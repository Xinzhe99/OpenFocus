#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace openfocus {

struct CancelRequested : public std::runtime_error {
    CancelRequested() : std::runtime_error("OpenFocus native operation cancelled") {}
};

inline int clamp_threads(int num_threads, int64_t num_tasks) {
    if (num_tasks <= 1) {
        return 1;
    }
    int hw = static_cast<int>(std::thread::hardware_concurrency());
    if (hw <= 0) {
        hw = 4;
    }
    int requested = num_threads <= 0 ? hw : num_threads;
    int capped = static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(requested, num_tasks)));
    return capped;
}

namespace detail {

inline thread_local bool g_in_pool_worker = false;

class PersistentThreadPool {
public:
    template <typename Fn>
    void run(int workers, Fn&& worker_loop) {
        if (workers <= 1 || g_in_pool_worker) {
            worker_loop();
            return;
        }
        std::unique_lock<std::mutex> dispatch_lock(dispatch_mutex_);
        const int helpers = workers - 1;
        ensure_workers_locked(helpers);

        // Bind a type-erased trampoline to avoid std::function heap allocation.
        auto trampoline = [](void* ctx) {
            (*static_cast<std::remove_reference_t<Fn>*>(ctx))();
        };

        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            if (stopping_) {
                dispatch_lock.unlock();
                worker_loop();
                return;
            }
            job_fn_ = trampoline;
            job_ctx_ = const_cast<void*>(static_cast<const void*>(&worker_loop));
            active_helpers_ = helpers;
            remaining_helpers_.store(helpers, std::memory_order_release);
            ++generation_;
        }
        cv_start_.notify_all();

        worker_loop();

        if (remaining_helpers_.load(std::memory_order_acquire) != 0) {
            std::unique_lock<std::mutex> lock(state_mutex_);
            cv_done_.wait(lock, [this]() {
                return remaining_helpers_.load(std::memory_order_acquire) == 0;
            });
        }
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            job_fn_ = nullptr;
            job_ctx_ = nullptr;
            active_helpers_ = 0;
        }
    }

    void shutdown() {
        std::lock_guard<std::mutex> dispatch_lock(dispatch_mutex_);
        std::vector<std::thread> to_join;
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            stopping_ = true;
            ++generation_;
            to_join.swap(threads_);
        }
        cv_start_.notify_all();
        for (auto& t : to_join) {
            if (t.joinable()) {
                t.join();
            }
        }
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            stopping_ = false;
        }
    }

private:
    void ensure_workers_locked(int helpers) {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (stopping_) {
            return;
        }
        while (static_cast<int>(threads_.size()) < helpers) {
            const int id = static_cast<int>(threads_.size());
            const uint64_t initial_gen = generation_;
            threads_.emplace_back([this, id, initial_gen]() {
                g_in_pool_worker = true;
                uint64_t seen_gen = initial_gen;
                for (;;) {
                    void (*fn)(void*) = nullptr;
                    void* ctx = nullptr;
                    {
                        std::unique_lock<std::mutex> ul(state_mutex_);
                        cv_start_.wait(ul, [this, seen_gen]() {
                            return stopping_ || generation_ != seen_gen;
                        });
                        if (stopping_) {
                            return;
                        }
                        seen_gen = generation_;
                        if (id < active_helpers_) {
                            fn = job_fn_;
                            ctx = job_ctx_;
                        }
                    }
                    if (fn != nullptr) {
                        fn(ctx);
                        if (remaining_helpers_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                            std::lock_guard<std::mutex> ul(state_mutex_);
                            cv_done_.notify_one();
                        }
                    }
                }
            });
        }
    }

    std::mutex dispatch_mutex_;
    std::mutex state_mutex_;
    std::condition_variable cv_start_;
    std::condition_variable cv_done_;
    std::vector<std::thread> threads_;
    void (*job_fn_)(void*) = nullptr;
    void* job_ctx_ = nullptr;
    int active_helpers_ = 0;
    std::atomic<int> remaining_helpers_{0};
    uint64_t generation_ = 0;
    bool stopping_ = false;
};

inline PersistentThreadPool& global_thread_pool() {
    // Intentionally heap-allocated so Windows DLL unload never joins threads
    // under the loader lock; Python atexit calls shutdown_thread_pool() cleanly.
    static PersistentThreadPool* pool = new PersistentThreadPool();
    return *pool;
}

}  // namespace detail

inline void shutdown_thread_pool() {
    detail::global_thread_pool().shutdown();
}

template <typename Fn>
inline void parallel_bands(
    int64_t total,
    int64_t band,
    int num_threads,
    Fn&& fn,
    const std::atomic<bool>* cancel_flag = nullptr
) {
    if (total <= 0) {
        return;
    }
    const int64_t step = std::max<int64_t>(1, band);
    const int64_t num_tasks = (total + step - 1) / step;
    const int workers = clamp_threads(num_threads, num_tasks);

    auto check_cancel = [cancel_flag]() {
        if (cancel_flag != nullptr && cancel_flag->load(std::memory_order_relaxed)) {
            throw CancelRequested();
        }
    };

    if (workers <= 1) {
        for (int64_t start = 0; start < total; start += step) {
            check_cancel();
            const int64_t stop = std::min<int64_t>(total, start + step);
            fn(start, stop);
        }
        check_cancel();
        return;
    }

    std::atomic<int64_t> next_task{0};
    std::atomic<bool> failed{false};
    std::exception_ptr first_error = nullptr;
    std::mutex err_mutex;

    auto worker_loop = [&]() {
        try {
            while (!failed.load(std::memory_order_relaxed)) {
                if (cancel_flag != nullptr && cancel_flag->load(std::memory_order_relaxed)) {
                    throw CancelRequested();
                }
                const int64_t task_idx = next_task.fetch_add(1, std::memory_order_relaxed);
                if (task_idx >= num_tasks) {
                    break;
                }
                const int64_t start = task_idx * step;
                const int64_t stop = std::min<int64_t>(total, start + step);
                fn(start, stop);
            }
        } catch (...) {
            failed.store(true, std::memory_order_relaxed);
            std::lock_guard<std::mutex> lock(err_mutex);
            if (!first_error) {
                first_error = std::current_exception();
            }
        }
    };

    detail::global_thread_pool().run(workers, worker_loop);

    if (first_error) {
        std::rethrow_exception(first_error);
    }
    check_cancel();
}

}  // namespace openfocus
