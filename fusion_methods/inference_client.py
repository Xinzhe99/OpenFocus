"""Cancellable transport for the optional accelerator interpreter."""
import json
import mmap
import os
import queue
import subprocess
import threading
import time
import uuid

import numpy as np

from core.inference_process import spawn
from fusion_methods.neural_common import cancel


class WorkerError(RuntimeError):
    """A native engine rejected the current request."""


class WorkerLost(WorkerError):
    """The process or its protocol can no longer be used."""


class Worker:
    start_timeout = 60
    run_timeout = 300

    def __init__(self, command, should_cancel):
        self.process = spawn(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL)
        self.replies = queue.Queue()
        self.requests = queue.Queue()
        self.reader = threading.Thread(target=self._read, daemon=True, name='neural-worker-replies')
        try:
            self.reader.start()
            reply = self._receive(self.start_timeout, should_cancel)
            engines = reply.get('backends')
            if reply.get('ready') is not True or not isinstance(engines, list):
                raise WorkerLost(str(reply.get('error', 'Invalid worker startup response')))
            self.backends = tuple(k for k in engines if k in ('TORCH_CUDA', 'TENSORRT'))
            self.device = str(reply.get('device') or '')
        except BaseException:
            self.close()
            raise

    def _read(self):
        try:
            for line in iter(lambda: self.process.stdout.readline(65537), b''):
                if len(line) > 65536:
                    break
                reply = json.loads(line)
                if not isinstance(reply, dict):
                    break
                self.replies.put(reply)
                request = self.requests.get()
                if request is None:
                    break
                self.process.stdin.write(json.dumps(request).encode('ascii') + b'\n')
                self.process.stdin.flush()
        except (ValueError, OSError):
            pass
        finally:
            self.replies.put(None)

    def _receive(self, timeout, should_cancel):
        deadline = time.monotonic() + timeout
        while True:
            cancel(should_cancel)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise WorkerLost('Accelerator worker timed out')
            try:
                reply = self.replies.get(timeout=min(remaining, 0.05))
            except queue.Empty:
                continue
            if reply is None:
                raise WorkerLost('Accelerator worker exited or sent an invalid response')
            return reply

    def run(self, backend, model, planes, cache, should_cancel):
        if os.name == 'nt':
            from core.resource_policy import probe_memory, require_memory
            size = planes.size * np.dtype(np.float32).itemsize
            require_memory(size, probe_memory())
            name = 'Local\\OpenFocus-' + uuid.uuid4().hex
            region = mmap.mmap(-1, size, tagname=name, access=mmap.ACCESS_WRITE)
            # NumPy holds the mapping as its base through all output views.
            # Windows releases the pagefile section when its final view closes.
            mapped = np.ndarray(planes.shape, np.float32, buffer=region)
            storage = {'mapping': name}
        else:
            from core.frame_storage import create_frame
            mapped = create_frame(planes.shape, np.float32)
            storage = {'tensor': os.path.abspath(mapped.filename)}
        np.copyto(mapped, planes)
        cancel(should_cancel)
        request = {'op': 'run', 'backend': backend, 'model': model,
                   **storage, 'shape': list(planes.shape),
                   'cache': os.path.abspath(cache)}
        try:
            self.requests.put(request)
            reply = self._receive(self.run_timeout, should_cancel)
            if reply.get('ok') is True and reply.get('backend') == backend:
                return mapped
            if reply.get('error'):
                raise WorkerError(str(reply['error'])[:512])
            raise WorkerLost('Invalid accelerator inference response')
        except (OSError, WorkerLost) as exc:
            self.close()
            raise WorkerLost(str(exc)) from None
        except WorkerError:
            raise
        except BaseException:
            # The worker may still be accessing the mapping on cancellation.
            self.close()
            raise

    def close(self):
        process = self.process
        self.requests.put(None)
        if process.poll() is None:
            process.kill()
        try:
            process.wait(timeout=3)
        finally:
            # The I/O thread owns both pipes until it exits.
            if self.reader.is_alive():
                self.reader.join(timeout=1)
            if not self.reader.is_alive():
                process.stdin.close()
                process.stdout.close()
