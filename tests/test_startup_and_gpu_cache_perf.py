import os
import time
from types import SimpleNamespace

import core._background_focus_gpu as bg_gpu
import core._ecc_area_gpu as ecc_gpu
import core._perspective_gpu as persp_gpu
import fusion_methods._openfocus_gpu as ogpu
import fusion_methods._openfocus_gpu_a as ogpu_a
import fusion_methods._openfocus_gpu_c as ogpu_c
import fusion_methods._openfocus_gpu_dct as ogpu_dct
import fusion_methods._openfocus_gpu_dtcwt as ogpu_dtcwt
import fusion_methods._openfocus_gpu_gfg as ogpu_gfg
import fusion_methods._openfocus_gpu_gff as ogpu_gff
from utils import diagnostics, logging_setup


def test_scandir_log_and_crash_pruning_and_listing(monkeypatch, tmp_path):
    now = time.time()
    old = now - (logging_setup.RETENTION_DAYS + 2) * 86400

    log_dir = tmp_path / "logs"
    crash_dir = tmp_path / "fallback"
    log_dir.mkdir()
    crash_dir.mkdir()

    stale_log = log_dir / "openfocus-20200101T000000_000000-2147483647.log"
    active_log = log_dir / f"openfocus-20200101T000000_000000-{os.getpid()}.log"
    legacy_log = log_dir / "openfocus.log.1"
    unrelated_log = log_dir / "notes.txt"

    for p in (stale_log, active_log, legacy_log, unrelated_log):
        p.write_text(p.name, encoding="utf-8")

    os.utime(stale_log, (old, old))
    os.utime(active_log, (old + 10, old + 10))
    os.utime(legacy_log, (now - 10, now - 10))
    os.utime(unrelated_log, (old, old))

    logging_setup._prune_old_logs(log_dir)
    assert not stale_log.exists()
    assert active_log.exists()
    assert legacy_log.exists()
    assert unrelated_log.exists()

    monkeypatch.setattr(logging_setup, "logs_dir", lambda: log_dir)
    listed = logging_setup.all_log_files()
    assert [p.name for p in listed] == [
        legacy_log.name,
        active_log.name,
    ]

    stale_crash = crash_dir / "openfocus-20200101T000000_000000-2147483647.crash.txt"
    active_crash = crash_dir / f"openfocus-20200101T000000_000000-{os.getpid()}.crash.txt"
    non_crash_session = crash_dir / "openfocus-20200101T000000_000000-2147483647.log"
    for p in (stale_crash, active_crash, non_crash_session):
        p.write_text(p.name, encoding="utf-8")
        os.utime(p, (old, old))

    diagnostics._prune_fallback_crashes(crash_dir)
    assert not stale_crash.exists()
    assert active_crash.exists()
    assert non_crash_session.exists()


def _make_fake_opencl():
    stats = {"contexts": 0, "source_builds": 0, "binary_builds": 0}

    class FakeBuffer:
        def __init__(self, *_args, **_kwargs):
            pass

        def release(self):
            pass

    class FakeKernel:
        def __init__(self, _prog, name):
            self.name = name

        def get_work_group_info(self, _info, _dev):
            return 256

    class FakeProgram:
        def __init__(self, ctx, *args):
            self.ctx = ctx
            if len(args) == 1 and isinstance(args[0], str):
                self._from_binary = False
                self._tag = f"BIN:{hash(args[0])}".encode("ascii")
            else:
                self._from_binary = True
                self._tag = args[1][0]

        def build(self, **_kwargs):
            if self._from_binary:
                stats["binary_builds"] += 1
            else:
                stats["source_builds"] += 1
            return self

        def get_info(self, _param):
            return [self._tag]

    class FakeContext:
        def __init__(self, devices=None):
            stats["contexts"] += 1
            self.devices = devices

    class FakeQueue:
        def __init__(self, ctx, device=None):
            self.ctx = ctx
            self.device = device

        def finish(self):
            pass

    cl = SimpleNamespace(
        Context=lambda devs: FakeContext(devices=devs),
        CommandQueue=FakeQueue,
        Program=FakeProgram,
        Kernel=FakeKernel,
        Buffer=FakeBuffer,
        mem_flags=SimpleNamespace(READ_ONLY=1, COPY_HOST_PTR=2),
        program_info=SimpleNamespace(BINARIES=1),
        device_fp_config=SimpleNamespace(CORRECTLY_ROUNDED_DIVIDE_SQRT=1),
        kernel_work_group_info=SimpleNamespace(WORK_GROUP_SIZE=1),
    )
    device = SimpleNamespace(
        int_ptr=0xCAFEBABE,
        name="FakeGPU",
        vendor="TestVendor",
        driver_version="1.0",
        version="OpenCL 1.2",
        platform=SimpleNamespace(name="FakePlat", vendor="TestVendor", version="1.2"),
        global_mem_size=8 * 1024**3,
        max_work_group_size=256,
        max_work_item_sizes=(256, 256, 64),
        local_mem_size=65536,
        single_fp_config=1,
    )
    return cl, device, stats


def test_shared_opencl_context_reused_and_invalidated_across_gpu_modules(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENFOCUS_OPENCL_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(ogpu, '_SHARED_CONTEXT', None)
    monkeypatch.setattr(persp_gpu, "_self_check", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(persp_gpu, "_program", None)
    monkeypatch.setattr(bg_gpu, "_PROGRAM_CACHE", {})
    for mod in (ogpu, ogpu_a, ogpu_c, ogpu_dct, ogpu_dtcwt, ogpu_gff, ogpu_gfg, ecc_gpu):
        mod._invalidate_program_cache()

    try:
        cl, device, stats = _make_fake_opencl()
        merge_ctx, merge_q, _, _ = ogpu._get_or_build_merge_program(cl, device)
        ogpu._get_or_build_depth_program(cl, device, merge_ctx)
        res_a = ogpu_a._get_or_build_program(cl, device)
        res_c = ogpu_c._get_or_build_program(cl, device)
        res_dct = ogpu_dct._get_or_build_program(cl, device)
        res_dtcwt = ogpu_dtcwt._get_or_build_program(cl, device)
        res_gff = ogpu_gff._get_or_build_program(cl, device)
        res_gfg = ogpu_gfg._get_or_build_program(cl, device)
        res_bg = bg_gpu._get_or_build_kernel(cl, device)
        res_ecc = ecc_gpu._get_or_build_program(cl, device)
        res_persp = persp_gpu._verified_program(cl, device)

        assert stats["contexts"] == 1
        stages = (res_a, res_c, res_dct, res_dtcwt, res_gff, res_gfg, res_bg, res_ecc, res_persp)
        assert all(result[0] is merge_ctx for result in stages)
        # Finishing an ECC or warp queue must never wait on the fusion queue.
        assert len({id(merge_q), *(id(result[1]) for result in stages)}) == len(stages) + 1

        # Disk binaries were written on cold build for every module (including DCT/DTCWT/GFF/GFG/BG).
        cached_files = list(tmp_path.glob("*.clbin"))
        assert len(cached_files) == stats['source_builds']

        # Rotate the context on the same device and rebuild from its disk binaries.
        cold_builds = stats['source_builds']
        ogpu_dct._invalidate_program_cache(merge_ctx)
        new_ctx_dct, _, _ = ogpu_dct._get_or_build_program(cl, device)
        assert new_ctx_dct is not merge_ctx
        assert stats['contexts'] == 2
        assert stats['binary_builds'] == 1

        # An old job may finish failing after another stage has already recovered.
        ogpu_a._invalidate_program_cache(merge_ctx)
        new_ctx_a, _, _ = ogpu_a._get_or_build_program(cl, device)
        new_ctx_persp = persp_gpu._verified_program(cl, device)[0]
        assert new_ctx_a is new_ctx_persp is new_ctx_dct
        assert stats['contexts'] == 2
        assert stats['source_builds'] == cold_builds
        assert stats['binary_builds'] == 3
    finally:
        for mod in (ogpu, ogpu_a, ogpu_c, ogpu_dct, ogpu_dtcwt, ogpu_gff, ogpu_gfg, ecc_gpu):
            mod._invalidate_program_cache()
