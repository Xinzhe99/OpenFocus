"""Depth map -> textured OBJ export (utils/mesh_export.py)."""
import os

import numpy as np
import pytest

from utils.mesh_export import export_obj


def test_obj_export_writes_consistent_mesh(tmp_path):
    h, w = 120, 200
    yy, xx = np.mgrid[0:h, 0:w]
    index01 = (xx / (w - 1)).astype(np.float32)          # ramp
    tex = np.full((h, w, 3), 40000, np.uint16)           # 16-bit texture
    out = export_obj(os.path.join(str(tmp_path), "模型.obj"), index01, tex,
                     relief=0.1, max_side=50)
    obj, mtl, png, metadata = out["files"]
    assert all(os.path.exists(p) for p in out["files"])
    lines = open(obj, encoding="utf-8").read().splitlines()
    v = [l for l in lines if l.startswith("v ")]
    vt = [l for l in lines if l.startswith("vt ")]
    f = [l for l in lines if l.startswith("f ")]
    assert len(v) == len(vt) == out["vertices"] == 50 * 30
    assert len(f) == out["faces"] == 2 * 49 * 29
    # every face index is a valid 1-based vertex
    ids = np.array([int(t.split("/")[0]) for l in f for t in l.split()[1:]])
    assert ids.min() == 1 and ids.max() == len(v)
    # relief: z spans 0 .. relief * longest side along the ramp
    z = np.array([float(l.split()[3]) for l in v])
    # area-downsampled endpoints average a few pixels of the ramp
    assert z.min() < 0.05 * 0.1 * w and z.max() > 0.95 * 0.1 * w
    assert "map_Kd 模型_texture.png" in open(mtl, encoding="utf-8").read()


def test_invert_flips_relief(tmp_path):
    index01 = np.zeros((20, 20), np.float32)
    out = export_obj(os.path.join(str(tmp_path), "m.obj"), index01,
                     np.zeros((20, 20, 3), np.uint8), relief=0.5, invert=True)
    z = [float(l.split()[3]) for l in open(out["files"][0]) if l.startswith("v ")]
    assert min(z) == pytest.approx(9.5)  # pixel centres span 0 through 19


def test_physical_positions_preserve_excluded_capture_gaps():
    from utils.mesh_export import physical_depth
    index = np.array([[0, 0.5, 1, np.nan]], np.float32)
    depth, positions = physical_depth(index, [5, 2, 0], 0, 2)
    assert positions == [10, 4, 0]
    np.testing.assert_array_equal(depth[:, :3], [[10, 4, 0]])
    assert np.isnan(depth[0, 3])


def test_sidecar_collision_preserves_existing_files(tmp_path):
    old = tmp_path / "model_texture.png"
    old.write_bytes(b"existing texture")
    image = np.zeros((8, 8), np.float32)
    with pytest.raises(FileExistsError):
        export_obj(tmp_path / "model.obj", image, np.zeros((8, 8, 3), np.uint8))
    assert old.read_bytes() == b"existing texture"
    assert not (tmp_path / "model.obj").exists()


def test_group_commit_failure_restores_every_original(tmp_path, monkeypatch):
    import utils.mesh_export as mesh
    path = tmp_path / "model.obj"
    files = mesh.obj_paths(path)
    for file in files:
        with open(file, "wb") as stream:
            stream.write(file.encode())
    original = mesh.os.replace
    def fail_once(source, target):
        if str(target).endswith("model.mtl") and "backup-" not in str(source):
            raise OSError("commit failed")
        original(source, target)
    monkeypatch.setattr(mesh.os, "replace", fail_once)
    with pytest.raises(OSError, match="commit failed"):
        mesh.export_obj(path, np.zeros((8, 8), np.float32),
                        np.zeros((8, 8, 3), np.uint8), overwrite=True)
    for file in files:
        with open(file, "rb") as stream:
            assert stream.read() == file.encode()


def test_cancel_before_commit_leaves_no_group(tmp_path):
    from utils.mesh_export import MeshExportCancelled
    cancelled = False
    def progress(stage):
        nonlocal cancelled
        if stage == "texture":
            cancelled = True
    with pytest.raises(MeshExportCancelled):
        export_obj(tmp_path / "model.obj", np.zeros((8, 8), np.float32),
                   np.zeros((8, 8, 3), np.uint8), progress=progress,
                   should_cancel=lambda: cancelled)
    assert list(tmp_path.iterdir()) == []


def test_physical_mesh_uses_supplied_height_and_omits_holes(tmp_path):
    height = np.array([[0, 1e-7, 2e-7], [0, 1e-7, np.nan],
                       [0, 1e-7, 2e-7]], np.float32)
    info = export_obj(tmp_path / "µm.obj", np.zeros_like(height),
                      np.full((3, 3, 3), 40000, np.uint16),
                      height_field=height, pixel_size_um=1.0)
    vertices = [list(map(float, line.split()[1:]))
                for line in open(info["files"][0], encoding="utf-8")
                if line.startswith("v ")]
    assert info["faces"] == 4
    assert len(vertices) == 8
    assert vertices[1][2] == pytest.approx(1e-7)
    assert vertices[2][0] - vertices[0][0] == pytest.approx(2.0)


@pytest.mark.parametrize('index', range(4))
@pytest.mark.parametrize('overwrite', [False, True])
def test_directory_target_is_never_moved(tmp_path, index, overwrite):
    from pathlib import Path
    from utils.mesh_export import obj_paths
    path = tmp_path / 'model.obj'
    files = [Path(p) for p in obj_paths(path)]
    for i, target in enumerate(files):
        if i == index:
            (target / 'nested').mkdir(parents=True)
            (target / 'nested' / 'sentinel').write_bytes(b'keep')
        else:
            target.write_bytes(b'old')
    with pytest.raises(ValueError):
        export_obj(path, np.zeros((2, 2), np.float32), np.zeros((2, 2), np.uint8), overwrite=overwrite)
    assert (files[index] / 'nested' / 'sentinel').read_bytes() == b'keep'
    assert all(p.read_bytes() == b'old' for i, p in enumerate(files) if i != index)


@pytest.mark.parametrize('index', range(4))
def test_symlink_target_is_never_replaced(tmp_path, index):
    from pathlib import Path
    from utils.mesh_export import obj_paths
    files = [Path(p) for p in obj_paths(tmp_path / 'model.obj')]
    source = tmp_path / 'original'
    source.write_bytes(b'keep')
    try:
        files[index].symlink_to(source)
    except OSError:
        pytest.skip('symbolic links unavailable')
    with pytest.raises(ValueError):
        export_obj(files[0], np.zeros((2, 2), np.float32), np.zeros((2, 2), np.uint8), overwrite=True)
    assert files[index].is_symlink() and source.read_bytes() == b'keep'


def test_failed_restore_retains_backup_and_restores_other_files(tmp_path, monkeypatch):
    import utils.mesh_export as mesh
    files = mesh.obj_paths(tmp_path / 'model.obj')
    for name in files:
        with open(name, 'wb') as stream:
            stream.write(os.path.basename(name).encode())
    replace = os.replace
    def fail_commit_and_one_restore(src, dst):
        if os.path.basename(src) == 'model_texture.png' and '.openfocus-mesh-' in src:
            raise OSError('commit failed')
        if os.path.basename(src) == 'backup-model.mtl':
            raise OSError('restore failed')
        return replace(src, dst)
    monkeypatch.setattr(mesh.os, 'replace', fail_commit_and_one_restore)
    with pytest.raises(OSError, match='restore failed') as error:
        mesh.export_obj(files[0], np.zeros((2, 2), np.float32), np.zeros((2, 2), np.uint8), overwrite=True)
    backups = list(tmp_path.glob('.openfocus-mesh-*/backup-model.mtl'))
    assert len(backups) == 1 and backups[0].read_bytes() == b'model.mtl'
    assert str(backups[0]) in str(error.value)
    for name in (files[0], files[2], files[3]):
        with open(name, 'rb') as stream:
            assert stream.read() == os.path.basename(name).encode()


def test_directory_swapped_during_backup_is_restored(tmp_path, monkeypatch):
    import utils.mesh_export as mesh
    from pathlib import Path
    files = [Path(p) for p in mesh.obj_paths(tmp_path / 'model.obj')]
    for target in files:
        target.write_bytes(b'old')
    replace = os.replace
    swapped = False
    def swap_before_backup(src, dst):
        nonlocal swapped
        if src == str(files[1]) and not swapped:
            swapped = True
            files[1].unlink()
            files[1].mkdir()
            (files[1] / 'sentinel').write_bytes(b'keep')
        return replace(src, dst)
    monkeypatch.setattr(mesh.os, 'replace', swap_before_backup)
    with pytest.raises(ValueError):
        mesh.export_obj(files[0], np.zeros((2, 2), np.float32), np.zeros((2, 2), np.uint8), overwrite=True)
    assert (files[1] / 'sentinel').read_bytes() == b'keep'
    assert all(p.read_bytes() == b'old' for i, p in enumerate(files) if i != 1)
    assert not list(tmp_path.glob('.openfocus-mesh-*'))
