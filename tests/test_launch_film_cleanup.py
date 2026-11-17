import subprocess
import sys

import pytest

from tools import launch_film


@pytest.mark.parametrize("failure", [RuntimeError("encode failed"), KeyboardInterrupt()])
def test_failed_film_export_removes_intermediate_frames(tmp_path, monkeypatch, failure):
    frames = []

    class Frame:
        def save(self, path):
            path.write_bytes(b"frame")
            frames.append(path)

    class Film:
        def __init__(self, stack):
            pass

        def frame(self, time):
            return Frame()

    monkeypatch.setattr(launch_film, "Film", Film)
    monkeypatch.setattr(launch_film, "FPS", 1)
    monkeypatch.setattr(launch_film.tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(launch_film, "synth_audio", lambda duration, path: path.write_bytes(b"audio"))
    monkeypatch.setattr(launch_film.shutil, "which", lambda name: "ffmpeg")
    monkeypatch.setattr(sys, "argv", ["launch_film.py", "--stack", str(tmp_path)])

    def fail_encode(command, check):
        assert frames and all(path.is_file() for path in frames)
        assert (frames[0].parent / "score.wav").is_file()
        raise failure

    monkeypatch.setattr(subprocess, "run", fail_encode)
    with pytest.raises(type(failure)) as error:
        launch_film.main()

    assert error.value is failure
    assert not frames[0].parent.exists()
