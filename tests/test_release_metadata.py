"""Release identity: tag equality and safe branch acceptance labels."""
import pytest

from tools import release_metadata as rm


def test_tag_build_keeps_the_exact_version():
    identity = rm.resolve_identity(
        "refs/tags/v1.75.3", "v1.75.3", "42", "1.75.3")
    assert identity == {
        "version": "1.75.3",
        "label": "v1.75.3",
        "channel": "release",
        "tag": "v1.75.3",
    }


def test_tag_build_rejects_a_mismatched_app_version():
    with pytest.raises(rm.ReleaseIdentityError, match="APP_VERSION"):
        rm.resolve_identity(
            "refs/tags/v1.75.4", "v1.75.4", "42", "1.75.3")


def test_branch_dispatch_gets_a_safe_preview_label():
    identity = rm.resolve_identity(
        "refs/heads/feature/gpu-preview", "feature/gpu-preview", "421", "1.75.3")
    assert identity["channel"] == "preview"
    assert identity["label"] == "v1.75.3-preview.421"
    assert identity["tag"] == ""
    assert rm.SAFE_LABEL_RE.fullmatch(identity["label"])


def test_branch_dispatch_falls_back_to_a_short_sha():
    long_sha = "a1b2c3d4e5f60718"
    identity = rm.resolve_identity(
        "refs/heads/main", "main", "", "1.75.3", long_sha)
    assert identity["label"] == "v1.75.3-preview.a1b2c3d4"


def test_branch_dispatch_needs_some_distinguishing_input():
    with pytest.raises(rm.ReleaseIdentityError, match="RUN_NUMBER"):
        rm.resolve_identity("refs/heads/main", "main", "", "1.75.3", "")


def test_unsafe_versions_are_rejected_before_packaging():
    with pytest.raises(rm.ReleaseIdentityError, match="dotted numeric"):
        rm.resolve_identity("refs/heads/main", "main", "1", "1.75.3/evil")


def test_cli_writes_github_output_and_reports_tag_mismatch(tmp_path):
    output = tmp_path / "github-output"
    assert rm.main([
        "--version", "1.75.3",
        "--ref", "refs/heads/main",
        "--ref-name", "main",
        "--run-number", "7",
        "--sha", "deadbeef",
        "--github-output", str(output),
    ]) == 0
    values = dict(line.split("=", 1) for line in
                  output.read_text(encoding="utf-8").splitlines())
    assert values == {
        "version": "1.75.3",
        "label": "v1.75.3-preview.7",
        "channel": "preview",
        "tag": "",
    }

    assert rm.main([
        "--version", "1.75.3",
        "--ref", "refs/tags/v1.75.4",
        "--ref-name", "v1.75.4",
    ]) == 1
