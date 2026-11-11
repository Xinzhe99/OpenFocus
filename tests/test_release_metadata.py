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

def test_beta_tag_keeps_numeric_version_and_exact_label():
    assert rm.resolve_identity('refs/tags/v1.77.0-beta.1', 'v1.77.0-beta.1',
                               '42', '1.77.0') == {
        'version': '1.77.0', 'label': 'v1.77.0-beta.1',
        'channel': 'beta', 'tag': 'v1.77.0-beta.1',
    }


@pytest.mark.parametrize('tag', ['v1.77.1-beta.1', 'v1.77.0-beta.0',
                                 'v1.77.0-beta.01', 'v1.77.0-beta.x',
                                 'v1.77.0-rc.1', 'v1.77.0-beta.1-extra'])
def test_invalid_beta_tags_are_rejected(tag):
    with pytest.raises(rm.ReleaseIdentityError):
        rm.resolve_identity('refs/tags/' + tag, tag, '42', '1.77.0')


@pytest.mark.parametrize('version', ['1.77.0.999999999.4', '1.999999999.01',
                                     '1.0999999999.4', '999999999.1',
                                     '1.77.0.999999999', '1.77.0.999999999.0'])
def test_reserved_beta_alias_versions_cannot_be_published(version):
    tag = 'v' + version
    with pytest.raises(rm.ReleaseIdentityError, match='reserved beta alias'):
        rm.resolve_identity('refs/tags/' + tag, tag, '42', version)


@pytest.mark.parametrize('version', ['1.77.0', '1.77.0.4', '1.77.0.999999998.0'])
def test_other_numeric_versions_remain_valid(version):
    tag = 'v' + version
    assert rm.resolve_identity('refs/tags/' + tag, tag, '42', version)['tag'] == tag


def test_tag_ref_must_match_name():
    with pytest.raises(rm.ReleaseIdentityError, match='ref and tag name'):
        rm.resolve_identity('refs/tags/v1.77.0', 'v1.77.0-beta.1', '42', '1.77.0')


@pytest.mark.parametrize('ref,name,expected', [
    ('refs/tags/v1.77.0-beta.1', 'v1.77.0-beta.1', '-beta.1'),
    ('refs/tags/v1.77.0', 'v1.77.0', ''),
    ('refs/heads/main', 'main', ''),
])
def test_build_marker_matches_resolved_channel(tmp_path, ref, name, expected):
    marker = tmp_path / 'build_identity.py'
    assert rm.main(['--version', '1.77.0', '--ref', ref, '--ref-name', name,
                    '--run-number', '123', '--write-build-identity', str(marker)]) == 0
    namespace = {}
    exec(marker.read_text(encoding='utf-8'), namespace)
    assert namespace['BETA_SUFFIX'] == expected


def test_beta_runtime_identity_reaches_report_metadata(monkeypatch):
    from types import SimpleNamespace
    import build_identity
    import constants
    from utils.reporting import report_metadata
    monkeypatch.setattr(constants, 'APP_VERSION', '1.77.0')
    monkeypatch.setattr(build_identity, 'BETA_SUFFIX', '-beta.1')
    bundle = SimpleNamespace(fingerprint='abc', platform='windows', subject='error',
                             summary='error', log_excerpt='', occurred_at='2026-11-11')
    assert constants.app_display_version() == '1.77.0-beta.1'
    assert report_metadata(bundle, source='manual', locale='en')['app_version'] == '1.77.0-beta.1'
    assert constants.APP_VERSION == '1.77.0'


def test_pypi_verification_rejects_beta_identity(tmp_path):
    dist = tmp_path / 'dist'
    dist.mkdir()
    (dist / 'openfocus.whl').write_bytes(b'wheel')
    (dist / 'openfocus.tar.gz').write_bytes(b'source')
    identity = tmp_path / 'identity.json'
    args = ['--identity', str(identity), '--dist', str(dist), '--sha', 'a' * 40,
            '--version', '1.77.0', '--channel', 'beta', '--tag', 'v1.77.0-beta.1']
    rm.pypi_main(['create', *args])
    with pytest.raises(rm.ReleaseIdentityError, match='only an exact release tag'):
        rm.pypi_main(['verify', *args])
    with pytest.raises(rm.ReleaseIdentityError, match='only an exact release tag'):
        rm.pypi_main(['tag', *args])


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


def test_cli_requires_full_checkout_sha(tmp_path):
    output = tmp_path / "output"
    args = ["--version", "1.75.3", "--ref", "refs/tags/v1.75.3",
            "--ref-name", "v1.75.3", "--github-output", str(output)]
    assert rm.main(args + ["--checkout-sha", "deadbeef"]) == 1
    assert rm.main(args + ["--checkout-sha", "a" * 40]) == 0
    assert "sha=" + "a" * 40 in output.read_text()


def test_distribution_identity_detects_same_size_replacement(tmp_path):
    (tmp_path / "package.whl").write_bytes(b"wheel")
    source = tmp_path / "package.tar.gz"
    source.write_bytes(b"source")
    records = rm.dist_records(tmp_path)
    source.write_bytes(b"tamper")
    assert records != rm.dist_records(tmp_path)


def test_desktop_identity_rejects_wrong_run_and_duplicate_assets():
    identity = {"tag": "v1.2", "version": "1.2", "sha": "a" * 40}
    asset = {"name": "package.zip", "size": 1, "sha256": "b" * 64}
    desktop = {**identity, "schema_version": 1, "run_id": 123,
               "run_attempt": 1, "assets": [asset]}
    rm.validate_release_identity(identity, desktop)
    with pytest.raises(rm.ReleaseIdentityError):
        rm.validate_release_identity(identity, {**desktop, "run_attempt": 0})
    with pytest.raises(rm.ReleaseIdentityError):
        rm.validate_release_identity(identity, {**desktop, "sha": "c" * 40})
    with pytest.raises(rm.ReleaseIdentityError):
        rm.validate_release_identity(identity, {**desktop, "assets": [asset, asset]})


def test_reused_jobs_must_be_successful_in_the_same_run():
    sha = "a" * 40
    names = [f"release-tests / pytest ({system})" for system in
             ("ubuntu-latest", "windows-latest", "macos-latest")]
    names += ["release-tests / Installed wheel without GUI",
              "release-tests / Release test summary", "Publish GitHub Release"]
    jobs = [{"name": name, "head_sha": sha, "status": "completed",
             "conclusion": "success"} for name in names]
    assert rm.required_jobs_success([jobs[:1], jobs[1:]], sha)
    assert not rm.required_jobs_success([jobs[:-1]], sha)
    failed = {**jobs[0], "conclusion": "failure"}
    assert not rm.required_jobs_success([[failed], jobs], sha)
    assert not rm.required_jobs_success([jobs], "b" * 40)


def test_existing_pypi_file_must_have_matching_digest(tmp_path, monkeypatch):
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "package.whl").write_bytes(b"wheel")
    (dist / "package.tar.gz").write_bytes(b"source")
    identity = {"version": "1.2", "assets": rm.dist_records(dist)}
    item = identity["assets"][0]
    monkeypatch.setattr(rm, "json_request", lambda *args, **kwargs: {
        "urls": [{"filename": item["name"], "digests": {"sha256": "0" * 64}}]})
    with pytest.raises(rm.ReleaseIdentityError, match="different digest"):
        rm.prepare_missing_pypi(identity, dist, tmp_path / "upload")


@pytest.mark.parametrize("conclusion", ["failure", "cancelled"])
def test_desktop_bound_attempt_failure_never_uses_another_run(monkeypatch, conclusion):
    import io
    import json

    identity = {"tag": "v1.2", "version": "1.2", "sha": "a" * 40}
    desktop = {**identity, "schema_version": 1, "run_id": 42, "run_attempt": 1,
               "assets": [{"name": "asset.zip", "size": 3, "sha256": "b" * 64}]}
    release = {"id": 7, "tag_name": "v1.2", "draft": False, "prerelease": False,
               "assets": [{"name": "OpenFocus-v1.2-build.json", "url": "identity"},
                          {"name": "asset.zip", "size": 3, "digest": "sha256:" + "b" * 64}]}
    calls = []

    def api(url, **kwargs):
        calls.append(url)
        if "/releases/tags/" in url:
            return release
        if "/releases/7/assets?" in url:
            return release["assets"]
        if url.endswith("/actions/runs/42"):
            return {"run_attempt": 2}
        if url.endswith("/actions/runs/42/attempts/2"):
            return {"id": 42, "run_attempt": 2, "path": ".github/workflows/release.yml",
                    "head_sha": identity["sha"], "status": "completed", "conclusion": conclusion}
        raise AssertionError(url)

    monkeypatch.setattr(rm, "check_tag", lambda identity: None)
    monkeypatch.setattr(rm, "json_request", api)
    monkeypatch.setattr(rm, "request", lambda *args, **kwargs: io.BytesIO(json.dumps(desktop).encode()))
    with pytest.raises(rm.ReleaseIdentityError, match="failed or was cancelled"):
        rm.wait_for_desktop(identity, "https://example.test", "token", timeout=1, interval=0)
    assert not any("attempts/1" in url for url in calls)


def test_missing_release_digest_is_verified_by_download(monkeypatch):
    import hashlib
    import io

    payload = b"verified asset"
    record = {"name": "asset.zip", "size": len(payload),
              "sha256": hashlib.sha256(payload).hexdigest()}
    release = {"assets": [{"name": "identity.json"},
                          {"name": "asset.zip", "size": len(payload), "url": "asset"}]}
    monkeypatch.setattr(rm, "request", lambda *args, **kwargs: io.BytesIO(payload))
    rm.verify_remote_assets(release, {"assets": [record]}, "identity.json", "token")
    monkeypatch.setattr(rm, "request", lambda *args, **kwargs: io.BytesIO(b"x" * len(payload)))
    with pytest.raises(rm.ReleaseIdentityError, match="digest mismatch"):
        rm.verify_remote_assets(release, {"assets": [record]}, "identity.json", "token")


def test_remote_tag_movement_is_rejected(monkeypatch):
    identity = {"tag": "v1.2", "sha": "a" * 40}
    monkeypatch.setattr(rm, "remote_tag_sha", lambda tag: "b" * 40)
    with pytest.raises(rm.ReleaseIdentityError, match="remote tag"):
        rm.check_tag(identity)


def test_existing_matching_pypi_file_is_excluded(tmp_path, monkeypatch):
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "package.whl").write_bytes(b"wheel")
    (dist / "package.tar.gz").write_bytes(b"source")
    identity = {"version": "1.2", "assets": rm.dist_records(dist)}
    item = identity["assets"][0]
    monkeypatch.setattr(rm, "json_request", lambda *args, **kwargs: {
        "urls": [{"filename": item["name"], "digests": {"sha256": item["sha256"]}}]})
    upload = tmp_path / "upload"
    assert rm.prepare_missing_pypi(identity, dist, upload) == 1
    assert {path.name for path in upload.iterdir()} == {identity["assets"][1]["name"]}


def test_desktop_release_waits_for_bound_successful_attempt(monkeypatch):
    import io
    import json

    identity = {"tag": "v1.2", "version": "1.2", "sha": "a" * 40}
    desktop = {**identity, "schema_version": 1, "run_id": 42, "run_attempt": 1,
               "assets": [{"name": "asset.zip", "size": 3, "sha256": "b" * 64}]}
    release = {"id": 7, "tag_name": "v1.2", "draft": False, "prerelease": False,
               "assets": [{"name": "OpenFocus-v1.2-build.json", "url": "identity"},
                          {"name": "asset.zip", "size": 3, "digest": "sha256:" + "b" * 64}]}
    names = [f"release-tests / pytest ({system})" for system in
             ("ubuntu-latest", "windows-latest", "macos-latest")]
    names += ["release-tests / Installed wheel without GUI",
              "release-tests / Release test summary", "Publish GitHub Release"]
    jobs = [{"name": name, "head_sha": identity["sha"], "status": "completed",
             "conclusion": "success"} for name in names]
    calls = []

    def api(url, **kwargs):
        calls.append(url)
        if "/releases/tags/" in url:
            return release
        if "/releases/7/assets?" in url:
            return release["assets"]
        if url.endswith("/actions/runs/42"):
            return {"run_attempt": 2}
        if url.endswith("/actions/runs/42/attempts/2"):
            return {"id": 42, "run_attempt": 2, "path": ".github/workflows/release.yml",
                    "head_sha": identity["sha"], "status": "completed", "conclusion": "success"}
        raise AssertionError(url)

    monkeypatch.setattr(rm, "check_tag", lambda identity: None)
    monkeypatch.setattr(rm, "json_request", api)
    monkeypatch.setattr(rm, "request", lambda *args, **kwargs: io.BytesIO(json.dumps(desktop).encode()))
    monkeypatch.setattr(rm, "api_jobs", lambda base, run_id, attempt, token:
                        jobs[:1] if attempt == 2 else jobs[1:])
    rm.wait_for_desktop(identity, "https://example.test", "token", timeout=1, interval=0)
    assert calls[-1].endswith("/actions/runs/42/attempts/2")


def test_release_asset_listing_reads_all_pages(monkeypatch):
    from urllib.parse import parse_qs, urlsplit

    first = [{"name": f"asset-{index}"} for index in range(100)]
    last = [{"name": "identity.json"}]
    calls = []

    def api(url, **kwargs):
        calls.append(url)
        page = parse_qs(urlsplit(url).query)["page"]
        return {"1": first, "2": last}[page[0]]

    monkeypatch.setattr(rm, "json_request", api)
    assert rm.api_release_assets("https://example.test", 7, "token") == first + last
    assert len(calls) == 2
