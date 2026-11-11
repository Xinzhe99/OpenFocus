"""Resolve the release identity used by the release workflow.

A stable tag matches ``constants.APP_VERSION``; a beta tag appends
``-beta.<positive integer>``. A branch ``workflow_dispatch`` run is an
acceptance build and never reaches the release job.

Run from the repository root:
    python tools/release_metadata.py --github-output "$GITHUB_OUTPUT"
"""
from __future__ import annotations

import argparse
import os
import re
import hashlib
import json
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TAG_REF_PREFIX = "refs/tags/"
# File names, installer versions and shell arguments all use this label.
SAFE_LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
VERSION_RE = re.compile(r"^[0-9]+(?:\.[0-9]+)*$")


class ReleaseIdentityError(ValueError):
    """The requested tag/version/label cannot be used for a build."""


def read_app_version() -> str:
    """Read the single source of truth without importing the whole app."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    import constants

    return str(constants.APP_VERSION)


def resolve_identity(ref: str, ref_name: str, run_number: str, version: str,
                     sha: str = "") -> dict:
    """Return version/label/channel/tag for one workflow run.

    Stable and beta tags retain their exact names. Branch builds receive a
    distinct safe preview label and cannot be published as a release.
    """
    version = str(version).strip()
    if not VERSION_RE.fullmatch(version):
        raise ReleaseIdentityError(
            f"constants.APP_VERSION '{version}' is not a dotted numeric version")
    if any(int(part) == 999999999 for part in version.split('.')):
        raise ReleaseIdentityError("constants.APP_VERSION uses the reserved beta alias namespace")

    ref = str(ref or "")
    ref_name = str(ref_name or "")
    if ref.startswith(TAG_REF_PREFIX):
        expected = "v" + version
        if ref != TAG_REF_PREFIX + ref_name:
            raise ReleaseIdentityError("tag ref and tag name differ")
        if ref_name == expected:
            channel = "release"
        elif re.fullmatch(re.escape(expected) + r"-beta\.[1-9][0-9]*", ref_name):
            channel = "beta"
        else:
            raise ReleaseIdentityError(
                f"constants.py APP_VERSION ({expected}) != tag ({ref_name or ref}).")
        label = ref_name
        tag = ref_name
    else:
        suffix = str(run_number or "").strip()
        if not suffix.isdigit():
            suffix = re.sub(r"[^A-Za-z0-9]", "", str(sha or ""))[:8]
        if not suffix:
            raise ReleaseIdentityError(
                "A branch build needs GITHUB_RUN_NUMBER or GITHUB_SHA for its label")
        label = f"v{version}-preview.{suffix}"
        channel = "preview"
        tag = ""

    if not SAFE_LABEL_RE.fullmatch(label):
        raise ReleaseIdentityError(
            f"Build label '{label}' is not safe for file names and installers")
    return {"version": version, "label": label, "channel": channel, "tag": tag}


def _append_github_output(path: str, values: dict) -> None:
    with open(path, "a", encoding="utf-8") as stream:
        for key, value in values.items():
            stream.write(f"{key}={value}\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--github-output", default="",
                        help="Append resolved values to this GITHUB_OUTPUT file")
    parser.add_argument("--version", default="",
                        help="Override constants.APP_VERSION (tests only)")
    parser.add_argument("--ref", default=os.environ.get("GITHUB_REF", ""))
    parser.add_argument("--ref-name", default=os.environ.get("GITHUB_REF_NAME", ""))
    parser.add_argument("--run-number", default=os.environ.get("GITHUB_RUN_NUMBER", ""))
    parser.add_argument("--sha", default=os.environ.get("GITHUB_SHA", ""))
    parser.add_argument("--checkout-sha", default="",
                        help="Full SHA of the actual checked-out commit")
    parser.add_argument("--write-build-identity", default="",
                        help="Write a Python marker for the packaged build")
    args = parser.parse_args(argv)

    try:
        version = args.version or read_app_version()
        identity = resolve_identity(args.ref, args.ref_name, args.run_number,
                                    version, args.sha)
        if args.checkout_sha:
            if not re.fullmatch(r"[0-9a-f]{40}", args.checkout_sha):
                raise ReleaseIdentityError("checkout SHA must be a full lowercase commit SHA")
            identity["sha"] = args.checkout_sha
    except ReleaseIdentityError as exc:
        print(f"::error::{exc}", file=sys.stderr)
        return 1
    if args.write_build_identity:
        suffix = identity["tag"][len("v" + version):] if identity["channel"] == "beta" else ""
        Path(args.write_build_identity).write_text(
            '"""Generated build-time beta identity."""\nBETA_SUFFIX = ' + repr(suffix) + '\n',
            encoding="utf-8")

    if args.github_output:
        _append_github_output(args.github_output, identity)
    print(
        f"{identity['channel']} build: version={identity['version']} "
        f"label={identity['label']}"
        + (f" tag={identity['tag']}" if identity["tag"] else "")
    )
    return 0


def file_record(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"name": path.name, "size": path.stat().st_size, "sha256": digest.hexdigest()}


def dist_records(directory):
    files = sorted(directory.iterdir())
    if (len(files) != 2 or any(not p.is_file() for p in files)
            or sum(p.name.endswith(".whl") for p in files) != 1
            or sum(p.name.endswith(".tar.gz") for p in files) != 1):
        raise ReleaseIdentityError("dist must contain one wheel and one source archive")
    records = [file_record(p) for p in files]
    if any(not item["size"] for item in records):
        raise ReleaseIdentityError("empty distribution")
    return records


def request(url, *, token=None, accept="application/vnd.github+json"):
    headers = {"Accept": accept, "User-Agent": "OpenFocus-release-verifier"}
    if token:
        headers["Authorization"] = "Bearer " + token
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60)


def json_request(url, *, token=None, missing_ok=False):
    try:
        with request(url, token=token) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        if missing_ok and exc.code == 404:
            return None
        raise


def remote_tag_sha(tag):
    output = subprocess.check_output(["git", "ls-remote", "--tags", "origin",
        "refs/tags/" + tag, "refs/tags/" + tag + "^{}"], text=True)
    refs = {ref: sha for sha, ref in (line.split() for line in output.splitlines())}
    return refs.get("refs/tags/" + tag + "^{}", refs.get("refs/tags/" + tag))


def check_tag(identity):
    if remote_tag_sha(identity["tag"]) != identity["sha"]:
        raise ReleaseIdentityError("remote tag no longer points to the build commit")


def validate_release_identity(identity, desktop):
    if (type(desktop.get("schema_version")) is not int or desktop["schema_version"] != 1
            or any(desktop.get(key) != identity[key] for key in ("tag", "version", "sha"))
            or any(type(desktop.get(key)) is not int or desktop[key] < 1
                   for key in ("run_id", "run_attempt"))):
        raise ReleaseIdentityError("desktop build identity mismatch")
    records = desktop.get("assets")
    if not isinstance(records, list) or not records:
        raise ReleaseIdentityError("desktop identity has no assets")
    names = set()
    for item in records:
        if (not isinstance(item, dict) or not isinstance(item.get("name"), str)
                or not item["name"] or item["name"] in names
                or type(item.get("size")) is not int or item["size"] <= 0
                or not isinstance(item.get("sha256"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"])):
            raise ReleaseIdentityError("invalid desktop asset record")
        names.add(item["name"])


def verify_remote_assets(release, desktop, identity_name, token):
    assets = release.get("assets", [])
    by_name = {asset["name"]: asset for asset in assets}
    expected = {item["name"]: item for item in desktop["assets"]}
    if identity_name in expected:
        raise ReleaseIdentityError("desktop identity cannot list itself as an asset")
    if len(by_name) != len(assets) or set(by_name) != set(expected) | {identity_name}:
        raise ReleaseIdentityError("published release asset set differs from its identity")
    for name, record in expected.items():
        asset = by_name[name]
        if asset.get("size") != record["size"]:
            raise ReleaseIdentityError(f"release asset size mismatch: {name}")
        digest = asset.get("digest")
        if not digest:
            sha = hashlib.sha256()
            size = 0
            with request(asset["url"], token=token, accept="application/octet-stream") as response:
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    sha.update(block)
                    size += len(block)
            if size != record["size"]:
                raise ReleaseIdentityError(f"downloaded asset size mismatch: {name}")
            digest = "sha256:" + sha.hexdigest()
        if digest != "sha256:" + record["sha256"]:
            raise ReleaseIdentityError(f"release asset digest mismatch: {name}")


def required_jobs_success(attempt_jobs, sha):
    required = {f"release-tests / pytest ({os_name})" for os_name in
                ("ubuntu-latest", "windows-latest", "macos-latest")}
    required.update({"release-tests / Installed wheel without GUI",
                     "release-tests / Release test summary", "Publish GitHub Release"})
    found = {}
    for jobs in attempt_jobs:
        for job in jobs:
            if job.get("name") in required and job["name"] not in found:
                found[job["name"]] = job
    return (set(found) == required and all(
        job.get("head_sha") == sha and job.get("status") == "completed"
        and job.get("conclusion") == "success" for job in found.values()))


def api_jobs(base, run_id, attempt, token):
    result = []
    page = 1
    while True:
        data = json_request(f"{base}/actions/runs/{run_id}/attempts/{attempt}/jobs?per_page=100&page={page}", token=token)
        jobs = data["jobs"]
        result.extend(jobs)
        if len(jobs) < 100:
            return result
        page += 1


def api_release_assets(base, release_id, token):
    assets = []
    page = 1
    while True:
        batch = json_request(f"{base}/releases/{release_id}/assets?per_page=100&page={page}", token=token)
        if not isinstance(batch, list):
            raise ReleaseIdentityError("invalid release asset response")
        assets.extend(batch)
        if len(batch) < 100:
            return assets
        page += 1


def wait_for_desktop(identity, base, token, timeout=5400, interval=30):
    deadline = time.monotonic() + timeout
    check_tag(identity)
    selected = None
    while time.monotonic() < deadline:
        if selected is None:
            tag = urllib.parse.quote(identity["tag"], safe="")
            release = json_request(f"{base}/releases/tags/{tag}", token=token, missing_ok=True)
            if release and not release.get("draft"):
                if release.get("prerelease") or release.get("tag_name") != identity["tag"]:
                    raise ReleaseIdentityError("published release is not the requested stable tag")
                release_id = release.get("id")
                if type(release_id) is not int or release_id < 1:
                    raise ReleaseIdentityError("invalid release ID")
                release["assets"] = api_release_assets(base, release_id, token)
                name = f"OpenFocus-{identity['tag']}-build.json"
                matching = [asset for asset in release.get("assets", []) if asset.get("name") == name]
                if len(matching) != 1:
                    raise ReleaseIdentityError("published release has no unique build identity")
                with request(matching[0]["url"], token=token, accept="application/octet-stream") as response:
                    desktop = json.load(response)
                validate_release_identity(identity, desktop)
                verify_remote_assets(release, desktop, name, token)
                run = json_request(f"{base}/actions/runs/{desktop['run_id']}", token=token)
                attempt = run.get("run_attempt")
                if type(attempt) is not int or attempt < desktop["run_attempt"]:
                    raise ReleaseIdentityError("invalid desktop workflow attempt")
                selected = (desktop, attempt)
        if selected:
            desktop, attempt = selected
            run = json_request(f"{base}/actions/runs/{desktop['run_id']}/attempts/{attempt}", token=token)
            if (run.get("id") != desktop["run_id"] or run.get("run_attempt") != attempt
                    or run.get("path", "").split("@", 1)[0] != ".github/workflows/release.yml"
                    or run.get("head_sha") != identity["sha"]):
                raise ReleaseIdentityError("desktop workflow run identity mismatch")
            if run.get("status") == "completed":
                if run.get("conclusion") != "success":
                    raise ReleaseIdentityError("selected desktop workflow attempt failed or was cancelled")
                jobs = [api_jobs(base, desktop["run_id"], number, token)
                        for number in range(attempt, desktop["run_attempt"] - 1, -1)]
                if not required_jobs_success(jobs, identity["sha"]):
                    raise ReleaseIdentityError("desktop test and wheel jobs did not all succeed")
                return
        time.sleep(interval)
    raise ReleaseIdentityError("timed out waiting for the bound desktop release attempt")


def prepare_missing_pypi(identity, dist, output):
    version = urllib.parse.quote(identity["version"], safe="")
    data = json_request(f"https://pypi.org/pypi/openfocus/{version}/json", missing_ok=True)
    urls = [] if data is None else data.get("urls")
    if not isinstance(urls, list):
        raise ReleaseIdentityError("cannot verify existing PyPI files")
    existing = {}
    for item in urls:
        name = item.get("filename")
        if not name or name in existing:
            raise ReleaseIdentityError("invalid PyPI release file list")
        existing[name] = item.get("digests", {}).get("sha256")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ReleaseIdentityError("upload directory must be empty")
    for record in identity["assets"]:
        name = record["name"]
        if name in existing:
            if existing[name] != record["sha256"]:
                raise ReleaseIdentityError(f"PyPI file has an unverifiable or different digest: {name}")
        else:
            shutil.copyfile(dist / name, output / name)
    return len(list(output.iterdir()))


def pypi_main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("create", "verify", "tag"))
    parser.add_argument("--identity", required=True, type=Path)
    parser.add_argument("--dist", type=Path, default=Path("dist"))
    parser.add_argument("--upload-dir", type=Path)
    parser.add_argument("--sha", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--channel", required=True)
    parser.add_argument("--tag", default="")
    args = parser.parse_args(argv)
    expected = {"sha": args.sha, "version": args.version, "channel": args.channel, "tag": args.tag}
    if not re.fullmatch(r"[0-9a-f]{40}", args.sha):
        raise ReleaseIdentityError("invalid checkout SHA")
    if args.action == "create":
        expected["assets"] = dist_records(args.dist)
        args.identity.parent.mkdir(parents=True, exist_ok=True)
        args.identity.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
        return
    identity = json.loads(args.identity.read_text(encoding="utf-8"))
    if any(identity.get(key) != value for key, value in expected.items()):
        raise ReleaseIdentityError("distribution identity differs from build outputs")
    if args.channel != "release" or args.tag != "v" + args.version:
        raise ReleaseIdentityError("only an exact release tag may be published")
    if identity.get("assets") != dist_records(args.dist):
        raise ReleaseIdentityError("distribution files differ from build identity")
    if args.action == "tag":
        check_tag(identity)
        return
    base = os.environ.get("GITHUB_API_URL", "https://api.github.com") + "/repos/" + os.environ["GITHUB_REPOSITORY"]
    wait_for_desktop(identity, base, os.environ["GH_TOKEN"])
    if args.upload_dir is None:
        raise ReleaseIdentityError("upload directory is required")
    count = prepare_missing_pypi(identity, args.dist, args.upload_dir)
    _append_github_output(os.environ["GITHUB_OUTPUT"], {"upload_count": count})


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "pypi":
        try:
            pypi_main(sys.argv[2:])
        except (ReleaseIdentityError, OSError, ValueError, KeyError) as exc:
            print(f"::error::{exc}", file=sys.stderr)
            raise SystemExit(1)
    else:
        raise SystemExit(main())
