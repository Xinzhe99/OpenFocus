"""Resolve the release identity used by the release workflow.

A tag build must match ``constants.APP_VERSION`` exactly. A branch
``workflow_dispatch`` run is an acceptance build: it gets a safe preview label
(``v<version>-preview.<run>``) for file names and never reaches the release job.

Run from the repository root:
    python tools/release_metadata.py --github-output "$GITHUB_OUTPUT"
"""
from __future__ import annotations

import argparse
import os
import re
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

    Tags are release builds and keep the exact ``v<version>`` name. Anything
    else is a preview: a run number (or short commit when GitHub does not
    provide one) keeps concurrent branch builds distinguishable, and the label
    contains no slashes or spaces so every packaging command can use it
    verbatim.
    """
    version = str(version).strip()
    if not VERSION_RE.fullmatch(version):
        raise ReleaseIdentityError(
            f"constants.APP_VERSION '{version}' is not a dotted numeric version")

    ref = str(ref or "")
    ref_name = str(ref_name or "")
    if ref.startswith(TAG_REF_PREFIX):
        expected = "v" + version
        if ref_name != expected:
            raise ReleaseIdentityError(
                f"constants.py APP_VERSION ({expected}) != tag ({ref_name or ref}).")
        label = ref_name
        channel = "release"
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
    args = parser.parse_args(argv)

    try:
        version = args.version or read_app_version()
        identity = resolve_identity(args.ref, args.ref_name, args.run_number,
                                    version, args.sha)
    except ReleaseIdentityError as exc:
        print(f"::error::{exc}", file=sys.stderr)
        return 1

    if args.github_output:
        _append_github_output(args.github_output, identity)
    print(
        f"{identity['channel']} build: version={identity['version']} "
        f"label={identity['label']}"
        + (f" tag={identity['tag']}" if identity["tag"] else "")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
