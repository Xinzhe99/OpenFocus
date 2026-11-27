"""Register document types and ensure valid bundle version metadata before signing."""
import plistlib
import sys
from pathlib import Path
from typing import Optional

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from constants import APP_VERSION


def configure_bundle(bundle: Path, version: Optional[str] = None) -> None:
    path = bundle / "Contents" / "Info.plist"
    with path.open("rb") as stream:
        info = plistlib.load(stream)
    resolved_version = (version or APP_VERSION).strip()
    if resolved_version:
        if info.get("CFBundleShortVersionString") in (None, "", "0.0.0"):
            info["CFBundleShortVersionString"] = resolved_version
        if info.get("CFBundleVersion") in (None, "", "0.0.0"):
            info["CFBundleVersion"] = resolved_version
    project_type = "top.openfocus.project"
    info["UTExportedTypeDeclarations"] = [{
        "UTTypeIdentifier": project_type,
        "UTTypeDescription": "OpenFocus Project",
        "UTTypeConformsTo": ["public.data", "public.content"],
        "UTTypeTagSpecification": {"public.filename-extension": ["ofproj"]},
    }]
    info["CFBundleDocumentTypes"] = [{
        "CFBundleTypeName": "OpenFocus Project",
        "CFBundleTypeRole": "Editor",
        "LSHandlerRank": "Owner",
        "LSItemContentTypes": [project_type],
        "NSExportableTypes": [project_type],
    }, {
        "CFBundleTypeName": "Source Stack",
        "CFBundleTypeRole": "Viewer",
        # Enable Dock drops without making OpenFocus the default media app.
        "LSHandlerRank": "None",
        "LSItemContentTypes": ["public.image", "public.movie", "public.folder"],
    }]
    with path.open("wb") as stream:
        plistlib.dump(info, stream)


if __name__ == "__main__":
    configure_bundle(Path(sys.argv[1]))
