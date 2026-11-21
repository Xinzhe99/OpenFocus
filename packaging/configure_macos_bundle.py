"""Register document types on the macOS bundle before signing."""
import plistlib
import sys
from pathlib import Path


def configure_bundle(bundle: Path) -> None:
    path = bundle / "Contents" / "Info.plist"
    with path.open("rb") as stream:
        info = plistlib.load(stream)
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
