"""The packaged .app advertises projects and Dock drop targets to macOS."""
import plistlib
import subprocess
import sys
from pathlib import Path


def test_bundle_can_receive_projects_and_media_drops(tmp_path):
    bundle = tmp_path / "OpenFocus.app"
    info = bundle / "Contents" / "Info.plist"
    info.parent.mkdir(parents=True)
    info.write_bytes(plistlib.dumps({"CFBundleExecutable": "OpenFocus", "CFBundleVersion": "7"}))
    script = Path(__file__).resolve().parents[1] / "packaging" / "configure_macos_bundle.py"
    subprocess.run([sys.executable, str(script), str(bundle)], check=True, timeout=10)
    data = plistlib.loads(info.read_bytes())
    assert data["CFBundleExecutable"] == "OpenFocus"
    assert data["CFBundleVersion"] == "7"
    project_type = next(item["UTTypeIdentifier"] for item in data["UTExportedTypeDeclarations"]
                        if "ofproj" in item["UTTypeTagSpecification"]["public.filename-extension"])
    by_type = {uti: role for role in data["CFBundleDocumentTypes"] for uti in role["LSItemContentTypes"]}
    assert by_type[project_type]["CFBundleTypeRole"] == "Editor"
    for media in ("public.image", "public.movie", "public.folder"):
        assert by_type[media]["LSHandlerRank"] == "None"
