"""Private environments and analytics injection for process tests."""
import os
from unittest.mock import patch


def sandbox_env(tmp_path, overrides=None, *, instance_name=None):
    scratch = tmp_path / "tmp"
    scratch.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(overrides or {})
    env.update({
        "OPENFOCUS_DATA_DIR": str(tmp_path / "data"),
        "OPENFOCUS_LOG_DIR": str(tmp_path / "logs"),
        "OPENFOCUS_CONTROL_FILE": str(tmp_path / "control.json"),
        "OPENFOCUS_INSTANCE_NAME": instance_name or f"OpenFocus-test-{os.getpid()}-{tmp_path.name}",
        "APPDATA": str(tmp_path / "appdata"),
        "XDG_CONFIG_HOME": str(tmp_path / "config"),
        "XDG_DATA_HOME": str(tmp_path / "data"),
        "HOME": str(tmp_path),
        "USERPROFILE": str(tmp_path),
        "TMP": str(scratch),
        "TEMP": str(scratch),
        "TMPDIR": str(scratch),
    })
    return env

def seed_test_account(root):
    """Give an isolated GUI a synthetic account using its normal token store."""
    from utils.account_client import TokenStore

    with patch.dict(os.environ, {'OPENFOCUS_DATA_DIR': str(root)}):
        store = TokenStore()
        store.save({'token': 'test-session-token', 'token_type': 'Bearer',
                    'expires_at': 9999999999, 'user': {'id': 'test-user'}})
    return store



def install_usage_sender():
    """Install in the child before constructing any real UsageAnalytics."""
    from utils import usage_analytics

    calls = []
    reporters = []

    def sender(payload):
        calls.append(payload)
        return False

    original_init = usage_analytics.UsageAnalytics.__init__

    def record_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        reporters.append(self.sender)

    usage_analytics._send = sender
    usage_analytics.UsageAnalytics.__init__ = record_init
    return sender, reporters, calls
