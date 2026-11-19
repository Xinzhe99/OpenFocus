"""Private environments and analytics injection for process tests."""
import os


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
        "LOCALAPPDATA": str(tmp_path / "localappdata"),
        "XDG_CONFIG_HOME": str(tmp_path / "config"),
        "XDG_DATA_HOME": str(tmp_path / "data"),
        "HOME": str(tmp_path),
        "USERPROFILE": str(tmp_path),
        "TMP": str(scratch),
        "TEMP": str(scratch),
        "TMPDIR": str(scratch),
    })
    return env

def install_test_account():
    """Install a saved synthetic session in this isolated child process."""
    from pathlib import Path
    from utils import account_client

    session_path = Path(os.environ["OPENFOCUS_DATA_DIR"]) / "test-account-session"
    token_store = account_client.TokenStore
    store = token_store(session_path)
    session = {'token': 'test-session-token', 'token_type': 'Bearer',
               'expires_at': 9999999999, 'user': {'id': 'test-user'}}
    store.save(session)
    assert store.load() == session

    def isolated_store(path=None):
        return token_store(session_path if path is None else path)

    account_client.TokenStore = isolated_store

    def unexpected_request(*args, **kwargs):
        raise AssertionError("An isolated GUI must not contact the account service")

    account_client._http = unexpected_request


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
