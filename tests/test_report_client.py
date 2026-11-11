import json
import os
import stat
import threading
import urllib.parse
from unittest.mock import patch

import pytest

from utils.report_client import MAX_BUNDLE_BYTES, ReportClient, ReportError


class Response:
    def __init__(self, status, data):
        self.status = status
        self.data = json.dumps(data).encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read(self, limit):
        return self.data[:limit]


class Server:
    def __init__(self, fail_uploads=0):
        self.requests = []
        self.fail_uploads = fail_uploads
        self.creations = 0

    def open(self, request, timeout):
        self.requests.append((request, timeout))
        if not request.get_header("User-agent", "").startswith("OpenFocus/"):
            raise OSError("gateway rejected unknown client")
        if request.method == "POST":
            self.creations += 1
            return Response(201, {"id": "report-%d" % self.creations,
                                  "upload_token": "one-use-%d" % self.creations})
        if self.fail_uploads:
            self.fail_uploads -= 1
            raise OSError("private URL and secret")
        return Response(200, {"ok": True, "bytes": len(request.data)})


def test_protocol_and_no_network_on_construction(tmp_path):
    with patch("urllib.request.OpenerDirector.open", side_effect=AssertionError("unexpected network")):
        client = ReportClient(tmp_path)
        installation_id = client.installation_id
    server = Server()
    client._opener = server
    assert client.submit({"source": "manual"}, b"zip data", "account-secret", owner_id="user-a") == "sent"
    assert client.pending_count == 0
    post, put = (item[0] for item in server.requests)
    assert post.method == "POST"
    assert json.loads(post.data) == {"source": "manual", "installation_id": installation_id}
    assert post.get_header("Authorization") == "Bearer account-secret"
    assert put.method == "PUT" and put.data == b"zip data"
    assert put.get_header("Authorization") is None
    assert put.get_header("Content-type") == "application/zip"
    assert urllib.parse.parse_qs(urllib.parse.urlsplit(put.full_url).query) == {"token": ["one-use-1"]}
    assert all(0 < timeout <= 5 for _, timeout in server.requests)
    assert ReportClient(tmp_path).installation_id == installation_id


def test_failed_upload_gets_new_report_and_backoff_survives_restart(tmp_path):
    now = [1000]
    client = ReportClient(tmp_path, clock=lambda: now[0])
    server = Server(fail_uploads=2)
    client._opener = server
    assert client.submit({"source": "manual"}, b"bundle", "account-secret", owner_id="user-a") == "queued"
    state = (tmp_path / "reports.json").read_text()
    assert "account-secret" not in state and "one-use-1" not in state
    client = ReportClient(tmp_path, clock=lambda: now[0])
    client._opener = server
    assert client.retry_pending("account-secret", owner_id="user-a") == 0 and server.creations == 1
    now[0] += 60
    assert client.retry_pending("account-secret", owner_id="user-a") == 0 and server.creations == 2
    now[0] += 119
    assert client.retry_pending("account-secret", owner_id="user-a") == 0 and server.creations == 2
    now[0] += 1
    assert client.retry_pending("account-secret", owner_id="user-a") == 1 and server.creations == 3
    assert client.pending_count == 0
    urls = [request.full_url for request, _ in server.requests if request.method == "PUT"]
    assert len(set(urls)) == 3


def test_queue_cap_and_attempt_limit(tmp_path):
    now = [0]
    client = ReportClient(tmp_path, clock=lambda: now[0])
    client._opener = Server(fail_uploads=100)
    for subject in ("first", "second", "third", "fourth"):
        assert client.submit({"subject": subject}, b"zip") == "queued"
    queue = json.loads((tmp_path / "reports.json").read_text())["pending"]
    assert [entry["metadata"]["subject"] for entry in queue] == ["second", "third", "fourth"]
    now[0] = 60
    assert client.retry_pending() == 0
    now[0] = 180
    assert client.retry_pending() == 0
    assert client.pending_count == 0
    assert client._opener.creations == 10


def test_bounds_private_storage_and_sanitized_errors(tmp_path):
    client = ReportClient(tmp_path / "private", timeout=100)
    with pytest.raises(ReportError, match="2 MiB"):
        client.submit({}, b"x" * (MAX_BUNDLE_BYTES + 1))
    with pytest.raises(ReportError, match="metadata"):
        client.submit({"token": "secret"}, b"zip")
    with pytest.raises(ReportError, match="metadata"):
        client.submit({"summary": "x" * 16384}, b"zip")
    client._opener = Server(fail_uploads=1)
    assert client.submit({}, b"zip", "account-secret", owner_id="user-a") == "queued"
    if os.name != "nt":
        assert stat.S_IMODE((tmp_path / "private" / "reports.json").stat().st_mode) == 0o600
        assert stat.S_IMODE((tmp_path / "private").stat().st_mode) == 0o700
    assert client._opener.requests[0][1] == 5


def test_interrupted_final_attempt_is_not_repeated(tmp_path):
    now = [0]
    client = ReportClient(tmp_path, clock=lambda: now[0])
    client._opener = Server(fail_uploads=10)
    client.submit({}, b"zip")
    path = tmp_path / "reports.json"
    state = json.loads(path.read_text())
    state["pending"][0]["attempts"] = 3
    path.write_text(json.dumps(state))
    now[0] = 1000
    assert client.retry_pending() == 0
    assert client.pending_count == 0
    assert client._opener.creations == 1


def test_retry_respects_owner_across_logout_and_account_switch(tmp_path):
    now = [0]
    client = ReportClient(tmp_path, clock=lambda: now[0])
    server = Server(fail_uploads=2)
    client._opener = server
    assert client.submit({"subject": "anonymous"}, b"zip") == "queued"
    assert client.submit({"subject": "account"}, b"zip", "token-a", owner_id="user-a") == "queued"
    now[0] = 60
    assert client.retry_pending("token-b", owner_id="user-b") == 1
    anonymous_post = [req for req, _ in server.requests if req.method == "POST"][-1]
    assert json.loads(anonymous_post.data)["subject"] == "anonymous"
    assert anonymous_post.get_header("Authorization") is None
    assert client.pending_count == 1
    assert client.retry_pending() == 0
    assert client.pending_count == 1
    assert client.retry_pending("new-token-a", owner_id="user-a") == 1
    account_post = [req for req, _ in server.requests if req.method == "POST"][-1]
    assert account_post.get_header("Authorization") == "Bearer new-token-a"
    assert client.pending_count == 0


def test_owner_and_token_must_be_supplied_together(tmp_path):
    client = ReportClient(tmp_path)
    with pytest.raises(ReportError):
        client.submit({}, b"zip", "token")
    with pytest.raises(ReportError):
        client.retry_pending(owner_id="user-a")
    assert client.pending_count == 0


def test_offline_account_enqueue_waits_for_matching_session(tmp_path):
    client = ReportClient(tmp_path)
    with patch("urllib.request.OpenerDirector.open", side_effect=AssertionError("unexpected network")):
        assert client.enqueue({"subject": "offline"}, b"zip", owner_id="user-a") == "queued"
    state = json.loads((tmp_path / "reports.json").read_text())
    assert state["pending"][0]["owner_id"] == "user-a"
    assert state["pending"][0]["attempts"] == 0
    client._opener = Server()
    assert client.retry_pending() == 0
    assert client.retry_pending("token-b", owner_id="user-b") == 0
    assert not client._opener.requests
    assert client.retry_pending("token-a", owner_id="user-a") == 1
    post = client._opener.requests[0][0]
    assert post.get_header("Authorization") == "Bearer token-a"
    assert client.pending_count == 0


def test_retry_skips_automatic_reports_when_disabled(tmp_path):
    client = ReportClient(tmp_path)
    client.enqueue({"source": "auto"}, b"auto")
    client.enqueue({"source": "manual"}, b"manual")
    client._opener = Server()
    assert client.retry_pending(include_auto=False) == 1
    posts = [req for req, _ in client._opener.requests if req.method == "POST"]
    assert [json.loads(req.data)["source"] for req in posts] == ["manual"]
    state = json.loads((tmp_path / "reports.json").read_text())
    assert len(state["pending"]) == 1
    assert state["pending"][0]["metadata"]["source"] == "auto"
    assert state["pending"][0]["attempts"] == 0
    assert client.retry_pending(include_auto=True) == 1
    assert client.pending_count == 0


def test_discard_automatic_preserves_manual_reports_and_never_sends(tmp_path):
    client = ReportClient(tmp_path)
    client.enqueue({"source": "auto"}, b"auto", owner_id="user-a")
    client.enqueue({"source": "manual"}, b"manual")
    client.enqueue({"source": "auto"}, b"auto-anonymous")
    with patch("urllib.request.OpenerDirector.open", side_effect=AssertionError("unexpected network")):
        assert client.discard_automatic() == 2
        assert client.discard_automatic() == 0
    state = json.loads((tmp_path / "reports.json").read_text())
    assert [entry["metadata"]["source"] for entry in state["pending"]] == ["manual"]
    assert ReportClient(tmp_path).pending_count == 1
    client._opener = Server()
    assert client.retry_pending() == 1
    assert json.loads(client._opener.requests[0][0].data)["source"] == "manual"


def test_enqueue_and_submit_persists_before_resolving_and_retains_failure(tmp_path):
    client = ReportClient(tmp_path)
    client._opener = Server()
    observed = []

    def unavailable():
        state = json.loads((tmp_path / "reports.json").read_text())
        observed.append(state["pending"][0]["owner_id"])
        raise OSError("private token response")

    assert client.enqueue_and_submit({"source": "auto"}, b"zip", owner_id="user-a",
                                     token_resolver=unavailable) == "queued"
    assert not client._opener.requests
    assert observed == ["user-a"]
    assert client.retry_pending("other-token", owner_id="user-b") == 0
    assert client.retry_pending("token-a", owner_id="user-a") == 1


def test_enqueue_and_submit_resolver_runs_outside_lock_and_observes_opt_out(tmp_path):
    client = ReportClient(tmp_path)
    client._opener = Server()
    started, release = threading.Event(), threading.Event()
    result = []

    def resolve():
        started.set()
        assert release.wait(2)
        return "token-a"

    worker = threading.Thread(target=lambda: result.append(client.enqueue_and_submit(
        {"source": "auto"}, b"zip", owner_id="user-a", token_resolver=resolve)))
    worker.start()
    try:
        assert started.wait(2)
        assert client.discard_automatic() == 1
    finally:
        release.set()
        worker.join(2)
    assert result == ["dropped"]
    assert not client._opener.requests


def test_enqueue_and_submit_sends_with_owner_token_or_anonymously(tmp_path):
    client = ReportClient(tmp_path)
    client._opener = Server()
    assert client.enqueue_and_submit({}, b"zip", owner_id="user-a",
                                     token_resolver=lambda: "token-a") == "sent"
    assert client.enqueue_and_submit({}, b"zip") == "sent"
    posts = [request for request, _ in client._opener.requests if request.method == "POST"]
    assert posts[0].get_header("Authorization") == "Bearer token-a"
    assert posts[1].get_header("Authorization") is None


@pytest.mark.parametrize('signed_in,token', [(False, None), (True, None), (True, 'invalid'), (True, 'token-a')])
def test_account_default_retry_never_sends_anonymous_automatic(
        tmp_path, monkeypatch, signed_in, token):
    from utils import reporting

    client = ReportClient(tmp_path)
    client._opener = Server()
    client.enqueue({'source': 'auto', 'subject': 'anonymous'}, b'auto')
    client.enqueue({'source': 'manual', 'subject': 'manual'}, b'manual')
    client.enqueue({'source': 'auto', 'subject': 'account'}, b'account', owner_id='user-a')
    monkeypatch.setattr(reporting, 'ReportClient', lambda _path: client)

    class Account:
        def current_user(self):
            return {'id': 'user-a'} if signed_in else None

        def token(self, *, expected_owner):
            if token == 'invalid':
                from utils.account_client import AccountError
                raise AccountError('invalid_session')
            assert expected_owner == 'user-a'
            return token

    reporting.retry_reports(Account(), include_auto=True, account_only_auto=True)
    posts = [request for request, _ in client._opener.requests if request.method == 'POST']
    subjects = [json.loads(request.data)['subject'] for request in posts]
    assert 'manual' in subjects
    assert 'anonymous' not in subjects
    assert ('account' in subjects) == (signed_in and token == 'token-a')
    assert client.pending_count == (1 if signed_in and token == 'token-a' else 2)


def test_explicit_opt_in_retries_anonymous_automatic(tmp_path, monkeypatch):
    from utils import reporting

    client = ReportClient(tmp_path)
    client._opener = Server()
    client.enqueue({'source': 'auto'}, b'auto')
    monkeypatch.setattr(reporting, 'ReportClient', lambda _path: client)
    account = type('Account', (), {'current_user': lambda self: None})()
    reporting.retry_reports(account, include_auto=True)
    assert client.pending_count == 0


def test_account_default_send_rejects_signed_out_session(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from utils import reporting

    client = ReportClient(tmp_path)
    client._opener = Server()
    monkeypatch.setattr(reporting, 'ReportClient', lambda _path: client)
    account = SimpleNamespace(current_user=lambda: None, is_signed_in=lambda: False)
    with pytest.raises(ReportError):
        reporting.send_bundle(None, source='auto', locale='en', account=account,
                              require_account=True)
    assert not client._opener.requests
    assert client.pending_count == 0


@pytest.mark.parametrize('signed_in,account_only,expected', [
    (False, True, False), (True, True, True), (False, False, True),
])
def test_shutdown_uses_persisted_session_for_default_scope(
        qapp, tmp_path, monkeypatch, signed_in, account_only, expected):
    from types import SimpleNamespace
    from ui import shutdown_reports

    account = SimpleNamespace(is_signed_in=lambda: signed_in)
    monkeypatch.setattr(shutdown_reports, 'AccountClient', lambda: account)
    monkeypatch.setattr(shutdown_reports, 'user_data_dir', lambda: tmp_path)
    monkeypatch.setattr(shutdown_reports, 'session_has_error', lambda: True)
    monkeypatch.setattr(shutdown_reports, 'build_report_bundle',
                        lambda: SimpleNamespace(has_error=True))
    sent = []
    monkeypatch.setattr(shutdown_reports, 'send_bundle',
                        lambda *args, **kwargs: sent.append(kwargs))
    shutdown_reports._ShutdownReport(None, 'en', True, account_only)._run()
    assert bool(sent) is expected
    if sent:
        assert sent[0]['account'] is account
        assert sent[0]['require_account'] is account_only
