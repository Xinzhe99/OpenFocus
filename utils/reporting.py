"""Prepare and deliver consented, sanitized diagnostic reports."""
import sys

from constants import app_display_version
from utils.account_client import AccountClient, AccountError
from utils.app_paths import user_data_dir
from utils.report_client import ReportClient, ReportError


def build_channel():
    return "release" if getattr(sys, "frozen", False) else "development"


def report_metadata(bundle, *, source, locale, severity='error'):
    return {
        'fingerprint': bundle.fingerprint,
        'app_version': app_display_version(),
        'platform': bundle.platform,
        'severity': severity,
        'source': source,
        'build_channel': build_channel(),
        'locale': 'zh' if locale == 'zh' else 'en',
        'subject': bundle.subject,
        'summary': bundle.summary,
        'log_excerpt': bundle.log_excerpt,
        'occurred_at': bundle.occurred_at,
    }


def send_bundle(bundle, *, source, locale, account=None, severity='error', require_account=False,
                permission=None, cancel=None, deadline=None):
    """Send from a background worker, retaining account scope if offline."""
    reports = ReportClient(user_data_dir() / 'reports')
    client = account if account is not None else AccountClient()
    user = client.current_user()
    owner_id = user.get('id') if isinstance(user, dict) else None
    if require_account and (not client.is_signed_in() or not owner_id):
        raise ReportError('Account identity unavailable')
    if client.is_signed_in() and (not isinstance(owner_id, str) or not owner_id):
        raise ReportError('Account identity unavailable')
    if source == 'auto' and build_channel() == 'development':
        return None
    metadata = report_metadata(bundle, source=source, locale=locale, severity=severity)
    if owner_id:
        return reports.enqueue_and_submit(
            metadata, bundle.data, owner_id=owner_id,
            token_resolver=lambda: client.token(expected_owner=owner_id),
            permission=permission, cancel=cancel, deadline=deadline)
    return reports.enqueue_and_submit(metadata, bundle.data, permission=permission,
                                      cancel=cancel, deadline=deadline)


def retry_reports(account=None, *, include_auto=False, account_only_auto=False,
                  permission=None, cancel=None, deadline=None):
    """Retry queued reports only under their original account or anonymously."""
    reports = ReportClient(user_data_dir() / 'reports')
    client = account if account is not None else AccountClient()
    user = client.current_user()
    owner_id = user.get('id') if isinstance(user, dict) else None
    if owner_id:
        try:
            token = client.token(expected_owner=owner_id)
        except AccountError:
            return reports.retry_pending(include_auto=include_auto and not account_only_auto,
                                         permission=permission, cancel=cancel, deadline=deadline)
        if token:
            return reports.retry_pending(token, owner_id=owner_id, include_auto=include_auto,
                                         include_anonymous_auto=not account_only_auto,
                                         permission=permission, cancel=cancel, deadline=deadline)
    return reports.retry_pending(include_auto=include_auto and not account_only_auto,
                                 permission=permission, cancel=cancel, deadline=deadline)
