"""Consumer-facing website download and availability contracts."""

from html.parser import HTMLParser
from pathlib import Path
import shutil
import subprocess

import pytest


class Elements(HTMLParser):
    def __init__(self):
        super().__init__()
        self.items = []

    def handle_starttag(self, tag, attrs):
        self.items.append((tag, dict(attrs)))


def test_localized_pages_keep_linux_download_safe_without_javascript():
    from tools import build_site

    template = (build_site.SITE / 'template.html').read_text(encoding='utf-8')
    for locale in build_site.LOCALES.values():
        page = build_site.render(template, build_site.context_for(locale), locale['code'])
        elements = Elements()
        elements.feed(page)
        linux = next(attrs for tag, attrs in elements.items if attrs.get('id') == 'linux-download')
        status = next(attrs for tag, attrs in elements.items if attrs.get('id') == 'linux-status')
        nav = next(attrs for tag, attrs in elements.items if 'data-url-linux' in attrs)
        assert linux['href'] == '/download/latest/linux-desktop'
        assert 'hidden' in linux
        assert status['role'] == 'status'
        assert nav['data-url-linux'] == '#download'
        assert 'OpenFocus-v1.77.0-linux-x86_64.tar.gz' in page
        assert './OpenFocus/OpenFocus' in page
        assert 'pip install openfocus' in page
        assert locale['download']['linux_unknown'] in page
        assert locale['download']['script_label'] in page


def test_linux_download_release_states():
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node.js is needed for the website interaction tests')
    test_file = Path(__file__).with_name('site_download.test.mjs')
    result = subprocess.run([node, '--test', str(test_file)], capture_output=True,
                            text=True, encoding='utf-8', timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
