"""Cloudflare Worker contracts and built installer portability."""

from pathlib import Path
import shutil
import subprocess

import pytest


def test_site_worker_contract():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed for the site worker tests")

    test_file = Path(__file__).with_name("site_worker.test.mjs")
    completed = subprocess.run(
        [node, "--test", str(test_file)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.fixture
def built_site(tmp_path, monkeypatch):
    from tools import build_site

    original_root = Path(__file__).resolve().parent.parent
    root = tmp_path / "checkout"
    site = root / "site"
    site.mkdir(parents=True)
    for name in ("template.html", "contributors_template.html", "contributors.json", "style.css", "app.js", "install.ps1"):
        shutil.copyfile(original_root / "site" / name, site / name)
    installer = (original_root / "site" / "install.sh").read_text(encoding="utf-8")
    (site / "install.sh").write_bytes(installer.replace("\n", "\r\n").encode("utf-8"))
    for name in ("OpenFocus.png", "ui.png", "benchmark.png",
                 "readme/method-comparison_poster.jpg", "readme/retouch_poster.jpg"):
        target = root / "assets" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original_root / "assets" / name, target)

    dist = site / "dist"
    dist.mkdir()
    qr_files = ("wechat-group.png", "wechat-official-account.jpg")
    for name in qr_files:
        (dist / name).write_bytes(b"stale")

    monkeypatch.setattr(build_site, "ROOT", root)
    monkeypatch.setattr(build_site, "SITE", site)
    monkeypatch.setattr(build_site, "DIST", dist)
    monkeypatch.setattr(build_site, "ASSETS", dist / "assets")
    assert build_site.main() == 0
    return dist


def test_built_site_drops_legacy_qr_files(built_site):
    headers = (built_site / "_headers").read_text(encoding="utf-8")
    for name in ("wechat-group.png", "wechat-official-account.jpg"):
        assert not (built_site / name).exists()
        assert f"/{name}\n" not in headers


def test_built_installer_runs_shell_syntax_check_after_crlf_checkout(built_site):
    shell = shutil.which("sh")
    if shell is None:
        pytest.skip("A POSIX shell is needed to validate the installer")
    result = subprocess.run([shell, "-n", str(built_site / "install.sh")],
                            capture_output=True, text=True, encoding="utf-8")
    assert result.returncode == 0, result.stderr




def test_theme_state_uses_template_script():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed for the site theme test")
    root = Path(__file__).resolve().parent.parent
    css = (root / "site/style.css").read_text(encoding="utf-8")
    assert ".theme { display: none; }" in css
    assert ".theme[data-enhanced] { display: inline-grid; }" in css
    program = r"""
const fs = require('fs'), vm = require('vm'), assert = require('assert');
const template = fs.readFileSync(process.argv[1], 'utf8');
const head = template.match(/<script>\s*([\s\S]*?)<\/script>/)[1];
const app = fs.readFileSync(process.argv[2], 'utf8');
const buttonCode = app.slice(0, app.indexOf('  // The navigation download')) + '})();';
function page(saved, blocked, systemDark) {
  const store = new Map(saved ? [['theme', saved]] : []);
  const media = {matches: systemDark, addEventListener: (_, fn) => media.change = fn};
  const root = {dataset: {}, classList: {add() {}}};
  const meta = {};
  const button = {dataset: {}, addEventListener: (_, fn) => button.click = fn};
  const document = {documentElement: root, querySelector: s => s === '#theme' ? button : meta};
  const localStorage = {
    getItem: k => { if (blocked) throw Error('blocked'); return store.get(k) || null; },
    setItem: (k, v) => { if (blocked) throw Error('blocked'); store.set(k, v); }
  };
  const context = {document, localStorage, matchMedia: () => media, window: {}};
  vm.runInNewContext(head, context);
  assert.equal('enhanced' in button.dataset, false);
  vm.runInNewContext(buttonCode, context);
  assert.equal('enhanced' in button.dataset, true);
  return {root, meta, media, store, button, win: context.window};
}
for (const blocked of [false, true]) {
  const p = page(null, blocked, true);
  assert.equal(p.win.__themeMode(), blocked ? 'system' : 'light');
  if (blocked) p.button.click(); // system -> light
  for (const expected of ['dark', 'system', 'light']) {
    p.button.click();
    assert.equal(p.win.__themeMode(), expected);
    assert.equal(p.root.dataset.mode, expected);
    if (expected === 'system') {
      if (!blocked) assert.equal(p.store.get('theme'), 'system');
      assert.equal(p.root.dataset.theme, 'dark');
      p.media.matches = false; p.media.change();
      assert.equal(p.root.dataset.theme, 'light');
      assert.equal(p.meta.content, '#fafafa');
      p.media.matches = true; p.media.change();
      assert.equal(p.root.dataset.theme, 'dark');
      assert.equal(p.meta.content, '#0b0b0c');
    }
  }
  p.media.matches = false; p.media.change();
  assert.equal(p.root.dataset.theme, 'light');
}
for (const mode of ['light', 'dark', 'system']) {
  const p = page(mode, false, true);
  assert.equal(p.win.__themeMode(), mode);
  assert.equal(p.root.dataset.theme, mode === 'system' ? 'dark' : mode);
}
"""
    result = subprocess.run(
        [node, "-e", program, str(root / "site/template.html"), str(root / "site/app.js")],
        capture_output=True, text=True, encoding="utf-8", timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_built_contributors_pages_render_external_reporters_without_maintainer(built_site):
    en_html = (built_site / "contributors/index.html").read_text(encoding="utf-8")
    zh_html = (built_site / "zh/contributors/index.html").read_text(encoding="utf-8")
    sitemap = (built_site / "sitemap.xml").read_text(encoding="utf-8")
    headers = (built_site / "_headers").read_text(encoding="utf-8")

    for handle in ("AlanRockefeller", "m1ll3t", "taraskin45-glitch", "hzweifei", "Phreemann", "pixelsurium"):
        assert f"@{handle}" in en_html
        assert f"@{handle}" in zh_html
    for version in ("v2.0.0", "v1.52", "v1.26", "v1.9"):
        assert version in en_html
        assert version in zh_html
    for issue in ("#329", "#294", "#246", "#5", "#4", "#2"):
        assert issue in en_html
        assert issue in zh_html

    assert "@Xinzhe99" not in en_html
    assert "@Xinzhe99" not in zh_html
    assert "<details" in en_html and "5 more" in en_html
    assert "<details" in zh_html and "还有 5 项" in zh_html
    assert 'href="/contributors/?lang=en"' in en_html
    assert 'href="/zh/contributors/?lang=zh"' in zh_html
    assert "https://openfocus.top/contributors/" in sitemap
    assert "https://openfocus.top/zh/contributors/" in sitemap
    assert "/contributors/\n" in headers
    assert "/zh/contributors/\n" in headers


def test_contributors_loader_rejects_excluded_author(tmp_path, monkeypatch):
    import json
    from tools import build_site

    site = tmp_path / "site"
    site.mkdir()
    (site / "contributors.json").write_text(json.dumps({
        "excluded_authors": ["Xinzhe99"],
        "contributors": [{
            "handle": "Xinzhe99",
            "url": "https://github.com/Xinzhe99",
            "contributions": [{"issue": 1, "version": "v1.0", "date": "2026-01-01", "en": "a", "zh": "a"}],
        }],
    }), encoding="utf-8")
    monkeypatch.setattr(build_site, "SITE", site)
    with pytest.raises(ValueError, match="excluded contributor"):
        build_site.load_contributors()


def test_sync_contributors_merges_promotes_and_filters():
    from tools import sync_contributors

    existing = {
        "excluded_authors": ["Xinzhe99"],
        "opt_out_authors": ["NoCreditUser"],
        "excluded_issues": [10],
        "contributors": [
            {
                "handle": "AlanRockefeller",
                "platform": "GitHub",
                "url": "https://github.com/AlanRockefeller",
                "contributions": [
                    {
                        "issue": 329,
                        "version": "next",
                        "date": "2026-11-28",
                        "kind": "bug",
                        "en": "Curated English text.",
                        "zh": "人工润色中文。",
                    }
                ],
            },
            {
                "handle": "ReopenedUser",
                "platform": "GitHub",
                "url": "https://github.com/ReopenedUser",
                "contributions": [
                    {
                        "issue": 99,
                        "version": "next",
                        "date": "2026-11-20",
                        "kind": "bug",
                        "en": "Stale",
                        "zh": "Stale",
                    }
                ],
            },
        ],
    }
    issues = [
        {"number": 329, "state": "closed", "closed_at": "2026-11-28T07:00:00Z", "user": {"login": "AlanRockefeller"}, "title": "Raw title"},
        {"number": 345, "state": "closed", "closed_at": "2026-11-29T08:00:00Z", "user": {"login": "NewReporter"}, "title": "[Feature] Add dark grid", "labels": [{"name": "enhancement"}]},
        {"number": 346, "state": "closed", "closed_at": "2026-11-29T09:00:00Z", "user": {"login": "Xinzhe99"}, "title": "Maintainer issue"},
        {"number": 347, "state": "closed", "closed_at": "2026-11-29T09:00:00Z", "user": {"login": "NoCreditUser"}, "title": "Opt out"},
        {"number": 348, "state": "closed", "closed_at": "2026-11-29T09:00:00Z", "user": {"login": "bot[bot]"}, "title": "Bot"},
        {"number": 10, "state": "closed", "closed_at": "2026-11-29T09:00:00Z", "user": {"login": "SomeUser"}, "title": "Excluded issue"},
        {"number": 349, "state": "closed", "closed_at": "2026-11-29T09:00:00Z", "user": {"login": "SomeUser"}, "title": "PR item", "pull_request": {}},
        {"number": 350, "state": "closed", "state_reason": "not_planned", "closed_at": "2026-11-29T09:00:00Z", "user": {"login": "SomeUser"}, "title": "Not planned"},
    ]
    releases = [
        {"tag_name": "v2.0.1-beta.1", "draft": False, "prerelease": True, "published_at": "2026-11-28T10:00:00Z"},
        {"tag_name": "v2.0.1", "draft": False, "prerelease": False, "published_at": "2026-11-28T12:00:00Z"},
    ]

    merged = sync_contributors.merge_contributors(existing, issues, releases)
    handles = [p["handle"] for p in merged["contributors"]]
    assert handles == ["NewReporter", "AlanRockefeller"]
    assert merged["contributors"][0]["contributions"][0]["version"] == "next"
    assert merged["contributors"][0]["contributions"][0]["kind"] == "feature"
    assert merged["contributors"][0]["contributions"][0]["en"] == "Add dark grid"
    # Issue #329 closed before v2.0.1 is promoted automatically while keeping curated EN/ZH copy.
    alan_item = merged["contributors"][1]["contributions"][0]
    assert alan_item["version"] == "v2.0.1"
    assert alan_item["en"] == "Curated English text."
    assert alan_item["zh"] == "人工润色中文。"
