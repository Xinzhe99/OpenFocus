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


def test_built_installer_runs_shell_syntax_check_after_crlf_checkout(tmp_path, monkeypatch):
    shell = shutil.which("sh")
    if shell is None:
        pytest.skip("A POSIX shell is needed to validate the installer")

    from tools import build_site

    original_root = Path(__file__).resolve().parent.parent
    root = tmp_path / "checkout"
    site = root / "site"
    site.mkdir(parents=True)
    for name in ("template.html", "style.css", "app.js", "install.ps1"):
        shutil.copyfile(original_root / "site" / name, site / name)
    installer = (original_root / "site" / "install.sh").read_text(encoding="utf-8")
    (site / "install.sh").write_bytes(installer.replace("\n", "\r\n").encode("utf-8"))
    for name in ("OpenFocus.png", "ui.png", "benchmark.png",
                 "readme/method-comparison_poster.jpg", "readme/retouch_poster.jpg"):
        target = root / "assets" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original_root / "assets" / name, target)

    monkeypatch.setattr(build_site, "ROOT", root)
    monkeypatch.setattr(build_site, "SITE", site)
    monkeypatch.setattr(build_site, "DIST", site / "dist")
    monkeypatch.setattr(build_site, "ASSETS", site / "dist" / "assets")
    assert build_site.main() == 0
    result = subprocess.run([shell, "-n", str(site / "dist" / "install.sh")],
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
