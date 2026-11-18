import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import vm from 'node:vm';

const script = await readFile(new URL('../site/app.js', import.meta.url), 'utf8');

async function visit(response) {
  const navDownload = { dataset: { urlWin: '/windows', urlMac: '/mac', urlLinux: '#download' }, href: '/windows' };
  const status = { dataset: { available: 'available', unavailable: 'not available yet' }, textContent: 'availability unknown' };
  const download = { hidden: true, href: '/download/latest/linux-desktop' };
  const install = { hidden: true };
  const command = { textContent: '' };
  const version = { textContent: '' };
  const document = {
    documentElement: { dataset: {} },
    querySelector(selector) {
      return { '#linux-status': status, '#linux-download': download, '#linux-install': install,
        '#linux-command': command }[selector] || null;
    },
    querySelectorAll(selector) {
      return { '[data-url-win]': [navDownload], '[data-release-tag]': [version] }[selector] || [];
    },
  };
  const context = { document, window: {}, navigator: { userAgent: 'Mozilla/5.0 (X11; Linux x86_64)' },
    fetch: async () => { if (response instanceof Error) throw response; return response; } };
  vm.runInNewContext(script, context);
  await new Promise((resolve) => setImmediate(resolve));
  return { status, download, navDownload, install, command, version };
}

test('v1.76.0 displays a stable version but has no Linux archive', async () => {
  const page = await visit({ ok: true, json: async () => ({ tag: 'v1.76.0', assets: {} }) });
  assert.equal(page.version.textContent, 'v1.76.0 · ');
  assert.equal(page.status.textContent, 'not available yet');
  assert.equal(page.download.hidden, true);
  assert.equal(page.install.hidden, true);
  assert.equal(page.navDownload.href, '#download');
});

test('beta metadata never sets public download or version UI', async () => {
  for (const tag of ['v1.77.0-beta.7', 'v1.77.0.999999999.7']) {
    const page = await visit({ ok: true, json: async () => ({ tag, assets: { 'linux-desktop': {} } }) });
    assert.equal(page.version.textContent, '');
    assert.equal(page.status.textContent, 'availability unknown');
    assert.equal(page.download.hidden, true);
    assert.equal(page.install.hidden, true);
  }
});

test('Linux desktop becomes available from release metadata', async () => {
  const page = await visit({ ok: true, json: async () => ({ tag: 'v1.78.0', assets: { 'linux-desktop': { source: 'r2', url: null } } }) });
  assert.equal(page.status.textContent, 'available');
  assert.equal(page.download.hidden, false);
  assert.equal(page.download.href, '/download/latest/linux-desktop');
  assert.equal(page.install.hidden, false);
  assert.equal(page.command.textContent,
    'mkdir openfocus-v1.78.0 &&\ntar -xzf OpenFocus-v1.78.0-linux-x86_64.tar.gz -C openfocus-v1.78.0 &&\ncd openfocus-v1.78.0 &&\n./OpenFocus/OpenFocus');
  assert.equal(page.navDownload.href, '#download');
});

test('metadata failure leaves availability unknown and the download hidden', async () => {
  for (const response of [new Error('network failure'), { ok: false }, { ok: true, json: async () => ({ tag: 'v1.77.0' }) }]) {
    const page = await visit(response);
    assert.equal(page.status.textContent, 'availability unknown');
    assert.equal(page.download.hidden, true);
    assert.equal(page.install.hidden, true);
    assert.equal(page.navDownload.href, '#download');
  }
});
