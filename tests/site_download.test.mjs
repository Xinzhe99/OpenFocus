import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import vm from 'node:vm';

const script = await readFile(new URL('../site/app.js', import.meta.url), 'utf8');

async function visit(response) {
  const navDownload = { dataset: { urlWin: '/windows', urlMac: '/mac', urlLinux: '#download' }, href: '/windows' };
  const status = { dataset: { available: 'available', unavailable: 'not available yet' }, textContent: 'availability unknown' };
  const download = { hidden: true, href: '/download/latest/linux-desktop' };
  const document = {
    documentElement: { dataset: {} },
    querySelector(selector) { return ({ '#linux-status': status, '#linux-download': download })[selector] || null; },
    querySelectorAll(selector) { return selector === '[data-url-win]' ? [navDownload] : []; },
  };
  const context = { document, window: {}, navigator: { userAgent: 'Mozilla/5.0 (X11; Linux x86_64)' },
    fetch: async () => { if (response instanceof Error) throw response; return response; } };
  vm.runInNewContext(script, context);
  await new Promise((resolve) => setImmediate(resolve));
  return { status, download, navDownload };
}

for (const tag of ['v1.76.0', 'v1.77.0-beta.7']) {
  test(`${tag}: Linux desktop is unavailable`, async () => {
    const page = await visit({ ok: true, json: async () => ({ tag, assets: {} }) });
    assert.equal(page.status.textContent, 'not available yet');
    assert.equal(page.download.hidden, true);
    assert.equal(page.navDownload.href, '#download');
  });
}

test('Linux desktop becomes available from release metadata', async () => {
  const page = await visit({ ok: true, json: async () => ({ tag: 'v1.77.0', assets: { 'linux-desktop': { source: 'r2', url: null } } }) });
  assert.equal(page.status.textContent, 'available');
  assert.equal(page.download.hidden, false);
  assert.equal(page.download.href, '/download/latest/linux-desktop');
  assert.equal(page.navDownload.href, '#download');
});

test('metadata failure leaves availability unknown and the download hidden', async () => {
  for (const response of [new Error('network failure'), { ok: false }, { ok: true, json: async () => ({ tag: 'v1.77.0' }) }]) {
    const page = await visit(response);
    assert.equal(page.status.textContent, 'availability unknown');
    assert.equal(page.download.hidden, true);
    assert.equal(page.navDownload.href, '#download');
  }
});
