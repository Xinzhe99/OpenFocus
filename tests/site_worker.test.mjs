import test, { after } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";

const root = new URL("../", import.meta.url);
const nativeFetch = globalThis.fetch;
const tempDir = await mkdtemp(path.join(tmpdir(), "openfocus-worker-"));
const modulePath = path.join(tempDir, "worker.mjs");
await writeFile(modulePath, await readFile(new URL("site/worker.js", root), "utf8"));
const { default: worker } = await import(pathToFileURL(modulePath).href);

after(async () => {
  globalThis.fetch = nativeFetch;
  await rm(tempDir, { recursive: true, force: true }).catch(() => {});
});

const TAG = "v1.0.0";
const ETAG = '"r2-etag-1"';
const PAYLOAD = Uint8Array.from({ length: 10 }, (_, index) => 48 + index);
const NAMES = {
  "windows-setup": `OpenFocus-${TAG}-setup.exe`,
  "windows-zip": `OpenFocus-${TAG}-windows-x64.zip`,
  "macos-dmg": `OpenFocus-${TAG}-macos.dmg`,
  "macos-zip": `OpenFocus-${TAG}-macos-arm64.zip`,
};

function releaseBody() {
  return {
    tag_name: TAG,
    published_at: "2026-01-02T03:04:05Z",
    assets: Object.entries(NAMES).map(([, name]) => ({
      name,
      size: PAYLOAD.length,
      browser_download_url:
        `https://github.com/Xinzhe99/OpenFocus/releases/download/${TAG}/${name}`,
    })),
  };
}

function installCache() {
  const entries = new Map();
  Object.defineProperty(globalThis, "caches", {
    configurable: true,
    writable: true,
    value: {
      default: {
        async match(request) {
          const key = typeof request === "string" ? request : request.url;
          const hit = entries.get(key);
          return hit ? hit.clone() : undefined;
        },
        async put(request, response) {
          const key = typeof request === "string" ? request : request.url;
          entries.set(key, response.clone());
        },
      },
    },
  });
  return entries;
}

function installReleaseFetch({ fail = false, release = releaseBody() } = {}) {
  let calls = 0;
  globalThis.fetch = async (input) => {
    calls += 1;
    const url = typeof input === "string" ? input : input.url;
    if (!url.includes("api.github.com")) {
      throw new Error(`unexpected fetch: ${url}`);
    }
    if (fail) return new Response("upstream unavailable", { status: 503 });
    return new Response(JSON.stringify(release), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  };
  return () => calls;
}

function fakeObject(bytes) {
  return {
    size: bytes.length,
    etag: ETAG.slice(1, -1),
    httpEtag: ETAG,
    uploaded: new Date("2026-01-02T03:04:05Z"),
    writeHttpMetadata(headers) {
      headers.set("Content-Type", "application/octet-stream");
      headers.set("Cache-Control", "public, max-age=31536000");
    },
  };
}

function bucket(objects = {}) {
  const stored = new Map(
    Object.entries(objects).map(([key, bytes]) => [key, fakeObject(bytes)]),
  );
  return {
    async head(key) {
      return stored.get(key) || null;
    },
    async get(key, options) {
      const object = stored.get(key);
      if (!object) return null;
      let bytes = objects[key];
      const range = options && options.range;
      if (range) bytes = bytes.slice(range.offset, range.offset + range.length);
      return {
        ...object,
        body: new Blob([bytes]).stream(),
      };
    },
  };
}

function assets() {
  const calls = [];
  return {
    calls,
    async fetch(request) {
      const pathname = new URL(request.url).pathname;
      calls.push(pathname);
      return new Response(`asset:${pathname}`, { status: 200 });
    },
  };
}

function makeEnv(objects = {}) {
  return { DOWNLOADS: bucket(objects), ASSETS: assets() };
}

async function run(url, { method = "GET", headers = {}, body, env } = {}) {
  const request = new Request(url, { method, headers, body });
  const pending = [];
  const response = await worker.fetch(request, env, {
    waitUntil(promise) {
      pending.push(promise);
    },
  });
  await Promise.all(pending);
  return response;
}

test("single byte ranges return 206 with exact bytes and a valid Content-Range", async () => {
  installCache();
  installReleaseFetch();
  const env = makeEnv({ [NAMES["windows-setup"]]: PAYLOAD });

  const response = await run("https://openfocus.top/download/latest/windows-setup", {
    headers: { Range: "bytes=2-4" },
    env,
  });

  assert.equal(response.status, 206);
  assert.equal(response.headers.get("content-range"), "bytes 2-4/10");
  assert.equal(response.headers.get("content-length"), "3");
  assert.equal(response.headers.get("accept-ranges"), "bytes");
  assert.equal(response.headers.get("etag"), ETAG);
  assert.equal(response.headers.get("cache-control"), "no-store");
  assert.equal(await response.text(), "234");
});

test("suffix ranges address the tail without emitting an empty Content-Range", async () => {
  installCache();
  installReleaseFetch();
  const env = makeEnv({ [NAMES["windows-setup"]]: PAYLOAD });

  let response = await run("https://openfocus.top/download/latest/windows-setup", {
    headers: { Range: "bytes=-3" },
    env,
  });
  assert.equal(response.status, 206);
  assert.equal(response.headers.get("content-range"), "bytes 7-9/10");
  assert.equal(await response.text(), "789");

  response = await run("https://openfocus.top/download/latest/windows-setup", {
    headers: { Range: "bytes=-50" },
    env,
  });
  assert.equal(response.status, 206);
  assert.equal(response.headers.get("content-range"), "bytes 0-9/10");
  assert.equal(await response.text(), "0123456789");
});

test("malformed, multi and out-of-bounds ranges return 416 with bytes */size", async () => {
  installCache();
  installReleaseFetch();
  const env = makeEnv({ [NAMES["windows-setup"]]: PAYLOAD });

  for (const range of ["bytes=10-", "bytes=5-4", "bytes=-0", "bytes=0-1,4-5"]) {
    const response = await run("https://openfocus.top/download/latest/windows-setup", {
      headers: { Range: range },
      env,
    });
    assert.equal(response.status, 416, range);
    assert.equal(response.headers.get("content-range"), "bytes */10", range);
    assert.equal(await response.text(), "", range);
  }
});

test("If-Range only authorises a resume for the same strong ETag", async () => {
  installCache();
  installReleaseFetch();
  const env = makeEnv({ [NAMES["windows-setup"]]: PAYLOAD });

  let response = await run("https://openfocus.top/download/latest/windows-setup", {
    headers: { Range: "bytes=2-4", "If-Range": ETAG },
    env,
  });
  assert.equal(response.status, 206);
  assert.equal(await response.text(), "234");

  response = await run("https://openfocus.top/download/latest/windows-setup", {
    headers: { Range: "bytes=2-4", "If-Range": '"another-release"' },
    env,
  });
  assert.equal(response.status, 200);
  assert.equal(response.headers.get("content-range"), null);
  assert.equal(response.headers.get("etag"), ETAG);
  assert.equal(response.headers.get("content-length"), "10");
  assert.equal(response.headers.get("cache-control"), "no-store");
  assert.equal(await response.text(), "0123456789");

  response = await run("https://openfocus.top/download/latest/windows-setup", {
    headers: { Range: "bytes=2-4", "If-Range": `W/${ETAG}` },
    env,
  });
  assert.equal(response.status, 200);
  assert.equal(await response.text(), "0123456789");
});

test("If-None-Match returns 304 for the current object", async () => {
  installCache();
  installReleaseFetch();
  const env = makeEnv({ [NAMES["windows-setup"]]: PAYLOAD });

  const response = await run("https://openfocus.top/download/latest/windows-setup", {
    headers: { "If-None-Match": ETAG },
    env,
  });
  assert.equal(response.status, 304);
  assert.equal(response.headers.get("etag"), ETAG);
  assert.equal(await response.text(), "");
});

test("HEAD mirrors GET status and headers without a body", async () => {
  installCache();
  installReleaseFetch();
  const env = makeEnv({ [NAMES["windows-setup"]]: PAYLOAD });

  let response = await run("https://openfocus.top/download/latest/windows-setup", {
    method: "HEAD",
    env,
  });
  assert.equal(response.status, 200);
  assert.equal(response.headers.get("content-length"), "10");
  assert.equal(await response.text(), "");

  response = await run("https://openfocus.top/download/latest/windows-setup", {
    method: "HEAD",
    headers: { Range: "bytes=0-3" },
    env,
  });
  assert.equal(response.status, 206);
  assert.equal(response.headers.get("content-range"), "bytes 0-3/10");
  assert.equal(response.headers.get("content-length"), "4");
  assert.equal(await response.text(), "");
});

test("an absent R2 object redirects a plain GET but refuses to guess at a range", async () => {
  installCache();
  installReleaseFetch();
  const env = makeEnv();

  let response = await run("https://openfocus.top/download/latest/windows-setup", { env });
  assert.equal(response.status, 302);
  assert.equal(
    response.headers.get("location"),
    `https://github.com/Xinzhe99/OpenFocus/releases/download/${TAG}/${NAMES["windows-setup"]}`,
  );
  assert.equal(response.headers.get("cache-control"), "no-store");

  response = await run("https://openfocus.top/download/latest/windows-setup", {
    headers: { Range: "bytes=0-3" },
    env,
  });
  assert.equal(response.status, 503);
  assert.equal(response.headers.get("location"), null);
  assert.match(await response.text(), /range/i);
});

test("upstream release failures stay 502 instead of becoming 404", async () => {
  installCache();
  installReleaseFetch({ fail: true });
  const env = makeEnv({ [NAMES["windows-setup"]]: PAYLOAD });

  let response = await run("https://openfocus.top/download/latest/windows-setup", { env });
  assert.equal(response.status, 502);
  assert.equal(response.headers.get("cache-control"), "no-store");
  assert.match(await response.text(), /GitHub release API returned 503/);

  response = await run("https://openfocus.top/api/latest", { env });
  assert.equal(response.status, 502);
  assert.match(await response.text(), /503/);
});

test("mirror metadata keeps downloads available when GitHub is rate limited", async () => {
  installCache();
  installReleaseFetch({ fail: true });
  const release = releaseBody();
  const manifest = {
    tag: release.tag_name,
    published: release.published_at,
    assets: Object.fromEntries(Object.entries(NAMES).map(([kind, name]) => {
      const asset = release.assets.find((item) => item.name === name);
      return [kind, { name, size: asset.size, url: asset.browser_download_url }];
    })),
  };
  const env = makeEnv({
    "openfocus-meta/latest-release.json": new TextEncoder().encode(JSON.stringify(manifest)),
    [NAMES["windows-setup"]]: PAYLOAD,
  });
  const latest = await run("https://openfocus.top/api/latest", { env });
  assert.equal(latest.status, 200);
  assert.equal(latest.headers.get("x-openfocus-release-source"), "mirror");
  assert.equal((await latest.json()).tag, TAG);
  const download = await run("https://openfocus.top/download/latest/windows-setup", {
    env, headers: { Range: "bytes=2-4" },
  });
  assert.equal(download.status, 206);
  assert.equal(download.headers.get("content-range"), "bytes 2-4/10");
  assert.equal(await download.text(), "234");
});

test("split Windows release is served from its complete, verified mirror", async () => {
  installCache();
  const release = releaseBody();
  release.assets = release.assets.filter((asset) => asset.name !== NAMES["windows-zip"]);
  installReleaseFetch({ release });
  const manifest = {
    tag: TAG,
    assets: Object.fromEntries(Object.entries(NAMES).map(([kind, name]) => [kind, {
      name, size: PAYLOAD.length, digest: `sha256:${"a".repeat(64)}`,
      ...(kind === "windows-zip" ? { source: "r2", url: null } : {
        url: `https://github.com/Xinzhe99/OpenFocus/releases/download/${TAG}/${name}`,
      }),
    }])),
  };
  const metadata = new TextEncoder().encode(JSON.stringify(manifest));
  const env = makeEnv({
    "openfocus-meta/latest-release.json": metadata,
    [NAMES["windows-zip"]]: PAYLOAD,
  });
  const latest = await run("https://openfocus.top/api/latest", { env });
  assert.equal((await latest.json()).assets["windows-zip"].source, "r2");
  const download = await run("https://openfocus.top/download/latest/windows-zip", {
    env, headers: { Range: "bytes=2-4" },
  });
  assert.equal(download.status, 206);
  assert.equal(await download.text(), "234");
  const updates = await run("https://openfocus.top/api/updates", { env });
  assert.equal((await updates.json()).assets["windows-zip"].digest,
               `sha256:${"a".repeat(64)}`);
});

test("a missing split mirror returns 503 instead of redirecting to a nonexistent file", async () => {
  installCache();
  const release = releaseBody();
  release.assets = release.assets.filter((asset) => asset.name !== NAMES["windows-zip"]);
  installReleaseFetch({ release });
  const manifest = {
    tag: TAG,
    assets: Object.fromEntries(Object.entries(NAMES).map(([kind, name]) => [kind, {
      name, size: PAYLOAD.length, digest: `sha256:${"b".repeat(64)}`,
      ...(kind === "windows-zip" ? { source: "r2", url: null } : {
        url: `https://github.com/Xinzhe99/OpenFocus/releases/download/${TAG}/${name}`,
      }),
    }])),
  };
  const env = makeEnv({
    "openfocus-meta/latest-release.json": new TextEncoder().encode(JSON.stringify(manifest)),
  });
  for (const headers of [{}, { Range: "bytes=0-2" }]) {
    const response = await run("https://openfocus.top/download/latest/windows-zip", { env, headers });
    assert.equal(response.status, 503);
    assert.equal(response.headers.get("location"), null);
  }
});

test("an older mirror never supplies assets for a newer GitHub release", async () => {
  installCache();
  const release = releaseBody();
  release.tag_name = "v1.0.1";
  release.assets = release.assets.filter((asset) => asset.name !== NAMES["windows-zip"]);
  installReleaseFetch({ release });
  const manifest = {
    tag: TAG,
    assets: Object.fromEntries(Object.entries(NAMES).map(([kind, name]) => [kind, {
      name, size: PAYLOAD.length, digest: `sha256:${"c".repeat(64)}`,
      ...(kind === "windows-zip" ? { source: "r2", url: null } : {
        url: `https://github.com/Xinzhe99/OpenFocus/releases/download/${TAG}/${name}`,
      }),
    }])),
  };
  const env = makeEnv({
    "openfocus-meta/latest-release.json": new TextEncoder().encode(JSON.stringify(manifest)),
  });
  const latest = await run("https://openfocus.top/api/latest", { env });
  const data = await latest.json();
  assert.equal(data.tag, "v1.0.1");
  assert.equal(data.assets["windows-zip"], null);
  const download = await run("https://openfocus.top/download/latest/windows-zip", { env });
  assert.equal(download.status, 503);
  assert.equal(download.headers.get("Retry-After"), "60");
});


test("invalid mirror metadata cannot redirect downloads", async () => {
  installCache();
  installReleaseFetch({ fail: true });
  const env = makeEnv({
    "openfocus-meta/latest-release.json": new TextEncoder().encode(JSON.stringify({tag: TAG, assets: {}})),
  });
  const response = await run("https://openfocus.top/api/latest", { env });
  assert.equal(response.status, 502);
  assert.match(await response.text(), /GitHub release API returned 503/);
});

test("split mirror metadata without a SHA-256 is rejected", async () => {
  installCache();
  installReleaseFetch({ fail: true });
  const manifest = {
    tag: TAG,
    assets: Object.fromEntries(Object.entries(NAMES).map(([kind, name]) => [kind, {
      name, size: PAYLOAD.length,
      ...(kind === "windows-zip" ? { source: "r2", url: null } : {
        url: `https://github.com/Xinzhe99/OpenFocus/releases/download/${TAG}/${name}`,
      }),
    }])),
  };
  const env = makeEnv({
    "openfocus-meta/latest-release.json": new TextEncoder().encode(JSON.stringify(manifest)),
  });
  const response = await run("https://openfocus.top/api/updates", { env });
  assert.equal(response.status, 503);
});

test("updates serves only the R2 manifest, including optional release notes and digests", async () => {
  const calls = installReleaseFetch({ fail: true });
  const release = releaseBody();
  const manifest = {
    tag: TAG,
    published: release.published_at,
    assets: Object.fromEntries(Object.entries(NAMES).map(([kind, name]) => [kind, {
      name, size: PAYLOAD.length,
      url: `https://github.com/Xinzhe99/OpenFocus/releases/download/${TAG}/${name}`,
      digest: `sha256:${"a".repeat(64)}`,
    }])),
    releases: [{ tag_name: TAG, body: "Release notes" }],
  };
  const key = "openfocus-meta/latest-release.json";
  const env = makeEnv({ [key]: new TextEncoder().encode(JSON.stringify(manifest)) });
  let response = await run("https://openfocus.top/api/updates", { env });
  assert.equal(response.status, 200);
  assert.deepEqual(await response.json(), manifest);
  response = await run("https://openfocus.top/api/updates", { env, method: "HEAD" });
  assert.equal(response.status, 200);
  assert.equal(await response.text(), "");
  assert.equal(calls(), 0);

  manifest.releases = Array.from({ length: 30 }, (_, index) => ({
    tag_name: `v1.0.${index}`,
    body: "Release notes ".repeat(320),
  }));
  const largeManifest = new TextEncoder().encode(JSON.stringify(manifest));
  assert.ok(largeManifest.length > 100000);
  response = await run("https://openfocus.top/api/updates", { env: makeEnv({ [key]: largeManifest }) });
  assert.equal(response.status, 200);
  assert.deepEqual(await response.json(), manifest);

  delete manifest.releases;
  for (const asset of Object.values(manifest.assets)) delete asset.digest;
  const legacy = makeEnv({ [key]: new TextEncoder().encode(JSON.stringify(manifest)) });
  response = await run("https://openfocus.top/api/updates", { env: legacy });
  assert.deepEqual(await response.json(), manifest);
  for (const unavailable of [makeEnv(), makeEnv({ [key]: new TextEncoder().encode("{}") }), { DOWNLOADS: { get: async () => { throw new Error("R2 unavailable"); } } }]) {
    response = await run("https://openfocus.top/api/updates", { env: unavailable });
    assert.equal(response.status, 503);
    assert.equal(response.headers.get("location"), null);
  }
  response = await run("https://openfocus.top/api/updates", { env, method: "POST" });
  assert.equal(response.status, 405);
  assert.equal(response.headers.get("allow"), "GET, HEAD");
  assert.equal(calls(), 0);
});

test("versioned downloads use only R2 with range, validators and HEAD", async () => {
  const calls = installReleaseFetch({ fail: true });
  const env = makeEnv(Object.fromEntries(Object.values(NAMES).map((name) => [name, PAYLOAD])));
  for (const kind of Object.keys(NAMES)) {
    const url = `https://openfocus.top/download/releases/${TAG}/${kind}`;
    let response = await run(url, { env, headers: { Range: "bytes=2-4" } });
    assert.equal(response.status, 206);
    assert.equal(response.headers.get("content-range"), "bytes 2-4/10");
    assert.equal(await response.text(), "234");
    response = await run(url, { env, method: "HEAD" });
    assert.equal(response.status, 200);
    assert.equal(response.headers.get("etag"), ETAG);
    assert.equal(await response.text(), "");
    response = await run(url, { env, headers: { "If-None-Match": ETAG } });
    assert.equal(response.status, 304);
  }
  const twoPartEnv = makeEnv({ "OpenFocus-v1.10-setup.exe": PAYLOAD });
  const twoPart = await run("https://openfocus.top/download/releases/v1.10/windows-setup", { env: twoPartEnv });
  assert.equal(twoPart.status, 200);
  assert.equal(await twoPart.text(), "0123456789");
  const missing = `https://openfocus.top/download/releases/v9.9.9/windows-setup`;
  for (const method of ["GET", "HEAD"]) {
    const response = await run(missing, { env, method });
    assert.equal(response.status, 503);
    assert.equal(response.headers.get("location"), null);
  }
  for (const path of ["v1.0.0/linux-deb", "v1.0.0-rc1/windows-setup", "v1..0/windows-setup", "v1.0.0/windows-setup/extra"]) {
    const response = await run(`https://openfocus.top/download/releases/${path}`, { env });
    assert.equal(response.status, 404);
  }
  const response = await run(`https://openfocus.top/download/releases/${TAG}/windows-setup`, { env, method: "POST" });
  assert.equal(response.status, 405);
  assert.equal(response.headers.get("allow"), "GET, HEAD");
  assert.equal(calls(), 0);
});

test("unknown kinds are 404 and non-read methods are 405", async () => {
  installCache();
  installReleaseFetch();
  const env = makeEnv({ [NAMES["windows-setup"]]: PAYLOAD });

  let response = await run("https://openfocus.top/download/latest/linux-deb", { env });
  assert.equal(response.status, 404);

  response = await run("https://openfocus.top/download/latest/windows-setup", {
    method: "POST",
    env,
  });
  assert.equal(response.status, 405);
  assert.equal(response.headers.get("allow"), "GET, HEAD");
});

test("latest metadata is no-store for clients while the edge caches it", async () => {
  installCache();
  const calls = installReleaseFetch();
  const env = makeEnv();

  let response = await run("https://openfocus.top/api/latest", { env });
  assert.equal(response.status, 200);
  assert.equal(response.headers.get("cache-control"), "no-store");
  const data = await response.json();
  assert.equal(data.tag, TAG);
  assert.equal(data.assets["windows-setup"].name, NAMES["windows-setup"]);
  assert.equal(calls(), 1);

  response = await run("https://openfocus.top/api/latest", { env });
  assert.equal(response.status, 200);
  assert.equal((await response.json()).tag, TAG);
  assert.equal(calls(), 1, "the second request should come from the edge cache");
});

test("www canonicalises static paths before they reach the asset binding", async () => {
  installCache();
  installReleaseFetch();
  const env = makeEnv();

  const response = await run("https://www.openfocus.top/zh/style.css?theme=dark", { env });
  assert.equal(response.status, 301);
  assert.equal(
    response.headers.get("location"),
    "https://openfocus.top/zh/style.css?theme=dark",
  );
  assert.deepEqual(env.ASSETS.calls, []);
});

test("Chinese preference redirects / to /zh/ and other static paths pass through", async () => {
  installCache();
  installReleaseFetch();
  const env = makeEnv();

  let response = await run("https://openfocus.top/", {
    headers: { "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8" },
    env,
  });
  assert.equal(response.status, 302);
  assert.equal(response.headers.get("location"), "/zh/");

  response = await run("https://openfocus.top/zh/", { env });
  assert.equal(response.status, 200);
  assert.equal(await response.text(), "asset:/zh/");
});

test("wrangler sends every route through the worker so www can canonicalise", async () => {
  const config = JSON.parse(
    await readFile(new URL("site/wrangler.jsonc", root), "utf8"),
  );
  assert.equal(config.assets.run_worker_first, true);
});

function analyticsDb() {
  const writes = [];
  const batches = [];
  return {
    writes, batches,
    prepare(sql) {
      return {
        sql, values: [],
        bind(...values) { this.values = values; return this; },
        async run() { writes.push({ sql: this.sql, values: this.values }); return { success: true }; },
      };
    },
    async batch(statements) {
      batches.push(statements.map(({ sql, values }) => ({ sql, values })));
      return statements.map(() => ({ success: true, results: [] }));
    },
  };
}

test("download analytics counts only successful starts with resolved version, route and source", async () => {
  installCache();
  installReleaseFetch();
  const db = analyticsDb();
  const env = { ...makeEnv({ [NAMES["windows-zip"]]: PAYLOAD }), ANALYTICS: db };
  const latest = "https://openfocus.top/download/latest/windows-zip";
  const pinned = `https://openfocus.top/download/releases/${TAG}/windows-zip`;
  assert.equal((await run(latest, { env })).status, 200);
  assert.equal((await run(latest, { env, headers: { Range: "bytes=0-2" } })).status, 206);
  assert.equal((await run(latest, { env, headers: { Range: "bytes=2-" } })).status, 206);
  assert.equal((await run(latest, { env, headers: { Range: "bytes=20-" } })).status, 416);
  assert.equal((await run(latest, { env, headers: { "If-None-Match": ETAG } })).status, 304);
  assert.equal((await run(latest, { env, method: "HEAD" })).status, 200);
  assert.equal((await run(pinned, { env })).status, 200);
  assert.equal((await run("https://openfocus.top/download/latest/macos-dmg", { env })).status, 302);
  assert.equal((await run("https://openfocus.top/download/latest/macos-dmg", { env, method: "HEAD" })).status, 302);
  assert.deepEqual(db.writes.map((write) => write.values), [
    [TAG, "windows-zip", "website", "r2"],
    [TAG, "windows-zip", "website", "r2"],
    [TAG, "windows-zip", "in-app", "r2"],
    [TAG, "macos-dmg", "website", "github"],
  ]);
  assert.match(db.writes[0].sql, /ON CONFLICT.*count = count \+ 1/);
  env.ANALYTICS = { prepare() { throw new Error("database down"); } };
  assert.equal((await run(latest, { env })).status, 200);
  delete env.ANALYTICS;
  assert.equal((await run(latest, { env })).status, 200);
});

test("usage accepts only bounded reports and stores no request metadata", async () => {
  const db = analyticsDb();
  const env = { ANALYTICS: db };
  const url = "https://openfocus.top/api/usage";
  const payload = { installation_id: "8d5d5ea6-9e3c-4a5e-9e7d-444444444444", version: "1.2.3", platform: "windows" };
  const post = (value, headers = {}) => run(url, { method: "POST", env, headers: { "Content-Type": "application/json", ...headers }, body: JSON.stringify(value) });
  assert.equal((await run(url, { env })).status, 405);
  assert.equal((await post(payload, { "User-Agent": "private-agent", "CF-Connecting-IP": "192.0.2.1" })).status, 204);
  assert.equal((await post(payload)).status, 204);
  assert.deepEqual(db.batches[0][1].values, Object.values(payload));
  assert.match(db.batches[0][1].sql, /ON CONFLICT.*DO NOTHING/);
  assert.match(db.batches[0][0].sql, /-89 days/);
  assert.equal((await post({ ...payload, filename: "private" })).status, 400);
  assert.equal((await post({ ...payload, installation_id: "8d5d5ea6-9e3c-1a5e-9e7d-444444444444" })).status, 400);
  assert.equal((await post({ ...payload, version: "1.2.3/extra" })).status, 400);
  assert.equal((await post({ ...payload, platform: "other" })).status, 400);
  assert.equal((await post({ ...payload, version: "1".repeat(1100) })).status, 413);
  assert.equal((await run(url, { method: "POST", env, body: "{}" })).status, 415);
  assert.equal((await run(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" })).status, 503);
  assert.equal((await post(payload, { "Content-Length": "9999" })).status, 413);
  const failing = { ANALYTICS: { prepare: db.prepare, async batch() { throw new Error("offline"); } } };
  assert.equal((await run(url, { method: "POST", env: failing, headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) })).status, 503);
});

test("private statistics require a configured bearer secret and are read-only", async () => {
  const db = analyticsDb();
  const url = "https://openfocus.top/api/admin/stats";
  const env = { ANALYTICS: db, ANALYTICS_ADMIN_SECRET: "secret-for-test" };
  assert.equal((await run(url, { env: { ANALYTICS: db } })).status, 503);
  assert.equal((await run(url, { env })).status, 401);
  assert.equal((await run(url, { env, headers: { Authorization: "Bearer incorrect" } })).status, 401);
  assert.equal((await run(url, { env, method: "POST" })).status, 405);
  const response = await run(url, { env, headers: { Authorization: "Bearer secret-for-test" } });
  assert.equal(response.status, 200);
  assert.equal(response.headers.get("Cache-Control"), "no-store");
  assert.deepEqual(await response.json(), { downloads: [], active_30_days: [] });
  assert.ok(db.batches[0].every(({ sql }) => /^SELECT /.test(sql)));
  assert.match(db.batches[0][1].sql, /COUNT\(DISTINCT installation_id\)/);
});

test("scheduled cleanup removes expired pseudonymous activity", async () => {
  const db = analyticsDb();
  const pending = [];
  await worker.scheduled({}, { ANALYTICS: db }, { waitUntil(promise) { pending.push(promise); } });
  await Promise.all(pending);
  assert.equal(db.writes.length, 1);
  assert.match(db.writes[0].sql, /^DELETE FROM active_installations WHERE day < date\('now', '-89 days'\)$/);
});
