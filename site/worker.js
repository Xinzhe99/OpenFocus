/**
 * OpenFocus site worker.
 *
 * - "/" redirects to /zh/ when the browser prefers Chinese (Accept-Language +
 *   a `lang` cookie set by the language menu).
 * - "/api/latest" returns the latest GitHub release, cached at the edge.
 * - "/download/latest/<kind>" streams the matching R2 object with byte-range
 *   support, or redirects to the GitHub asset when the mirror has no object.
 * - "/api/updates" and "/download/releases/<tag>/<kind>" use R2 only.
 *
 * With `run_worker_first: true` in wrangler.jsonc every request reaches this
 * worker, so `www` canonicalisation also applies to static paths. Everything
 * that is not handled below is passed to the static asset binding unchanged.
 */

const REPO = "Xinzhe99/OpenFocus";
const RELEASE_API = `https://api.github.com/repos/${REPO}/releases/latest`;
const CACHE_URL = "https://openfocus.internal/latest-release.json";
const MIRROR_MANIFEST = "openfocus-meta/latest-release.json";
const DOWNLOAD_PREFIX = "/download/latest/";
const VERSIONED_DOWNLOAD_PREFIX = "/download/releases/";
const DOWNLOAD_SUFFIXES = {
  "windows-setup": "setup.exe",
  "windows-zip": "windows-x64.zip",
  "macos-dmg": "macos.dmg",
  "macos-zip": "macos-arm64.zip",
};
const DOWNLOAD_KINDS = new Set([
  "windows-setup",
  "windows-zip",
  "macos-dmg",
  "macos-zip",
]);
const EDGE_CACHE_SECONDS = 300;
const NO_STORE = "no-store";
// /download/latest/* is a mutable URL even though the R2 object it resolves to
// is versioned, so no downstream cache may pin the response.
const OBJECT_CACHE = NO_STORE;

function jsonHeaders() {
  return {
    "Content-Type": "application/json; charset=utf-8",
    "Cache-Control": NO_STORE,
    "X-Content-Type-Options": "nosniff",
  };
}

function textResponse(message, status, headers = {}, method = "GET") {
  const merged = new Headers(headers);
  if (!merged.has("Content-Type")) {
    merged.set("Content-Type", "text/plain; charset=utf-8");
  }
  if (!merged.has("Cache-Control")) merged.set("Cache-Control", NO_STORE);
  merged.set("X-Content-Type-Options", "nosniff");
  merged.set("Content-Length", String(new TextEncoder().encode(message).byteLength));
  return new Response(method === "HEAD" ? null : message, {
    status,
    headers: merged,
  });
}

function badGateway(message, method = "GET") {
  const body = JSON.stringify({ error: message });
  return new Response(method === "HEAD" ? null : body, {
    status: 502,
    headers: {
      ...jsonHeaders(),
      "Content-Length": String(new TextEncoder().encode(body).byteLength),
      "Retry-After": "60",
    },
  });
}

function preferredLanguage(request) {
  const cookie = request.headers.get("Cookie") || "";
  const chosen = cookie.match(/(?:^|;\s*)lang=(en|zh)(?:;|$)/);
  if (chosen) return chosen[1];

  const header = request.headers.get("Accept-Language") || "";
  const tags = header
    .split(",")
    .map((part) => {
      const [tag, ...params] = part.trim().split(";");
      let q = 1;
      for (const param of params) {
        const match = param.trim().match(/^q=([0-9.]+)$/);
        if (match) q = Number(match[1]);
      }
      return { tag: tag.toLowerCase(), q };
    })
    .filter((entry) => entry.tag && entry.q > 0)
    .sort((a, b) => b.q - a.q);

  return tags.length && tags[0].tag.startsWith("zh") ? "zh" : "en";
}

function githubAssetUrl(value) {
  if (typeof value !== "string") return null;
  try {
    const parsed = new URL(value);
    if (parsed.protocol !== "https:" || parsed.hostname !== "github.com") {
      return null;
    }
    if (!parsed.pathname.startsWith(`/${REPO}/releases/download/`)) return null;
    return parsed.toString();
  } catch {
    return null;
  }
}

function safeObjectKey(value) {
  if (typeof value !== "string" || value.length > 512) return null;
  return /^[A-Za-z0-9][A-Za-z0-9._-]*$/.test(value) ? value : null;
}

function releasePayload(release) {
  const assets = Array.isArray(release.assets) ? release.assets : [];
  const pick = (pattern) => {
    const asset = assets.find(
      (item) =>
        item &&
        typeof item.name === "string" &&
        pattern.test(item.name),
    );
    if (!asset) return null;
    return {
      name: asset.name,
      size: Number(asset.size) || 0,
      url: githubAssetUrl(asset.browser_download_url),
    };
  };

  return {
    tag: release.tag_name,
    published: release.published_at || null,
    assets: {
      "windows-setup": pick(/-setup\.exe$/i),
      "windows-zip": pick(/-windows-x64\.zip$/i),
      "macos-dmg": pick(/-macos\.dmg$/i),
      "macos-zip": pick(/-macos-arm64\.zip$/i),
    },
  };
}
// The mirror workflow publishes this manifest only after uploading all four
// assets. It keeps downloads available when GitHub's shared-IP rate limit is
// exhausted; the response identifies the mirror as its metadata source.
async function mirroredRelease(env) {
  if (!env || !env.DOWNLOADS || typeof env.DOWNLOADS.get !== "function") return null;
  try {
    const object = await env.DOWNLOADS.get(MIRROR_MANIFEST);
    if (!object || !object.body || object.size > 5000000) return null;
    const data = await new Response(object.body).json();
    if (!data || typeof data.tag !== "string" || !/^v[0-9A-Za-z._+-]+$/.test(data.tag)) return null;
    const suffixes = {
      "windows-setup": "setup.exe",
      "windows-zip": "windows-x64.zip",
      "macos-dmg": "macos.dmg",
      "macos-zip": "macos-arm64.zip",
    };
    for (const [kind, suffix] of Object.entries(suffixes)) {
      const asset = data.assets && data.assets[kind];
      const expected = `OpenFocus-${data.tag}-${suffix}`;
      const remote = asset && githubAssetUrl(asset.url);
      const mirrorOnly = asset && asset.source === "r2" && asset.url === null &&
        kind.startsWith("windows-") &&
        /^sha256:[0-9a-fA-F]{64}$/.test(asset.digest || "");
      if (!asset || asset.name !== expected || !safeObjectKey(asset.name) ||
          !Number.isSafeInteger(asset.size) || asset.size <= 0) return null;
      if (!mirrorOnly && (!remote || decodeURIComponent(new URL(remote).pathname) !==
          `/${REPO}/releases/download/${data.tag}/${expected}`)) return null;
    }
    return data;
  } catch {
    return null;
  }
}

async function latestFailure(message, env) {
  const mirror = await mirroredRelease(env);
  if (!mirror) return badGateway(message);
  return new Response(JSON.stringify(mirror), {
    headers: { ...jsonHeaders(), "X-OpenFocus-Release-Source": "mirror" },
  });
}


async function latestRelease(ctx, env) {
  const cache = typeof caches !== "undefined" ? caches.default : null;
  const key = new Request(CACHE_URL);

  if (cache) {
    try {
      const hit = await cache.match(key);
      if (hit) {
        const body = await hit.text();
        return new Response(body, { status: 200, headers: jsonHeaders() });
      }
    } catch {
      // A cache miss or an unavailable cache must not fail the request.
    }
  }

  let upstream;
  try {
    upstream = await fetch(RELEASE_API, {
      headers: {
        "User-Agent": "openfocus-site",
        Accept: "application/vnd.github+json",
      },
    });
  } catch {
    return latestFailure("GitHub release API is unreachable", env);
  }
  if (!upstream.ok) {
    return latestFailure(`GitHub release API returned ${upstream.status}`, env);
  }

  let release;
  try {
    release = await upstream.json();
  } catch {
    return latestFailure("GitHub release API returned invalid JSON", env);
  }
  if (
    !release ||
    typeof release.tag_name !== "string" ||
    !release.tag_name ||
    !Array.isArray(release.assets)
  ) {
    return latestFailure("GitHub release API returned an incomplete release", env);
  }

  const payload = releasePayload(release);
  if (Object.values(payload.assets).some((asset) => !asset)) {
    const mirror = await mirroredRelease(env);
    if (mirror && mirror.tag === payload.tag) {
      for (const kind of DOWNLOAD_KINDS) {
        if (!payload.assets[kind]) payload.assets[kind] = mirror.assets[kind];
      }
    }
  }
  const body = JSON.stringify(payload);
  // Do not cache a release while its split Windows assets are still mirroring.
  if (cache && Object.values(payload.assets).every(Boolean)) {
    const cacheable = new Response(body, {
      headers: {
        "Content-Type": "application/json; charset=utf-8",
        "Cache-Control": `public, max-age=${EDGE_CACHE_SECONDS}`,
      },
    });
    try {
      const stored = cache.put(key, cacheable);
      if (ctx && typeof ctx.waitUntil === "function") {
        ctx.waitUntil(stored.catch(() => {}));
      } else {
        await stored.catch(() => {});
      }
    } catch {
      // Caching is an optimisation; the response is still correct without it.
    }
  }

  return new Response(body, { headers: jsonHeaders() });
}

/**
 * Parse one `bytes=` range against a known representation size.
 *
 * Returns null when the header is not a byte range (the caller then serves the
 * full representation), `{ unsatisfiable: true }` for a syntactically valid
 * byte range that cannot be satisfied, and `{ offset, length }` otherwise.
 * Multiple ranges are intentionally rejected: the download endpoint only
 * promises single ranges, and a 416 is safer than guessing multipart output.
 */
function parseRange(value, size) {
  const match = /^\s*bytes\s*=\s*(.*)$/i.exec(value);
  if (!match) return null;
  const spec = match[1].trim();
  if (!spec || spec.includes(",")) return { unsatisfiable: true };

  const parts = /^(\d*)\s*-\s*(\d*)$/.exec(spec);
  if (!parts) return { unsatisfiable: true };
  const first = parts[1];
  const last = parts[2];
  if (first === "" && last === "") return { unsatisfiable: true };
  if (!Number.isSafeInteger(size) || size <= 0) return { unsatisfiable: true };

  if (first === "") {
    const suffix = Number(last);
    if (!Number.isSafeInteger(suffix) || suffix <= 0) {
      return { unsatisfiable: true };
    }
    const length = Math.min(suffix, size);
    return { offset: size - length, length };
  }

  const offset = Number(first);
  if (!Number.isSafeInteger(offset) || offset >= size) {
    return { unsatisfiable: true };
  }
  let end = size - 1;
  if (last !== "") {
    const requestedEnd = Number(last);
    if (!Number.isSafeInteger(requestedEnd) || requestedEnd < offset) {
      return { unsatisfiable: true };
    }
    end = Math.min(requestedEnd, size - 1);
  }
  return { offset, length: end - offset + 1 };
}

function entityTag(object) {
  if (object && typeof object.httpEtag === "string" && object.httpEtag) {
    return object.httpEtag;
  }
  if (object && typeof object.etag === "string" && object.etag) {
    return object.etag.startsWith('"') ? object.etag : `"${object.etag}"`;
  }
  return null;
}

function withoutWeakPrefix(tag) {
  return tag.startsWith("W/") ? tag.slice(2) : tag;
}

function ifNoneMatchMatches(header, etag) {
  if (!header) return false;
  const candidates = header.split(",").map((part) => part.trim()).filter(Boolean);
  if (candidates.includes("*")) return true;
  if (!etag) return false;
  const current = withoutWeakPrefix(etag.trim());
  return candidates.some(
    (candidate) => withoutWeakPrefix(candidate) === current,
  );
}

/**
 * Strong comparison for If-Range. A weak tag or an unparseable date never
 * authorises a partial response: serving the full representation is the only
 * way to guarantee that a resume cannot append bytes from another object.
 */
function ifRangeMatches(header, etag, uploaded) {
  const value = header.trim();
  if (!value) return false;
  if (value.startsWith('"') || value.startsWith("W/")) {
    if (value.startsWith("W/")) return false;
    return etag !== null && value === etag;
  }

  const date = Date.parse(value);
  if (!Number.isFinite(date)) return false;
  const modified = uploaded instanceof Date ? uploaded.getTime() : Date.parse(uploaded);
  if (!Number.isFinite(modified)) return false;
  return modified <= date + 1000;
}

function objectHeaders(head, name) {
  const headers = new Headers();
  if (typeof head.writeHttpMetadata === "function") {
    head.writeHttpMetadata(headers);
  }
  headers.set("Content-Type", "application/octet-stream");
  headers.set("Content-Disposition", `attachment; filename="${name}"`);
  headers.set("Accept-Ranges", "bytes");
  headers.set("X-Content-Type-Options", "nosniff");
  headers.set("Cache-Control", OBJECT_CACHE);
  if (head.uploaded instanceof Date) {
    headers.set("Last-Modified", head.uploaded.toUTCString());
  }
  const etag = entityTag(head);
  if (etag) headers.set("ETag", etag);
  return headers;
}

/**
 * Build a response from R2, or return null when the object is absent or the
 * storage call failed. The caller decides how to handle an unavailable object.
 */
async function r2Download(request, env, key, name) {
  if (!env || !env.DOWNLOADS || typeof env.DOWNLOADS.head !== "function") {
    return null;
  }

  let head;
  try {
    head = await env.DOWNLOADS.head(key);
  } catch {
    return null;
  }
  if (!head) return null;

  const method = (request.method || "GET").toUpperCase();
  const size = Number(head.size) || 0;
  const etag = entityTag(head);
  const rangeHeader = request.headers.get("Range");

  if (ifNoneMatchMatches(request.headers.get("If-None-Match"), etag)) {
    const headers = new Headers({
      "Cache-Control": OBJECT_CACHE,
      "Accept-Ranges": "bytes",
      "X-Content-Type-Options": "nosniff",
    });
    if (etag) headers.set("ETag", etag);
    return new Response(null, { status: 304, headers });
  }

  let range = null;
  if (rangeHeader) {
    const ifRange = request.headers.get("If-Range");
    if (ifRange === null || ifRangeMatches(ifRange, etag, head.uploaded)) {
      range = parseRange(rangeHeader, size);
    }
  }

  const headers = objectHeaders(head, name);

  if (range && range.unsatisfiable) {
    headers.delete("Content-Disposition");
    headers.set("Content-Range", `bytes */${size}`);
    headers.set("Content-Length", "0");
    headers.set("Cache-Control", NO_STORE);
    return new Response(null, { status: 416, headers });
  }

  if (range) {
    const last = range.offset + range.length - 1;
    headers.set("Content-Range", `bytes ${range.offset}-${last}/${size}`);
    headers.set("Content-Length", String(range.length));
    headers.set("Cache-Control", NO_STORE);
  } else {
    headers.set("Content-Length", String(size));
  }

  const status = range ? 206 : 200;
  if (method === "HEAD" || size === 0) {
    return new Response(null, { status, headers });
  }

  let object;
  try {
    object = await env.DOWNLOADS.get(
      key,
      range ? { range: { offset: range.offset, length: range.length } } : undefined,
    );
  } catch {
    return null;
  }
  if (!object || !object.body) return null;

  // If the object changed between head() and get(), do not pair the old
  // validator with new bytes. The caller falls back rather than returning a
  // response that a resumed download could append incorrectly.
  const fetchedTag = entityTag(object);
  if (etag && fetchedTag && etag !== fetchedTag) return null;

  return new Response(object.body, { status, headers });
}

async function handleDownload(request, env, ctx) {
  const method = (request.method || "GET").toUpperCase();
  if (method !== "GET" && method !== "HEAD") {
    return textResponse("Method Not Allowed", 405, { Allow: "GET, HEAD" }, method);
  }

  const kind = new URL(request.url).pathname.slice(DOWNLOAD_PREFIX.length);
  if (!DOWNLOAD_KINDS.has(kind)) {
    return textResponse("Unknown download kind", 404, {}, method);
  }

  let latest;
  try {
    latest = await latestRelease(ctx, env);
  } catch {
    return badGateway("Latest release lookup failed", method);
  }
  if (!latest.ok) {
    if (method === "HEAD") {
      return new Response(null, { status: latest.status, headers: latest.headers });
    }
    return latest;
  }

  let data;
  try {
    data = await latest.json();
  } catch {
    return badGateway("Latest release metadata is invalid", method);
  }
  const asset = data && data.assets ? data.assets[kind] : null;
  if (!asset) {
    return textResponse(`The latest ${kind} download is not ready; retry the download.`,
      503, { "Retry-After": "60" }, method);
  }

  const name = safeObjectKey(asset.name);
  const remote = githubAssetUrl(asset.url);
  if (!name || (!remote && asset.source !== "r2")) {
    return badGateway("The latest release has an invalid asset URL", method);
  }

  const mirrored = await r2Download(request, env, name, name);
  if (mirrored) return mirrored;

  if (!remote || request.headers.get("Range")) {
    return textResponse(
      remote ? "The download mirror is not ready for this range; retry the download."
        : "The download mirror is not ready; retry or use the release page for recovery.",
      503,
      { "Retry-After": "60" },
      method,
    );
  }

  return new Response(null, {
    status: 302,
    headers: { Location: remote, "Cache-Control": NO_STORE },
  });
}

async function handleVersionedDownload(request, env) {
  const method = (request.method || "GET").toUpperCase();
  if (method !== "GET" && method !== "HEAD") {
    return textResponse("Method Not Allowed", 405, { Allow: "GET, HEAD" }, method);
  }
  const parts = new URL(request.url).pathname.slice(VERSIONED_DOWNLOAD_PREFIX.length).split("/");
  const [tag, kind] = parts;
  if (parts.length !== 2 || !/^v[0-9]+(?:\.[0-9]+)*$/.test(tag) || !DOWNLOAD_KINDS.has(kind)) {
    return textResponse("Unknown release download", 404, {}, method);
  }
  const name = `OpenFocus-${tag}-${DOWNLOAD_SUFFIXES[kind]}`;
  const response = await r2Download(request, env, name, name);
  return response || textResponse("The download mirror is not ready; retry the download.", 503, { "Retry-After": "60" }, method);
}

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    const method = (request.method || "GET").toUpperCase();

    if (url.hostname === "www.openfocus.top") {
      const target = new URL(request.url);
      target.protocol = "https:";
      target.hostname = "openfocus.top";
      target.port = "";
      return new Response(null, {
        status: 301,
        headers: {
          Location: target.toString(),
          "Cache-Control": "public, max-age=3600",
        },
      });
    }

    if (url.pathname === "/api/updates") {
      if (method !== "GET" && method !== "HEAD") {
        return textResponse("Method Not Allowed", 405, { Allow: "GET, HEAD" }, method);
      }
      const mirror = await mirroredRelease(env);
      if (!mirror) return textResponse("Update mirror is unavailable", 503, { "Retry-After": "60" }, method);
      return new Response(method === "HEAD" ? null : JSON.stringify(mirror), {
        headers: { ...jsonHeaders(), "X-OpenFocus-Release-Source": "mirror" },
      });
    }

    if (url.pathname.startsWith(VERSIONED_DOWNLOAD_PREFIX)) {
      return handleVersionedDownload(request, env);
    }

    if (url.pathname === "/api/latest") {
      if (method !== "GET" && method !== "HEAD") {
        return textResponse("Method Not Allowed", 405, { Allow: "GET, HEAD" }, method);
      }
      const response = await latestRelease(ctx, env);
      if (method === "HEAD") {
        return new Response(null, { status: response.status, headers: response.headers });
      }
      return response;
    }

    if (url.pathname.startsWith(DOWNLOAD_PREFIX)) {
      return handleDownload(request, env, ctx);
    }

    if (url.pathname === "/" && (method === "GET" || method === "HEAD")) {
      if (preferredLanguage(request) === "zh") {
        return new Response(null, {
          status: 302,
          headers: {
            Location: "/zh/",
            Vary: "Accept-Language, Cookie",
            "Cache-Control": NO_STORE,
          },
        });
      }
    }

    if (!env || !env.ASSETS || typeof env.ASSETS.fetch !== "function") {
      return textResponse("Not Found", 404, {}, method);
    }

    return env.ASSETS.fetch(request);
  },
};
