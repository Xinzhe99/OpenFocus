/**
 * OpenFocus site worker.
 *
 * - "/" negotiates English/Chinese from an explicit `lang` query, the
 *   language-menu cookie, or Accept-Language, preserving query parameters.
 * - "/api/latest" returns the latest GitHub release, cached at the edge.
 * - "/download/latest/<kind>" streams the matching R2 object with byte-range
 *   support, or redirects to the GitHub asset when the mirror has no object.
 * - "/api/updates" and versioned full and delta downloads use R2 only.
 * - The two WeChat QR paths proxy the current PNGs from the account service.
 *
 * With `run_worker_first: true` in wrangler.jsonc every request reaches this
 * worker, so `www` canonicalisation also applies to static paths. Everything
 * that is not handled below is passed to the static asset binding unchanged.
 */

const REPO = "Xinzhe99/OpenFocus";
const RELEASE_API = `https://api.github.com/repos/${REPO}/releases/latest`;
const CACHE_URL = "https://openfocus.internal/latest-release-v2.json";
const MIRROR_MANIFEST = "openfocus-meta/latest-release.json";
const PREVIEW_MANIFEST = "openfocus-meta/beta-release.json";
const DOWNLOAD_PREFIX = "/download/latest/";
const VERSIONED_DOWNLOAD_PREFIX = "/download/releases/";
const DOWNLOAD_SUFFIXES = {
  "windows-setup": "setup.exe",
  "windows-zip": "windows-x64.zip",
  "macos-dmg": "macos.dmg",
  "macos-zip": "macos-arm64.zip",
  "linux-desktop": "linux-x86_64.tar.gz",
};
const DELTA_FULL_KINDS = new Map([
  ["windows-x64", "windows-zip"],
  ["macos-arm64", "macos-zip"],
  ["linux-x86_64", "linux-desktop"],
]);
const DOWNLOAD_KINDS = new Set([
  "windows-setup",
  "windows-zip",
  "macos-dmg",
  "macos-zip",
  "linux-desktop",
]);
const EDGE_CACHE_SECONDS = 300;
const NO_STORE = "no-store";
const QR_ORIGIN = "https://account.openfocus.top";
const QR_ROUTES = new Map([
  ["/wechat-group.png", "wechat-group"],
  ["/wechat-official-account.jpg", "wechat-official-account"],
]);
const QR_MAX_BYTES = 512 * 1024;
const QR_MAX_DIMENSION = 1024;
const PNG_SIGNATURE = [137, 80, 78, 71, 13, 10, 26, 10];
const LOCALIZED_ROUTES = new Map([
  ["/", { en: "/", zh: "/zh/", lang: null, home: true }],
  ["/zh/", { en: "/", zh: "/zh/", lang: "zh", home: true }],
  ["/contributors", { en: "/contributors/", zh: "/zh/contributors/", lang: null, canonicalize: true, home: false }],
  ["/contributors/", { en: "/contributors/", zh: "/zh/contributors/", lang: null, canonicalize: false, home: false }],
  ["/zh/contributors", { en: "/contributors/", zh: "/zh/contributors/", lang: "zh", canonicalize: true, home: false }],
  ["/zh/contributors/", { en: "/contributors/", zh: "/zh/contributors/", lang: "zh", canonicalize: false, home: false }],
]);
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

async function readQrPng(response) {
  const declaredLength = response.headers.get("Content-Length");
  const limit = declaredLength === null ? QR_MAX_BYTES : Number(declaredLength);
  if (!response.body || limit < 45 || limit > QR_MAX_BYTES ||
      (declaredLength !== null && !/^\d+$/.test(declaredLength)) ||
      response.headers.get("Content-Type")?.toLowerCase() !== "image/png") {
    response.body?.cancel().catch(() => {});
    return null;
  }

  const buffer = new Uint8Array(limit);
  let size = 0;
  const reader = response.body.getReader();
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    if (size + value.byteLength > limit) {
      reader.cancel().catch(() => {});
      return null;
    }
    buffer.set(value, size);
    size += value.byteLength;
  }
  if (size < 45 || (declaredLength !== null && size !== limit) ||
      !PNG_SIGNATURE.every((byte, offset) => buffer[offset] === byte)) return null;

  const header = new DataView(buffer.buffer);
  if (header.getUint32(8) !== 13 || header.getUint32(12) !== 0x49484452) return null; // IHDR
  if (header.getUint32(size - 12) !== 0 || header.getUint32(size - 8) !== 0x49454e44 ||
      header.getUint32(size - 4) !== 0xae426082) return null; // IEND
  const width = header.getUint32(16);
  const height = header.getUint32(20);
  return width > 0 && height > 0 && width <= QR_MAX_DIMENSION && height <= QR_MAX_DIMENSION
    ? buffer.subarray(0, size) : null;
}

async function proxyQr(id, method) {
  if (method !== "GET" && method !== "HEAD") {
    return textResponse("Method Not Allowed", 405, { Allow: "GET, HEAD" }, method);
  }

  try {
    // The account API exposes GET. Validate the same representation for HEAD.
    const upstream = await fetch(`${QR_ORIGIN}/v1/qr/${id}`, {
      method: "GET",
      headers: { Accept: "image/png", "User-Agent": "OpenFocus/Website" },
      credentials: "omit",
      redirect: "manual",
      cache: "no-store",
    });
    if (upstream.status !== 200) {
      upstream.body?.cancel().catch(() => {});
      if (upstream.status === 404) {
        return textResponse("QR code is unpublished", 404, {}, method);
      }
    } else {
      const png = await readQrPng(upstream);
      if (png) {
        return new Response(method === "HEAD" ? null : png, {
          headers: {
            "Content-Type": "image/png",
            "Content-Length": String(png.byteLength),
            "Cache-Control": NO_STORE,
            "X-Content-Type-Options": "nosniff",
          },
        });
      }
    }
  } catch {
    // Network and body read failures use the same public upstream error.
  }
  return textResponse("QR code service is unavailable", 502, {}, method);
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

  const supported = tags.find((entry) => /^(en|zh)(?:-|$)/.test(entry.tag));
  return supported?.tag === "zh" || supported?.tag.startsWith("zh-") ? "zh" : "en";
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

function linuxRequired(tag) {
  if (!/^v[0-9]+(?:\.[0-9]+)*$/.test(tag)) return false;
  const [major, minor = 0] = tag.slice(1).split(".").map(Number);
  return major > 1 || (major === 1 && minor >= 77);
}

function validLinuxAsset(asset, tag) {
  const name = `OpenFocus-${tag}-${DOWNLOAD_SUFFIXES["linux-desktop"]}`;
  const remote = asset && githubAssetUrl(asset.url);
  return !!asset && asset.name === name && safeObjectKey(name) &&
    Number.isSafeInteger(asset.size) && asset.size > 0 && !!remote &&
    remote === `https://github.com/${REPO}/releases/download/${tag}/${name}` &&
    (!asset.digest || /^sha256:[0-9a-fA-F]{64}$/.test(asset.digest));
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
      ...(typeof asset.digest === "string" && /^sha256:[0-9a-fA-F]{64}$/.test(asset.digest)
        ? { digest: asset.digest } : {}),
    };
  };
  const linuxName = `OpenFocus-${release.tag_name}-${DOWNLOAD_SUFFIXES["linux-desktop"]}`;
  const linux = assets.find((item) => item && item.name === linuxName);
  const linuxAsset = linux ? {
    name: linux.name, size: linux.size, url: githubAssetUrl(linux.browser_download_url),
    ...(linux.digest ? { digest: linux.digest } : {}),
  } : null;

  return {
    tag: release.tag_name,
    published: release.published_at || null,
    assets: {
      "windows-setup": pick(/-setup\.exe$/i),
      "windows-zip": pick(/-windows-x64\.zip$/i),
      "macos-dmg": pick(/-macos\.dmg$/i),
      "macos-zip": pick(/-macos-arm64\.zip$/i),
      "linux-desktop": validLinuxAsset(linuxAsset, release.tag_name) ? linuxAsset : null,
    },
  };
}
// The mirror workflow publishes metadata only after uploading all required
// assets. It keeps downloads available when GitHub's shared-IP rate limit is
// exhausted; the response identifies the mirror as its metadata source.
async function mirroredRelease(env, key = MIRROR_MANIFEST) {
  if (!env || !env.DOWNLOADS || typeof env.DOWNLOADS.get !== "function") return null;
  try {
    const object = await env.DOWNLOADS.get(key);
    if (!object || !object.body || object.size > 5000000) return null;
    const data = await new Response(object.body).json();
    if (!data || typeof data.tag !== "string" || !/^v[0-9A-Za-z._+-]+$/.test(data.tag)) return null;
    if (key === MIRROR_MANIFEST && (!/^v[0-9]+(?:\.[0-9]+)*$/.test(data.tag) ||
        data.tag.split(".").some((part) => /^0*999999999$/.test(part)) || data.actual_tag)) return null;
    if (key === PREVIEW_MANIFEST && !/^v[0-9]+(?:\.[0-9]+)*-beta\.[1-9][0-9]*$/.test(data.tag)) return null;
    if (key === PREVIEW_MANIFEST && data.tag.split("-beta.")[0].split(".").includes("999999999")) return null;
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
    const linux = data.assets && data.assets["linux-desktop"];
    if (linux && (!validLinuxAsset(linux, data.tag) ||
        !/^sha256:[0-9a-fA-F]{64}$/.test(linux.digest || ""))) {
      delete data.assets["linux-desktop"];
    }
    if ("deltas" in data) data.deltas = validDeltas(data);
    if (key === MIRROR_MANIFEST && Array.isArray(data.releases)) {
      data.releases = data.releases.filter((release) => release &&
        /^v[0-9]+(?:\.[0-9]+)*$/.test(release.tag_name || "") &&
        !release.tag_name.split(".").some((part) => /^0*999999999$/.test(part)));
    }
    return data;
  } catch {
    return null;
  }
}

async function latestFailure(message, env) {
  const mirror = await mirroredRelease(env);
  if (!mirror) return badGateway(message);
  delete mirror.deltas;
  return new Response(JSON.stringify(mirror), {
    headers: { ...jsonHeaders(), "X-OpenFocus-Release-Source": "mirror" },
  });
}


async function stableRelease(ctx, env) {
  const cache = typeof caches !== "undefined" ? caches.default : null;
  const key = new Request(CACHE_URL);

  if (cache) {
    try {
      const hit = await cache.match(key);
      if (hit) {
        const data = await hit.json();
        if (/^v[0-9]+(?:\.[0-9]+)*$/.test(data.tag || "") &&
            !data.tag.split(".").some((part) => /^0*999999999$/.test(part)) && !data.actual_tag) {
          return new Response(JSON.stringify(data), { status: 200, headers: jsonHeaders() });
        }
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
    !/^v[0-9]+(?:\.[0-9]+)*$/.test(release.tag_name) ||
    release.tag_name.split(".").some((part) => /^0*999999999$/.test(part)) || release.prerelease || release.draft ||
    !Array.isArray(release.assets)
  ) {
    return latestFailure("GitHub release API returned an incomplete release", env);
  }

  const payload = releasePayload(release);
  const requiredKinds = [...DOWNLOAD_KINDS].filter((kind) =>
    kind !== "linux-desktop" || linuxRequired(payload.tag));
  if (requiredKinds.some((kind) => !payload.assets[kind])) {
    const mirror = await mirroredRelease(env);
    if (mirror && mirror.tag === payload.tag) {
      for (const kind of DOWNLOAD_KINDS) {
        if (!payload.assets[kind] && mirror.assets[kind]) payload.assets[kind] = mirror.assets[kind];
      }
    }
  }
  const body = JSON.stringify(payload);
  // Do not cache a release while its split Windows assets are still mirroring.
  if (cache && requiredKinds.every((kind) => payload.assets[kind])) {
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

// Compare numeric components as strings so even unusually large versions stay exact.
function compareVersions(left, right) {
  const parse = (tag) => {
    const match = /^v([0-9]+(?:\.[0-9]+)*)(?:-beta\.([1-9][0-9]*))?$/.exec(tag);
    return match && { parts: match[1].split("."), beta: match[2] };
  };
  const a = parse(left);
  const b = parse(right);
  if (!a || !b) return null;
  const compareNumber = (x, y) => {
    x = x.replace(/^0+/, "") || "0";
    y = y.replace(/^0+/, "") || "0";
    return x.length - y.length || (x < y ? -1 : x > y ? 1 : 0);
  };
  for (let i = 0; i < Math.max(a.parts.length, b.parts.length); i++) {
    const difference = compareNumber(a.parts[i] || "0", b.parts[i] || "0");
    if (difference) return difference;
  }
  if (!a.beta || !b.beta) return a.beta ? -1 : b.beta ? 1 : 0;
  return compareNumber(a.beta, b.beta);
}

function deltaName(tag, base, platform) {
  const pattern = /^v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(?:-beta\.[1-9][0-9]*)?$/;
  if (typeof tag !== "string" || typeof base !== "string" ||
      !pattern.test(tag) || !pattern.test(base) ||
      [tag, base].some((version) => version.split("-beta.")[0].split(".").includes("999999999")) ||
      compareVersions(base, tag) >= 0 || !DELTA_FULL_KINDS.has(platform)) return null;
  if (base.includes("-beta.") && base.split("-beta.")[0] !== tag.split("-beta.")[0]) return null;
  return `OpenFocus-${tag}-${platform}-from-${base}.ofdelta.zip`;
}

function validDeltas(data) {
  if (!Array.isArray(data.deltas)) return [];
  const deltas = new Map();
  const duplicates = new Set();
  for (const entry of data.deltas) {
    if (!entry) continue;
    const { name, size, digest, from, platform } = entry;
    const full = data.assets[DELTA_FULL_KINDS.get(platform)];
    if (!full || !name || name !== deltaName(data.tag, from, platform) ||
        !Number.isSafeInteger(size) || size <= 0 || size >= full.size ||
        size >= 2147483648 || typeof digest !== "string" || !/^sha256:[0-9a-fA-F]{64}$/.test(digest)) continue;
    if (deltas.has(name)) duplicates.add(name);
    deltas.set(name, { name, size, digest, from, platform });
  }
  return [...deltas.values()].filter((entry) => !duplicates.has(entry.name));
}

function betaAlias(tag) {
  const match = /^(v[0-9]+(?:\.[0-9]+)*)-beta\.([1-9][0-9]*)$/.exec(tag);
  if (!match || match[1].split(".").includes("999999999")) return null;
  return `${match[1]}.999999999.${match[2]}`;
}

function previewUpdate(preview) {
  const alias = betaAlias(preview.tag);
  if (!alias || Object.values(preview.assets).some((asset) =>
    !asset || !/^sha256:[0-9a-fA-F]{64}$/.test(asset.digest || ""))) return null;
  const assets = Object.fromEntries(Object.entries(DOWNLOAD_SUFFIXES)
    .filter(([kind]) => preview.assets[kind])
    .map(([kind, suffix]) => [kind, {
      ...preview.assets[kind], name: `OpenFocus-${alias}-${suffix}`,
    }]));
  const result = { ...preview, tag: alias, actual_tag: preview.tag, assets };
  if (Array.isArray(preview.releases)) {
    result.releases = preview.releases.flatMap((release) => {
      if (!release || typeof release.tag_name !== "string") return [];
      const tag = betaAlias(release.tag_name) ||
        (/^v[0-9]+(?:\.[0-9]+)*$/.test(release.tag_name) ? release.tag_name : null);
      return tag ? [{ ...release, tag_name: tag }] : [];
    });
  }
  return result;
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

function recordWebsiteTraffic(db, ctx, downloadKind) {
  if (!ctx?.waitUntil) return;
  const now = Math.floor(Date.now() / 1000);
  const platform = downloadKind?.slice(0, downloadKind.indexOf("-"));
  try {
    ctx.waitUntil(Promise.resolve().then(() => db.prepare(`INSERT INTO website_traffic_daily
      (day, page_views, download_starts, windows_downloads, macos_downloads, linux_downloads)
      VALUES (date(?, 'unixepoch', '+8 hours'), ?, ?, ?, ?, ?)
      ON CONFLICT(day) DO UPDATE SET page_views = page_views + excluded.page_views,
      download_starts = download_starts + excluded.download_starts,
      windows_downloads = COALESCE(windows_downloads, 0) + excluded.windows_downloads,
      macos_downloads = COALESCE(macos_downloads, 0) + excluded.macos_downloads,
      linux_downloads = COALESCE(linux_downloads, 0) + excluded.linux_downloads`)
      .bind(now, Number(downloadKind === null), Number(downloadKind !== null),
        Number(platform === "windows"), Number(platform === "macos"), Number(platform === "linux"))
      .run()).catch(() => {}));
  } catch {
    // Statistics must not delay or break the page or file response.
  }
}

function isPrefetch(request) {
  return request.headers.get("Sec-Purpose")?.includes("prefetch") || request.headers.get("Purpose") === "prefetch";
}

// Analytics is optional and must never interfere with serving a download.
function recordDownload(request, response, env, ctx, version, kind, channel, source) {
  const siteAnalytics = channel === "website" && env?.SITE_ANALYTICS;
  if ((!siteAnalytics && !env?.ANALYTICS) || request.method !== "GET" || !ctx?.waitUntil) return;
  if (response.status !== 200 && response.status !== 302 &&
      !(response.status === 206 && /^bytes 0-\d+\/\d+$/.test(response.headers.get("Content-Range") || ""))) return;
  if (siteAnalytics && new URL(request.url).hostname === "openfocus.top" && !isPrefetch(request)) {
    recordWebsiteTraffic(siteAnalytics, ctx, kind);
  }
  if (!env.ANALYTICS || typeof version !== "string" || !/^v[0-9A-Za-z._+-]{1,80}$/.test(version)) return;
  try {
    const operation = Promise.resolve().then(() => env.ANALYTICS.prepare(
      "INSERT INTO download_counts (day, version, kind, channel, source, count) VALUES (date('now'), ?, ?, ?, ?, 1) " +
      "ON CONFLICT(day, version, kind, channel, source) DO UPDATE SET count = count + 1"
    ).bind(version, kind, channel, source).run()).catch(() => {});
    ctx.waitUntil(operation);
  } catch {
    // A missing or broken analytics binding must not break the response.
  }
}

async function usageRequest(request, env) {
  if (request.method !== "POST") return textResponse("Method Not Allowed", 405, { Allow: "POST" }, request.method);
  if (!env?.ANALYTICS) return textResponse("Analytics unavailable", 503);
  if (!/^application\/json(?:\s*;|\s*$)/i.test(request.headers.get("Content-Type") || "")) {
    return textResponse("Expected JSON", 415);
  }
  const declaredLength = request.headers.get("Content-Length");
  if (declaredLength !== null && (!/^\d+$/.test(declaredLength) || Number(declaredLength) > 1024)) {
    return textResponse("Request too large", 413);
  }
  let payload;
  try {
    const reader = request.body?.getReader();
    if (!reader) return textResponse("Invalid JSON", 400);
    const chunks = [];
    let size = 0;
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      size += value.byteLength;
      if (size > 1024) {
        reader.cancel().catch(() => {});
        return textResponse("Request too large", 413);
      }
      chunks.push(value);
    }
    const bytes = new Uint8Array(size);
    let offset = 0;
    for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.byteLength; }
    payload = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes));
  } catch {
    return textResponse("Invalid JSON", 400);
  }
  if (!payload || typeof payload !== "object" || Array.isArray(payload) ||
      Object.keys(payload).sort().join(",") !== "installation_id,platform,version" ||
      typeof payload.installation_id !== "string" ||
      !/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(payload.installation_id) ||
      typeof payload.version !== "string" || !/^[0-9]+(?:\.[0-9]+){1,3}$/.test(payload.version) || payload.version.length > 40 ||
      !["windows", "macos", "linux"].includes(payload.platform)) {
    return textResponse("Invalid usage report", 400);
  }
  try {
    await env.ANALYTICS.batch([
      env.ANALYTICS.prepare("DELETE FROM active_installations WHERE day < date('now', '-89 days')"),
      env.ANALYTICS.prepare("INSERT INTO active_installations (day, installation_id, version, platform) VALUES (date('now'), ?, ?, ?) ON CONFLICT(day, installation_id, version, platform) DO NOTHING")
        .bind(payload.installation_id.toLowerCase(), payload.version, payload.platform),
    ]);
    return new Response(null, { status: 204, headers: { "Cache-Control": NO_STORE } });
  } catch {
    return textResponse("Analytics unavailable", 503);
  }
}

async function adminStats(request, env) {
  if (request.method !== "GET") return textResponse("Method Not Allowed", 405, { Allow: "GET" }, request.method);
  if (typeof env?.ANALYTICS_ADMIN_SECRET !== "string" || !env.ANALYTICS_ADMIN_SECRET || !env.ANALYTICS) {
    return textResponse("Analytics unavailable", 503);
  }
  const authorization = request.headers.get("Authorization") || "";
  const candidate = authorization.startsWith("Bearer ") ? authorization.slice(7) : "";
  const encoder = new TextEncoder();
  const [expected, actual] = await Promise.all([
    crypto.subtle.digest("SHA-256", encoder.encode(env.ANALYTICS_ADMIN_SECRET)),
    crypto.subtle.digest("SHA-256", encoder.encode(candidate)),
  ]);
  const a = new Uint8Array(expected), b = new Uint8Array(actual);
  let difference = 0;
  for (let i = 0; i < a.length; i++) difference |= a[i] ^ b[i];
  if (!candidate || difference !== 0) return textResponse("Unauthorized", 401, { "WWW-Authenticate": "Bearer" });
  try {
    const [downloads, active] = await env.ANALYTICS.batch([
      env.ANALYTICS.prepare("SELECT day, version, kind, channel, source, count FROM download_counts ORDER BY day DESC, version, kind, channel, source"),
      env.ANALYTICS.prepare("SELECT version, platform, COUNT(DISTINCT installation_id) AS installations FROM active_installations WHERE day >= date('now', '-29 days') AND day <= date('now') GROUP BY version, platform ORDER BY version, platform"),
    ]);
    return new Response(JSON.stringify({ downloads: downloads.results, active_30_days: active.results }), { headers: jsonHeaders() });
  } catch {
    return textResponse("Analytics unavailable", 503);
  }
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
    latest = await stableRelease(ctx, env);
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
  if (mirrored) {
    recordDownload(request, mirrored, env, ctx, data.tag, kind, "website", "r2");
    return mirrored;
  }

  if (!remote || request.headers.get("Range")) {
    return textResponse(
      remote ? "The download mirror is not ready for this range; retry the download."
        : "The download mirror is not ready; retry or use the release page for recovery.",
      503,
      { "Retry-After": "60" },
      method,
    );
  }

  const response = new Response(null, {
    status: 302,
    headers: { Location: remote, "Cache-Control": NO_STORE },
  });
  recordDownload(request, response, env, ctx, data.tag, kind, "website", "github");
  return response;
}

async function handleVersionedDownload(request, env, ctx) {
  const method = (request.method || "GET").toUpperCase();
  if (method !== "GET" && method !== "HEAD") {
    return textResponse("Method Not Allowed", 405, { Allow: "GET, HEAD" }, method);
  }
  const parts = new URL(request.url).pathname.slice(VERSIONED_DOWNLOAD_PREFIX.length).split("/");
  const [tag, kind] = parts;
  if (kind === "delta") {
    const [, , base, platform] = parts;
    const name = parts.length === 4 && deltaName(tag, base, platform);
    if (!name) return textResponse("Unknown release delta", 404, {}, method);
    // The canonical name is the whole R2 key. Older pinned requests may still
    // finish after the corresponding metadata has advanced to a newer release.
    const response = await r2Download(request, env, name, name);
    if (response) recordDownload(request, response, env, ctx, tag, kind, "in-app", "r2");
    return response || textResponse("The delta mirror is not ready; retry the download.", 503, { "Retry-After": "60" }, method);
  }
  const alias = /^(v[0-9]+(?:\.[0-9]+)*)\.999999999\.([1-9][0-9]*)$/.exec(tag);
  const objectTag = alias && !alias[1].split(".").some((part) => /^0*999999999$/.test(part))
    ? `${alias[1]}-beta.${alias[2]}` : tag;
  if (parts.length !== 2 || !/^v[0-9]+(?:\.[0-9]+)*(?:-beta\.[1-9][0-9]*)?$/.test(tag) ||
      (tag.split("-beta.")[0].split(".").some((part) => /^0*999999999$/.test(part)) && !alias) ||
      (alias && objectTag === tag) || !DOWNLOAD_KINDS.has(kind)) {
    return textResponse("Unknown release download", 404, {}, method);
  }
  const name = `OpenFocus-${objectTag}-${DOWNLOAD_SUFFIXES[kind]}`;
  const downloadName = `OpenFocus-${tag}-${DOWNLOAD_SUFFIXES[kind]}`;
  const response = await r2Download(request, env, name, downloadName);
  if (response) recordDownload(request, response, env, ctx, tag, kind, "in-app", "r2");
  return response || textResponse("The download mirror is not ready; retry the download.", 503, { "Retry-After": "60" }, method);
}

export default {
  async scheduled(_event, env, ctx) {
    if (!env?.ANALYTICS) return;
    ctx.waitUntil(Promise.resolve().then(() => env.ANALYTICS.prepare(
      "DELETE FROM active_installations WHERE day < date('now', '-89 days')"
    ).run()).catch(() => {}));
  },
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    const method = (request.method || "GET").toUpperCase();

    const qrId = QR_ROUTES.get(url.pathname);
    if (qrId) return proxyQr(qrId, method);

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
    if (url.pathname === "/api/usage") return usageRequest(request, env);
    if (url.pathname === "/api/admin/stats") return adminStats(request, env);


    if (url.pathname === "/api/updates") {
      if (method !== "GET" && method !== "HEAD") {
        return textResponse("Method Not Allowed", 405, { Allow: "GET, HEAD" }, method);
      }
      const channel = url.searchParams.get("channel") || "stable";
      if ((channel !== "stable" && channel !== "beta") || url.searchParams.getAll("channel").length > 1) {
        return textResponse("Unknown update channel", 400, {}, method);
      }
      const [stable, preview] = await Promise.all([
        mirroredRelease(env), channel === "beta" ? mirroredRelease(env, PREVIEW_MANIFEST) : null,
      ]);
      const candidate = preview && previewUpdate(preview);
      const mirror = candidate && (!stable || compareVersions(preview.tag, stable.tag) > 0) ? candidate : stable;
      if (!mirror) return textResponse("Update mirror is unavailable", 503, { "Retry-After": "60" }, method);
      return new Response(method === "HEAD" ? null : JSON.stringify(mirror), {
        headers: { ...jsonHeaders(), "X-OpenFocus-Release-Source": "mirror" },
      });
    }

    if (url.pathname.startsWith(VERSIONED_DOWNLOAD_PREFIX)) {
      return handleVersionedDownload(request, env, ctx);
    }

    if (url.pathname === "/api/latest") {
      if (method !== "GET" && method !== "HEAD") {
        return textResponse("Method Not Allowed", 405, { Allow: "GET, HEAD" }, method);
      }
      const response = await stableRelease(ctx, env);
      if (method === "HEAD") {
        return new Response(null, { status: response.status, headers: response.headers });
      }
      return response;
    }

    if (url.pathname.startsWith(DOWNLOAD_PREFIX)) {
      return handleDownload(request, env, ctx);
    }

    let languageHeaders = null;
    const route = LOCALIZED_ROUTES.get(url.pathname);
    const homepage = Boolean(route?.home);
    if (route && (method === "GET" || method === "HEAD")) {
      const query = url.searchParams.get("lang");
      const explicit = query === "en" || query === "zh";
      const lang = explicit ? query : route.lang || preferredLanguage(request);
      if (route.lang !== "zh" || explicit || route.canonicalize) {
        languageHeaders = new Headers({
          Vary: "Accept-Language, Cookie",
          "Cache-Control": NO_STORE,
        });
        if (explicit) {
          languageHeaders.set("Set-Cookie", `lang=${lang}; Path=/; Max-Age=31536000; SameSite=Lax${url.protocol === "https:" ? "; Secure" : ""}`);
        }
      }
      const pathname = lang === "zh" ? route.zh : route.en;
      if (url.pathname !== pathname) {
        languageHeaders.set("Location", pathname + url.search);
        return new Response(null, { status: 302, headers: languageHeaders });
      }
    }

    if (!env || !env.ASSETS || typeof env.ASSETS.fetch !== "function") {
      return textResponse("Not Found", 404, {}, method);
    }

    const response = await env.ASSETS.fetch(request);
    if (env.SITE_ANALYTICS && homepage && url.hostname === "openfocus.top" && method === "GET" &&
        (response.status === 304 || (response.status === 200 && response.headers.get("Content-Type")?.startsWith("text/html"))) &&
        !isPrefetch(request)) {
      recordWebsiteTraffic(env.SITE_ANALYTICS, ctx, null);
    }
    if (!languageHeaders) return response;
    const headers = new Headers(response.headers);
    for (const [key, value] of languageHeaders) headers.set(key, value);
    return new Response(method === "HEAD" ? null : response.body, {
      status: response.status,
      statusText: response.statusText,
      headers,
    });
  },
};
