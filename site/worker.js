/**
 * OpenFocus site worker.
 *
 * - "/" redirects to /zh/ when the browser prefers Chinese (magpie-style,
 *   Accept-Language + a `lang` cookie set by the language menu).
 * - "/api/latest" returns the latest GitHub release, cached at the edge.
 * - "/download/latest/<kind>" redirects to that release's asset.
 *
 * Everything else is served as a static asset without invoking this Worker.
 */

const REPO = "Xinzhe99/OpenFocus";
const RELEASE_API = `https://api.github.com/repos/${REPO}/releases/latest`;
const CACHE_URL = "https://openfocus.internal/latest-release.json";

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

async function latestRelease(ctx) {
  const cache = caches.default;
  const key = new Request(CACHE_URL);
  const hit = await cache.match(key);
  if (hit) return hit;

  const upstream = await fetch(RELEASE_API, {
    headers: {
      "User-Agent": "openfocus-site",
      Accept: "application/vnd.github+json",
    },
  });
  if (!upstream.ok) {
    return new Response("{}", {
      status: 502,
      headers: { "Content-Type": "application/json", "Cache-Control": "no-store" },
    });
  }

  const release = await upstream.json();
  const pick = (pattern) => {
    const asset = (release.assets || []).find((item) => pattern.test(item.name));
    return asset
      ? { name: asset.name, size: asset.size, url: asset.browser_download_url }
      : null;
  };

  const body = JSON.stringify({
    tag: release.tag_name,
    published: release.published_at,
    assets: {
      "windows-setup": pick(/-setup\.exe$/i),
      "windows-zip": pick(/-windows-x64\.zip$/i),
      "macos-dmg": pick(/-macos\.dmg$/i),
      "macos-zip": pick(/-macos-arm64\.zip$/i),
    },
  });

  const response = new Response(body, {
    headers: {
      "Content-Type": "application/json; charset=utf-8",
      "Cache-Control": "public, max-age=300",
    },
  });
  ctx.waitUntil(cache.put(key, response.clone()));
  return response;
}

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);

    if (url.hostname === "www.openfocus.top") {
      const target = new URL(request.url);
      target.hostname = "openfocus.top";
      return Response.redirect(target.toString(), 301);
    }

    if (url.pathname === "/api/latest") {
      return latestRelease(ctx);
    }

    if (url.pathname.startsWith("/download/latest/")) {
      const kind = url.pathname.slice("/download/latest/".length);
      let data = {};
      try {
        data = await latestRelease(ctx).then((response) => response.json());
      } catch {}
      const asset = data.assets && data.assets[kind];
      if (!asset) return new Response("Unknown download", { status: 404 });

      let object = null;
      const rangeHeader = request.headers.get("Range");
      try {
        object = await env.DOWNLOADS.get(
          asset.name,
          rangeHeader ? { range: request.headers } : undefined,
        );
      } catch {}

      if (object) {
        const headers = new Headers();
        object.writeHttpMetadata(headers);
        headers.set("Content-Type", "application/octet-stream");
        headers.set("Content-Disposition", `attachment; filename="${asset.name}"`);
        headers.set("Accept-Ranges", "bytes");
        const range = object.range;
        if (rangeHeader && range && range.offset !== undefined) {
          const length = range.length ?? object.size - range.offset;
          headers.set("Content-Range", `bytes ${range.offset}-${range.offset + length - 1}/${object.size}`);
          headers.set("Content-Length", String(length));
          headers.set("Cache-Control", "no-store");
          return new Response(object.body, { status: 206, headers });
        }
        if (rangeHeader && range && range.suffix !== undefined) {
          headers.set("Content-Range", `bytes -${range.suffix}/${object.size}`);
          headers.set("Content-Length", String(range.suffix));
          headers.set("Cache-Control", "no-store");
          return new Response(object.body, { status: 206, headers });
        }
        headers.set("Content-Length", String(object.size));
        headers.set("Cache-Control", "public, max-age=86400");
        return new Response(object.body, { status: 200, headers });
      }

      return new Response(null, {
        status: 302,
        headers: { Location: asset.url, "Cache-Control": "public, max-age=300" },
      });
    }

    if (url.pathname === "/" && request.method === "GET") {
      if (preferredLanguage(request) === "zh") {
        return new Response(null, {
          status: 302,
          headers: {
            Location: "/zh/",
            Vary: "Accept-Language, Cookie",
            "Cache-Control": "no-store",
          },
        });
      }
    }

    return env.ASSETS.fetch(request);
  },
};
