# OpenFocus website

Static product site for OpenFocus: English at `/`, Simplified Chinese at
`/zh/`. No framework, no runtime, no third-party requests — one HTML file per
language with CSS and JavaScript inlined, deployed as Cloudflare Workers
static assets.

Full operating manual (architecture, R2 mirror, release checklist,
troubleshooting): [`docs/WEBSITE_OPERATIONS.md`](../docs/WEBSITE_OPERATIONS.md).

## Build

```bash
python tools/build_site.py
```

Output goes to `site/dist/` (git-ignored). The build also writes `robots.txt`,
`sitemap.xml`, `_headers` and a bilingual `404.html`, and derives the icon,
Open Graph image and WebP screenshots from `assets/`.

Set the public Worker origin so canonical URLs, Open Graph tags, the sitemap
and no-JavaScript download links point at the correct deployment:

```bash
SITE_ORIGIN=https://openfocus.example.com python tools/build_site.py
```

On PowerShell:

```powershell
$env:SITE_ORIGIN = "https://openfocus.example.com"; python tools/build_site.py
```

## Preview

```bash
python -m http.server 8788 --directory site/dist
```

`http://127.0.0.1:8788/` is English, `/zh/` is Chinese. This static server
does not implement the Worker download/API routes or the custom 404 page.

## Deploy

A production deploy needs an authenticated Cloudflare account that can reach
the `openfocus.top` zone, the `openfocus-site` Worker, the
`openfocus-downloads` R2 bucket and, if enabled, the analytics D1 database.
Use your account-specific config when analytics is enabled: deploying the
unbound repository template would turn statistics off.

```bash
npx wrangler login
npx wrangler deploy -c path/to/wrangler.analytics.jsonc
```

`site/wrangler.jsonc` binds the production custom domains and the R2 bucket,
so Wrangler deploys to that Worker. It is not an account-free deploy: if the
account cannot see the zone, Worker or bucket, the command fails instead of
creating a public site.

To check that the Worker bundles and the configuration parses without
uploading anything:

```bash
npx wrangler deploy -c site/wrangler.jsonc --dry-run
```

`--temporary` creates a temporary preview account. It cannot bind
`openfocus.top`, `www.openfocus.top` or the existing `openfocus-downloads`
bucket, so it is not a preview path for this configuration.

## Optional analytics

Download starts can be aggregated in D1, and the desktop app can send daily
activity reports by default (disable in Settings → Advanced Settings). Existing
explicit opt-outs are preserved. The Worker keeps downloads working if D1
is absent or unavailable. The admin statistics API requires a server-only
bearer secret. See [analytics operations](../docs/WEBSITE_OPERATIONS.md#optional-private-analytics)
for database creation, binding, migrations, secret setup, privacy, retention and
the limits of these counts. No analytics database ID or secret is in this repo.
The English and Chinese home pages include a privacy section linked from the
footer; publish it before enabling the analytics binding.

## Editing

| File | Holds |
|---|---|
| `site/site_content.py` | Copy, links and download cards; the current desktop version comes from `/api/latest` |
| `site/template.html` | Page structure; `{{placeholders}}` resolve from the content |
| `site/style.css` | Design tokens (light/dark), layout, responsive rules |
| `site/app.js` | Theme, OS-aware navigation download, Linux desktop availability, copy button and menus |
| `site/worker.js` | `/` language negotiation; `www` → apex; `/api/latest` and `/download/latest/*` for website downloads; mirror-only `/api/updates` and `/download/releases/<tag>/<kind>` for desktop updates |
| `site/install.sh` | macOS app installer + Linux CLI installer (`curl … | sh`, `--dry-run`) |
| `site/install.ps1` | Windows installer (`irm … | iex`, `OPENFOCUS_DRY_RUN=1`) |
| `site/wrangler.jsonc` | Worker name and static-asset routing |
| `tools/build_site.py` | Build: assets, pages, sitemap, headers, 404 |

Adding a language: add one dict to `LOCALES` with the same keys, rebuild. The
language menu, `hreflang` tags and sitemap pick it up automatically.

The theme button cycles through light, dark and system modes. Each selection is
saved when browser storage is available; if storage is blocked, the selection
and system changes still work for the current page. With JavaScript disabled,
the English and Chinese links and all navigation links remain available,
including on mobile screens.

The Worker resolves `/api/latest` and `/download/latest/*` from release
metadata. Linux x86-64 desktop downloads are shown only when `/api/latest`
reports `assets["linux-desktop"]`; without JavaScript or when metadata cannot
be fetched, the page explains that availability cannot be checked. Historical
releases may not include the desktop archive. Linux navigation points to the
download section, and `install.sh` continues to install the separate PyPI
headless CLI on Linux. Windows and macOS links work without JavaScript. With
JavaScript enabled, the desktop version and file sizes come from live metadata.
Complete GitHub metadata is cached for five minutes per data center; clients
receive `no-store`. The Worker falls back to R2 metadata when GitHub is
unavailable. Range requests require the R2 object.

## Download mirror (R2)

Installers are mirrored to Cloudflare R2 and served from
`/download/latest/<kind>`. The Worker streams the complete R2 object; a full
request can fall back to GitHub when the complete asset exists there.

Desktop updates use `GET` or `HEAD /api/updates?channel=stable` for stable
releases, or `?channel=beta` to select the newer verified stable or beta R2
manifest, without contacting GitHub. An omitted channel defaults to stable.
A beta uses a numeric compatibility tag of the form
`v<version>.999999999.<beta-number>` and its real tag in `actual_tag`.
Clients validate the association and display the beta tag. Versioned alias
downloads continue to resolve to the same beta bytes. Metadata carries the
asset sizes and SHA-256 digests, including `linux-desktop` when available.

`GET` and `HEAD /download/releases/<tag>/<kind>` serve stable, beta and
compatibility-alias downloads from R2, with byte ranges and ETag validation.
The recognized kinds are `windows-setup`, `windows-zip`, `macos-dmg`,
`macos-zip` and `linux-desktop`. Invalid paths return `404`; missing objects return `503` without a
GitHub redirect. Stable objects may be pruned after a newer stable release;
beta objects are retained for version-pinned and cached update requests.

- Bucket: `openfocus-downloads` (binding `DOWNLOADS`)
- Workflow: `.github/workflows/r2-mirror.yml` runs after a successful
  **Build & Publish Release** run, on manually published releases, and on
  manual dispatch. It verifies and uploads complete release files, including
  Linux desktop when present, before publishing either `openfocus-meta/latest-release.json` for the latest stable
  release or `openfocus-meta/beta-release.json` for the newest beta. Beta
  mirroring never replaces the stable manifest. Stable pruning preserves beta
  assets; beta mirrors do not prune. Runs are serialized.
- Secrets: `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY` (an R2
  API token with Object Read & Write on the bucket).
- Free tier: 10 GB-month storage, 1M Class A and 10M Class B operations per
  month; egress is free.

## Mainland China

Cloudflare's free network is best-effort from mainland China: it often works
because the site is small, static and edge-cached, but it has no SLA. A custom
domain on Cloudflare still resolves to an overseas edge unless the account has
Cloudflare China Network (Enterprise + ICP filing). For a guaranteed mainland
deployment, put the same `site/dist/` bundle behind an ICP-filed CDN such as
Alibaba Cloud OSS + CDN or Tencent Cloud COS + CDN.
