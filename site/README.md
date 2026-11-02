# OpenFocus website

Static marketing site for OpenFocus: English at `/`, Simplified Chinese at
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

Set the public origin so canonical URLs, Open Graph tags and the sitemap are
correct:

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

`http://127.0.0.1:8788/` is English, `/zh/` is Chinese. The local server does
not serve the custom 404 page; Cloudflare does.

## Deploy

A production deploy needs an authenticated Cloudflare account that can reach
the `openfocus.top` zone, the `openfocus-site` Worker and the
`openfocus-downloads` R2 bucket:

```bash
npx wrangler login
npx wrangler deploy -c site/wrangler.jsonc
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

## Editing

| File | Holds |
|---|---|
| `site/site_content.py` | All copy, links, release tag and download assets |
| `site/template.html` | Page structure; `{{placeholders}}` resolve from the content |
| `site/style.css` | Design tokens (light/dark), layout, responsive rules |
| `site/app.js` | Theme, OS-aware download button, copy button, menus |
| `site/worker.js` | `/` language negotiation; `www` → apex on every path; `/api/latest` (5-minute per-data-center Worker cache, client `no-store`); `/download/latest/*` (R2 stream with range/ETag, GitHub `302` fallback for full requests, `503` when a range needs a missing object) |
| `site/install.sh` | macOS app installer + Linux CLI installer (`curl … | sh`, `--dry-run`) |
| `site/install.ps1` | Windows installer (`irm … | iex`, `OPENFOCUS_DRY_RUN=1`) |
| `site/wrangler.jsonc` | Worker name and static-asset routing |
| `tools/build_site.py` | Build: assets, pages, sitemap, headers, 404 |

Adding a language: add one dict to `LOCALES` with the same keys, rebuild. The
language menu, `hreflang` tags and sitemap pick it up automatically.

Releasing: the Worker resolves `/api/latest` from GitHub and serves
`/download/latest/*` from R2 (falling back to a `302` GitHub asset URL for a
full request), so the page's version label, file sizes and download links
follow the latest release without a rebuild. `/api/latest` clients receive
`no-store`; the Worker itself caches the payload for five minutes per data
center. The values in `site/site_content.py` are the no-JavaScript fallback —
bump `RELEASE` when a release is cut so that fallback stays current.

## Download mirror (R2)

Installers are mirrored to Cloudflare R2 and served from
`/download/latest/<kind>`. The Worker streams the R2 object when it exists and
falls back to the GitHub release asset for a full request when it does not.

- Bucket: `openfocus-downloads` (binding `DOWNLOADS`)
- Workflow: `.github/workflows/r2-mirror.yml` runs after a successful
  **Build & Publish Release** run for a `v*` tag (the release event is
  suppressed for `GITHUB_TOKEN` releases), on manually published releases, and
  on manual dispatch (`tag` input). It downloads the four assets with `gh`,
  verifies exactly four non-empty assets, uploads them through the R2 S3
  endpoint under their release filenames, and prunes only older recognized
  `OpenFocus-v*` keys after confirming the mirrored tag is the latest release.
  Runs are serialized so an older re-run cannot prune a newer tag.
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
