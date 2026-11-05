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
| `site/site_content.py` | Copy, links and download cards; static version comes from `constants.APP_VERSION` |
| `site/template.html` | Page structure; `{{placeholders}}` resolve from the content |
| `site/style.css` | Design tokens (light/dark), layout, responsive rules |
| `site/app.js` | Theme, OS-aware navigation download, copy button, menus; the hero shows both desktop downloads |
| `site/worker.js` | `/` language negotiation; `www` → apex; `/api/latest` and `/download/latest/*` for website downloads; mirror-only `/api/updates` and `/download/releases/<tag>/<kind>` for desktop updates |
| `site/install.sh` | macOS app installer + Linux CLI installer (`curl … | sh`, `--dry-run`) |
| `site/install.ps1` | Windows installer (`irm … | iex`, `OPENFOCUS_DRY_RUN=1`) |
| `site/wrangler.jsonc` | Worker name and static-asset routing |
| `tools/build_site.py` | Build: assets, pages, sitemap, headers, 404 |

Adding a language: add one dict to `LOCALES` with the same keys, rebuild. The
language menu, `hreflang` tags and sitemap pick it up automatically.

Releasing: the Worker resolves `/api/latest` from GitHub and serves
`/download/latest/*` from R2. For a split Windows release it fills missing
complete assets from the validated same-tag mirror manifest. A complete GitHub
asset is a fallback for full requests; a mirror-only asset or range request
returns `503` until R2 is ready. Complete GitHub metadata is cached for five
minutes per data center, while clients receive `no-store`. If GitHub is
unavailable, the Worker serves the last complete mirror manifest with
`X-OpenFocus-Release-Source: mirror`.
Without JavaScript, download links still use `/download/latest/*`. Static
version labels and JSON-LD are generated from `constants.APP_VERSION`; rebuild
and redeploy when those labels must match the latest release.
File sizes are shown only when live release metadata is available, avoiding
stale sizes in static pages.

## Download mirror (R2)

Installers are mirrored to Cloudflare R2 and served from
`/download/latest/<kind>`. The Worker streams the complete R2 object; a full
request can fall back to GitHub when the complete asset exists there.

Desktop updates use `GET` or `HEAD /api/updates` to read the R2 manifest
directly. It contains the latest mirrored tag, publication date and four
assets; newer manifests also include each asset's GitHub SHA-256 `digest` and
up to 30 public stable releases as `{tag_name, body}`. Older manifests can
omit both fields. An unavailable or invalid manifest returns `503` without
contacting GitHub.

`GET` and `HEAD /download/releases/<tag>/<kind>` serve only R2 objects,
supporting byte ranges and ETag validation. Tags use `v` followed by numeric
components separated by dots, such as `v1.10` or `v1.2.3`. Kinds are
`windows-setup`, `windows-zip`, `macos-dmg` and `macos-zip`. Invalid paths
return `404`, and missing or unavailable objects return `503` without a GitHub
redirect. Older objects are pruned after the latest release is mirrored, so a
previously advertised version may become unavailable.

- Bucket: `openfocus-downloads` (binding `DOWNLOADS`)
- Workflow: `.github/workflows/r2-mirror.yml` runs after a successful
  **Build & Publish Release** run for a `v*` tag (the release event is
  suppressed for `GITHUB_TOKEN` releases), on manually published releases, and
  on manual dispatch (`tag` input). It downloads and verifies Windows parts,
  reassembles oversized files, downloads the macOS assets, and uploads the four
  complete files under their release filenames. Only after all uploads succeed
  does it publish digests and stable release notes under
  `openfocus-meta/latest-release.json` for the current latest release.
  After the five-minute metadata cache expires, it prunes older recognized
  `OpenFocus-v*` keys. Runs are serialized.
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
