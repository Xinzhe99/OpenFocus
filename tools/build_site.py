"""Build the OpenFocus product site into site/dist.

Zero dependencies beyond Pillow (already a runtime dependency of the app).

    python tools/build_site.py
    SITE_ORIGIN=http://127.0.0.1:8788 python tools/build_site.py   # local preview

The default origin is https://openfocus.top; set SITE_ORIGIN to override the
canonical URLs, sitemap, Open Graph tags and installer commands.

The output is plain static files: one HTML page per language, inlined CSS
and JavaScript, optimised images, robots.txt, sitemap.xml, _headers and a
404 page. See site/README.md for the production deployment configuration.
"""

from __future__ import annotations

import html
import json
import re
import shutil
import sys
from datetime import date
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
DIST = SITE / "dist"
ASSETS = DIST / "assets"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SITE))
from site_content import LOCALES, ORIGIN, RELEASE, REPO, MANUAL  # noqa: E402

BENCH_LINK = f"{REPO}/blob/main/tools/synthetic_benchmark.py"
CLI_LINKS = dict(MANUAL)


def flatten(data: dict, prefix: str = "") -> dict:
    out = {}
    for key, value in data.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(flatten(value, f"{name}."))
        else:
            out[name] = value
    return out


def esc(value: str) -> str:
    return html.escape(value, quote=True)


def render_features(features: list[dict]) -> str:
    return "".join(
        f"<div><h3>{esc(item['title'])}</h3><p>{item['body']}</p></div>"
        for item in features
    )


def render_bullets(items: list[dict]) -> str:
    return "".join(
        f"<li><b>{esc(item['b'])}</b><span>{esc(item['t'])}</span></li>"
        for item in items
    )


def render_chips(chips: list[str]) -> str:
    return "".join(f"<li>{esc(chip)}</li>" for chip in chips)


def render_downloads(locale: dict) -> str:
    action = esc(locale["download"]["action"])
    cards = []
    for card in locale["download"]["cards"]:
        kind = esc(card["kind"])
        size = f" · {esc(card['size'])}" if card.get("size") else ""
        kind_attr = f' data-kind="{esc(card["kind_id"])}"' if card.get("kind_id") else ""
        cards.append(
            '<a class="plat" href="{url}"{kind_attr} rel="noopener">'
            "<b>{os}</b><span>{kind}{size}</span><i>{action} →</i></a>".format(
                url=esc(card["url"]),
                kind_attr=kind_attr,
                os=esc(card["os"]),
                kind=kind,
                size=size,
                action=action,
            )
        )
    return "".join(cards)


def render_lang_menu(current: dict) -> str:
    links = []
    for code, locale in LOCALES.items():
        href = f"/{locale['path']}?lang={code}"
        current_attr = ' aria-current="page"' if code == current["code"] else ""
        links.append(
            f'<a href="{href}" hreflang="{code}" lang="{locale["html_lang"]}"'
            f"{current_attr}>{esc(locale['switch_label'])}</a>"
        )
    return "".join(links)


def render_alternates() -> str:
    rows = [
        f'<link rel="alternate" hreflang="{code}" href="{ORIGIN}/{locale["path"]}">'
        for code, locale in LOCALES.items()
    ]
    rows.append(f'<link rel="alternate" hreflang="x-default" href="{ORIGIN}/">')
    return "\n".join(rows)


def render_footer_links(locale: dict) -> str:
    return "".join(
        f'<a href="{esc(url)}" rel="noopener">{esc(label)}</a>'
        for label, url in locale["footer"]["links"]
    )


def render_cli(lines: list[str]) -> str:
    out = []
    for line in lines:
        text = esc(line)
        text = re.sub(r"(--[a-z-]+)", r'<span class="flag">\1</span>', text)
        out.append(text)
    return "\n".join(out)


def json_ld(locale: dict, canonical: str) -> str:
    data = {
        "@context": "https://schema.org",
        "@type": "SoftwareApplication",
        "name": "OpenFocus",
        "applicationCategory": "MultimediaApplication",
        "operatingSystem": "Windows 10, Windows 11, macOS 13+",
        "description": locale["desc"],
        "url": canonical,
        "downloadUrl": RELEASE["windows_setup"],
        "inLanguage": locale["html_lang"],
        "license": "https://opensource.org/license/mit",
        "offers": {"@type": "Offer", "price": "0", "priceCurrency": "USD"},
        "sameAs": [REPO],
    }
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def context_for(locale: dict) -> dict:
    canonical = f"{ORIGIN}/{locale['path']}"
    ctx = flatten(locale)
    ctx.update(
        {
            "css": (SITE / "style.css").read_text(encoding="utf-8"),
            "js": (SITE / "app.js").read_text(encoding="utf-8"),
            "canonical": canonical,
            "alternates": render_alternates(),
            "jsonld": json_ld(locale, canonical),
            "origin": ORIGIN,
            "skip": locale["skip"],
            "repo": REPO,
            "releases": f"{REPO}/releases",
            "manual": MANUAL[locale["code"]],
            "bench_link": BENCH_LINK,
            "cli_link": CLI_LINKS[locale["code"]],
            "dl_windows": RELEASE["windows_setup"],
            "dl_macos": RELEASE["macos_dmg"],
            "dl_linux": RELEASE["pypi"],
            "chips": render_chips(locale["hero"]["chips"]),
            "features": render_features(locale["why"]["features"]),
            "workflow_bullets": render_bullets(locale["workflow"]["bullets"]),
            "detail_bullets": render_bullets(locale["detail"]["bullets"]),
            "download_cards": render_downloads(locale),
            "footer_links": render_footer_links(locale),
            "lang_menu": render_lang_menu(locale),
            "cli_lines": render_cli(locale["cli"]["lines"]),
        }
    )
    return ctx


TOKEN = re.compile(r"\{\{\s*([a-zA-Z0-9_.]+)\s*\}\}")


def render(template: str, ctx: dict, page: str) -> str:
    missing = set()

    def replace(match: re.Match) -> str:
        key = match.group(1)
        if key not in ctx:
            missing.add(key)
            return match.group(0)
        value = ctx[key]
        if not isinstance(value, str):
            raise TypeError(f"{page}: {key} is {type(value).__name__}, expected str")
        return value

    out = TOKEN.sub(replace, template)
    if missing:
        raise KeyError(f"{page}: unresolved placeholders: {', '.join(sorted(missing))}")
    return out


def cover(image: Image.Image, width: int, height: int) -> Image.Image:
    ratio = max(width / image.width, height / image.height)
    resized = image.resize(
        (max(1, round(image.width * ratio)), max(1, round(image.height * ratio))),
        Image.Resampling.LANCZOS,
    )
    left = (resized.width - width) // 2
    top = (resized.height - height) // 2
    return resized.crop((left, top, left + width, top + height))


def save_webp(source: Path, target: Path, width: int, quality: int = 84) -> None:
    image = Image.open(source)
    if image.width > width:
        height = round(image.height * width / image.width)
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGBA" if "A" in image.getbands() else "RGB")
    image.save(target, "WEBP", quality=quality, method=6)


def build_assets() -> list[Path]:
    ASSETS.mkdir(parents=True, exist_ok=True)
    made: list[Path] = []

    icon = Image.open(ROOT / "assets/OpenFocus.png").convert("RGBA")
    icon.resize((192, 192), Image.Resampling.LANCZOS).save(ASSETS / "icon-192.png", optimize=True)
    icon.resize((180, 180), Image.Resampling.LANCZOS).save(
        ASSETS / "apple-touch-icon.png", optimize=True
    )
    icon.save(DIST / "favicon.ico", sizes=[(16, 16), (32, 32), (48, 48)])
    made += [ASSETS / "icon-192.png", ASSETS / "apple-touch-icon.png", DIST / "favicon.ico"]

    save_webp(ROOT / "assets/ui.png", ASSETS / "ui.webp", 1600, 86)
    save_webp(ROOT / "assets/readme/method-comparison_poster.jpg",
              ASSETS / "method-comparison.webp", 1200)
    save_webp(ROOT / "assets/readme/retouch_poster.jpg", ASSETS / "retouch.webp", 1200)
    shutil.copyfile(ROOT / "assets/benchmark.png", ASSETS / "benchmark.png")
    made += [ASSETS / "ui.webp", ASSETS / "method-comparison.webp",
             ASSETS / "retouch.webp", ASSETS / "benchmark.png"]

    og = cover(Image.open(ROOT / "assets/ui.png").convert("RGB"), 1200, 630)
    og.save(ASSETS / "og.jpg", "JPEG", quality=86, optimize=True, progressive=True)
    made.append(ASSETS / "og.jpg")
    return made


def not_found(css: str) -> str:
    en, zh = LOCALES["en"], LOCALES["zh"]
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>404 — OpenFocus</title>
<meta name="robots" content="noindex">
<style>{css}</style>
</head>
<body>
<main class="wrap" style="padding:96px 0 120px;text-align:center">
<p class="eyebrow">404</p>
<h2>{esc(en['notfound']['title'])}</h2>
<p class="sub" style="margin-inline:auto">{esc(en['notfound']['body'])}</p>
<p class="lede" style="margin:0 auto 30px;color:var(--ink2)">{esc(zh['notfound']['body'])}</p>
<div class="cta" style="justify-content:center">
<a class="btn" href="/">{esc(en['notfound']['home'])}</a>
<a class="btn ghost" href="/zh/">{esc(zh['notfound']['home'])}</a>
</div>
</main>
</body>
</html>
"""


def main() -> int:
    shutil.rmtree(DIST, ignore_errors=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    template = (SITE / "template.html").read_text(encoding="utf-8")

    pages = []
    for code, locale in LOCALES.items():
        ctx = context_for(locale)
        page = render(template, ctx, code)
        target = DIST / locale["path"] / "index.html"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(page, encoding="utf-8", newline="\n")
        pages.append((target, len(page.encode("utf-8"))))

    assets = build_assets()

    css = (SITE / "style.css").read_text(encoding="utf-8")
    (DIST / "404.html").write_text(not_found(css), encoding="utf-8", newline="\n")

    for installer in ("install.sh", "install.ps1"):
        text = (SITE / installer).read_text(encoding="utf-8")
        (DIST / installer).write_text(text, encoding="utf-8", newline="\n")

    (DIST / "robots.txt").write_text(
        f"User-agent: *\nAllow: /\n\nSitemap: {ORIGIN}/sitemap.xml\n",
        encoding="utf-8", newline="\n",
    )
    urls = "\n".join(
        f"  <url><loc>{ORIGIN}/{locale['path']}</loc>"
        f"<lastmod>{date.today().isoformat()}</lastmod></url>"
        for locale in LOCALES.values()
    )
    (DIST / "sitemap.xml").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        f"{urls}\n</urlset>\n",
        encoding="utf-8", newline="\n",
    )
    (DIST / "_headers").write_text(
        "/*\n"
        "  X-Content-Type-Options: nosniff\n"
        "  Referrer-Policy: strict-origin-when-cross-origin\n"
        "  X-Frame-Options: DENY\n"
        "\n"
        "/\n"
        "  Cache-Control: public, max-age=0, must-revalidate\n"
        "  Vary: Accept-Language, Cookie\n"
        "\n"
        "/zh/\n"
        "  Cache-Control: public, max-age=0, must-revalidate\n"
        "\n"
        "/404.html\n"
        "  Cache-Control: public, max-age=0, must-revalidate\n"
        "\n"
        "/install.sh\n"
        "  Content-Type: text/plain; charset=utf-8\n"
        "  Cache-Control: public, max-age=300\n"
        "\n"
        "/install.ps1\n"
        "  Content-Type: text/plain; charset=utf-8\n"
        "  Cache-Control: public, max-age=300\n"
        "\n"
        "/assets/*\n"
        "  Cache-Control: public, max-age=86400\n",
        encoding="utf-8", newline="\n",
    )

    print(f"origin: {ORIGIN}")
    for target, size in pages:
        print(f"  {target.relative_to(ROOT)}  {size:,} bytes")
    print(f"  {len(assets)} assets, {sum(a.stat().st_size for a in assets):,} bytes")
    print(f"  dist: {DIST.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
