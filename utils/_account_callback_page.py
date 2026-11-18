"""Self-contained receipt for the desktop sign-in listener."""

from base64 import b64encode
from pathlib import Path


_COPY = {
    "en": {
        "received": (
            "Return to OpenFocus",
            "Your sign-in request reached the desktop app. Your sign-in status will appear there.",
        ),
        "not_signed_in": (
            "No browser session",
            "This browser has no active OpenFocus session. Return to the desktop app and choose another sign-in method.",
        ),
        "error": (
            "Sign-in interrupted",
            "Return to the OpenFocus desktop app for details and try again.",
        ),
        "close": "You can close this tab.",
    },
    "zh": {
        "received": (
            "返回 OpenFocus 应用",
            "登录请求已送达桌面应用。登录结果会在应用中显示。",
        ),
        "not_signed_in": (
            "浏览器尚未登录",
            "当前浏览器没有可用的 OpenFocus 账号。请返回桌面应用，选择其他登录方式。",
        ),
        "error": (
            "登录未能继续",
            "请返回 OpenFocus 桌面应用，查看提示后重试。",
        ),
        "close": "现在可以关闭此标签页。",
    },
}

_STYLE = """
:root {
    color-scheme: light dark;
    --canvas: #f4f6fb;
    --surface: #fff;
    --text: #192337;
    --muted: #4b586d;
    --border: #dde4ef;
    --glow: #e2ebff;
    --status: #2259bf;
    --status-bg: #e8f0ff;
    --warning: #8c4905;
    --warning-bg: #fff0da;
    --shadow: 0 20px 60px -24px #243d6b40;
}
@media (prefers-color-scheme: dark) {
    :root {
        --canvas: #101522;
        --surface: #192132;
        --text: #f2f5fc;
        --muted: #b8c4d6;
        --border: #33415a;
        --glow: #202f4d;
        --status: #b2cbff;
        --status-bg: #273e69;
        --warning: #ffd69e;
        --warning-bg: #49341e;
        --shadow: 0 20px 60px -24px #0009;
    }
}
* { box-sizing: border-box; }
body {
    min-height: 100vh;
    margin: 0;
    padding: clamp(1rem, 5vw, 3rem);
    display: grid;
    place-items: center;
    background: radial-gradient(ellipse at top, var(--glow), transparent 64%), var(--canvas);
    color: var(--text);
    font: 1rem/1.6 system-ui, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
}
main {
    width: min(100%, 33rem);
    padding: clamp(1.5rem, 5vw, 2.5rem);
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 1.5rem;
    box-shadow: var(--shadow);
}
.brand {
    display: flex;
    align-items: center;
    gap: .625rem;
    margin-bottom: 1.75rem;
    font-weight: 650;
}
.status {
    display: grid;
    place-items: center;
    width: 3rem;
    height: 3rem;
    border-radius: .875rem;
    background: var(--status-bg);
    color: var(--status);
    font-size: 1.5rem;
    font-weight: 700;
    line-height: 1;
}
.status.warning {
    background: var(--warning-bg);
    color: var(--warning);
}
h1 {
    margin: .875rem 0 .5rem;
    font-size: clamp(1.5rem, 5vw, 2rem);
    line-height: 1.25;
    letter-spacing: -.02em;
}
p {
    margin: 0;
    color: var(--muted);
}
footer {
    margin-top: 1.5rem;
    padding-top: 1rem;
    border-top: 1px solid var(--border);
    font-size: .875rem;
}
"""


def render(outcome, language):
    # Only fixed copy is rendered. Callback parameters stay in the listener.
    lang = "zh" if language == "zh" else "en"
    copy = _COPY[lang]
    title, message = copy[outcome]
    status_class, symbol = ("status", "✓") if outcome == "received" else ("status warning", "!")
    logo = ""
    try:
        icon = (Path(__file__).parent.parent / "assets" / "OpenFocus.ico").read_bytes()
    except OSError:
        # A missing decorative asset should never interrupt the sign-in handoff.
        pass
    else:
        logo = f'<img src="data:image/x-icon;base64,{b64encode(icon).decode("ascii")}" width="36" height="36" alt="">'
    return f"""<!doctype html>
<html lang="{lang}">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{title}</title>
    <style>{_STYLE}</style>
</head>
<body>
    <main aria-labelledby="heading">
        <div class="brand">{logo}<span>OpenFocus</span></div>
        <div class="{status_class}" aria-hidden="true">{symbol}</div>
        <h1 id="heading">{title}</h1>
        <p>{message}</p>
        <footer><p>{copy['close']}</p></footer>
    </main>
</body>
</html>
""".encode("utf-8")
