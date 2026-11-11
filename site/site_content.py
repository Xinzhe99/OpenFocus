"""Copy and release metadata for the OpenFocus website.

The build script (tools/build_site.py) renders one page per entry in
LOCALES.  Adding a language means adding one dict here and a
locales/<code>.py entry is *not* needed: the site is independent of the
application's UI language packs.

Placeholders are resolved by dotted key, so every ``{{key}}`` in
template.html must exist under the matching locale.
"""

import os


REPO = "https://github.com/Xinzhe99/OpenFocus"
RELEASES = f"{REPO}/releases"
MANUAL = {
    "en": f"{REPO}/blob/main/docs/USER_MANUAL_EN.md",
    "zh": f"{REPO}/blob/main/docs/USER_MANUAL_ZH.md",
}

# Download endpoints resolve the newest complete release even without JavaScript.
ORIGIN = os.environ.get("SITE_ORIGIN", "").strip().rstrip("/") or "https://openfocus.top"
RELEASE = {
    "page": f"{RELEASES}/latest",
    "windows_setup": f"{ORIGIN}/download/latest/windows-setup",
    "windows_zip": f"{ORIGIN}/download/latest/windows-zip",
    "macos_dmg": f"{ORIGIN}/download/latest/macos-dmg",
    "macos_zip": f"{ORIGIN}/download/latest/macos-zip",
    "pypi": "https://pypi.org/project/openfocus/",
}


def _en() -> dict:
    return {
        "code": "en",
        "html_lang": "en",
        "og_locale": "en_US",
        "path": "",
        "switch_label": "English",
        "skip": "Skip to content",
        "title": "OpenFocus — focus stacking, pixel by pixel",
        "desc": (
            "OpenFocus aligns a focus stack and fuses it into one all-in-focus "
            "image, choosing each pixel from its sharpest source frame: ten "
            "fusion algorithms including neural StackMFF V3, 16-bit end to "
            "end, depth maps, retouch and a headless CLI."
        ),
        "nav": {
            "features": "Features",
            "workflow": "Workflow",
            "benchmark": "Benchmark",
            "cli": "CLI",
            "docs": "Docs",
            "github": "GitHub",
            "account": "Sign in",
            "account_url": "https://account.openfocus.top/account?lang=en",
            "download": "Download",
            "theme": "Theme",
            "language": "Language",
            "menu": "Menu",
        },
        "hero": {
            "title": 'Every pixel,<br>its sharpest frame.',
            "lede": (
                "A focus stack is a pile of near-misses. OpenFocus aligns the "
                "frames, decides which one is sharpest at every pixel, and "
                "blends them into a single image — for macro, microscopy and "
                "industrial inspection."
            ),
            "chips": [
                "10 fusion methods",
                "5 alignment modes",
                "16-bit end to end",
                "12 UI languages",
                "MIT application; StackMFF V1/V2 models have separate terms",
            ],
            "cta_windows": "Download for Windows",
            "cta_macos": "Download for macOS",
            "cta_github": "View on GitHub",
            "desktop_label": "Desktop downloads (may be beta):",
            "meta": "Windows 10/11 · macOS Apple Silicon · Linux CLI on PyPI",
            "shot_alt": (
                "The OpenFocus window: a nine-frame stack, a wipe comparison "
                "and a 50 mm scale bar on the fused result."
            ),
        },
        "why": {
            "eyebrow": "Why OpenFocus",
            "title": "Focus stacking, without the guesswork.",
            "sub": (
                "Ten fusion algorithms, a registration step that fits your "
                "rig, and a workspace that shows you the result at every zoom."
            ),
            "features": [
                {
                    "title": "AI fusion and classical methods",
                    "body": (
                        "StackMFF V3 is OpenFocus's current state-of-the-art AI "
                        "fusion model, estimating the best source frame for each "
                        "pixel. V2 and V1 join Guided Filter, DCT, DTCWT, GFG-FGF "
                        "and OpenFocus A/B/C."
                    ),
                },
                {
                    "title": "Alignment that matches the rig",
                    "body": (
                        "No alignment, SIFT homography, ECC, homography refined "
                        "by ECC, or a bounded similarity search for translation, "
                        "rotation and scale."
                    ),
                },
                {
                    "title": "16-bit, end to end",
                    "body": (
                        "Load, fuse and export without dropping to 8-bit. µm/px "
                        "can come from TIFF metadata, and multi-page TIFF Z-stacks "
                        "stay intact."
                    ),
                },
                {
                    "title": "Compare, then commit",
                    "body": (
                        "Render every method once and inspect them at a shared "
                        "crop. Adopt the winner, or copy its parameters back to "
                        "the main window."
                    ),
                },
                {
                    "title": "Retouch from the source",
                    "body": (
                        "Paint detail back from the frame that actually had it: "
                        "shared zoom and pan, soft and restore brushes, "
                        "whole-stroke undo."
                    ),
                },
                {
                    "title": "Headless when you need it",
                    "body": (
                        "The same pipeline as <code>openfocus</code>, with batch "
                        "mode, <code>--dry-run</code>, <code>--resume</code> and "
                        "JSON output for scripting."
                    ),
                },
            ],
        },
        "workflow": {
            "eyebrow": "Workflow",
            "title": "Load. Align. Fuse. Retouch.",
            "sub": (
                "The stack browser and sharpness curve tell you which frame "
                "owns which detail before you render."
            ),
            "bullets": [
                {
                    "b": "Wipe compare",
                    "t": "Drag the divider between a source frame and the result at any zoom.",
                },
                {
                    "b": "Method comparison",
                    "t": "Fuse the full image once per method, then compare the same detail side by side.",
                },
                {
                    "b": "Depth map & 3D",
                    "t": "Seven depth-capable methods, colour maps, manual physical Z calibration and textured OBJ export.",
                },
                {
                    "b": "Scale bars and labels",
                    "t": "Calibrate scale bars from metadata or enter a value, and add labels to results.",
                },
            ],
            "img_alt": "Side-by-side comparison of the fusion methods at the same crop.",
        },
        "detail": {
            "eyebrow": "Retouch",
            "title": "Fix the one percent the blend got wrong.",
            "sub": (
                "The retouch workspace is locked to the frames that produced the "
                "result, so every stroke is traceable. Depth maps come from the "
                "same stack."
            ),
            "bullets": [
                {
                    "b": "Source-locked brush",
                    "t": "Pick the frame, paint, and the result updates in place.",
                },
                {
                    "b": "Restore and undo",
                    "t": "Soft brush, restore brush, whole-stroke undo, and a guard for pixels that will not be exported.",
                },
                {
                    "b": "Depth and surface",
                    "t": "Seven depth-capable methods, research colour maps, and a textured OBJ of the surface.",
                },
            ],
            "img_alt": "The retouch workspace with a source frame beside the fused result.",
        },
        "bench": {
            "eyebrow": "Benchmark",
            "title": "Measured against ground truth.",
            "sub": (
                "A 16-frame synthetic stack with noise σ = 3, rendered through a "
                "known depth map. Higher PSNR/SSIM and a lower ghost ratio are "
                "better."
            ),
            "img_alt": (
                "Benchmark chart: StackMFF V3 leads the fusion methods with "
                "37.13 dB PSNR, 0.9910 SSIM and 0.97% ghosting."
            ),
            "link": "Reproduce with tools/synthetic_benchmark.py",
        },
        "cli": {
            "eyebrow": "Headless",
            "title": "The same pipeline, without the window.",
            "sub": (
                "Install on Windows, macOS or Linux. The model, its weights, "
                "HEIC and camera-RAW decoding are all in one wheel."
            ),
            "install": "pip install openfocus",
            "copy": "Copy",
            "copied": "Copied",
            "lines": [
                "openfocus -i ./stack -o fused.png",
                "openfocus -i ./stackA ./stackB -d results -m stackmffv3",
                "openfocus -i ./stack -m dtcwt -a ecc --threads 8",
            ],
            "note": (
                "<code>--dry-run</code>, <code>--resume</code>, "
                "<code>--json</code> and every fusion and registration "
                "parameter are available."
            ),
            "link": "Command reference",
        },
        "download": {
            "eyebrow": "Download",
            "title": "Install OpenFocus",
            "sub": (
                "Windows code signing, when available, is provided free by "
                "SignPath.io with a SignPath Foundation certificate. Check each "
                "release's notes for its signing status. If SmartScreen appears, "
                "choose More info → Run anyway; "
                "the first start can take 30–60 seconds. On macOS, run xattr -cr, "
                "or allow the app in System Settings → Privacy & Security → "
                "Open Anyway."
            ),
            "cards": [
                {
                    "os": "Windows 10/11 · x64",
                    "kind": "Installer",
                    "kind_id": "windows-setup",
                    "url": RELEASE["windows_setup"],
                    "primary": True,
                },
                {
                    "os": "Windows 10/11 · x64",
                    "kind": "Portable ZIP",
                    "kind_id": "windows-zip",
                    "url": RELEASE["windows_zip"],
                },
                {
                    "os": "macOS Apple Silicon",
                    "kind": "Disk image",
                    "kind_id": "macos-dmg",
                    "url": RELEASE["macos_dmg"],
                    "primary": True,
                },
                {
                    "os": "macOS Apple Silicon",
                    "kind": "Portable ZIP",
                    "kind_id": "macos-zip",
                    "url": RELEASE["macos_zip"],
                },
                {
                    "os": "Linux · any OS",
                    "kind": "pip install openfocus",
                    "size": "PyPI",
                    "url": RELEASE["pypi"],
                },
            ],
            "action": "Download",
            "script_label": "Terminal install · macOS / Linux",
            "ps_label": "PowerShell · Windows",
            "copy": "Copy",
            "copied": "Copied",
            "all": "All releases and checksums",
            "channel_notice": "Windows and macOS downloads and updates may be beta releases. The Linux CLI on PyPI is a stable release.",
            "note": "Desktop downloads · MIT application; StackMFF V1/V2 models have separate terms",
        },
        "privacy": {
            "title": "Privacy",
            "accounts": "The optional account service stores your email address, display name, linked sign-in provider identifiers, sessions and device labels, consent choices and the service records needed to operate your account. These records are separate from desktop usage statistics and are not covered by the 90-day statistics retention period.",
            "reports": "Optional diagnostic reports can include software and device details, image counts and dimensions, processing settings, interface actions and sanitized error information. They exclude image pixels, image and project files, file names, paths and account credentials. You can review manual reports before sending them; after desktop sign-in, automatic error reporting is enabled by default unless you previously disabled it. You can change it in Advanced Settings. Reports may be sent anonymously or linked to your account.",
            "controls": "You can revoke sessions or delete your account from the account page. Deletion removes your contact profile, linked sign-in identities and sessions; some account, consent and administrative records may remain. Product email is optional: choose whether to opt in during account setup, and use the unsubscribe link in those emails to opt out. Sign-in links and service notices are separate.",
            "manage": "Manage your account",
            "cookies": "The website remembers your language in a cookie and your theme in local storage. The account area also uses cookies for sign-in sessions, request protection, pending sign-in and language preferences.",
            "providers": "Cloudflare hosts the website and account service. If you choose a third-party sign-in provider, that provider processes your sign-in. Resend delivers account and notification emails. These services may process request or delivery information separately. For personal-data requests, contact support@openfocus.top.",
            "downloads": "When download statistics are enabled, the website records aggregate download starts by date, version, asset type, route and source. Retries, scanners and incomplete downloads can be counted; these are not unique users or completed downloads.",
            "usage": "Desktop usage reporting is enabled by default when no preference exists; a previous explicit opt-out is preserved. Usage reports send only a random installation ID, application version and operating system to estimate active installations. They do not include images, paths, hardware details or logs. Turn reporting off in Settings → Advanced Settings → Share usage statistics, then choose OK, to stop future reports and delete the local ID. Cancel discards changes. A request already in flight may still complete.",
            "retention": "Usage records are kept for up to 90 UTC calendar days; operational failures can delay deletion. Download totals remain until an operator deletes them. The statistics database does not store IP addresses, user agents or hardware identifiers. Cloudflare may keep separate request logs, including network information.",
        },
        "footer": {
            "tagline": "Every pixel, its sharpest frame.",
            "links": [
                ("GitHub", REPO),
                ("Docs", MANUAL["en"]),
                ("Changelog", f"{REPO}/blob/main/CHANGELOG.md"),
                ("Code signing policy", f"{REPO}/blob/main/docs/CODE_SIGNING_POLICY.md"),
                ("Privacy", "#privacy"),
                ("Terms of Service", "https://account.openfocus.top/terms?lang=en"),
                ("Cite", "https://doi.org/10.5281/zenodo.23050823"),
            ],
            "license": "© 2026 OpenFocus Contributors · MIT application; StackMFF V1/V2 models have separate terms",
        },
        "notfound": {
            "title": "That page is out of focus.",
            "body": "The link may have moved. The home page is a good place to refocus.",
            "home": "Back to OpenFocus",
        },
    }


def _zh() -> dict:
    return {
        "code": "zh",
        "html_lang": "zh-CN",
        "og_locale": "zh_CN",
        "path": "zh/",
        "switch_label": "简体中文",
        "skip": "跳到主要内容",
        "title": "OpenFocus — 像素级焦点堆栈融合",
        "desc": (
            "OpenFocus 先对齐焦点堆栈，再将多张不同焦点的照片融合为一张"
            "全清晰图像：支持十种融合算法（包括 AI 模型 StackMFF V3）、"
            "16 位处理、深度图、源帧修饰和命令行批处理。"
        ),
        "nav": {
            "features": "功能",
            "workflow": "工作流",
            "benchmark": "基准",
            "cli": "命令行",
            "docs": "文档",
            "github": "GitHub",
            "account": "登录",
            "account_url": "https://account.openfocus.top/account?lang=zh",
            "download": "下载",
            "theme": "主题",
            "language": "语言",
            "menu": "菜单",
        },
        "hero": {
            "title": "每一像素，<br>皆取自最清晰的一帧。",
            "lede": (
                "焦点堆栈，是一叠“此处清晰、彼处模糊”的照片。OpenFocus 先对齐"
                "每一帧，再逐像素判断何处最清晰，最终融成一张全清晰图像——微距、"
                "显微与工业检测皆可适用。"
            ),
            "chips": [
                "十种融合算法",
                "五种配准模式",
                "全程 16-bit",
                "十二种界面语言",
                "MIT 应用；StackMFF V1/V2 模型另有许可",
            ],
            "cta_windows": "下载 Windows 版",
            "cta_macos": "下载 macOS 版",
            "cta_github": "查看源码",
            "desktop_label": "桌面版下载（可能为测试版）：",
            "meta": "Windows 10/11 · macOS Apple Silicon · Linux 命令行请见 PyPI",
            "shot_alt": "OpenFocus 主窗口：九帧源图、拖拽对比，以及融合结果上的 50 mm 比例尺。",
        },
        "why": {
            "eyebrow": "为何选择 OpenFocus",
            "title": "焦点堆叠，不必靠猜。",
            "sub": "十种融合算法、贴合不同拍摄方式的配准，以及可在任意缩放下逐像素检视的工作区。",
            "features": [
                {
                    "title": "AI 融合与经典算法",
                    "body": (
                        "StackMFF V3 是 OpenFocus 目前最先进的 AI 融合模型，"
                        "逐像素判断最合适的源帧；也可选择 V2、V1、引导滤波、"
                        "DCT、DTCWT、GFG-FGF 和 OpenFocus A/B/C。"
                    ),
                },
                {
                    "title": "配准，随拍摄方式而定",
                    "body": (
                        "不对齐、SIFT 单应性、ECC、单应性→ECC 精修，或用于平移、旋转"
                        "和缩放的受限相似变换搜索——总有一种适合你的拍摄方式。"
                    ),
                },
                {
                    "title": "全程 16-bit",
                    "body": (
                        "从载入、融合到导出，全程不降至 8-bit；可从 TIFF 元数据 "
                        "读取 µm/px，多页 TIFF Z 堆栈保持原样。"
                    ),
                },
                {
                    "title": "先比较，再定稿",
                    "body": (
                        "一次渲染全部算法，于同一裁切处并排审视；择其优者定稿，"
                        "或将其参数回填至主窗口。"
                    ),
                },
                {
                    "title": "从源帧修饰结果",
                    "body": (
                        "从清晰的源帧取回细节：同步缩放和平移，"
                        "使用柔边画笔或恢复画笔，并可逐笔撤销。"
                    ),
                },
                {
                    "title": "需要时，可完全无界面运行",
                    "body": (
                        "与 <code>openfocus</code> 同一套流程，支持批处理、"
                        "<code>--dry-run</code>、<code>--resume</code> 与 JSON "
                        "输出，便于脚本化。"
                    ),
                },
            ],
        },
        "workflow": {
            "eyebrow": "工作流程",
            "title": "载入、对齐、融合、修饰。",
            "sub": "借助源图列表和清晰度曲线，在渲染前了解各帧的清晰程度。",
            "bullets": [
                {
                    "b": "拖拽对比",
                    "t": "在任意缩放下拖动分隔线，比较源帧与融合结果。",
                },
                {
                    "b": "算法对比",
                    "t": "每种算法默认融合整图，再并排查看相同的局部细节。",
                },
                {
                    "b": "深度图与 3D",
                    "t": "七种方法可估计焦点位置，支持科研级配色、手动标定物理 Z 及纹理 OBJ 导出。",
                },
                {
                    "b": "比例尺与标注",
                    "t": "从元数据标定比例尺或手动输入尺寸，并为结果添加标注。",
                },
            ],
            "img_alt": "同一裁切处并排比较各融合算法的结果。",
        },
        "detail": {
            "eyebrow": "修饰",
            "title": "融合后的细节，由你完善。",
            "sub": "修饰工作区只使用实际参与融合的源帧，每一笔都可追溯，方便找回细节、修复光晕；深度图也同样来自这组源帧。",
            "bullets": [
                {
                    "b": "源帧画笔",
                    "t": "选择一张源帧，在结果上涂抹即可取回对应细节。",
                },
                {
                    "b": "恢复与撤销",
                    "t": "支持柔边画笔、恢复画笔和逐笔撤销；源帧没有有效像素的边缘会自动跳过。",
                },
                {
                    "b": "深度与表面",
                    "t": "七种方法可估计焦点位置，支持科研级配色和带纹理的表面 OBJ。",
                },
            ],
            "img_alt": "修饰工作区：源帧与融合结果并排显示。",
        },
        "bench": {
            "eyebrow": "基准测试",
            "title": "用已知真值评估效果。",
            "sub": "基准使用经已知深度图渲染的 16 帧合成图像，噪声 σ = 3；PSNR、SSIM 越高，重影比例越低，效果越好。",
            "img_alt": "基准图：StackMFF V3 以 37.13 dB PSNR、0.9910 SSIM 和 0.97% 的重影比例领先各融合算法。",
            "link": "以 tools/synthetic_benchmark.py 复现",
        },
        "cli": {
            "eyebrow": "命令行",
            "title": "同一套流程，无需窗口。",
            "sub": "Windows、macOS、Linux 皆可安装；模型、权重、HEIC 与相机 RAW 解码，皆在同一安装包内。",
            "install": "pip install openfocus",
            "copy": "复制",
            "copied": "已复制",
            "lines": [
                "openfocus -i ./stack -o fused.png",
                "openfocus -i ./stackA ./stackB -d results -m stackmffv3",
                "openfocus -i ./stack -m dtcwt -a ecc --threads 8",
            ],
            "note": (
                "支持 <code>--dry-run</code>、<code>--resume</code>、"
                "<code>--json</code>，以及全部融合与配准参数。"
            ),
            "link": "命令参考",
        },
        "download": {
            "eyebrow": "下载",
            "title": "安装 OpenFocus",
            "sub": "Windows 代码签名启用后，由 SignPath.io 免费提供签名服务、SignPath Foundation 签发证书；请查看各版本说明中的签名状态。若被 SmartScreen 拦截，请点“更多信息 → 仍要运行”；首次启动可能需要 30–60 秒。macOS 请执行 xattr -cr，或在“系统设置 → 隐私与安全性”中点“仍要打开”。",
            "cards": [
                {
                    "os": "Windows 10/11 · x64",
                    "kind": "安装程序",
                    "kind_id": "windows-setup",
                    "url": RELEASE["windows_setup"],
                    "primary": True,
                },
                {
                    "os": "Windows 10/11 · x64",
                    "kind": "便携 ZIP",
                    "kind_id": "windows-zip",
                    "url": RELEASE["windows_zip"],
                },
                {
                    "os": "macOS Apple Silicon",
                    "kind": "磁盘映像",
                    "kind_id": "macos-dmg",
                    "url": RELEASE["macos_dmg"],
                    "primary": True,
                },
                {
                    "os": "macOS Apple Silicon",
                    "kind": "便携 ZIP",
                    "kind_id": "macos-zip",
                    "url": RELEASE["macos_zip"],
                },
                {
                    "os": "Linux · 任意系统",
                    "kind": "pip install openfocus",
                    "size": "PyPI",
                    "url": RELEASE["pypi"],
                },
            ],
            "action": "下载",
            "script_label": "终端安装 · macOS / Linux",
            "ps_label": "PowerShell · Windows",
            "copy": "复制",
            "copied": "已复制",
            "all": "所有版本与校验值",
            "channel_notice": "Windows 和 macOS 的下载及更新可能是测试版；PyPI 上的 Linux 命令行版为稳定版。",
            "note": "桌面版下载 · MIT 应用；StackMFF V1/V2 模型另有许可",
        },
        "privacy": {
            "title": "隐私",
            "accounts": "可选账号服务为提供账号功能，会保存你的邮箱、显示名称、关联登录平台的用户标识、会话与设备标签、同意选择，以及运营账号所需的服务记录。这些记录与桌面使用统计分开保存，不适用统计数据的 90 天保留期限。",
            "reports": "可选错误报告可能包含软件和设备信息、图像数量与尺寸、处理设置、界面操作及脱敏后的错误信息，不包含图像像素、图像或项目文件、文件名、路径或账号凭据。手动发送前可查看报告；登录桌面应用后，默认启用自动错误报告，此前明确关闭的设置会保留。你可以在高级设置中更改此选项。报告可以匿名发送，也可以关联账号。",
            "controls": "你可以在账号页面撤销会话或删除账号。删除会清除联系资料、关联登录身份和会话；部分账号、同意和管理记录可能保留。产品邮件为可选项：可在账号设置流程中自主选择接收，并通过邮件中的退订链接取消订阅。登录链接和服务通知另行发送。",
            "manage": "管理你的账号",
            "cookies": "网站使用 Cookie 记住语言偏好，使用本地存储记住主题。账号区还使用 Cookie 维持登录会话、提供请求保护、跟踪待完成的登录和记住语言偏好。",
            "providers": "Cloudflare 托管网站及账号服务。若选择第三方登录平台，该平台会处理你的登录。Resend 负责发送账号及通知邮件。这些服务可能另行处理请求或投递信息。如需提出个人信息相关请求，请联系 support@openfocus.top。",
            "downloads": "启用下载统计后，网站按日期、版本、文件类型、下载路径和来源汇总下载开始次数。重试、扫描或未完成的下载也可能计入；这个数字不代表独立用户或完成下载的次数。",
            "usage": "未保存偏好时，桌面使用统计默认开启；此前明确关闭的偏好会保留。使用统计报告仅发送随机安装标识、应用版本和操作系统，用于估计活跃安装量，不包含图像、路径、硬件信息或日志。可在“设置 → 高级设置 → 分享使用统计”关闭并确定，停止后续报告并删除本地标识；取消则放弃修改。已经发出的请求仍可能完成。",
            "retention": "使用记录最多保留 90 个 UTC 日历日；系统故障可能延迟删除。下载汇总保留到管理员删除为止。统计数据库不保存 IP 地址、用户代理或硬件标识。Cloudflare 可能另行保存包含网络信息的请求日志。",
        },
        "footer": {
            "tagline": "每一像素，皆取自最清晰的一帧。",
            "links": [
                ("GitHub", REPO),
                ("文档", MANUAL["zh"]),
                ("更新日志", f"{REPO}/blob/main/CHANGELOG.md"),
                ("代码签名政策", f"{REPO}/blob/main/docs/CODE_SIGNING_POLICY.md"),
                ("隐私", "#privacy"),
                ("服务条款", "https://account.openfocus.top/terms?lang=zh"),
                ("引用", "https://doi.org/10.5281/zenodo.23050823"),
            ],
            "license": "© 2026 OpenFocus Contributors · MIT 应用；StackMFF V1/V2 模型另有许可",
        },
        "notfound": {
            "title": "这个页面失焦了。",
            "body": "页面可能已移动，请返回首页。",
            "home": "返回 OpenFocus 首页",
        },
    }


LOCALES = {"en": _en(), "zh": _zh()}

