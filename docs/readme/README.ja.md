<p align="center">
  <a href="https://github.com/Xinzhe99/OpenFocus">
    <img src="../../assets/OpenFocus.png" alt="OpenFocus" width="128">
  </a>
</p>

<h1 align="center">OpenFocus</h1>

<p align="center"><strong>すべての焦点面を、1 枚の全焦点画像に。</strong></p>

<p align="center">
  10 種類の融合アルゴリズム、5 つの位置合わせオプション（位置合わせなしを含む）、マクロ・顕微鏡・産業用スタック向けのソース固定レタッチ。16 ビットを一貫して保持し、ヘッドレス CLI と 12 言語の UI を備えます。
</p>

<p align="center">
  <a href="https://github.com/Xinzhe99/OpenFocus/releases"><img src="https://img.shields.io/github/v/release/Xinzhe99/OpenFocus?label=release&color=2f81f7" alt="Release"></a>
  <a href="https://github.com/Xinzhe99/OpenFocus/stargazers"><img src="https://img.shields.io/github/stars/Xinzhe99/OpenFocus?color=e3b341" alt="Stars"></a>
  <a href="../../docs/THIRD_PARTY_NOTICES.md"><img src="https://img.shields.io/badge/license-MIT%20%2B%20model%20terms-3fb950" alt="License: MIT + model terms"></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Windows-10%2F11-0078D7?logo=windows&logoColor=white" alt="Windows 10/11">
  <img src="https://img.shields.io/badge/macOS-Apple%20Silicon-000000?logo=apple&logoColor=white" alt="macOS Apple Silicon">
  <img src="https://img.shields.io/badge/Linux-CLI-FCC624?logo=linux&logoColor=black" alt="Linux CLI">
  <a href="https://pypi.org/project/openfocus/"><img src="https://img.shields.io/badge/PyPI-openfocus-3775A9?logo=pypi&logoColor=white" alt="PyPI"></a>
</p>

<p align="center">
  <a href="#install">ダウンロード</a> ·
  <a href="../USER_MANUAL_JA.md">ドキュメント</a> ·
  <a href="../../CHANGELOG.md">変更履歴</a> ·
  <a href="https://doi.org/10.5281/zenodo.23050823">DOI</a>
</p>

<p align="center">
  <a href="../../README.md">English</a> ·
  <a href="README.zh-CN.md">简体中文</a> ·
  <strong>日本語</strong> ·
  <a href="README.es.md">Español</a> ·
  <a href="README.de.md">Deutsch</a> ·
  <a href="README.fr.md">Français</a> ·
  <a href="README.ru.md">Русский</a> ·
  <a href="README.ko.md">한국어</a> ·
  <a href="README.pt.md">Português</a> ·
  <a href="README.it.md">Italiano</a> ·
  <a href="README.nl.md">Nederlands</a> ·
  <a href="README.pl.md">Polski</a>
</p>

<p align="center">
  <img src="../../assets/ui.png" alt="OpenFocus の画面" width="880">
</p>

## 📢 お知らせ

> [!NOTE]
> **v1.75.0** — グループ融合、再利用可能な処理ワークフロー、透明 PNG スタック、位置合わせ診断、強化された深度エクスポート、刷新されたデスクトップワークフロー。対応入力と制限は変更履歴を参照してください。

以前のリリースは [変更履歴](../../CHANGELOG.md) にあります。

📸 OpenFocus で作成した画像を公開する場合は、次のような注記を添えていただけると幸いです。

> Created with OpenFocus — https://github.com/Xinzhe99/OpenFocus

必須ではありませんが、大変ありがたく思います。

<a id="features"></a>
## 機能

<table>
<tr>
<td width="46%" valign="middle">

### AI 融合：StackMFF V3

StackMFF V3 は AI で各フレームの鮮明な部分を組み合わせ、焦点の合った画像を生成します。

[StackMFF V3 →](https://github.com/Xinzhe99/StackMFF-V3) · [ベンチマーク →](../../tools/synthetic_benchmark.py)

</td>
<td width="54%">

<a href="../../assets/readme/ai.mp4"><picture><source srcset="../../assets/readme/ai.gif" type="image/gif"><img src="../../assets/readme/ai_poster.jpg" alt="同じスタックでの StackMFF V3 とガイドフィルタ" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### A/B ワイプ比較

分割線を動かして、ソースフレームと融合結果を直接比較できます。

[ユーザーマニュアル →](../USER_MANUAL_JA.md)

</td>
<td width="54%">

<a href="../../assets/readme/wipe.mp4"><picture><source srcset="../../assets/readme/wipe.gif" type="image/gif"><img src="../../assets/readme/wipe_poster.jpg" alt="ソースフレームと融合結果の A/B 比較" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### 手法比較

融合手法を並べて比較し、選んだ結果を出力に採用するか、設定をメインウィンドウに適用できます。

</td>
<td width="54%">

<a href="../../assets/readme/method-comparison.mp4"><picture><source srcset="../../assets/readme/method-comparison.gif" type="image/gif"><img src="../../assets/readme/method-comparison_poster.jpg" alt="8 つの融合手法の並列比較" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### 深度マップと 3D

深度マップを生成し、手動の Z 校正とテクスチャ付き 3D OBJ の書き出しに対応します。

</td>
<td width="54%">

<a href="../../assets/readme/depth.mp4"><picture><source srcset="../../assets/readme/depth.gif" type="image/gif"><img src="../../assets/readme/depth_poster.jpg" alt="同じスタックの深度マップと 3D 曲面" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### ソース固定レタッチ

結果の生成に使ったソースフレームからブラシでレタッチし、ソースを含むレイヤー付き TIFF を書き出せます。自動背景は、スタックで乱れた背景を実際の1フレームに置き換えます。

</td>
<td width="54%">

<a href="../../assets/readme/retouch.mp4"><picture><source srcset="../../assets/readme/retouch.gif" type="image/gif"><img src="../../assets/readme/retouch_poster.jpg" alt="ソースフレームと結果を並べたレタッチ作業領域" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### バッチ処理

複数のスタックを一括処理し、必要に応じてグループ融合でメモリ使用量を抑えられます。

</td>
<td width="54%">

<a href="../../assets/readme/batch.mp4"><picture><source srcset="../../assets/readme/batch.gif" type="image/gif"><img src="../../assets/readme/batch_poster.jpg" alt="バッチ処理ダイアログ" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### 出版向けスケールバー

TIFF メタデータまたは手入力で校正したスケールバーを画像に書き出せます。

[ユーザーマニュアル →](../USER_MANUAL_JA.md)

</td>
<td width="54%">

<a href="../../assets/readme/scalebar.mp4"><picture><source srcset="../../assets/readme/scalebar.gif" type="image/gif"><img src="../../assets/readme/scalebar_poster.jpg" alt="ライブプレビュー付きスケールバーダイアログ" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### ラベルとキャプション

ソース、位置合わせ済みスタック、出力に、フレーム範囲とスタイルを指定したラベルやキャプションを追加できます。

</td>
<td width="54%">

<a href="../../assets/readme/labels.mp4"><picture><source srcset="../../assets/readme/labels.gif" type="image/gif"><img src="../../assets/readme/labels_poster.jpg" alt="ラベルとキャプション" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Agent から操作可能

トークンで保護されたローカルチャネルを通じて、Agent がウィンドウ操作、画像処理、結果の書き出しを行えます。

[プロトコル →](../../docs/AGENT_CONTROL_SPEC.md) · [スキル →](../../skills/openfocus-control/SKILL.md)

</td>
<td width="54%">

<a href="../../assets/readme/agent.mp4"><picture><source srcset="../../assets/readme/agent.gif" type="image/gif"><img src="../../assets/readme/agent_poster.jpg" alt="Agent が実 GUI を操作：読み込み、AI レンダリング、比較、レタッチ、ワイプ、深度、ラベル" width="100%" /></picture></a>

</td>
</tr>
</table>

**そのほか：**

- **フォーカススタッキング作業フロー**——スタック閲覧、シャープネス曲線、クイックプレビュー、ROI レンダリング、手法ごとの設定を再起動後も保持。
- **顕微鏡向け**——16 ビットを一貫保持、TIFF µm/px、フラットフィールド補正、センサー欠陥補正、マルチページ TIFF Z スタック、テクスチャ付き OBJ 書き出し。
- **カラープロファイル**——単体の PNG、TIFF、JPEG、WebP 書き出しでは全ソース共通の RGB ICC プロファイルを保持し、混在または未設定の場合は付与も色変換もしません。
- **透明 PNG**——同じ矩形の一定不透明度マスクを持つスタックを原寸で融合・保存できます（制限はユーザーマニュアルを参照）。
- **ヘッドレス CLI**——GUI と同じパイプラインを `openfocus` コマンドとして提供。バッチ、`--dry-run`、`--resume`、JSON 出力に対応。
- **プロジェクトと復元**——`.ofproj` セッション、クラッシュ自動復元、スタック編集と注釈の undo/redo。
- **12 の UI 言語**——システムロケールから自動判定し、`locales/<code>.py` で言語を追加できます。
- **省メモリ**——ニューラル重みは AI 手法の使用時だけ読み込みます。

<a id="install"></a>
## ⬇️ インストール

最新の Windows または macOS ビルドは [OpenFocus 公式サイト](https://openfocus.top/#download) からダウンロードしてください。同じファイルと旧バージョンは [GitHub Releases](https://github.com/Xinzhe99/OpenFocus/releases) にもあります。

インストールとトラブルシューティングは[ユーザーマニュアル](../USER_MANUAL_JA.md)を参照してください。

ターミナルからもインストールできます。Linux にはヘッドレス CLI が入り、macOS は Apple Silicon が必要です。DMG は `/Applications` に、ターミナルからは `~/Applications` にインストールします。

```bash
curl -fsSL https://openfocus.top/install.sh | sh -s -- --dry-run
curl -fsSL https://openfocus.top/install.sh | sh
```

Windows では PowerShell からインストーラーを実行します。

```powershell
irm https://openfocus.top/install.ps1 | iex
```

Windows のセットアップを直接取得することもできます。

```powershell
curl.exe -fL https://openfocus.top/download/latest/windows-setup -o OpenFocus-setup.exe
```

### 🤖 Agent 連携

デスクトップ版はトークン保護されたローカル Agent チャネルを自動で開始します。`openfocus ctl` を使うと、実行中のウィンドウに型付きコマンドや汎用 UI 操作を送れます。Agent はスタックの読み込み、レンダリング、書き出し、ダイアログの表示と入力、ファイル選択への応答、マウス・キーボードイベントの送信、スクリーンショットの取得ができます。

- スキル：[`skills/openfocus-control/SKILL.md`](../../skills/openfocus-control/SKILL.md)
- プロトコルとコマンドリファレンス：[`docs/AGENT_CONTROL_SPEC.md`](../../docs/AGENT_CONTROL_SPEC.md)
- エンドツーエンド実行：`python tools/agent_gui_smoke.py --launch`
- 全メニュー操作のスイープ：`python tools/agent_action_sweep.py --launch`

### 💻 コマンドライン

Windows・macOS・Linux では CLI もインストールできます。同じパイプラインを GUI なしで動かせます。

```bash
pip install openfocus
openfocus -i ./stack_folder -o ./result/fused.png
```

詳しくは[CLI マニュアル](../USER_MANUAL_JA.md#14-コマンドラインインターフェース)を参照してください。

<a id="community"></a>
## コミュニティとサポート

質問やご意見は [Discord コミュニティ](https://discord.gg/afzmwgPt)へ。できるだけ早く返信しますが、北京時間（UTC+8）を使っているため、時差によって返信が遅れる場合があります。

- **ドキュメント** — [ユーザーマニュアル](../USER_MANUAL_JA.md)（12 言語） · [変更履歴](../../CHANGELOG.md) · [パフォーマンス](../../docs/PERFORMANCE.md) · [融合手法](../../docs/FUSION_METHODS.md)
- **公式サイト** — [openfocus.top](https://openfocus.top/)（ダウンロード、インストールスクリプト、ミラー）
- **引用** — [CITATION.cff](../../CITATION.cff)。各リリースは Zenodo に保存されています（[DOI](https://doi.org/10.5281/zenodo.23050823)）
- **フィードバック** — 不具合、提案、アルゴリズムの改善は [issue](https://github.com/Xinzhe99/OpenFocus/issues) へ
- **クレジット** — 各融合・位置合わせアルゴリズムは元の研究ライセンスに従います。StackMFF V3 は [Xinzhe99/StackMFF-V3](https://github.com/Xinzhe99/StackMFF-V3) にあります

<a id="license"></a>
## ライセンス

OpenFocus アプリケーションは [MIT ライセンス](../../LICENSE) で公開されています。

**StackMFF のライセンス。** 新しい V1/V2 の重みとラッパーには StackMFF V3 License 1.0 が適用されます。複製、配布、他のプロジェクトへの組み込み、商用利用には権利者の事前の書面による許可が必要です。従来同梱されている StackMFF V3 の v2.0 スナップショットは引き続き MIT ライセンスです。詳細は[サードパーティーに関する通知](../../docs/THIRD_PARTY_NOTICES.md)を参照してください。

<a id="developing"></a>
## 開発

チェックアウトから実行：

```bash
conda create -n openfocus python=3.10
conda activate openfocus
pip install -r requirements.txt
python main.py
```

PyInstaller でのパッケージングは [docs/BUILD_COMMANDS.md](../../docs/BUILD_COMMANDS.md) を参照してください。

<a id="star-history"></a>
## ⭐ Star 履歴

[![Star History Chart](https://api.star-history.com/svg?repos=Xinzhe99/OpenFocus&type=Date)](https://star-history.com/#Xinzhe99/OpenFocus&Date)
