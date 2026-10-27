<p align="center">
  <a href="https://github.com/Xinzhe99/OpenFocus">
    <img src="assets/OpenFocus.png" alt="OpenFocus" width="128">
  </a>
</p>

<h1 align="center">OpenFocus</h1>

<p align="center"><strong>すべての焦点面を、1 枚の全焦点画像に。</strong></p>

<p align="center">
  8 つの合成アルゴリズム、3 つの位置合わせモード、マクロ・顕微鏡・産業用スタック向けのソース固定レタッチ。16 ビットを一貫して保持し、ヘッドレス CLI と 12 言語の UI を備えます。
</p>

<p align="center">
  <a href="https://github.com/Xinzhe99/OpenFocus/releases"><img src="https://img.shields.io/github/v/release/Xinzhe99/OpenFocus?label=release&color=2f81f7" alt="Release"></a>
  <a href="https://github.com/Xinzhe99/OpenFocus/stargazers"><img src="https://img.shields.io/github/stars/Xinzhe99/OpenFocus?color=e3b341" alt="Stars"></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/license-MIT-3fb950" alt="License: MIT"></a>
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
  <a href="./docs/USER_MANUAL_JA.md">ドキュメント</a> ·
  <a href="./CHANGELOG.md">変更履歴</a> ·
  <a href="https://doi.org/10.5281/zenodo.23050823">DOI</a>
</p>

<p align="center">
  <a href="README.md">English</a> ·
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
  <img src="assets/ui.png" alt="OpenFocus の画面" width="880">
</p>

## 📢 News

> [!NOTE]
> **v1.71.0** — アルゴリズムは不変のレジストリで宣言されるようになり、新しい融合・位置合わせ手法は型付きパラメータを持つ 1 つの記述子だけでレンダー、バッチ、比較、深度、CLI に現れます。起動ごとに独立したログをローテーション保存し、タスク ID でパラメータ、所要時間、完全なスタックトレースを関連付けます。フィードバック ZIP にはクラッシュレポート、RAM/GPU 情報、依存バージョンが入ります。診断は Qt と数値ライブラリのインポート前に開始するため、起動時の失敗も記録されます。8 つのアルゴリズムは v1.70.0 とバイト単位で同一です。

<a id="features"></a>
## 機能

<table>

<tr>
<td width="46%" valign="middle">

### AI 合成：StackMFF V3

既定の手法は、各画素を最も確からしいソースフレームにニューラルネットで分類します。同じ位置合わせ結果で、文字の輪郭はガイド付きフィルタより鋭く、面はより滑らか（ラプラシアン鮮鋭度 27.1 対 28.5）。真値のある合成スタックでは PSNR と SSIM の両方で全手法を上回り、ゴースト画素も古典手法より少なくなります。

[StackMFF V3 →](https://github.com/Xinzhe99/StackMFF-V3) · [ベンチマーク →](tools/synthetic_benchmark.py)

</td>
<td width="54%">

<a href="assets/readme/ai.mp4"><picture><source srcset="assets/readme/ai.gif" type="image/gif"><img src="assets/readme/ai_poster.jpg" alt="AI 合成：StackMFF V3" width="100%" /></picture></a>

</td>
</tr>

<tr>
<td width="46%" valign="middle">

### A/B ワイプ比較

分割線をソースフレームと合成結果の上で動かし、ブレンドが何を変えたかを確認。位置ずれ、縁のハロー、失われた細部が一目で分かります。

[ユーザーマニュアル →](./docs/USER_MANUAL_JA.md)

</td>
<td width="54%">

<a href="assets/readme/wipe.mp4"><picture><source srcset="assets/readme/wipe.gif" type="image/gif"><img src="assets/readme/wipe_poster.jpg" alt="ソースフレームと合成結果の A/B 比較" width="100%" /></picture></a>

</td>
</tr>

<tr>
<td width="46%" valign="middle">

### 手法比較

すべての手法を一度にレンダリングし、同じクロップで並べて確認。勝った結果を新しい出力として採用するか、パラメータをメインウィンドウに戻せます。

</td>
<td width="54%">

<a href="assets/readme/method-comparison.mp4"><picture><source srcset="assets/readme/method-comparison.gif" type="image/gif"><img src="assets/readme/method-comparison_poster.jpg" alt="8 つの合成手法の並列比較" width="100%" /></picture></a>

</td>
</tr>

<tr>
<td width="46%" valign="middle">

### 深度マップと 3D

5 つのフォーカス尺度で画素ごとの焦点位置を推定。研究用カラーマップ、TIFF メタデータからの物理 Z 校正、テクスチャ付き OBJ 曲面の書き出しに対応。

</td>
<td width="54%">

<a href="assets/readme/depth.mp4"><picture><source srcset="assets/readme/depth.gif" type="image/gif"><img src="assets/readme/depth_poster.jpg" alt="同じスタックの深度マップと 3D 曲面" width="100%" /></picture></a>

</td>
</tr>

<tr>
<td width="46%" valign="middle">

### ソース固定レタッチ

結果を実際に生成したフレームから描画。共有ズーム/パン、ソースフレームスライダー、軟らかいブラシ、復元ブラシ、ストローク単位の undo、未書き出しピクセルの保護。

</td>
<td width="54%">

<a href="assets/readme/retouch.mp4"><picture><source srcset="assets/readme/retouch.gif" type="image/gif"><img src="assets/readme/retouch_poster.jpg" alt="ソースフレームと結果を並べたレタッチ作業領域" width="100%" /></picture></a>

</td>
</tr>

<tr>
<td width="46%" valign="middle">

### バッチ処理

スタックの入ったフォルダを指定し、手法と出力先を選んで放置。位置合わせのみならスタックごとに 1 フォルダ、合成なら合成画像だけを書き出して一覧に表示します。

</td>
<td width="54%">

<a href="assets/readme/batch.mp4"><picture><source srcset="assets/readme/batch.gif" type="image/gif"><img src="assets/readme/batch_poster.jpg" alt="バッチ処理ダイアログ" width="100%" /></picture></a>

</td>
</tr>

<tr>
<td width="46%" valign="middle">

### 出版向けスケールバー

TIFF メタデータまたは手入力で校正し、実際のフォントと実寸で描画。単体画像、スタック、TIFF、GIF、バッチ出力のすべてに焼き込みます。 読みやすく収まらない場合は、表示と異なる短いバーを描かずに明示します。

[ユーザーマニュアル →](./docs/USER_MANUAL_JA.md)

</td>
<td width="54%">

<a href="assets/readme/scalebar.mp4"><picture><source srcset="assets/readme/scalebar.gif" type="image/gif"><img src="assets/readme/scalebar_poster.jpg" alt="ライブプレビュー付きスケールバーダイアログ" width="100%" /></picture></a>

</td>
</tr>

<tr>
<td width="46%" valign="middle">

### ラベルとキャプション

キャプションを管理されたリストとして保持します。各項目は対象（ソース、位置合わせ済みスタック、特定の出力）、フレーム範囲、スタイルを個別に持ちます。プレビューとすべての書き出しが同じ対象とフレーム番号を使うため、`{value}` は保存・名前を付けて保存・ドラッグ・すべて書き出しで同じ値を示します。

</td>
<td width="54%">

<a href="assets/readme/labels.mp4"><picture><source srcset="assets/readme/labels.gif" type="image/gif"><img src="assets/readme/labels_poster.jpg" alt="ラベルとキャプション" width="100%" /></picture></a>

</td>
</tr>
</table>

**その他：**

- **フォーカススタッキング作業フロー**——スタック閲覧、シャープネス曲線、クイックプレビュー、ROI レンダリング、手法ごとの設定を再起動後も保持。
- **顕微鏡向け**——16 ビットを一貫保持、TIFF µm/px、フラットフィールド補正、複数ページ TIFF Z スタック。
- **ヘッドレス CLI**——GUI と同じパイプラインを `openfocus` として提供。バッチ、`--dry-run`、`--resume`、JSON 出力に対応。
- **プロジェクトと復元**——`.ofproj` セッション、クラッシュ自動復元、スタック編集と注釈の undo/redo。
- **12 の UI 言語**——English、简体中文、日本語、Español、Deutsch、Français、Русский、한국어、Português、Italiano、Nederlands、Polski。システムロケールから自動判定し、新しい言語は `locales_<code>.py` 1 ファイルで追加できます。
- **省メモリ**——ニューラル重みは AI 手法を使うときだけ読み込み、古典手法のセッションは約 110 MB で待機します（260 MB ではなく）。

<a id="install"></a>
## ⬇️ インストール

セットアップ不要 — [Releases](https://github.com/Xinzhe99/OpenFocus/releases) からビルドを取得してください：

| プラットフォーム | ファイル | 種類 |
|------------------|----------|------|
| Windows 10/11（64 ビット） | `OpenFocus-*-setup.exe` | **インストーラー**（推奨） |
| Windows 10/11（64 ビット） | `OpenFocus-*-windows-x64.zip` | ポータブル（解凍して実行） |
| macOS Apple Silicon（M1–M4） | `OpenFocus-*-macos.dmg` | ディスクイメージ（Applications へドラッグ） |
| macOS Apple Silicon（M1–M4） | `OpenFocus-*-macos-arm64.zip` | ポータブル（解凍して実行） |

アプリは未署名のため、初回起動で警告が出る場合があります：**Windows** は「詳細情報 → 実行」を選びます（初回は 30〜60 秒かかることがあります）。**macOS** は右クリック →「開く」、またはターミナルで `xattr -cr /Applications/OpenFocus.app` を実行します。

### 💻 コマンドライン

Windows・macOS・Linux では CLI もインストールできます。同じパイプラインを GUI なしで動かせます：

```bash
pip install openfocus
openfocus -i ./stack_folder -o ./result/fused.png
```

AI モデル、重み、HEIC とカメラ RAW のデコードはすべてこの 1 回のインストールに含まれます。Python 3.13 以降では `dtcwt` が省略され、DTCWT の項目は説明付きで無効になります。

<a id="community"></a>
## コミュニティとサポート

- **ドキュメント** — [ユーザーマニュアル](./docs/USER_MANUAL_JA.md)（12 言語） · [変更履歴](./CHANGELOG.md) · [パフォーマンス](./docs/PERFORMANCE.md) · [アルゴリズム比較](./docs/ALGORITHM_PARITY.md)
- **引用** — [CITATION.cff](./CITATION.cff)。各リリースは Zenodo に保存されています（[DOI](https://doi.org/10.5281/zenodo.23050823)）
- **フィードバック** — 不具合、提案、アルゴリズムの改善は [issue](https://github.com/Xinzhe99/OpenFocus/issues) へ
- **クレジット** — 各合成・位置合わせアルゴリズムは元の研究ライセンスに従います。StackMFF V3 は [Xinzhe99/StackMFF-V3](https://github.com/Xinzhe99/StackMFF-V3) にあります

<a id="developing"></a>
## 開発

チェックアウトから実行：

```bash
conda create -n openfocus python=3.10
conda activate openfocus
pip install -r requirements.txt
python main.py
```

PyInstaller でのパッケージングは [docs/BUILD_COMMANDS.md](./docs/BUILD_COMMANDS.md) を参照してください。

<a id="star-history"></a>
## ⭐ Star 履歴

[![Star History Chart](https://api.star-history.com/svg?repos=Xinzhe99/OpenFocus&type=Date)](https://star-history.com/#Xinzhe99/OpenFocus&Date)
