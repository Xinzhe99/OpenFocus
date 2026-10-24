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

# <img src="assets/OpenFocus.png" alt="OpenFocus Logo" width="120"> OpenFocus

OpenFocus は、商用グレードのフォーカススタッキング品質を目指すオープンソースの多重焦点画像の位置合わせ・合成ワークステーションです。完全オープンソースで、拡張も容易です。

<p align="left">
  <a href="https://www.python.org/downloads/release/python-3100/"><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+" /></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?logo=open-source-initiative&logoColor=white" alt="License: MIT" /></a>
  <a href="https://github.com/Xinzhe99/OpenFocus"><img src="https://img.shields.io/badge/GitHub-Repository-181717?logo=github&logoColor=white" alt="GitHub Repository" /></a>
  <a href="https://github.com/Xinzhe99/OpenFocus/releases"><img src="https://img.shields.io/badge/Windows%20%7C%20macOS-Download-0078D7?logo=github&logoColor=white" alt="Download" /></a>
  <a href="https://doi.org/10.5281/zenodo.23050823"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.23050823-0078D7?logo=doi&logoColor=white" alt="DOI" /></a>
</p>

<p align="center">
  <img src="assets/ui.png" alt="OpenFocus の画面" width="820">
</p>

> **v1.68.0** — バッチ出力が選択モードに従います。位置合わせのみなら `<スタック名>_aligned` フォルダを作成して一覧に表示し、融合時は融合画像のみを表示、整列を保存しない場合は融合画像だけを書き出します。ソースフォルダに残る `*_registered*` コピーはフレームとして読み込まれません。`docs/ux-specs` の監査に沿ってラベル、スケールバー、手法比較、深度/モデル出力、更新、レタッチも改修しました。

## ダウンロードとインストール

[Releases](https://github.com/Xinzhe99/OpenFocus/releases) からビルドを取得できます。

| プラットフォーム | ファイル | 種類 |
|------------------|----------|------|
| Windows 10/11（64 ビット） | `OpenFocus-*-setup.exe` | **インストーラー**（推奨） |
| Windows 10/11（64 ビット） | `OpenFocus-*-windows-x64.zip` | ポータブル（解凍して実行） |
| macOS Apple Silicon（M1–M4） | `OpenFocus-*-macos.dmg` | ディスクイメージ（Applications へドラッグ） |
| macOS Apple Silicon（M1–M4） | `OpenFocus-*-macos-arm64.zip` | ポータブル（解凍して実行） |

アプリは未署名のため、初回起動時に警告が出る場合があります。Windows は「詳細情報 → 実行」、macOS は右クリック →「開く」、または `xattr -cr /Applications/OpenFocus.app` を実行してください。

## 主な機能

- **8 種類の合成アルゴリズム** — Guided Filter、DCT、DTCWT、GFG-FGF、StackMFF-V3（ニューラル）、OpenFocus A/B/C。
- **3 種類の位置合わせ** — 高速相似変換、ECC、ホモグラフィ。位置合わせなしの合成も可能。
- **フォーカススタック Workflow** — スタック表示、A/B ワイプ比較、深度マップ、手法比較、バッチ処理、クイックプレビュー。
- **ソース固定のレタッチ** — 実際に合成に使ったフレームから取得。共有ズーム/位置、ソースフレームスライダー、軟らかいブラシ、復元ブラシ、ストローク単位の undo/redo、未書き出し保護。
- **顕微鏡・出版向け** — 8/16 ビット、TIFF の µm/px、フラットフィールド補正、スケールバー、マルチページ TIFF Z スタック、テクスチャ付き OBJ 出力。
- **ヘッドレス CLI** — GUI と同じ処理を `openfocus` で実行。バッチ、`--dry-run`、`--resume`、JSON 出力に対応。
- **12 言語の UI** — English、简体中文、日本語、Español、Deutsch、Français、Русский、한국어、Português、Italiano、Nederlands、Polski。

## クイックスタート

```bash
# すべてのアルゴリズム、AI モデル、HEIC/RAW 対応を含む CLI
pip install openfocus

# フォルダ内の画像を合成
openfocus -i ./stack_folder -o ./result/fused.png

# DTCWT + ECC、8 スレッド
openfocus -i ./stack_folder -o ./result/fused.png -m dtcwt -a ecc -t 8
```

ソースから GUI を起動：

```bash
conda create -n openfocus python=3.10
conda activate openfocus
pip install -r requirements.txt
python main.py
```

## ドキュメント

- [ユーザーマニュアル（日本語）](./docs/USER_MANUAL_JA.md)
- [変更履歴](./CHANGELOG.md)
- [性能と大規模スタック](./docs/PERFORMANCE.md)
- [ビルドコマンド](./docs/BUILD_COMMANDS.md)
- [アルゴリズム互換性](./docs/ALGORITHM_PARITY.md)

詳細な英語版は [README.md](./README.md) を参照してください。

## 引用とライセンス

研究や業務で OpenFocus を利用する場合は、[CITATION.cff](./CITATION.cff) の引用情報をご利用ください。

本プロジェクトは [MIT License](./LICENSE) で公開されています。各融合・位置合わせアルゴリズムの原論文と実装には、それぞれのライセンスが適用されます。
