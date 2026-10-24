<p align="center">
  <a href="https://github.com/Xinzhe99/OpenFocus">
    <img src="assets/OpenFocus.png" alt="OpenFocus" width="128">
  </a>
</p>

<h1 align="center">OpenFocus</h1>

<p align="center"><strong>Jede Schärfeebene. Ein durchgehend scharfes Bild.</strong></p>

<p align="center">
  Acht Fusionsverfahren, drei Registrierungsmodi und eine quellengebundene Retusche für Makro-, Mikroskopie- und Industriestapel. 16 Bit durchgängig, eine Headless-CLI und eine Oberfläche in zwölf Sprachen.
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
  <a href="#download">Download</a> ·
  <a href="./docs/USER_MANUAL_DE.md">Dokumentation</a> ·
  <a href="#command-line-usage">CLI</a> ·
  <a href="./CHANGELOG.md">Änderungen</a> ·
  <a href="https://doi.org/10.5281/zenodo.23050823">DOI</a>
</p>

<p align="center">
  <a href="README.md">English</a> ·
  <a href="README.zh-CN.md">简体中文</a> ·
  <a href="README.ja.md">日本語</a> ·
  <a href="README.es.md">Español</a> ·
  <strong>Deutsch</strong> ·
  <a href="README.fr.md">Français</a> ·
  <a href="README.ru.md">Русский</a> ·
  <a href="README.ko.md">한국어</a> ·
  <a href="README.pt.md">Português</a> ·
  <a href="README.it.md">Italiano</a> ·
  <a href="README.nl.md">Nederlands</a> ·
  <a href="README.pl.md">Polski</a>
</p>

<p align="center">
  <img src="assets/ui.png" alt="OpenFocus-Oberfläche" width="880">
</p>

## 📢 News

> [!NOTE]
> **v1.68.1** — der Batch-Export folgt dem gewählten Modus: Nur-Ausrichten erzeugt einen `<Stapel>_aligned`-Ordner und listet ihn, Fusion listet nur das fused Bild, und ohne Ausrichtungs-Export wird nur dieses geschrieben. Übrig gebliebene `*_registered*`-Kopien im Quellordner werden nicht mehr als Frames geladen. Ebenfalls überarbeitet: Labels, Maßstabsbalken, Methodenvergleich, Tiefen-/Modellexport, Updates und Retusche gemäß dem Audit in `docs/ux-specs`.

<a id="download"></a>
## Download & Installation

Eine Build-Version gibt es auf der [Releases](https://github.com/Xinzhe99/OpenFocus/releases)-Seite:

| Plattform | Datei | Typ |
|-----------|-------|-----|
| Windows 10/11 (64 Bit) | `OpenFocus-*-setup.exe` | **Installer** (empfohlen) |
| Windows 10/11 (64 Bit) | `OpenFocus-*-windows-x64.zip` | Portabel (entpacken und starten) |
| macOS Apple Silicon (M1–M4) | `OpenFocus-*-macos.dmg` | Disk-Image (in „Programme“ ziehen) |
| macOS Apple Silicon (M1–M4) | `OpenFocus-*-macos-arm64.zip` | Portabel (entpacken) |

Die App ist nicht signiert. Beim ersten Start kann ein Sicherheitshinweis erscheinen: unter Windows „Weitere Informationen → Trotzdem ausführen“, unter macOS Rechtsklick → Öffnen oder `xattr -cr /Applications/OpenFocus.app`.

## Funktionen

- **Acht Fusionsverfahren** — Guided Filter, DCT, DTCWT, GFG-FGF, StackMFF-V3 (neuronal) und OpenFocus A/B/C.
- **Drei Registrierungsmodi** — schnelle Ähnlichkeitstransformation, ECC und Homographie; Fusion ist auch ohne Registrierung möglich.
- **Focus-Stacking-Workflow** — Stack-Browser, A/B-Wipe, Tiefenkarten, Methodenvergleich, Stapelverarbeitung und Schnellvorschau.
- **Quellengebundene Retusche** — Pixel aus den tatsächlich verwendeten Quellbildern, gemeinsamer Zoom/Position, Quellbild-Regler, weicher Pinsel, Wiederherstellungspinsel, Undo/Redo pro Strich und Schutz nicht exportierter Ergebnisse.
- **Mikroskopie- und publikationsbereit** — 16 Bit durchgängig, µm/px aus TIFF, Flat-Field-Korrektur, Maßstabsbalken, mehrseitige TIFF-Z-Stacks und texturierte OBJ-Exporte.
- **Headless-CLI** — dieselbe Pipeline als `openfocus`, inklusive Batch, `--dry-run`, `--resume` und JSON-Ausgabe.
- **Zwölf UI-Sprachen** — English, 简体中文, 日本語, Español, Deutsch, Français, Русский, 한국어, Português, Italiano, Nederlands, Polski.

<a id="command-line-usage"></a>
## Schnellstart

```bash
# Vollständige CLI: alle Verfahren, KI-Modell, HEIC- und Kamera-RAW-Unterstützung
pip install openfocus

# Einen Bildordner fusionieren
openfocus -i ./stack_folder -o ./result/fused.png

# DTCWT mit ECC und 8 Threads
openfocus -i ./stack_folder -o ./result/fused.png -m dtcwt -a ecc -t 8
```

GUI aus dem Quellcode starten:

```bash
conda create -n openfocus python=3.10
conda activate openfocus
pip install -r requirements.txt
python main.py
```

## Dokumentation

- [Benutzerhandbuch (Deutsch)](./docs/USER_MANUAL_DE.md)
- [Changelog](./CHANGELOG.md)
- [Performance und große Stacks](./docs/PERFORMANCE.md)
- [Build-Befehle](./docs/BUILD_COMMANDS.md)
- [Algorithmus-Parität](./docs/ALGORITHM_PARITY.md)

Die vollständige englische Referenz steht in [README.md](./README.md).

## Zitation und Lizenz

Wenn OpenFocus Ihre Arbeit unterstützt, zitieren Sie das Projekt bitte über [CITATION.cff](./CITATION.cff).

Das Projekt steht unter der [MIT-Lizenz](./LICENSE). Für die einzelnen Fusions- und Registrierungsalgorithmen gelten zusätzlich deren Original-Lizenzen.
