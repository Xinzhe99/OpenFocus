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

# <img src="assets/OpenFocus.png" alt="OpenFocus Logo" width="120"> OpenFocus

OpenFocus ist eine Open-Source-Workstation für die Registrierung und Fusion von Multifokus-Bildern. Die Qualität orientiert sich an kommerzieller Focus-Stacking-Software; der Code ist vollständig offen und erweiterbar.

<p align="left">
  <a href="https://www.python.org/downloads/release/python-3100/"><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+" /></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?logo=open-source-initiative&logoColor=white" alt="License: MIT" /></a>
  <a href="https://github.com/Xinzhe99/OpenFocus"><img src="https://img.shields.io/badge/GitHub-Repository-181717?logo=github&logoColor=white" alt="GitHub Repository" /></a>
  <a href="https://github.com/Xinzhe99/OpenFocus/releases"><img src="https://img.shields.io/badge/Windows%20%7C%20macOS-Download-0078D7?logo=github&logoColor=white" alt="Download" /></a>
  <a href="https://doi.org/10.5281/zenodo.23050823"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.23050823-0078D7?logo=doi&logoColor=white" alt="DOI" /></a>
</p>

<p align="center">
  <img src="assets/ui.png" alt="OpenFocus-Oberfläche" width="820">
</p>

> **v1.68.0** — der Batch-Export folgt dem gewählten Modus: Nur-Ausrichten erzeugt einen `<Stapel>_aligned`-Ordner und listet ihn, Fusion listet nur das fused Bild, und ohne Ausrichtungs-Export wird nur dieses geschrieben. Übrig gebliebene `*_registered*`-Kopien im Quellordner werden nicht mehr als Frames geladen. Ebenfalls überarbeitet: Labels, Maßstabsbalken, Methodenvergleich, Tiefen-/Modellexport, Updates und Retusche gemäß dem Audit in `docs/ux-specs`.

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
