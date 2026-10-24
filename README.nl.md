<p align="center">
  <a href="README.md">English</a> ·
  <a href="README.zh-CN.md">简体中文</a> ·
  <a href="README.ja.md">日本語</a> ·
  <a href="README.es.md">Español</a> ·
  <a href="README.de.md">Deutsch</a> ·
  <a href="README.fr.md">Français</a> ·
  <a href="README.ru.md">Русский</a> ·
  <a href="README.ko.md">한국어</a> ·
  <a href="README.pt.md">Português</a> ·
  <a href="README.it.md">Italiano</a> ·
  <strong>Nederlands</strong> ·
  <a href="README.pl.md">Polski</a>
</p>

# <img src="assets/OpenFocus.png" alt="OpenFocus Logo" width="120"> OpenFocus

OpenFocus is een open-source werkstation voor het uitlijnen en fuseren van multifocus-afbeeldingen. De kwaliteit richt zich op commerciële focus-stackingsoftware; de code is volledig open en eenvoudig uit te breiden.

<p align="left">
  <a href="https://www.python.org/downloads/release/python-3100/"><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+" /></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?logo=open-source-initiative&logoColor=white" alt="License: MIT" /></a>
  <a href="https://github.com/Xinzhe99/OpenFocus"><img src="https://img.shields.io/badge/GitHub-Repository-181717?logo=github&logoColor=white" alt="GitHub Repository" /></a>
  <a href="https://github.com/Xinzhe99/OpenFocus/releases"><img src="https://img.shields.io/badge/Windows%20%7C%20macOS-Download-0078D7?logo=github&logoColor=white" alt="Download" /></a>
  <a href="https://doi.org/10.5281/zenodo.23050823"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.23050823-0078D7?logo=doi&logoColor=white" alt="DOI" /></a>
</p>

<p align="center">
  <img src="assets/ui.png" alt="OpenFocus-interface" width="820">
</p>

> **v1.68.0** — batchexport volgt de gekozen modus: alleen uitlijnen maakt één `<stack>_aligned`-map en toont die in de lijst, fusie toont alleen het gefuseerde beeld, en zonder uitlijning exporteren wordt alleen dat beeld geschreven. Achtergebleven `*_registered*`-kopieën in de bronmap worden niet meer als frames geladen. Ook labels, schaalbalken, methodevergelijking, diepte-/modelexport, updates en retouche zijn herzien volgens de audit in `docs/ux-specs`.

## Download en installatie

Download een build via [Releases](https://github.com/Xinzhe99/OpenFocus/releases):

| Platform | Bestand | Type |
|----------|---------|------|
| Windows 10/11 (64-bit) | `OpenFocus-*-setup.exe` | **Installer** (aanbevolen) |
| Windows 10/11 (64-bit) | `OpenFocus-*-windows-x64.zip` | Draagbaar (uitpakken en starten) |
| macOS Apple Silicon (M1–M4) | `OpenFocus-*-macos.dmg` | Schijfimage (naar Programma's slepen) |
| macOS Apple Silicon (M1–M4) | `OpenFocus-*-macos-arm64.zip` | Draagbaar (uitpakken) |

De app is niet ondertekend. Bij de eerste start kan Windows vragen om “Meer informatie → Toch uitvoeren”; op macOS: rechtermuisknop → Open of `xattr -cr /Applications/OpenFocus.app`.

## Functies

- **Acht fusiemethoden** — Guided Filter, DCT, DTCWT, GFG-FGF, StackMFF-V3 (neuraal) en OpenFocus A/B/C.
- **Drie uitlijnmodi** — snelle similariteitstransformatie, ECC en homografie; fuseren zonder uitlijning kan ook.
- **Focus-stackingworkflow** — stackbrowser, A/B-veegvergelijking, dieptekaarten, methodevergelijking, batchverwerking en snelle preview.
- **Retouche met herkomst** — pixels uit de frames die het resultaat werkelijk hebben gemaakt, gedeelde zoom/positie, bronframeslider, zachte penseel, herstelpenseel, undo/redo per streek en bescherming van niet-geëxporteerde uitvoer.
- **Klaar voor microscopie en publicatie** — 16-bit van begin tot eind, µm/px uit TIFF, flat-fieldcorrectie, schaalbalken, meerbladige TIFF Z-stacks en getextureerde OBJ-export.
- **Headless CLI** — dezelfde pipeline via `openfocus`, met batch, `--dry-run`, `--resume` en JSON-uitvoer.
- **Twaalf UI-talen** — English, 简体中文, 日本語, Español, Deutsch, Français, Русский, 한국어, Português, Italiano, Nederlands, Polski.

## Snel starten

```bash
# Volledige CLI: alle methoden, AI-model, HEIC en camera-RAW
pip install openfocus

# Een map met afbeeldingen fuseren
openfocus -i ./stack_folder -o ./result/fused.png

# DTCWT met ECC en 8 threads
openfocus -i ./stack_folder -o ./result/fused.png -m dtcwt -a ecc -t 8
```

GUI vanuit de broncode starten:

```bash
conda create -n openfocus python=3.10
conda activate openfocus
pip install -r requirements.txt
python main.py
```

## Documentatie

- [Gebruikershandleiding (Nederlands)](./docs/USER_MANUAL_NL.md)
- [Wijzigingslog](./CHANGELOG.md)
- [Prestaties en grote stacks](./docs/PERFORMANCE.md)
- [Buildcommando's](./docs/BUILD_COMMANDS.md)
- [Algoritmepariteit](./docs/ALGORITHM_PARITY.md)

De volledige Engelse referentie staat in [README.md](./README.md).

## Citatie en licentie

Als OpenFocus je werk helpt, citeer het project via [CITATION.cff](./CITATION.cff).

Het project valt onder de [MIT-licentie](./LICENSE). Elk fusie-/uitlijningsalgoritme behoudt daarnaast zijn oorspronkelijke licentie.
