<p align="center">
  <a href="README.md">English</a> ·
  <a href="README.zh-CN.md">简体中文</a> ·
  <a href="README.ja.md">日本語</a> ·
  <a href="README.es.md">Español</a> ·
  <a href="README.de.md">Deutsch</a> ·
  <strong>Français</strong> ·
  <a href="README.ru.md">Русский</a> ·
  <a href="README.ko.md">한국어</a> ·
  <a href="README.pt.md">Português</a> ·
  <a href="README.it.md">Italiano</a> ·
  <a href="README.nl.md">Nederlands</a> ·
  <a href="README.pl.md">Polski</a>
</p>

# <img src="assets/OpenFocus.png" alt="OpenFocus Logo" width="120"> OpenFocus

OpenFocus est un poste de travail open source pour le recalage et la fusion d’images à mise au point multiple, avec une qualité comparable aux logiciels commerciaux et une architecture facile à étendre.

<p align="left">
  <a href="https://www.python.org/downloads/release/python-3100/"><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+" /></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?logo=open-source-initiative&logoColor=white" alt="License: MIT" /></a>
  <a href="https://github.com/Xinzhe99/OpenFocus"><img src="https://img.shields.io/badge/GitHub-Repository-181717?logo=github&logoColor=white" alt="GitHub Repository" /></a>
  <a href="https://github.com/Xinzhe99/OpenFocus/releases"><img src="https://img.shields.io/badge/Windows%20%7C%20macOS-Download-0078D7?logo=github&logoColor=white" alt="Download" /></a>
  <a href="https://doi.org/10.5281/zenodo.23050823"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.23050823-0078D7?logo=doi&logoColor=white" alt="DOI" /></a>
</p>

<p align="center">
  <img src="assets/ui.png" alt="Interface d’OpenFocus" width="820">
</p>

> **v1.68.0** — l'export par lots suit le mode choisi : l'alignement seul crée un dossier `<pile>_aligned` et l'affiche, la fusion n'affiche que l'image fusionnée, et sans export de l'alignement seule cette image est écrite. Les copies `*_registered*` restées dans le dossier source ne sont plus chargées comme images. Retouche également des libellés, barres d'échelle, comparaison de méthodes, export de profondeur/modèle, mises à jour et retouche d'après l'audit `docs/ux-specs`.

## Téléchargement et installation

Téléchargez une version depuis [Releases](https://github.com/Xinzhe99/OpenFocus/releases) :

| Plateforme | Fichier | Type |
|------------|---------|------|
| Windows 10/11 (64 bits) | `OpenFocus-*-setup.exe` | **Installateur** (recommandé) |
| Windows 10/11 (64 bits) | `OpenFocus-*-windows-x64.zip` | Portable (décompresser et lancer) |
| macOS Apple Silicon (M1–M4) | `OpenFocus-*-macos.dmg` | Image disque (glisser dans Applications) |
| macOS Apple Silicon (M1–M4) | `OpenFocus-*-macos-arm64.zip` | Portable (décompresser) |

L’application n’est pas signée : au premier lancement, Windows peut demander « Informations complémentaires → Exécuter quand même » ; sur macOS, clic droit → Ouvrir, ou `xattr -cr /Applications/OpenFocus.app`.

## Fonctionnalités

- **Huit méthodes de fusion** — Guided Filter, DCT, DTCWT, GFG-FGF, StackMFF-V3 (réseau de neurones) et OpenFocus A/B/C.
- **Trois modes de recalage** — similarité rapide, ECC et homographie ; la fusion sans recalage reste possible.
- **Workflow focus stacking** — navigateur de pile, comparaison A/B, cartes de profondeur, comparaison de méthodes, traitement par lots et aperçu rapide.
- **Retouche liée à la source** — pixels issus des images réellement fusionnées, zoom/position partagés, curseur de source, pinceau doux, pinceau de restauration, annulation/rétablissement par trait et protection des sorties non exportées.
- **Prêt pour la microscopie et la publication** — 16 bits de bout en bout, µm/px depuis les TIFF, correction de champ plat, barres d’échelle, piles Z TIFF multipages et export OBJ texturé.
- **CLI sans interface** — le même pipeline via `openfocus`, avec lots, `--dry-run`, `--resume` et sortie JSON.
- **Douze langues d’interface** — English, 简体中文, 日本語, Español, Deutsch, Français, Русский, 한국어, Português, Italiano, Nederlands, Polski.

## Démarrage rapide

```bash
# CLI complète : toutes les méthodes, modèle IA, HEIC et RAW appareil photo
pip install openfocus

# Fusionner un dossier d’images
openfocus -i ./stack_folder -o ./result/fused.png

# DTCWT avec recalage ECC et 8 threads
openfocus -i ./stack_folder -o ./result/fused.png -m dtcwt -a ecc -t 8
```

Lancer l’interface depuis les sources :

```bash
conda create -n openfocus python=3.10
conda activate openfocus
pip install -r requirements.txt
python main.py
```

## Documentation

- [Manuel utilisateur (français)](./docs/USER_MANUAL_FR.md)
- [Journal des modifications](./CHANGELOG.md)
- [Performances et grandes piles](./docs/PERFORMANCE.md)
- [Commandes de compilation](./docs/BUILD_COMMANDS.md)
- [Parité des algorithmes](./docs/ALGORITHM_PARITY.md)

La référence complète en anglais se trouve dans [README.md](./README.md).

## Citation et licence

Si OpenFocus vous aide dans vos travaux, citez le projet via [CITATION.cff](./CITATION.cff).

Le projet est distribué sous [licence MIT](./LICENSE). Chaque algorithme de fusion/recalage conserve également sa licence d’origine.
