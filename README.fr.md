<p align="center">
  <a href="https://github.com/Xinzhe99/OpenFocus">
    <img src="assets/OpenFocus.png" alt="OpenFocus" width="128">
  </a>
</p>

<h1 align="center">OpenFocus</h1>

<p align="center"><strong>Tous les plans de mise au point. Une seule image nette partout.</strong></p>

<p align="center">
  Dix algorithmes de fusion, trois modes de recalage et un espace de retouche lié à la source pour la macro, la microscopie et les piles industrielles. 16 bits de bout en bout, une CLI sans interface et une UI en douze langues.
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
  <a href="#install">Télécharger</a> ·
  <a href="./docs/USER_MANUAL_FR.md">Documentation</a> ·
  <a href="./CHANGELOG.md">Journal</a> ·
  <a href="https://doi.org/10.5281/zenodo.23050823">DOI</a>
</p>

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

<p align="center">
  <img src="assets/ui.png" alt="Interface d’OpenFocus" width="880">
</p>

## 📢 News

> [!NOTE]
> **v1.72.0** — StackMFF et StackMFF V2 rejoignent la liste : dix méthodes dans une seule installation — le réseau 3D d'origine (indépendant de l'ordre, sans réglage), le modèle récurrent V2 avec cartes de profondeur (conserver l'ordre de mise au point) et StackMFF V3 au nom unifié. Chaque sélecteur regroupe par IA, domaine spatial et domaine transformé, et chaque méthode s'explique au survol. Le mélange doux flou de V2 est retiré ; son échantillonnage de référence reste inchangé.

<a id="features"></a>
## Fonctionnalités

<table>

<tr>
<td width="46%" valign="middle">

### Fusion IA : StackMFF V3

La méthode par défaut classe chaque pixel selon son image source la plus probable via un réseau de neurones. Sur la même pile recalée, les contours des lettres restent plus nets et les surfaces plus propres qu'avec Guided Filter (netteté laplacienne 27,1 contre 28,5) ; face à une vérité synthétique, elle devance toutes les méthodes en PSNR et SSIM, avec moins de pixels fantômes que tout algorithme classique.

[StackMFF V3 →](https://github.com/Xinzhe99/StackMFF V3) · [Benchmark →](tools/synthetic_benchmark.py)

</td>
<td width="54%">

<a href="assets/readme/ai.mp4"><picture><source srcset="assets/readme/ai.gif" type="image/gif"><img src="assets/readme/ai_poster.jpg" alt="Fusion IA : StackMFF V3" width="100%" /></picture></a>

</td>
</tr>

<tr>
<td width="46%" valign="middle">

### Comparaison A/B

Faites glisser le séparateur entre une image source et le résultat : vous voyez immédiatement ce que la fusion a changé — dérive de recalage, halos sur les bords et détail perdu, à tout zoom.

[Manuel →](./docs/USER_MANUAL_FR.md)

</td>
<td width="54%">

<a href="assets/readme/wipe.mp4"><picture><source srcset="assets/readme/wipe.gif" type="image/gif"><img src="assets/readme/wipe_poster.jpg" alt="Comparaison A/B entre une image et le résultat" width="100%" /></picture></a>

</td>
</tr>

<tr>
<td width="46%" valign="middle">

### Comparaison de méthodes

Rendez toutes les méthodes une fois, comparez-les côte à côte sur le même recadrage, puis adoptez la gagnante comme nouvelle sortie ou copiez ses paramètres dans la fenêtre principale.

</td>
<td width="54%">

<a href="assets/readme/method-comparison.mp4"><picture><source srcset="assets/readme/method-comparison.gif" type="image/gif"><img src="assets/readme/method-comparison_poster.jpg" alt="Huit méthodes de fusion en parallèle" width="100%" /></picture></a>

</td>
</tr>

<tr>
<td width="46%" valign="middle">

### Carte de profondeur et 3D

Position de mise au point par pixel via cinq mesures, palettes scientifiques, calibration Z physique depuis les métadonnées TIFF et export OBJ texturé de la surface.

</td>
<td width="54%">

<a href="assets/readme/depth.mp4"><picture><source srcset="assets/readme/depth.gif" type="image/gif"><img src="assets/readme/depth_poster.jpg" alt="Carte de profondeur et surface 3D de la même pile" width="100%" /></picture></a>

</td>
</tr>

<tr>
<td width="46%" valign="middle">

### Retouche liée à la source

Peignez depuis les images qui ont réellement produit le résultat : zoom et déplacement partagés, curseur d'image source, pinceau doux, pinceau de restauration, annulation par trait et protection des pixels non exportés.

</td>
<td width="54%">

<a href="assets/readme/retouch.mp4"><picture><source srcset="assets/readme/retouch.gif" type="image/gif"><img src="assets/readme/retouch_poster.jpg" alt="Espace de retouche avec image source et résultat" width="100%" /></picture></a>

</td>
</tr>

<tr>
<td width="46%" valign="middle">

### Traitement par lots

Pointez un dossier de piles, choisissez méthode et destination, puis laissez tourner. L'alignement seul écrit un dossier par pile ; la fusion n'écrit et n'affiche que l'image fusionnée.

</td>
<td width="54%">

<a href="assets/readme/batch.mp4"><picture><source srcset="assets/readme/batch.gif" type="image/gif"><img src="assets/readme/batch_poster.jpg" alt="Boîte de dialogue de traitement par lots" width="100%" /></picture></a>

</td>
</tr>

<tr>
<td width="46%" valign="middle">

### Barres d'échelle de publication

Calibrées depuis les métadonnées TIFF ou saisies à la main, dessinées avec la police et la longueur réelles, et gravées dans chaque export : image, pile, TIFF, GIF et lot. Si elle ne peut pas être lisible, le dialogue le signale au lieu de dessiner une barre plus courte que son étiquette.

[Manuel →](./docs/USER_MANUAL_FR.md)

</td>
<td width="54%">

<a href="assets/readme/scalebar.mp4"><picture><source srcset="assets/readme/scalebar.gif" type="image/gif"><img src="assets/readme/scalebar_poster.jpg" alt="Boîte de dialogue de barre d'échelle avec aperçu" width="100%" /></picture></a>

</td>
</tr>

<tr>
<td width="46%" valign="middle">

### Libellés et légendes

Les légendes forment une liste gérée : chacune a sa portée (source, pile recalée ou sortie précise), sa plage d'images et son style. L'aperçu et tous les exports utilisent la même cible et le même numéro d'image, donc `{value}` se lit de la même façon dans Enregistrer, Enregistrer sous, le glisser-déposer et Tout exporter.

</td>
<td width="54%">

<a href="assets/readme/labels.mp4"><picture><source srcset="assets/readme/labels.gif" type="image/gif"><img src="assets/readme/labels_poster.jpg" alt="Libellés et légendes" width="100%" /></picture></a>

</td>
</tr>
</table>

**Également inclus :**

- **Flux de focus stacking** — navigateur de piles, courbe de netteté, aperçu rapide, rendu ROI et réglages par méthode mémorisés entre sessions.
- **Prêt pour la microscopie** — 16 bits de bout en bout, µm/px depuis TIFF, correction de champ plat, Z-stacks TIFF multipages.
- **CLI sans interface** — la même chaîne que `openfocus`, avec lots, `--dry-run`, `--resume` et sortie JSON.
- **Projets et récupération** — sessions `.ofproj`, récupération après plantage, annuler/rétablir sur les piles et les annotations.
- **Douze langues d'interface** — English, 简体中文, 日本語, Español, Deutsch, Français, Русский, 한국어, Português, Italiano, Nederlands, Polski ; détectées du système, une nouvelle langue est un fichier `locales_<code>.py`.
- **Économe en mémoire** — les poids neuronaux ne se chargent qu'avec la méthode AI ; une session classique reste autour de 110 Mo au lieu de 260 Mo.

<a id="install"></a>
## ⬇️ Installation

Aucune configuration — récupérez une version sur [Releases](https://github.com/Xinzhe99/OpenFocus/releases) :

| Plateforme | Fichier | Type |
|------------|---------|------|
| Windows 10/11 (64 bits) | `OpenFocus-*-setup.exe` | **Installateur** (recommandé) |
| Windows 10/11 (64 bits) | `OpenFocus-*-windows-x64.zip` | Portable (décompresser et lancer) |
| macOS Apple Silicon (M1–M4) | `OpenFocus-*-macos.dmg` | Image disque (glisser dans Applications) |
| macOS Apple Silicon (M1–M4) | `OpenFocus-*-macos-arm64.zip` | Portable (décompresser) |

L'application n'est pas signée : le premier lancement peut afficher un avertissement. **Windows** SmartScreen → « Informations complémentaires » → « Exécuter quand même » (le premier démarrage peut prendre 30–60 s) ; **macOS** clic droit → Ouvrir, ou `xattr -cr /Applications/OpenFocus.app` dans le Terminal.

### 💻 Ligne de commande

Ou installez la CLI sous Windows, macOS et Linux — la même chaîne, sans pile graphique :

```bash
pip install openfocus
openfocus -i ./stack_folder -o ./result/fused.png
```

Le modèle IA, ses poids, le décodage HEIC et RAW y sont tous inclus. Sous Python 3.13+, `dtcwt` est ignoré et l'entrée DTCWT reste désactivée avec une explication.

<a id="community"></a>
## Communauté et support

- **Documentation** — [Manuel](./docs/USER_MANUAL_FR.md) (douze langues) · [Journal](./CHANGELOG.md) · [Performance](./docs/PERFORMANCE.md) · [Parité des algorithmes](./docs/ALGORITHM_PARITY.md)
- **Citation** — [CITATION.cff](./CITATION.cff) ; chaque version est archivée sur Zenodo ([DOI](https://doi.org/10.5281/zenodo.23050823))
- **Retours** — ouvrez une [issue](https://github.com/Xinzhe99/OpenFocus/issues) pour les bugs, idées et travaux d'algorithmes
- **Crédits** — chaque algorithme de fusion et de recalage conserve sa licence de recherche d'origine ; StackMFF V3 est sur [Xinzhe99/StackMFF V3](https://github.com/Xinzhe99/StackMFF V3)

<a id="developing"></a>
## Développement

Lancer depuis le dépôt :

```bash
conda create -n openfocus python=3.10
conda activate openfocus
pip install -r requirements.txt
python main.py
```

Empaquetage avec PyInstaller : [docs/BUILD_COMMANDS.md](./docs/BUILD_COMMANDS.md).

<a id="star-history"></a>
## ⭐ Historique des stars

[![Star History Chart](https://api.star-history.com/svg?repos=Xinzhe99/OpenFocus&type=Date)](https://star-history.com/#Xinzhe99/OpenFocus&Date)
