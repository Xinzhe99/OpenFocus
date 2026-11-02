<p align="center">
  <a href="https://github.com/Xinzhe99/OpenFocus">
    <img src="../../assets/OpenFocus.png" alt="OpenFocus" width="128">
  </a>
</p>

<h1 align="center">OpenFocus</h1>

<p align="center"><strong>Chaque pixel, depuis son image la plus nette.</strong></p>

<p align="center">
  Dix algorithmes de fusion, quatre modes de recalage et un espace de retouche lié à la source pour la macro, la microscopie et les piles industrielles. 16 bits de bout en bout, une CLI sans interface et une interface en douze langues.
</p>

<p align="center">
  <a href="https://github.com/Xinzhe99/OpenFocus/releases"><img src="https://img.shields.io/github/v/release/Xinzhe99/OpenFocus?label=release&color=2f81f7" alt="Version"></a>
  <a href="https://github.com/Xinzhe99/OpenFocus/stargazers"><img src="https://img.shields.io/github/stars/Xinzhe99/OpenFocus?color=e3b341" alt="Étoiles"></a>
  <a href="../../LICENSE"><img src="https://img.shields.io/badge/license-MIT-3fb950" alt="Licence : MIT"></a>
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
  <a href="../USER_MANUAL_FR.md">Documentation</a> ·
  <a href="../../CHANGELOG.md">Journal des modifications</a> ·
  <a href="https://doi.org/10.5281/zenodo.23050823">DOI</a>
</p>

<p align="center">
  <a href="../../README.md">English</a> ·
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
  <img src="../../assets/ui.png" alt="Interface d'OpenFocus" width="880">
</p>

## 📢 Actualités

> [!NOTE]
> **v1.74.0** — Refonte des fusions OpenFocus A/B/C et du recalage rapide. L'application de bureau lance désormais automatiquement son canal d'agent local authentifié, si bien que `openfocus ctl` peut piloter la fenêtre en cours d'exécution sans indicateur de lancement particulier.

Les versions précédentes figurent dans le [journal des modifications](../../CHANGELOG.md).

📸 Si vous publiez des images créées avec OpenFocus, vous pouvez ajouter une mention telle que :

> Créé avec OpenFocus — https://github.com/Xinzhe99/OpenFocus

Ce n'est pas obligatoire, mais vivement apprécié.

<a id="features"></a>
## Fonctionnalités

<table>
<tr>
<td width="46%" valign="middle">

### Fusion IA : StackMFF V3

La méthode par défaut utilise un réseau de neurones pour estimer l'image source la plus probable de chaque pixel, puis mélange les contributions prédites ou choisit l'image la plus probable. Le banc d'essai synthétique compare les méthodes à une référence connue ; les résultats dépendent de la pile et des réglages.

[StackMFF V3 →](https://github.com/Xinzhe99/StackMFF-V3) · [Banc d'essai →](../../tools/synthetic_benchmark.py)

</td>
<td width="54%">

<a href="../../assets/readme/ai.mp4"><picture><source srcset="../../assets/readme/ai.gif" type="image/gif"><img src="../../assets/readme/ai_poster.jpg" alt="StackMFF V3 face au filtre guidé sur la même pile" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Comparaison par balayage

Faites glisser le séparateur entre une image et le résultat fusionné pour voir exactement ce que le mélange a changé : dérive de recalage, halos sur les bords et détail perdu, à n'importe quel zoom.

[Manuel →](../USER_MANUAL_FR.md)

</td>
<td width="54%">

<a href="../../assets/readme/wipe.mp4"><picture><source srcset="../../assets/readme/wipe.gif" type="image/gif"><img src="../../assets/readme/wipe_poster.jpg" alt="Comparaison A/B entre une image source et le résultat fusionné" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Comparaison de méthodes

Rendez toutes les méthodes une fois, comparez-les côte à côte sur le même recadrage, puis adoptez la gagnante comme nouvelle sortie ou copiez ses paramètres dans la fenêtre principale.

</td>
<td width="54%">

<a href="../../assets/readme/method-comparison.mp4"><picture><source srcset="../../assets/readme/method-comparison.gif" type="image/gif"><img src="../../assets/readme/method-comparison_poster.jpg" alt="Comparaison côte à côte de huit méthodes de fusion" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Carte de profondeur et 3D

Position de mise au point par pixel selon cinq mesures, palettes scientifiques, calibration Z physique à partir des métadonnées TIFF et export OBJ texturé de la surface.

</td>
<td width="54%">

<a href="../../assets/readme/depth.mp4"><picture><source srcset="../../assets/readme/depth.gif" type="image/gif"><img src="../../assets/readme/depth_poster.jpg" alt="Carte de profondeur et surface 3D de la même pile" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Retouche liée à la source

Peignez depuis les images qui ont réellement produit le résultat : zoom et déplacement partagés, curseur d'image source, pinceau doux, pinceau de restauration, annulation par trait et protection des pixels non exportés. Appliquez la retouche comme nouvelle sortie, puis **Exporter un TIFF multicouche** pour l'enregistrer avec ses images sources. Les libellés et barres d'échelle actifs ne sont pas inclus. GIMP 3.2.6 sous Windows ouvre le résultat et les calques sources ; les autres éditeurs n'ont pas été testés.

</td>
<td width="54%">

<a href="../../assets/readme/retouch.mp4"><picture><source srcset="../../assets/readme/retouch.gif" type="image/gif"><img src="../../assets/readme/retouch_poster.jpg" alt="Espace de retouche avec image source et résultat" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Traitement par lots

Indiquez un dossier de piles, choisissez une méthode et une destination, puis laissez tourner. Le mode alignement seul écrit un dossier par pile ; la fusion écrit l'image fusionnée et ne liste que celle-ci. La fusion groupée facultative échange l'équivalence exacte avec la pile complète contre une mémoire et un espace disque temporaire bornés.

</td>
<td width="54%">

<a href="../../assets/readme/batch.mp4"><picture><source srcset="../../assets/readme/batch.gif" type="image/gif"><img src="../../assets/readme/batch_poster.jpg" alt="Boîte de dialogue de traitement par lots" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Barres d'échelle de publication

Calibrées depuis les métadonnées TIFF ou saisies à la main, dessinées avec la police et la longueur réelles, et intégrées à chaque export : images seules, piles, TIFF multipage, GIF et lots. Une barre qui ne peut pas être dessinée lisiblement l'indique au lieu d'afficher une longueur qu'elle ne trace pas.

[Manuel →](../USER_MANUAL_FR.md)

</td>
<td width="54%">

<a href="../../assets/readme/scalebar.mp4"><picture><source srcset="../../assets/readme/scalebar.gif" type="image/gif"><img src="../../assets/readme/scalebar_poster.jpg" alt="Boîte de dialogue de barre d'échelle avec aperçu" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Légendes et libellés

Gérez une liste de légendes, chacune avec sa portée (pile source, pile alignée ou une sortie précise), sa plage d'images et son style. L'aperçu et chaque export utilisent la même cible et le même numéro d'image ; `{value}` se lit donc de la même façon dans Enregistrer, Enregistrer sous, le glisser-déposer et Tout exporter.

</td>
<td width="54%">

<a href="../../assets/readme/labels.mp4"><picture><source srcset="../../assets/readme/labels.gif" type="image/gif"><img src="../../assets/readme/labels_poster.jpg" alt="Gestionnaire de libellés avec aperçu" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Pilotable par un agent

Chaque fenêtre, widget et menu est accessible via un canal local protégé par jeton : un agent charge une pile, rend, retouche, compare, calcule la profondeur, exporte et répond aux boîtes de dialogue par commande typée, chemin de widget ou vrais événements souris et clavier. Aucune capture d'écran ni OCR n'est nécessaire ; le canal démarre avec l'application de bureau.

[Protocole →](../AGENT_CONTROL_SPEC.md) · [Skill →](../../skills/openfocus-control/SKILL.md)

</td>
<td width="54%">

<a href="../../assets/readme/agent.mp4"><picture><source srcset="../../assets/readme/agent.gif" type="image/gif"><img src="../../assets/readme/agent_poster.jpg" alt="Un agent pilote l'interface en direct" width="100%" /></picture></a>

</td>
</tr>
</table>

**Également inclus :**

- **Flux de travail focus stacking** — navigateur de piles, courbe de netteté, aperçu rapide, rendu ROI et réglages par méthode mémorisés entre les sessions.
- **Prêt pour la microscopie** — 16 bits de bout en bout, µm/px depuis les métadonnées TIFF, correction de champ plat par carte de poussière, piles Z TIFF multipage et export OBJ texturé.
- **Profils colorimétriques** — les exports PNG, TIFF, JPEG et WebP d'une seule image conservent un profil ICC RVB commun lorsque toutes les images sources ont le même profil. Les piles mixtes ou sans profil sont exportées sans profil ; les pixels ne sont pas convertis entre espaces colorimétriques.
- **PNG transparent** — les piles dont les masques rectangulaires à opacité constante coïncident peuvent être fusionnées en pleine taille et enregistrées en PNG. Voir les restrictions dans le manuel.
- **CLI sans interface** — le même pipeline sous `openfocus`, avec mode lot, `--dry-run`, `--resume` et sortie JSON pour les scripts.
- **Pilotable par un agent** — un canal local protégé par jeton pilote l'interface en direct (commandes de pipeline typées et contrôle générique des widgets, menus, souris et clavier), avec une skill, un exécuteur de fumée à 14 contrôles et un balayage complet des actions de menu.
- **Projets et récupération** — sessions `.ofproj`, récupération automatique après plantage, annuler/rétablir sur les modifications de pile et les annotations.
- **Douze langues d'interface** — English, 中文, 日本語, Español, Deutsch, Français, Русский, 한국어, Português, Italiano, Nederlands, Polski. Détection depuis la locale du système ; une nouvelle langue correspond à un fichier `locales/<code>.py`.
- **Peu gourmand en mémoire** — le réseau de neurones n'est chargé que lorsque la méthode IA est utilisée ; une session avec les méthodes classiques reste donc autour de 110 Mo au lieu de 260 Mo.

<a id="install"></a>
## ⬇️ Installation

Téléchargez la dernière version Windows ou macOS depuis le [site web d'OpenFocus](https://openfocus.top/#download). Les mêmes fichiers et les versions antérieures sont aussi disponibles sur [GitHub Releases](https://github.com/Xinzhe99/OpenFocus/releases).

| Plateforme | Fichier | Type |
|------------|---------|------|
| Windows 10/11 (64 bits) | `OpenFocus-*-setup.exe` | **Programme d'installation** (recommandé) |
| Windows 10/11 (64 bits) | `OpenFocus-*-windows-x64.zip` | Portable (décompresser et lancer) |
| macOS Apple Silicon (M1–M4) | `OpenFocus-*-macos.dmg` | Image disque (glisser dans Applications) |
| macOS Apple Silicon (M1–M4) | `OpenFocus-*-macos-arm64.zip` | Portable (décompresser) |

L'application n'est pas signée : le premier lancement peut afficher un avertissement. **Windows** : SmartScreen → « Informations complémentaires » → « Exécuter quand même » (le premier démarrage peut prendre 30 à 60 s). **macOS** : clic droit → Ouvrir, ou `xattr -cr /Applications/OpenFocus.app` dans le Terminal.

Pour installer depuis un terminal, prévisualisez d'abord la commande macOS ou Linux, puis exécutez-la sans `--dry-run`. Sous Linux, cela installe la CLI sans interface ; les versions macOS exigent Apple Silicon et s'installent dans `~/Applications`.

```bash
curl -fsSL https://openfocus.top/install.sh | sh -s -- --dry-run
curl -fsSL https://openfocus.top/install.sh | sh
```

Sous Windows, lancez l'installateur depuis PowerShell :

```powershell
irm https://openfocus.top/install.ps1 | iex
```

Le téléchargement direct du programme d'installation Windows est aussi disponible avec
`curl -fL https://openfocus.top/download/latest/windows-setup -o OpenFocus-setup.exe`.

### 🤖 Intégration d'agent

L'application de bureau démarre automatiquement un canal d'agent local protégé par jeton. Utilisez `openfocus ctl` pour des commandes typées ou des actions d'interface génériques sur la fenêtre en direct. Les agents peuvent charger des piles, rendre, exporter, ouvrir et remplir des boîtes de dialogue, répondre aux sélecteurs de fichiers, envoyer des événements souris et clavier et prendre des captures d'écran.

- Skill : [`skills/openfocus-control/SKILL.md`](../../skills/openfocus-control/SKILL.md)
- Protocole et référence des commandes : [`docs/AGENT_CONTROL_SPEC.md`](../AGENT_CONTROL_SPEC.md)
- Exécuteur de bout en bout : `python tools/agent_gui_smoke.py --launch`
- Balayage de toutes les actions : `python tools/agent_action_sweep.py --launch`

### 💻 Utilisation en ligne de commande

Vous pouvez aussi installer la CLI sous Windows, macOS et Linux — le même pipeline, sans la pile graphique :

```bash
pip install openfocus
openfocus -i ./stack_folder -o ./result/fused.png
```

Le modèle IA, ses poids, le décodage HEIC et RAW d'appareil photo sont tous inclus dans cette installation unique. `dtcwt` est ignoré à partir de Python 3.13, où l'entrée DTCWT reste désactivée avec une infobulle.

Les points de contrôle de retouche peuvent être rejoués sans interface avec la base fusionnée d'origine et les images sources traitées exactes, dans l'ordre enregistré :

```bash
openfocus retouch replay edits.json --base fused.png --sources processed1.png processed2.png --output retouched.tif
```

Le point de contrôle vérifie la base, les pixels sources, la géométrie et les masques de validité. Exportez les images traitées elles-mêmes (y compris tout recadrage de recalage ou de ROI), pas la pile d'entrée non traitée. Le rejeu refuse les contextes incompatibles et n'écrase jamais ses entrées. La sortie est en PNG ou TIFF et conserve la profondeur 8 ou 16 bits.

Les recettes enregistrent les réglages de traitement sans interface pour les réutiliser et les déplacer :

```bash
openfocus recipe create workflow.json -i ./stack_folder -o ./result/fused.png -m dct --cpu
openfocus recipe run workflow.json --dry-run
openfocus recipe run workflow.json
```

Le format JSON de version 1 contient `version`, un tableau `input`, soit `output`, soit `output_dir`, et un objet `options`. Les chemins sont relatifs au fichier de recette ; déplacez la recette avec ses fichiers d'entrée pour conserver ces relations. La création capture les réglages de la CLI et les paramètres d'algorithme résolus. Les champs inconnus, les valeurs invalides et les versions non prises en charge sont rejetés. Une exécution à blanc vérifie l'entrée sans écrire de fichiers de résultat. Les fichiers de projet existants restent inchangés.

Pour capturer les réglages d'un `.ofproj` existant, sélectionnez explicitement la nouvelle entrée et la nouvelle sortie. Les sources d'origine du projet n'ont pas besoin d'être présentes pour extraire les réglages :

```bash
openfocus recipe create workflow.json --project session.ofproj -i ./new_stack -o ./result/fused.png
```

Cette opération transfère à la recette CLI les réglages de fusion, de recalage, de luminosité, d'ordre, de netteté et de tuiles, et incorpore les réglages de traitement pour l'interface. Les exclusions d'images, l'échelle de chargement, les libellés et les modifications ne sont pas transférés ; préparez la nouvelle pile d'entrée en conséquence. Les projets comportant des transformations de source, notamment des corrections de champ plat appliquées, doivent d'abord exporter leur pile source corrigée. Les recettes exigent des sorties hors de leurs dossiers d'entrée et n'acceptent jamais un fichier d'entrée comme sortie. Les options de regroupement peuvent être capturées depuis les arguments de la CLI.

Dans l'interface, Fichier → Enregistrer un modèle de traitement enregistre les réglages sans chemins de source. Fichier → Appliquer un modèle de traitement applique un modèle, un projet ou une recette issue d'un projet à la pile actuelle sans remplacer ses entrées, transformations, exclusions ni libellés. Un modèle n'est pas lui-même une recette exécutable. Les réglages d'interface intégrés à une recette sont indépendants de ses options CLI ; si vous modifiez le JSON, gardez les deux cohérents.

La même API sans interface est disponible sans importer PyQt :

```python
from core.recipes import capture_project_recipe, capture_recipe, load_recipe, save_recipe, run_recipe

capture_recipe("workflow.json", ["-i", "stack_folder", "-o", "result/fused.png", "-m", "dct"])
capture_project_recipe("from-project.json", "session.ofproj",
                       ["-i", "new_stack", "-o", "result/fused.png"])
recipe = load_recipe("workflow.json")  # dictionnaire validé avec chemins relatifs
save_recipe("workflow.json", recipe)
exit_code = run_recipe("workflow.json", dry_run=True)  # utilisez False pour traiter
```

`capture_recipe` interprète ses arguments comme des chemins CLI ordinaires relatifs au répertoire courant ; `save_recipe` interprète les chemins stockés relativement à la recette. La recette enregistre les réglages et les références, y compris une empreinte de contenu pour un masque de défauts de capteur explicitement sélectionné. Elle n'enregistre pas les pixels d'entrée, les versions logicielles, le matériel ni les sommes de contrôle de la pile source ; conservez-les séparément lorsque la provenance compte. Des réglages identiques ne garantissent pas des pixels identiques d'une version logicielle, d'un pilote ou d'un appareil à l'autre ; les opérations neuronales sur GPU peuvent être non déterministes. Utilisez `--cpu` lorsqu'une exécution CPU est requise, et conservez le logiciel installé et les entrées.

<a id="community"></a>
## Communauté et support

- **Documentation** — [Manuel](../USER_MANUAL_FR.md) (douze langues) · [Journal](../../CHANGELOG.md) · [Performances](../PERFORMANCE.md) · [Méthodes de fusion](../FUSION_METHODS.md)
- **Citation** — [CITATION.cff](../../CITATION.cff) ; chaque version est archivée sur Zenodo ([DOI](https://doi.org/10.5281/zenodo.23050823))
- **Retours** — ouvrez un [issue](https://github.com/Xinzhe99/OpenFocus/issues) pour les bogues, les idées et le travail sur les algorithmes
- **Crédits** — chaque algorithme de fusion et de recalage conserve sa licence de recherche d'origine ; StackMFF V3 se trouve sur [Xinzhe99/StackMFF-V3](https://github.com/Xinzhe99/StackMFF-V3)
- **Rapports de sécurité** — [@ranas-mukminov](https://github.com/ranas-mukminov) a signalé la fenêtre de permissions du jeton de découverte ([PR #9](https://github.com/Xinzhe99/OpenFocus/pull/9))

<a id="license"></a>
## Licence

L'application OpenFocus est publiée sous [licence MIT](../../LICENSE).

**Exception de la série StackMFF.** Les algorithmes de fusion StackMFF — actuellement StackMFF V3 et toute future édition de pointe — restent la propriété du projet OpenFocus, qui conserve tous les droits sur la série StackMFF. Vous pouvez les utiliser au sein d'OpenFocus pour vos propres images et votre travail interne. Copier, porter ou migrer ces algorithmes vers un autre projet open source nécessite une autorisation écrite préalable.

<a id="developing"></a>
## Développement

Exécution depuis un clone :

```bash
conda create -n openfocus python=3.10
conda activate openfocus
pip install -r requirements.txt
python main.py
```

Empaquetage avec PyInstaller : [docs/BUILD_COMMANDS.md](../BUILD_COMMANDS.md).

Ajouter un algorithme de fusion ou de recalage : [CONTRIBUTING.md](../../CONTRIBUTING.md).
La [spécification d'extensibilité](../ALGORITHM_EXTENSIBILITY_SPEC.md) définit les contrats partagés de recalage, de paramètres, d'exécution et de persistance.

<a id="star-history"></a>
## ⭐ Historique des étoiles

[![Star History Chart](https://api.star-history.com/svg?repos=Xinzhe99/OpenFocus&type=Date)](https://star-history.com/#Xinzhe99/OpenFocus&Date)
