<p align="center">
  <a href="https://github.com/Xinzhe99/OpenFocus">
    <img src="../../assets/OpenFocus.png" alt="OpenFocus" width="128">
  </a>
</p>

<h1 align="center">OpenFocus</h1>

<p align="center"><strong>Chaque pixel, depuis son image la plus nette.</strong></p>

<p align="center">
  Dix algorithmes de fusion, cinq options de recalage (dont aucune) et un espace de retouche lié à la source pour la macro, la microscopie et les piles industrielles. 16 bits de bout en bout, une CLI sans interface et une interface en douze langues.
</p>

<p align="center">
  <a href="https://github.com/Xinzhe99/OpenFocus/releases"><img src="https://img.shields.io/github/v/release/Xinzhe99/OpenFocus?label=release&color=2f81f7" alt="Version"></a>
  <a href="https://github.com/Xinzhe99/OpenFocus/stargazers"><img src="https://img.shields.io/github/stars/Xinzhe99/OpenFocus?color=e3b341" alt="Étoiles"></a>
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
> **v1.77.0, version candidate** — Arrière-plan automatique et sessions de retouche restaurables, aperçus JPEG/RAW à mémoire limitée et inspection des pixels d’origine, fusion CPU selon le budget RAM et état d’exécution GPU plus clair. Consultez le journal des modifications pour les entrées prises en charge et les limitations.

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

StackMFF V3 utilise l'IA pour combiner les parties nettes des images d'une pile.

[StackMFF V3 →](https://github.com/Xinzhe99/StackMFF-V3) · [Banc d'essai →](../../tools/synthetic_benchmark.py)

</td>
<td width="54%">

<a href="../../assets/readme/ai.mp4"><picture><source srcset="../../assets/readme/ai.gif" type="image/gif"><img src="../../assets/readme/ai_poster.jpg" alt="StackMFF V3 face au filtre guidé sur la même pile" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Comparaison par balayage

Faites glisser le séparateur pour comparer une image source au résultat fusionné.

[Manuel →](../USER_MANUAL_FR.md)

</td>
<td width="54%">

<a href="../../assets/readme/wipe.mp4"><picture><source srcset="../../assets/readme/wipe.gif" type="image/gif"><img src="../../assets/readme/wipe_poster.jpg" alt="Comparaison A/B entre une image source et le résultat fusionné" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Comparaison de méthodes

Comparez les méthodes côte à côte et réutilisez le résultat ou les paramètres de votre choix.

</td>
<td width="54%">

<a href="../../assets/readme/method-comparison.mp4"><picture><source srcset="../../assets/readme/method-comparison.gif" type="image/gif"><img src="../../assets/readme/method-comparison_poster.jpg" alt="Comparaison côte à côte de huit méthodes de fusion" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Carte de profondeur et 3D

Créez une carte de profondeur et exportez une surface 3D texturée.

</td>
<td width="54%">

<a href="../../assets/readme/depth.mp4"><picture><source srcset="../../assets/readme/depth.gif" type="image/gif"><img src="../../assets/readme/depth_poster.jpg" alt="Carte de profondeur et surface 3D de la même pile" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Retouche liée à la source

Retouchez le résultat en peignant à partir des images sources et exportez-le avec celles-ci en TIFF multicouche. L'arrière-plan automatique remplace un arrière-plan empilé irrégulier par une vraie image.

</td>
<td width="54%">

<a href="../../assets/readme/retouch.mp4"><picture><source srcset="../../assets/readme/retouch.gif" type="image/gif"><img src="../../assets/readme/retouch_poster.jpg" alt="Espace de retouche avec image source et résultat" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Traitement par lots

Traitez plusieurs piles à la fois, avec une fusion groupée facultative pour limiter la mémoire et l'espace temporaire.

</td>
<td width="54%">

<a href="../../assets/readme/batch.mp4"><picture><source srcset="../../assets/readme/batch.gif" type="image/gif"><img src="../../assets/readme/batch_poster.jpg" alt="Boîte de dialogue de traitement par lots" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Barres d'échelle de publication

Ajoutez des barres d'échelle calibrées à partir des métadonnées TIFF ou saisies manuellement.

[Manuel →](../USER_MANUAL_FR.md)

</td>
<td width="54%">

<a href="../../assets/readme/scalebar.mp4"><picture><source srcset="../../assets/readme/scalebar.gif" type="image/gif"><img src="../../assets/readme/scalebar_poster.jpg" alt="Boîte de dialogue de barre d'échelle avec aperçu" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Légendes et libellés

Ajoutez des légendes et libellés avec une portée, une plage d'images et un style définis.

</td>
<td width="54%">

<a href="../../assets/readme/labels.mp4"><picture><source srcset="../../assets/readme/labels.gif" type="image/gif"><img src="../../assets/readme/labels_poster.jpg" alt="Gestionnaire de libellés avec aperçu" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Pilotable par un agent

Un agent peut piloter l'application via un canal local protégé par jeton.

[Protocole →](../AGENT_CONTROL_SPEC.md) · [Skill →](../../skills/openfocus-control/SKILL.md)

</td>
<td width="54%">

<a href="../../assets/readme/agent.mp4"><picture><source srcset="../../assets/readme/agent.gif" type="image/gif"><img src="../../assets/readme/agent_poster.jpg" alt="Un agent pilote l'interface en direct" width="100%" /></picture></a>

</td>
</tr>
</table>

**Également inclus :**

- **Focus stacking** — navigateur de piles, courbe de netteté, aperçu rapide, rendu ROI et réglages mémorisés.
- **Microscopie** — 16 bits, calibration TIFF, correction de champ plat et piles Z TIFF multipage.
- **Profils colorimétriques** — conservation du profil ICC RVB commun dans les exports individuels compatibles.
- **PNG transparent** — fusion des piles aux masques rectangulaires compatibles ; restrictions dans le manuel.
- **CLI sans interface** — lots, `--dry-run`, `--resume` et sortie JSON sous `openfocus`.
- **Projets et récupération** — sessions `.ofproj`, récupération automatique et annuler/rétablir.
- **Douze langues** — détection automatique et ajout par fichier `locales/<code>.py`.
- **Mémoire** — le réseau neuronal est chargé uniquement pour la méthode IA.

<a id="install"></a>
## ⬇️ Installation

Téléchargez la dernière version Windows ou macOS depuis le [site web d'OpenFocus](https://openfocus.top/#download). Les mêmes fichiers et les versions antérieures sont aussi disponibles sur [GitHub Releases](https://github.com/Xinzhe99/OpenFocus/releases).

Consultez le [manuel](../USER_MANUAL_FR.md) pour l'installation et le dépannage.

Sous macOS Apple Silicon ou Linux, installez depuis le terminal ; sous Linux, seule la CLI sans interface est installée et, sous macOS, l'application est placée dans `~/Applications` :

```bash
curl -fsSL https://openfocus.top/install.sh | sh -s -- --dry-run # facultatif : aperçu
curl -fsSL https://openfocus.top/install.sh | sh
```

Sous Windows, lancez l'installateur depuis PowerShell :

```powershell
irm https://openfocus.top/install.ps1 | iex
```

Téléchargement direct du programme d'installation Windows :

```powershell
curl.exe -fL https://openfocus.top/download/latest/windows-setup -o OpenFocus-setup.exe
```

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

Consultez le [manuel de la CLI](../USER_MANUAL_FR.md#14-interface-en-ligne-de-commande).

<a id="community"></a>
## Communauté et support

Des questions, idées ou commentaires ? Rejoignez notre [communauté Discord](https://discord.gg/afzmwgPt). Nous répondons dès que possible ; nous utilisons l'heure de Pékin (UTC+8), et le décalage horaire peut retarder nos réponses.

- **Documentation** — [Manuel](../USER_MANUAL_FR.md) (douze langues) · [Journal](../../CHANGELOG.md) · [Performances](../PERFORMANCE.md) · [Méthodes de fusion](../FUSION_METHODS.md)
- **Citation** — [CITATION.cff](../../CITATION.cff) ; chaque version est archivée sur Zenodo ([DOI](https://doi.org/10.5281/zenodo.23050823))
- **Retours** — ouvrez un [issue](https://github.com/Xinzhe99/OpenFocus/issues) pour les bogues, les idées et le travail sur les algorithmes
- **Crédits** — chaque algorithme de fusion et de recalage conserve sa licence de recherche d'origine ; StackMFF V3 se trouve sur [Xinzhe99/StackMFF-V3](https://github.com/Xinzhe99/StackMFF-V3)

<a id="license"></a>
## Licence

L'application OpenFocus est publiée sous [licence MIT](../../LICENSE).

**Licences StackMFF.** Les nouveaux poids et wrappers V1/V2 sont soumis à StackMFF V3 License 1.0. Leur copie, distribution, intégration à d'autres projets ou utilisation commerciale nécessite l'autorisation écrite préalable du titulaire des droits. L'instantané v2.0 original de StackMFF V3 intégré au projet reste sous licence MIT. Voir les [avis relatifs aux tiers](../../docs/THIRD_PARTY_NOTICES.md).

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
