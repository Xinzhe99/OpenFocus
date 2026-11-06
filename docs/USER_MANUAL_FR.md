# Manuel d'utilisation d'OpenFocus

## Table des matières

1. [Introduction](#1-introduction)
2. [Prise en main](#2-prise-en-main)
3. [Vue d'ensemble de l'interface principale](#3-vue-densemble-de-linterface-principale)
4. [Importer des images et des vidéos](#4-importer-des-images-et-des-vidéos)
5. [Affichage et navigation dans les images](#5-affichage-et-navigation-dans-les-images)
6. [Vue de comparaison par balayage](#6-vue-de-comparaison-par-balayage)
7. [Recalage d'images](#7-recalage-dimages)
8. [Méthodes de fusion d'images](#8-méthodes-de-fusion-dimages)
9. [Transformations d'images](#9-transformations-dimages)
10. [Retouche, libellés et annotations](#10-retouche-libellés-et-annotations)
11. [Paramètres et configuration](#11-paramètres-et-configuration)
12. [Traitement par lots](#12-traitement-par-lots)
13. [Exportation des résultats](#13-exportation-des-résultats)
14. [Interface en ligne de commande](#14-interface-en-ligne-de-commande)
15. [Raccourcis clavier](#15-raccourcis-clavier)
16. [Dépannage](#16-dépannage)

---

## 1. Introduction

OpenFocus est une application de bureau professionnelle de fusion d'images multifocus, conçue pour les chercheurs, les ingénieurs et les professionnels du traitement d'images. Elle traite des séquences d'images prises à différents plans de mise au point grâce à des algorithmes avancés de recalage et de fusion, et produit des images entièrement nettes avec une profondeur de champ totale.

### Fonctionnalités principales

- **Fusion d'images multifocus** : combine plusieurs images mises au point à des distances différentes en une seule image entièrement nette
- **Dix algorithmes de fusion** : filtre guidé, DCT, DTCWT, GFG-FGF, StackMFF V1/V2/V3, OpenFocus A/B/C, plus le mode alignement seul
- **Recalage d'images** : aligne les séquences désalignées avec OpenFocus (rapide), ECC, l'homographie ou l'enchaînement homographie → ECC
- **Arrière-plan automatique** : remplacer un arrière-plan empilé irrégulier par une vraie image et voir où aucune image n'a vu la scène nettement
- **Vue de comparaison par balayage** : vérifie l'alignement et compare les résultats avec un séparateur A/B
- **Fichiers de projet** : restaurent les références d'entrée et les réglages reproductibles ; les pixels du résultat s'exportent séparément
- **Modèles de traitement** : enregistrent et réappliquent les réglages de fusion, de recalage et de ressources sur toute pile chargée
- **TIFF multipage** : les piles Z d'un seul fichier se chargent page par page et se réexportent dans un seul fichier
- **Carte de profondeur** : cartes de position de mise au point par pixel avec des palettes de pseudocouleurs personnalisées
- **Barre d'échelle** : barres d'échelle de qualité publication avec calibration automatique en µm/px
- **Récupération automatique après plantage** : la dernière session de travail est proposée au redémarrage
- **Multilingue** : douze langues d'interface — English, 中文, 日本語, Español, Deutsch, Français, Русский, 한국어, Português, Italiano, Nederlands, Polski — avec détection automatique de la langue du système
- **CLI sans interface et pour agents** : scriptez des pipelines headless pour la CI ou pilotez une fenêtre en cours d'exécution via le canal de contrôle local
- **Export flexible** : enregistrez les résultats en images individuelles, dossiers ou animations GIF
- **Transformations d'images** : rotation, retournement et redimensionnement des piles
- **Libellés** : ajoutez du texte à la pile source, à la pile alignée ou à des sorties précises
- **Réglages persistants** : vos préférences, la langue et les piles récemment ouvertes survivent aux redémarrages

### Formats d'entrée pris en charge

- Fichiers image : JPG/JPEG, PNG, BMP, TIFF, WebP et autres formats courants
- Fichiers vidéo : MP4, AVI, MOV, MKV, WMV, FLV et WEBM (décodés en séquences d'images)
- Dossiers contenant des séquences d'images numérotées
- Fichiers RAW d'appareil photo (DNG, CR2/CR3, NEF, ARW, RAF, ORF, RW2 et autres formats RAW courants). Le décodage utilise `rawpy` (LibRaw), que `pip install openfocus` installe et que les versions Windows/macOS embarquent

Les piles PNG transparentes peuvent être fusionnées dans l'application de bureau, le traitement par lots de bureau ou l'interface en ligne de commande, sans fusion groupée. Chaque image doit être un PNG 8 ou 16 bits de mêmes dimensions et de même profondeur de bits, avec le même masque alpha. Le masque doit être nul en dehors d'un rectangle non vide unique et présenter une opacité constante non nulle dans tout le rectangle. Une opacité constante partiellement transparente est autorisée. Le rectangle visible est fusionné indépendamment : les couleurs masquées ne peuvent donc pas influencer la mesure de netteté ni les limites des tuiles. La sortie PNG conserve la taille d'origine du canevas et l'opacité ; les couleurs masquées de la sortie sont nulles. Les vues source et résultat du bureau affichent la transparence sur un damier.

Chargez en taille réelle, choisissez Aucun alignement, désactivez l'aperçu rapide et la ROI, puis rendez et enregistrez en PNG. Le mode par lots exige une méthode de fusion et une sortie PNG sans enregistrer de pile alignée. L'alignement, le redimensionnement ou la réduction, la correction de champ plat, les cartes de profondeur, la comparaison de méthodes, la retouche et les barres d'échelle ne sont pas pris en charge pour les données transparentes, et les libellés ne peuvent pas être incrustés dans un export. Une méthode de fusion qui modifie les dimensions du rectangle est refusée. Les masques variables ou à bords adoucis, les masques différents entre images, l'entrée TIFF transparente et les sorties non PNG, y compris GIF et TIFF multipage, ne sont pas pris en charge. La CLI refuse aussi le nettoyage du capteur et les cartes de profondeur pour une entrée transparente.

Par exemple, l'équivalent en CLI est `python main.py --input ./png-stack --output ./result.png --method guided_filter --align none`.

### Formats de sortie pris en charge

Dans **Enregistrer sous**, le TIFF ordinaire enregistre le résultat comme une seule image ; le TIFF multicouche inclut les images sources comme calques. Linear DNG est toujours affiché, mais son enregistrement exige un résultat linéaire appareil sans modifications incorporées, issu de **Ouvrir un RAW natif** et doté de la calibration de l’appareil.

- Images individuelles : JPG, PNG, BMP, TIFF, WebP
- Séquences d'images : enregistrées sous forme de dossiers
- Animations : format GIF
- Modèles 3D : OBJ avec MTL, texture PNG et métadonnées JSON

---

## 2. Prise en main

### Lancer l'application

La façon préférée d'installer une version packagée est le site du projet, **https://openfocus.top** : il pointe vers l'installateur Windows actuel, la version macOS et le paquet Linux, avec des scripts d'installation pour le terminal. Préférez-le à une recompilation : ces artefacts sont la sortie exacte du pipeline de publication. `pip install openfocus` installe uniquement l'interface en ligne de commande headless (le paquet wheel ne dépend volontairement pas de PyQt6) ; utilisez une version de bureau ou une copie du code source pour l'interface graphique.

Pour installer depuis un terminal, prévisualisez d'abord la commande puis exécutez-la sans `--dry-run` : sous macOS ou Linux, `curl -fsSL https://openfocus.top/install.sh | sh -s -- --dry-run` puis `curl -fsSL https://openfocus.top/install.sh | sh` (la CLI est installée sous Linux) ; sous Windows, `irm https://openfocus.top/install.ps1 | iex` dans PowerShell.

Après l'installation, lancez OpenFocus de l'une des façons suivantes :

1. **Windows (exécutable)** : double-cliquez sur le raccourci ou l'exécutable OpenFocus
2. **macOS** : glissez `OpenFocus.app` dans Applications et ouvrez-le depuis ce dossier
3. **Source Python** : exécutez `pip install -r requirements.txt`, puis `python main.py` dans votre environnement activé

Le premier lancement ouvre la boîte de bienvenue, où vous pouvez choisir le thème et charger la pile de démonstration fournie. Un profil neuf utilise le thème clair par défaut ; le choix est mémorisé.

### Flux de travail de démarrage rapide

1. Importez votre séquence d'images (Fichier → Ouvrir un dossier ou glisser-déposer)
2. Configurez les réglages de fusion dans le panneau de droite
3. Le recalage démarre sur **ECC** ; choisissez **Aucun alignement** si les images sont déjà alignées
4. Ajustez la taille du noyau si nécessaire
5. *(facultatif)* Ouvrez **Paramètres de rendu** et cochez **Aperçu rapide** pour tester d'abord les paramètres sur un brouillon réduit
6. Cliquez sur **Lancer le rendu** pour lancer le traitement
7. Exportez les résultats

---

## 3. Vue d'ensemble de l'interface principale

L'interface d'OpenFocus comporte trois zones principales :

### Panneau gauche : zone d'affichage des images

La partie gauche de la fenêtre affiche vos images en vue divisée :

- **Panneau image source (haut)** : affiche la séquence d'images d'entrée
- **Panneau image de résultat (bas)** : affiche les résultats de fusion ou de recalage
- **Curseur de navigation** : utilisez le curseur sous chaque panneau pour parcourir les images
- **Libellés d'information** : indiquent la position de l'image courante et ses dimensions

Vous pouvez redimensionner les panneaux en faisant glisser le séparateur qui les sépare.

### Panneau droit : panneau de configuration

Le panneau droit regroupe toutes les options de configuration, de haut en bas :

1. **Réglages de fusion** : choix de l'algorithme et de ses paramètres
2. **Réglages de recalage** : activation ou désactivation des méthodes d'alignement
3. **Boutons d'action** : lancer le rendu, réinitialiser, ouvrir les paramètres de rendu et lancer toutes les méthodes. La boîte **Paramètres de rendu** contient l'aperçu rapide, l'ordre des images, l'égalisation de l'exposition, la suppression de la dérive de luminosité et la netteté de sortie
4. **Liste des images sources** : toutes les images chargées avec leur nom de fichier
5. **Liste des sorties** : les résultats générés

### Barre de menus

La barre de menus supérieure donne accès à toutes les fonctions :

- **Fichier** : importation et exportation, projets, modèles de traitement et liste des fichiers récents
- **Édition** : transformations d'images, retouche et libellés
- **Outils** : traitement par lots, comparaison de méthodes, carte de profondeur, correction de champ plat, barre d'échelle et libellés
- **Paramètres** : paramètres avancés (performances, recalage et diagnostics d'alignement, nettoyage du capteur, nommage des sorties, IA et démarrage), accélération GPU, thème et langue
- **Aide** : informations sur l'environnement, contact, recherche de mises à jour et dossier des journaux

---

## 4. Importer des images et des vidéos

### Ouvrir un dossier d'images

1. Allez dans **Fichier → Ouvrir un dossier** ou appuyez sur `Ctrl+O`
2. Sélectionnez un dossier contenant votre séquence d'images
3. Une boîte de dialogue d'échelle de chargement apparaît : choisissez l'échelle (100 % = résolution complète)
4. Cliquez sur OK pour charger les images

Cochez **Ne plus demander** dans cette boîte pour toujours charger à l'échelle choisie sans invite ; **Paramètres → Paramètres avancés → Demander l'échelle de chargement à l'importation** réactive l'invite. Une pile déjà chargée peut encore être réduite avec **Édition → Redimensionner**.

Le logiciel effectue automatiquement les opérations suivantes :

- Détection et tri des fichiers image par nom, dans un ordre naturel (par exemple `frame2` avant `frame10`)
- Affichage de la première image dans le panneau source
- Remplissage de la liste des images sources avec toutes les images

### Ouvrir des fichiers vidéo

1. Allez dans **Fichier → Ouvrir une vidéo** ou appuyez sur `Ctrl+Shift+O`
2. Sélectionnez un fichier vidéo (MP4, AVI, MOV, MKV, WMV, FLV, WEBM)
3. La vidéo est décodée automatiquement en images individuelles
4. Chaque image devient un élément de la pile

### Fichiers de projet

Fichier → Enregistrer le projet stocke les références d'entrée et les réglages reproductibles dans un `.ofproj` : ordre exact des images (y compris les pages TIFF et les doublons), exclusions, rotations et retournements, échelle de chargement, paramètres de rendu par méthode, libellés, style de barre d'échelle et image courante. Les pixels du résultat et de la retouche doivent être exportés séparément. Les opérations non reconstructibles à partir des fichiers sources exigent d'enregistrer d'abord une nouvelle pile d'entrée.

Fichier → Ouvrir le projet reconstruit les images enregistrées avant de remplacer la pile actuelle. Si les sources ont été déplacées, choisissez la nouvelle racine source et confirmez la correspondance complète des anciens et nouveaux chemins. Enregistrez à nouveau le projet pour conserver les références déplacées.

Un projet se comporte comme un document : la barre de titre affiche son nom avec un `*` tant que des modifications ne sont pas enregistrées, et fermer l'application ou ouvrir un autre projet demande confirmation avant de les abandonner. Les images sources sont référencées, pas copiées.

### Modèles de traitement

Un modèle de traitement contient des réglages, jamais de chemins source. **Fichier → Enregistrer un modèle de traitement…** écrit un `.oftemplate` avec la méthode et les paramètres de fusion actuels, la méthode et les paramètres de recalage, les options de luminosité par méthode, le nombre de threads, les réglages de tuiles, la largeur de réduction du recalage, la taille de lot IA de StackMFF, l'accentuation, l'ordre inversé et l'indicateur de réduction du bruit du capteur. Le chemin d'un masque de défauts n'est délibérément pas stocké.

**Fichier → Appliquer un modèle de traitement…** applique ces réglages à la pile déjà chargée ; il ne remplace pas les sources actuelles. Le sélecteur accepte aussi un projet `.ofproj` (les réglages sont extraits sans nécessiter ses sources) et une recette `.json` contenant des réglages de projet. Appliquer un modèle marque le projet comme modifié et peut changer les paramètres des algorithmes et les réglages de ressources ; vérifiez donc le panneau de droite avant de rendre. Les recettes restent le format headless et incluent un chemin de masque de défauts ainsi qu'une empreinte.

### Pile de démonstration

La boîte de dialogue de bienvenue propose **Charger la pile de démonstration** : six photos d'exemple fournies, issues d'une vraie pile de mise au point, pour essayer immédiatement le flux complet sans images personnelles. Utile aussi pour vérifier une installation.

### Courbe de netteté

Une petite courbe au-dessus du curseur de navigation source montre la netteté de chaque image (normalisée). Les images floues apparaissent comme des creux marqués d'un point rouge. Cliquez n'importe où sur la courbe pour aller à cette image ; survolez-la pour afficher son numéro.

### Piles TIFF multipage

Les logiciels de microscopie (ZEN, NIS-Elements, ImageJ/Fiji) exportent souvent toute une pile Z dans UN seul fichier TIFF multipage. OpenFocus lit chaque page de ce fichier comme une image individuelle de la pile : déposez ou ouvrez le fichier et la pile apparaît complète (les images sont nommées `<file>_page_001`, etc.). Les TIFF d'une seule page se comportent exactement comme avant. L'inverse fonctionne aussi : **Fichier → Enregistrer la pile → (Pile recalée / Pile d'entrée) → Enregistrer en TIFF multipage** écrit toute la pile dans un seul `.tif`, en conservant la profondeur 16 bits.

### Barre d'échelle (figures de publication)

Si la cible choisie ne possède pas de métadonnées utilisables, la calibration manuelle est présélectionnée et attend une taille de pixel valide.

Outils → Barre d'échelle… configure une barre d'échelle intégrée aux images exportées. La taille du pixel (µm/px) est détectée automatiquement dans les métadonnées TIFF (convention `unit=micron` d'ImageJ, `PhysicalSizeX` d'OME-TIFF ou unité de résolution explicite en centimètres), de sorte que les piles de microscopie (y compris multipage) se calibrent seules. La résolution d'imprimante (72/300 dpi et autres) n'est délibérément *pas* traitée comme une calibration, et une valeur saisie manuellement l'emporte toujours sur les métadonnées détectées : une barre fausse mais plausible dans une figure est pire que pas de barre. Choisissez le coin et la couleur (Auto inverse selon le fond) ; la boîte de dialogue affiche un aperçu en direct. Une fois la barre activée, chaque export (image seule, piles en dossiers, TIFF multipage, GIF, tout exporter, lots) porte une barre de style publication : une longueur arrondie de la série 1-2-5 (~12 % de la largeur de l'image) avec un texte contourné et lisible. La CLI l'accepte via `--scale-bar` (calibration automatique) et `--scale-um <um/px>`.

L'aperçu est dessiné à la géométrie native en pixels de la cible avant d'être ajusté à la zone visible. Ajuster, 100 %, le déplacement et la navigation entre images ne modifient que la vue. Appliquer valide une seule modification annulable ; Annuler conserve la calibration précédente. Les sorties historiques gardent leur propre calibration et leur style lorsqu'une autre pile source est chargée.

Pour la retirer, utilisez **Outils → Barre d'échelle → Supprimer la barre d'échelle** : *De la pile source* et *Du résultat* limitent la portée « Appliquer à » au canevas restant, tandis que *De partout* désactive la barre. Les deux canevas et le balayage A/B sont redessinés immédiatement ; retirer la barre d'un canevas qui ne l'a jamais affichée laisse le réglage inchangé.

### Carte de profondeur (position de mise au point)

**Outils → Carte de profondeur…** estime les positions de mise au point à partir de la mesure d'activité de la méthode sélectionnée. Ce n'est ni une surface mesurée ni une carte exacte de l'origine des pixels : les méthodes pyramidales/ondelettes estiment l'activité et les variantes StackMFF indiquent une position d'image estimée. Choisissez les images cochées actuelles ou la source enregistrée de la sortie sélectionnée. Une provenance historique manquante n'est jamais remplacée par la pile actuelle. La source et la texture restent liées à cet instantané. Les cartes de profondeur ne sont pas disponibles pour les piles PNG transparentes, ni dans l'interface ni dans la CLI.

Les sept méthodes compatibles avec la profondeur — DCT, DTCWT, GFG-FGF, OpenFocus B, OpenFocus C, StackMFF V2 et StackMFF V3 — exigent un clic explicite sur **Calculer**. StackMFF V1 n'a pas de carte de profondeur. Ouvrir la boîte de dialogue ou changer la mesure, la source, le recalage ou les paramètres ne lance jamais le calcul. L'éditeur de paramètres apparaît juste sous le sélecteur de mesure : le noyau médian DCT et le noyau initial GFG utilisent des valeurs impaires de 1 à 51 (7 par défaut). OpenFocus B utilise un rayon de réponse de mise au point et un lissage de profondeur ; OpenFocus C utilise **Lissage du mélange** pour lisser sa carte de profondeur, sans rayon. Les niveaux de décomposition DTCWT se modifient ici comme brouillon local, initialisé depuis les paramètres de la source. StackMFF V3 affiche son propre mode de sortie IA (mélange doux par défaut, ou sélection stricte) et une explication au lieu d'un noyau numérique : la carte indique une position d'image continue, pas une distance physique. Le mode doux utilise la position pondérée par les probabilités ; le mode strict, l'image la plus probable par pixel. StackMFF V2 n'a aucun paramètre et indique la couche focale échantillonnée par son réseau. Les brouillons partent des paramètres de la sortie correspondante ou des réglages mémorisés de la méthode dans la fenêtre principale. Ils sont mémorisés par source et par méthode tant que la boîte est ouverte ; les modifier ne change ni la fenêtre principale, ni le projet, ni la sortie enregistrée.

Changer les réglages de calcul marque la carte 2D précédente comme **non à jour**, vide la surface 3D et bloque tous les exports, y compris colorisés, bruts, physiques et OBJ, jusqu'à la réussite d'un nouveau calcul. Un calcul actif est annulé de façon coopérative ; une fois l'annulation terminée, cliquez de nouveau sur **Calculer**. Aucun calcul suivant n'est mis en file. Calculer reste visible et est désactivé pendant le calcul, l'annulation ou l'export.

- **Palette** : niveaux de gris (standard de publication), 15 palettes scientifiques (Turbo, Viridis, Magma, …) et les vôtres — *Nouveau schéma personnalisé…* construit un dégradé à partir d'un nombre quelconque d'arrêts de couleur (double-cliquez sur un arrêt pour modifier sa position et sa couleur) ; les schémas personnalisés sont enregistrés et proposés dans la liste.
- **Inverser / Gamma / Lissage** : inverse l'axe de profondeur, rééquilibre la tonalité (gamma < 1 éclaircit les indices bas) et supprime le grain avec une passe respectueuse des contours.
- **Superposition** : disponible uniquement pour la sortie associée avec des coordonnées correspondantes.
- **Export brut** : le TIFF 16 bits stocke un indice normalisé codé de 0 à 65535 ; ce ne sont pas des données flottantes sans perte ni une distance physique. JPEG/BMP sont refusés.
- **TIFF d'indices et TIFF d'image alternative** : exporte l'estimation de profondeur actuelle sous forme d'indices d'image 16 bits non signés ou d'image assemblée à partir des pixels sources. Les valeurs 1–N renvoient à l'ordre exact des images utilisé en entrée du calcul de profondeur, après recalage et réordonnancement ; 0 marque les pixels invalides. 65 535 images au maximum sont prises en charge. Les positions fractionnaires estimées sont arrondies à l'image la plus proche, les valeurs intermédiaires étant arrondies au supérieur. Ces indices estiment la position de mise au point ; ce n'est pas la vraie provenance des pixels d'un résultat fusionné. L'image alternative sélectionne le pixel de l'image d'entrée de profondeur correspondante à chaque emplacement valide, les pixels invalides étant mis à zéro. Lorsqu'une entrée est en 16 bits, les valeurs 8 bits sont promues en 16 bits en les multipliant par 257. Les exports utilisent la ROI de la carte de profondeur et conservent les zones invalides à zéro.
- **Profondeur physique** : saisissez le début Z, le pas positif, l'unité et le sens d'acquisition, puis confirmez la correspondance de capture d'origine. Passer de la pile actuelle à une sortie efface la confirmation, même si les noms et les indices correspondent ; reconfirmez-la avant d'exporter. Les images exclues conservent leurs écarts. Le TIFF Float32 stocke des micromètres, NaN pour les données manquantes et les métadonnées source. Un OBJ physique nécessite aussi une calibration XY ; il utilise une échelle Z physique 1:1.
- **Modèle 3D / Texture 3D** : choisissez le côté le plus long de la grille (600 sommets par défaut). Chaque nouvelle boîte de dialogue utilise **Automatique (couleurs réelles)** : la sortie associée n'est utilisée que si son identité et ses coordonnées/ROI correspondent à la carte terminée ; sinon, la première image participante de ce calcul terminé est utilisée, après recalage et réordonnancement éventuel. Le nom de la sortie ou de l'image résolue est affiché. Vous pouvez choisir explicitement la sortie correspondante, une image participante ou **Pseudocouleur de profondeur** ; les images indisponibles sont signalées, jamais remplacées par des sorties sans rapport. Cette texture est partagée par l'aperçu 3D et le PNG de texture de l'OBJ. La carte 2D garde sa propre palette et superposition : changer sa palette ou son Gamma ne recolore pas une texture en couleurs réelles. La pseudocouleur explicite suit la palette, le Gamma et l'inversion. L'aperçu stabilisé et l'export partagent le même champ de hauteur échantillonné, sans lissage caché supplémentaire. Le relief relatif est exprimé en unités d'image.
- **Export du modèle** : OBJ, MTL, texture PNG et métadonnées JSON forment un groupe de fichiers ; conservez-les ensemble. L'export s'exécute en arrière-plan. Les conflits sont vérifiés pour chaque membre, la préparation est mise en attente, l'annulation conserve les fichiers existants et un échec de validation annule les remplacements. Enregistrer un modèle n'exporte pas son image de fusion parente et ne supprime pas la protection des modifications non enregistrées de cette image.

Le calcul s'exécute en arrière-plan avec progression et annulation ; les méthodes IA réutilisent le même traitement par tuiles économe en mémoire que la fusion IA, ce qui permet de traiter les piles profondes. L'équivalent CLI est `--depth-map [path]` avec `--depth-colormap` ; il n'écrit que la carte colorisée (par défaut `<output>_depth.png`), ne fonctionne que pour les méthodes compatibles avec la profondeur ci-dessus et est refusé pour une entrée transparente ou groupée. L'interface exporte en plus la profondeur brute, les indices d'image, les images alternatives, la profondeur physique et les modèles 3D.

### Récupération automatique après plantage

OpenFocus conserve en continu un instantané de votre session de travail (pile, réglages, libellés, image courante). Si l'application se ferme de façon inattendue — plantage, coupure de courant ou arrêt forcé —, le lancement suivant le détecte et propose **Restaurer la session** pour revenir exactement où vous étiez. Une fermeture propre n'affiche jamais l'invite. Les données de récupération se trouvent à côté de vos réglages (les installations portables les gardent près de l'exécutable, pour qu'elles vous suivent).

### Fichiers récents

Chaque pile ouverte est mémorisée dans **Fichier → Ouvrir récent** (jusqu'à 8 entrées, numérotées dans le sous-menu). Cliquez sur une entrée pour la rouvrir ; il n'existe pas de raccourci clavier global pour les fichiers récents. Rouvrir la même pile la place en tête ; les entrées dont les fichiers n'existent plus peuvent être retirées après confirmation, et **Effacer la liste des récents** vide le menu.

### Glisser-déposer

Vous pouvez aussi glisser un dossier directement sur la fenêtre de l'application pour importer des images. Déposer d'autres fichiers sur une pile existante les ajoute à la fin (après vérification de la taille).

### Orientation EXIF

Les fichiers JPEG/TIFF d'appareil photo portant une balise EXIF Orientation sont automatiquement pivotés ou retournés à l'importation, afin que les prises de vue verticales s'affichent droites.

### Sources 16 bits

Les entrées PNG/TIFF 16 bits sont détectées automatiquement et conservées de bout en bout : la fusion s'exécute en pleine profondeur et les résultats peuvent être réenregistrés en PNG/TIFF 16 bits (les exports JPG/BMP sont convertis en 8 bits, ces formats ne pouvant pas stocker 16 bits).

Les fichiers RAW d'appareil photo sont développés en 16 bits (balance des blancs de l'appareil, gamma sRGB) avec l'éclaircissement automatique désactivé, afin que chaque image de la pile reçoive le même mappage d'exposition ; une exposition automatique par image ferait scintiller la luminosité entre les images. Enregistrez en TIFF ou PNG pour conserver les données 16 bits.

### Profils colorimétriques

Les exports PNG, TIFF, JPEG et WebP d'une image seule conservent le profil ICC RVB lorsque toutes les images sources ont le même profil intégré. Les piles mixtes ou sans profil sont exportées sans profil ICC. OpenFocus conserve le profil ; il ne convertit pas les pixels sources vers un espace colorimétrique commun et ne gère pas la couleur de l'aperçu. Convertissez les images à profils différents vers un espace commun avant de les charger. L'entrée RAW est développée en sRGB et n'hérite pas du profil de l'appareil.

Pour un TIFF multipage, toutes les pages du fichier doivent avoir le même profil RVB valide ; sinon, leurs images sont traitées comme sans profil. Les images vidéo sont également traitées comme sans profil. Les images ajoutées et celles sélectionnées pour un rendu contribuent à son profil, qui reste attaché à ce résultat pour les exports ultérieurs.

### Vider la pile

Pour retirer toutes les images chargées, allez dans **Fichier → Vider la pile** ou appuyez sur `Ctrl+W`.

---

## 5. Affichage et navigation dans les images

### Parcourir la pile d'images

- **Curseur** : faites glisser le curseur sous le panneau source pour parcourir les images
- **Liste de fichiers** : cliquez sur un nom de fichier dans la liste des images sources du panneau droit ; lorsque la liste a le focus, Haut/Bas déplacent la sélection et le panneau source la suit
- **Clavier** : il n'existe pas de raccourci global image précédente/suivante ; donnez d'abord le focus à la liste source

### Examiner les images similaires

Cliquez sur **Examiner les images** sous l'en-tête de la liste source pour trouver des images adjacentes dont les pixels sont presque identiques. Le bouton devient **Annuler l'examen** pendant l'analyse. Chaque ligne suggérée indique son image de référence et la différence ; inspectez les deux avant d'utiliser la case source pour exclure une image de la fusion. Les suggestions ne modifient jamais les cases ni ne suppriment de fichiers. Cette comparaison prudente peut manquer des images similaires ; les changements limités à une petite région nette ne sont pas moyennés. Les résultats sont abandonnés si la pile change. Les suggestions ne s'affichent que si l'image suggérée et sa référence sont toutes deux incluses ; modifier une case actualise les suggestions affichées.

### Zoom et déplacement

- **Zoom avant/arrière** : molette de la souris sur l'image
- **Zoom fin** : maintenez `Ctrl` et utilisez la molette pour des incréments plus petits
- **Ajuster à la fenêtre** : double-cliquez sur l'image pour basculer entre ajustement et vue à 100 %
- **Déplacer** : cliquez-glissez pour déplacer l'image dans le panneau

### Navigation synchronisée

Lorsque vous naviguez dans le panneau source, le panneau de résultat affiche l'image de résultat correspondante si elle existe.

---

## 6. Vue de comparaison par balayage

La vue Balayage superpose deux images dans un même cadre avec un séparateur vertical déplaçable, ce qui rend directement visibles les désalignements et les différences de qualité de part et d'autre.

### Ouvrir la vue Balayage

1. Rendez au moins un résultat (fusion ou recalage)
2. Appuyez sur le commutateur **Balayage** en haut à droite de la barre de titre du panneau de résultat
3. La zone de résultat passe en vue de comparaison ; appuyez de nouveau sur **Balayage** pour revenir

### Lire la vue

- **Côté A (à gauche du séparateur)** : une image source — par défaut, elle suit le curseur du panneau source
- **Côté B (à droite du séparateur)** : le rendu le plus récent
- **Séparateur blanc + poignée ronde** : faites-le glisser horizontalement pour déplacer la division
- Déplacer les bords qui traversent le séparateur révèle des erreurs de recalage difficiles à voir côte à côte

### Choisir ce que l'on compare

La barre en bas de la vue de comparaison comporte un groupe par côté :

- **Côté A** : cochez **Suivre la source** (par défaut) pour suivre le curseur du panneau source, ou décochez pour fixer le côté A sur n'importe quelle image avec son propre curseur
- **Côté B** : cochez **Dernier résultat** (par défaut) pour toujours afficher le rendu le plus récent, ou décochez pour parcourir l'historique des sorties avec son curseur

Les deux côtés se mettent à jour en direct : nouveaux rendus, déplacements du curseur source et suppressions d'images sont reflétés immédiatement.

### Zoom et déplacement

La vue de comparaison partage une seule transformation pour les deux images :

- **Zoom** : molette (zoom autour du curseur)
- **Déplacer** : cliquez-glissez partout sauf sur le séparateur
- **Réinitialiser** : double-clic

Le mode Balayage et le mode ROI partagent le panneau de résultat : entrer dans l'un sort de l'autre. Vider la pile quitte aussi le mode Balayage.

---

## 7. Recalage d'images

Le recalage corrige le désalignement spatial entre les images de votre pile. Il est indispensable lorsque les images présentent de légers décalages ou des changements de perspective.

### Méthodes de recalage

#### ECC (Enhanced Correlation Coefficient)

- Utilise une optimisation pour trouver le meilleur alignement selon la corrélation
- Adapté aux désalignements subtils
- Précision inférieure au pixel
- Plus lent que l'homographie, mais plus précis

#### Homographie

- Repose sur la mise en correspondance de points caractéristiques (SIFT/ORB)
- Gère des transformations géométriques plus importantes
- Plus rapide que l'ECC pour un alignement initial
- Adaptée aux images présentant d'importants changements de perspective

#### OpenFocus (rapide)

- Alignement par similarité bornée (translation + rotation + échelle), avec égalisation de luminosité par image et recadrage automatique
- Recherche des transformations de similarité bornées par différences d'images échantillonnées et applique un warp par interpolation Lanczos
- Les images décalées de plus de **8 % de la taille de l'image restent non alignées** plutôt que partiellement corrigées. Modifiez la limite avec `--align-shift-limit` en CLI
- Pour une géométrie non rigide ou extrême, préférez ECC ou l'homographie
- **Gestion des images rejetées** : par défaut, le traitement continue avec l'image non alignée. Choisissez **Arrêter sur image rejetée** pour faire échouer le rendu ou l'élément de lot avant de produire un résultat lorsque la qualité du détecteur est inférieure à 0,06 ou que le mouvement estimé dépasse la limite de décalage configurée, 8° de rotation ou 5 % de changement d'échelle. Les images sont numérotées à partir de 1 dans l'ordre de la pile traitée. La référence géométrique est l'image 1 par défaut ; une correspondance directe faible peut être retentée sur l'image adjacente plus proche de la référence. Cette politique détecte les estimations rejetées, pas toutes les correspondances incorrectes : une texture faible, des motifs répétés, une occlusion, un mouvement important ou des changements de mise au point peuvent encore tromper le recalage. Inspectez le résultat.
- En CLI, sélectionnez la même politique avec `--align openfocus --align-param 'failure_policy="fail"'` (avec le quoting adapté à votre shell). Les éléments de lot appliquent la politique choisie indépendamment. La valeur par défaut est `"continue"`.
- Réglez **Image de référence** sur un nombre à base 1 dans l'ordre de la pile incluse et traitée pour ancrer la géométrie sur cette image. L'ordre est conservé ; l'égalisation de luminosité utilise sa propre référence. Chaque pile de lot doit contenir ce numéro, sinon l'élément échoue. Exemple CLI : `--align openfocus --align-param reference_frame=3`. Choisir une référence ne corrige ni la parallaxe ni une mauvaise correspondance ; inspectez le résultat.

Les rapports d'alignement identifient la référence et le résultat de chaque image. Un repli signifie que la méthode n'a pas pu établir l'alignement ou dépend d'une estimation précédente échouée ; la raison indiquée distingue les estimations faibles et le mouvement hors limites lorsqu'il est connu. Les valeurs de qualité sont propres à chaque méthode, sans garantie d'exactitude. Les transformations projettent les coordonnées sources sur le canevas de référence de l'étape de recalage avant le recadrage commun ; les méthodes composites rapportent leurs étapes séparément. L'homographie et l'ECC acceptent aussi le paramètre facultatif `failure_policy="fail"` pour s'arrêter lorsque leurs algorithmes détectent un échec. La valeur par défaut reste `"continue"` ; inspectez les résultats même lorsque l'alignement réussit.

Pour inspecter un rendu, activez **Paramètres → Paramètres avancés → Recalage → Collecter les diagnostics d'alignement**, puis relancez le rendu. L'alignement est recalculé au lieu de réutiliser les images en cache. Le statut de fin met en évidence les replis ; ouvrez **Paramètres → Paramètres avancés → Recalage → Diagnostics d'alignement** pour le rapport par image. Il indique les raisons d'échec disponibles, les scores propres à la méthode, le nombre de correspondances, les images référencées et les transformations. Les numéros et noms de référence renvoient à la liste source d'origine, même lorsque des images sont exclues. Les rendus par lots et groupés ne fournissent pas ce rapport.

### Activer le recalage

1. Dans le panneau droit, repérez le groupe **Recalage**
2. Choisissez une méthode dans la liste : **Aucun alignement**, **OpenFocus (rapide)**, **Aligner (ECC)**, **Aligner (homographie)** ou **Homographie → ECC** (enchaîné)
3. Choisissez **Gestion des images rejetées** dans **Paramètres → Paramètres avancés → Recalage** ; toute modification s'applique aux méthodes. Pour OpenFocus (rapide), choisissez aussi une image de référence.
4. **Collecter les diagnostics d'alignement** se trouve désormais dans Paramètres avancés (voir ci-dessus)

### Quand utiliser le recalage

- Vos images présentent un désalignement visible
- Les images ont été prises à main levée
- La scène présente de la parallaxe (les objets proches se déplacent entre les images)
- Les résultats montrent des artefacts de dédoublement ou de fantôme

### Paramètres de recalage

Les réglages supplémentaires se trouvent dans **Paramètres → Paramètres avancés → Recalage** :

- **Largeur de réduction** : résolution de prétraitement pour le recalage (1024 px par défaut)
- Valeurs plus faibles = traitement plus rapide, précision éventuellement moindre
- Valeurs plus élevées = traitement plus lent, précision éventuellement meilleure
- **Collecter les diagnostics d'alignement** et **Cache de recalage** se trouvent dans la même section

---

## 8. Méthodes de fusion d'images

OpenFocus propose dix algorithmes de fusion (plus un mode alignement seul). Un profil neuf démarre sur **StackMFF V3** avec le recalage **ECC** ; les listes ci-dessous décrivent tous les choix. Chacun présente des caractéristiques adaptées à différents types d'images.

### Filtre guidé

- **Algorithme** : filtrage guidé pour une fusion qui préserve les contours
- **Idéal pour** : usage général, bon compromis vitesse/qualité
- **Avantages** : bonne préservation des contours, traitement rapide
- **Paramètre** : taille du noyau (réglable de 1 à 51, valeurs impaires uniquement)
  - Noyau plus petit : plus de détail local, risque de bruit accru
  - Noyau plus grand : résultat plus lisse, meilleure suppression du bruit

### DCT (Discrete Cosine Transform)

- **Algorithme** : fusion dans le domaine fréquentiel via les coefficients DCT
- **Idéal pour** : images riches en textures, imagerie scientifique
- **Avantages** : conserve les textures fines, base mathématique solide
- **Paramètre** : la taille du noyau contrôle la fenêtre de traitement

### DTCWT (Dual-Tree Complex Wavelet Transform)

- **Algorithme** : fusion dans le domaine des ondelettes multi-échelles
- **Idéal pour** : scènes complexes à plusieurs niveaux de détail
- **Avantages** : excellente analyse multi-échelles, bonne directionalité
- **Règle de fusion** : une seule décision par pixel et par niveau pour toutes les orientations et canaux de couleur (N voies, cohérence majoritaire vérifiée) ; la bande passe-bas suit la décision la plus grossière au lieu de moyenner toutes les images
- **Paramètre** : niveaux d'ondelette (1–6, 4 par défaut) dans le panneau droit. La méthode n'utilise pas de noyau spatial, donc le curseur de noyau est masqué.

### GFG-FGF (gradient à quatre voisins de type gaussien + filtre guidé rapide)

- **Algorithme** : implémentation de Fu et al. (*Laser & Optoelectronics Progress*, 2024) : une carte de netteté par différence de flou |I − mean(I)| est enrichie par l'opérateur de gradient à quatre voisins de type gaussien (GFG), affinée par un filtre guidé rapide, décidée par pixel (les égalités sont tranchées par une mesure de petite région de Sobel 3×3), puis les cartes de décision binaires sont à nouveau lissées par le filtre guidé rapide avant le mélange pondéré.
- **Idéal pour** : microscopie et images à régions nettes bien définies
- **Avantages** : rapide, limites de régions nettes ; la mémoire ne croît pas avec le nombre d'images
- **Paramètre** : noyau = taille du filtre moyen de la carte de netteté grossière

### StackMFF (V1, apprentissage profond)

- **Algorithme** : un réseau d'attention à une seule échelle mélange la luminance des sources avec des poids softmax ; Cb/Cr proviennent de l'image source dont la luminance est la plus proche de la luminance fusionnée
- **Idéal pour** : fusion neuronale d'une pile de mise au point complète
- **Avantages** : sans paramètres, couleurs dans la gamme ; la luminance fusionnée est indépendante de l'ordre des images, mais la chrominance peut varier en cas d'égalité des écarts de luminance
- **Remarque** : environ 156 172 paramètres ; les grandes piles gagnent à utiliser le GPU

### StackMFF V2 (apprentissage profond)

- **Algorithme** : un réseau d'attention à une seule échelle normalise la profondeur de mise au point attendue et échantillonne l'image source la plus proche. Les indices de profondeur dépendent de l'ordre de la pile. **Les images doivent conserver leur ordre de mise au point d'origine** — l'inversion est désactivée pour cette méthode
- **Cartes de profondeur** : prises en charge ; indique la couche focale échantillonnée
- **Idéal pour** : balayages de mise au point continus où un modèle très petit et rapide compte
- **Prérequis** : ONNX Runtime et le modèle ONNX fourni.

### StackMFF V3 (apprentissage profond)

C'est la méthode présélectionnée dans une nouvelle installation.

- **Algorithme** : classification de la source pixel par pixel (ONNX Runtime). Le réseau prédit, pour chaque pixel, l'image d'entrée dont il provient, puis mélange les images selon ces probabilités ou en choisit une
- **Mode de sortie IA** (panneau droit, uniquement pour cette méthode) :
  - **Mélange doux (pondéré)** — par défaut ; chaque pixel est un mélange des images pondéré par les probabilités. Transitions douces, sans coutures dures ; correspond à la configuration publiée dans l'article
  - **Sélection stricte (image unique)** — chaque pixel est copié depuis l'image la plus probable. Sélection plus nette, sans interpolation entre images ; une couture au niveau du pixel peut apparaître là où deux images sont à égalité
- **Idéal pour** : qualité maximale lorsqu'un GPU est disponible
- **Avantages** : invariant à l'ordre (l'ordre des images ne change jamais le résultat), gère les piles selon les ressources disponibles, aucun réglage par méthode, optimisation automatique
- **Prérequis** : ONNX Runtime et le modèle ONNX fourni.

Chaque tuile neuronale utilise toutes les images. Les images et les données temporaires de recalage peuvent être stockées sur disque ; l’espace libre et la mémoire disponible limitent la pile. Le pavage spatial V2 est une approximation de l’inférence sur l’image entière.


### OpenFocus A / B / C

- **OpenFocus A** : lisse sa carte de réponse de netteté multi-échelles, puis sélectionne les pixels nets.
- **OpenFocus B** : mélange les images voisines guidé par une carte de profondeur lissée.
- **OpenFocus C** : reconstruit les bandes multi-échelles guidé par une carte de profondeur lissée.
- **Paramètres** : A et B utilisent le curseur de noyau pour régler l'échelle de réponse de netteté ; C n'a pas de rayon. Les trois ont un paramètre de lissage qui contrôle la force des transitions.

**Lissage** : un seul champ numérique sous le curseur de noyau, dont le sens dépend de la méthode sélectionnée (plage 0–8, pas 0,2) :

| Méthode | Libellé | Défaut | Effet |
|---|---|---|---|
| OpenFocus A | Lissage du mélange | 2,0 | Contrôle le lissage de la carte de réponse de netteté avant la sélection des pixels |
| OpenFocus B | Lissage de profondeur | 1,2 | Contrôle le lissage de la carte de profondeur qui guide le mélange |
| OpenFocus C | Lissage du mélange | 3,0 | Contrôle le lissage de la carte de profondeur qui guide la reconstruction multi-échelles |

Des valeurs plus élevées augmentent le lissage des transitions. Chaque méthode mémorise sa propre valeur entre les sessions, et le traitement par lots applique la même règle.

C n'a pas de paramètre Rayon ; son curseur de noyau est donc masqué.

CLI : `--depth-smoothing VALUE` (B) et `--blend-smoothing VALUE` (A, C).

**Les options de luminosité sont propres à chaque méthode.** *Égaliser l'exposition avant la fusion* et *Supprimer la dérive de luminosité* sont mémorisées séparément pour chaque méthode de fusion, car elles répondent à une question propre à chaque pile plutôt qu'à une préférence globale. Les deux sont activées par défaut pour OpenFocus A/B/C (qui n'ont pas de calibration d'exposition intégrée) ; les autres méthodes conservent les valeurs historiques (égalisation activée, suppression de dérive désactivée). Changer de méthode affiche les valeurs de cette méthode, qui survivent au redémarrage. **Réinitialiser** efface les valeurs mémorisées de luminosité, lissage et rayon de toutes les méthodes.

**Accentuer la sortie fusionnée** applique un léger masque flou après la fusion, avant la conversion du résultat vers sa profondeur de sortie. Désactivé par défaut, il s'active dans les options de fusion principales ou du lot ; l'option CLI équivalente est `--sharpen-output`. Cela affecte l'image fusionnée, pas les images recalées ni les cartes de profondeur. Cela peut rendre les contours plus nets, mais aussi amplifier le bruit et les halos. Le réglage est enregistré avec le projet et mémorisé entre les sessions.

### Alignement seul (sans fusion)

Sélectionnez **Alignement seul (sans fusion)** pour recaler la pile et ajouter une entrée de pile à la liste des sorties. Les images restent en mémoire jusqu'à leur export ; l'alignement seul ne crée pas de cache supplémentaire ni n'écrit à côté des fichiers sources. Enregistrez l'entrée pour créer un dossier dans la destination choisie. Les noms de fichiers d'origine sont conservés, avec l'extension du format choisi.

### Sélectionner une méthode de fusion

1. Dans le panneau droit, repérez le groupe **Fusion**
2. Choisissez une méthode dans la **liste déroulante** (une seule est toujours sélectionnée ; la première entrée recale sans fusionner)
3. Ajustez la taille du noyau le cas échéant — le libellé indique ce que signifie le noyau pour la méthode choisie
4. Cliquez sur **Lancer le rendu** pour traiter

### Guide de comparaison des algorithmes

| Algorithme | Vitesse | Qualité | Idéal pour |
|-----------|-------|---------|----------|
| Filtre guidé | Rapide | Bonne | Usage général |
| DCT | Moyenne | Bonne | Textures |
| DTCWT | Moyenne | Très bonne | Scènes complexes |
| GFG-FGF | Rapide | Bonne | Régions nettes |
| StackMFF (V1) | Lente (GPU) | Non évaluée ici | Luminance indépendante de l'ordre ; chrominance variable en cas d'égalité des écarts de luminance |
| StackMFF V2 | Rapide (GPU) | Non évaluée ici | Balayages continus avec un modèle minuscule |
| StackMFF V3 | Lente (GPU) | Excellente | Qualité maximale |
| OpenFocus A | Dépend de la pile | Dépend de la pile | Sélection multi-échelles des pixels nets |
| OpenFocus B | Dépend de la pile | Dépend de la pile | Mélange d'images voisines guidé par la profondeur |
| OpenFocus C | Dépend de la pile | Dépend de la pile | Reconstruction multi-échelles guidée par la profondeur |

Ouvrez **Outils → Comparaison de méthodes** pour régler plusieurs méthodes indépendamment. Le nom, les libellés de paramètres et les contrôles de chaque méthode restent groupés lorsque la fenêtre se rétrécit. Les modifications actualisent l'aperçu automatiquement. **Aperçu ROI (rapide)** fusionne la zone sélectionnée avec le contexte environnant ; **Fusion pleine image** vérifie le résultat sur toute l'image et permet d'inspecter une autre zone sans relancer le rendu. Les résultats ROI peuvent différer de ceux en pleine image, car les statistiques d'exposition et certains algorithmes dépendent de l'étendue de l'image.
La **fusion pleine image** est sélectionnée par défaut.

Une indication visible explique la sélection : cliquez sur un résultat ou utilisez `Tab` pour lui donner le focus, puis `Entrée` ou `Espace` pour le sélectionner.

Utilisez **Taille de l'aperçu (px)** au-dessus des volets de droite pour choisir un carré de 160 à 960 pixels, en faisant glisser le curseur ou en saisissant un nombre. Toutes les méthodes utilisent la même taille et s'adaptent à la largeur disponible. La taille est mémorisée pour la prochaine ouverture. La modifier conserve l'agrandissement courant et ajuste le carré d'inspection autour de son centre, en le limitant aux bords de l'image.

Appuyez sur **F11** ou sur le bouton **Plein écran (F11)** pour utiliser tout l'écran ; le texte du bouton devient **Quitter le plein écran (F11)**. La barre de titre peut aussi agrandir la fenêtre. **Échap** quitte d'abord le plein écran et ne ferme la boîte qu'à la deuxième pression. Le retour du plein écran restaure la taille précédente de la fenêtre, et la grille de résultats se réorganise selon la largeur disponible dans les deux cas.

Faites glisser l'image source avec le bouton gauche pour déplacer le carré d'inspection blanc ; relâcher le bouton le laisse en place. Les volets de droite affichent exactement cette région de la même pile alignée. Utilisez la molette pour changer l'agrandissement de la région, le glisser-déposer avec le bouton droit pour déplacer la source, **Ctrl+molette** pour zoomer la source et **F** pour ajuster. Changer d'image source ou redimensionner la fenêtre conserve la sélection. Annuler arrête le travail en cours ; fermer attend que l'opération native actuelle se termine en sécurité.

### Aperçu rapide (rendu brouillon)

Régler les paramètres sur de grandes piles est lent en pleine résolution. Ouvrez **Paramètres de rendu** et cochez **Aperçu rapide** : OpenFocus produit d'abord un brouillon à la moitié de la taille linéaire (jamais moins de 320 px sur le côté long), généralement plusieurs fois plus rapide, recalage compris. Le message de fin (et l'infobulle du résultat le plus récent) indique **Qualité : aperçu**, et le brouillon n'est **pas** ajouté à l'historique des sorties. Décochez la case et relancez le rendu pour obtenir le résultat en pleine résolution. Le mode aperçu ne s'applique pas aux rendus ROI et ne pollue jamais le cache d'alignement pleine résolution.

---

## 9. Transformations d'images

### Rotation

1. Allez dans **Édition → Pivoter**
2. Choisissez le type de rotation :
   - **90° horaire** : pivote toutes les images de 90 degrés vers la droite
   - **90° antihoraire** : pivote toutes les images de 90 degrés vers la gauche
   - **180°** : retourne les images sens dessus dessous

### Retournement

1. Allez dans **Édition → Retourner**
2. Choisissez le type de retournement :
   - **Retournement horizontal** : miroir de gauche à droite
   - **Retournement vertical** : miroir de haut en bas

### Redimensionnement

1. Allez dans **Édition → Redimensionner**
2. Une boîte de dialogue de redimensionnement apparaît
3. Saisissez les nouvelles dimensions ou un pourcentage d'échelle
4. Toutes les images de la pile sont redimensionnées proportionnellement

### Ordre des transformations

Les transformations s'appliquent à toute la pile d'images, en conservant l'alignement entre les images.

---

## 10. Retouche, libellés et annotations

### Retoucher le résultat (Édition → Retoucher le résultat…, `Ctrl+T`)

La fusion peut encore laisser un défaut local — un halo autour d'un contour très contrasté, un poil pris dans la mauvaise image, du bruit dans une zone plane. L'espace de travail copie les pixels des images exactes qui ont produit le résultat sélectionné :

1. Sélectionnez une sortie dans la liste (clic droit → *Retoucher le résultat…* fonctionne aussi) et appuyez sur `Ctrl+T`. Sans sélection, un sélecteur liste tous les résultats et indique ceux qui ont des images sources enregistrées.
2. L'image source est à gauche, le résultat à droite. Les deux partagent le zoom et la position : la molette zoome, le glisser avec le bouton droit ou central (ou Espace+glisser) déplace, `F` ajuste et `1` affiche 100 %.
3. Choisissez une vignette, faites glisser le curseur d'image ou appuyez sur `,` / `.`. Un trait est verrouillé sur cette image et les réglages de pinceau courants ; l'image ne peut pas changer avant la fin du trait.
4. Choisissez **Pinceau source** (`B`) ou **Pinceau de restauration** (`E`). Le diamètre, la dureté et l'opacité disposent d'un curseur et d'un champ numérique ; `[` / `]` ajuste le diamètre. `Ctrl` + molette vers le haut augmente le diamètre du pinceau, vers le bas le diminue ; la molette seule continue de zoomer. Le pinceau de restauration ramène les pixels à l'image que vous avez ouverte, pas à une sortie antérieure.
5. Maintenez `Tab` pour voir le résultat avant cette session et `Alt` pour jeter un œil à l'image source. La référence de netteté est facultative, ne couvre que le canevas source et utilise une échelle commune à toutes les images.
6. `Ctrl+Z` / `Ctrl+Y` annulent et rétablissent des traits entiers. **Tout restaurer…** est une seule étape annulable. **Appliquer comme nouvelle sortie** ajoute une entrée nommée `<nom> · Retouché N` et la sélectionne ; l'original reste intact. La nouvelle entrée est marquée *Non exportée* jusqu'à son export depuis la fenêtre principale, et OpenFocus la protège de l'éviction automatique de l'historique et d'une fermeture sans décision.

Appliquez la retouche comme nouvelle sortie, sélectionnez-la et ouvrez **Enregistrer sous**. Choisissez TIFF multicouche dans la liste des formats pour enregistrer le résultat avec ses images sources. Les libellés et barres d'échelle actifs ne sont pas inclus ; les modifications d'annotation non enregistrées restent protégées. La géométrie source et les masques de validité sont conservés. Le résultat et les sources doivent être des images RVB ou en niveaux de gris opaques de même profondeur 8 ou 16 bits, et chaque rectangle source doit tenir dans le résultat. L'export prend en charge jusqu'à 128 sources ; les noms de source sont limités à 255 caractères et ne peuvent pas contenir de NUL ni de caractères hors du plan multilingue de base, comme beaucoup d'emoji. Il refuse les contextes dangereux et les tâches estimées à plus de 512 Mio de mémoire de travail ou dépassant le budget de RAM disponible, en signalant l'échec plutôt qu'en écrivant un export incomplet. GIMP 3.2.6 sous Windows importe les calques 8 et 16 bits, y compris leurs décalages et le remplissage transparent de recalage. Le résultat est visible et les calques sources masqués ; GIMP ajoute aussi un calque d'arrière-plan visible. Les autres éditeurs n'ont pas été testés.

Pour une correction tonale locale, sélectionnez **Pinceau d'exposition**, réglez sa valeur en centièmes de diaphragme (100 double les valeurs de pixel, -100 les divise par deux) et peignez sur le résultat. L'aperçu se met à jour pendant la peinture. Chaque trait ajuste le résultat courant une fois, même là où ses touches se chevauchent ; un autre trait s'y ajoute. Utilisez l'opacité et la dureté du pinceau pour contrôler le mélange. Les hautes lumières peuvent être écrêtées en éclaircissant. Maintenez **Avant** pour comparer avec le résultat tel qu'ouvert, appuyez sur Échap pendant un trait pour l'annuler, ou utilisez annuler/rétablir après le trait. Ce sont des modifications de pixels, pas des calques réglables. Appliquez comme nouvelle sortie et exportez la nouvelle entrée pour enregistrer les pixels modifiés ; un projet ne stocke pas ces pixels.

**Réglages du résultat…** modifie tout le résultat de retouche courant. Son aperçu échantillonné se met à jour lorsque vous changez l'exposition et le contraste ; cochez **Avant** pour comparer avec le résultat avant l'ouverture du réglage. Les deux contrôles utilisent des centièmes de diaphragme : l'exposition multiplie les valeurs de pixel, puis le contraste met à l'échelle leur distance au gris moyen. Des valeurs extrêmes peuvent écrêter les hautes lumières ou les ombres. **OK** applique le réglage comme une seule modification annulable ; **Annuler** laisse le résultat inchangé. Pendant le traitement, Annuler demande un retour en arrière. L'image complète doit tenir dans la limite mémoire de l'historique de retouche pour cette opération annulable. Publiez et exportez le résultat comme décrit ci-dessus.

La provenance voyage avec la sortie : les images alignées, les décalages de recadrage/collage ROI, le rognage des bords DCT et la profondeur de bits sont enregistrés au moment du rendu. Les fichiers de lot, les résultats hérités et les entrées de dossier d'alignement seul n'ont pas de source enregistrée : ils sont donc listés comme indisponibles au lieu d'être devinés depuis la pile actuelle. Les traits non appliqués et l'historique d'annulation ne durent que le temps de la session ; enregistrer un projet ne les stocke pas.

Pour un fichier image existant, choisissez **Édition → Retoucher des fichiers image…**, puis sélectionnez le résultat et une ou plusieurs images sources. Les sources doivent déjà être alignées sur le résultat dans la même grille de pixels ; des dimensions identiques ne prouvent pas l'alignement. Les images PNG, TIFF, JPEG et BMP d'une seule page, de dimensions verticales, canaux et profondeur 8 ou 16 bits identiques, sont acceptées ; l'orientation EXIF est appliquée au chargement. Les images avec alpha et multipage ne sont pas prises en charge. Les images sélectionnées sont fournies par l'utilisateur, pas issues d'une provenance de rendu enregistrée. Appliquer crée une sortie séparée non exportée ; enregistrez-la depuis la liste des sorties. Les fichiers choisis sont protégés contre l'écrasement pendant cette session, y compris les exports de cartes de profondeur et de maillages 3D. Comme pour les autres résultats, les fichiers de projet ne stockent ni ces pixels ni l'historique du pinceau.

### Remplacer l'arrière-plan (Arrière-plan automatique)

L'empilement rend l'arrière-plan net en même temps que le sujet. L'arrière-plan
d'une image empilée devient alors souvent un patchwork de flous différents, avec
un halo autour du sujet. **Arrière-plan automatique…** dans l'espace de retouche
le remplace par une vraie image de la même pile. Le sujet reste tel qu'empilé.
L'arrière-plan est dérivé d'une image et du résultat empilé ; le traitement peut
reconstruire, ajuster et mélanger ses pixels.

1. Ouvrez le résultat dans l'espace de retouche et cliquez sur
   **Arrière-plan automatique…**.
2. Choisissez l'arrière-plan :
   - **Automatique** (recommandé) utilise l'arrière-plan doux, sauf si la pile
     a enregistré une grande partie de l'arrière-plan nette. Il utilise alors
     l'arrière-plan net.
   - **Arrière-plan doux** utilise l'image où l'arrière-plan est le plus flou.
   - **Arrière-plan net** utilise l'image dont la mise au point est la plus
     lointaine.
3. Laissez **Nettoyer les halos des bords** activé pour retirer les halos le
   long des bords du sujet. Il ne dépasse jamais ce que les images ont
   enregistré. Désactivez-le pour garder le rendu des bords de la pile.
4. Cochez **Conserver les parties floues rattachées au sujet** quand des parties
   du sujet sont juste hors de la profondeur de champ, comme le bout arrondi
   d'une épine. Des objets d'arrière-plan voisins du sujet peuvent aussi rester
   tels qu'empilés, vérifiez donc le résultat.
5. Cliquez sur **Remplacer l'arrière-plan**. Une pile de 15 images de
   20 mégapixels prend environ une minute et demande environ 3,5 Go de mémoire
   libre. **Annuler** l'arrête.

La modification est une seule étape annulable, et les traits déjà peints sur le
sujet sont conservés. Utilisez le **Pinceau de restauration** pour retrouver les
pixels empilés là où vous les préférez, par exemple un brin d'herbe flou devant
le sujet. Si le sujet remplit presque toute l'image, comme une scène de sol
forestier, l'image reste telle qu'empilée et une note l'indique.

Après un passage, **Afficher les données manquantes** colore les endroits
qu'aucune image n'a vus nettement. Ce sont les endroits à vérifier ou à
retoucher à la main :

- magenta : arrière-plan qu'aucune image n'a enregistré sans le sujet devant
- cyan : sujet caché à l'appareil
- jaune : sujet hors de la plage de mise au point de la pile
- ardoise : zones trop lisses pour juger

**Enregistrer la carte des données manquantes…** écrit l'aperçu coloré et un
masque noir et blanc par couleur. L'arrière-plan automatique a besoin des images
qui ont produit le résultat et n'est pas disponible pour les résultats de zone
(ROI). Les nouveaux points de contrôle de fond restaurent exactement l'image,
l'historique accessible et les diagnostics sans recalculer l'arrière-plan.

### Gestion des libellés

Ouvrez **Outils → Libellés** après avoir chargé des images ou produit un résultat. Chaque libellé a sa propre cible : la pile source, la pile recalée ou les résultats de sortie. Les libellés de sortie peuvent s'appliquer à tous les résultats ou au résultat sélectionné par son nom ; renommer ou réordonner ce résultat ne change pas l'association.

Utilisez **Ajouter** pour créer un autre libellé, sélectionnez une entrée pour la modifier et utilisez sa case pour l'activer ou la désactiver. Saisissez un texte fixe ou insérez `{value}` pour un numéro d'image. Le numéro est la valeur de départ plus l'indice d'image (base zéro) multiplié par l'intervalle ; une sortie unique utilise toujours la valeur de départ.

Choisissez toutes les images, l'image courante ou une plage personnalisée comme `3-5,8`. Des numéros d'image invalides ou indisponibles empêchent l'application du brouillon. Les plages courante et personnalisée suivent les identités sources sélectionnées lorsque la pile est réordonnée ; les libellés « toutes les images » s'appliquent aussi aux images ajoutées. Les cibles manquantes restent associées à leurs identités d'origine jusqu'à ce que vous choisissiez explicitement une nouvelle cible.

L'aperçu utilise la cible réelle. Naviguez avec Précédent/Suivant ou saisissez son numéro d'image, choisissez **Ajuster** ou **100 %**, et faites glisser l'image pour la déplacer à 100 %. La position et la taille de police utilisent les pixels de l'image. Une police de remplacement ou un dessin échoué est signalé. Les libellés conservent la profondeur de bits d'origine lorsqu'ils sont exportés en PNG ou TIFF.

**Appliquer et fermer** valide tout le brouillon comme une seule modification annulable. Annuler, Échap et fermer la boîte le rejettent. Sélectionner un autre libellé conserve les modifications du brouillon. **Supprimer** ne retire que l'entrée sélectionnée, y compris la dernière. Les actions de suppression du menu effacent indépendamment les libellés source, de pile recalée, de sortie ou tous les libellés, et peuvent aussi être annulées. Appliquer des libellés modifie l'affichage de la session ; exportez une image pour écrire les annotations dans un fichier.

---

## 11. Paramètres et configuration

Ouvrez **Paramètres → Paramètres avancés** pour les préférences de performance, de recalage, de nettoyage du capteur, de nommage des sorties, d'IA StackMFF et de démarrage. Les modifications restent des brouillons jusqu'à **OK** ; Annuler, Échap et fermer les rejettent. Réduire la taille de tuile met immédiatement à jour la limite de chevauchement visible à la moitié d'une tuile. La limite de lot enregistrée de la section IA est conservée, mais l'inférence actuelle soumet une tuile par appel ; la largeur de recalage peut modifier la précision de l'alignement. La collecte des diagnostics d'alignement est une case de la section Recalage.

Les téléchargements automatiques de mises à jour s'exécutent lorsque l'application est inactive et ne s'installent jamais seuls. Un téléchargement terminé reste prêt jusqu'à ce que vous choisissiez explicitement **Redémarrer et mettre à jour**. Le redémarrage protège les modifications actives et le travail non enregistré, puis écrit les données de récupération avant de lancer le remplacement. Annuler un téléchargement ou une préparation attend la limite de sécurité courante. Une version préparée complète et plus récente peut être installée hors ligne.

La correction de champ plat modifie les pixels de travail de la session sans écraser les fichiers sources. Sélectionnez la référence blanche et une image d'aperçu, puis régénérez l'aperçu après avoir changé la source ou l'échelle. Les images de référence et de travail/base doivent partager le même domaine de coordonnées et la même taille. Appliquer prépare les deux piles avant validation ; la suppression facultative de l'image de référence fait partie de la même étape d'annulation. Exportez les images sources corrigées lorsque vous devez conserver ces pixels hors de la session.

Le traitement par lots accepte une image de champ blanc facultative dans ses options. Choisissez une référence défocalisée à la résolution d'origine des images ; elle est mise à l'échelle avec la pile d'entrée et appliquée avant le recalage, avec la taille de poussière par défaut. Une référence illisible ou de taille incompatible fait échouer visiblement la pile concernée. La référence est exclue des entrées de dossier, y compris le traitement groupé ; au moins deux images sources doivent rester pour la fusion. La correction de champ plat ne peut pas être combinée avec une entrée transparente. Laissez le champ vide pour traiter sans correction. La CLI accepte aussi la référence avec `--dust-map WHITE`.

Le nettoyage du capteur est facultatif et s'exécute sur les images d'origine en coordonnées capteur avant le recalage. Sélectionnez un masque de défauts explicite pour corriger des pixels chauds ou morts connus ; OpenFocus n'infère pas de masque à partir des images du spécimen. Le masque doit être une image monocanal 8 ou 16 bits de mêmes dimensions que la source : zéro signifie non affecté, et 1 ou la valeur maximale du canal marque les défauts. Vérifiez son étendue avant le traitement. Les masques couvrant plus de 10 % de l'image, ou les défauts sans voisin non masqué, sont refusés. Conservez le masque à la résolution d'origine et ne le sélectionnez pas comme image source. Pour le traitement dans l'interface, chargez les images sources à pleine échelle lors de l'utilisation d'un masque de défauts ; le masque n'est jamais redimensionné. Le nettoyage du capteur (y compris la réduction du bruit) ne peut pas être appliqué à une pile de l'interface après une rotation, un retournement, une correction de champ plat ou toute autre modification de source : rechargez les images d'origine ou désactivez le nettoyage. Cela vaut aussi pour les projets restaurés avec de tels historiques de source. Le traitement ROI préaligné ne peut pas être combiné avec le nettoyage du capteur ; quittez le mode ROI et rendez la pile complète. Les réglages de nettoyage appartiennent à la session et au projet enregistré, pas aux préférences globales de l'application. Le lot et la CLI traitent leurs entrées d'origine séparément, avec le nettoyage du capteur avant la correction de champ plat et le recalage. Les masques de défauts exigent des fichiers sources à orientation EXIF neutre. Les fichiers orientés, les RAW d'appareil photo et les sources dont l'orientation ne peut pas être établie sont refusés, car leurs pixels décodés peuvent ne plus correspondre aux coordonnées du capteur. Utilisez des images sources et un masque dans la même orientation de pixels connue, avec des métadonnées d'orientation neutres. La réduction du bruit seule ne nécessite pas de masque de défauts.

La réduction de bruit du capteur facultative est indépendante du masque de défauts et désactivée par défaut. Inspectez le résultat en taille réelle pour décider si son lissage convient au spécimen. Le nettoyage du capteur conserve les fichiers sources et la profondeur d'image 8 ou 16 bits ; il est distinct de la correction de champ plat. La CLI accepte `--defect-mask-path MASK` et `--sensor-denoise` ; les recettes et l'API de recettes Python capturent ces options. Les recettes enregistrent une empreinte du masque et refusent un masque manquant ou modifié. Le nettoyage CLI ne peut pas être combiné avec une entrée transparente. Le traitement groupé applique le nettoyage aux images d'origine, pas aux résultats intermédiaires du groupe.

Les recettes exigent que leur sortie se trouve hors de tous les dossiers d'entrée et refusent une entrée explicitement sélectionnée comme sortie. Choisissez un emplacement de sortie séparé pour protéger les images sources et éviter d'inclure un résultat antérieur lors d'une nouvelle exécution. Les réglages de traitement d'un projet peuvent être réutilisés comme modèle ou recette même si ses sources d'origine ont été déplacées : **Fichier → Enregistrer un modèle de traitement…** stocke les réglages sans chemins source et **Fichier → Appliquer un modèle de traitement…** les applique à une pile chargée, tandis que `openfocus recipe create --project` écrit une recette headless qui nécessite encore des chemins `--input` explicites. Les modèles transportent l'indicateur de réduction du bruit du capteur mais pas le chemin du masque de défauts ; les recettes stockent le chemin du masque et son empreinte. L'ouverture du projet lui-même exige toujours ses sources et toute référence de champ plat enregistrée.

Lorsqu'une référence de fichier est appliquée dans Outils à l'échelle de base de la pile, un projet enregistré conserve son chemin absolu, le hash du contenu et la taille de poussière, et la réapplique une fois aux sources d'origine à l'ouverture. Gardez le fichier de référence inchangé et en place : une référence manquante ou modifiée empêche l'ouverture du projet. Une référence sélectionnée depuis la pile, ou une correction à une autre échelle de travail/base, exige encore d'exporter la pile source corrigée avant d'enregistrer un projet. Si l'échelle de travail change après la correction, rétablissez l'échelle de correction avant d'enregistrer ou exportez la pile corrigée.

**Aide → Envoyer des commentaires** ouvre les journaux locaux. Copier copie le journal visible de la session ; le ZIP de commentaires contient les journaux de session et de l'application complète, ainsi que les informations d'environnement. Relisez le paquet avant de l'envoyer vous-même. Ouvrir GitHub ne fait qu'ouvrir le tracker. Annuler un export laisse les fichiers inchangés ; une erreur conserve le fichier de destination précédent et affiche le chemin en échec.

Chaque lancement écrit son propre journal rotatif, de sorte que les tâches GUI/CLI simultanées conservent des enregistrements séparés. Les identifiants de tâche relient paramètres, durée et résultats terminés/annulés/échoués ; les erreurs inattendues incluent une trace complète. Le ZIP de commentaires contient aussi le rapport de plantage courant et les versions des dépendances. Utilisez **Aide → Ouvrir le dossier des journaux** ou le chemin affiché sur la page de commentaires pour trouver les fichiers. Les journaux tournent à 1 Mo avec trois sauvegardes par processus ; les journaux de sessions inactives de plus de 14 jours sont supprimés. Pour des journaux de support détaillés, définissez `OPENFOCUS_LOG_LEVEL=DEBUG` avant le lancement ; `OPENFOCUS_LOG_DIR` permet de choisir un autre dossier.

### Paramètres des tuiles (optimisation de la mémoire)

Pour les grandes images, OpenFocus utilise un traitement par tuiles afin d'éviter les problèmes de mémoire.

Accès : **Paramètres → Paramètres avancés → Performances** :

- **Taille de bloc de tuile** : taille de chaque tuile de traitement (1024 px par défaut)
- **Chevauchement des tuiles** : chevauchement entre tuiles pour un mélange sans couture (256 px par défaut)
- **Seuil de tuilage** : taille d'image au-delà de laquelle les tuiles sont utilisées (2048 px par défaut)

**Recommandations** :

- Tuiles plus petites : mémoire réduite, traitement plus lent
- Tuiles plus grandes : mémoire accrue, traitement plus rapide
- Augmentez le chevauchement si des coutures apparaissent dans les résultats

### Paramètres de recalage

Accès : **Paramètres → Paramètres avancés → Recalage** :

- **Largeur de réduction** : résolution de prétraitement pour le recalage
  - Défaut : 1024 px
  - Plus faible pour la vitesse, plus élevée pour la précision
- **Collecter les diagnostics d'alignement** : recalcule l'alignement au prochain rendu et active **Paramètres → Paramètres avancés → Recalage → Diagnostics d'alignement**

### Référence de performances

Voir [PERFORMANCE.md](./PERFORMANCE.md) pour les débits et la mémoire mesurés sur de grandes piles réelles, ainsi que des conseils pratiques pour les piles profondes et les entrées haute résolution.

Le panneau d'état affiche la mémoire physique utilisée et totale lorsque le système peut fournir les deux mesures ; sinon, il affiche `N/D`.

### Paramètres du nombre de threads

Contrôlez le traitement parallèle du CPU :

Accès : **Paramètres → Paramètres avancés → Performances** :

- `0` sélectionne l'ordonnancement automatique et constitue la valeur par défaut des nouveaux réglages.
- Les valeurs positives sont des limites supérieures ; les limites enregistrées survivent aux mises à jour.
- La capacité CPU, le mode de ressources et la marge mémoire peuvent réduire la concurrence. Plus de threads ne rend pas nécessairement un rendu plus rapide.

### Mode de ressources

**Paramètres → Paramètres avancés → Performances → Mode de ressources** décide de la part de la machine qu'un rendu peut utiliser. Il ne modifie jamais l'image fusionnée.

| Mode | Comportement |
|---|---|
| Équilibré (par défaut) | Laisse libres environ 20 % des cœurs physiques et une réserve de mémoire pour le bureau |
| Réactif | Utilise environ la moitié des cœurs et une fenêtre de travail plus petite ; rendus plus lents, machine plus réactive |
| Maximum | Utilise tous les cœurs physiques autorisés par le budget mémoire |

Le planificateur tient compte des cœurs physiques, de l'affinité et des quotas CPU, et exécute une seule couche parallèle à la fois. Les tâches lourdes partagent l'admission pendant la préparation et la publication ; l'annulation attend la fin de l'appel natif actif. Une mémoire inconnue ou insuffisante produit une erreur de ressources. La grille de tuiles choisie reste inchangée ; choisir des tuiles plus petites peut modifier les pixels de sortie.

### Accélération GPU

StackMFF V1/V2/V3 utilisent ONNX Runtime. DirectML sous Windows et CoreML sous macOS peuvent accélérer l’inférence ; sinon le CPU est utilisé. La disponibilité du fournisseur ne prouve pas une exécution sur GPU. La prise en charge d’autres GPU reste à valider.

- Basculez via **Paramètres → Accélération GPU**
- La décocher force les méthodes neuronales sur le CPU — utile lorsque les pilotes GPU posent problème
- Le panneau d'état affiche `N/D` lorsqu'aucun accélérateur n'est utilisé ou disponible, et `CPU*` lorsqu'un accélérateur existe mais que la méthode sélectionnée est réservée au CPU
- Le choix s'applique au rendu suivant et est mémorisé

### Taille de lot IA

Accès : **Paramètres → Paramètres avancés → IA** :

- La limite de lot enregistrée est de 2 par défaut ; la politique d'exécution actuelle plafonne l'inférence réelle à une tuile par appel sur tous les appareils.
- Augmenter cette limite n'active pas de lots GPU non validés et ne réduit pas la taille des tuiles spatiales.

### Thèmes

Deux thèmes d'interface sont disponibles : **Clair (par défaut)** et **Sombre**. Changez via **Paramètres → Thème** ; le choix s'applique immédiatement (la barre de titre Windows suit) et est mémorisé. Au premier lancement, l'utilisateur choisit un thème dans la boîte de bienvenue ; un choix enregistré l'emporte toujours.

### Mode portable

Créez un fichier vide nommé `OpenFocus.portable` à côté de `OpenFocus.exe` (le ZIP portable en contient un) : toutes les données utilisateur — réglages, journaux, fichiers récents — sont stockées dans un dossier `OpenFocusData` à côté de l'application au lieu de votre profil, ce qui permet d'emporter l'installation sur une clé USB. Supprimez le marqueur pour revenir au profil normal.

### Métadonnées EXIF

Les métadonnées source (modèle d'appareil, exposition, orientation, etc.) sont conservées : l'orientation est appliquée au chargement, et les exports PNG/TIFF/WebP intègrent automatiquement l'EXIF d'origine. Les exports JPG/BMP ne peuvent pas stocker l'EXIF complet et sont écrits sans lui.

### Entrée HEIC/HEIF

Les photos iPhone se chargent directement. Nécessite `pillow-heif`, que `pip install openfocus` installe et que les versions packagées embarquent.

### Mises à jour

**Aide → Rechercher des mises à jour** compare votre version aux informations de `openfocus.top`. Une version plus récente est téléchargée depuis ce site en arrière-plan pendant que l'application est inactive ; une fois la version préparée, la bannière et la zone d'état proposent **Redémarrer et mettre à jour**, qui l'applique hors ligne. Les versions qui ne peuvent pas se mettre à jour elles-mêmes ouvrent la page de téléchargement du site. La vérification automatique s'exécute silencieusement au plus une fois par jour.

### Cache de recalage

OpenFocus peut stocker les résultats de recalage dans un dossier `.openfocus_cache` à côté de la pile source. Rouvrir la même pile (fichiers inchangés) évite alors totalement l'alignement. Activez ou désactivez via **Paramètres → Paramètres avancés → Recalage → Cache de recalage** ; les entrées obsolètes sont remplacées automatiquement.

### Paramètres persistants

Tout ce qui suit survit automatiquement au redémarrage : nombre de threads, mode de ressources, paramètres de tuiles, largeur de réduction du recalage, taille de lot de la fusion IA (StackMFF V3), accélération GPU, cache de recalage et cases de diagnostics d'alignement, langue de l'interface, taille de fenêtre et disposition des panneaux, format d'export par glisser-déposer, options de luminosité par méthode, valeurs de lissage et de rayon, et piles récemment ouvertes. Les boîtes de paramètres enregistrent dès confirmation ; tout est aussi enregistré à la fermeture. Les préférences sont stockées par utilisateur dans `OpenFocus/OpenFocus.ini` (format INI de QSettings).

### Langue de l'interface

Changez à tout moment via **Paramètres → Langue**, qui liste les douze paquets (English, 中文, 日本語, Español, Deutsch, Français, Русский, 한국어, Português, Italiano, Nederlands, Polski). Le changement s'applique immédiatement et est mémorisé pour les sessions futures.

### Vérification automatique des mises à jour

Une fois par jour, quelques secondes après le lancement, OpenFocus vérifie silencieusement `openfocus.top`. Les notes de version proviennent aussi de ce site. Si une version plus récente existe, une bannière apparaît au-dessus des panneaux d'image : la version, **Mettre à jour maintenant** (applique une version préparée ou démarre le téléchargement), **Nouveautés** (notes de version) et un bouton de fermeture. La fermer masque la bannière pour cette version seulement ; la version suivante la réaffiche, et **Aide → Rechercher des mises à jour** la réactive immédiatement.

Pendant le téléchargement en arrière-plan, la barre d'état affiche la progression. Vous pouvez aussi vérifier manuellement à tout moment via **Aide → Rechercher des mises à jour**.

---

## 12. Traitement par lots

Traitez plusieurs dossiers automatiquement :

1. Allez dans **Outils → Traitement par lots**
2. Dans la boîte de dialogue :
   - **Ajouter des dossiers** : sélectionnez des dossiers contenant des séquences d'images
   - **Glisser des dossiers** : déposez un ou plusieurs dossiers de piles n'importe où dans la boîte (ou sur la liste) pour tous les mettre en file ; le dépôt passe en mode multi-dossiers et les vignettes apparaissent au fur et à mesure du décodage
   - **Retirer des dossiers** : retire les dossiers sélectionnés de la liste
3. Configurez la sortie :
   - **Enregistrer dans un sous-dossier** : crée les résultats dans chaque dossier d'entrée
   - **Enregistrer dans le même dossier** : enregistre à côté de l'entrée en conservant les fichiers existants
   - **Dossier personnalisé** : indiquez l'emplacement de sortie
4. Vérifiez les réglages de fusion et de recalage courants
5. Cliquez sur **Lancer le traitement par lot**

La boîte de dialogue affiche la progression en temps réel. Vous pouvez annuler le traitement à tout moment.

### Remarques sur le traitement par lots

- Part d'un instantané des réglages de la fenêtre principale ; les contrôles du lot les remplacent.
- Chaque dossier ou groupe divisé est traité indépendamment par le même pipeline.
- **Traiter les grandes piles par groupes** est une case facultative du mode multi-dossiers. Lorsqu'elle est cochée, **Taille du groupe** (minimum 2, 4 par défaut) et **Chevauchement** (de 0 à la taille du groupe moins 2) apparaissent ; lorsqu'elle est décochée, ces contrôles sont masqués et le regroupement est désactivé. Le chevauchement de bureau commence à 1 et est limité à la plage autorisée lorsque la taille change. Les groupes sont fusionnés par tours en utilisant de l'espace disque temporaire ; la progression identifie chaque tour. La fusion, l'égalisation d'exposition, la correction de base et le recalage utilisent les images et références propres à chaque groupe, y compris les tours intermédiaires, si bien que le résultat s'approche du traitement de la pile complète en une fois. Le regroupement exige une méthode de fusion ; la profondeur par image d'origine, la provenance et les exports d'images alignées sont indisponibles. Le mode dossier divisé n'est pas pris en charge, la case y est donc refusée. Les groupes recalés utilisent leur première image comme référence de coordonnées. Chaque recalage retire une bordure prudente pour l'interpolation, et les tours suivants croisent les rectangles survivants, y compris toute queue d'une seule image. L'image finale est donc recadrée. Chaque groupe recalé doit mesurer plus de 8 pixels dans les deux dimensions ; des rotations difficiles ou des transformations projectives peuvent ne laisser aucun rectangle sûr et produire une erreur. Utilisez des entrées préalignées sans recalage ou désactivez le regroupement si cela se produit.
- L'entrée groupée doit avoir des dimensions orientées identiques et être constituée de fichiers d'une seule image dont les en-têtes sont lisibles par Pillow. Les RAW d'appareil photo, les fichiers multipage et les fichiers avec canal alpha ou transparence ne sont pas pris en charge, y compris les piles PNG transparentes. Retirez l'alpha ou désactivez le regroupement pour traiter ces piles. Les essais à blanc de la CLI vérifient ces restrictions. En ligne de commande, un exemple est `python main.py --input stack --output fused.png --group-size 8 --group-overlap 1`. L'entrée groupée CLI doit être un dossier et ne peut pas être combinée avec `--depth-map`. Une référence d'alignement OpenFocus différente pour toute la pile ne peut pas être combinée avec le regroupement. Une référence de champ plat est appliquée aux images d'origine avant le recalage ; l'accentuation de sortie est appliquée une fois après le groupe final.
- **Ce qui est écrit et où, selon le mode.** Avec **alignement seul**, la destination choisie reçoit un dossier `<pile>_aligned` par pile et le panneau de sortie liste ce dossier. Avec **fusion + enregistrer les alignées**, l'image fusionnée est écrite sous `<pile>.<ext>` dans la destination et ses images alignées vont dans `<pile>_aligned` à côté ; le panneau de sortie ne liste que le résultat fusionné. Avec **fusion sans export d'alignement**, seule l'image fusionnée est écrite. Le panneau de sortie nomme les dossiers exactement comme sur le disque.
- Les images alignées conservent les noms d'origine. Les noms en double et les dossiers existants reçoivent des suffixes numériques. Les fichiers sources ne sont jamais remplacés.
- Les copies `*_registered*` résiduelles d'anciennes versions ne sont pas chargées comme images lorsque leur fichier d'origine est présent dans le même dossier ; le message de chargement indique combien ont été ignorées.
- PNG/TIFF conservent les données 16 bits ; choisir explicitement JPG/BMP convertit en 8 bits.
- L'annulation attend l'arrêt du worker. Les fichiers validés restent disponibles ; les tâches partielles, échouées, annulées et non démarrées sont comptées séparément.

---

## 13. Exportation des résultats

### Enregistrer un résultat unique

1. Sélectionnez le résultat à enregistrer dans la liste des sorties
2. Allez dans **Fichier → Enregistrer** ou appuyez sur `Ctrl+S`
3. Choisissez le format et l'emplacement
4. Cliquez sur Enregistrer

Dans **Paramètres → Paramètres avancés → Nommage des sorties**, un préfixe de nom de fichier facultatif aide à identifier les résultats nouvellement rendus. L'exemple se met à jour pendant la saisie ; utilisez jusqu'à 40 lettres, chiffres, traits de soulignement ou traits d'union. Laissez vide pour conserver le nommage d'origine. Confirmez avec **OK** pour le mémoriser, ou annulez pour abandonner la modification. Le préfixe fait partie des nouveaux noms de la liste des sorties et de leurs noms d'export suggérés ; les résultats existants gardent leurs noms. Vous pouvez toujours modifier un nom dans la boîte d'enregistrement. Cette préférence ne renomme pas les fichiers sources et ne change pas les noms de sortie des lots.

### Format d'export par glisser-déposer

Faire glisser des résultats depuis la liste des sorties écrit des fichiers image au format choisi dans le menu contextuel de la liste (**Format de glisser-déposer** : JPG, PNG, WebP ou TIFF). Le choix est mémorisé.

### Enregistrer la pile recalée

1. Allez dans **Fichier → Enregistrer la pile → Pile recalée**
2. Choisissez le format d'enregistrement :
   - **Enregistrer comme dossier** : choisissez une destination et un encodage ; chaque pile reçoit un nouveau sous-dossier contenant ses noms d'images d'origine
   - **Enregistrer comme GIF** : crée un GIF animé de la séquence alignée

### Enregistrer la pile d'entrée

1. Allez dans **Fichier → Enregistrer la pile → Pile d'entrée**
2. Choisissez le format d'enregistrement :
   - **Enregistrer comme dossier** : enregistre les images d'origine comme images séparées
   - **Enregistrer comme GIF** : crée un GIF animé de la séquence d'entrée

### Formats d'export

- **JPG** : compressé, bon rapport qualité/taille
- **PNG** : sans perte, conserve les données 8 et 16 bits
- **BMP** : non compressé, convertit les données 16 bits en 8 bits
- **TIFF** : conserve les données 8 et 16 bits ; l'export multipage garde toute la pile dans un seul fichier
- **GIF** : animation à palette 8 bits ; une annulation laisse intacte une destination existante

---

## 14. Interface en ligne de commande

`pip install openfocus` inclut le modèle IA, ses poids et le décodage HEIC et RAW d'appareil photo. `dtcwt` est ignoré à partir de Python 3.13, où l'entrée DTCWT reste désactivée avec une infobulle.

OpenFocus peut s'exécuter sans interface graphique — utile pour les pipelines scriptés et l'intégration continue. Le script de console `openfocus` (installé par `pip install openfocus`) et `python main.py` depuis une copie du code exposent les mêmes commandes. Les options de fusion fonctionnent avec les deux formes ; les sous-commandes sont :

```bash
openfocus version                       # affiche la version
openfocus describe                      # JSON : méthodes, paramètres, capacités, commandes de contrôle
openfocus gui [--control] [--socket NAME] [paths…]   # démarre l'application de bureau
openfocus ctl <command> [flags]          # pilote une fenêtre en cours (voir AGENT_CONTROL_SPEC.md)
openfocus recipe create FILE --input … --output … [processing flags]
openfocus recipe run FILE [--dry-run]    # rejoue une recette headless enregistrée
openfocus retouch replay CHECKPOINT --base BASE --sources … --output RESULT
```

`openfocus gui` démarre en arrière-plan ; `--control` attend le canal de contrôle et imprime son JSON de connexion. `openfocus describe` et `version` ne nécessitent aucune entrée. Le workflow destiné aux agents est documenté dans [AGENT_CONTROL_SPEC.md](./AGENT_CONTROL_SPEC.md) et `skills/openfocus-control/SKILL.md`.

Depuis un environnement Python (ou à côté de l'exécutable packagé) :

```bash
# Fusionne un dossier avec la méthode IA par défaut
python main.py --input ./stack_folder --output ./result/fused.png

# DTCWT avec recalage ECC, 8 threads
python main.py -i ./stack_folder -o ./result/fused.png -m dtcwt -a ecc -t 8

# StackMFF V3 forcé sur le CPU avec une taille de tuile personnalisée
python main.py -i ./stack_folder -o ./result/fused.png -m stackmffv3 --cpu --tile-size 512

# Liste de fichiers explicite ; les fichiers vidéo sont aussi acceptés en entrée
python main.py -i img1.jpg img2.jpg img3.jpg -o fused.png -m gfgfgf

# Une seule pile Z TIFF multipage avec barre d'échelle calibrée par métadonnées
python main.py -i ./zstack.tif -o fused.png -m dtcwt --scale-bar

# Inspecte les images adjacentes similaires sans fusionner ni modifier l'entrée
python main.py -i ./stack_folder --suggest-duplicates
```

`--suggest-duplicates` accepte un dossier ou une liste ordonnée de fichiers image d'une seule image. Il imprime un objet JSON avec les tableaux `files` et `suggestions` ; `index` et `reference_index` de chaque suggestion sont des positions à base zéro dans `files`, et `difference` est la différence moyenne normalisée des échantillons. Ajoutez `--json report.json` pour enregistrer le même objet. Ce mode d'examen ne se combine pas avec `--output`, `--output-dir`, `--json-progress`, `--dry-run` ni les options de nettoyage du capteur. Le rapport compare les fichiers adjacents dans l'ordre d'entrée fourni, sans savoir quelles images vous pourriez exclure ensuite. Si vous excluez une référence, reconsidérez son successeur ou révisez les fichiers restants dans leur ordre prévu.

Par exemple, ajoutez `--align openfocus --alignment-report alignment.json` à un rendu unique. L'objet JSON contient `method`, `input`, `frame_count` et `entries`. Le `frame_index` de chaque entrée est à base zéro dans l'ordre d'entrée décodé, avant le tri de fusion ; les images multipage se développent dans l'ordre des pages et les images vidéo dans l'ordre de décodage. Les méthodes composites incluent des entrées séparées pour chaque étape. Le rapport est écrit de façon atomique après un traitement réussi et ne peut remplacer ni une entrée ni une autre sortie. Les modes lot, groupé et à blanc refusent cette option.

| Option | Description |
|--------|-------------|
| `--suggest-duplicates` | Signale les images adjacentes presque identiques pour examen manuel ; ne les fusionne ni ne les exclut |
| `--input, -i` | Dossier(s) d'images, fichier vidéo, un ou plusieurs fichiers image, ou TIFF multipage |
| `--output, -o` | Chemin de l'image de sortie (l'extension détermine le format) |
| `--output-dir, -d` | Mode lot : fusionne chaque dossier d'entrée et écrit `<dossier>.png` par pile |
| `--resume` | Mode lot : ignore les dossiers dont la sortie existe déjà |
| `--config FILE` | Mode lot : surcharges JSON par dossier, p. ex. `{"stackA": {"method": "dct"}}` |
| `--method, -m` | `stackmffv3` (défaut), `stackmffv2`, `stackmff`, `openfocus_a`, `openfocus_b`, `openfocus_c`, `guided_filter`, `dct`, `dtcwt`, `gfgfgf` |
| `--param KEY=JSON_VALUE` | Définit tout paramètre de fusion ; répétable et prioritaire sur les options héritées |
| `--align, -a` | `none` (défaut), `homography`, `ecc`, `both`, `openfocus` |
| `--align-param KEY=JSON_VALUE` | Définit tout paramètre de recalage ; répétable |
| `--align-shift-limit` | Recalage `openfocus` uniquement : décalage maximal accepté en fraction de la taille de l'image (0,08 par défaut ; au-delà, l'image reste non alignée) |
| `--alignment-report PATH` | Enregistre les résultats d'alignement par image en JSON pour un rendu unique non groupé avec alignement ; les lots et groupes ne sont pas pris en charge |
| `--exposure-match` | Égalise la moyenne de chaque image sur celle de la dernière avant la fusion (activé par défaut ; `--no-exposure-match` le désactive). Gains pour toute la pile, donc la sortie tuilée reste sans couture ; s'applique à toutes les méthodes et à la référence de luminosité de l'aligneur OpenFocus |
| `--base-frame` | Étape post-fusion facultative : aligne les basses fréquences de la sortie sur l'image `last`/`first` (`none` par défaut) |
| `--sort` | Ordre des images pour `openfocus_a/b/c` : `asc` (défaut, ordre des fichiers), `desc` (inverse un balayage de loin en proche), `off` |
| `--sharpen-output` | Applique un léger masque flou à la sortie fusionnée (désactivé par défaut) |
| `--dust-map WHITE` | Correction de champ plat (carte de poussière) à partir d'une prise de champ blanc défocalisée |
| `--defect-mask-path MASK` | Masque de défauts en coordonnées capteur réparé avant le recalage |
| `--sensor-denoise` | Réduction du bruit du capteur prudente avant le recalage |
| `--kernel, -k` | Taille du noyau (forcée impaire) ; contrôle l'échelle de réponse de netteté pour `openfocus_a` et `openfocus_b` ; C n'a pas de rayon |
| `--depth-smoothing VALUE` | `openfocus_b` uniquement : force du lissage des transitions de profondeur (1,2 par défaut) |
| `--blend-smoothing VALUE` | `openfocus_a` / `openfocus_c` uniquement : force du lissage des transitions (2,0 pour A, 3,0 pour C par défaut) |
| `--ai-mode {soft,hard}` | Sortie StackMFF V3 : mélange pondéré par les probabilités (défaut) ou image la plus probable par pixel |
| `--threads, -t` | Limite supérieure de threads de travail ; `0` = automatique (défaut), les valeurs positives plafonnent la concurrence |
| `--resource-preference` | `balanced` (défaut), `responsive` ou `maximum` ; contrôle l'ordonnancement des ressources |
| `--batch-size` | Limite supérieure du lot de tuiles IA (2 par défaut) ; l'inférence actuelle utilise une tuile par appel |
| `--cpu` | Force le CPU même si un GPU est disponible |
| `--downscale` | Largeur de réduction pour la détection de caractéristiques du recalage |
| `--tile-enabled` / `--no-tile-enabled` | Active la fusion par tuiles lorsque nécessaire (défaut : activée) |
| `--tile-size` | Taille de bloc de tuile pour les grandes images (défaut : automatique) |
| `--tile-overlap` | Chevauchement des tuiles en pixels (défaut : automatique) |
| `--tile-threshold` | Taille minimale d'image déclenchant le tuilage (défaut : automatique) |
| `--group-size` | Fusionne un dossier en groupes de ce nombre d'images ; `0` désactive le regroupement (défaut). Dossiers uniquement ; pas de `--depth-map`, d'entrée RAW/alpha/multipage ni d'image de référence OpenFocus non par défaut |
| `--group-overlap` | Images partagées par groupes consécutifs ; `0` par défaut, au plus la taille du groupe moins 2 |
| `--scale-bar` | Dessine une barre d'échelle (calibration détectée automatiquement dans les métadonnées TIFF) |
| `--scale-um` | Taille de pixel en µm/px, prioritaire sur les métadonnées (implique `--scale-bar`) |
| `--depth-map [path]` | Enregistre aussi une carte de position de mise au point colorisée (défaut `<output>_depth.png`) ; méthodes compatibles avec la profondeur uniquement, ni groupées ni transparentes. En mode lot, seul le chemin par défaut est autorisé |
| `--depth-colormap` | Palette pour `--depth-map` : turbo, viridis, jet, gray, … (turbo par défaut) |
| `--json FILE` | Écrit un résumé de résultat lisible par machine dans FILE à la fin |
| `--json-progress` | Imprime un objet JSON de progression par étape sur stdout (NDJSON) ; la prose va sur stderr |
| `--dry-run` | Valide les entrées et imprime le plan sans fusionner |

Il n'existe pas de valeur `--method none` : la CLI fusionne toujours. Utilisez l'application de bureau pour une sortie alignement seul.

### Recettes de traitement

Une recette est un fichier JSON portable d'options de traitement CLI avec des chemins relatifs à la recette. Créez-en une à partir des options que vous passeriez à une exécution normale, puis rejouez-la :

```bash
openfocus recipe create line.json --input ./stack --output ./out/fused.tif \
  --method dtcwt --align ecc --threads 8
openfocus recipe run line.json
openfocus recipe run line.json --dry-run
```

`recipe create --project project.ofproj --input ./stack --output ./out/fused.tif` capture les réglages de traitement d'un projet sur des entrées choisies explicitement ; les sources du projet peuvent avoir été déplacées. Les recettes refusent les contrôles transitoires tels que `--dry-run`, `--resume`, `--json` et `--config`, et exigent que la sortie soit hors de tous les dossiers d'entrée (et différente d'une entrée explicitement sélectionnée). Un masque de défauts est stocké par chemin relatif avec une empreinte SHA-256, de sorte qu'un masque manquant ou modifié fait échouer l'exécution. Les modifications de source du projet (rotation, retournement, champ plat) ne peuvent pas être capturées ; exportez d'abord la pile corrigée.

La capture du projet transfère à la recette CLI les réglages de fusion, de recalage, de luminosité, d'ordre, de netteté et de tuiles, et incorpore les réglages de traitement pour l'interface. Les exclusions d'images, l'échelle de chargement, les libellés et les modifications ne sont pas transférés ; préparez la nouvelle pile d'entrée en conséquence. Les options de regroupement peuvent être capturées depuis les arguments de la CLI.

Le format JSON de version 1 contient `version`, un tableau `input`, soit `output`, soit `output_dir`, et un objet `options`. Les chemins sont relatifs au fichier de recette ; déplacez la recette avec ses fichiers d'entrée pour conserver ces relations. La création capture les réglages de la CLI et les paramètres d'algorithme résolus. Les champs inconnus, les valeurs invalides et les versions non prises en charge sont rejetés. Une exécution à blanc vérifie l'entrée sans écrire de fichiers de résultat. Les fichiers de projet existants restent inchangés.

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

### Rejouer un point de contrôle de retouche

Utilisez les commandes d'enregistrement et de chargement des points de contrôle de l'espace de retouche avec la base d'origine et les sources traitées. Les sessions avec arrière-plan automatique, y compris les analyses réussies sans changement de pixels, enregistrent des instantanés ZIP v2 `.ofretouch` sans perte avec métadonnées JSON et données NumPy : l'image de travail exacte, les tuiles accessibles d'annulation/rétablissement, les états de diagnostic et la provenance des sources. Le chargement v2 ne recalcule pas l'arrière-plan. Les sessions limitées au pinceau et aux tonalités peuvent utiliser le JSON v1 compact ; les anciens points de contrôle de fond en JSON v1 restent rejouables.

La vérification des sources s'exécute en arrière-plan et peut être annulée. Avant de remplacer la destination, l'enregistrement recharge le fichier préparé et vérifie l'égalité exacte de l'image actuelle, des tuiles accessibles d'annulation/rétablissement et des états de diagnostic. Une source incompatible, un échec de vérification, une annulation ou un manque de mémoire laissent les modifications actuelles et tout point de contrôle existant inchangés.

Si la limite d'enregistrement du point de contrôle empêche une nouvelle modification, appliquez les modifications en cours et démarrez une nouvelle session de retouche pour continuer.

Enregistrez un point de contrôle de retouche depuis l'espace de retouche, puis fournissez l'image de base d'origine et chaque image source traitée dans le même ordre que la session :

```bash
openfocus retouch replay edits.ofretouch --base fused.png --sources frame1.png frame2.png --output retouched.tif
```

Les deux formats nécessitent toujours la base d'origine et toutes les images réellement utilisées pour la retouche, dans l'ordre de la session, après recalage et tout recadrage ROI ou autre traitement, avec leurs valeurs de pixel et profondeur de bits d'origine. Le chargement reconstruit les décalages d'image et le comportement d'alignement enregistrés et vérifie la base, les images ordonnées et les masques de validité ; les pixels, géométries ou masques incompatibles sont refusés. Chaque chemin doit contenir une image. La sortie doit être PNG ou TIFF et conserve la profondeur de bits de la base. Le point de contrôle et les images d'entrée ne peuvent pas servir de sortie.

Codes de sortie : `0` succès, `1` erreur de traitement, `2` erreur d'utilisation. `python main.py --help` affiche toutes les options ; lancer avec un simple chemin (`python main.py C:\stack`) ouvre toujours l'interface avec ce dossier chargé.

---

## 15. Raccourcis clavier

### Opérations sur les fichiers

| Raccourci | Action |
|----------|--------|
| `Ctrl+O` | Ouvrir un dossier |
| `Ctrl+Shift+O` | Ouvrir une vidéo |
| `Ctrl+Alt+O` | Ouvrir un projet |
| `Ctrl+Alt+S` | Enregistrer le projet |
| `Ctrl+S` | Enregistrer le résultat |
| `Ctrl+Shift+S` | Enregistrer la pile recalée |
| `Ctrl+W` | Vider la pile |
| `Ctrl+Q` | Quitter |
| `Ctrl+R` | Lancer le rendu |
| `Ctrl+Shift+R` | Comparer toutes les méthodes |
| `Ctrl+T` | Retoucher le résultat |
| `Ctrl+M` | Carte de profondeur |
| `Ctrl+E` | Basculer la comparaison par balayage |
| `Ctrl+Shift+B` | Traitement par lots |
| `Ctrl+Z` / `Ctrl+Y` | Annuler / Rétablir (modifications de pile) |

### Navigation

| Raccourci | Action |
|----------|--------|
| `Haut` / `Bas` (liste source active) | Image précédente / suivante |
| `Molette` | Zoom avant/arrière |
| `Ctrl + molette` | Zoom fin |
| `Double-clic` | Basculer ajustement/100 % |

### Général

| Raccourci | Action |
|----------|--------|
| `Espace` (maintenu, pointeur sur une image) | Mode déplacement temporaire |
| `Suppr` | Supprime l'élément de liste actif ; efface aussi la ROI lorsque le pointeur est sur l'image source |

---

## 16. Dépannage

### Problèmes courants

#### « Aucune image chargée »

- Vérifiez que vous avez d'abord importé un dossier d'images ou une vidéo
- Vérifiez que le dossier contient des formats d'image pris en charge

#### Les résultats de fusion sont flous ou dédoublés

- Activez le recalage (choisissez homographie / ECC / OpenFocus rapide)
- Assurez-vous que les images sont correctement alignées avant la fusion
- Essayez un autre algorithme de fusion
- Augmentez la taille du noyau pour un résultat plus lisse

#### IA indisponible

- Vérifiez ONNX Runtime et le modèle fourni.
- Vérifiez via **Aide → Informations sur l'environnement**

#### Erreurs de mémoire insuffisante

- Activez le traitement par tuiles dans **Paramètres → Paramètres avancés → Performances**
- Réduisez la taille de bloc de tuile
- Utilisez des noyaux plus petits
- Redimensionnez les images

#### Le recalage échoue

- Utilisez **OpenFocus (rapide)** pour un mouvement rigide borné, ou activez **Arrêter sur image rejetée** pour échouer tôt au lieu d'accepter des images non alignées
- Réduisez la largeur de réduction dans **Paramètres → Paramètres avancés → Recalage**
- Assurez-vous que les images ont assez de contraste et de caractéristiques
- Activez **Collecter les diagnostics d'alignement**, relancez le rendu et consultez **Paramètres → Paramètres avancés → Recalage → Diagnostics d'alignement** pour les images en échec

#### Traitement lent

- Laissez les threads sur automatique (`0`) ; la capacité CPU, le mode de ressources et la marge mémoire plafonnent déjà la concurrence
- Utilisez le mode de ressources Maximum pour un lot non surveillé, ou le filtre guidé / GFG-FGF au lieu d'une méthode IA
- Réduisez la résolution des images ou utilisez **Aperçu rapide** pendant les réglages

#### Avertissement Windows SmartScreen au premier lancement

La signature des binaires Windows dépend de la release concernée (voir les notes de version) ; la signature n'est pas garantie. Téléchargez-les depuis **https://openfocus.top** plutôt que depuis un miroir tiers. Si SmartScreen vous avertit, cliquez sur **Informations complémentaires → Exécuter quand même**. Le premier lancement après décompression peut prendre 30 à 60 secondes.

#### macOS indique que l'application « ne peut pas être ouverte » ou « est endommagée »

La version macOS n'est pas notarizée. Téléchargez la version Apple Silicon depuis **https://openfocus.top**. Si vous installez depuis le DMG en déplaçant `OpenFocus.app` dans Applications, exécutez :

```bash
xattr -cr /Applications/OpenFocus.app
```

Si vous installez depuis le terminal avec `install.sh`, l'application se trouve dans `~/Applications` ; exécutez alors `xattr -cr ~/Applications/OpenFocus.app`.

ou cliquez droit sur l'application et choisissez **Ouvrir**. Les versions macOS publiées sont pour Apple Silicon (M1/M2/M3/M4) ; les Mac Intel peuvent s'exécuter depuis les sources.

### Obtenir de l'aide

1. **Informations sur l'environnement** : allez dans **Aide → Informations sur l'environnement** pour vérifier votre installation
2. **Contacter le support** : allez dans **Aide → Nous contacter** pour les informations de support

---

## Annexe : détails des algorithmes de fusion

### Filtre guidé

Le filtre guidé réalise un lissage préservant les contours à l'aide d'une image de guidage. Il calcule des transformations linéaires locales entre le guidage et les images d'entrée, puis les applique pour produire la sortie. Cette méthode excelle à préserver les contours tout en réduisant le bruit.

### DCT (Discrete Cosine Transform)

La DCT transforme les images dans le domaine fréquentiel, où les décisions de fusion reposent sur la magnitude des coefficients. Les composantes haute fréquence (détails) sont conservées tandis que les basses fréquences (zones lisses) sont fusionnées selon l'énergie locale.

### DTCWT (Dual-Tree Complex Wavelet Transform)

La DTCWT fournit une décomposition multi-échelles et multidirectionnelle des images. Elle capture les caractéristiques à différentes échelles et orientations, ce qui permet des règles de fusion sophistiquées préservant à la fois la texture et la structure.

### GFG-FGF

Implémente Fu et al. (2024) : la carte de netteté grossière |I − mean(I)| est enrichie par le gradient à quatre voisins de type gaussien — le gabarit 3×3 T = [1 2 1; 2 4 2; 1 2 1]/16 donne P = I⊗T, et G = (|I(x±1,y) − P|)/2 + (|I(x,y±1) − P|)/2, les valeurs sous T0 = 0,005 étant écartées — puis affinée par un filtre guidé rapide. Chaque pixel choisit l'image au plus grand score de netteté (égalités : mesure de petite région de Sobel 3×3), et les cartes de décision binaires sont lissées par le filtre guidé rapide avant le mélange pondéré.

### Fusion IA (StackMFF V3)

Approche d'apprentissage profond utilisant un réseau neuronal entraîné sur de grands ensembles de paires d'images multifocus. Le réseau apprend automatiquement des stratégies de fusion optimales et fournit des résultats de pointe lorsqu'il est correctement configuré.

---

*OpenFocus est un projet open source. Pour les mises à jour, les rapports de bogues et les demandes de fonctionnalités, consultez le dépôt du projet.*
