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
10. [Retouche, étiquettes et annotations](#10-retouche-étiquettes-et-annotations)
11. [Paramètres et configuration](#11-paramètres-et-configuration)
12. [Traitement par lot](#12-traitement-par-lot)
13. [Exporter les résultats](#13-exporter-les-résultats)
14. [Interface en ligne de commande](#14-interface-en-ligne-de-commande)
15. [Raccourcis clavier](#15-raccourcis-clavier)
16. [Dépannage](#16-dépannage)

---

## 1. Introduction

OpenFocus est une application de bureau professionnelle de fusion d'images multi-focus, conçue pour les chercheurs, les ingénieurs et les professionnels du traitement d'image. Le logiciel traite des séquences d'images prises à des mises au point différentes à l'aide d'algorithmes avancés de recalage et de fusion, afin de produire des images entièrement nettes avec une profondeur de champ totale.

### Fonctionnalités principales

- **Fusion d'images multi-focus** : combine plusieurs images ayant des points de mise au point différents en une seule image entièrement nette
- **Algorithmes de fusion multiples** : choisissez parmi Filtre guidé, DCT, DTCWT, GFG-FGF et la fusion IA (apprentissage profond)
- **Recalage d'images** : alignez des séquences mal alignées avec les méthodes ECC ou Homographie
- **Vue de comparaison par balayage** : inspectez l'alignement et comparez les résultats avec un séparateur A/B
- **Fichiers de projet** : enregistrez et restaurez des sessions de travail complètes
- **TIFF multipage** : les piles Z dans un fichier unique sont chargées page par page et exportées dans un seul fichier
- **Carte de profondeur** : cartes de position de mise au point par pixel, avec des palettes pseudo-couleur personnalisées
- **Barre d'échelle** : barres d'échelle de qualité publication, avec calibration automatique µm/px
- **Récupération automatique après plantage** : la dernière session de travail est proposée à la restauration après un plantage
- **Multilingue** : anglais, 中文, 日本語, español, avec détection automatique de la langue du système
- **CLI sans interface** : pilotez OpenFocus depuis la ligne de commande pour l'automatisation et l'intégration continue
- **Export flexible** : enregistrez les résultats sous forme d'images individuelles, de dossiers ou d'animations GIF
- **Transformations d'images** : pivotez, retournez et redimensionnez les piles d'images
- **Ajout d'étiquettes** : ajoutez des étiquettes texte aux piles recalées et aux piles d'entrée
- **Paramètres persistants** : vos préférences, la langue et les piles récemment ouvertes sont conservées entre les sessions

### Formats d'entrée pris en charge

- Fichiers image : JPG/JPEG, PNG, BMP, TIFF, WebP et autres formats courants
- Fichiers vidéo : format MP4 (décodé automatiquement en séquences d'images)
- Dossiers contenant des séquences d'images numérotées
- Fichiers RAW d'appareil photo (DNG, CR2/CR3, NEF, ARW, RAF, ORF, RW2 et autres
  formats RAW courants). Le décodage utilise `rawpy` (LibRaw), que
  `pip install openfocus` installe et que les versions Windows/macOS embarquent

### Formats de sortie pris en charge

- Images individuelles : JPG, PNG, BMP, TIFF, WebP
- Séquences d'images : enregistrées sous forme de dossiers
- Animations : format GIF

---

## 2. Prise en main

### Lancer l'application

Après l'installation, lancez OpenFocus de l'une des façons suivantes :

1. **Windows (exécutable)** : double-cliquez sur le raccourci ou le fichier exécutable OpenFocus
2. **Code source Python** : exécutez `python main.py` dans votre environnement conda activé

Au démarrage, la fenêtre principale de l'application apparaît avec une interface en thème sombre.

### Flux de travail de démarrage rapide

1. Importez votre séquence d'images (Fichier → Ouvrir un dossier, ou par glisser-déposer)
2. Configurez les réglages de fusion dans le panneau de droite
3. Activez éventuellement le recalage pour l'alignement
4. Ajustez la taille du noyau si nécessaire
5. *(facultatif)* Cochez **Aperçu rapide** pour tester d'abord les paramètres sur un brouillon réduit
6. Cliquez sur **Lancer le rendu** pour démarrer le traitement
7. Exportez vos résultats

---

## 3. Vue d'ensemble de l'interface principale

L'interface d'OpenFocus se compose de trois zones principales :

### Panneau gauche : zone d'affichage des images

La partie gauche de la fenêtre affiche vos images en vue partagée :

- **Panneau de l'image source (haut)** : affiche la séquence d'images d'entrée
- **Panneau de l'image de résultat (bas)** : affiche les résultats de fusion ou de recalage
- **Curseur de navigation** : utilisez le curseur situé sous chaque panneau pour parcourir les images
- **Étiquettes d'information** : affichent la position de l'image courante et ses dimensions

Vous pouvez redimensionner les panneaux en faisant glisser la poignée du séparateur entre eux.

### Panneau droit : panneau de configuration

Le panneau de droite contient toutes les options de configuration, organisées verticalement :

1. **Réglages de fusion** : sélectionnez l'algorithme de fusion et la taille du noyau
2. **Réglages de recalage** : activez ou désactivez les méthodes d'alignement
3. **Boutons d'action** : Réinitialiser et Lancer le rendu
4. **Liste des images sources** : affiche toutes les images chargées avec leurs noms de fichier
5. **Liste de sortie** : affiche les résultats générés

### Barre de menus

La barre de menus supérieure donne accès à toutes les fonctions :

- **Fichier** : opérations d'import et d'export, et liste des fichiers récents
- **Édition** : transformations d'images et étiquettes
- **Outils** : traitement par lot, carte de profondeur et carte de poussière
- **Paramètres** : accélération GPU, langue, tuiles, recalage et nombre de threads
- **Aide** : informations sur l'environnement, nous contacter, recherche de mises à jour et dossier des journaux

---

## 4. Importer des images et des vidéos

### Ouvrir un dossier d'images

1. Allez dans **Fichier → Ouvrir un dossier** ou appuyez sur `Ctrl+O`
2. Sélectionnez un dossier contenant votre séquence d'images
3. Une boîte de dialogue d'échelle de chargement apparaît — choisissez l'échelle (100 % = pleine résolution)
4. Cliquez sur OK pour charger les images

Cochez **Ne plus demander** dans cette boîte de dialogue pour toujours charger à
l'échelle choisie sans confirmation ; **Paramètres → Demander l'échelle de
chargement à l'import** réactive la demande. Une pile chargée peut toujours
être réduite ensuite avec **Édition → Redimensionner**.

Le logiciel effectue automatiquement les opérations suivantes :
- Détecter et trier les fichiers image par nom
- Afficher la première image dans le panneau source
- Remplir la liste des images sources avec toutes les images

### Ouvrir des fichiers vidéo

1. Allez dans **Fichier → Ouvrir une vidéo** ou appuyez sur `Ctrl+Shift+O`
2. Sélectionnez un fichier vidéo (MP4, AVI, MOV, MKV, WMV, FLV, WEBM)
3. La vidéo est automatiquement décodée en images individuelles
4. Chaque image rejoint la pile d'images

### Fichiers de projet

Fichier → Enregistrer le projet enregistre toute votre session de travail dans un
fichier `.ofproj` : la liste exacte des fichiers sources (dans l'ordre),
l'échelle de chargement, la méthode de fusion, les options de recalage, la
taille du noyau, les réglages de tuiles/threads/recalage, les configurations
d'étiquettes et l'image courante. Fichier → Ouvrir un projet… reconstruit
cette session à l'identique (les images sources sont référencées, pas copiées —
laissez-les en place). Le titre de la fenêtre indique le projet ouvert.

### Pile de démonstration

La boîte de dialogue de bienvenue propose **Charger la pile de démo** : six
photos d'exemple fournies, issues d'une véritable pile de mise au point, pour
essayer immédiatement l'ensemble du flux de travail sans vos propres images.
Utile aussi pour vérifier une installation.

### Courbe de netteté

Une petite courbe au-dessus du curseur de navigation source affiche la netteté
par image (normalisée). Les images floues apparaissent comme des creux marqués
d'un point rouge. Cliquez n'importe où sur la courbe pour accéder à cette
image ; le survol affiche le numéro de l'image.

### Piles TIFF multipage

Les logiciels de microscopie (ZEN, NIS-Elements, ImageJ/Fiji) exportent souvent
une pile Z entière dans UN SEUL fichier TIFF multipage. OpenFocus lit chaque
page d'un tel fichier comme une image distincte de la pile d'images — déposez ou
ouvrez le fichier et la pile apparaît complète (les images sont nommées
`<fichier>_page_001`, etc.). Les TIFF d'une seule page se comportent exactement
comme avant. L'inverse fonctionne aussi :
**Fichier → Enregistrer la pile → (Pile recalée / Pile d'entrée) → Enregistrer en TIFF multipage**
écrit la pile entière dans un seul `.tif`, en préservant la profondeur 16 bits.

### Barre d'échelle (figures de publication)

Paramètres → Barre d'échelle… configure une barre d'échelle incrustée dans les
images exportées. La taille du pixel (µm/px) est détectée automatiquement dans
les métadonnées TIFF — convention `unit=micron` d'ImageJ, `PhysicalSizeX` d'OME-TIFF,
ou une unité de résolution explicite en centimètres — de sorte que les piles de
microscopie (y compris les TIFF multipage) se calibrent d'elles-mêmes. La
résolution d'imprimante (72/300 dpi et autres) n'est délibérément *pas* traitée
comme une calibration, et une valeur saisie manuellement l'emporte toujours sur
les métadonnées détectées : une barre fausse d'apparence plausible dans une
figure est pire que pas de barre du tout. Choisissez le coin
et la couleur (Auto inverse par rapport au fond) ; la boîte de dialogue affiche un aperçu
en direct. Une fois la barre activée, chaque export (image seule, piles de dossiers,
TIFF multipage, GIF, export global, traitement par lot) comporte une barre de style publication :
une longueur arrondie 1-2-5 (environ 12 % de la largeur de l'image) avec un texte
contouré et lisible. La CLI la prend en charge via `--scale-bar` (calibration automatique) et
`--scale-um <um/px>`.

### Carte de profondeur (position de mise au point)

**Outils → Carte de profondeur…** estime les positions de mise au point avec la mesure
d’activité de la méthode choisie (DCT, DTCWT, GFG-FGF, StackMFF V3, OpenFocus B/C).
Ce n’est ni une surface mesurée ni une carte exacte de l’origine des pixels fusionnés.
Choisissez les images actuellement cochées ou la source enregistrée de la sortie sélectionnée.
Une provenance manquante n’est jamais remplacée par la pile actuelle. Calcul, texture et
export restent liés au même instantané de la source.

Les six méthodes exigent un clic explicite sur **Calculer**. Ouvrir la fenêtre ou changer
la mesure, la source, le recalage ou les paramètres ne lance aucun calcul. L’éditeur se
trouve juste sous le sélecteur de mesure : les noyaux médian DCT et initial GFG prennent
des valeurs impaires de 1 à 51 (7 par défaut) ; B utilise une fenêtre de variance et C une
fenêtre d’énergie passe-bande, impaires de 1 à 21 px (11 par défaut). Les niveaux DTCWT
sont modifiables ici dans un brouillon local initialisé avec les paramètres de la source.
StackMFF V3 affiche son propre mode de sortie IA (mélange doux par défaut, ou
sélection dure) et une explication, sans noyau numérique : le mélange doux estime une
position continue pondérée par les probabilités, la sélection dure prend l’image la plus
probable par pixel. Ni l’une ni l’autre n’est une distance physique. Les brouillons partent des paramètres de la sortie correspondante ou
des valeurs mémorisées par méthode dans la fenêtre principale. Ils sont conservés par
source et méthode pendant cette ouverture, sans modifier fenêtre principale, projet ni sortie.

Modifier les réglages de calcul marque durablement l’ancienne carte 2D comme **non à jour**,
efface la surface 3D et bloque tous les exports, y compris les données brutes, la profondeur
physique et l’OBJ, jusqu’à la réussite d’un nouveau calcul. Le travail actif est annulé de
façon coopérative. Ensuite, cliquez à nouveau sur **Calculer** : aucun calcul n’est mis en
attente ou relancé automatiquement. Calculer reste visible et est désactivé pendant le
calcul, l’annulation et l’export du modèle.

- **Palette** : niveaux de gris (standard de publication) plus 15 palettes
  scientifiques (Turbo, Viridis, Magma, …) et
  les vôtres — *Nouveau schéma personnalisé…* construit un dégradé à partir d'un nombre
  quelconque de points de couleur (double-cliquez sur un point pour modifier sa position et sa couleur) ; les schémas
  personnalisés sont enregistrés et proposés dans la liste déroulante.
- **Inverser / Gamma / Lissage** : inversez l'axe de profondeur, rééquilibrez la tonalité
  (gamma < 1 éclaircit les indices faibles) et supprimez le grain avec une passe
  préservant les contours.
- **Superposition** : mélange la carte 2D avec la sortie associée à opacité réglable,
  uniquement si leurs domaines de coordonnées correspondent.
- **Export** : la version colorisée accepte PNG/JPEG/TIFF/BMP. Le TIFF brut 16 bits
  code les indices de mise au point normalisés de 0 à 65535, sans être des données
  flottantes sans perte ni des distances physiques. JPEG/BMP sont refusés pour les données brutes.
- **Modèle 3D / Texture 3D** : le grand côté du maillage compte 600 sommets par défaut.
  Chaque nouvelle ouverture choisit **Automatique (couleurs réelles)** : la sortie associée
  est utilisée uniquement si son identité et son domaine de coordonnées/ROI correspondent
  à la carte terminée. Sinon, la texture vient de la première image participante du calcul
  terminé, après recalage et selon l’ordre effectivement utilisé. Le nom de la sortie ou
  de l’image utilisée est affiché. Vous pouvez choisir explicitement la sortie correspondante,
  une image participante ou **Pseudo-couleurs de profondeur** ; une texture indisponible est
  signalée. L’aperçu 3D et le PNG de texture OBJ partagent la même image. La carte 2D garde sa
  propre palette et sa superposition ; palette et Gamma ne recolorent pas les photos.
  Seule la texture en pseudo-couleurs suit la palette, le Gamma et l’inversion. L’aperçu au
  repos et l’export partagent le même champ de hauteurs échantillonné, sans lissage caché
  supplémentaire. Le relief relatif s’exprime en unités d’image.

Le calcul s'exécute en arrière-plan avec progression et annulation ; la méthode
IA réutilise le même traitement par tuiles économe en mémoire que la fusion IA,
ce qui permet de traiter les piles profondes. L'équivalent en CLI est `--depth-map [path]` avec `--depth-colormap`.

### Récupération automatique après plantage

OpenFocus conserve en continu un instantané de votre session de travail (pile,
réglages, étiquettes, image courante). Si l'application se ferme de façon inattendue — plantage, coupure de courant ou arrêt forcé — le lancement suivant le détecte et
propose **Restaurer la session** pour revenir exactement là où vous en étiez. Les
fermetures normales n'affichent jamais cette demande. Les données de récupération se trouvent à côté de vos
paramètres (les installations portables les conservent à côté de l'exécutable, elles vous suivent donc).

### Fichiers récents

Chaque pile que vous ouvrez est mémorisée dans **Fichier → Ouvrir récent** (8 entrées au maximum). Appuyez sur `Alt+1`–`Alt+8` pour rouvrir une entrée directement. Rouvrir la même pile la place en tête de liste ; les entrées dont les fichiers n'existent plus peuvent être retirées après confirmation, et **Effacer la liste des récents** vide le menu.

### Glisser-déposer

Vous pouvez aussi faire glisser un dossier directement sur la fenêtre de l'application pour importer des images. Déposer d'autres fichiers sur une pile existante les ajoute à la suite (après vérification des dimensions).

### Orientation EXIF

Les fichiers JPEG/TIFF d'appareil photo portant une balise EXIF Orientation sont
pivotés ou corrigés automatiquement lors de l'import, afin que les prises de vue en
portrait s'affichent à l'endroit.

### Sources 16 bits

Les entrées PNG/TIFF 16 bits sont détectées automatiquement et préservées de bout en bout :
la fusion s'effectue en pleine profondeur et les résultats peuvent être réenregistrés en PNG/TIFF 16 bits
(les exports JPG/BMP sont convertis en 8 bits, car ces formats ne peuvent pas stocker 16 bits).

Les fichiers RAW d'appareil photo sont développés en 16 bits (balance des blancs de l'appareil, gamma sRGB)
avec l'éclairage automatique désactivé, afin que chaque image d'une pile reçoive la même
mise en correspondance d'exposition ; une exposition automatique par image ferait scintiller la luminosité
d'une image à l'autre. Enregistrez en TIFF ou PNG pour conserver les données 16 bits.

### Vider la pile

Pour retirer toutes les images chargées, allez dans **Fichier → Vider la pile** ou appuyez sur `Ctrl+W`.

---

## 5. Affichage et navigation dans les images

### Parcourir la pile d'images

- **Curseur** : faites glisser le curseur sous le panneau source pour parcourir les images
- **Liste de fichiers** : cliquez sur un nom de fichier dans la liste des images sources du panneau droit
- **Clavier** : utilisez les flèches gauche et droite pour naviguer

### Zoom et déplacement

- **Zoom avant/arrière** : molette de la souris au-dessus de l'image
- **Zoom fin** : maintenez `Ctrl` + molette pour des incréments plus petits
- **Ajuster à la fenêtre** : double-cliquez sur l'image pour basculer entre l'affichage ajusté et 100 %
- **Déplacer** : cliquez et faites glisser pour déplacer l'image dans le panneau

### Navigation synchronisée

Lorsque vous naviguez dans le panneau source, le panneau de résultat affiche l'image de résultat correspondante, si elle est disponible.

---

## 6. Vue de comparaison par balayage

La vue Balayage superpose deux images dans un même cadre avec un séparateur vertical déplaçable, ce qui rend immédiatement visibles les désalignements et les différences de qualité de part et d'autre du séparateur.

### Ouvrir la vue Balayage

1. Générez au moins un résultat (fusion ou recalage)
2. Appuyez sur le bouton **Balayage** dans le coin supérieur droit de la barre de titre du panneau de résultat
3. La zone de résultat bascule en vue de comparaison ; appuyez à nouveau sur **Balayage** pour revenir

### Lire la vue

- **Côté A (à gauche du séparateur)** : une image source — par défaut, elle suit le curseur du panneau source
- **Côté B (à droite du séparateur)** : le rendu le plus récent
- **Séparateur blanc + poignée ronde** : faites glisser horizontalement pour déplacer la séparation
- Faire glisser les bords qui franchissent le séparateur révèle des erreurs de recalage difficiles à voir dans des panneaux côte à côte

### Choisir ce que l'on compare

La barre en bas de la vue de comparaison comporte un groupe par côté :

- **Côté A** — cochez **Suivre la source** (par défaut) pour suivre le curseur du panneau source, ou décochez pour fixer le côté A sur une image donnée avec son propre curseur
- **Côté B** — cochez **Dernier résultat** (par défaut) pour toujours afficher le rendu le plus récent, ou décochez pour parcourir l'historique de vos sorties avec son curseur

Les deux côtés se mettent à jour en direct : les nouveaux rendus, les déplacements du curseur source et les suppressions d'images sont reflétés immédiatement.

### Zoom et déplacement

La vue de comparaison applique une seule transformation aux deux images :

- **Zoom** : molette de la souris (le zoom s'effectue autour du curseur)
- **Déplacer** : cliquez et faites glisser n'importe où sauf sur le séparateur
- **Réinitialiser** : double-clic

Le mode Balayage et le mode ROI partagent le panneau de résultat : entrer dans l'un fait sortir de l'autre. Vider la pile fait également sortir du mode Balayage.

---

## 7. Recalage d'images

Le recalage d'images corrige le désalignement spatial entre les images de votre pile. Il est indispensable lorsque les images présentent de légers décalages ou des changements de perspective.

### Méthodes de recalage

#### ECC (Enhanced Correlation Coefficient)

- Utilise une optimisation pour trouver le meilleur alignement d'après la corrélation
- Adapté aux désalignements subtils
- Fournit une précision subpixel
- Plus lent que l'homographie mais plus précis

#### Homographie

- Repose sur la mise en correspondance de points caractéristiques (SIFT/ORB)
- Gère des transformations géométriques plus importantes
- Plus rapide que l'ECC pour un alignement initial
- Adapté aux images présentant d'importants changements de perspective

#### OpenFocus (rapide)

- Alignement par similarité bornée (translation + rotation + échelle),
  avec égalisation de la luminosité par image et recadrage automatique
- Moins d'une seconde par image — bien plus rapide que l'ECC ou l'homographie
- Les images décalées de plus de **8 % de leur taille restent non alignées**
  plutôt que partiellement corrigées. Modifiez ce seuil avec l'option CLI
  `--align-shift-limit`
- Pour une géométrie non rigide ou extrême, préférez l'ECC ou l'homographie

### Activer le recalage

1. Dans le panneau droit, repérez le groupe **Recalage**
2. Choisissez une méthode dans la liste déroulante : **Aucun alignement**, **OpenFocus (rapide)**,
   **ECC** ou **Homographie**
3. L'alignement rapide OpenFocus peut aussi être enchaîné avant l'ECC ou
   l'homographie (d'abord une passe globale grossière, puis la passe locale précise)

### Quand utiliser le recalage

- Vos images présentent un désalignement visible
- Les images ont été prises à main levée
- La scène présente de la parallaxe (les objets proches se décalent d'une image à l'autre)
- Les résultats montrent des artefacts de dédoublement ou de fantôme

### Paramètres de recalage

Accédez aux réglages supplémentaires via **Paramètres → Paramètres de recalage** :
- **Largeur de réduction** : contrôle la résolution de prétraitement (1024 px par défaut)
- Des valeurs plus faibles = traitement plus rapide, précision éventuellement moindre
- Des valeurs plus élevées = traitement plus lent, précision éventuellement meilleure

---

## 8. Méthodes de fusion d'images

OpenFocus propose huit algorithmes de fusion (plus un mode d'alignement seul). Chacun présente des caractéristiques différentes adaptées à divers types d'images.

### Filtre guidé (par défaut)

- **Algorithme** : utilise un filtrage guidé pour une fusion préservant les contours
- **Idéal pour** : usage général, compromis équilibré entre vitesse et qualité
- **Avantages** : bonne préservation des contours, traitement rapide
- **Paramètre** : taille du noyau (réglable de 1 à 51, valeurs impaires uniquement)
  - Noyau plus petit : plus de détail local, mais potentiellement plus de bruit
  - Noyau plus grand : résultats plus doux, meilleure suppression du bruit

### DCT (Discrete Cosine Transform)

- **Algorithme** : fusion dans le domaine fréquentiel à l'aide des coefficients DCT
- **Idéal pour** : images riches en textures, imagerie scientifique
- **Avantages** : préserve les textures fines, fondement mathématique solide
- **Paramètre** : la taille du noyau contrôle la fenêtre de traitement

### DTCWT (Dual-Tree Complex Wavelet Transform)

- **Algorithme** : fusion multi-échelle dans le domaine des ondelettes
- **Idéal pour** : scènes complexes comportant plusieurs niveaux de détail
- **Avantages** : excellente analyse multi-échelle, bonne directionalité
- **Règle de fusion** : une décision par pixel et par niveau pour toutes les orientations et
  canaux de couleur (à N voies, avec vérification de cohérence majoritaire) ; la bande passe-bas
  suit la décision la plus grossière au lieu de moyenner toutes les images
- **Paramètre** : niveaux d’ondelettes (1–6, 4 par défaut) dans le panneau
  droit. La méthode n’a pas de noyau spatial : le curseur de noyau est masqué.

### GFG-FGF (Gaussian-like Four-neighbourhood Gradient + Fast Guided Filter)

- **Algorithme** : implémentation de Fu et al. (*Laser & Optoelectronics
  Progress*, 2024) : une carte de mise au point par différence de flou |I − mean(I)| est enrichie
  par l'opérateur de gradient à quatre voisins de type gaussien (GFG), affinée
  par un filtre guidé rapide, décidée pixel par pixel (les égalités étant tranchées par une mesure
  de petite région de Sobel 3×3), puis les cartes de décision binaires sont lissées une nouvelle fois
  par le filtre guidé rapide avant le mélange pondéré.
- **Idéal pour** : microscopie et images présentant des zones nettes bien délimitées
- **Avantages** : rapide, frontières de zones nettes précises ; la mémoire n'augmente pas
  avec le nombre d'images
- **Paramètre** : Noyau = taille du filtre moyenne de la carte de mise au point grossière

### StackMFF-V3 (apprentissage profond)

- **Algorithme** : classification de la source pixel par pixel (nécessite PyTorch). Le
  réseau prédit, pour chaque pixel, de quelle image d'entrée il provient, puis
  soit mélange les images selon ces probabilités, soit choisit une seule image
- **Mode de sortie IA** (panneau droit, uniquement pour cette méthode) :
  - **Mélange doux (pondéré)** — par défaut ; chaque pixel est un mélange pondéré par les probabilités
    des images. Transitions douces, sans coutures marquées ; correspond à la configuration
    publiée dans l'article
  - **Sélection stricte (image unique)** — chaque pixel est copié depuis l'image la plus
    probable. Sélection plus nette, sans interpolation entre images ; une
    couture au niveau du pixel peut apparaître là où deux images sont à égalité
- **Idéal pour** : résultats de la plus haute qualité lorsque le GPU est disponible
- **Avantages** : invariant à l'ordre (l'ordre des images ne change jamais le résultat),
  gère n'importe quelle longueur de pile, aucun réglage par méthode, optimisation automatique
- **Prérequis** : installation de PyTorch, GPU CUDA facultatif (sans torchvision)
- **Remarque** : peut être indisponible si PyTorch n'est pas installé

### OpenFocus A / B / C

- **Algorithme** : moyenne pondérée par la variance, sélection par carte de profondeur de variance, et
  fusion par pyramide laplacienne
- **OpenFocus A** : moyenne pondérée par la variance avec a priori sur la dernière image —
  rapide, couleurs stables sur les piles courtes
- **OpenFocus B** : sélection par carte de profondeur de variance — prend l'image la plus nette
  par pixel ; idéal sur les surfaces lisses et en microscopie
- **OpenFocus C** : mélange par bande de fréquence de la pyramide laplacienne — rendu naturel sur
  les sujets texturés, renforce le contraste local
- **Paramètres** : A et B utilisent une Fenêtre de variance (curseur de noyau) ; C n'a pas de
  rayon. Toutes les trois prennent un sigma de Lissage.

**Lissage** : une seule zone numérique sous le curseur de noyau, dont la signification dépend de la
méthode sélectionnée (plage 0–8, pas de 0,2) :

| Méthode | Libellé | Défaut | Effet |
|---|---|---|---|
| OpenFocus A | Lissage du mélange (σ) | 2,0 | Lissage gaussien de la carte de pondération par image — ampleur du mélange des zones nettes des images voisines |
| OpenFocus B | Lissage de profondeur (σ) | 1,2 | Lissage gaussien de la carte de profondeur — largeur de la transition entre images |
| OpenFocus C | Lissage du mélange (σ) | 3,0 | Lissage gaussien de la carte de pondération de chaque bande de la pyramide |

0 conserve le choix le plus net par pixel (détail le plus fin, mais du grain sur les
piles bruitées) ; des valeurs plus grandes adoucissent les transitions et les halos. Chaque méthode mémorise
sa propre valeur d'une session à l'autre, et le traitement par lot applique la même règle.

C n'a pas de Rayon : la méthode C d'Helicon n'expose que le Lissage, le curseur de noyau
est donc masqué pour C plutôt que d'agir dans le vide.

CLI : `--depth-smoothing SIGMA` (B) et `--blend-smoothing SIGMA` (A, C).

**Les options de luminosité sont propres à chaque méthode.** *Harmoniser l'exposition avant la fusion* et
*Supprimer la dérive de luminosité* sont mémorisés séparément pour chaque méthode de fusion,
car ils répondent à une question propre à chaque pile plutôt qu'à une préférence globale.
Les deux sont activés par défaut pour OpenFocus A/B/C (qui n'ont pas de calibration d'exposition
intégrée) ; les autres méthodes conservent les valeurs par défaut historiques (harmonisation de l'exposition
activée, suppression de la dérive désactivée). Changer de méthode affiche les valeurs propres à cette méthode.

### Alignement seul (sans fusion)

Sélectionnez **Alignement seul (sans fusion)** comme méthode pour recaler la pile
et envoyer les images alignées dans la liste de sortie sans fusion — pratique pour
évaluer la qualité du recalage ou exporter des images alignées.

### Sélectionner une méthode de fusion

1. Dans le panneau droit, repérez le groupe **Fusion**
2. Choisissez une méthode dans la **liste déroulante** (une seule est toujours sélectionnée ;
   la première entrée recale sans fusionner)
3. Ajustez la taille du noyau si applicable — le libellé indique ce que le noyau
   signifie pour la méthode sélectionnée
4. Cliquez sur **Lancer le rendu** pour traiter

### Guide de comparaison des algorithmes

| Algorithme | Vitesse | Qualité | Idéal pour |
|-----------|-------|---------|----------|
| Filtre guidé | Rapide | Bonne | Usage général |
| DCT | Moyenne | Bonne | Textures |
| DTCWT | Moyenne | Très bonne | Scènes complexes |
| GFG-FGF | Rapide | Bonne | Zones nettes |
| Fusion IA (StackMFF-V3) | Lente (GPU) | Excellente | Qualité maximale |
| OpenFocus A | Très rapide | Très bonne | Piles courtes, stabilité des couleurs |
| OpenFocus B | Très rapide | Très bonne | Surfaces lisses, microscopie |
| OpenFocus C | Rapide | Très bonne | Sujets texturés, rendu naturel |

### Aperçu rapide (rendu brouillon)

Régler les paramètres sur de grandes piles est lent en pleine résolution. Cochez
**Aperçu rapide** au-dessus du bouton de rendu : OpenFocus calcule d'abord un
brouillon réduit (côté le plus long 1200 px) — généralement plusieurs fois plus rapide,
recalage compris. L'aperçu est affiché avec la mention « Aperçu » dans
le résumé de la barre d'état après le rendu et n'est **pas** ajouté à l'historique des sorties. Décochez
la case et relancez le rendu pour obtenir le résultat en pleine résolution. Le mode aperçu
ne s'applique pas aux rendus ROI et ne pollue jamais le cache de recalage
en pleine résolution.

---

## 9. Transformations d'images

### Pivoter

1. Allez dans **Édition → Pivoter**
2. Choisissez le type de rotation :
   - **90° horaire** : pivote toutes les images de 90 degrés vers la droite
   - **90° antihoraire** : pivote toutes les images de 90 degrés vers la gauche
   - **180°** : retourne les images sens dessus dessous

### Retourner

1. Allez dans **Édition → Retourner**
2. Choisissez le type de retournement :
   - **Retourner horizontalement** : effet miroir de gauche à droite
   - **Retourner verticalement** : effet miroir de haut en bas

### Redimensionner

1. Allez dans **Édition → Redimensionner**
2. Une boîte de dialogue s'ouvre avec les options de redimensionnement
3. Saisissez les nouvelles dimensions ou un pourcentage d'échelle
4. Toutes les images de la pile sont redimensionnées proportionnellement

### Ordre des transformations

Les transformations s'appliquent à l'ensemble de la pile d'images, en conservant l'alignement entre les images.

---

## 10. Retouche, étiquettes et annotations

### Retoucher le résultat (Édition → Retoucher le résultat…, `Ctrl+T`)

La fusion peut encore laisser un défaut local — un halo autour d'un contour très contrasté,
un poil provenant de la mauvaise image, du bruit dans une zone uniforme. Le pinceau de
retouche copie les pixels d'une image source sur le résultat :

1. Générez un résultat (ou sélectionnez-en un dans la liste de sortie ; un clic droit → *Retoucher
   le résultat…* fonctionne aussi) et appuyez sur `Ctrl+T`.
2. Peignez sur le défaut avec le bouton gauche de la souris. Avec **Choisir automatiquement
   l'image la plus nette par trait** activé (par défaut), chaque trait sélectionne d'abord
   l'image qui contient le plus de détail sous le pinceau — corriger un halo ne demande généralement
   qu'un seul glissement. Désactivez l'option pour choisir vous-même l'image avec le curseur
   ou `,` / `.`.
3. Ajustez le **rayon** (`[` / `]`), la **dureté** (des bords doux masquent la couture) et
   l'**opacité**. Les touches qui se chevauchent dans un même trait ne s'accumulent jamais.
4. **Afficher l'image source** (`Tab`) prévisualise l'image depuis laquelle vous peignez.
   La molette zoome, un glissement droit/milieu déplace, `F` ajuste l'image.
5. `Ctrl+Z` / `Ctrl+Y` annulent et rétablissent des traits entiers. **Appliquer comme nouvelle sortie**
   ajoute l'image retouchée à la liste de sortie (suffixe `_Retouched`) ; la
   sortie d'origine reste intacte.

Le pinceau utilise les images alignées qui ont produit le résultat, il coïncide donc
avec lui ; les résultats 16 bits restent en 16 bits. Les rendus ROI et les piles redimensionnées ne peuvent pas
être retouchés (leurs images ne correspondent plus à la géométrie du résultat).

### Ajouter des étiquettes

1. Assurez-vous d'avoir généré des résultats ou chargé des images
2. Allez dans **Édition → Ajouter une étiquette**
3. Saisissez le texte de l'étiquette dans la boîte de dialogue
4. Les étiquettes sont ajoutées à la fois à la pile recalée et à la pile d'entrée

### Supprimer des étiquettes

- **Supprimer les étiquettes de la pile recalée** : retire les étiquettes des images alignées
- **Supprimer les étiquettes de la pile d'entrée** : retire les étiquettes des images d'origine

Accédez à ces options via le sous-menu **Édition → Supprimer l'étiquette**.

---

## 11. Paramètres et configuration

### Paramètres des tuiles (optimisation de la mémoire)

Pour les grandes images, OpenFocus utilise un traitement par tuiles afin d'éviter les problèmes de mémoire.

Accessible via **Paramètres → Paramètres des tuiles** :

- **Taille des blocs** : taille de chaque tuile de traitement (1024 px par défaut)
- **Chevauchement des tuiles** : chevauchement entre tuiles pour un mélange sans couture (256 px par défaut)
- **Seuil de tuilage** : taille d'image au-delà de laquelle les tuiles sont utilisées (2048 px par défaut)

**Recommandations** :
- Blocs plus petits : moins de mémoire utilisée, traitement plus lent
- Blocs plus grands : plus de mémoire utilisée, traitement plus rapide
- Augmentez le chevauchement si des coutures visibles apparaissent dans les résultats

### Paramètres de recalage

Accessibles via **Paramètres → Paramètres de recalage** :

- **Largeur de réduction** : résolution de prétraitement pour le recalage
  - Par défaut : 1024 px
  - Plus faible pour la vitesse, plus élevée pour la précision

### Référence de performances

Consultez [PERFORMANCE.md](./PERFORMANCE.md) pour les débits et les chiffres de mémoire mesurés
sur de grandes piles réelles, ainsi que des conseils pratiques pour les piles profondes et
les entrées haute résolution.

### Paramètres du nombre de threads

Contrôlez le traitement parallèle sur le CPU :

Accessibles via **Paramètres → Paramètres des threads** :

- `0` sélectionne la planification automatique, valeur initiale des nouveaux réglages.
- Les valeurs positives plafonnent les threads ; les limites enregistrées sont conservées.
- La capacité CPU, le mode de ressources et la mémoire disponible peuvent réduire la concurrence.

### Activation de l'accélération GPU

OpenFocus n'utilise le GPU (CUDA/MPS) que pour le modèle de réseau de neurones IA (StackMFF-V3) ; les algorithmes classiques s'exécutent toujours sur le CPU.

- Activez ou désactivez via **Paramètres → Accélération GPU**
- Décocher force tout le traitement sur le CPU — utile lorsque les pilotes GPU posent problème
- Le panneau d'état affiche `CPU*` tant que le GPU est désactivé
- Le choix s'applique au rendu suivant et est mémorisé

### Taille de lot IA

Accessible via **Paramètres → Taille de lot IA** :

- Le plafond enregistré est de 2 par défaut ; une seule tuile est actuellement traitée par appel sur chaque appareil.
- Augmenter ce plafond n'active pas les lots parallèles sur GPU.

### Thèmes

Deux thèmes d'interface sont disponibles : **Sombre (par défaut)** et **Clair**. Basculez via
**Paramètres → Thème** ; le choix s'applique immédiatement (la barre de titre Windows suit)
et est mémorisé. Au premier lancement, l'utilisateur choisit un thème dans la boîte de dialogue
de bienvenue.

### Mode portable

Créez un fichier vide nommé `OpenFocus.portable` à côté de `OpenFocus.exe`
(l'archive portable en contient déjà un) : toutes les données utilisateur — paramètres, journaux,
fichiers récents — sont alors stockées dans un dossier `OpenFocusData` à côté de l'application
au lieu de votre profil utilisateur, si bien que l'installation tient sur une clé
USB. Supprimez le marqueur pour revenir au profil normal.

### Métadonnées EXIF

Les métadonnées source (modèle d'appareil, exposition, orientation, etc.) sont conservées :
l'orientation est appliquée au chargement, et les exports PNG/TIFF/WebP intègrent automatiquement
l'EXIF d'origine. Les exports JPG/BMP ne peuvent pas stocker l'EXIF complet et
sont écrits sans lui.

### Entrée HEIC/HEIF

Les photos d'iPhone se chargent directement. Nécessite le paquet facultatif `pillow-heif`
(`pip install pillow-heif` ; inclus dans les versions packagées).

### Mises à jour

Les fichiers de projet (`.ofproj`) se comportent comme un document : la barre de titre affiche
le nom du projet avec un `*` tant que des modifications ne sont pas enregistrées, et la fermeture de l'application
ou l'ouverture d'un autre projet demande confirmation avant de les abandonner. Le projet stocke
la liste des images, l'échelle de chargement, toutes les options de rendu, les étiquettes et l'image
courante — les images elles-mêmes sont référencées, pas copiées.

Aide → Rechercher des mises à jour compare votre version à celle des versions GitHub.
**Télécharger et installer** récupère le nouvel installateur dans l'application (avec progression)
et le lance ; Ouvrir la page de téléchargement mène à la liste des versions. La vérification
s'effectue silencieusement au plus une fois par jour.

### Cache de recalage

OpenFocus peut stocker les résultats de recalage dans un dossier `.openfocus_cache`
à côté de la pile source. La réouverture de la même pile (fichiers inchangés)
évite alors complètement l'alignement. Basculez via **Paramètres → Cache de
recalage** ; les entrées obsolètes sont remplacées automatiquement.

### Paramètres persistants

Tous les éléments suivants survivent automatiquement à un redémarrage de l'application :
nombre de threads, paramètres des tuiles, largeur de réduction du recalage, taille de lot de la fusion IA (StackMFF-V3),
activation de l'accélération GPU, activation du cache de recalage,
langue de l'interface, taille de la fenêtre et disposition des panneaux, format d'export par glisser-déposer
et piles récemment ouvertes. Les boîtes de dialogue de réglages enregistrent dès que vous validez ;
tout est également enregistré à la fermeture de l'application. Les préférences sont
stockées par utilisateur dans `OpenFocus/OpenFocus.ini` (format INI de QSettings).

### Langue de l'interface

Changez à tout moment via **Paramètres → Langue → English / 中文**. Le changement
s'applique immédiatement et est mémorisé pour les sessions suivantes.

### Vérification automatique des mises à jour

Une fois par jour, quelques secondes après le lancement, OpenFocus interroge silencieusement GitHub
pour une version plus récente. Si elle existe, une bannière apparaît au-dessus des panneaux
d'images : la version, **Mettre à jour** (télécharge et redémarre en une étape),
**Nouveautés** (notes de version) et un bouton de fermeture. Fermer la bannière la masque
pour cette version uniquement — la version suivante la fera réapparaître, et
**Aide → Rechercher des mises à jour** la réaffiche immédiatement.

Pendant le téléchargement en arrière-plan, la barre d'état affiche la progression.
Vous pouvez aussi vérifier manuellement à tout moment via **Aide → Rechercher des mises à jour**.

---

## 12. Traitement par lot

Traitez plusieurs dossiers automatiquement :

1. Allez dans **Outils → Traitement par lot** ou cliquez sur l'option de traitement par lot
2. Dans la boîte de dialogue de traitement par lot :
   - **Ajouter des dossiers** : sélectionnez les dossiers contenant des séquences d'images
   - **Retirer la sélection** : retire les dossiers sélectionnés de la liste
   - **Tout effacer** : retire tous les dossiers
3. Configurez les réglages de sortie :
   - **Créer un sous-dossier dans le dossier source** : crée les résultats dans chaque dossier d'entrée
   - **Identique au dossier source** : écrase les fichiers ou enregistre à côté des entrées
   - **Spécifier un dossier de sortie** : indiquez l'emplacement de sortie
4. Vérifiez les réglages de fusion et de recalage actuels
5. Cliquez sur **Lancer le traitement par lot** pour commencer

La boîte de dialogue de traitement par lot affiche la progression en temps réel. Vous pouvez annuler le traitement à tout moment.

### Remarques sur le traitement par lot

- Utilise les mêmes réglages de fusion et de recalage que le traitement manuel
- Chaque dossier est traité indépendamment
- Les noms de fichiers incluent l'horodatage et les paramètres pour identification

---

## 13. Exporter les résultats

### Enregistrer un résultat unique

1. Sélectionnez le résultat à enregistrer dans la liste de sortie
2. Allez dans **Fichier → Enregistrer** ou appuyez sur `Ctrl+S`
3. Choisissez le format et l'emplacement
4. Cliquez sur Enregistrer

### Format d'export par glisser-déposer

Faire glisser des résultats depuis la liste de sortie écrit des fichiers image au format
choisi dans le menu contextuel de la liste de sortie (**Format de glisser-déposer** : JPG,
PNG ou TIFF). Le choix est mémorisé.

### Enregistrer la pile recalée

1. Allez dans **Fichier → Enregistrer la pile → Pile recalée**
2. Choisissez le format d'enregistrement :
   - **Enregistrer dans un dossier** : enregistre chaque image alignée dans un fichier image distinct
   - **Enregistrer en GIF** : crée un GIF animé de la séquence alignée

### Enregistrer la pile d'entrée

1. Allez dans **Fichier → Enregistrer la pile → Pile d'entrée**
2. Choisissez le format d'enregistrement :
   - **Enregistrer dans un dossier** : enregistre les images d'origine dans des fichiers distincts
   - **Enregistrer en GIF** : crée un GIF animé de la séquence d'entrée

### Formats d'export

- **JPG** : compressé, bon rapport qualité/taille
- **PNG** : sans perte, prend en charge la transparence
- **BMP** : non compressé, qualité maximale
- **TIFF** : haute qualité, prend en charge les couches
- **GIF** : format d'animation

---

## 14. Interface en ligne de commande

OpenFocus peut fonctionner sans interface graphique — utile pour les pipelines scriptés et l'intégration continue.
Depuis un environnement Python (ou à côté de l'exécutable packagé) :

```bash
# Fusionne un dossier avec le filtre guidé par défaut
python main.py --input ./stack_folder --output ./result/fused.png

# DTCWT avec recalage ECC, 8 threads
python main.py -i ./stack_folder -o ./result/fused.png -m dtcwt -a ecc -t 8

# StackMFF-V3 forcé sur le CPU avec une taille de tuile personnalisée
python main.py -i ./stack_folder -o ./result/fused.png -m stackmffv3 --cpu --tile-size 512

# Liste de fichiers explicite ; les fichiers vidéo sont aussi acceptés en entrée
python main.py -i img1.jpg img2.jpg img3.jpg -o fused.png -m gfgfgf

# Pile Z TIFF multipage unique avec barre d'échelle calibrée par les métadonnées
python main.py -i ./zstack.tif -o fused.png -m dtcwt --scale-bar
```

| Option | Description |
|--------|-------------|
| `--input, -i` | Dossier d'images, fichier vidéo, ou un ou plusieurs fichiers image |
| `--output, -o` | Chemin de l'image de sortie (l'extension définit le format) |
| `--method, -m` | `stackmffv3` (par défaut), `none`, `openfocus_a`, `openfocus_b`, `openfocus_c`, `guided_filter`, `dct`, `dtcwt`, `gfgfgf` |
| `--align, -a` | `none` (par défaut), `homography`, `ecc`, `both`, `openfocus` |
| `--align-shift-limit` | alignement `openfocus` uniquement : décalage maximal accepté en fraction de la taille de l'image (0,08 par défaut ; au-delà, les images restent non alignées) |
| `--exposure-match` | Harmonise la moyenne de chaque image sur celle de la dernière image avant la fusion (activé par défaut ; `--no-exposure-match` le désactive). Gains calculés sur toute la pile, donc la sortie en tuiles reste sans couture ; s'applique à toutes les méthodes et à la référence de luminosité de l'aligneur OpenFocus |
| `--base-frame` | Étape post-fusion facultative : aligne les basses fréquences de la sortie sur l'image `last`/`first` (`none` par défaut) |
| `--sort` | Ordre des images pour `openfocus_a/b/c` : `asc` (par défaut, ordre des fichiers), `desc` (inverse un balayage loin → près), `off` |
| `--dust-map WHITE` | Correction de champ plat (carte de poussière) à partir d'une prise de vue défocalisée d'une zone blanche |
| `--kernel, -k` | Taille du noyau (forcée impaire ; pour la famille openfocus, c'est la source du rayon de la fenêtre de variance) |
| `--depth-smoothing SIGMA` | `openfocus_b` uniquement : sigma gaussien de la transition de profondeur entre images (1,2 par défaut ; 0 conserve le détail le plus fin) |
| `--blend-smoothing SIGMA` | `openfocus_a` / `openfocus_c` uniquement : sigma gaussien de la carte de pondération par image (2,0 par défaut pour A, 3,0 pour C ; 0 conserve le choix le plus net par pixel) |
| `--threads, -t` | Plafond de threads ; `0` = automatique (par défaut), les valeurs positives limitent la concurrence |
| `--resource-preference` | Mode de ressources : `balanced` (par défaut), `responsive` ou `maximum` |
| `--batch-size` | Plafond de tuiles IA par lot (2 par défaut) ; actuellement une tuile par appel |
| `--cpu` | Force le CPU même si un GPU est disponible |
| `--downscale` | Largeur de réduction pour la détection de caractéristiques du recalage |
| `--tile-size` | Taille des blocs de tuilage pour les grandes images |
| `--output-dir, -d` | Mode lot : fusionne chaque dossier d'entrée et écrit `<dossier>.png` par pile |
| `--scale-bar` | Dessine une barre d'échelle (calibration détectée automatiquement dans les métadonnées TIFF) |
| `--depth-map [path]` | Enregistre aussi une carte de position de mise au point (profondeur) colorisée (par défaut : `<sortie>_depth.png`) |
| `--depth-colormap` | Palette pour `--depth-map` : turbo, viridis, jet, gray, … (turbo par défaut) |
| `--scale-um` | Taille du pixel en µm/px, prioritaire sur les métadonnées (implique `--scale-bar`) |

Codes de sortie : `0` succès, `1` erreur de traitement, `2` erreur d'utilisation. Exécuter
`python main.py --help` affiche toutes les options ; lancer avec un simple chemin
(`python main.py C:\stack`) ouvre toujours l'interface graphique avec ce dossier chargé.

---

## 15. Raccourcis clavier

### Opérations sur les fichiers

| Raccourci | Action |
|----------|--------|
| `Ctrl+O` | Ouvrir un dossier |
| `Ctrl+Shift+O` | Ouvrir une vidéo |
| `Ctrl+S` | Enregistrer le résultat |
| `Ctrl+Shift+S` | Enregistrer la pile recalée |
| `Ctrl+W` | Vider la pile |
| `Ctrl+Q` | Quitter |
| `Ctrl+R` | Lancer le rendu |
| `Ctrl+Shift+R` | Tout comparer (toutes les méthodes) |
| `Ctrl+T` | Retoucher le résultat |
| `Ctrl+M` | Carte de profondeur |
| `Ctrl+E` | Basculer la comparaison par balayage |
| `Ctrl+Shift+B` | Traitement par lot |
| `Ctrl+Z` / `Ctrl+Y` | Annuler / Rétablir (modifications de la pile) |

### Navigation

| Raccourci | Action |
|----------|--------|
| `Flèche gauche` | Image précédente |
| `Flèche droite` | Image suivante |
| `Molette de la souris` | Zoom avant/arrière |
| `Ctrl + molette` | Zoom fin |
| `Double-clic` | Basculer entre l'affichage ajusté et 100 % |

### Général

| Raccourci | Action |
|----------|--------|
| `Espace` | Basculer le mode déplacement |
| `Suppr` | Supprimer l'élément sélectionné |

---

## 16. Dépannage

### Problèmes courants

#### « Aucune image chargée »

- Vérifiez que vous avez d'abord importé un dossier d'images ou une vidéo
- Vérifiez que le dossier contient des formats d'image pris en charge

#### Les résultats de fusion sont flous ou dédoublés

- Activez le recalage d'images (choisissez Homographie / ECC / OpenFocus rapide dans la liste déroulante)
- Assurez-vous que les images sont correctement alignées avant la fusion
- Essayez un autre algorithme de fusion
- Augmentez la taille du noyau pour des résultats plus doux

#### IA indisponible

- Installez PyTorch : `pip install torch torchvision`
- Pour le support GPU, installez PyTorch avec CUDA
- Vérifiez via **Aide → Informations sur l'environnement**

#### Erreurs de mémoire insuffisante

- Activez le traitement par tuiles dans **Paramètres → Paramètres des tuiles**
- Réduisez la taille des blocs
- Utilisez des noyaux plus petits
- Redimensionnez les images à des dimensions plus faibles

#### Le recalage échoue

- N'essayez qu'une seule méthode de recalage à la fois
- Réduisez la largeur de réduction dans **Paramètres → Paramètres de recalage**
- Assurez-vous que les images ont un contraste et des caractéristiques suffisants

#### Traitement lent

- Augmentez le nombre de threads dans **Paramètres → Paramètres des threads**
- Utilisez un CPU plus puissant
- Réduisez la résolution des images
- Utilisez des algorithmes de fusion plus simples (filtre guidé plutôt que fusion IA)

#### Avertissement Windows SmartScreen au premier lancement

Les binaires publiés ne sont pas signés. Cliquez sur **Informations complémentaires → Exécuter quand même**.
Le premier lancement après décompression peut prendre 30 à 60 secondes.

#### macOS indique que l'application « ne peut pas être ouverte » ou « est endommagée »

L'application n'est ni signée ni notariée. Après avoir déplacé `OpenFocus.app` dans
Applications, exécutez :

```bash
xattr -cr /Applications/OpenFocus.app
```

ou cliquez avec le bouton droit sur l'application et choisissez **Ouvrir**. Les versions macOS publiées sont destinées
à Apple Silicon (M1/M2/M3/M4) ; les Mac Intel peuvent exécuter le code source.

### Obtenir de l'aide

1. **Informations sur l'environnement** : allez dans **Aide → Informations sur l'environnement** pour vérifier votre installation
2. **Contacter l'assistance** : allez dans **Aide → Nous contacter** pour les informations d'assistance

---

## Annexe : détails des algorithmes de fusion

### Filtre guidé

Le filtre guidé réalise un lissage préservant les contours à l'aide d'une image de guidage. Il calcule des transformations linéaires locales entre l'image de guidage et l'image d'entrée, puis les applique pour produire la sortie. Cette méthode excelle à préserver les contours tout en réduisant le bruit.

### DCT (Discrete Cosine Transform)

La DCT transforme les images dans le domaine fréquentiel, où les décisions de fusion sont prises d'après l'amplitude des coefficients. Les composantes haute fréquence (détails) sont préservées, tandis que les composantes basse fréquence (zones lisses) sont fusionnées d'après l'énergie locale.

### DTCWT (Dual-Tree Complex Wavelet Transform)

La DTCWT fournit une décomposition multi-échelle et multidirectionnelle des images. Elle capture les caractéristiques à différentes échelles et orientations, ce qui permet des règles de fusion sophistiquées préservant à la fois la texture et la structure.

### GFG-FGF

Implémente Fu et al. (2024) : la carte de mise au point grossière |I − mean(I)| est enrichie
par le gradient à quatre voisins de type gaussien — le gabarit 3×3 T = [1 2 1; 2 4 2; 1 2 1]/16 donne P = I⊗T, et G = (|I(x±1,y) − P|)/2 + (|I(x,y±1) − P|)/2, les valeurs inférieures à T0 = 0,005 étant écartées — puis affinée par un filtre guidé rapide. Chaque pixel choisit l'image dont la valeur de netteté est la plus grande (égalités : mesure de petite région de Sobel 3×3), et les cartes de décision binaires sont lissées par le filtre guidé rapide avant le mélange pondéré.

### Fusion IA (StackMFF-V3)

Une approche par apprentissage profond utilisant un réseau de neurones entraîné sur de grands ensembles de paires d'images multi-focus. Le réseau apprend automatiquement les stratégies de fusion optimales, offrant des résultats de l'état de l'art lorsqu'il est correctement configuré.

---

*OpenFocus est un projet open source. Pour les mises à jour, les rapports de bogues et les demandes de fonctionnalités, veuillez consulter le dépôt du projet.*
