# OpenFocus Benutzerhandbuch

## Inhaltsverzeichnis

1. [Einleitung](#1-einleitung)
2. [Erste Schritte](#2-erste-schritte)
3. [Überblick über die Hauptoberfläche](#3-überblick-über-die-hauptoberfläche)
4. [Bilder und Videos importieren](#4-bilder-und-videos-importieren)
5. [Bilder ansehen und navigieren](#5-bilder-ansehen-und-navigieren)
6. [Wischvergleich](#6-wischvergleich)
7. [Bildregistrierung](#7-bildregistrierung)
8. [Fusionsmethoden](#8-fusionsmethoden)
9. [Bildtransformationen](#9-bildtransformationen)
10. [Retusche, Beschriftungen und Anmerkungen](#10-retusche-beschriftungen-und-anmerkungen)
11. [Einstellungen und Konfiguration](#11-einstellungen-und-konfiguration)
12. [Stapelverarbeitung](#12-stapelverarbeitung)
13. [Ergebnisse exportieren](#13-ergebnisse-exportieren)
14. [Kommandozeilenschnittstelle](#14-kommandozeilenschnittstelle)
15. [Tastenkürzel](#15-tastenkürzel)
16. [Fehlerbehebung](#16-fehlerbehebung)

---

## 1. Einleitung

OpenFocus ist eine professionelle Desktop-Anwendung zur Multifokus-Bildfusion für Forschende, Ingenieurinnen und Ingenieure sowie Fachleute der Bildverarbeitung. Die Software verarbeitet Multifokus-Bildsequenzen mit ausgefeilten Registrierungs- und Fusionsalgorithmen und erzeugt daraus durchgehend scharfe Bilder mit voller Schärfentiefe.

### Hauptfunktionen

- **Multifokus-Bildfusion**: Mehrere Bilder mit unterschiedlichen Fokuspunkten zu einem einzigen vollständig scharfen Bild kombinieren
- **Mehrere Fusionsalgorithmen**: Guided Filter, DCT, DTCWT, GFG-FGF und KI-Fusion (Deep Learning) zur Auswahl
- **Bildregistrierung**: Fehlausgerichtete Bildsequenzen mit ECC oder Homographie ausrichten
- **Wischvergleich**: Ausrichtung prüfen und Ergebnisse mit einem A/B-Trenner vergleichen
- **Projektdateien**: vollständige Arbeitssitzungen speichern und wiederherstellen
- **Mehrseitige TIFF**: Z-Stapel aus einer einzigen Datei werden seitenweise geladen und wieder in eine Datei exportiert
- **Tiefenkarte**: Fokuspositionskarten pro Pixel mit eigenen Pseudofarbschemata
- **Maßstabsbalken**: publikationsreife Maßstäbe mit automatischer µm/px-Kalibrierung
- **Automatische Absturzwiederherstellung**: nach einem Absturz wird die letzte Arbeitssitzung zum Wiederherstellen angeboten
- **Mehrsprachig**: Englisch, 中文, 日本語, Español mit automatischer Erkennung der Systemsprache
- **Headless-CLI**: OpenFocus für Automatisierung und CI per Kommandozeile skripten
- **Flexibler Export**: Ergebnisse als einzelne Bilder, Ordner oder GIF-Animationen speichern
- **Bildtransformationen**: Bildstapel drehen, spiegeln und skalieren
- **Beschriftungen**: Textbeschriftungen zu registrierten Stapeln und Eingabe-Stapeln hinzufügen
- **Persistente Einstellungen**: Voreinstellungen, Sprache und zuletzt geöffnete Stapel überstehen einen Neustart

### Unterstützte Eingabeformate

- Bilddateien: JPG/JPEG, PNG, BMP, TIFF, WebP und weitere gängige Formate
- Videodateien: MP4 (wird automatisch in Bildsequenzen dekodiert)
- Ordner mit nummerierten Bildsequenzen
- Kamera-RAW-Dateien (DNG, CR2/CR3, NEF, ARW, RAF, ORF, RW2 und weitere gängige
  RAW-Formate). Die Dekodierung übernimmt `rawpy` (LibRaw), das durch `pip install openfocus`
  mitinstalliert wird und in den Windows-/macOS-Builds enthalten ist

### Unterstützte Ausgabeformate

- Einzelne Bilder: JPG, PNG, BMP, TIFF, WebP
- Bildsequenzen: als Ordner gespeichert
- Animationen: GIF-Format

---

## 2. Erste Schritte

### Anwendung starten

Start OpenFocus nach der Installation so:

1. **Windows (ausführbare Datei)**: Doppelklick auf die OpenFocus-Verknüpfung oder die ausführbare Datei
2. **Python-Quellcode**: `python main.py` in der aktivierten conda-Umgebung ausführen

Nach dem Start erscheint das Hauptfenster mit einer dunklen Oberfläche.

### Schnellstart-Ablauf

1. Bildsequenz importieren (Datei → Ordner öffnen oder per Drag-and-drop)
2. Fusionseinstellungen im rechten Bereich konfigurieren
3. Optional die Registrierung für die Ausrichtung aktivieren
4. Bei Bedarf die Kernelgröße anpassen
5. *(optional)* **Schnellvorschau** aktivieren, um die Parameter zuerst an einem verkleinerten Entwurf zu testen
6. Auf „Rendern starten“ klicken, um die Verarbeitung zu beginnen
7. Ergebnisse exportieren

---

## 3. Überblick über die Hauptoberfläche

Die Oberfläche von OpenFocus besteht aus drei Hauptbereichen:

### Linker Bereich: Bildanzeige

Der linke Teil des Fensters zeigt die Bilder in geteilter Ansicht:

- **Quellbildbereich (oben)**: zeigt den eingegebenen Bildstapel
- **Ergebnisbereich (unten)**: zeigt die Fusions- oder Registrierungsergebnisse
- **Navigationsregler**: mit dem Regler unter jedem Bereich durch die Einzelbilder blättern
- **Bildinfo-Beschriftungen**: zeigen die aktuelle Bildposition und die Bildabmessungen

Die Größe der Bereiche lässt sich durch Ziehen des Trenners dazwischen ändern.

### Rechter Bereich: Bedienfeld

Der rechte Bereich enthält alle Konfigurationsoptionen, vertikal gegliedert:

1. **Fusionseinstellungen**: Fusionsalgorithmus und Kernelgröße wählen
2. **Registrierungseinstellungen**: Ausrichtungsmethoden aktivieren bzw. deaktivieren
3. **Aktionsschaltflächen**: Zurücksetzen und Rendern starten
4. **Liste der Quellbilder**: alle geladenen Bilder mit Dateinamen
5. **Ausgabeliste**: die erzeugten Ergebnisse

### Menüleiste

Die Menüleiste am oberen Rand bietet Zugriff auf alle Funktionen:

- **Datei**: Import- und Exportvorgänge sowie die Liste der zuletzt geöffneten Dateien
- **Bearbeiten**: Bildtransformationen und Beschriftungen
- **Werkzeuge**: Stapelverarbeitung
- **Einstellungen**: GPU-Beschleunigung, Sprache, Kacheln, Registrierung und Thread-Konfiguration
- **Hilfe**: Umgebungsinformationen, Kontaktinformationen, Update-Prüfung und der Log-Ordner

---

## 4. Bilder und Videos importieren

### Bildordner öffnen

1. Auf **Datei → Ordner öffnen** gehen oder `Ctrl+O` drücken
2. Einen Ordner mit der Bildsequenz auswählen
3. Ein Dialog zur Ladeskalierung erscheint – die Skalierung wählen (100 % = volle Auflösung)
4. Auf OK klicken, um die Bilder zu laden

Im selben Dialog **Nicht mehr fragen – immer mit dieser Skalierung laden** ankreuzen,
um künftig immer mit der gewählten Skalierung ohne Rückfrage zu laden;
**Einstellungen → Beim Import nach Ladeskalierung fragen** schaltet die Rückfrage
wieder ein. Ein geladener Stack lässt sich später jederzeit mit
**Bearbeiten → Größe ändern** verkleinern.

Die Software wird automatisch:
- Bilddateien erkennen und nach Dateinamen sortieren
- das erste Bild im Quellbildbereich anzeigen
- die Liste der Quellbilder mit allen Einzelbildern füllen

### Videodateien öffnen

1. Auf **Datei → Video öffnen** gehen oder `Ctrl+Shift+O` drücken
2. Eine Videodatei auswählen (MP4, AVI, MOV, MKV, WMV, FLV, WEBM)
3. Das Video wird automatisch in einzelne Bilder dekodiert
4. Jedes Einzelbild wird Teil des Bildstapels

### Projektdateien

Datei → Projekt speichern legt die gesamte Arbeitssitzung als `.ofproj`-Datei
ab: die exakte Quellenliste (in Reihenfolge), die Ladeskalierung, die Fusionsmethode,
die Registrierungsoptionen, die Kernelgröße, Kachel-/Thread-/Registrierungseinstellungen,
die Beschriftungskonfigurationen und das aktuelle Einzelbild. Datei → Projekt öffnen
baut diese Sitzung exakt wieder auf (Quellbilder werden referenziert, nicht kopiert – sie
müssen deshalb am selben Ort bleiben). Der Fenstertitel zeigt das geöffnete Projekt.

### Demo-Stack

Der Willkommensdialog bietet **Demo-Stack laden**: sechs mitgelieferte Beispielfotos
aus einem echten Fokus-Stack, damit der komplette Ablauf sofort ohne eigene Bilder
ausprobiert werden kann. Auch praktisch, um eine Installation zu prüfen.

### Schärfekurve

Eine kleine Kurve über dem Navigationsregler der Quelle zeigt die Schärfe pro
Einzelbild (normiert). Unscharfe Bilder erscheinen als Einbrüche mit rotem
Punkt. Ein Klick auf die Kurve springt zu diesem Bild; beim Überfahren mit der
Maus wird die Bildnummer eingeblendet.

### Mehrseitige TIFF-Stapel

Mikroskopiesoftware (ZEN, NIS-Elements, ImageJ/Fiji) exportiert einen ganzen
Z-Stapel häufig als EINE mehrseitige TIFF-Datei. OpenFocus liest jede Seite einer
solchen Datei als eigenes Einzelbild des Stapels – die Datei ablegen oder öffnen,
und der Stapel erscheint vollständig (die Einzelbilder heißen `<file>_page_001` und
so weiter). Einseitige TIFFs verhalten sich genau wie bisher. Auch der umgekehrte
Weg funktioniert: **Datei → Stack speichern → (Registrierter / Eingabe-Stack) → Als mehrseitige
TIFF speichern** schreibt den gesamten Stapel zurück in eine einzige `.tif` und
behält dabei die 16-Bit-Tiefe bei.

### Maßstabsbalken (Publikationsabbildungen)

Einstellungen → Maßstabsbalken… konfiguriert einen Maßstabsbalken, der in
exportierte Bilder eingebrannt wird. Die Pixelgröße (µm/px) wird automatisch aus
den TIFF-Metadaten erkannt – über die `unit=micron`-Konvention von ImageJ, das
`PhysicalSizeX` von OME-TIFF oder eine explizite Auflösungseinheit in Zentimetern –,
sodass sich Mikroskopie-Stapel (auch mehrseitige TIFFs) selbst kalibrieren.
Druckerauflösungen (72/300 dpi und ähnliche) werden bewusst *nicht* als Kalibrierung
behandelt, und ein
manuell eingegebener Wert hat immer Vorrang vor erkannten Metadaten: ein plausibel
aussehender falscher Balken in einer Abbildung ist schlimmer als gar keiner.
Ecke und Farbe wählen – **Auto** invertiert gegenüber dem Hintergrund; der Dialog
zeigt eine Live-Vorschau. Bei aktiviertem Balken trägt jeder Export (Einzelbild,
Ordner-Stapel, mehrseitige TIFF, GIF, Alle exportieren, Stapelverarbeitung) einen
Balken im Publikationsstil: eine gerundete 1-2-5-Länge (etwa 12 % der Bildbreite)
mit konturierter, gut lesbarer Beschriftung. Die CLI unterstützt ihn über
`--scale-bar` (automatische Kalibrierung) und `--scale-um <um/px>`.

### Tiefenkarte (Fokusposition)

**Werkzeuge → Tiefenkarte…** schätzt Fokuspositionen mit dem Aktivitätsmaß der gewählten
Methode (DCT, DTCWT, GFG-FGF, StackMFF V3, OpenFocus B/C). Die Karte ist weder eine
gemessene Oberfläche noch eine genaue Zuordnung der Pixel des Fusionsergebnisses.
Als Quelle dienen die aktuell markierten Bilder oder die aufgezeichnete Quelle der
gewählten Ausgabe; fehlende Herkunftsdaten werden nicht durch den aktuellen Stapel
ersetzt. Berechnung, Textur und Export bleiben an denselben Quellstand gebunden.

Alle sechs Methoden erfordern einen ausdrücklichen Klick auf **Berechnen**. Öffnen,
Wechsel von Maß, Quelle oder Registrierung und Parameteränderungen starten keine
Berechnung. Direkt unter der Maßauswahl stehen die editierbaren Parameter: DCT-Median-
und GFG-Anfangskernel sind ungerade 1–51 (Standard 7), das Varianzfenster von B und das
Bandpassenergiefenster von C ungerade 1–21 px (Standard 11). Die DTCWT-Zerlegungstiefe
ist hier als lokaler Entwurf bearbeitbar und beginnt mit den Quellparametern. StackMFF V3
zeigt stattdessen einen eigenen KI-Ausgabemodus (standardmäßig weiche Mischung oder harte
Auswahl) und eine Erläuterung: Weiche Mischung schätzt eine wahrscheinlichkeitsgewichtete
kontinuierliche Fokusposition, harte Auswahl nimmt pro Pixel das wahrscheinlichste Bild.
Beides ist keine physikalische Entfernung.
Die Entwürfe beginnen mit den Parametern der passenden Ausgabe oder den gespeicherten
Methodenwerten des Hauptfensters. Sie bleiben während dieses Dialogs pro Quelle und
Methode erhalten, ohne Hauptfenster, Projekt oder Ausgabeprotokoll zu ändern.

Geänderte Berechnungseinstellungen markieren die alte 2D-Karte dauerhaft als **nicht
aktuell**, leeren die 3D-Oberfläche und sperren alle Exporte, auch Rohdaten, physikalische
Tiefe und OBJ, bis eine neue Berechnung gelingt. Laufende Arbeit wird kooperativ
abgebrochen. Danach erneut **Berechnen** drücken; es gibt keine automatische Fortsetzung
oder Warteschlange. Berechnen bleibt sichtbar und ist bei Berechnung, Abbruch und
Modellexport deaktiviert.

- **Farbtabelle**: Graustufen (Publikationsstandard) sowie 15 wissenschaftliche
  Paletten (Turbo, Viridis, Magma, …) und eigene – *Neues eigenes Schema…* baut
  einen Verlauf aus beliebig vielen Farbstützpunkten (Doppelklick auf einen
  Stützpunkt bearbeitet Position und Farbe); eigene Schemata werden gespeichert und
  im Dropdown angeboten.
- **Tiefenachse umkehren / Gamma / Glättung**: die Tiefenachse umdrehen, den
  Tonwert neu ausbalancieren (Gamma < 1 hellt niedrige Indizes auf) und Sprenkel
  mit einem kantenbewahrenden Durchgang entfernen.
- **Überlagerung**: Die 2D-Karte lässt sich mit einstellbarer Deckkraft über die
  zugehörige Ausgabe legen, wenn deren Koordinatenbereich übereinstimmt.
- **Export**: *Koloriert speichern…* bietet PNG/JPEG/TIFF/BMP. Der rohe 16-Bit-TIFF
  kodiert normalisierte Fokusindizes als 0–65535, keine verlustfreien Fließkommawerte
  oder physikalischen Entfernungen. JPEG/BMP werden für Rohdaten abgelehnt.
- **3D-Modell / 3D-Textur**: Die längste Netzseite hat standardmäßig 600 Knoten.
  Jeder neue Dialog beginnt mit **Automatisch (Echtfarben)**: Die zugehörige Ausgabe
  wird nur verwendet, wenn Identität und Koordinatenbereich/ROI zur fertigen Karte
  passen. Sonst dient das erste beteiligte Bild der abgeschlossenen Berechnung als
  Textur, nach Registrierung und tatsächlicher Bildsortierung. Der verwendete Ausgabe-
  oder Bildname wird angezeigt. Passende Ausgabe, einzelnes beteiligtes Bild und
  **Tiefenpseudofarben** sind ausdrücklich wählbar; fehlende Texturen werden gemeldet.
  3D-Vorschau und OBJ-Textur-PNG nutzen dieselbe Bildquelle. Die 2D-Karte behält ihre
  eigene Farbkarte und Überlagerung; Palette und Gamma färben Echtfotos nicht um.
  Nur die gewählte Pseudofarbtextur folgt Palette, Gamma und Umkehrung.
  Ruhende Vorschau und Export verwenden dasselbe abgetastete Höhenfeld ohne zusätzliche
  versteckte Glättung. Das relative Relief ist eine Darstellung in Bildeinheiten.

Die Berechnung läuft im Hintergrund mit Fortschritt und Abbruchmöglichkeit; die
KI-Methode nutzt dieselbe speicherschonende Kachelung wie die KI-Fusion, sodass auch
tiefe Stapel funktionieren. Das CLI-Äquivalent ist `--depth-map [path]` mit
`--depth-colormap`.

### Automatische Absturzwiederherstellung

OpenFocus hält fortlaufend einen Schnappschuss der Arbeitssitzung (Stapel,
Einstellungen, Beschriftungen, aktuelles Einzelbild). Wird die Anwendung unerwartet
beendet – durch einen Absturz, Stromausfall oder gewaltsames Beenden –, erkennt dies
der nächste Start und bietet **Sitzung wiederherstellen** an, um genau zum letzten
Stand zurückzukehren. Nach einem sauberen Beenden erscheint die Abfrage nie.
Wiederherstellungsdaten liegen neben den Einstellungen (portable Installationen
behalten sie neben der ausführbaren Datei, sodass sie mitwandern).

### Zuletzt geöffnete Dateien

Jeder geöffnete Stapel wird unter **Datei → Zuletzt geöffnet** gemerkt (bis zu 8 Einträge).
Mit `Alt+1`–`Alt+8` lässt sich ein Eintrag direkt wieder öffnen. Ein erneut geöffneter
Stapel rückt an die Spitze; Einträge, deren Dateien nicht mehr existieren, lassen sich
nach einer Rückfrage entfernen, und **Liste „Zuletzt geöffnet“ leeren** räumt das Menü.

### Drag-and-drop

Ein Ordner lässt sich auch direkt auf das Anwendungsfenster ziehen, um Bilder zu
importieren. Werden weitere Dateien auf einen vorhandenen Stapel abgelegt, werden sie
angehängt (nach einer Größenprüfung).

### EXIF-Ausrichtung

Kamera-JPEG/TIFF-Dateien mit EXIF-Orientation-Tag werden beim Import automatisch
gedreht bzw. gespiegelt, sodass Hochformat-Aufnahmen aufrecht erscheinen.

### 16-Bit-Quellen

16-Bit-PNG/TIFF-Eingaben werden automatisch erkannt und durchgehend erhalten:
Die Fusion läuft in voller Bittiefe, und Ergebnisse lassen sich wieder als 16-Bit-PNG/TIFF
speichern (JPG/BMP-Exporte werden auf 8 Bit reduziert, da diese Formate keine 16 Bit
speichern können).

Kamera-RAW-Dateien werden mit 16 Bit entwickelt (Kameraweißabgleich, sRGB-Gamma)
und mit deaktivierter Automatikhelligkeit, damit jedes Bild eines Stapels dieselbe
Belichtungszuordnung erhält; eine automatische Belichtung pro Bild würde die
Helligkeit zwischen den Bildern flackern lassen. Zum Erhalt der 16-Bit-Daten als
TIFF oder PNG speichern.

### Stapel leeren

Um alle geladenen Bilder zu entfernen, auf **Datei → Stack leeren** gehen oder `Ctrl+W` drücken.

---

## 5. Bilder ansehen und navigieren

### Im Bildstapel navigieren

- **Regler**: den Regler unter dem Quellbereich ziehen, um durch die Einzelbilder zu blättern
- **Dateiliste**: auf einen Dateinamen in der Quellenliste des rechten Bereichs klicken
- **Tastatur**: mit den Pfeiltasten (links/rechts) navigieren

### Zoomen und Verschieben

- **Hinein-/Herauszoomen**: Mausrad über dem Bild
- **Feinzoomen**: `Ctrl` + Mausrad für kleinere Schritte
- **An Fenster anpassen**: Doppelklick auf das Bild schaltet zwischen Einpassen und 100 %
- **Verschieben**: Klicken und ziehen, um das Bild im Bereich zu bewegen

### Synchronisierte Navigation

Wenn im Quellbereich navigiert wird, zeigt der Ergebnisbereich das entsprechende
Ergebnisbild an, sofern vorhanden.

---

## 6. Wischvergleich

Die Wischansicht legt zwei Bilder mit einem verschiebbaren vertikalen Trenner
übereinander und macht Fehlausrichtungen und Qualitätsunterschiede entlang der
Trennlinie direkt sichtbar.

### Wischansicht öffnen

1. Mindestens ein Ergebnis rendern (Fusion oder Registrierung)
2. Den Schalter **Wischen** oben rechts in der Titelleiste des Ergebnisbereichs drücken
3. Der Ergebnisbereich wechselt in die Vergleichsansicht; erneutes Drücken von **Wischen** kehrt zurück

### Die Ansicht lesen

- **Seite A (links vom Trenner)**: ein Quellbild – standardmäßig folgt es dem Regler des Quellbereichs
- **Seite B (rechts vom Trenner)**: das zuletzt gerenderte Ergebnis
- **Weißer Trenner mit rundem Griff**: horizontal ziehen, um die Trennlinie zu verschieben
- Wer Kanten über den Trenner zieht, erkennt Registrierungsfehler, die in nebeneinanderliegenden Bereichen kaum auffallen

### Vergleichsobjekte wählen

Die Leiste am unteren Rand der Vergleichsansicht hat eine Gruppe pro Seite:

- **Seite A** – **Quelle folgen** ankreuzen (Standard), um dem Regler des Quellbereichs zu folgen, oder abwählen, um Seite A mit einem eigenen Regler auf ein beliebiges Bild zu fixieren
- **Seite B** – **Neuestes Ergebnis** ankreuzen (Standard), um immer das neueste Rendering zu zeigen, oder abwählen, um mit dem Regler durch den Ausgabeverlauf zu blättern

Beide Seiten aktualisieren sich live: neue Renderings, Bewegungen des Quellreglers und
gelöschte Einzelbilder erscheinen sofort.

### Zoomen und Verschieben

Die Vergleichsansicht verwendet eine gemeinsame Transformation für beide Bilder:

- **Zoom**: Mausrad (zoomt um den Cursor)
- **Verschieben**: überall außer auf dem Trenner klicken und ziehen
- **Zurücksetzen**: Doppelklick

Wischmodus und ROI-Modus teilen sich den Ergebnisbereich, der eine schließt den
anderen daher aus. Das Leeren des Stapels beendet ebenfalls den Wischmodus.

---

## 7. Bildregistrierung

Die Bildregistrierung korrigiert räumliche Fehlausrichtungen zwischen den Bildern
eines Stapels. Sie ist unverzichtbar, wenn Bilder leichte Verschiebungen oder
Perspektivänderungen aufweisen.

### Registrierungsmethoden

#### ECC (Enhanced Correlation Coefficient)

- Nutzt eine Optimierung, um die beste Ausrichtung anhand der Korrelation zu finden
- Geeignet für subtile Fehlausrichtungen
- Liefert Subpixel-Genauigkeit
- Langsamer als Homographie, aber präziser

#### Homographie

- Basiert auf Merkmalspunkt-Matching (SIFT/ORB-Merkmale)
- Bewältigt größere geometrische Transformationen
- Für die Erstausrichtung schneller als ECC
- Gut geeignet bei deutlichen Perspektivänderungen

#### OpenFocus (schnell)

- Begrenzte Ähnlichkeitsausrichtung (Verschiebung + Rotation + Skalierung),
  mit Helligkeitsabgleich pro Bild und automatischem Zuschnitt
- Unter einer Sekunde pro Bild – deutlich schneller als ECC oder Homographie
- Bilder, die sich um mehr als **8 % der Bildgröße verschieben, bleiben unausgerichtet**,
  statt teilweise korrigiert zu werden. Über die CLI-Option
  `--align-shift-limit` änderbar
- Für nicht-starre oder extreme Geometrien stattdessen ECC oder Homographie wählen

### Registrierung aktivieren

1. Im rechten Bereich die Gruppe **Registrierung** suchen
2. Eine Methode im Dropdown wählen: **Keine Ausrichtung**, **OpenFocus (schnell)**,
   **ECC** oder **Homographie**
3. Die schnelle OpenFocus-Ausrichtung lässt sich auch vor ECC oder Homographie
   schalten (erst ein grober globaler Durchgang, dann der präzise lokale Durchgang)

### Wann Registrierung sinnvoll ist

- Die Bilder sind sichtbar fehlausgerichtet
- Die Aufnahmen wurden aus der Hand gemacht
- Die Szene hat Parallaxe (nähere Objekte verschieben sich zwischen den Bildern)
- Ergebnisse zeigen Geisterbilder oder Doppelkonturen

### Registrierungseinstellungen

Weitere Einstellungen über **Einstellungen → Registrierungseinstellungen**:
- **Verkleinerungsbreite**: Auflösung der Vorverarbeitung (Standard: 1024 px)
- Kleinere Werte = schnellere Verarbeitung, eventuell geringere Genauigkeit
- Größere Werte = langsamere Verarbeitung, eventuell höhere Genauigkeit

---

## 8. Fusionsmethoden

OpenFocus bietet zehn Fusionsalgorithmen (plus einen reinen Ausrichtungsmodus). Jeder
hat eigene Eigenschaften, die zu unterschiedlichen Bildtypen passen.

### Guided Filter (Standard)

- **Algorithmus**: kantenbewahrende Fusion über Guided Filtering
- **Am besten für**: allgemeine Zwecke, ausgewogener Kompromiss aus Geschwindigkeit und Qualität
- **Vorteile**: gute Kantenerhaltung, schnelle Verarbeitung
- **Parameter**: Kernelgröße (einstellbar von 1–51, nur ungerade Werte)
  - Kleinerer Kernel: mehr lokales Detail, potenziell mehr Rauschen
  - Größerer Kernel: glattere Ergebnisse, bessere Rauschunterdrückung

### DCT (Diskrete Kosinustransformation)

- **Algorithmus**: Fusion im Frequenzbereich über DCT-Koeffizienten
- **Am besten für**: texturreiche Bilder, wissenschaftliche Bildgebung
- **Vorteile**: erhält feine Texturen, mathematisch fundiert
- **Parameter**: Die Kernelgröße bestimmt das Verarbeitungsfenster

### DTCWT (Dual-Tree-Complex-Wavelet-Transformation)

- **Algorithmus**: Fusion im mehrskaligen Wavelet-Bereich
- **Am besten für**: komplexe Szenen mit mehreren Detailstufen
- **Vorteile**: hervorragende mehrskalige Analyse, gute Richtungsauflösung
- **Fusionsregel**: eine Entscheidung pro Pixel und Stufe für alle Orientierungen und
  Farbkanäle (N-fach, mit Mehrheitskonsistenzprüfung); das Tiefpassband
  folgt der gröbsten Entscheidung, statt alle Bilder zu mitteln
- **Parameter**: Wavelet-Ebenen (1–6, Standard 4) im rechten Bereich. Die
  Methode hat keinen räumlichen Kernel, daher ist der Kernelregler ausgeblendet.

### GFG-FGF (Gaußartiger Vier-Nachbarschafts-Gradient + schneller Guided Filter)

- **Algorithmus**: Implementierung nach Fu et al. (*Laser & Optoelectronics
  Progress*, 2024): Eine Unschärfedifferenz-Fokuskarte |I − mean(I)| wird
  mit dem gaußartigen Vier-Nachbarschafts-Gradienten (GFG) angereichert, durch einen
  schnellen Guided Filter verfeinert, pro Pixel entschieden (Gleichstand über ein
  3×3-Sobel-Kleinbereichsmaß aufgelöst) und die binären Entscheidungskarten werden vor
  der gewichteten Mischung erneut mit dem schnellen Guided Filter geglättet.
- **Am besten für**: Mikroskopie und Bilder mit klaren Fokusbereichen
- **Vorteile**: schnell, scharfe Grenzen der Fokusbereiche; der Speicherbedarf wächst
  nicht mit der Bildanzahl
- **Parameter**: Kernel = Größe des Mittelwertfilters der groben Fokuskarte

### StackMFF (V1, Deep Learning)

- **Algorithmus**: ein 3D-U-Net, das die fusionierte Luma des gesamten Stapels
  in einem Durchlauf vorhersagt; Cb/Cr stammen aus dem nächstgelegenen Quellbild
- **Am besten für**: Stapel ohne feste Fokusreihenfolge oder ohne kontinuierliche
  Aufnahmeserie
- **Vorteile**: reihenfolgeunabhängig, keine Parameter, Farben bleiben im Gamut
- **Hinweis**: 6 Mio. Parameter; große Stapel profitieren von einer GPU

### StackMFF V2 (Deep Learning)

- **Algorithmus**: Fokusschätzung pro Ebene, verfeinert durch eine
  bidirektionale rekurrente Schicht; das Netz regressiert den Fokusindex und
  tastet den Stapel ab. **Die Bilder müssen in ihrer ursprünglichen
  Fokusreihenfolge bleiben** — Umkehren ist deaktiviert
- **Tiefenkarten**: unterstützt; meldet die abgetastete Fokusebene
- **Am besten für**: kontinuierliche Fokusserien mit sehr kleinem, schnellem Modell
- **Voraussetzungen**: PyTorch; etwa 0,05 Mio. Parameter; keine Einstellungen
### StackMFF V3 (Deep Learning)

- **Algorithmus**: pixelweise Quellklassifikation (erfordert PyTorch). Das
  Netzwerk sagt für jedes Pixel voraus, aus welchem Eingabebild es stammt, und mischt
  dann die Bilder entsprechend diesen Wahrscheinlichkeiten oder wählt ein Bild aus
- **KI-Ausgabe** (rechter Bereich, nur für diese Methode):
  - **Weiche Mischung (gewichtet)** – Standard; jedes Pixel ist eine
    wahrscheinlichkeitsgewichtete Mischung der Bilder. Weiche Übergänge, keine harten
    Nähte; entspricht der im Paper beschriebenen Konfiguration
  - **Harte Auswahl (einzelnes Bild)** – jedes Pixel wird aus dem
    wahrscheinlichsten Bild kopiert. Schärfere Auswahl ohne Interpolation zwischen den
    Bildern; wo zwei Bilder gleichauf liegen, kann eine Naht auf Pixelebene entstehen
- **Am besten für**: höchste Qualität, wenn eine GPU verfügbar ist
- **Vorteile**: reihenfolgeunabhängig (die Bildreihenfolge ändert das Ergebnis nie),
  beliebige Stapellänge, keine methodenspezifische Abstimmung, automatische Optimierung
- **Voraussetzungen**: PyTorch-Installation, optional CUDA-GPU (kein torchvision)
- **Hinweis**: Möglicherweise nicht verfügbar, wenn PyTorch nicht installiert ist

### OpenFocus A / B / C

- **Algorithmus**: varianzgewichteter Mittelwert, Varianz-Tiefenkarten-Auswahl und
  Laplace-Pyramiden-Fusion
- **OpenFocus A**: varianzgewichteter Mittelwert mit Prior des letzten Bildes –
  schnell, farbstabil bei kurzen Stapeln
- **OpenFocus B**: Varianz-Tiefenkarten-Auswahl – wählt pro Pixel das schärfste
  Bild; am besten auf glatten Oberflächen und in der Mikroskopie
- **OpenFocus C**: Laplace-Pyramiden-Mischung pro Frequenzband – natürliche Wirkung
  bei texturierten Motiven, hebt den lokalen Kontrast
- **Parameter**: A und B verwenden ein Varianzfenster (Kernelregler); C hat keinen
  Radius. Alle drei verwenden ein Glättungs-Sigma.

**Glättung**: ein Drehfeld unter dem Kernelregler, dessen Bedeutung von der gewählten
Methode abhängt (Bereich 0–8, Schrittweite 0,2):

| Methode | Beschriftung | Standard | Wirkung |
|---|---|---|---|
| OpenFocus A | Mischungsglättung (σ) | 2,0 | Gauß-Glättung der Gewichtskarte pro Bild – wie breit die scharfen Bereiche benachbarter Bilder gemischt werden |
| OpenFocus B | Tiefenglätung (σ) | 1,2 | Gauß-Glättung der Tiefenkarte – Breite des Übergangs zwischen den Bildern |
| OpenFocus C | Mischungsglättung (σ) | 3,0 | Gauß-Glättung der Gewichtskarte jedes Pyramidenbands |

0 behält die schärfste Auswahl pro Pixel (feinstes Detail, aber Sprenkel auf
verrauschten Stapeln); größere Werte machen Übergänge und Halos weicher. Jede Methode
merkt sich ihren eigenen Wert über Sitzungen hinweg, und die Stapelverarbeitung nutzt
dieselbe Regel.

C hat keinen Radius: Helicons Methode C bietet nur Glättung, daher wird der
Kernelregler für C ausgeblendet, statt stillschweigend wirkungslos zu sein.

CLI: `--depth-smoothing SIGMA` (B) und `--blend-smoothing SIGMA` (A, C).

**Helligkeitsoptionen gelten pro Methode.** *Belichtung vor der Fusion angleichen* und
*Helligkeitsdrift entfernen* werden für jede Fusionsmethode separat gespeichert, weil
sie eine Frage pro Stapel beantworten und keine globale Voreinstellung sind.
Beide sind für OpenFocus A/B/C standardmäßig an (diese Methoden haben keine eingebaute
Belichtungskalibrierung); die übrigen Methoden behalten die bisherigen Standards
(Belichtungsabgleich an, Driftentfernung aus). Beim Methodenwechsel werden die
jeweiligen Werte dieser Methode angezeigt.

### Nur Ausrichtung (keine Fusion)

**Nur Ausrichtung (keine Fusion)** als Methode wählen, um den Stapel zu registrieren
und die ausgerichteten Bilder ohne Fusion in die Ausgabeliste zu übernehmen – praktisch,
um die Registrierungsqualität zu prüfen oder ausgerichtete Bilder zu exportieren.

### Fusionsmethode wählen

1. Im rechten Bereich die Gruppe **Fusion** suchen
2. Eine Methode im **Dropdown** wählen (genau eine ist immer aktiv;
   der erste Eintrag registriert ohne Fusion)
3. Bei Bedarf die Kernelgröße anpassen – die Beschriftung zeigt, was der Kernel
   für die gewählte Methode bedeutet
4. **▶ Rendern starten** klicken, um zu verarbeiten

### Algorithmus-Vergleich

| Algorithmus | Geschwindigkeit | Qualität | Am besten für |
|-----------|-------|---------|----------|
| Guided Filter | Schnell | Gut | Allgemeine Nutzung |
| DCT | Mittel | Gut | Texturen |
| DTCWT | Mittel | Sehr gut | Komplexe Szenen |
| GFG-FGF | Schnell | Gut | Fokusbereiche |
| KI-Fusion (StackMFF V3) | Langsam (GPU) | Hervorragend | Beste Qualität |
| OpenFocus A | Sehr schnell | Sehr gut | Kurze Stapel, Farbstabilität |
| OpenFocus B | Sehr schnell | Sehr gut | Glatte Oberflächen, Mikroskopie |
| OpenFocus C | Schnell | Sehr gut | Texturierte Motive, natürliche Wirkung |

### Schnellvorschau (Entwurfsrendering)

Das Abstimmen von Parametern auf großen Stapeln ist in voller Auflösung langsam.
**Schnellvorschau** über der Rendern-Schaltfläche aktivieren: OpenFocus rendert dann
zuerst einen verkleinerten Entwurf (längste Seite 1200 px) – typischerweise mehrere
Male schneller, einschließlich Registrierung. Die Vorschau erscheint nach dem Rendern
in der Statusleisten-Zusammenfassung als „Qualität: Vorschau“ und wird **nicht** zum
Ausgabeverlauf hinzugefügt. Häkchen entfernen und erneut rendern, um das Ergebnis in
voller Auflösung zu erhalten. Der Vorschaumodus gilt nicht für ROI-Renderings und
verunreinigt nie den Ausrichtungs-Cache in voller Auflösung.

---

## 9. Bildtransformationen

### Drehen

1. Auf **Bearbeiten → Drehen** gehen
2. Drehungsart wählen:
   - **90° im Uhrzeigersinn**: alle Bilder um 90 Grad nach rechts drehen
   - **90° gegen den Uhrzeigersinn**: alle Bilder um 90 Grad nach links drehen
   - **180°**: Bilder auf den Kopf stellen

### Spiegeln

1. Auf **Bearbeiten → Spiegeln** gehen
2. Spiegelungsart wählen:
   - **Horizontal spiegeln**: von links nach rechts spiegeln
   - **Vertikal spiegeln**: von oben nach unten spiegeln

### Größe ändern

1. Auf **Bearbeiten → Größe ändern** gehen
2. Ein Dialog mit den Skalierungsoptionen erscheint
3. Neue Abmessungen oder Skalierungsprozentsatz eingeben
4. Alle Bilder im Stapel werden proportional skaliert

### Reihenfolge der Transformationen

Transformationen wirken auf den gesamten Bildstapel und erhalten die Ausrichtung
zwischen den Einzelbildern.

---

## 10. Retusche, Beschriftungen und Anmerkungen

### Ergebnis retuschieren (Bearbeiten → Ergebnis retuschieren…, `Ctrl+T`)

Die Fusion kann örtliche Fehler hinterlassen – einen Halo an einer kontrastreichen Kante,
ein Härchen aus dem falschen Bild, Rauschen in einer gleichmäßigen Fläche. Der
Retuschepinsel kopiert Pixel aus einem Quellbild auf das Ergebnis:

1. Ein Ergebnis rendern (oder eines in der Ausgabeliste auswählen; Rechtsklick →
   *Ergebnis retuschieren…* funktioniert ebenfalls) und `Ctrl+T` drücken.
2. Den Fehler mit der linken Maustaste übermalen. Mit aktivierter Option **Pro Strich
   schärfstes Bild automatisch wählen** (Standard) sucht jeder Strich zuerst das Bild
   mit dem meisten Detail unter dem Pinsel – ein Halo ist meist mit einem Zug behoben.
   Deaktivieren, um das Bild selbst über den Regler oder `,` / `.` zu wählen.
3. **Radius** (`[` / `]`), **Härte** (weiche Kanten verbergen die Naht) und
   **Deckkraft** anpassen. Überlappende Tupfer innerhalb eines Strichs addieren sich nie.
4. **Quellbild anzeigen** (`Tab`) zeigt das Bild, aus dem gemalt wird.
   Das Mausrad zoomt, Rechts-/Mittelklick-Ziehen verschiebt, `F` passt das Bild ein.
5. `Ctrl+Z` / `Ctrl+Y` machen ganze Striche rückgängig bzw. wiederholen sie.
   **Als neue Ausgabe übernehmen** fügt das retuschierte Bild der Ausgabeliste hinzu
   (Suffix `_Retouched`); die ursprüngliche Ausgabe bleibt unverändert.

Der Pinsel verwendet die ausgerichteten Bilder, aus denen das Ergebnis entstand, und
passt daher exakt dazu; 16-Bit-Ergebnisse bleiben 16 Bit. ROI-Renderings und skalierte
Stapel lassen sich nicht retuschieren (ihre Bilder passen nicht mehr zur Geometrie des
Ergebnisses).

### Beschriftungen hinzufügen

1. Sicherstellen, dass Ergebnisse gerendert oder Bilder geladen sind
2. Auf **Bearbeiten → Beschriftung hinzufügen** gehen
3. Den Beschriftungstext im Dialog eingeben
4. Die Beschriftungen werden sowohl dem registrierten Stapel als auch dem Eingabe-Stack hinzugefügt

### Beschriftungen löschen

- **Beschriftungen des registrierten Stacks löschen**: entfernt Beschriftungen aus ausgerichteten Bildern
- **Beschriftungen des Eingabe-Stacks löschen**: entfernt Beschriftungen aus den Originalbildern

Diese Optionen finden sich im Untermenü **Bearbeiten → Beschriftung löschen**.

---

## 11. Einstellungen und Konfiguration

### Kachel-Einstellungen (Speicheroptimierung)

Für große Bilder verwendet OpenFocus eine kachelbasierte Verarbeitung, um Speicherprobleme zu vermeiden.

Zugriff über **Einstellungen → Kachel-Einstellungen**:

- **Kachelblockgröße**: Größe jeder Verarbeitungskachel (Standard: 1024 px)
- **Kachelüberlappung**: Überlappung zwischen Kacheln für nahtloses Zusammensetzen (Standard: 256 px)
- **Kachelschwelle**: Bildgröße, ab der Kacheln verwendet werden (Standard: 2048 px)

**Empfehlungen**:
- Kleinere Blockgröße: geringerer Speicherbedarf, langsamere Verarbeitung
- Größere Blockgröße: höherer Speicherbedarf, schnellere Verarbeitung
- Überlappung erhöhen, wenn im Ergebnis sichtbare Nähte auftreten

### Registrierungseinstellungen

Zugriff über **Einstellungen → Registrierungseinstellungen**:

- **Verkleinerungsbreite**: Auflösung der Vorverarbeitung für die Registrierung
  - Standard: 1024 px
  - Niedriger für Geschwindigkeit, höher für Genauigkeit

### Leistungsreferenz

Siehe [PERFORMANCE.md](./PERFORMANCE.md) für gemessene Durchsatz- und Speicherwerte
auf großen realen Stapeln sowie praktische Hinweise für tiefe Stapel und
hochauflösende Eingaben.

### Thread-Einstellungen

Steuert die parallele Verarbeitung auf der CPU:

Zugriff über **Einstellungen → Thread-Einstellungen**:

- `0` bedeutet automatisch und ist die Vorgabe für neue Einstellungen.
- Positive Werte begrenzen die Thread-Anzahl; gespeicherte Grenzen bleiben erhalten.
- CPU-Kapazität, Ressourcenmodus und freier Speicher können die Parallelität verringern.

### GPU-Beschleunigung

OpenFocus nutzt die GPU (CUDA/MPS) nur für das neuronale KI-Modell (StackMFF V3); klassische Algorithmen laufen immer auf der CPU.

- Umschalten über **Einstellungen → GPU-Beschleunigung**
- Deaktivieren zwingt die gesamte Verarbeitung auf die CPU – nützlich, wenn GPU-Treiber Probleme bereiten
- Bei deaktivierter GPU zeigt das Statusfeld `CPU*`
- Die Wahl greift beim nächsten Rendern und wird gespeichert

### KI-Batchgröße

Zugriff über **Einstellungen → KI-Batchgröße**:

- Die gespeicherte Batch-Obergrenze ist standardmäßig 2; tatsächlich wird derzeit auf jedem Gerät eine Kachel pro Aufruf verarbeitet.
- Ein höherer Wert aktiviert keine parallele GPU-Batch-Verarbeitung.

### Oberflächendesign

Es stehen zwei Oberflächendesigns zur Verfügung: **Dunkel (Standard)** und **Hell**. Umschalten über
**Einstellungen → Design**; die Wahl wird sofort angewendet (die Windows-Titelleiste folgt) und
gespeichert. Beim ersten Start wird das Design im Willkommensdialog gewählt.

### Portabler Modus

Eine leere Datei namens `OpenFocus.portable` neben `OpenFocus.exe` anlegen
(das portable ZIP enthält bereits eine) – dann liegen alle Benutzerdaten (Einstellungen,
Logs, zuletzt geöffnete Dateien) in einem Ordner `OpenFocusData` neben der Anwendung
statt im Benutzerprofil, sodass die Installation auf einem USB-Stick mitgenommen werden
kann. Zum Zurückschalten auf das normale Profil die Markierungsdatei löschen.

### EXIF-Metadaten

Quellmetadaten (Kameramodell, Belichtung, Ausrichtung und so weiter) bleiben erhalten:
Die Ausrichtung wird beim Laden angewendet, und PNG/TIFF/WebP-Exporte betten die
ursprünglichen EXIF-Daten automatisch ein. JPG/BMP-Exporte können keine vollständigen
EXIF-Daten speichern und werden ohne sie geschrieben.

### HEIC/HEIF-Eingabe

iPhone-Fotos lassen sich direkt laden. Erfordert das optionale Paket `pillow-heif`
(`pip install pillow-heif`; in den gepackten Builds enthalten).

### Updates

Projektdateien (`.ofproj`) verhalten sich wie ein Dokument: Die Titelleiste zeigt den
Projektnamen mit einem `*`, solange ungespeicherte Änderungen bestehen, und beim
Schließen der Anwendung oder Öffnen eines anderen Projekts wird vor dem Verwerfen
gefragt. Das Projekt speichert die Bildliste, die Ladeskalierung, alle Rendering-Optionen,
die Beschriftungen und das aktuelle Einzelbild – die Bilder selbst werden referenziert,
nicht kopiert.

Hilfe → Nach Updates suchen vergleicht die eigene Version mit den GitHub-Releases.
**Herunterladen und installieren** lädt den neuen Installer in der App herunter (mit
Fortschrittsanzeige) und startet ihn; Downloads-Seite öffnen führt zur Release-Liste.
Die Prüfung läuft höchstens einmal täglich im Hintergrund.

### Registrierungs-Cache

OpenFocus kann Registrierungsergebnisse in einem Ordner `.openfocus_cache` neben dem
Quellstapel ablegen. Wird derselbe Stapel erneut geöffnet (Dateien unverändert),
entfällt die Ausrichtung vollständig. Umschalten über **Einstellungen →
Registrierungs-Cache**; veraltete Einträge werden automatisch ersetzt.

### Gespeicherte Einstellungen

Alles Folgende übersteht einen Neustart der Anwendung automatisch:
Thread-Anzahl, Kachelparameter, Verkleinerungsbreite der Registrierung, Batchgröße der
KI-Fusion (StackMFF V3), der Schalter für die GPU-Beschleunigung, der Schalter für den
Registrierungs-Cache, die Oberflächensprache, Fenstergröße/Panel-Layout, das
Exportformat beim Herausziehen und die zuletzt geöffneten Stapel. Einstellungsdialoge
speichern in dem Moment, in dem bestätigt wird; zusätzlich wird alles beim Beenden der
Anwendung gesichert. Die Voreinstellungen liegen pro Benutzer in `OpenFocus/OpenFocus.ini`
(QSettings-INI-Format).

### Oberflächensprache

Jederzeit über **Einstellungen → Sprache → English / 中文** umschaltbar. Die Änderung
greift sofort und wird für künftige Sitzungen gespeichert.

### Automatische Update-Prüfung

Einmal täglich, wenige Sekunden nach dem Start, prüft OpenFocus still auf GitHub nach
einer neueren Version. Existiert eine, erscheint ein Banner über den Bildbereichen: die
Version, **Jetzt aktualisieren** (lädt herunter und startet in einem Schritt neu),
**Neuerungen** (Versionshinweise) und eine Schaltfläche zum Ausblenden. Ausblenden
verbirgt das Banner nur für diese Version – die nächste Version bringt es zurück, und
**Hilfe → Nach Updates suchen** aktiviert es sofort wieder.

Während der Download im Hintergrund läuft, zeigt die Statusleiste den Fortschritt.
Eine manuelle Prüfung ist jederzeit über **Hilfe → Nach Updates suchen** möglich.

---

## 12. Stapelverarbeitung

Mehrere Ordner automatisch verarbeiten:

1. Auf **Werkzeuge → Stapelverarbeitung** gehen oder die Stapeloption anklicken
2. Im Stapeldialog:
   - **Ordner hinzufügen**: Ordner mit Bildsequenzen auswählen
   - **Auswahl entfernen**: ausgewählte Ordner aus der Liste entfernen
   - **Alle entfernen**: alle Ordner entfernen
3. Ausgabeeinstellungen konfigurieren:
   - **Unterordner im Quellordner anlegen**: erzeugt die Ergebnisse in jedem Eingabeordner
   - **Wie Quellordner**: überschreibt die Eingabe bzw. legt die Ergebnisse daneben ab
   - **Ausgabeordner festlegen**: Zielort festlegen
4. Die aktuellen Fusions-/Registrierungseinstellungen prüfen
5. **Stapelverarbeitung starten** anklicken, um zu beginnen

Der Stapeldialog zeigt den Fortschritt in Echtzeit. Die Verarbeitung kann jederzeit
abgebrochen werden.

### Hinweise zur Stapelverarbeitung

- Verwendet dieselben Fusions- und Registrierungseinstellungen wie die manuelle Verarbeitung
- Jeder Ordner wird unabhängig verarbeitet
- Dateinamen enthalten Zeitstempel und Parameter zur Identifikation

---

## 13. Ergebnisse exportieren

### Einzelnes Ergebnis speichern

1. Das gewünschte Ergebnis in der Ausgabeliste auswählen
2. Auf **Datei → Speichern** gehen oder `Ctrl+S` drücken
3. Format und Speicherort wählen
4. Auf Speichern klicken

### Exportformat beim Herausziehen

Werden Ergebnisse aus der Ausgabeliste herausgezogen, entstehen Bilddateien im Format,
das im Kontextmenü der Ausgabeliste unter **Format beim Herausziehen** gewählt wurde
(JPG, PNG oder TIFF). Die Wahl wird gespeichert.

### Registrierten Stapel speichern

1. Auf **Datei → Stack speichern → Registrierter Stack** gehen
2. Speicherformat wählen:
   - **Als Ordner speichern**: speichert jedes ausgerichtete Bild als eigene Bilddatei
   - **Als GIF speichern**: erzeugt eine animierte GIF-Datei aus der ausgerichteten Sequenz

### Eingabe-Stack speichern

1. Auf **Datei → Stack speichern → Eingabe-Stack** gehen
2. Speicherformat wählen:
   - **Als Ordner speichern**: speichert die Originalbilder als eigene Dateien
   - **Als GIF speichern**: erzeugt eine animierte GIF-Datei aus der Eingabesequenz

### Exportformate

- **JPG**: komprimiert, gutes Verhältnis von Qualität zu Dateigröße
- **PNG**: verlustfrei, unterstützt Transparenz
- **BMP**: unkomprimiert, maximale Qualität
- **TIFF**: hohe Qualität, unterstützt Ebenen
- **GIF**: Animationsformat

---

## 14. Kommandozeilenschnittstelle

OpenFocus kann ohne GUI laufen – nützlich für skriptgesteuerte Pipelines und CI.
Aus einer Python-Umgebung (oder neben der gepackten ausführbaren Datei):

```bash
# Einen Ordner mit dem standardmäßigen Guided Filter fusionieren
python main.py --input ./stack_folder --output ./result/fused.png

# DTCWT mit ECC-Registrierung, 8 Threads
python main.py -i ./stack_folder -o ./result/fused.png -m dtcwt -a ecc -t 8

# StackMFF V3 auf CPU erzwungen, mit eigener Kachelgröße
python main.py -i ./stack_folder -o ./result/fused.png -m stackmffv3 --cpu --tile-size 512

# Explizite Dateiliste; Videodateien werden ebenfalls als Eingabe akzeptiert
python main.py -i img1.jpg img2.jpg img3.jpg -o fused.png -m gfgfgf

# Einzelner mehrseitiger TIFF-Z-Stapel mit metadatenkalibriertem Maßstabsbalken
python main.py -i ./zstack.tif -o fused.png -m dtcwt --scale-bar
```

| Option | Beschreibung |
|--------|-------------|
| `--input, -i` | Bildordner, Videodatei oder eine bzw. mehrere Bilddateien |
| `--output, -o` | Pfad des Ausgabebildes (die Endung bestimmt das Format) |
| `--method, -m` | `stackmffv3` (Standard), `none`, `openfocus_a`, `openfocus_b`, `openfocus_c`, `guided_filter`, `dct`, `dtcwt`, `gfgfgf` |
| `--align, -a` | `none` (Standard), `homography`, `ecc`, `both`, `openfocus` |
| `--align-shift-limit` | Nur für die `openfocus`-Ausrichtung: maximal akzeptierte Verschiebung als Anteil der Bildgröße (Standard 0,08; darüber liegende Bilder bleiben unausgerichtet) |
| `--exposure-match` | Den Mittelwert jedes Bildes vor der Fusion an das letzte Bild angleichen (Standard: an; `--no-exposure-match` deaktiviert). Verstärkungen gelten für den ganzen Stapel, sodass gekachelte Ausgaben nahtlos bleiben; wirkt bei jeder Methode und auf die Helligkeitsreferenz des OpenFocus-Ausrichters |
| `--base-frame` | Optionaler Schritt nach der Fusion: die niedrigen Frequenzen der Ausgabe an das `last`/`first`-Bild angleichen (Standard `none`) |
| `--sort` | Bildreihenfolge für `openfocus_a/b/c`: `asc` (Standard, Dateireihenfolge), `desc` (kehrt einen Fern-nach-Nah-Durchlauf um), `off` |
| `--dust-map WHITE` | Flat-Field-Korrektur (Staubkarte) aus einer defokussierten Weißfeldaufnahme |
| `--kernel, -k` | Kernelgröße (ungerade erzwungen; bei der OpenFocus-Familie die Quelle für den Radius des Varianzfensters) |
| `--depth-smoothing SIGMA` | Nur `openfocus_b`: Gauß-Sigma des Tiefenübergangs von Bild zu Bild (Standard 1,2; 0 behält das feinste Detail) |
| `--blend-smoothing SIGMA` | Nur `openfocus_a` / `openfocus_c`: Gauß-Sigma der Gewichtskarte pro Bild (Standard 2,0 für A, 3,0 für C; 0 behält die schärfste Auswahl pro Pixel) |
| `--threads, -t` | Thread-Obergrenze; `0` = automatisch (Standard), positive Werte begrenzen die Parallelität |
| `--resource-preference` | Ressourcenmodus: `balanced` (Standard), `responsive` oder `maximum` |
| `--batch-size` | KI-Kachel-Batch-Obergrenze (Standard 2); tatsächlich derzeit eine Kachel pro Aufruf |
| `--cpu` | CPU erzwingen, auch wenn eine GPU verfügbar ist |
| `--downscale` | Verkleinerungsbreite der Merkmalserkennung bei der Registrierung |
| `--tile-size` | Kachelblockgröße für große Bilder |
| `--output-dir, -d` | Stapelmodus: jeden Eingabeordner fusionieren und pro Stapel `<folder>.png` schreiben |
| `--scale-bar` | Maßstabsbalken zeichnen (Kalibrierung automatisch aus den TIFF-Metadaten erkannt) |
| `--depth-map [path]` | Zusätzlich eine kolorierte Tiefenkarte (Fokusposition) speichern (Standard: `<output>_depth.png`) |
| `--depth-colormap` | Farbtabelle für `--depth-map`: turbo, viridis, jet, gray, … (Standard turbo) |
| `--scale-um` | Pixelgröße in µm/px, überschreibt die Metadaten (impliziert `--scale-bar`) |

Exit-Codes: `0` Erfolg, `1` Verarbeitungsfehler, `2` Nutzungsfehler. Der Aufruf
`python main.py --help` gibt alle Optionen aus; ein Start mit einem reinen Pfad
(`python main.py C:\stack`) öffnet weiterhin die GUI mit diesem Ordner.

---

## 15. Tastenkürzel

### Dateioperationen

| Tastenkürzel | Aktion |
|----------|--------|
| `Ctrl+O` | Ordner öffnen |
| `Ctrl+Shift+O` | Video öffnen |
| `Ctrl+S` | Ergebnis speichern |
| `Ctrl+Shift+S` | Registrierten Stack speichern |
| `Ctrl+W` | Stack leeren |
| `Ctrl+Q` | Beenden |
| `Ctrl+R` | Rendern starten |
| `Ctrl+Shift+R` | Alle vergleichen |
| `Ctrl+T` | Ergebnis retuschieren |
| `Ctrl+M` | Tiefenkarte |
| `Ctrl+E` | Wischvergleich umschalten |
| `Ctrl+Shift+B` | Stapelverarbeitung |
| `Ctrl+Z` / `Ctrl+Y` | Rückgängig / Wiederholen (Stapelbearbeitungen) |

### Navigation

| Tastenkürzel | Aktion |
|----------|--------|
| `Pfeil links` | Vorheriges Bild |
| `Pfeil rechts` | Nächstes Bild |
| `Mausrad` | Hinein-/Herauszoomen |
| `Ctrl + Mausrad` | Feinzoomen |
| `Doppelklick` | Zwischen Einpassen und 100-%-Ansicht wechseln |

### Allgemein

| Tastenkürzel | Aktion |
|----------|--------|
| `Leertaste` | Verschiebemodus umschalten |
| `Entf` | Ausgewähltes Element löschen |

---

## 16. Fehlerbehebung

### Häufige Probleme

#### „Keine Bilder geladen“

- Sicherstellen, dass zuerst ein Bildordner oder ein Video importiert wurde
- Prüfen, ob der Ordner unterstützte Bildformate enthält

#### Fusionsergebnisse wirken unscharf oder doppelt

- Die Bildregistrierung aktivieren (im Dropdown eine Methode wählen: Homographie / ECC / OpenFocus schnell)
- Sicherstellen, dass die Bilder vor der Fusion korrekt ausgerichtet sind
- Einen anderen Fusionsalgorithmus versuchen
- Die Kernelgröße für glattere Ergebnisse erhöhen

#### KI nicht verfügbar

- PyTorch installieren: `pip install torch torchvision`
- Für GPU-Unterstützung eine CUDA-fähige PyTorch-Version installieren
- Über **Hilfe → Umgebungsinformationen** prüfen

#### Fehler wegen zu wenig Arbeitsspeicher

- Die Kachelverarbeitung in **Einstellungen → Kachel-Einstellungen** aktivieren
- Die Kachelblockgröße verringern
- Kleinere Kernelgrößen verwenden
- Die Bilder auf kleinere Abmessungen skalieren

#### Registrierung schlägt fehl

- Immer nur eine Registrierungsmethode gleichzeitig verwenden
- Die Verkleinerungsbreite in **Einstellungen → Registrierungseinstellungen** reduzieren
- Sicherstellen, dass die Bilder ausreichend Kontrast und Merkmale haben

#### Langsame Verarbeitung

- Die Thread-Anzahl in **Einstellungen → Thread-Einstellungen** erhöhen
- Eine leistungsfähigere CPU verwenden
- Die Bildauflösung reduzieren
- Einfachere Fusionsalgorithmen verwenden (Guided Filter statt KI-Fusion)

#### Windows-SmartScreen-Warnung beim ersten Start

Die veröffentlichten Binärdateien sind nicht signiert. Auf **Weitere Informationen →
Trotzdem ausführen** klicken. Der erste Start nach dem Entpacken kann 30–60 Sekunden dauern.

#### macOS meldet, die App „kann nicht geöffnet werden“ oder „ist beschädigt“

Die App ist nicht signiert und nicht notarisiert. Nach dem Verschieben von
`OpenFocus.app` in den Ordner Programme ausführen:

```bash
xattr -cr /Applications/OpenFocus.app
```

oder mit der rechten Maustaste auf die App klicken und **Öffnen** wählen. Die
veröffentlichten macOS-Builds sind für Apple Silicon (M1/M2/M3/M4); Intel-Macs können
aus dem Quellcode laufen.

### Hilfe erhalten

1. **Umgebungsinformationen**: Über **Hilfe → Umgebungsinformationen** die Installation prüfen
2. **Kontakt zum Support**: Über **Hilfe → Kontakt** Supportinformationen aufrufen

---

## Anhang: Details zu den Fusionsalgorithmen

### Guided Filter

Der Guided Filter führt eine kantenbewahrende Glättung mit einem Führungsbild durch.
Er berechnet lokale lineare Transformationen zwischen Führungs- und Eingabebild und
wendet diese auf die Ausgabe an. Diese Methode zeichnet sich dadurch aus, dass sie
Kanten erhält und zugleich Rauschen reduziert.

### DCT (Diskrete Kosinustransformation)

DCT überführt Bilder in den Frequenzbereich, in dem Fusionsentscheidungen anhand der
Koeffizientenbeträge getroffen werden. Hochfrequente Anteile (Details) bleiben erhalten,
während niederfrequente Anteile (gleichmäßige Bereiche) anhand der lokalen Energie
zusammengeführt werden.

### DTCWT (Dual-Tree-Complex-Wavelet-Transformation)

DTCWT liefert eine mehrskalige, mehrdirektionale Zerlegung von Bildern. Es erfasst
Merkmale auf verschiedenen Skalen und in verschiedenen Orientierungen und ermöglicht
damit ausgefeilte Fusionsregeln, die sowohl Textur- als auch Strukturinformationen
erhalten.

### GFG-FGF

Implementiert Fu et al. (2024): Die grobe Fokuskarte |I − mean(I)| wird mit dem
gaußartigen Vier-Nachbarschafts-Gradienten angereichert – die 3×3-Schablone
T = [1 2 1; 2 4 2; 1 2 1]/16 ergibt P = I⊗T, und
G = (|I(x±1,y) − P|)/2 + (|I(x,y±1) − P|)/2, wobei Werte unterhalb von T0 = 0,005
verworfen werden – und anschließend durch einen schnellen Guided Filter verfeinert.
Jedes Pixel wählt das Bild mit dem größten Fokuswert (bei Gleichstand: 3×3-Sobel-
Kleinbereichsmaß), und die binären Entscheidungskarten werden vor der gewichteten
Mischung mit dem schnellen Guided Filter geglättet.

### KI-Fusion (StackMFF V3)

Ein Deep-Learning-Ansatz mit einem neuronalen Netz, das auf großen Datensätzen von
Multifokus-Bildpaaren trainiert wurde. Das Netz erlernt optimale Fusionsstrategien
automatisch und liefert bei richtiger Konfiguration Ergebnisse auf dem aktuellen Stand
der Technik.

---

*OpenFocus ist ein Open-Source-Projekt. Für Updates, Fehlerberichte und Funktionswünsche
besuchen Sie bitte das Projekt-Repository.*
