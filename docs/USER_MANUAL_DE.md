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
- **Zehn Fusionsverfahren**: Guided Filter, DCT, DTCWT, GFG-FGF, StackMFF, StackMFF V2, StackMFF V3 sowie OpenFocus A, B und C; zusätzlich ein reiner Ausrichtungsmodus
- **Bildregistrierung**: Fehlausgerichtete Bildsequenzen mit ECC, Homographie, Homographie mit anschließender ECC-Verfeinerung oder der schnellen OpenFocus-Ausrichtung ausrichten
- **Automatischer Hintergrund**: einen unruhigen gestackten Hintergrund durch ein echtes Einzelbild ersetzen und sehen, wo kein Bild die Szene klar erfasst hat
- **Quellengebundene Retusche**: Pixel aus den Bildern malen, die das Ergebnis tatsächlich erzeugt haben, mit Quell- und Wiederherstellungspinsel, Belichtungspinsel und Rückgängig pro Strich
- **Wischvergleich**: Ausrichtung prüfen und Ergebnisse mit einem A/B-Trenner vergleichen
- **Projektdateien**: reproduzierbare Einstellungen und Quellverweise in `.ofproj`-Dateien speichern; Ergebnis- und Retuschepixel werden getrennt exportiert
- **Verarbeitungsvorlagen**: Fusions-, Registrierungs- und Ressourceneinstellungen als `.oftemplate` speichern und auf jeden geladenen Stapel anwenden
- **Mehrseitige TIFF**: Z-Stapel aus einer einzigen Datei werden seitenweise geladen und wieder in eine Datei exportiert
- **Tiefenkarte**: Fokuspositionskarten pro Pixel mit wissenschaftlichen Farbskalen, Bildindex- und physikalischen Exporten sowie 3D-Oberfläche
- **Maßstabsbalken**: publikationsreife Maßstäbe mit automatischer µm/px-Kalibrierung aus TIFF-Metadaten
- **Automatische Absturzwiederherstellung**: nach einem Absturz wird die letzte Arbeitssitzung zum Wiederherstellen angeboten
- **Zwölf Oberflächensprachen**: English, 中文, 日本語, Español, Deutsch, Français, Русский, 한국어, Português, Italiano, Nederlands und Polski, mit automatischer Erkennung der Systemsprache
- **Headless- und Agent-CLI**: headless Pipelines für CI skripten oder ein laufendes Fenster über den lokalen Kontrollkanal steuern, einschließlich Rezepten und Retusche-Wiedergabe
- **Flexibler Export**: Ergebnisse als einzelne Bilder, Ordner, mehrseitige TIFF, TIFF mit Ebenen oder GIF-Animationen speichern
- **Bildtransformationen**: Bildstapel drehen, spiegeln und skalieren
- **Beschriftungen**: verwaltete Textbeschriftungen für Quell-, registrierte und Ergebnisstapel
- **Persistente Einstellungen**: Voreinstellungen, Sprache und zuletzt geöffnete Stapel überstehen einen Neustart

### Unterstützte Eingabeformate

- Bilddateien: JPG/JPEG, PNG, BMP, TIFF, WebP und weitere gängige Formate
- Videodateien: MP4, AVI, MOV, MKV, WMV, FLV und WEBM (werden automatisch in Bildsequenzen dekodiert)
- Ordner mit nummerierten Bildsequenzen
- Kamera-RAW-Dateien (DNG, CR2/CR3, NEF, ARW, RAF, ORF, RW2 und weitere gängige RAW-Formate). Die Dekodierung übernimmt `rawpy` (LibRaw), das durch `pip install openfocus` mitinstalliert wird und in den Windows-/macOS-Builds enthalten ist

Transparente PNG-Stapel lassen sich in der Desktop-App, in der Stapelverarbeitung und in der Kommandozeile ohne Gruppenfusion verarbeiten. Jedes Bild muss ein 8- oder 16-Bit-PNG gleicher Abmessungen und Bittiefe mit derselben Alphamaske sein. Die Maske muss außerhalb genau eines nicht leeren Rechtecks null sein und innerhalb dieses Rechtecks eine konstante, von null verschiedene Deckkraft haben; teilweise transparente konstante Deckkraft ist zulässig. Nur das sichtbare Rechteck wird fusioniert, sodass verdeckte Farben weder die Fokusmessung noch Kachelgrenzen beeinflussen. Die PNG-Ausgabe behält Leinwandgröße und Deckkraft; verdeckte Ausgabefarben sind null. Quell- und Ergebnisansicht zeigen Transparenz vor einem Schachbrettmuster.

Laden Sie in voller Größe, wählen Sie keine Ausrichtung, deaktivieren Sie Schnellvorschau und ROI, rendern Sie und speichern Sie als PNG. Die Stapelverarbeitung verlangt ein Fusionsverfahren und eine PNG-Ausgabe ohne Export eines ausgerichteten Stapels. Ausrichtung, Größenänderung oder Downsampling, Flachfeldkorrektur, Tiefenkarten, Methodenvergleich, Retusche und Maßstabsbalken sind für transparente Daten nicht verfügbar; Beschriftungen lassen sich nicht in einen Export einbrennen. Ein Fusionsverfahren, das die Abmessungen des Rechtecks ändert, wird abgelehnt. Abweichende oder weich auslaufende Masken, unterschiedliche Masken pro Bild, transparente TIFF-Eingaben und Nicht-PNG-Ausgaben – einschließlich GIF und mehrseitiger TIFF – sind nicht möglich. Die CLI lehnt außerdem Sensorbereinigung und Tiefenkarten für transparente Eingaben ab.

Das CLI-Äquivalent lautet beispielsweise `python main.py --input ./png-stack --output ./result.png --method guided_filter --align none`.

### Unterstützte Ausgabeformate

Unter **Speichern unter** speichert gewöhnliches TIFF das Ergebnis als einzelnes Bild; TIFF mit Ebenen enthält die Quellbilder als Ebenen. Linear DNG wird immer angezeigt, kann aber nur mit einem kameralinearen Ergebnis ohne eingebrannte Bearbeitungen aus **Natives RAW öffnen** mit Kamerakalibrierung gespeichert werden.

- Einzelne Bilder: JPG, PNG, BMP, TIFF, WebP
- Bildsequenzen: als Ordner gespeichert
- Animationen: GIF-Format
- Mehrseitige TIFF: der gesamte Stapel in einer Datei, mit 8 oder 16 Bit
- TIFF mit Ebenen: Retuscheergebnis und Quellbilder als Ebenen

---

## 2. Erste Schritte

### Installation

Laden Sie den aktuellen Windows- oder macOS-Build von der [OpenFocus-Website](https://openfocus.top/#download). Dieselben Dateien und ältere Versionen stehen auch unter [GitHub Releases](https://github.com/Xinzhe99/OpenFocus/releases) bereit.

| Plattform | Datei | Typ |
|-----------|-------|-----|
| Windows 10/11 (64 Bit) | `OpenFocus-*-setup.exe` | **Installer** (empfohlen) |
| Windows 10/11 (64 Bit) | `OpenFocus-*-windows-x64.zip` | Portabel (entpacken und starten) |
| macOS Apple Silicon (M1–M4) | `OpenFocus-*-macos.dmg` | Disk-Image (in „Programme“ ziehen) |
| macOS Apple Silicon (M1–M4) | `OpenFocus-*-macos-arm64.zip` | Portabel (entpacken) |

Ob die Windows-Builds signiert sind, hängt vom jeweiligen Release ab (siehe Release-Notes); eine Signatur ist nicht garantiert. Beim ersten Start kann daher eine Sicherheitsabfrage erscheinen: unter **Windows** SmartScreen → „Weitere Informationen“ → „Trotzdem ausführen“ (der erste Start kann 30–60 s dauern). Die macOS-Builds sind nicht notarisiert; nach der DMG-Installation (in „Programme“) führen Sie bei Bedarf `xattr -cr /Applications/OpenFocus.app` aus, nach der Terminal-Installation (nach `~/Applications`) `xattr -cr ~/Applications/OpenFocus.app`, oder Sie klicken mit der rechten Maustaste auf die App und wählen **Öffnen**.

Für die Installation im Terminal zuerst den Befehl für macOS oder Linux mit `--dry-run` prüfen und anschließend ohne diese Option ausführen. Unter Linux installiert das die Headless-CLI; macOS-Builds erfordern Apple Silicon und werden nach `~/Applications` installiert.

```bash
curl -fsSL https://openfocus.top/install.sh | sh -s -- --dry-run
curl -fsSL https://openfocus.top/install.sh | sh
```

Unter Windows den Installer aus PowerShell starten:

```powershell
irm https://openfocus.top/install.ps1 | iex
```

Alternativ die CLI unter Windows, macOS und Linux installieren – dieselbe Pipeline, ohne GUI-Stack (das Wheel hängt bewusst nicht von PyQt6 ab):

```bash
pip install openfocus
openfocus -i ./stack_folder -o ./result/fused.png
```

### Anwendung starten

Nach der Installation starten Sie OpenFocus so:

1. **Windows (ausführbare Datei)**: Doppelklick auf die OpenFocus-Verknüpfung oder die ausführbare Datei
2. **macOS**: `OpenFocus.app` aus dem Ordner „Programme“ öffnen
3. **Python-Quellcode**: `pip install -r requirements.txt` ausführen und anschließend `python main.py` in der aktivierten Umgebung starten

Der erste Start öffnet den Willkommensdialog, in dem Sie das Design wählen und den mitgelieferten Demo-Stapel laden können. Ein frisches Profil verwendet standardmäßig das helle Design; die Auswahl wird gespeichert.

### Schnellstart-Ablauf

1. Bildsequenz importieren (Datei → Ordner öffnen oder per Drag-and-drop)
2. Fusionseinstellungen im rechten Bereich konfigurieren
3. Die Registrierung startet mit **ECC**; wählen Sie **Keine Ausrichtung**, wenn die Bilder bereits ausgerichtet sind
4. Kernelgröße oder methodspezifische Parameter bei Bedarf ändern
5. *(optional)* **Rendereinstellungen** öffnen und **Schnellvorschau** aktivieren, um die Parameter zuerst an einem verkleinerten Entwurf zu testen
6. Auf **▶ Rendern starten** klicken, um die Verarbeitung zu beginnen
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

1. **Fusionseinstellungen**: Fusionsverfahren und dessen Parameter wählen
2. **Registrierungseinstellungen**: Ausrichtungsmethode und zugehörige Parameter wählen
3. **Rendereinstellungen**: Der Dialog enthält Schnellvorschau, Bildreihenfolge, Belichtungsabgleich, Entfernung der Helligkeitsdrift und Ausgabeschärfung
4. **Aktionsschaltflächen**: Rendern starten sowie Zurücksetzen, Rendereinstellungen und alle Methoden rendern
5. **Liste der Quellbilder**: alle geladenen Bilder mit Dateinamen und Auswahlkästchen
6. **Ausgabeliste**: die erzeugten Ergebnisse

### Menüleiste

Die Menüleiste am oberen Rand bietet Zugriff auf alle Funktionen:

- **Datei**: Importieren, Projekte, Verarbeitungsvorlagen, Speichern, Stapel speichern und die Liste der zuletzt geöffneten Dateien
- **Bearbeiten**: Bildtransformationen, Retusche und Bearbeitung einzelner Bilddateien
- **Werkzeuge**: Bilder zusammenfügen, Stapelverarbeitung, Methodenvergleich, Tiefenkarte, Staubkarte (Flat-Field), Maßstabsbalken und Beschriftungen
- **Einstellungen**: Erweiterte Einstellungen (Registrierung und Ausrichtungsdiagnose), GPU-Beschleunigung, Design und Sprache
- **Hilfe**: Schnellstart, Umgebungsinformationen, Feedback, Kontakt, Log-Ordner und Update-Prüfung

**Werkzeuge → Bilder zusammenfügen…** öffnet einen separaten Arbeitsbereich für bereits fusionierte Mikroskopbereiche und Kamerapanoramen; der aktuelle Stapel bleibt unverändert. Die Registerkarte **Projektassistent** bündelt Import, Ausrichtung und Kontrollpunkte, **Quellbilder** enthält Liste, Raster und manuelle Platzierung, **Panoramaeditor** enthält Naht-, Überblendungs-, Belichtungs-/Farb-, Ausgabe-, Projektions-, Kamera- und HDR-Einstellungen sowie Vorschau-Überlagerungen. Mit **Dateien hinzufügen…**, **Ordner öffnen** oder **Ergebnisse hinzufügen…** kopieren Sie ausgewählte Ergebnisse der aktuellen Sitzung, oder Sie ziehen Dateien bzw. Ordner auf Liste oder Leinwand. Der Ordnerimport umfasst nur unterstützte Dateien direkt im Ordner, keine Unterordner. Importe folgen der natürlichen Dateinamenreihenfolge, ignorieren doppelte Pfade und fügen jede Seite mehrseitiger TIFFs als eigenes Bild hinzu. Ein ausgewähltes Bild lässt sich mit Entf oder Rücktaste aus dem Arbeitsbereich entfernen, ohne die Datei zu löschen.

**Automatisch ausrichten** sucht gemeinsame Merkmale auf begrenzten Miniaturbildern (maskierte und transparente Pixel werden ignoriert) und passt jede verbundene Überlappung an. Der Planarmodus schätzt Drehung, Skalierung und Verschiebung; der Panoramamodus passt zusätzlich Kamera- und Objektivmodell an. Bilder ohne genügend gemeinsame Merkmale bleiben als nicht zugeordnet markiert. Im Planarmodus: Ziehen Sie sie oder geben Sie X/Y in Pixeln, Skalierung in Prozent und Drehung in Grad ein, auch aus bekannten Objektträgerkoordinaten. Bildinhalt kann die Lage eines merkmalslosen, nicht überlappenden Bereichs nicht ableiten, und automatische Zuordnung ist keine physikalische Kalibrierung. Für ein bekanntes Objektträger-Raster legen Sie Zeilen, Spalten, Aufnahmereihenfolge, Richtung, Startecke und geschätzte Überlappung fest und drücken **Rasterpositionen anwenden**. Die Zellenzahl muss der Bildzahl entsprechen, und die Rasterplatzierung gilt nur im Planarmodus. Änderungen an den Bedienelementen lassen die aktuellen Positionen bis zum Anwenden unberührt; Anwenden ersetzt sie und löscht Zuordnungsverbindungen. Prüfen Sie jede Überlappung und passen Sie X/Y, Skalierung und Drehung vor dem Speichern an.

**Kontrollpunkte** öffnet einen Editor mit zwei Vorschauen. Wählen Sie Bild A und B, klicken Sie in beiden Vorschauen auf dasselbe Merkmal und drücken Sie **Punktpaar hinzufügen**; gespeichert werden native Quellpixel, auch wenn Vorschauen und Marker skaliert sind. Die Tabelle zeigt Aktivstatus, Bilder, A/B-Koordinaten, manuelle oder automatische Herkunft und Restfehler in Pixeln. Korrigieren Sie Koordinaten, entfernen Sie bei verdächtigen Punkten **Aktiv**, oder verwenden Sie **Auswahl löschen**, **Schlechteste 10% löschen** und **Nicht benachbarte Paare löschen**. **Optimieren** passt alle aktiven Paare an und meldet die Ausrichtungsqualität: aktive Punkte, schwache Paare, unverbundene Bilder und p95-Restfehler in Pixeln. Ein Paar braucht mindestens vier aktive Punkte, um die Anpassung zu stützen. Automatische Punkte überstehen manuelle Änderungen; prüfen Sie wiederholte Muster, Parallaxe und Präparatbewegung. Kontrollpunktänderungen erzeugen keine neue Vollvorschau: Drücken Sie nach Bearbeiten oder Optimieren erneut **Überblendung anzeigen**.

**Quellmaske** bearbeitet die Maske eines Bildes in Quellauflösung. Der Pinselradius wird in Quellpixeln angegeben; **Ausschließen** färbt Pixel rot, **Einschließen** stellt sie wieder her, **Maske löschen** entfernt die Grundlage, und **Maske importieren…** nimmt ein PNG oder TIFF auf, dessen Abmessungen exakt der Quelle entsprechen (PNG: Alpha, Graustufen oder BGR; TIFF: Graustufen oder BGR ohne Alpha). **Anwenden** übernimmt die Striche, **Abbrechen** behält die vorherige Maske. Ausgeschlossene Pixel tragen weder zu Nähten noch zur Merkmalszuordnung oder zu Ausgabepixeln bei; eine Maske oder RGBA-Quelle ermöglicht Transparenz in der Ausgabe.

Der **Panoramaeditor** steuert die eigentliche Überblendung. Unter **Optimale Nähte suchen** sucht **Inhaltsbasierte Naht (DP)** Nähte am Inhalt; **Graphenschnitt** wird nur angeboten, wenn der OpenCV-Build ihn enthält; **Weiche Überblendung** überspringt die Inhaltssuche. Nahtgenauigkeit 1/2/4/8, halbe Übergangsbreite, Kosten nach Farbe oder Farbe und Gradient sowie das Verfeinern von Überlappungsschleifen steuern diese Suche; bei weicher Überblendung sind Genauigkeit, Kosten und Schleifenverfeinerung deaktiviert. **Überblendung** bietet eine echte Mehrbandpyramide (1–7 Stufen, Standard 5), weiche Überblendung oder harte Auswahl ohne Überlappung. **Belichtung und Farbe** bietet Aus, Belichtungskorrektur, Farbbalance oder lokales Verstärkungsfeld, optional mit **In linearem Licht mischen**. Im Panoramamodus stehen objektivbezogene Felder je Bild bereit: Gier-, Nick- und Rollwinkel, horizontaler Bildwinkel, radiale Verzeichnung a/b/c und Objektivzentrum-Versätze. Vignettierung k1–k3 und Belichtungszeit in Sekunden (0 behält die Dateimetadaten, falls vorhanden) gelten für jedes ausgewählte Bild. Diese Werte sind Benutzereingaben oder Schätzungen, keine abgeleitete physikalische Kalibrierung.

**Projektion** wählt **Ebenes Mikroskopmosaik** mit affinem X/Y/Skalierung/Drehung und Rasterunterstützung oder ein **Geradliniges**, **Zylindrisches** oder **Equirektangulares** Panorama, das jede Quelle über Gier-, Nick- und Rollwinkel, Bildwinkel und Objektivmodell ihrer Kamera abbildet; der Panoramamaßstab ist Pixel pro Radiant, und **Bildwinkel optimieren** kann den Kamera-Bildwinkel anpassen. Geradlinige Projektion meldet einen Fehler beim Überschreiten des Horizonts, zylindrische an einem Pol; wählen Sie für solche Szenen equirektangular. Unter **HDR / Belichtungsfusion** mischt **Belichtungsfusion** jede vollständige Reihe aufeinanderfolgender unterschiedlicher Belichtungen, während **HDR (Debevec + Reinhard)** eine Antwortkurve kalibriert und tonemappt. Stellen Sie die Zahl aufeinanderfolgender Belichtungen je Gruppe, die Belichtungszeiten (0 nutzt die Metadaten der jeweiligen Datei, falls vorhanden) und Breite/Zielhelligkeit der Fusion bzw. HDR-Belichtungsskala/Weißpunkt ein. Debevec verlangt für jede Quelle eine positive Belichtungszeit und ausgeschaltete Belichtungskorrektur; HDR lässt sich nicht mit linearer Lichtmischung kombinieren, und unvollständige Gruppen oder unverträgliche photometrische Kombinationen melden einen Fehler.

**Überblendung anzeigen** erfasst den aktuellen globalen Plan und rendert eine begrenzte Vorschau (bis 1024 Pixel lange Kante). **Nähte anzeigen**, **Bildnummern anzeigen** und **Ausgeschlossene Bereiche anzeigen** legen die gewählten Beschriftungen und die Maske darüber; ausgewählte Kontrollpunkte erscheinen als Marker. Die Deckkraft der Quelle betrifft nur die Vorschau im Arbeitsbereich und gelangt nie in die exportierte Überblendung; X/Y, Skalierung und Drehung bestimmen die tatsächliche Platzierung im Planarmodus. Nicht zugeordnete Panoramabilder platzieren Sie über Gier-, Nick- und Rollwinkel sowie den Bildwinkel der Kamera. Der Vorschau-Deckkraftwert bleibt bei Importen und Layoutänderungen erhalten. Jede Änderung an Quelle, Platzierung, Maske, Kontrollpunkt oder Editoreinstellung macht den Plan ungültig; drücken Sie nach Bearbeiten oder Optimieren erneut **Überblendung anzeigen**. Beim Speichern wird der erfasste Plan nur wiederverwendet, wenn Quellen und Einstellungen unverändert sind; andernfalls wird neu geplant und gerendert. Die Ausgabesteuerung bietet Automatisch, 8 Bit oder 16 Bit (Automatisch ist 16 Bit bei einer 16-Bit-Quelle oder HDR, sonst 8 Bit), Ausgabegröße 0,01–10, normalisierten Beschnitt, Transparenz Automatisch/Ein/Aus (Automatisch aktiviert sie bei Panoramaprojektion, maskierten oder RGBA-Quellen und unbedeckter Leinwand) sowie eine Farbprofilstrategie: Automatisch behält ein gemeinsames Profil und wandelt gemischte Profile nach sRGB, sRGB wandelt um, erstes Profil behalten kopiert das erste Tag ohne Pixelumwandlung, kein Profil lässt das Tag weg. Gemischte 16-Bit-Profile kann dieses Backend nicht umwandeln; wählen Sie erstes Profil behalten oder kein Profil, es gibt keine stille Reduktion auf 8 Bit. **Quell-EXIF behalten** bettet bereinigte Kamera-, Objektiv-, Belichtungs-, Datums- und GPS-EXIF der ersten Quelle ein, verwirft Pixel-Layout-Tags und setzt die Ausrichtung zurück. Speichern Sie als TIFF oder PNG. TIFF nutzt diskgestützte Kachelkomposition, streifenweises Schreiben (Ziel 4 MiB pro Streifen, mindestens eine vollständige Ausgabezeile) und bei Bedarf BigTIFF; große PNGs werden in Zeilenbändern kodiert. Sorgen Sie für ausreichend temporären Speicher, Ausgabespeicher und RAM: Die Ausgabe in voller Größe wird auf die Platte gestreamt, Planung und Vorschauen belegen jedoch Arbeitsspeicher. Es sind genügend gemeinsame Merkmale oder manuelle Punkte nötig, und kein bildbasiertes Verfahren leitet den physikalischen Maßstab ab. Quellen und Einstellungen sind während eines Vorgangs eingefroren; verwenden Sie **Abbrechen** und warten Sie auf das Ende.

---

## 4. Bilder und Videos importieren

### Bildordner öffnen

1. Auf **Datei → Ordner öffnen** gehen oder `Ctrl+O` drücken
2. Einen Ordner mit der Bildsequenz auswählen
3. Ein Dialog zur Ladeskalierung erscheint – die Skalierung wählen (100 % = volle Auflösung)
4. Auf OK klicken, um die Bilder zu laden

Im selben Dialog **Nicht mehr fragen – immer mit dieser Skalierung laden** ankreuzen, um künftig ohne Rückfrage mit der gewählten Skalierung zu laden; **Einstellungen → Erweiterte Einstellungen… → Start und Updates → Beim Import nach Ladeskalierung fragen** schaltet die Rückfrage wieder ein. Ein geladener Stapel lässt sich später jederzeit mit **Bearbeiten → Größe ändern** verkleinern.

Die Software wird automatisch:

- Bilddateien erkennen und natürlich nach Dateinamen sortieren
- das erste Bild im Quellbildbereich anzeigen
- die Liste der Quellbilder mit allen Einzelbildern füllen

### Videodateien öffnen

1. Auf **Datei → Video öffnen** gehen oder `Ctrl+Shift+O` drücken
2. Eine Videodatei auswählen (MP4, AVI, MOV, MKV, WMV, FLV, WEBM)
3. Das Video wird automatisch in einzelne Bilder dekodiert
4. Jedes Einzelbild wird Teil des Bildstapels

### Projektdateien

**Datei → Projekt speichern** legt Quellverweise und reproduzierbare Einstellungen in einer `.ofproj`-Datei ab: die exakte Bildreihenfolge (einschließlich TIFF-Seiten und doppelter Bilder), Ausschlüsse, Drehungen und Spiegelungen, Ladeskalierung, Renderingparameter pro Verfahren, Beschriftungen, Maßstabsbalkenstil und das aktuelle Einzelbild. Ergebnis- und Retuschepixel müssen getrennt exportiert werden. Vorgänge, die sich nicht aus den Quelldateien rekonstruieren lassen, erfordern vorher das Speichern eines neuen Eingabestapels.

**Datei → Projekt öffnen…** baut die aufgezeichneten Bilder auf, bevor der aktuelle Stapel ersetzt wird. Wurden Quellen verschoben, den neuen Quellordner wählen und die vollständige Zuordnung alter zu neuer Pfade bestätigen. Anschließend das Projekt erneut speichern, damit die neuen Verweise erhalten bleiben. Ein Projekt verhält sich wie ein Dokument: Die Titelleiste zeigt den Namen mit einem `*`, solange Änderungen ungespeichert sind; beim Schließen der Anwendung oder Öffnen eines anderen Projekts wird vor dem Verwerfen gefragt. Quellbilder werden referenziert, nicht kopiert.

### Verarbeitungsvorlagen

Eine Verarbeitungsvorlage enthält Einstellungen, niemals Quellpfade. **Datei → Verarbeitungsvorlage speichern…** schreibt eine `.oftemplate` mit dem aktuellen Fusionsverfahren und seinen Parametern, der Registrierungsmethode und ihren Parametern, den Helligkeitsoptionen pro Methode, Thread-Anzahl, Kacheleinstellungen, Registrierungsbreite, StackMFF-AI-Batchgröße, Schärfung, umgekehrter Reihenfolge und dem Schalter für Sensorentrauschung. Ein Defektmaskenpfad wird bewusst nicht gespeichert.

**Datei → Verarbeitungsvorlage anwenden…** wendet diese Einstellungen auf den bereits geladenen Stapel an, ohne die aktuellen Quellen zu ersetzen. Der Auswahldialog akzeptiert auch ein `.ofproj` (die Einstellungen werden ohne die Projektquellen extrahiert) und ein Rezept `.json` mit Projekt-Einstellungen. Das Anwenden markiert das Projekt als geändert und kann Algorithmusparameter und Ressourceneinstellungen verändern; prüfen Sie den rechten Bereich vor dem Rendern. Rezepte bleiben das Headless-Format und enthalten den Defektmaskenpfad samt Fingerabdruck.

### Demo-Stack

Der Willkommensdialog bietet **Demo-Stack laden**: sechs mitgelieferte Beispielfotos aus einem echten Fokus-Stack, damit der komplette Ablauf sofort ohne eigene Bilder ausprobiert werden kann. Auch praktisch, um eine Installation zu prüfen.

### Schärfekurve

Eine kleine Kurve über dem Navigationsregler der Quelle zeigt die Schärfe pro Einzelbild (normiert). Unscharfe Bilder erscheinen als Einbrüche mit rotem Punkt. Ein Klick auf die Kurve springt zu diesem Bild; beim Überfahren mit der Maus wird die Bildnummer eingeblendet.

### Mehrseitige TIFF-Stapel

Mikroskopiesoftware (ZEN, NIS-Elements, ImageJ/Fiji) exportiert einen ganzen Z-Stapel häufig als EINE mehrseitige TIFF-Datei. OpenFocus liest jede Seite einer solchen Datei als eigenes Einzelbild des Stapels – die Datei ablegen oder öffnen, und der Stapel erscheint vollständig (die Einzelbilder heißen `<file>_page_001` und so weiter). Einseitige TIFFs verhalten sich genau wie bisher. Auch der umgekehrte Weg funktioniert: **Datei → Stack speichern → Registrierter Stack / Eingabe-Stack → Als mehrseitige TIFF speichern** schreibt den gesamten Stapel zurück in eine einzige `.tif` und behält dabei die 16-Bit-Tiefe bei.

### Maßstabsbalken (Publikationsabbildungen)

Fehlen für das gewählte Ziel nutzbare Metadaten, wird die manuelle Kalibrierung vorausgewählt und eine gültige Pixelgröße benötigt.

**Werkzeuge → Maßstabsbalken…** konfiguriert einen Maßstabsbalken, der in exportierte Bilder eingebrannt wird. Die Pixelgröße (µm/px) wird automatisch aus den TIFF-Metadaten erkannt – über die `unit=micron`-Konvention von ImageJ, das `PhysicalSizeX` von OME-TIFF oder eine explizite Auflösungseinheit in Zentimetern –, sodass sich Mikroskopie-Stapel (auch mehrseitige TIFFs) selbst kalibrieren. Druckerauflösungen (72/300 dpi und ähnliche) werden bewusst *nicht* als Kalibrierung behandelt, und ein manuell eingegebener Wert hat immer Vorrang vor erkannten Metadaten: ein plausibel aussehender falscher Balken in einer Abbildung ist schlimmer als gar keiner. Ecke und Farbe wählen – **Auto** invertiert gegenüber dem Hintergrund; der Dialog zeigt eine Live-Vorschau. Bei aktiviertem Balken trägt jeder Export (Einzelbild, Ordner-Stapel, mehrseitige TIFF, GIF, Alle exportieren, Stapelverarbeitung) einen Balken im Publikationsstil: eine gerundete 1-2-5-Länge (etwa 12 % der Bildbreite) mit konturierter, gut lesbarer Beschriftung. Die CLI unterstützt ihn über `--scale-bar` (automatische Kalibrierung) und `--scale-um <um/px>`.

Die Vorschau zeichnet in der nativen Pixelgeometrie des Ziels, bevor sie in den sichtbaren Bereich eingepasst wird. Einpassen, 100 %, Verschieben und Bildwechsel ändern nur die Ansicht. **Übernehmen** erzeugt einen rückgängig machbaren Schritt; **Abbrechen** behält die bisherige Kalibrierung. Historische Ausgaben behalten ihre eigene Kalibrierung und ihren Stil, wenn ein anderer Quellstapel geladen wird.

Entfernen lässt sich der Balken über **Werkzeuge → Skalenbalken entfernen**: *Aus dem Quellstapel* und *Aus dem Ergebnis* verengen den Geltungsbereich „Übernehmen auf“ auf die verbleibende Leinwand, während *Überall* den Balken abschaltet. Beide Leinwände und der A/B-Wischvergleich werden sofort neu gezeichnet; das Entfernen aus einer Leinwand, die den Balken nie zeigte, lässt die Einstellung unverändert.

### Tiefenkarte (Fokusposition)

**Werkzeuge → Tiefenkarte…** schätzt Fokuspositionen mit dem Aktivitätsmaß der gewählten Methode. Das ist keine gemessene Oberfläche und keine exakte Pixel-Herkunftskarte: Pyramiden- und Wavelet-Verfahren schätzen Aktivität, die tiefenfähigen StackMFF-Varianten melden eine geschätzte Bildposition. Als Quelle dienen entweder die aktuell markierten Bilder oder die aufgezeichnete Quelle der gewählten Ausgabe. Fehlende historische Herkunftsdaten werden niemals durch den aktuellen Stapel ersetzt; Quelle und Textur bleiben an diesen Stand gebunden. Tiefenkarten sind für transparente PNG-Stapel nicht verfügbar – weder in der GUI noch in der CLI.

Alle sieben tiefenfähigen Methoden – DCT, DTCWT, GFG-FGF, OpenFocus B, OpenFocus C, StackMFF V2 und StackMFF V3 – erfordern einen ausdrücklichen Klick auf **Berechnen**. StackMFF V1 hat keine Tiefenkarte. Öffnen des Dialogs oder Wechsel von Maß, Quelle, Registrierung und Parametern starten keine Berechnung. Der Parametereditor erscheint direkt unter der Maßauswahl: DCT-Median- und GFG-Anfangskernel sind ungerade Werte von 1 bis 51 (Standard 7). OpenFocus B verwendet einen Fokusradius und eine Tiefenglättung; OpenFocus C verwendet **Mischungsglättung** ohne Radius. Die DTCWT-Zerlegungstiefe ist hier als lokaler Entwurf bearbeitbar und beginnt mit den Quellparametern. StackMFF V3 zeigt einen eigenen KI-Ausgabemodus – **Weiche Mischung** (Standard) oder **Harte Auswahl** – und eine Erläuterung statt eines numerischen Kerns: Die Karte meldet eine kontinuierliche Bildposition, keine physikalische Entfernung. Weiche Mischung nutzt die wahrscheinlichkeitsgewichtete Position; harte Auswahl nimmt pro Pixel das wahrscheinlichste Bild. StackMFF V2 hat keine Parameter und meldet die vom Netz abgetastete Fokusebene.

Entwürfe beginnen mit den Parametern der passenden Ausgabe oder den gespeicherten Methodenwerten des Hauptfensters. Sie bleiben während dieses Dialogs pro Quelle und Methode erhalten; Änderungen daran wirken weder auf das Hauptfenster noch auf das Projekt oder eine aufgezeichnete Ausgabe.

Geänderte Berechnungseinstellungen markieren die vorherige 2D-Karte als **nicht aktuell**, leeren die 3D-Oberfläche und sperren jeden Export – auch koloriert, roh, physikalisch und OBJ –, bis eine neue Berechnung gelingt. Eine laufende Berechnung wird kooperativ abgebrochen; nach Abschluss des Abbruchs erneut **Berechnen** drücken. Es wird keine weitere Berechnung in die Warteschlange gestellt. **Berechnen** bleibt sichtbar und ist während Berechnung, Abbruch und Export deaktiviert.

- **Farbtabelle**: Graustufen (Publikationsstandard) sowie 15 wissenschaftliche Paletten (Turbo, Viridis, Magma, …) und eigene – *Neues eigenes Schema…* baut einen Verlauf aus beliebig vielen Farbstützpunkten (Doppelklick auf einen Stützpunkt bearbeitet Position und Farbe); eigene Schemata werden gespeichert und im Dropdown angeboten.
- **Umkehren / Gamma / Glättung**: die Tiefenachse umdrehen, den Tonwert neu ausbalancieren (Gamma < 1 hellt niedrige Indizes auf) und Sprenkel mit einem kantenbewahrenden Durchgang entfernen.
- **Überlagerung**: nur für die zugehörige Ausgabe mit übereinstimmenden Koordinaten verfügbar.
- **Rohdatenexport**: Ein 16-Bit-TIFF speichert einen normalisierten Index, kodiert von 0 bis 65535; es sind keine verlustfreien Fließkommadaten und keine physikalische Entfernung. JPEG und BMP werden abgelehnt.
- **Index-TIFF und alternatives Bild als TIFF**: exportieren die aktuelle Tiefenschätzung als vorzeichenlose 16-Bit-Bildindizes oder als ein aus Quellpixeln zusammengesetztes Bild. Indexwerte 1–N beziehen sich auf die exakte Bildreihenfolge, die als Eingabe der Tiefenberechnung diente, nach Registrierung und etwaiger Umsortierung; 0 markiert ungültige Pixel. Höchstens 65.535 Bilder werden unterstützt. Gebrochene geschätzte Positionen werden auf das nächste Bild gerundet, halbe Werte aufwärts. Diese Indizes schätzen die Fokusposition; sie sind nicht die wahre Pixel-Herkunft eines Fusionsergebnisses. Das alternative Bild wählt an jeder gültigen Stelle das Pixel des entsprechenden Tiefen-Eingabebilds und setzt ungültige Pixel auf null. Ist eine Eingabe 16 Bit tief, werden 8-Bit-Werte durch Multiplikation mit 257 auf 16 Bit angehoben. Exporte verwenden das ROI der Tiefenkarte und erhalten ungültige Bereiche als null.
- **Physikalische Tiefe**: Z-Start, positiver Schritt, Einheit und Aufnahmerichtung eingeben und die ursprüngliche Aufnahmezuordnung bestätigen. Ausgeschlossene Bilder behalten ihre Lücken. Ein Float32-TIFF speichert Mikrometer, NaN für fehlende Daten und Quellmetadaten. Ein physikalisches OBJ benötigt zusätzlich eine XY-Kalibrierung; es verwendet einen physikalischen Z-Maßstab von 1:1.
  Beim Wechsel der Quelle wird die Bestätigung zurückgesetzt; bestätigen Sie die Aufnahmereihenfolge erneut.
- **3D-Modell / 3D-Textur**: die längste Seite des Netzes wählen (standardmäßig 600 Knoten). Jeder neu geöffnete Dialog beginnt mit **Automatisch (Echtfarben)**: Die zugehörige Ausgabe wird nur verwendet, wenn Identität und Koordinaten/ROI zur abgeschlossenen Karte passen; andernfalls dient das erste beteiligte Bild dieser abgeschlossenen Berechnung als Textur, nach Registrierung und etwaiger Bildumsortierung. Der aufgelöste Ausgabe- oder Bildname wird angezeigt. Passende Ausgabe, ein beteiligtes Bild und **Tiefenpseudofarben** sind ausdrücklich wählbar; nicht verfügbare Bilder werden gemeldet, nie durch fremde Ausgaben ersetzt. Diese Textur teilen sich die 3D-Vorschau und das OBJ-Textur-PNG. Die 2D-Karte behält ihre eigene Farbkarte und Überlagerung: Eine Änderung von Palette oder Gamma färbt eine Echtfarbtextur nicht um. Ausdrückliche Pseudofarben folgen Palette, Gamma und Umkehrung. Fertige Vorschau und Export verwenden dasselbe abgetastete Höhenfeld ohne zusätzliche versteckte Glättung. Das relative Relief wird in Bildeinheiten angegeben.
- **Modellexport**: OBJ, MTL, Textur-PNG und JSON-Metadaten bilden eine Dateigruppe und sollten zusammenbleiben. Der Export läuft im Hintergrund. Konflikte werden für jedes Mitglied geprüft, die Vorbereitung wird zwischengespeichert, ein Abbruch erhält vorhandene Dateien, und Fehler beim Abschluss machen Ersetzungen rückgängig. Das Speichern eines Modells exportiert nicht das zugehörige Fusionsbild und hebt dessen Schutz vor ungespeicherten Änderungen nicht auf.

Die Berechnung läuft im Hintergrund mit Fortschritt und Abbruchmöglichkeit; die KI-Methoden nutzen dieselbe speicherschonende Kachelung wie die KI-Fusion, sodass auch tiefe Stapel funktionieren. Die CLI schreibt mit `--depth-map [Pfad]` und `--depth-colormap` nur die kolorierte Karte (Standard `<output>_depth.png`), nur für die oben genannten tiefenfähigen Methoden, und lehnt transparente oder gruppierte Eingaben ab; im Batch-Modus ist nur der Standardpfad erlaubt. Die GUI exportiert zusätzlich Rohdaten, Bildindizes, alternative Bilder, physikalische Tiefe und 3D-Modelle.

### Automatische Absturzwiederherstellung

OpenFocus hält fortlaufend einen Schnappschuss der Arbeitssitzung (Stapel, Einstellungen, Beschriftungen, aktuelles Einzelbild). Wird die Anwendung unerwartet beendet – durch einen Absturz, Stromausfall oder gewaltsames Beenden –, erkennt dies der nächste Start und bietet **Sitzung wiederherstellen** an, um genau zum letzten Stand zurückzukehren. Nach einem sauberen Beenden erscheint die Abfrage nie. Wiederherstellungsdaten liegen neben den Einstellungen (portable Installationen behalten sie neben der ausführbaren Datei, sodass sie mitwandern).

### Zuletzt geöffnete Dateien

Jeder geöffnete Stapel wird unter **Datei → Zuletzt geöffnet** gemerkt (bis zu 8 Einträge, im Untermenü nummeriert). Ein Klick auf einen Eintrag öffnet den Stapel erneut; ein globales Tastenkürzel für zuletzt geöffnete Dateien gibt es nicht. Ein erneut geöffneter Stapel rückt an die Spitze; Einträge, deren Dateien nicht mehr existieren, lassen sich nach einer Rückfrage entfernen, und **Liste „Zuletzt geöffnet“ leeren** räumt das Menü.

### Drag-and-drop

Ein Ordner lässt sich auch direkt auf das Anwendungsfenster ziehen, um Bilder zu importieren. Werden weitere Dateien auf einen vorhandenen Stapel abgelegt, werden sie angehängt (nach einer Größenprüfung).

### EXIF-Ausrichtung

Kamera-JPEG/TIFF-Dateien mit EXIF-Orientation-Tag werden beim Import automatisch gedreht bzw. gespiegelt, sodass Hochformat-Aufnahmen aufrecht erscheinen.

### 16-Bit-Quellen

16-Bit-PNG/TIFF-Eingaben werden automatisch erkannt und durchgehend erhalten: Die Fusion läuft in voller Bittiefe, und Ergebnisse lassen sich wieder als 16-Bit-PNG/TIFF speichern (JPG/BMP-Exporte werden auf 8 Bit reduziert, da diese Formate keine 16 Bit speichern können).

Kamera-RAW-Dateien werden mit 16 Bit entwickelt (Kameraweißabgleich, sRGB-Gamma) und mit deaktivierter Automatikhelligkeit, damit jedes Bild eines Stapels dieselbe Belichtungszuordnung erhält; eine automatische Belichtung pro Bild würde die Helligkeit zwischen den Bildern flackern lassen. Zum Erhalt der 16-Bit-Daten als TIFF oder PNG speichern.

### Farbprofile

Einzelbild-Exporte als PNG, TIFF, JPEG und WebP behalten das RGB-ICC-Profil, wenn jedes Quellbild dasselbe eingebettete Profil besitzt. Gemischte oder unmarkierte Stapel werden ohne ICC-Profil exportiert. OpenFocus bewahrt das Profil auf; es konvertiert die Quellpixel nicht in einen gemeinsamen Farbraum und verwaltet die Vorschau nicht farblich. Konvertieren Sie Bilder mit unterschiedlichen Profilen vor dem Laden in einen gemeinsamen Farbraum. RAW-Eingaben werden nach sRGB entwickelt und übernehmen kein Kameraprofil.

Bei einer mehrseitigen TIFF müssen alle Seiten in der Datei dasselbe gültige RGB-Profil besitzen; andernfalls gelten ihre Bilder als unmarkiert. Videobilder gelten ebenfalls als unmarkiert. Angehängte Bilder und die für ein Rendering ausgewählten Bilder tragen zu dessen Profil bei, das bei späteren Exporten dieses Ergebnisses erhalten bleibt.

### Stapel leeren

Um alle geladenen Bilder zu entfernen, auf **Datei → Stack leeren** gehen oder `Ctrl+W` drücken.

---

## 5. Bilder ansehen und navigieren

### Im Bildstapel navigieren

- **Regler**: den Regler unter dem Quellbereich ziehen, um durch die Einzelbilder zu blättern
- **Dateiliste**: auf einen Dateinamen in der Quellenliste des rechten Bereichs klicken; solange die Liste den Fokus hat, bewegen die Pfeiltasten oben/unten die Auswahl, und der Quellbereich folgt
- **Tastatur**: Es gibt kein globales Tastenkürzel für vorheriges/nächstes Bild; zuerst die Quellenliste fokussieren

### Ähnliche Bilder prüfen

Klicken Sie unter der Kopfzeile der Quellenliste auf **Bilder prüfen**, um benachbarte Bilder zu finden, deren Pixel nahezu identisch sind. Während der Analyse wechselt die Schaltfläche zu **Prüfung abbrechen**. Jede vorgeschlagene Zeile nennt ihr Referenzbild und den Unterschied; prüfen Sie beide, bevor Sie ein Bild über das Auswahlkästchen von der Fusion ausschließen. Vorschläge ändern niemals Auswahlkästchen oder löschen Dateien. Dieser konservative Vergleich kann ähnliche Bilder übersehen; Änderungen, die auf einen kleinen scharfen Bereich beschränkt sind, werden nicht herausgemittelt. Ergebnisse werden verworfen, wenn sich der Stapel ändert. Vorschläge erscheinen nur, wenn sowohl das vorgeschlagene Bild als auch sein Referenzbild einbezogen sind; eine Änderung der Auswahl aktualisiert die angezeigten Vorschläge.

### Zoomen und Verschieben

- **Hinein-/Herauszoomen**: Mausrad über Quelle oder Ergebnis, auch mit gedrückter `Ctrl`-Taste
- **Lupe**: linke Maustaste gedrückt halten; das Mausrad ändert die Vergrößerung, `Ctrl` + Mausrad die Größe des Quadrats. Beide Bereiche merken sich ihre Größe getrennt
- **An Fenster anpassen**: mit der mittleren Maustaste klicken
- **Verschieben**: mit der rechten Maustaste ziehen oder die Leertaste halten und mit links ziehen

Bei großen Quellbildern beginnt die Lupe bei nativen 100 %; die Prozentangabe beschreibt Gerätepixel pro Originalpixel. Die Hauptansicht erreicht 800 %. JPEG- und RAW-Stapel öffnen mit begrenzten Vorschauen, sofern verfügbar. Für genauere Prüfung lädt ein Hintergrundthread die Originalpixel und übergibt nur den sichtbaren Bereich an die Ansicht. Das ausgewählte Original bleibt für weitere Prüfungen erhalten; Wechseln oder Leeren gibt es frei. Die erste Originaldekodierung benötigt weiterhin genügend RAM und kann länger dauern als das Öffnen der Vorschau.

### Synchronisierte Navigation

Wenn im Quellbereich navigiert wird, zeigt der Ergebnisbereich das entsprechende Ergebnisbild an, sofern vorhanden.

---

## 6. Wischvergleich

Die Wischansicht legt zwei Bilder mit einem verschiebbaren vertikalen Trenner übereinander und macht Fehlausrichtungen und Qualitätsunterschiede entlang der Trennlinie direkt sichtbar.

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

Beide Seiten aktualisieren sich live: neue Renderings, Bewegungen des Quellreglers und gelöschte Einzelbilder erscheinen sofort.

### Zoomen und Verschieben

Die Vergleichsansicht verwendet eine gemeinsame Transformation für beide Bilder:

- **Zoom**: Mausrad (zoomt um den Cursor)
- **Verschieben**: überall außer auf dem Trenner klicken und ziehen
- **Zurücksetzen**: Doppelklick

Wischmodus und ROI-Modus teilen sich den Ergebnisbereich, der eine schließt den anderen daher aus. Das Leeren des Stapels beendet ebenfalls den Wischmodus.

---
## 7. Bildregistrierung

Die Bildregistrierung korrigiert räumliche Fehlausrichtungen zwischen den Bildern eines Stapels. Sie ist unverzichtbar, wenn Bilder leichte Verschiebungen oder Perspektivänderungen aufweisen.

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

#### Homographie → ECC

- Führt zuerst eine Homographie und danach eine ECC-Verfeinerung aus
- Für größere Perspektivänderungen, bei denen ECC allein zu instabil wäre

#### OpenFocus (schnell)

- Begrenzte Ähnlichkeitsausrichtung (Verschiebung + Rotation + Skalierung) mit automatischem Zuschnitt
- Sucht begrenzte Ähnlichkeitstransformationen über abgetastete Bilddifferenzen und verzerrt mit Lanczos-Interpolation
- Der Suchbereich für Verschiebungen beträgt standardmäßig 3 % der Bildgröße und lässt sich mit `--align-shift-limit` ändern. Die Suche verwendet das zuvor ausgerichtete Bild und die kumulierte Bewegung als Ausgangspunkt; die Gesamtverschiebung kann 3 % überschreiten.
- Für nicht-starre oder extreme Geometrien stattdessen ECC oder Homographie wählen
- Bei einem tatsächlichen Schätzfehler behält `failure_policy=continue` (Standard) das Bild unausgerichtet; `fail` bricht ab. Wenig Struktur, wiederholte Muster und Fokusänderungen können dennoch ungenaue Zuordnungen verursachen.
- In der CLI dieselbe Richtlinie mit `--align openfocus --align-param 'failure_policy="fail"'` wählen (shellgerechte Anführungszeichen verwenden). Stapel-Elemente verwenden die gewählte Richtlinie unabhängig. Standard ist `"continue"`.
- **Referenzbild** auf eine 1-basierte Nummer in der einbezogenen, verarbeiteten Stapelreihenfolge setzen, um die Geometrie an diesem Bild auszurichten. Die Bildreihenfolge bleibt erhalten; der Helligkeitsabgleich verwendet seine eigene Referenz. Jeder Stapel in der Stapelverarbeitung muss diese Nummer enthalten, sonst schlägt das Element fehl. CLI-Beispiel: `--align openfocus --align-param reference_frame=3`. Eine Referenz kann Parallaxe oder eine schlechte Zuordnung nicht korrigieren; prüfen Sie das Ergebnis.

Ausrichtungsberichte nennen die Referenz und das Ergebnis jedes Bildes. Ein Rückfall bedeutet, dass die Methode die Ausrichtung nicht herstellen konnte oder von einer früheren fehlgeschlagenen Schätzung abhängt; der gemeldete Grund unterscheidet schwache Schätzungen und Bewegung außerhalb der Grenzen, sofern bekannt. Qualitätswerte sind methodenspezifisch und keine Garantie für Genauigkeit. Transformationen bilden Quellkoordinaten auf die Referenzleinwand der Registrierungsstufe vor dem gemeinsamen Zuschnitt ab; zusammengesetzte Methoden melden ihre Stufen getrennt. Homographie und ECC akzeptieren ebenfalls den optionalen Registrierungsparameter `failure_policy="fail"`, um bei erkanntem Fehler zu stoppen. Standard bleibt `"continue"`; prüfen Sie Ergebnisse auch bei erfolgreicher Ausrichtung.

Um ein Rendering zu untersuchen, aktivieren Sie **Einstellungen → Erweiterte Einstellungen… → Registrierung → Ausrichtungsdiagnose erfassen** und rendern erneut. Dadurch wird die Ausrichtung neu berechnet, statt zwischengespeicherte Bilder wiederzuverwenden. Der Abschlussstatus hebt Rückfälle hervor; **Einstellungen → Erweiterte Einstellungen… → Registrierung → Ausrichtungsdiagnose** öffnet den Bericht pro Bild. Er zeigt verfügbare Fehlergründe, methodenspezifische Werte, Übereinstimmungszahlen, referenzierte Bilder und Transformationen. Referenznummern und -namen beziehen sich auf die ursprüngliche Quellenliste, auch wenn Bilder ausgeschlossen sind. Stapel- und gruppierte Renderings liefern diesen Bericht nicht.

### Registrierung aktivieren

1. Im rechten Bereich die Gruppe **Registrierung** suchen
2. Eine Methode im Dropdown wählen: **Keine Ausrichtung**, **OpenFocus (schnell)**, **ECC**, **Homographie** oder **Homographie → ECC**. Ohne gespeicherte Auswahl startet eine neue Sitzung mit ECC; die zuletzt gewählte Methode wird gespeichert.
3. Unter **Einstellungen → Erweiterte Einstellungen… → Registrierung** die **Behandlung abgelehnter Bilder** wählen. Änderungen gelten für die Methoden; für OpenFocus (schnell) lässt sich außerdem das Referenzbild festlegen.
4. **Ausrichtungsdiagnose erfassen** befindet sich jetzt unter **Einstellungen → Erweiterte Einstellungen… → Registrierung** (siehe oben).

### Wann Registrierung sinnvoll ist

- Die Bilder sind sichtbar fehlausgerichtet
- Die Aufnahmen wurden aus der Hand gemacht
- Die Szene hat Parallaxe (nähere Objekte verschieben sich zwischen den Bildern)
- Ergebnisse zeigen Geisterbilder oder Doppelkonturen

### Registrierungseinstellungen

Weitere Einstellungen über **Einstellungen → Erweiterte Einstellungen… → Registrierung**:

- **Verkleinerungsbreite**: Auflösung der Vorverarbeitung (Standard: 1024 px)
- Kleinere Werte = schnellere Verarbeitung, eventuell geringere Genauigkeit
- Größere Werte = langsamere Verarbeitung, eventuell höhere Genauigkeit
- **Ausrichtungsdiagnose erfassen** und **Registrierungs-Cache** befinden sich im selben Abschnitt

---

## 8. Fusionsmethoden

OpenFocus bietet zehn Fusionsalgorithmen plus einen reinen Ausrichtungsmodus. Ein frisches Profil startet mit **StackMFF V3** und **ECC**; die folgenden Abschnitte beschreiben jede Option. Jeder Algorithmus hat eigene Eigenschaften für unterschiedliche Bildtypen.

### Guided Filter

- **Algorithmus**: kantenbewahrende Fusion über Guided Filtering
- **Am besten für**: allgemeine Zwecke, ausgewogener Kompromiss aus Geschwindigkeit und Qualität
- **Vorteile**: gute Kantenerhaltung, schnelle Verarbeitung, ausschließlich CPU
- **Parameter**: Kernelgröße (einstellbar von 1–51, nur ungerade Werte)
  - Kleinerer Kernel: mehr lokales Detail, potenziell mehr Rauschen
  - Größerer Kernel: glattere Ergebnisse, bessere Rauschunterdrückung
- **Hinweis**: Guided Filter ist das klassische Standardverfahren, aber nicht die Voreinstellung. Ohne gespeicherte Auswahl startet eine neue Sitzung mit StackMFF V3.

### DCT (Diskrete Kosinustransformation)

- **Algorithmus**: Fusion im Frequenzbereich über DCT-Koeffizienten
- **Am besten für**: texturreiche Bilder, wissenschaftliche Bildgebung
- **Vorteile**: erhält feine Texturen, mathematisch fundiert
- **Parameter**: Die Kernelgröße bestimmt das Verarbeitungsfenster

### DTCWT (Dual-Tree-Complex-Wavelet-Transformation)

- **Algorithmus**: Fusion im mehrskaligen Wavelet-Bereich
- **Am besten für**: komplexe Szenen mit mehreren Detailstufen
- **Vorteile**: hervorragende mehrskalige Analyse, gute Richtungsauflösung
- **Fusionsregel**: eine Entscheidung pro Pixel und Stufe für alle Orientierungen und Farbkanäle (N-fach, mit Mehrheitskonsistenzprüfung); das Tiefpassband folgt der gröbsten Entscheidung, statt alle Bilder zu mitteln
- **Parameter**: Wavelet-Ebenen (1–6, Standard 4) im rechten Bereich. Die Methode hat keinen räumlichen Kernel, daher ist der Kernelregler ausgeblendet.

### GFG-FGF (Gaußartiger Vier-Nachbarschafts-Gradient + schneller Guided Filter)

- **Algorithmus**: Implementierung nach Fu et al. (*Laser & Optoelectronics Progress*, 2024): Eine Unschärfedifferenz-Fokuskarte |I − mean(I)| wird mit dem gaußartigen Vier-Nachbarschafts-Gradienten (GFG) angereichert, durch einen schnellen Guided Filter verfeinert, pro Pixel entschieden (Gleichstand über ein 3×3-Sobel-Kleinbereichsmaß aufgelöst) und die binären Entscheidungskarten werden vor der gewichteten Mischung erneut mit dem schnellen Guided Filter geglättet.
- **Am besten für**: Mikroskopie und Bilder mit klaren Fokusbereichen
- **Vorteile**: schnell, scharfe Grenzen der Fokusbereiche; der Speicherbedarf wächst nicht mit der Bildanzahl
- **Parameter**: Kernel = Größe des Mittelwertfilters der groben Fokuskarte

### StackMFF (V1, Deep Learning)

- **Algorithmus**: Ein einschaliges Attention-Netz mischt die Luma der Quellbilder mit Softmax-Gewichten; Cb/Cr stammen aus dem Quellbild, dessen Luma der fusionierten Luma am nächsten liegt
- **Am besten für**: neuronale Fusion eines vollständigen Fokusstapels
- **Vorteile**: keine Parameter, Farben bleiben im Gamut; die fusionierte Luma ist unabhängig von der Bildreihenfolge, die Chrominanz kann bei gleichem Luma-Abstand abweichen
- **Hinweis**: etwa 156.172 Parameter; große Stapel profitieren von einer GPU

### StackMFF V2 (Deep Learning)

- **Algorithmus**: Ein einschaliges Attention-Netz normalisiert die erwartete Fokustiefe und tastet das nächste Quellbild ab. Tiefenindizes hängen von der Stapelreihenfolge ab. **Die Bilder müssen in ihrer ursprünglichen Fokusreihenfolge bleiben** – Umkehren ist deaktiviert
- **Tiefenkarten**: unterstützt; meldet die abgetastete Fokusebene
- **Am besten für**: kontinuierliche Fokusserien mit sehr kleinem, schnellem Modell
- **Voraussetzungen**: ONNX Runtime und das mitgelieferte ONNX-Modell.

### StackMFF V3 (Deep Learning)

- **Algorithmus**: pixelweise Quellklassifikation (ONNX Runtime). Das Netzwerk sagt für jedes Pixel voraus, aus welchem Eingabebild es stammt, und mischt dann die Bilder entsprechend diesen Wahrscheinlichkeiten oder wählt ein Bild aus
- **KI-Ausgabe** (rechter Bereich, nur für diese Methode):
  - **Weiche Mischung (gewichtet)** – Standard; jedes Pixel ist eine wahrscheinlichkeitsgewichtete Mischung der Bilder. Weiche Übergänge, keine harten Nähte; entspricht der im Paper beschriebenen Konfiguration
  - **Harte Auswahl (einzelnes Bild)** – jedes Pixel wird aus dem wahrscheinlichsten Bild kopiert. Schärfere Auswahl ohne Interpolation zwischen den Bildern; wo zwei Bilder gleichauf liegen, kann eine Naht auf Pixelebene entstehen
- **Am besten für**: höchste Qualität, wenn eine GPU verfügbar ist
- **Vorteile**: reihenfolgeunabhängig (die Bildreihenfolge ändert das Ergebnis nie), Stapellängen innerhalb der verfügbaren Ressourcen, keine methodenspezifische Abstimmung, automatische Optimierung
- **Voraussetzungen**: ONNX Runtime und das mitgelieferte ONNX-Modell.

Jede neuronale Kachel verwendet alle Bilder. Quelldaten und Ausrichtungsdaten können auf der Festplatte liegen; freier Speicherplatz und Arbeitsspeicher begrenzen die Stapelgröße. Räumliche V2-Kachelung nähert die Berechnung des Gesamtbilds an.

Ist kein temporäres Verzeichnis angegeben, schätzt OpenFocus den Bedarf anhand bekannter Eingaben und temporärer Daten und berücksichtigt bei Platzmangel beschreibbare feste lokale Laufwerke. Ist das gewählte Verzeichnis nicht verfügbar, wird eine Warnung angezeigt und vorübergehend das System-Temporärverzeichnis verwendet; neue Aufgaben versuchen die gespeicherte Wahl erneut. Bei Platzmangel auf dem gewählten Laufwerk wird nicht automatisch gewechselt. Die Platzprüfung ist eine Momentaufnahme und keine Reservierung. Wählen Sie bei einem Speicherfehler in den erweiterten Einstellungen ein beschreibbares Verzeichnis.

- **Standard**: StackMFF V3.

### OpenFocus A / B / C

A nutzt eine mehrskalige Fokusantwort (Radius 8, Glättung 4); seine Glättung ist derzeit inaktiv.

B verwendet eine Tiefenkarte (Radius 8, Glättung 4, Feather 3). C verwendet eine gewichtete mehrstufige Pyramide (Glättung 4) und hat keinen Radius.

CLI: `--method openfocus_b --param radius=8 --param smoothing=4 --param feather=3`.

In der Oberfläche ist der Belichtungsabgleich für A/B/C standardmäßig an und nutzt das erste verarbeitete Bild; für andere Methoden ist er standardmäßig aus und nutzt das letzte. Explizit gespeicherte Einstellungen bleiben erhalten; bei aktiviertem A/B/C gilt das erste Bild. Die Korrektur der Helligkeitsdrift bleibt standardmäßig aus. Zurücksetzen stellt die Vorgaben jeder Methode wieder her. In der CLI ist der Belichtungsabgleich für alle Methoden standardmäßig an; `--no-exposure-match` schaltet ihn aus.

Bei Auswahl von A/B/C wird die OpenFocus-Ausrichtung verwendet, sofern keine Ausrichtung ausdrücklich gewählt oder ein Projekt wiederhergestellt wurde. Ihre Vorgaben sind Verschiebung 0,03 und Vergrößerung 5 %.

**Fusionsergebnis schärfen** wendet nach der Fusion eine milde Unscharfmaskierung an, bevor das Ergebnis in seine Ausgabebittiefe überführt wird. Sie ist standardmäßig aus und kann in den Haupt-Fusionsoptionen oder den Stapeloptionen aktiviert werden; die entsprechende CLI-Option ist `--sharpen-output`. Sie betrifft das fusionierte Bild, nicht registrierte Bilder oder Tiefenkarten. Kanten können deutlicher werden, Rauschen und Halos aber ebenfalls zunehmen. Die Einstellung wird im Projekt gespeichert und zwischen Sitzungen gemerkt.

### Nur Ausrichtung (keine Fusion)

**Nur Ausrichtung (keine Fusion)** wählen, um den Stapel zu registrieren und einen Stapel-Eintrag zur Ausgabeliste hinzuzufügen. Die Bilder bleiben im Speicher, bis Sie sie exportieren; der reine Ausrichtungsmodus legt keinen zusätzlichen Cache an und schreibt nichts neben die Quelldateien. Speichern Sie den Eintrag, um einen Ordner im gewählten Ziel zu erzeugen. Die ursprünglichen Dateinamen bleiben erhalten, mit der Endung des gewählten Formats.

### Fusionsmethode wählen

1. Im rechten Bereich die Gruppe **Fusion** suchen
2. Eine Methode im **Dropdown** wählen (genau eine ist immer aktiv; der erste Eintrag registriert ohne Fusion)
3. Bei Bedarf die Kernelgröße anpassen – die Beschriftung zeigt, was der Kernel für die gewählte Methode bedeutet
4. **▶ Rendern starten** anklicken, um zu verarbeiten

### Algorithmus-Vergleich

| Algorithmus | Geschwindigkeit | Qualität | Am besten für |
|-----------|-------|---------|----------|
| Guided Filter | Schnell | Gut | Allgemeine Nutzung |
| DCT | Mittel | Gut | Texturen |
| DTCWT | Mittel | Sehr gut | Komplexe Szenen |
| GFG-FGF | Schnell | Gut | Fokusbereiche |
| StackMFF (V1) | Langsam (GPU) | Hier nicht bewertet | Luma unabhängig von der Reihenfolge; Chrominanz kann bei gleichem Luma-Abstand abweichen |
| StackMFF V2 | Schnell (GPU) | Hier nicht bewertet | Kontinuierliche Fokusserien mit einem winzigen Modell |
| StackMFF V3 | Langsam (GPU) | Hervorragend | Beste Qualität |
| OpenFocus A | Abhängig vom Stapel | Abhängig vom Stapel | Mehrskalige Fokusantwort und Mischung detailarmer Bereiche |
| OpenFocus B | Abhängig vom Stapel | Abhängig vom Stapel | Tiefenkartengesteuerte Mischung benachbarter Bilder |
| OpenFocus C | — | — | Gewichtete mehrstufige Pyramide |

**Werkzeuge → Methoden vergleichen…** öffnet einen Dialog, in dem mehrere Verfahren unabhängig eingestellt werden. Name, Parameterbeschriftungen und Bedienelemente jeder Methode bleiben zusammen, wenn das Fenster schmaler wird. Änderungen aktualisieren die Vorschau automatisch. **ROI-Vorschau (schnell)** fusioniert den ausgewählten Bereich mit umgebendem Kontext; **Ganzbild-Fusion** prüft das Ergebnis des gesamten Bildes und erlaubt die Untersuchung eines anderen Bereichs ohne erneutes Rendern. ROI-Ergebnisse können von Ganzbild-Ergebnissen abweichen, weil Belichtungsstatistiken und einige Algorithmen von der Bildausdehnung abhängen.
Die **Ganzbild-Fusion** ist standardmäßig ausgewählt.

Ein sichtbarer Hinweis erklärt die Ergebnisauswahl: Ein Ergebnis anklicken oder es mit `Tab` fokussieren und mit `Enter` oder der Leertaste auswählen.

Über den rechten Bereichen wählt **Vorschaugröße (px)** eine quadratische Größe von 160 bis 960 Pixeln, per Regler oder Zahleneingabe. Alle Methoden verwenden dieselbe Größe und umbrechen passend zur verfügbaren Breite. Die Größe wird für das nächste Öffnen gemerkt. Eine Änderung behält die aktuelle Vergrößerung bei und passt das Prüfquadrat um seinen Mittelpunkt an, begrenzt auf die Bildränder.

**F11** oder die Schaltfläche **Vollbild (F11)** nutzen den gesamten Bildschirm; der Schaltflächentext wechselt zu **Vollbild beenden (F11)**. Die Titelleiste kann das Fenster ebenfalls maximieren. **Esc** verlässt zuerst den Vollbildmodus und schließt den Dialog erst beim zweiten Druck. Die Rückkehr aus dem Vollbild stellt die vorherige Fenstergröße wieder her, und das Ergebnisraster fließt in beiden Fällen auf die verfügbare Breite um.

Das Quellbild mit der linken Maustaste ziehen, um das weiße Prüfquadrat zu verschieben; beim Loslassen bleibt es stehen. Die rechten Bereiche zeigen genau diesen Ausschnitt aus demselben ausgerichteten Stapel. Mit dem Mausrad die Vergrößerung des Bereichs ändern, mit der rechten Maustaste die Quelle verschieben, mit **Strg+Mausrad** die Quelle zoomen und mit **F** einpassen. Ein Wechsel des Quellbilds oder eine Größenänderung des Fensters erhält die Auswahl. **Abbrechen** stoppt weitere Arbeit; das Schließen wartet, bis der aktuelle native Vorgang sicher zurückkehrt.

### Schnellvorschau (Entwurfsrendering)

Das Abstimmen von Parametern auf großen Stapeln ist in voller Auflösung langsam. **Rendereinstellungen** öffnen und **Schnellvorschau** aktivieren: OpenFocus rendert dann zuerst einen verkleinerten Entwurf (halbe Kantenlänge, mindestens 320 px auf der längsten Seite) – typischerweise mehrere Male schneller, einschließlich Registrierung. Die Abschlussmeldung (und der Tooltip des neuesten Ergebnisses) meldet **Qualität: Vorschau**; der Entwurf wird **nicht** zum Ausgabeverlauf hinzugefügt. Häkchen entfernen und erneut rendern, um das Ergebnis in voller Auflösung zu erhalten. Der Vorschaumodus gilt nicht für ROI-Renderings und verunreinigt nie den Ausrichtungs-Cache in voller Auflösung.

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

Transformationen wirken auf den gesamten Bildstapel und erhalten die Ausrichtung zwischen den Einzelbildern.

---

## 10. Retusche, Beschriftungen und Anmerkungen

### Ergebnis retuschieren (Bearbeiten → Ergebnis retuschieren…, `Ctrl+T`)

Die Fusion kann örtliche Fehler hinterlassen – einen Halo an einer kontrastreichen Kante, ein Härchen aus dem falschen Bild, Rauschen in einer gleichmäßigen Fläche. Der Retusche-Arbeitsbereich kopiert Pixel aus genau den Bildern, aus denen das gewählte Ergebnis erzeugt wurde:

1. Ein Ergebnis in der Ausgabeliste auswählen (Rechtsklick → *Ergebnis retuschieren…* funktioniert ebenfalls) und `Ctrl+T` drücken. Ohne Auswahl listet ein Auswahldialog jedes Ergebnis auf und markiert diejenigen mit aufgezeichneten Quellbildern.
2. Das Quellbild steht links, das Ergebnis rechts. Beide teilen sich Zoom und Position: Mausrad zoomt, Rechts-/Mittelklick-Ziehen (oder Leertaste+Ziehen) verschiebt, `F` passt ein, `1` zeigt 100 %.
3. Ein Vorschaubild wählen, den Bildregler ziehen oder `,` / `.` drücken. Ein Strich ist an dieses Bild und die aktuellen Pinseleinstellungen gebunden; das Bild kann erst nach dem Ende des Strichs gewechselt werden.
4. **Quellpinsel** (`B`) oder **Wiederherstellungspinsel** (`E`) wählen. Durchmesser, Härte und Deckkraft haben Regler und Zahlenfelder; `[` / `]` ändert den Durchmesser. `Ctrl` + Mausrad nach oben vergrößert, nach unten verkleinert den Pinseldurchmesser; das Mausrad ohne `Ctrl` zoomt weiterhin. Der Wiederherstellungspinsel setzt Pixel auf das Bild zurück, das Sie geöffnet haben, nicht auf eine ältere Ausgabe.
5. **Tab** gedrückt halten, um das Ergebnis vor dieser Sitzung zu sehen, und **Alt**, um einen Blick auf das Quellbild zu werfen. Die Schärfereferenz ist optional, deckt nur die Quellleinwand ab und verwendet einen gemeinsamen Maßstab für alle Bilder.
6. `Ctrl+Z` / `Ctrl+Y` machen ganze Striche rückgängig bzw. wiederholen sie. **Alles wiederherstellen…** ist ein rückgängig machbarer Schritt. **Als neue Ausgabe übernehmen** fügt einen Eintrag namens `<Name> · Retuschiert N` hinzu und wählt ihn aus; das Original bleibt unverändert. Der neue Eintrag ist als *Nicht exportiert* markiert, bis Sie ihn im Hauptfenster exportieren, und OpenFocus schützt ihn vor automatischer Verdrängung aus dem Verlauf und vor dem Beenden ohne Entscheidung.

Die Retusche als neue Ausgabe übernehmen, diese auswählen und **Speichern unter** öffnen. In der Formatliste TIFF mit Ebenen wählen, um das Ergebnis zusammen mit seinen Quellbildern zu speichern. Live-Beschriftungen und Maßstabsbalken sind nicht enthalten; ungespeicherte Anmerkungsänderungen bleiben geschützt. Quellgeometrie und Gültigkeitsmasken bleiben erhalten. Ergebnis und Quellen müssen undurchsichtige RGB- oder Graustufenbilder mit übereinstimmender 8- oder 16-Bit-Tiefe sein, und jedes Quellrechteck muss in das Ergebnis passen. Der Export unterstützt bis zu 128 Quellen; Quellnamen sind auf 255 Zeichen begrenzt und dürfen weder NUL noch Zeichen außerhalb der Basic Multilingual Plane enthalten, etwa viele Emojis. Unsichere Kontexte und Aufträge, die voraussichtlich mehr als 512 MiB Arbeitsspeicher oder das verfügbare RAM-Budget überschreiten, werden abgelehnt und der Fehler gemeldet, statt einen unvollständigen Export zu schreiben. GIMP 3.2.6 unter Windows importiert die 8- und 16-Bit-Ebenen einschließlich ihrer Versätze und der transparenten Registrierungsfüllung. Das Ergebnis ist sichtbar, die Quell-Ebenen sind ausgeblendet; GIMP fügt außerdem eine sichtbare Hintergrund-Ebene hinzu. Andere Editoren wurden nicht geprüft.

Für eine lokale Tonwertkorrektur **Belichtungspinsel** wählen, den Betrag in Hundertstel Blendenstufen einstellen (100 verdoppelt die Pixelwerte, −100 halbiert sie) und auf das Ergebnis malen. Die Vorschau aktualisiert sich während des Malens. Jeder Strich passt das aktuelle Ergebnis einmal an, auch wo sich seine Tupfer überlappen; ein weiterer Strich baut darauf auf. Pinseldeckkraft und -härte steuern die Mischung. Lichter können beim Aufhellen beschneiden. **Vorher** gedrückt halten, um mit dem Ergebnis beim Öffnen zu vergleichen, während eines Strichs **Esc** drücken, um ihn abzubrechen, oder nach dem Strich rückgängig machen und wiederholen. Das sind Pixelbearbeitungen, keine anpassbaren Ebenen. Als neue Ausgabe übernehmen und den neuen Eintrag exportieren, um die bearbeiteten Pixel zu speichern; ein Projekt speichert diese Pixel nicht.

**Ergebnis anpassen…** ändert das gesamte aktuelle Retuscheergebnis. Die abgetastete Vorschau aktualisiert sich während der Änderung von Belichtung und Kontrast; **Vorher** anklicken, um mit dem Ergebnis vor dem Öffnen der Anpassung zu vergleichen. Beide Regler verwenden Hundertstel Blendenstufen: Die Belichtung multipliziert die Pixelwerte, anschließend skaliert der Kontrast deren Abstand von Mittelgrau. Extreme Werte können Lichter oder Schatten beschneiden. **OK** übernimmt die Anpassung als einen rückgängig machbaren Schritt; **Abbrechen** lässt das Ergebnis unverändert. Während der Verarbeitung fordert **Abbrechen** ein Zurückrollen an. Das vollständige Bild muss für diesen rückgängig machbaren Vorgang in das Speicherlimit des Retuscheverlaufs passen. Das Ergebnis wie oben beschrieben übernehmen und exportieren.

Die Herkunft wird mit der Ausgabe gespeichert: ausgerichtete Bilder, ROI-Zuschnitt-/Einfügeversätze, DCT-Kantenbeschneidung und Bittiefe werden beim Rendern des Ergebnisses aufgezeichnet. Stapeldateien, ältere Ergebnisse und reine Ausrichtungsordner haben keine aufgezeichnete Quelle und werden daher als nicht verfügbar aufgeführt, statt aus dem aktuellen Stapel geraten zu werden. Nicht angewendete Striche und der Rückgängig-Verlauf gelten nur für die Sitzung; das Speichern eines Projekts legt sie nicht ab.

Für eine vorhandene Bilddatei **Bearbeiten → Bilddateien retuschieren…** wählen und dann das Ergebnis sowie ein oder mehrere Quellbilder auswählen. Die Quellen müssen bereits auf dasselbe Pixelraster ausgerichtet sein; gleiche Abmessungen allein belegen keine Ausrichtung. Einseitige PNG-, TIFF-, JPEG- und BMP-Bilder mit übereinstimmenden aufrechten Abmessungen, Kanälen und 8- oder 16-Bit-Tiefe werden akzeptiert; die EXIF-Ausrichtung wird beim Laden angewendet. Alpha- und mehrseitige Bilder werden nicht unterstützt. Die ausgewählten Bilder sind vom Benutzer bereitgestellt und keine aufgezeichnete Rendering-Herkunft. **Übernehmen** erzeugt eine separate, nicht exportierte Ausgabe; speichern Sie sie aus der Ausgabeliste. Die gewählten Dateien sind während dieser Sitzung vor Überschreiben geschützt, einschließlich Tiefenkarten- und 3D-Modell-Exporten. Wie bei anderen Ergebnissen speichern Projektdateien diese Pixel oder den Pinselverlauf nicht.

### Hintergrund ersetzen (Automatischer Hintergrund)

Beim Stacken wird der Hintergrund zusammen mit dem Motiv geschärft. Der
Hintergrund eines gestackten Bildes ist dann oft ein Flickwerk verschiedener
Unschärfen, mit einem Halo um das Motiv. **Automatischer Hintergrund…** im
Retusche-Arbeitsbereich ersetzt ihn durch ein echtes Einzelbild aus demselben
Stack. Das Motiv bleibt genau wie gestackt. Der Hintergrund wird aus einem
Einzelbild und dem gestackten Ergebnis abgeleitet; bei der Verarbeitung können
seine Pixel rekonstruiert, angepasst und überblendet werden.

1. Öffnen Sie das Ergebnis im Retusche-Arbeitsbereich und klicken Sie auf
   **Automatischer Hintergrund…**.
2. Wählen Sie den Hintergrund:
   - **Automatisch** (empfohlen) nimmt den weichen Hintergrund, außer der Stack
     hat einen Großteil des Hintergrunds scharf aufgenommen. Dann nimmt es den
     scharfen.
   - **Weicher Hintergrund** nimmt das Bild mit dem unschärfsten Hintergrund.
   - **Scharfer Hintergrund** nimmt das Bild mit dem fernsten Fokus.
3. Lassen Sie **Kantenhalos entfernen** eingeschaltet, um Halos entlang der
   Motivkanten zu entfernen. Es geht nie über das hinaus, was die Einzelbilder
   aufgenommen haben. Ausschalten behält den Kantenlook des Stacks.
4. Aktivieren Sie **Weiche, mit dem Motiv verbundene Teile behalten**, wenn
   Teile des Motivs knapp außerhalb der Schärfentiefe liegen, etwa das Köpfchen
   am Ende eines Stachels. Dabei können auch Hintergrundobjekte neben dem Motiv
   wie gestackt bleiben, prüfen Sie also das Ergebnis.
5. Klicken Sie auf **Hintergrund ersetzen**. Ein Stack aus 15 Bildern mit
   20 Megapixeln braucht etwa eine Minute und etwa 3,5 GB freien Speicher.
   **Abbrechen** stoppt den Vorgang.

Die Änderung ist ein rückgängig machbarer Schritt, und bereits gemalte Striche
auf dem Motiv bleiben erhalten. Mit dem **Wiederherstellungspinsel** holen Sie
gestackte Pixel überall dort zurück, wo Sie sie bevorzugen, zum Beispiel einen
unscharfen Grashalm vor dem Motiv. Füllt das Motiv fast das ganze Bild, wie bei
einer Waldbodenszene, bleibt das Bild wie gestackt und ein Hinweis erscheint.

Wenn Sie die Retusche auf einem bereits retuschierten Ergebnis öffnen,
analysiert Automatischer Hintergrund das gestackte Ergebnis, aus dem es stammt,
und legt den neuen Hintergrund auf Ihr retuschiertes Bild. Pinselarbeit am Motiv
bleibt erhalten, und ein früher ersetzter Hintergrund wird nie erneut
analysiert. Ein zweiter Durchlauf baut daher keine Säume auf.

Nach einem Durchlauf färbt **Fehlende Daten anzeigen** die Stellen ein, die kein
Bild sauber erfasst hat. Diese Stellen sollten Sie prüfen oder von Hand
retuschieren:

- Magenta: Hintergrund, den kein Bild ohne das Motiv davor aufgenommen hat
- Cyan: für die Kamera verdecktes Motiv
- Gelb: Motiv außerhalb des Fokusbereichs des Stacks
- Schiefer: zu glatt zum Beurteilen

**Karte fehlender Daten speichern…** schreibt die eingefärbte Vorschau und je
Farbe eine Schwarz-Weiß-Maske. Der automatische Hintergrund braucht die Bilder,
aus denen das Ergebnis entstand, und ist für Bereichsergebnisse (ROI) nicht
verfügbar. Neue Hintergrund-Checkpoints stellen Bild, erreichbaren Verlauf und
Diagnosedaten exakt wieder her, ohne den Hintergrund neu zu berechnen.

### Beschriftungen verwalten

**Werkzeuge → Beschriftungen** nach dem Laden von Bildern oder dem Erzeugen eines Ergebnisses öffnen. Jede Beschriftung hat ihr eigenes Ziel: den Quellstapel, den registrierten Stapel oder Ausgabeergebnisse. Ausgabebeschriftungen können für alle Ergebnisse oder für das ausgewählte Ergebnis nach seinem Namen gelten; Umbenennen oder Umsortieren dieses Ergebnisses ändert die Zuordnung nicht.

Mit **Hinzufügen** eine weitere Beschriftung anlegen, einen Eintrag zum Bearbeiten auswählen und über sein Kästchen aktivieren oder deaktivieren. Festen Text eingeben oder `{value}` für eine Bildnummer einfügen. Die Nummer ist der Startwert plus dem nullbasierten Bildindex mal dem Intervall; eine einzelne Ausgabe verwendet immer den Startwert.

Alle Bilder, das aktuelle Bild oder einen benutzerdefinierten Bereich wie `3-5,8` wählen. Ungültige oder nicht verfügbare Bildnummern verhindern das Anwenden des Entwurfs. Aktuelle und benutzerdefinierte Bereiche folgen den ausgewählten Quellidentitäten, wenn der Stapel umsortiert wird; Beschriftungen für alle Bilder gelten auch für angehängte Bilder. Fehlende Ziele bleiben mit ihren ursprünglichen Identitäten verbunden, bis Sie ausdrücklich ein neues Ziel wählen.

Die Vorschau verwendet das tatsächliche Ziel. Mit **Zurück/Weiter** navigieren oder die Bildnummer eingeben, **Einpassen** oder **100 %** wählen und das Bild bei 100 % durch Ziehen verschieben. Position und Schriftgröße verwenden Bildpixel. Eine Ersatzschrift oder ein fehlgeschlagenes Zeichnen wird gemeldet. Beschriftungen behalten beim Export als PNG oder TIFF die ursprüngliche Bittiefe.

**Anwenden und schließen** übernimmt den vollständigen Entwurf als einen rückgängig machbaren Schritt. Abbrechen, Esc und Schließen verwerfen ihn. Die Auswahl einer anderen Beschriftung behält Ihre Entwurfsänderungen. **Löschen** entfernt nur den ausgewählten Eintrag, auch den letzten. Die Entfernungsaktionen im Menü leeren Quellbeschriftungen, Beschriftungen des registrierten Stapels, Ausgabebeschriftungen oder alle Beschriftungen unabhängig voneinander und lassen sich ebenfalls rückgängig machen. Angewendete Beschriftungen ändern die Anzeige der Sitzung; exportieren Sie ein Bild, um die Anmerkungen in eine Datei zu schreiben.

---

## 11. Einstellungen und Konfiguration

**Einstellungen → Erweiterte Einstellungen…** öffnet Leistungs-, Registrierungs-, Sensorbereinigungs-, Ausgabebenennungs-, KI- und Starteinstellungen. Änderungen bleiben Entwürfe, bis Sie **OK** drücken; Abbrechen, Esc und Schließen verwerfen sie. Eine Verringerung der Kachelgröße aktualisiert sofort das sichtbare Überlappungslimit auf eine halbe Kachel. Die gespeicherte KI-Batch-Obergrenze bleibt erhalten; die Inferenz verwendet derzeit eine Kachel pro Aufruf. Die Registrierungsbreite kann die Genauigkeit der Ausrichtung verändern. Die Erfassung der Ausrichtungsdiagnose ist ein Kästchen im Abschnitt Registrierung.

Automatische Update-Downloads laufen im Leerlauf und installieren nie von selbst. Ein abgeschlossener Download bleibt bereit, bis Sie ausdrücklich **Aktualisieren und neu starten** wählen. Der Neustart schützt aktive Bearbeitungen und ungespeicherte Arbeit und schreibt anschließend Wiederherstellungsdaten, bevor die Ersatzversion startet. Das Abbrechen eines Downloads oder einer Vorbereitung wartet auf die aktuelle sichere Grenze. Ein vorhandener vollständiger, neuerer bereitgestellter Build kann offline installiert werden.

Die Flat-Field-Korrektur ändert die Arbeitssitzungspixel, ohne Quelldateien zu überschreiben. Über **Werkzeuge → Staubkarte (Flat-Field)** das Weißbild und ein Vorschaubild wählen und die Vorschau nach einer Änderung von Quelle oder Skalierung erneut erzeugen. Referenz- und Arbeits-/Basisbilder müssen denselben Koordinatenbereich und dieselbe Größe haben. Das Anwenden bereitet beide Stapel vor dem Übernehmen vor; das optionale Entfernen des Referenzbilds gehört zum selben Rückgängig-Schritt. Korrigierte Quellbilder exportieren, wenn diese Pixel außerhalb der Sitzung erhalten bleiben sollen.

Die Stapelverarbeitung akzeptiert optional ein Weißfeldbild in ihren Verarbeitungsoptionen. Ein defokussiertes Referenzbild in der ursprünglichen Bildauflösung wählen; es wird mit dem Eingabestapel skaliert und vor der Registrierung mit der Standardstaubgröße angewendet. Ein unlesbares Referenzbild oder eine Größenabweichung lässt den betroffenen Stapel sichtbar fehlschlagen. Das Referenzbild wird aus Ordner-Eingaben ausgeschlossen, auch bei gruppierter Verarbeitung; mindestens zwei Quellbilder müssen für die Fusion übrig bleiben. Flat-Field-Korrektur lässt sich nicht mit transparenten Eingaben kombinieren. Leer lassen, um ohne Korrektur zu verarbeiten. Die CLI akzeptiert die Referenz mit `--dust-map WHITE`.

Die Sensorbereinigung ist optional und läuft auf den Originalbildern in Sensorkoordinaten vor der Registrierung. In der Desktop-App finden Sie sie unter **Einstellungen → Erweiterte Einstellungen… → Sensorbereinigung**; der Stapeldialog bietet dieselben Steuerelemente. Eine ausdrückliche Defektmaske korrigiert bekannte Hot- oder Dead-Pixel; OpenFocus leitet keine Maske aus Probenbildern ab. Die Maske muss ein einkanaliges 8- oder 16-Bit-Bild mit denselben Abmessungen wie die Quelle sein: null bedeutet unberührt, und entweder 1 oder der maximale Kanalwert markiert Defekte. Prüfen Sie ihre Ausdehnung vor der Verarbeitung. Masken, die mehr als 10 % des Bildes abdecken, oder Defekte ohne unmaskierten Nachbarn werden abgelehnt. Die Maske in der Originalauflösung belassen und nicht als Quellbild auswählen. Für die GUI-Verarbeitung Quellbilder bei Verwendung einer Defektmaske in voller Skalierung laden; die Maske wird nie skaliert. Die Sensorbereinigung (einschließlich Entrauschung) kann nach einer Drehung, Spiegelung, Flat-Field-Korrektur oder anderen Quelländerung nicht auf einen GUI-Stapel angewendet werden: Originalbilder neu laden oder Bereinigung deaktivieren. Das gilt auch für wiederhergestellte Projekte mit solchen Quellverläufen. Vorab ausgerichtete ROI-Verarbeitung lässt sich nicht mit Sensorbereinigung kombinieren; ROI-Modus verlassen und den vollständigen Stapel rendern. Bereinigungseinstellungen gehören zur aktuellen Sitzung und zum gespeicherten Projekt, nicht zu globalen Anwendungseinstellungen. Stapelverarbeitung und CLI verarbeiten ihre Originaleingaben getrennt, mit Sensorbereinigung vor Flat-Field-Korrektur und Registrierung. Defektmasken erfordern Quelldateien mit neutraler EXIF-Ausrichtung. Orientierte Dateien, Kamera-RAW-Dateien und Quellen, deren Ausrichtung nicht feststellbar ist, werden abgelehnt, weil ihre dekodierten Pixel möglicherweise nicht mehr zu den Sensorkoordinaten passen. Verwenden Sie Quellbilder und eine Maske in derselben bekannten Pixelausrichtung mit neutralen Ausrichtungsmetadaten. Die reine Entrauschung benötigt keine Defektmaske.

Die optionale Sensorentrauschung ist unabhängig von der Defektmaske und standardmäßig aus. Prüfen Sie das Ergebnis in voller Größe, um zu entscheiden, ob die Glättung für die Probe geeignet ist. Die Sensorbereinigung bewahrt die Quelldateien und die 8- oder 16-Bit-Bildtiefe; sie ist von der Flat-Field-Korrektur getrennt. Die CLI akzeptiert `--defect-mask-path MASK` und `--sensor-denoise`; Rezepte und die Python-Rezept-API erfassen diese Optionen. Rezepte zeichnen einen Maskenfingerabdruck auf und lehnen eine fehlende oder geänderte Maske ab. CLI-Bereinigung lässt sich nicht mit transparenten Eingaben kombinieren. Gruppierte Verarbeitung wendet die Bereinigung auf Originalbilder an, nicht auf Zwischenergebnisse von Gruppen.

Rezepte verlangen einen Ausgabeort außerhalb jedes Eingabeordners und lehnen eine ausdrücklich gewählte Eingabe als Ausgabe ab. Einen separaten Ausgabeort wählen, um die Quellbilder zu schützen und zu vermeiden, dass ein früheres Ergebnis bei einem erneuten Lauf einbezogen wird. Verarbeitungseinstellungen lassen sich als Vorlage oder Rezept aus einem Projekt extrahieren, auch wenn seine ursprünglichen Quellen verschoben wurden. Eine Verarbeitungsvorlage überträgt den Schalter für Sensorentrauschung, aber keinen Defektmaskenpfad; ein Rezept speichert den Maskenpfad und seinen Fingerabdruck. Das Öffnen des Projekts selbst erfordert weiterhin seine Quellen und jede aufgezeichnete Flat-Field-Referenz.

Wird eine Dateireferenz in den Werkzeugen auf der Basisskalierung des Stapels angewendet, zeichnet ein gespeichertes Projekt ihren absoluten Pfad, ihren Inhaltshash und ihre Staubgröße auf und wendet sie beim Öffnen einmal auf die Originalquellen an. Die Referenzdatei unverändert und an ihrem Ort belassen: Eine fehlende oder geänderte Referenz verhindert das Öffnen des Projekts. Eine aus dem Stapel gewählte Referenz oder eine Korrektur auf einer anderen Arbeits-/Basisskalierung erfordert weiterhin den Export des korrigierten Quellstapels vor dem Speichern eines Projekts. Ändert sich die Arbeitsskalierung nach der Korrektur, vor dem Speichern auf die Korrekturskalierung zurückkehren oder den korrigierten Stapel exportieren.

**Hilfe → Feedback senden…** öffnet lokale Protokolle. Kopieren kopiert das sichtbare Sitzungsprotokoll; das Feedback-ZIP enthält Sitzungs- und vollständige Anwendungsprotokolle sowie Umgebungsinformationen. Prüfen Sie das Paket, bevor Sie es selbst einreichen. Das Öffnen von GitHub öffnet nur den Tracker. Ein abgebrochener Export lässt Dateien unverändert; bei einem Fehler bleibt die vorherige Zieldatei erhalten und der fehlgeschlagene Pfad wird angezeigt.

Jeder Start schreibt sein eigenes rotierendes Protokoll, sodass gleichzeitige GUI-/CLI-Aufträge getrennte Aufzeichnungen behalten. Aufgaben-IDs verknüpfen Parameter, Laufzeit und Ergebnisse (abgeschlossen/abgebrochen/fehlgeschlagen); unerwartete Fehler enthalten einen vollständigen Traceback. Das Feedback-ZIP enthält außerdem den aktuellen Absturzbericht und Abhängigkeitsversionen. Über **Hilfe → Log-Ordner öffnen** oder den auf der Feedback-Seite angezeigten Pfad finden Sie die Dateien. Protokolle rotieren bei 1 MB mit drei Sicherungen pro Prozess; inaktive Sitzungsprotokolle, die älter als 14 Tage sind, werden entfernt. Für ausführliche Support-Protokolle vor dem Start `OPENFOCUS_LOG_LEVEL=DEBUG` setzen; `OPENFOCUS_LOG_DIR` wählt optional einen anderen Protokollordner.

### Kachel-Einstellungen (Speicheroptimierung)

Für große Bilder verwendet OpenFocus eine kachelbasierte Verarbeitung, um Speicherprobleme zu vermeiden.

Zugriff über **Einstellungen → Erweiterte Einstellungen… → Leistung**:

- **Kachelgröße (px)**: Größe jeder Verarbeitungskachel (Standard: 1024 px)
- **Kachelüberlappung (px)**: Überlappung zwischen Kacheln für nahtloses Zusammensetzen (Standard: 256 px)
- **Kacheln ab (px)**: Bildgröße, ab der Kacheln verwendet werden (Standard: 2048 px)

**Empfehlungen**:

- Kleinere Blockgröße: geringerer Speicherbedarf, langsamere Verarbeitung
- Größere Blockgröße: höherer Speicherbedarf, schnellere Verarbeitung
- Überlappung erhöhen, wenn im Ergebnis sichtbare Nähte auftreten

Eine kleinere Kachel kann das Ergebnis um einige Stufen verschieben; die Überlappung ist auf die halbe Kachelgröße begrenzt.

### Registrierungseinstellungen

Zugriff über **Einstellungen → Erweiterte Einstellungen… → Registrierung**:

- **Verkleinerungsbreite**: Auflösung der Vorverarbeitung für die Registrierung
  - Standard: 1024 px
  - Niedriger für Geschwindigkeit, höher für Genauigkeit

### Leistungsreferenz

Siehe [PERFORMANCE.md](./PERFORMANCE.md) für gemessene Durchsatz- und Speicherwerte auf großen realen Stapeln sowie praktische Hinweise für tiefe Stapel und hochauflösende Eingaben.

Das Statusfeld zeigt belegten und gesamten physischen Speicher, wenn das Betriebssystem beide Werte liefern kann; andernfalls zeigt es `N/A`.

### Thread-Einstellungen

Steuert die parallele Verarbeitung auf der CPU:

Zugriff über **Einstellungen → Erweiterte Einstellungen… → Leistung**:

- `0` wählt die automatische Planung und ist der Standard für neue Einstellungen.
- Positive Werte sind Obergrenzen; gespeicherte Grenzen überstehen Upgrades.
- CPU-Kapazität, Ressourcenmodus und freier Speicher können die Parallelität verringern. Mehr Threads machen ein Rendering nicht zwangsläufig schneller.

### Ressourcenmodus

**Einstellungen → Erweiterte Einstellungen… → Leistung → Ressourcenmodus** entscheidet, wie viel der Rechner für ein Rendering verwenden darf. Der Modus ändert nie das fusionierte Bild.

| Modus | Verhalten |
|---|---|
| Ausgeglichen (Standard) | Hält rund 20 % der physischen Kerne und eine Speicherreserve für den Desktop frei |
| Reaktionsschnell | Verwendet etwa die Hälfte der Kerne und ein kleineres In-Flight-Fenster; langsamere Renderings, reaktionsfähigster Rechner |
| Maximal | Verwendet jeden physischen Kern, den das Speicherbudget zulässt |

Der Planer berücksichtigt physische Kerne, CPU-Affinität und Kontingente und führt jeweils eine parallele Ebene aus. Schwere Aufträge teilen sich die Zulassung über Vorbereitung und Veröffentlichung; ein Abbruch wartet, bis der aktive native Aufruf beendet ist. Unbekannter oder unzureichender Speicher erzeugt einen Ressourcenfehler. Das gewählte Kachelraster bleibt unverändert; kleinere Kacheln selbst zu wählen kann die Ausgabepixel verändern.

### GPU-Beschleunigung

StackMFF V1/V2/V3 nutzen ONNX Runtime. DirectML unter Windows und CoreML unter macOS können die Inferenz beschleunigen; andernfalls wird die CPU verwendet. Die Verfügbarkeit eines Providers belegt keine GPU-Ausführung. Weitere GPU-Unterstützung muss validiert werden.

- Umschalten über **Einstellungen → GPU-Beschleunigung**
- Deaktivieren zwingt die neuronalen Methoden auf die CPU – nützlich, wenn GPU-Treiber Probleme bereiten
- Das Statusfeld zeigt `N/A`, wenn kein Beschleuniger verwendet wird oder verfügbar ist, und `CPU*`, wenn ein Beschleuniger vorhanden ist, die gewählte Methode aber nur auf der CPU läuft
- Die Wahl greift beim nächsten Rendern und wird gespeichert

### KI-Batchgröße

Zugriff über **Einstellungen → Erweiterte Einstellungen… → KI**:

- Die gespeicherte Batch-Obergrenze ist standardmäßig 2; die aktuelle Ausführungsrichtlinie begrenzt die tatsächliche Inferenz auf jedem Gerät auf eine Kachel pro Aufruf.
- Ein höherer Wert aktiviert keine ungeprüfte GPU-Batch-Verarbeitung und verringert nicht die räumliche Kachelgröße.

### Oberflächendesign

Es stehen zwei Oberflächendesigns zur Verfügung: **Hell (Standard)** und **Dunkel**. Umschalten über **Einstellungen → Design**; die Wahl wird sofort angewendet (die Windows-Titelleiste folgt) und gespeichert. Beim ersten Start wird das Design im Willkommensdialog gewählt; eine gespeicherte Auswahl hat Vorrang.

### Portabler Modus

Eine leere Datei namens `OpenFocus.portable` neben `OpenFocus.exe` anlegen (das portable ZIP enthält bereits eine) – dann liegen alle Benutzerdaten (Einstellungen, Logs, zuletzt geöffnete Dateien) in einem Ordner `OpenFocusData` neben der Anwendung statt im Benutzerprofil, sodass die Installation auf einem USB-Stick mitgenommen werden kann. Zum Zurückschalten auf das normale Profil die Markierungsdatei löschen.

### EXIF-Metadaten

PNG-, TIFF-, JPEG- und WebP-Exporte behalten bereinigte Kamera-, Objektiv-, Belichtungs-, Datums- und GPS-Metadaten und setzen Orientation auf 1. BMP bettet kein EXIF ein. Überschreiten die EXIF- oder ICC-Daten eines JPEG die Formatgrenze, meldet OpenFocus einen eindeutigen Fehler; wählen Sie dann PNG oder TIFF. Überschreiten die bereinigten EXIF-Daten einer Quelle 16 MiB, verweigert OpenFocus das Laden; entfernen oder reduzieren Sie zuerst die EXIF-Daten dieser Datei.

### HEIC/HEIF-Eingabe

iPhone-Fotos lassen sich direkt laden. Erfordert `pillow-heif`; `pip install openfocus` installiert es, und die gepackten Builds enthalten es.

### Updates

**Hilfe → Nach Updates suchen** vergleicht die eigene Version mit den Versionsinformationen auf `openfocus.top`. Eine neuere Version wird von derselben Website im Hintergrund heruntergeladen, während die App im Leerlauf ist; sobald der Build bereitsteht, bieten Banner und Statusbereich **Aktualisieren und neu starten**, das ihn offline anwendet. Builds, die sich nicht selbst aktualisieren können, öffnen stattdessen die Downloadseite der Website. Die automatische Prüfung läuft höchstens einmal täglich still im Hintergrund.

### Registrierungs-Cache

OpenFocus kann Registrierungsergebnisse in einem Ordner `.openfocus_cache` neben dem Quellstapel ablegen. Wird derselbe Stapel erneut geöffnet (Dateien unverändert), entfällt die Ausrichtung vollständig. Umschalten über **Einstellungen → Erweiterte Einstellungen… → Registrierung → Registrierungs-Cache**; veraltete Einträge werden automatisch ersetzt. Der Schalter ist standardmäßig an.

### Gespeicherte Einstellungen

Alles Folgende übersteht einen Neustart der Anwendung automatisch: Thread-Anzahl, Ressourcenmodus, Kachelparameter, Verkleinerungsbreite der Registrierung, KI-Batchgröße (StackMFF V3), GPU-Beschleunigung, Registrierungs-Cache, Ausrichtungsdiagnose, Oberflächensprache, Fenstergröße/Panel-Layout, Exportformat beim Herausziehen, die Helligkeitsoptionen pro Methode, Glättungs- und Radiuswerte sowie die zuletzt geöffneten Stapel. Einstellungsdialoge speichern in dem Moment, in dem bestätigt wird; zusätzlich wird alles beim Beenden der Anwendung gesichert. Die Voreinstellungen liegen pro Benutzer in `OpenFocus/OpenFocus.ini` (QSettings-INI-Format).

### Oberflächensprache

Jederzeit über **Einstellungen → Sprache** umschaltbar; das Menü listet alle verfügbaren Sprachpakete – English, 中文, 日本語, Español, Deutsch, Français, Русский, 한국어, Português, Italiano, Nederlands und Polski. Die Änderung greift sofort und wird für künftige Sitzungen gespeichert. Eine neue Sprache ist eine einzige Datei `locales/<code>.py`.

### Automatische Update-Prüfung

Einmal täglich, wenige Sekunden nach dem Start, prüft OpenFocus still auf `openfocus.top` nach einer neueren Version. Auch die Versionshinweise stammen von dieser Website. Existiert eine neuere Version, erscheint ein Banner über den Bildbereichen: die Version, **Jetzt aktualisieren** (wendet einen bereitgestellten Build an oder startet den Download), **Neuerungen** (Versionshinweise) und eine Schaltfläche zum Ausblenden. Ausblenden verbirgt das Banner nur für diese Version – die nächste Version bringt es zurück, und **Hilfe → Nach Updates suchen** aktiviert es sofort wieder.

Während der Download im Hintergrund läuft, zeigt die Statusleiste den Fortschritt. Eine manuelle Prüfung ist jederzeit über **Hilfe → Nach Updates suchen** möglich.

---
## 12. Stapelverarbeitung

Mehrere Ordner automatisch verarbeiten:

1. Auf **Werkzeuge → Stapelverarbeitung** gehen
2. Im Stapeldialog:
   - **Ordner hinzufügen**: Ordner mit Bildsequenzen auswählen
   - **Stapelordner hierher ziehen**: einen oder mehrere Stapelordner auf den Dialog oder die Ordnerliste ziehen, um sie auf einmal einzureihen; das Ablegen wechselt in den Mehrordner-Modus, und die Vorschaubilder erscheinen während der Dekodierung
   - **Auswahl entfernen**: ausgewählte Ordner aus der Liste entfernen
3. Für einen einzelnen Ordner kann OpenFocus ihn automatisch in mehrere Stapel aufteilen: **Einzelner Ordner (automatisch in mehrere Stacks aufteilen)** wählen, die Aufteilungsmethode **Feste Anzahl** oder **Zeitschwelle** festlegen, die Vorschau prüfen und **Ordner wählen und aufteilen** anklicken.
4. Ausgabeeinstellungen konfigurieren:
   - **Unterordner im Quellordner anlegen**: erzeugt Ergebnisse in jedem Eingabeordner
   - **Wie Quellordner**: speichert neben der Eingabe und erhält vorhandene Dateien
   - **Ausgabeordner festlegen**: Zielort festlegen
5. Die aktuellen Fusions- und Registrierungseinstellungen prüfen
6. **Stapelverarbeitung starten** anklicken, um zu beginnen

Der Stapeldialog zeigt den Fortschritt in Echtzeit. Die Verarbeitung kann jederzeit abgebrochen werden.

### Hinweise zur Stapelverarbeitung

- Startet mit einem Schnappschuss der Einstellungen des Hauptfensters; Bedienelemente im Stapeldialog überschreiben sie.
- Jeder Ordner bzw. jede geteilte Gruppe wird unabhängig durch dieselbe Pipeline verarbeitet.
- **Gruppierte Fusion** ist eine optionale Möglichkeit, lange Stapel im Mehrordner-Modus zu verarbeiten. Aktivieren Sie das Kästchen **Große Stapel in Gruppen verarbeiten**; Gruppengröße und Überlappung werden erst dann eingeblendet. Die Gruppengröße muss mindestens 2 betragen, die Überlappung zwischen 0 und der Gruppengröße minus 2 liegen. Ist das Kästchen aus, sind die beiden Felder ausgeblendet und die Gruppierung ist deaktiviert. Die Überlappung im Desktop beginnt bei 1 und wird beim Ändern der Größe auf den erlaubten Bereich begrenzt. Gruppen werden in Runden mit temporärem Speicherplatz fusioniert; die Fortschrittsanzeige kennzeichnet jede Runde. Fusion, Belichtungsabgleich, Basiskorrektur und Registrierung verwenden die Bilder und Referenzen jeder Gruppe, auch in Zwischenrunden, sodass das Ergebnis die Verarbeitung des vollständigen Stapels annähert. Die Gruppierung erfordert ein Fusionsverfahren; Tiefe, Herkunft und Exporte ausgerichteter Bilder stehen nicht zur Verfügung. Der Split-Ordner-Modus wird nicht unterstützt; dort wird das Gruppierungskästchen abgelehnt. Registrierte Gruppen verwenden ihr erstes Bild als Koordinatenreferenz. Jede Registrierung entfernt einen konservativen Rand für die Interpolation, und spätere Runden schneiden die verbleibenden Rechtecke einschließlich eines einzelnen Restbilds zusammen. Das endgültige Bild wird daher beschnitten. Jede zu registrierende Gruppe muss in beiden Abmessungen größer als 8 Pixel sein; schwierige Drehungen oder projektive Transformationen können kein sicher nutzbares Rechteck hinterlassen und einen Fehler erzeugen. In diesem Fall vorab ausgerichtete Eingaben ohne Registrierung verwenden oder die Gruppierung deaktivieren.
- Gruppierte Eingaben müssen übereinstimmende orientierte Abmessungen haben und aus einseitigen Dateien mit für Pillow lesbaren Kopfzeilen bestehen. Kamera-RAW, mehrseitige Dateien und Dateien mit Alphakanälen oder Transparenz werden nicht unterstützt, einschließlich transparenter PNG-Stapel. Alpha entfernen oder die Gruppierung deaktivieren, um solche Stapel zu verarbeiten. CLI-Probeläufe prüfen diese Einschränkungen. Ein CLI-Beispiel lautet `python main.py --input stack --output fused.png --group-size 8 --group-overlap 1`. Gruppierte CLI-Eingaben müssen ein Ordner sein und lassen sich nicht mit `--depth-map` kombinieren. Eine andere stapelweite OpenFocus-Ausrichtungsreferenz lässt sich nicht mit der Gruppierung kombinieren. Eine Flat-Field-Referenz wird vor der Registrierung auf Originalbilder angewendet; die Ausgabeschärfung wird einmal nach der letzten Gruppe angewendet.
- **Was wohin gelangt, hängt vom Modus ab.** Bei **Nur Ausrichtung** erhält das gewählte Ziel einen Ordner `<Stapel>_aligned` pro Stapel, und die Ausgabeliste zeigt diesen Ordner. Bei **Fusion + ausgerichtete Bilder speichern** wird das fusionierte Bild als `<Stapel>.<ext>` im Ziel geschrieben, seine ausgerichteten Bilder landen daneben in `<Stapel>_aligned`; die Ausgabeliste zeigt nur das fusionierte Ergebnis. Bei **Fusion ohne Export der Ausrichtung** wird nur das fusionierte Bild geschrieben. Die Ausgabeliste benennt Ordner genau so, wie sie auf dem Datenträger heißen.
- Ausgerichtete Bilder behalten die ursprünglichen Dateinamen. Doppelte Namen und vorhandene Ordner erhalten numerische Suffixe. Quelldateien werden nie ersetzt.
- Übrig gebliebene `*_registered*`-Kopien älterer Versionen werden nicht als Bilder geladen, wenn ihre Originaldatei im selben Ordner vorhanden ist; die Lademeldung nennt die Anzahl der übersprungenen Dateien.
- PNG/TIFF erhalten 16-Bit-Daten; die ausdrückliche Wahl von JPG/BMP wandelt sie in 8 Bit um.
- Ein Abbruch wartet, bis der Worker stoppt. Abgeschlossene Dateien bleiben verfügbar; teilweise, fehlgeschlagene, abgebrochene und nicht gestartete Aufgaben werden getrennt gezählt.

---

## 13. Ergebnisse exportieren

### Einzelnes Ergebnis speichern

1. Das gewünschte Ergebnis in der Ausgabeliste auswählen
2. Auf **Datei → Speichern** gehen oder `Ctrl+S` drücken
3. Format und Speicherort wählen
4. Auf Speichern klicken

Unter **Einstellungen → Erweiterte Einstellungen… → Ausgabebenennung** hilft ein optionales Dateinamenpräfix, neu gerenderte Ergebnisse zu erkennen. Das Beispiel aktualisiert sich während der Eingabe; verwenden Sie bis zu 40 Buchstaben, Zahlen, Unterstriche oder Bindestriche. Leer lassen für die ursprüngliche Benennung. Mit **OK** bestätigen, um es zu merken, oder abbrechen, um die Änderung zu verwerfen. Das Präfix wird Teil neuer Namen in der Ausgabeliste und ihrer vorgeschlagenen Exportnamen; vorhandene Ergebnisse behalten ihre Namen. Im Speicherdialog lässt sich ein Name weiterhin bearbeiten. Diese Voreinstellung benennt keine Quelldateien um und ändert keine Batch-Ausgabenamen.

### Ein Ergebnis teilen

Klicken Sie mit der rechten Maustaste auf ein Ergebnis und öffnen Sie **Teilen**. **Vorschaubild kopieren (JPEG)** legt das aktuelle Ergebnis in voller Auflösung einschließlich Beschriftungen und Maßstabsbalken als echtes JPEG und dekodiertes Bitmap in die Zwischenablage. **Dateipfade kopieren** kopiert lokale Pfade als Text und Datei-URLs. Unverändert gespeicherte Ergebnisse verwenden die vorhandene Datei; ungespeicherte, geänderte oder fehlende Ergebnisse werden ohne Ordnerauswahl als PNG im System-Temporärverzeichnis erstellt und nicht als gespeichert markiert. Das System kann temporäre Dateien löschen; verwenden Sie **Speichern unter** für eine dauerhafte Datei. **Im Ordner anzeigen** öffnet Explorer, Finder oder den Dateimanager mit ausgewählter Datei. **Privaten Freigabelink erstellen** bleibt bis zur Verfügbarkeit von gehostetem Teilen deaktiviert.

### Exportformat beim Herausziehen

Werden Ergebnisse aus der Ausgabeliste herausgezogen, entstehen Bilddateien im Format, das im Kontextmenü der Ausgabeliste unter **Format beim Herausziehen** gewählt wurde (JPG, PNG, WebP oder TIFF). Die Wahl wird gespeichert.

### Registrierten Stapel speichern

1. Auf **Datei → Stack speichern → Registrierter Stack** gehen
2. Speicherformat wählen:
   - **Als Ordner speichern**: ein Ziel und eine Kodierung wählen; jeder Stapel erhält einen neuen Unterordner mit seinen ursprünglichen Dateinamen
   - **Als GIF speichern**: erzeugt eine animierte GIF-Datei aus der ausgerichteten Sequenz

### Eingabe-Stack speichern

1. Auf **Datei → Stack speichern → Eingabe-Stack** gehen
2. Speicherformat wählen:
   - **Als Ordner speichern**: speichert die Originalbilder als eigene Dateien
   - **Als GIF speichern**: erzeugt eine animierte GIF-Datei aus der Eingabesequenz

### Exportformate

- **JPG**: komprimiert, gutes Verhältnis von Qualität zu Dateigröße
- **PNG**: verlustfrei, erhält 8- und 16-Bit-Bilddaten
- **BMP**: unkomprimiert, wandelt 16-Bit-Bilddaten in 8 Bit um
- **TIFF**: erhält 8- und 16-Bit-Daten; der mehrseitige Export bewahrt den gesamten Stapel in einer Datei
- **GIF**: Animation mit 8-Bit-Palette; ein Abbruch lässt ein vorhandenes Ziel unverändert
- **TIFF mit Ebenen**: in der Formatliste von **Speichern unter**; speichert Ergebnis und Quellbilder mit Geometrie und Gültigkeitsmasken

---

## 14. Kommandozeilenschnittstelle

`pip install openfocus` installiert KI-Modell, Gewichte sowie HEIC- und Kamera-RAW-Dekodierung. Für DTCWT unter Python 3.13+ anschließend `python -m pip install --no-deps dtcwt==0.14.0` ausführen. Die NumPy-Versionsvorgabe des Pakets bleibt dabei laut `pip check` unerfüllt; OpenFocus stellt einen Kompatibilitätsadapter bereit.

OpenFocus kann ohne GUI laufen – nützlich für skriptgesteuerte Pipelines und CI. Das Konsolenskript `openfocus` (installiert durch `pip install openfocus`) und `python main.py` aus einem Checkout stellen dieselben Befehle bereit. Fusionsoptionen funktionieren in beiden Formen; die Unterbefehle sind:

```bash
openfocus version                       # Version ausgeben
openfocus describe                      # JSON: Methoden, Parameter, Fähigkeiten und Steuerbefehle
openfocus gui [--control] [--socket NAME] [Pfade…]   # Desktop-App starten
openfocus ctl <command> [flags]          # ein laufendes Fenster steuern (siehe AGENT_CONTROL_SPEC.md)
openfocus recipe create FILE --input … --output … [Verarbeitungsoptionen]
openfocus recipe run FILE [--dry-run]    # ein gespeichertes Headless-Rezept wiedergeben
openfocus retouch replay CHECKPOINT --base BASE --sources … --output RESULT
```

`openfocus gui` startet die App abgekoppelt; `--control` wartet auf den Steuerkanal und gibt dessen Verbindungs-JSON aus. `openfocus describe` und `version` benötigen keine Eingabe. Der Agent-Workflow ist in [AGENT_CONTROL_SPEC.md](./AGENT_CONTROL_SPEC.md) und `skills/openfocus-control/SKILL.md` dokumentiert.

Aus einer Python-Umgebung (oder neben der gepackten ausführbaren Datei):

```bash
# Einen Ordner mit dem standardmäßigen KI-Verfahren fusionieren
python main.py --input ./stack_folder --output ./result/fused.png

# DTCWT mit ECC-Registrierung, 8 Threads
python main.py -i ./stack_folder -o ./result/fused.png -m dtcwt -a ecc -t 8

# StackMFF V3 auf CPU erzwungen, mit eigener Kachelgröße
python main.py -i ./stack_folder -o ./result/fused.png -m stackmffv3 --cpu --tile-size 512

# Explizite Dateiliste; Videodateien werden ebenfalls als Eingabe akzeptiert
python main.py -i img1.jpg img2.jpg img3.jpg -o fused.png -m gfgfgf

# Einzelner mehrseitiger TIFF-Z-Stapel mit metadatenkalibriertem Maßstabsbalken
python main.py -i ./zstack.tif -o fused.png -m dtcwt --scale-bar

# Benachbarte ähnliche Bilder prüfen, ohne zu fusionieren oder die Eingabe zu ändern
python main.py -i ./stack_folder --suggest-duplicates
```

`--suggest-duplicates` akzeptiert einen Ordner oder eine geordnete Liste einseitiger Bilddateien. Es gibt ein JSON-Objekt mit den Arrays `files` und `suggestions` aus; `index` und `reference_index` jeder Empfehlung sind nullbasierte Positionen in `files`, und `difference` ist der mittlere normalisierte Stichprobenunterschied. Mit `--json report.json` lässt sich dasselbe Objekt speichern. Dieser Prüfmodus lässt sich nicht mit `--output`, `--output-dir`, `--json-progress`, `--dry-run` oder Sensorbereinigungsoptionen kombinieren. Der Bericht vergleicht benachbarte Dateien in der übergebenen Eingabereihenfolge, ohne zu wissen, welche Bilder Sie später ausschließen. Wenn Sie ein Referenzbild ausschließen, überdenken Sie seinen Nachfolger oder prüfen Sie die verbleibenden Dateien erneut in ihrer beabsichtigten Reihenfolge.

Für ein einzelnes Rendering lässt sich beispielsweise `--align openfocus --alignment-report alignment.json` hinzufügen. Das JSON-Objekt enthält `method`, `input`, `frame_count` und `entries`. `frame_index` jedes Eintrags ist nullbasiert in der dekodierten Eingabereihenfolge, vor der Fusionssortierung; mehrseitige Bilder werden in Seitenreihenfolge, Videobilder in Dekodierreihenfolge erweitert. Zusammengesetzte Methoden enthalten getrennte Einträge für jede Stufe. Der Bericht wird nach erfolgreicher Verarbeitung atomar geschrieben und darf weder eine Eingabe noch eine andere Ausgabe ersetzen. Stapel-, gruppierte und Probelauf-Modi lehnen diese Option ab.

| Option | Beschreibung |
|--------|-------------|
| `--input, -i` | Bildordner (auch mehrere), Videodatei, eine bzw. mehrere Bilddateien oder eine mehrseitige TIFF |
| `--output, -o` | Pfad des Ausgabebildes (die Endung bestimmt das Format) |
| `--output-dir, -d` | Stapelmodus: jeden Eingabeordner fusionieren und pro Stapel `<Ordner>.png` schreiben |
| `--method, -m` | `stackmffv3` (Standard), `openfocus_a`, `openfocus_b`, `openfocus_c`, `guided_filter`, `dct`, `dtcwt`, `gfgfgf`, `stackmff`, `stackmffv2` |
| `--align, -a` | `none` (Standard), `homography`, `ecc`, `both`, `openfocus` |
| `--param KEY=JSON_VALUE` | Fusionsparameter; für mehrere Schlüssel wiederholen; überschreibt ältere Flags |
| `--align-param KEY=JSON_VALUE` | Registrierungsparameter; für mehrere Schlüssel wiederholen |
| `--align-shift-limit` | Suchbereich für Verschiebungen als Anteil der Bildgröße (Standard 0,03); kumulierte Verschiebungen können größer sein. |
| `--alignment-report PATH` | Ergebnisse der Ausrichtung pro Bild als JSON für ein einzelnes, nicht gruppiertes Rendering mit aktivierter Ausrichtung speichern; Stapel- und gruppierte Renderings werden nicht unterstützt |
| `--exposure-match` | In der CLI für alle Methoden standardmäßig an; A/B/C verwenden das erste verarbeitete Bild, andere Methoden das letzte. `--no-exposure-match` schaltet aus. |
| `--base-frame` | Optionaler Schritt nach der Fusion: die niedrigen Frequenzen der Ausgabe an das `last`/`first`-Bild angleichen (Standard `none`) |
| `--sort` | Bildreihenfolge für `openfocus_a/b/c`: `asc` (Standard, Dateireihenfolge), `desc` (kehrt einen Fern-nach-Nah-Durchlauf um), `off` |
| `--frame-order` | Ordner als Eingabe: `auto` (Standard) sortiert die Bilder von nah nach fern nach den EXIF-Fokusschritten, wenn alle Bilder sie haben (OM-SYSTEM-Fokus-Bracketing beginnt in der Mitte), sonst nach Namen; `name` immer nach Dateiname. Einzeln angegebene Dateien behalten ihre Reihenfolge |
| `--dust-map WHITE` | Flat-Field-Korrektur (Staubkarte) aus einer defokussierten Weißfeldaufnahme |
| `--defect-mask-path MASK` | Sensordefektmaske; Werte ungleich null werden vor der Registrierung repariert |
| `--sensor-denoise` | Behutsame Sensorentrauschung vor der Registrierung |
| `--kernel, -k` | Nur für Methoden mit deklariertem `kernel_size` |
| `--sharpen-output` | Mildes Unscharfmaskieren auf das Fusionsergebnis anwenden (standardmäßig aus) |
| `--threads, -t` | Thread-Obergrenze; `0` = automatisch (Standard), positive Werte begrenzen die Parallelität |
| `--resource-preference` | Ressourcenmodus: `balanced` (Standard), `responsive` oder `maximum` |
| `--batch-size` | KI-Kachel-Batch-Obergrenze (Standard 2); die tatsächliche Inferenz verwendet derzeit eine Kachel pro Aufruf |
| `--ai-mode` | StackMFF V3: `soft` (Standard, wahrscheinlichkeitsgewichtete Mischung) oder `hard` (wahrscheinlichstes Bild pro Pixel) |
| `--cpu` | CPU erzwingen, auch wenn eine GPU verfügbar ist |
| `--downscale` | Verkleinerungsbreite der Merkmalserkennung bei der Registrierung |
| `--tile-size` | Kachelblockgröße für große Bilder (Standard: automatisch) |
| `--tile-enabled` / `--no-tile-enabled` | Gekachelte Fusion aktivieren bzw. deaktivieren (Standard: an) |
| `--tile-overlap` | Kachelüberlappung in Pixeln (Standard: automatisch) |
| `--tile-threshold` | Mindestbildgröße für die Kachelung (Standard: automatisch) |
| `--group-size` | Ordner in Gruppen dieser Größe fusionieren; `0` deaktiviert die Gruppierung (Standard). Nur Ordner; kein `--depth-map`, kein RAW-/Alpha-/mehrseitiger Eingang und keine nicht standardmäßige OpenFocus-Referenz |
| `--group-overlap` | Von aufeinanderfolgenden Gruppen geteilte Bilder; Standard `0`, höchstens Gruppengröße minus 2 |
| `--scale-bar` | Maßstabsbalken zeichnen (Kalibrierung automatisch aus den TIFF-Metadaten erkannt) |
| `--scale-um` | Pixelgröße in µm/px, überschreibt die Metadaten (impliziert `--scale-bar`) |
| `--depth-map [path]` | Zusätzlich eine kolorierte Tiefenkarte (Fokusposition) speichern (Standard `<output>_depth.png`); nur tiefenfähige Methoden, nicht gruppiert oder transparent. Im Batch-Modus ist nur der Standardpfad erlaubt |
| `--depth-colormap` | Farbtabelle für `--depth-map`: turbo, viridis, jet, gray, … (Standard turbo) |
| `--auto-background [preset]` | Den gestackten Hintergrund durch den eines ausgerichteten Einzelbilds ersetzen, wie Auto-Hintergrund in der Retusche: `auto` (Standard), `soft` (unschärfster Hintergrund) oder `sharp` (Fern-Fokus-Bild). Benötigt die ausgerichteten Bilder, daher nicht mit gruppierter Fusion |
| `--background-mode` | Auto-Hintergrund: `clean` (Standard) entfernt zusätzlich die Kantenhalos des Stacks, im Rahmen dessen, was die Bilder aufgezeichnet haben; `faithful` behält sie |
| `--keep-soft-parts` | Auto-Hintergrund: weiche, am Motiv hängende Teile behalten, auf die ein Bild fokussiert hat |
| `--save-stacked [path]` | Mit `--auto-background` zusätzlich das Ergebnis vor dem Ersetzen speichern (Standard `<output>_stacked<ext>`). Im Stapelmodus nur der Standardpfad |
| `--missing-data-map [path]` | Mit `--auto-background` zusätzlich die Fehlende-Daten-Karte und je Klasse eine 8-Bit-Maske speichern (Standard `<output>_missing.png`). Im Stapelmodus nur der Standardpfad |
| `--dry-run` | Eingaben prüfen und den Plan ausgeben, ohne zu fusionieren |
| `--resume` | Stapelmodus: Ordner überspringen, deren Ausgabedatei bereits existiert |
| `--json FILE` | Eine maschinenlesbare Ergebniszusammenfassung in FILE schreiben |
| `--json-progress` | Ein JSON-Fortschrittsobjekt pro Schritt auf stdout ausgeben (NDJSON); Prosa geht nach stderr |
| `--config FILE` | Stapelmodus: JSON-Datei mit Überschreibungen pro Ordner, z. B. `{"stackA": {"method": "dct", "kernel": 9}}` |
| `--suggest-duplicates` | Nahezu identische benachbarte Bilder zur manuellen Prüfung melden; fusioniert oder schließt sie nicht aus |

Es gibt keinen `--method none`-Wert: Die CLI fusioniert immer. Für reine Ausrichtungsausgaben die Desktop-App verwenden.

### Rezepte

Rezepte erfassen Einstellungen der Headless-Verarbeitung zur Wiederverwendung und zum Verschieben:

```bash
openfocus recipe create workflow.json -i ./stack_folder -o ./result/fused.png -m dct --cpu
openfocus recipe run workflow.json --dry-run
openfocus recipe run workflow.json
```

Das JSON-Format der Version 1 enthält `version`, ein `input`-Array, entweder `output` oder `output_dir` und ein `options`-Objekt. Pfade sind relativ zur Rezeptdatei; verschieben Sie das Rezept zusammen mit seinen Eingabedateien. Das Erstellen übernimmt die CLI-Einstellungen und die aufgelösten Algorithmusparameter. Unbekannte Felder, ungültige Werte und nicht unterstützte Versionen werden abgelehnt. Ein Probelauf prüft die Eingabe, ohne Ergebnisdateien zu schreiben. Bestehende Projektdateien bleiben unverändert.

Einstellungen aus einem vorhandenen `.ofproj` übernehmen:

```bash
openfocus recipe create workflow.json --project session.ofproj -i ./new_stack -o ./result/fused.png
```

Das überträgt Fusion, Registrierung, Helligkeit, Reihenfolge, Schärfung und Kacheleinstellungen in das CLI-Rezept und bettet die Verarbeitungseinstellungen für die GUI ein. Bildausschlüsse, Ladeskalierung, Beschriftungen und Bearbeitungen werden nicht übertragen. Projekte mit Quelltransformationen, einschließlich Flat-Field-Korrekturen, müssen zuerst ihren korrigierten Quellstapel exportieren. Rezepte verlangen Ausgaben außerhalb ihrer Eingabeordner und akzeptieren niemals eine Eingabedatei als Ausgabe. Sie lehnen transiente Steueroptionen wie `--dry-run`, `--resume`, `--json` und `--config` ab. Eine Defektmaske wird als relativer Pfad mit einem SHA-256-Fingerabdruck gespeichert; eine fehlende oder geänderte Maske lässt den Lauf fehlschlagen.

Gruppierungsoptionen lassen sich aus CLI-Argumenten übernehmen.

In der GUI speichert **Datei → Verarbeitungsvorlage speichern** Einstellungen ohne Quellpfade. **Datei → Verarbeitungsvorlage anwenden** wendet eine Vorlage, ein Projekt oder ein projektgestütztes Rezept auf den aktuellen Stapel an, ohne dessen Eingaben, Transformationen, Ausschlüsse oder Beschriftungen zu ersetzen. Eine Vorlage ist selbst kein ausführbares Rezept. Die in einem Rezept eingebetteten GUI-Einstellungen sind unabhängig von seinen CLI-Optionen; wenn Sie die JSON-Datei bearbeiten, halten Sie beides konsistent.

Dieselbe Headless-API ist ohne Import von PyQt verfügbar:

```python
from core.recipes import capture_project_recipe, capture_recipe, load_recipe, save_recipe, run_recipe

capture_recipe("workflow.json", ["-i", "stack_folder", "-o", "result/fused.png", "-m", "dct"])
capture_project_recipe("from-project.json", "session.ofproj",
                       ["-i", "new_stack", "-o", "result/fused.png"])
recipe = load_recipe("workflow.json")  # validiertes Dictionary mit relativen Pfaden
save_recipe("workflow.json", recipe)
exit_code = run_recipe("workflow.json", dry_run=True)  # False zum Verarbeiten
```

`capture_recipe` interpretiert seine Argumente als gewöhnliche CLI-Pfade relativ zum aktuellen Verzeichnis; `save_recipe` interpretiert gespeicherte Pfade relativ zum Rezept. Das Rezept zeichnet Einstellungen und Referenzen auf, einschließlich eines Inhaltsfingerabdrucks für eine ausdrücklich gewählte Sensordefektmaske. Es zeichnet nicht die Eingabepixel, Softwareversionen, Hardware oder Prüfsummen des Quellstapels auf; bewahren Sie diese separat auf, wenn Herkunft wichtig ist. Identische Einstellungen garantieren keine identischen Pixel über Softwareversionen, Treiber oder Geräte hinweg; GPU-neuronale Operationen können nicht deterministisch sein. Verwenden Sie `--cpu`, wenn eine CPU-Ausführung erforderlich ist, und bewahren Sie die installierte Software und die Eingaben auf.

### Retusche-Kontrollpunkte wiedergeben

Verwenden Sie die Kontrollpunkt-Speicher- und Ladefunktionen im Retusche-Arbeitsbereich mit der ursprünglichen Basis und den verarbeiteten Quellen. Sitzungen mit automatischem Hintergrund, auch erfolgreiche Analysen ohne Pixeländerung, speichern verlustfreie v2-`.ofretouch`-ZIP-Snapshots mit JSON-Metadaten und NumPy-Daten: das exakte Arbeitsbild, erreichbare Rückgängig-/Wiederholen-Kacheln, Diagnosezustände und Quellherkunft. Beim Laden von v2 wird der Hintergrund nicht neu berechnet. Reine Pinsel-/Tonwertsitzungen können kompaktes JSON v1 verwenden; ältere Hintergrund-Checkpoints in JSON v1 bleiben abspielbar.

Die Quellprüfung läuft in einem abbrechbaren Hintergrundvorgang. Vor dem Ersetzen der Zieldatei lädt das Speichern die vorbereitete Datei erneut und prüft das aktuelle Bild, erreichbare Rückgängig-/Wiederholen-Kacheln und Diagnosezustände auf exakte Gleichheit. Nicht passende Quellen, eine fehlgeschlagene Prüfung, Abbruch oder unzureichender Speicher lassen aktuelle Bearbeitungen und einen vorhandenen Kontrollpunkt unverändert.

Verhindert das Aufzeichnungslimit des Kontrollpunkts eine weitere Bearbeitung, übernehmen Sie die aktuellen Bearbeitungen und beginnen Sie eine neue Retuschesitzung, um fortzufahren.

Speichern Sie einen Retusche-Kontrollpunkt im Retusche-Arbeitsbereich und geben Sie anschließend das ursprüngliche Basisbild sowie jedes verarbeitete Quellbild in derselben Reihenfolge wie in der Sitzung an:

```bash
openfocus retouch replay edits.ofretouch --base fused.png --sources processed1.png processed2.png --output retouched.tif
```

Beide Formate benötigen weiterhin das ursprüngliche Basisbild und alle tatsächlich für die Retusche verwendeten Quellbilder in Sitzungsreihenfolge – nach Ausrichtung und einem etwaigen ROI-Zuschnitt oder anderer Verarbeitung – mit ihren ursprünglichen Pixelwerten und ihrer Bittiefe. Das Laden rekonstruiert die aufgezeichneten Bildversätze und das Ausrichtungsverhalten und prüft Basis, geordnete Bilder und Gültigkeitsmasken; unvereinbare Pixel, Geometrie oder Masken werden abgelehnt. Jeder Pfad muss genau ein Bild enthalten. Die Ausgabe muss PNG oder TIFF sein und behält die Bittiefe der Basis. Kontrollpunkt und Eingabebilder dürfen nicht als Ausgabe dienen.

Exit-Codes: `0` Erfolg, `1` Verarbeitungsfehler, `2` Nutzungsfehler. Der Aufruf `python main.py --help` gibt alle Optionen aus; ein Start mit einem reinen Pfad (`python main.py C:\stack`) öffnet weiterhin die GUI mit diesem Ordner.

---

## 15. Tastenkürzel

### Dateioperationen

| Tastenkürzel | Aktion |
|----------|--------|
| `Ctrl+O` | Ordner öffnen |
| `Ctrl+Shift+O` | Video öffnen |
| `Ctrl+Alt+O` | Projekt öffnen |
| `Ctrl+Alt+S` | Projekt speichern |
| `Ctrl+S` | Ergebnis speichern |
| `Ctrl+Shift+S` | Registrierten Stack speichern |
| `Ctrl+W` | Stack leeren |
| `Ctrl+Q` | Beenden |
| `Ctrl+R` | Rendern starten |
| `Ctrl+Shift+R` | Alle Methoden vergleichen |
| `Ctrl+T` | Ergebnis retuschieren |
| `Ctrl+M` | Tiefenkarte |
| `Ctrl+E` | Wischvergleich umschalten |
| `Ctrl+Shift+B` | Stapelverarbeitung |
| `Ctrl+Z` / `Ctrl+Y` | Rückgängig / Wiederholen (Stapelbearbeitungen) |

### Navigation

| Tastenkürzel | Aktion |
|----------|--------|
| `Pfeil oben` / `Pfeil unten` (Quellenliste fokussiert) | Vorheriges / nächstes Bild |
| `Mausrad` | Hinein-/Herauszoomen |
| `Ctrl + Mausrad` bei gedrückter linker Maustaste | Größe der Lupe ändern |
| `Mittlere Maustaste` | Bild an das Fenster anpassen |

### Allgemein

| Tastenkürzel | Aktion |
|----------|--------|
| `Leertaste` (gedrückt halten, Zeiger über einem Bild) | Temporärer Verschiebemodus |
| `Entf` | Fokussiertes Listenelement löschen; löscht außerdem die ROI, wenn der Zeiger über dem Quellbild liegt |

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

- Prüfen Sie ONNX Runtime und das mitgelieferte Modell.
- Über **Hilfe → Umgebungsinformationen** prüfen

#### Fehler wegen zu wenig Arbeitsspeicher

- Die Kachelverarbeitung in **Einstellungen → Erweiterte Einstellungen… → Leistung** aktivieren
- Die Kachelblockgröße verringern
- Kleinere Kernelgrößen verwenden
- Die Bilder auf kleinere Abmessungen skalieren

#### Registrierung schlägt fehl

- Für begrenzte starre Bewegung **OpenFocus (schnell)** wählen oder **Bei abgelehntem Bild stoppen** aktivieren, um früh fehlzuschlagen statt unausgerichtete Bilder zu akzeptieren
- Die Verkleinerungsbreite in **Einstellungen → Erweiterte Einstellungen… → Registrierung** reduzieren
- Sicherstellen, dass die Bilder ausreichend Kontrast und Merkmale haben
- Unter **Einstellungen → Erweiterte Einstellungen… → Registrierung** die **Ausrichtungsdiagnose erfassen** aktivieren, erneut rendern und **Einstellungen → Erweiterte Einstellungen… → Registrierung → Ausrichtungsdiagnose** für die betroffenen Bilder lesen

#### Langsame Verarbeitung

- Threads auf automatisch (`0`) lassen; CPU-Kapazität, Ressourcenmodus und Speicherreserve begrenzen die Parallelität bereits
- Für einen unbeaufsichtigten Stapel den Modus **Maximal** verwenden oder Guided Filter / GFG-FGF statt einer KI-Methode wählen
- Die Bildauflösung reduzieren oder während des Abstimmens die **Schnellvorschau** verwenden

#### Windows-SmartScreen-Warnung beim ersten Start

Ob die Windows-Binärdateien signiert sind, hängt vom jeweiligen Release ab (siehe Release-Notes); eine Signatur ist nicht garantiert. Laden Sie sie von **https://openfocus.top** statt von einem Drittanbieter-Spiegel. Falls SmartScreen warnt, klicken Sie auf **Weitere Informationen → Trotzdem ausführen**. Der erste Start nach dem Entpacken kann 30–60 Sekunden dauern.

#### macOS meldet, die App „kann nicht geöffnet werden“ oder „ist beschädigt“

Die macOS-Builds sind nicht notarisiert. Laden Sie den Apple-Silicon-Build von **https://openfocus.top**. Wenn Sie das DMG verwenden und `OpenFocus.app` in den Ordner Programme verschieben, führen Sie aus:

```bash
xattr -cr /Applications/OpenFocus.app
```

Bei einer Terminal-Installation mit `install.sh` liegt die App in `~/Applications`; führen Sie dann `xattr -cr ~/Applications/OpenFocus.app` aus.

oder mit der rechten Maustaste auf die App klicken und **Öffnen** wählen. Die veröffentlichten macOS-Builds sind für Apple Silicon (M1/M2/M3/M4); Intel-Macs können aus dem Quellcode laufen.

### Hilfe erhalten

1. **Umgebungsinformationen**: Über **Hilfe → Umgebungsinformationen** die Installation prüfen
2. **Kontakt zum Support**: Über **Hilfe → Kontakt** Supportinformationen aufrufen
3. **Feedback senden**: Über **Hilfe → Feedback senden…** Protokolle kopieren oder ein Paket zur manuellen Einreichung exportieren

---

## Anhang: Details zu den Fusionsalgorithmen

### Guided Filter

Der Guided Filter führt eine kantenbewahrende Glättung mit einem Führungsbild durch. Er berechnet lokale lineare Transformationen zwischen Führungs- und Eingabebild und wendet diese auf die Ausgabe an. Diese Methode zeichnet sich dadurch aus, dass sie Kanten erhält und zugleich Rauschen reduziert.

### DCT (Diskrete Kosinustransformation)

DCT überführt Bilder in den Frequenzbereich, in dem Fusionsentscheidungen anhand der Koeffizientenbeträge getroffen werden. Hochfrequente Anteile (Details) bleiben erhalten, während niederfrequente Anteile (gleichmäßige Bereiche) anhand der lokalen Energie zusammengeführt werden.

### DTCWT (Dual-Tree-Complex-Wavelet-Transformation)

DTCWT liefert eine mehrskalige, mehrdirektionale Zerlegung von Bildern. Es erfasst Merkmale auf verschiedenen Skalen und in verschiedenen Orientierungen und ermöglicht damit ausgefeilte Fusionsregeln, die sowohl Textur- als auch Strukturinformationen erhalten.

### GFG-FGF

Implementiert Fu et al. (2024): Die grobe Fokuskarte |I − mean(I)| wird mit dem gaußartigen Vier-Nachbarschafts-Gradienten angereichert – die 3×3-Schablone T = [1 2 1; 2 4 2; 1 2 1]/16 ergibt P = I⊗T, und G = (|I(x±1,y) − P|)/2 + (|I(x,y±1) − P|)/2, wobei Werte unterhalb von T0 = 0,005 verworfen werden – und anschließend durch einen schnellen Guided Filter verfeinert. Jedes Pixel wählt das Bild mit dem größten Fokuswert (bei Gleichstand: 3×3-Sobel-Kleinbereichsmaß), und die binären Entscheidungskarten werden vor der gewichteten Mischung mit dem schnellen Guided Filter geglättet.

### KI-Fusion (StackMFF V3)

Ein Deep-Learning-Ansatz mit einem neuronalen Netz, das auf großen Datensätzen von Multifokus-Bildpaaren trainiert wurde. Das Netz erlernt optimale Fusionsstrategien automatisch und liefert bei richtiger Konfiguration Ergebnisse auf dem aktuellen Stand der Technik.

---

*OpenFocus ist ein Open-Source-Projekt. Für Updates, Fehlerberichte und Funktionswünsche besuchen Sie bitte das Projekt-Repository oder https://openfocus.top.*
