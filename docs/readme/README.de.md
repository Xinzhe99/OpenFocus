<p align="center">
  <a href="https://github.com/Xinzhe99/OpenFocus">
    <img src="../../assets/OpenFocus.png" alt="OpenFocus" width="128">
  </a>
</p>

<h1 align="center">OpenFocus</h1>

<p align="center"><strong>Jedes Pixel aus seinem schärfsten Bild.</strong></p>

<p align="center">
  Zehn Fusionsverfahren, fünf Ausrichtungsoptionen und eine quellengebundene Retusche für Makro-, Mikroskopie- und Industriestapel. 16 Bit durchgängig, eine Headless-CLI und eine Oberfläche in zwölf Sprachen.
</p>

<p align="center">
  <a href="https://github.com/Xinzhe99/OpenFocus/releases"><img src="https://img.shields.io/github/v/release/Xinzhe99/OpenFocus?label=release&color=2f81f7" alt="Release"></a>
  <a href="https://github.com/Xinzhe99/OpenFocus/stargazers"><img src="https://img.shields.io/github/stars/Xinzhe99/OpenFocus?color=e3b341" alt="Stars"></a>
  <a href="../../LICENSE"><img src="https://img.shields.io/badge/license-MIT-3fb950" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Windows-10%2F11-0078D7?logo=windows&logoColor=white" alt="Windows 10/11">
  <img src="https://img.shields.io/badge/macOS-Apple%20Silicon-000000?logo=apple&logoColor=white" alt="macOS Apple Silicon">
  <img src="https://img.shields.io/badge/Linux-CLI-FCC624?logo=linux&logoColor=black" alt="Linux CLI">
  <a href="https://pypi.org/project/openfocus/"><img src="https://img.shields.io/badge/PyPI-openfocus-3775A9?logo=pypi&logoColor=white" alt="PyPI"></a>
</p>

<p align="center">
  <a href="#install">Download</a> ·
  <a href="../USER_MANUAL_DE.md">Dokumentation</a> ·
  <a href="../../CHANGELOG.md">Änderungsprotokoll</a> ·
  <a href="https://doi.org/10.5281/zenodo.23050823">DOI</a>
</p>

<p align="center">
  <a href="../../README.md">English</a> ·
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

<p align="center">
  <img src="../../assets/ui.png" alt="OpenFocus-Oberfläche" width="880">
</p>

## 📢 News

> [!NOTE]
> **v1.74.0** — Die OpenFocus-A/B/C-Fusion und die schnelle Ausrichtung wurden aktualisiert. Die Desktop-App startet ihren authentifizierten lokalen Agent-Kanal jetzt automatisch, sodass `openfocus ctl` das laufende Fenster ohne spezielles Startflag steuern kann.

Ältere Versionen stehen im [Änderungsprotokoll](../../CHANGELOG.md).

📸 Wenn Sie mit OpenFocus erstellte Bilder veröffentlichen, freuen wir uns über einen Hinweis wie:

> Erstellt mit OpenFocus — https://github.com/Xinzhe99/OpenFocus

Das ist nicht verpflichtend, aber sehr willkommen.

<a id="features"></a>
## Funktionen

<table>
<tr>
<td width="46%" valign="middle">

### KI-Fusion: StackMFF V3

Das Standardverfahren nutzt ein neuronales Netz, um für jeden Pixel das wahrscheinlichste Quellbild zu schätzen, und mischt anschließend die vorhergesagten Beiträge oder wählt das wahrscheinlichste Bild. Der synthetische Benchmark vergleicht die Verfahren mit einer bekannten Referenz; die Ergebnisse hängen vom Stapel und den Einstellungen ab.

[StackMFF V3 →](https://github.com/Xinzhe99/StackMFF-V3) · [Benchmark →](../../tools/synthetic_benchmark.py)

</td>
<td width="54%">

<a href="../../assets/readme/ai.mp4"><picture><source srcset="../../assets/readme/ai.gif" type="image/gif"><img src="../../assets/readme/ai_poster.jpg" alt="StackMFF V3 gegen Guided Filter auf demselben Stapel" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Wischvergleich

Ziehen Sie den Trenner über ein Quellbild und das Fusionsergebnis, um genau zu sehen, was die Fusion verändert hat – Ausrichtungsdrift, Ränder mit Halos und verlorene Details sind auf einen Blick sichtbar, in jeder Zoomstufe.

[Handbuch →](../USER_MANUAL_DE.md)

</td>
<td width="54%">

<a href="../../assets/readme/wipe.mp4"><picture><source srcset="../../assets/readme/wipe.gif" type="image/gif"><img src="../../assets/readme/wipe_poster.jpg" alt="A/B-Wischvergleich zwischen Quellbild und Fusionsergebnis" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Methodenvergleich

Alle Verfahren einmal rendern, nebeneinander am gleichen Ausschnitt prüfen und den Gewinner als neue Ausgabe übernehmen oder seine Parameter ins Hauptfenster zurückschreiben.

</td>
<td width="54%">

<a href="../../assets/readme/method-comparison.mp4"><picture><source srcset="../../assets/readme/method-comparison.gif" type="image/gif"><img src="../../assets/readme/method-comparison_poster.jpg" alt="Acht Fusionsverfahren im direkten Vergleich" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Tiefenkarte &amp; 3D

Fokusposition pro Pixel aus fünf Fokusmaßen, wissenschaftliche Farbskalen, physikalische Z-Kalibrierung aus TIFF-Metadaten und texturierter OBJ-Export der Oberfläche.

</td>
<td width="54%">

<a href="../../assets/readme/depth.mp4"><picture><source srcset="../../assets/readme/depth.gif" type="image/gif"><img src="../../assets/readme/depth_poster.jpg" alt="Tiefenkarte und 3D-Oberfläche desselben Stapels" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Quellengebundene Retusche

Malen Sie aus den Bildern, die das Ergebnis tatsächlich erzeugt haben: gemeinsamer Zoom und gemeinsame Verschiebung, Regler für das Quellbild, weicher Pinsel, Wiederherstellungspinsel, Rückgängig für ganze Striche und Schutz für nicht exportierte Pixel.
Übernehmen Sie die Retusche als neue Ausgabe und wählen Sie dann im Hauptfenster im Kontextmenü dieser Ausgabe **TIFF mit Ebenen exportieren**, um sie zusammen mit ihren Quellbildern zu speichern. Live-Beschriftungen und Maßstabsbalken sind nicht enthalten. GIMP 3.2.6 unter Windows öffnet das Ergebnis und die Quellbilder als Ebenen; andere Editoren wurden nicht geprüft.

</td>
<td width="54%">

<a href="../../assets/readme/retouch.mp4"><picture><source srcset="../../assets/readme/retouch.gif" type="image/gif"><img src="../../assets/readme/retouch_poster.jpg" alt="Retusche mit Quellbild und Ergebnis" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Stapelverarbeitung

Auf einen Ordner voller Stapel zeigen, Verfahren und Ziel wählen und laufen lassen. Läufe mit ausschließlicher Ausrichtung schreiben einen Ordner pro Stapel; Fusionsläufe schreiben das fusionierte Bild und listen genau dieses. Eine optionale Gruppenfusion tauscht exakte Gleichwertigkeit zum vollständigen Stapel gegen begrenzten Gruppenspeicher und temporären Speicherplatz.

</td>
<td width="54%">

<a href="../../assets/readme/batch.mp4"><picture><source srcset="../../assets/readme/batch.gif" type="image/gif"><img src="../../assets/readme/batch_poster.jpg" alt="Dialog für Stapelverarbeitung" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Maßstabsbalken für Publikationen

Aus TIFF-Metadaten oder manuell kalibriert, mit der echten Schrift in der echten Länge gezeichnet und in jeden Export eingebrannt – Einzelbilder, Stapel, mehrseitige TIFF, GIF und Batch-Ausgaben gleichermaßen. Ein Balken, der nicht lesbar hineinpasst, meldet das, statt eine Länge zu drucken, die er nicht zeichnet.

[Handbuch →](../USER_MANUAL_DE.md)

</td>
<td width="54%">

<a href="../../assets/readme/scalebar.mp4"><picture><source srcset="../../assets/readme/scalebar.gif" type="image/gif"><img src="../../assets/readme/scalebar_poster.jpg" alt="Maßstabsbalken-Dialog mit Live-Vorschau" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Beschriftungen und Legenden

Beschriftungen bleiben eine verwaltete Liste; jede hat einen eigenen Geltungsbereich (Quelle, ausgerichteter Stapel oder eine bestimmte Ausgabe), einen Bildbereich und einen Stil. Vorschau und jeder Export verwenden dasselbe Ziel und dieselbe Bildnummer, sodass `{value}` in Speichern, Speichern unter, Herausziehen und Alles exportieren gleich liest.

</td>
<td width="54%">

<a href="../../assets/readme/labels.mp4"><picture><source srcset="../../assets/readme/labels.gif" type="image/gif"><img src="../../assets/readme/labels_poster.jpg" alt="Beschriftungsverwaltung mit Live-Vorschau" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Agent-steuerbar

Jedes Fenster, jedes Widget und jedes Menü ist über einen token-geschützten lokalen Kanal ansprechbar: Ein Agent lädt einen Stapel, rendert, retuschiert, vergleicht, berechnet Tiefen, exportiert und beantwortet Dialoge – per typisiertem Befehl, per Widget-Pfad oder mit echten Maus- und Tastaturereignissen. Kein Screen-Scraping und keine OCR nötig; der Kanal startet mit der Desktop-App.

[Protokoll →](../AGENT_CONTROL_SPEC.md) · [Skill →](../../skills/openfocus-control/SKILL.md)

</td>
<td width="54%">

<a href="../../assets/readme/agent.mp4"><picture><source srcset="../../assets/readme/agent.gif" type="image/gif"><img src="../../assets/readme/agent_poster.jpg" alt="Ein Agent steuert die laufende GUI: laden, KI-Rendern, vergleichen, retuschieren, wischen, Tiefe, Beschriftungen" width="100%" /></picture></a>

</td>
</tr>
</table>

**Außerdem enthalten:**

- **Fokus-Stacking-Workflow** — Stapelbrowser, Schärfekurve, Schnellvorschau, ROI-Rendern und pro Verfahren gemerkte Einstellungen über Neustarts hinweg.
- **Mikroskopiebereit** — 16 Bit durchgängig, µm/px aus TIFF-Metadaten, Flat-Field-Korrektur per Staubkarte, mehrseitige TIFF-Z-Stapel, texturierter OBJ-Export.
- **Farbprofile** — Einzelbild-Exporte als PNG, TIFF, JPEG und WebP behalten ein gemeinsames RGB-ICC-Profil, wenn alle Quellbilder dasselbe Profil tragen. Gemischte oder unmarkierte Stapel erhalten kein Profil; die Pixel werden nicht farbkonvertiert.
- **Transparentes PNG** — Stapel mit übereinstimmenden rechteckigen Masken konstanter Deckkraft lassen sich in voller Größe fusionieren und als PNG speichern. Einschränkungen stehen im Handbuch.
- **Headless-CLI** — dieselbe Pipeline als `openfocus`, mit Batch-Modus, `--dry-run`, `--resume` und JSON-Ausgabe für Skripte.
- **Agent-steuerbar** — ein token-geschützter lokaler Kanal steuert die laufende GUI (typisierte Pipeline-Befehle sowie generische Widget-, Menü-, Maus- und Tastatursteuerung), mit Skill, 14-Prüfungen-Smoke-Runner und vollständigem Menüaktions-Durchlauf.
- **Verarbeitungsvorlagen** — Fusions-, Registrierungs- und Ressourceneinstellungen ohne Quellpfade als `.oftemplate` speichern und auf jeden geladenen Stapel anwenden; der Dialog akzeptiert auch Projekte und projektbasierte Rezepte.
- **Projekte und Wiederherstellung** — `.ofproj`-Sitzungen, automatische Absturzwiederherstellung, Rückgängig/Wiederholen für Stapelbearbeitungen und Beschriftungen.
- **Zwölf UI-Sprachen** — English, 中文, 日本語, Español, Deutsch, Français, Русский, 한국어, Português, Italiano, Nederlands, Polski. Automatisch aus der Systemsprache erkannt; eine neue Sprache ist eine einzige Datei `locales/<code>.py`.
- **Speicherschonend** — das neuronale Netz wird nur beim neuronalen Verfahren geladen; eine Sitzung mit klassischem Verfahren ruht bei rund 110 MB statt 260 MB.

<a id="install"></a>
## ⬇️ Installation

Laden Sie den aktuellen Windows- oder macOS-Build von der [OpenFocus-Website](https://openfocus.top/#download). Dieselben Dateien und ältere Versionen gibt es auch unter [GitHub Releases](https://github.com/Xinzhe99/OpenFocus/releases).

| Plattform | Datei | Typ |
|----------|------|------|
| Windows 10/11 (64 Bit) | `OpenFocus-*-setup.exe` | **Installer** (empfohlen) |
| Windows 10/11 (64 Bit) | `OpenFocus-*-windows-x64.zip` | Portabel (entpacken und starten) |
| macOS Apple Silicon (M1–M4) | `OpenFocus-*-macos.dmg` | Disk-Image (in „Programme“ ziehen) |
| macOS Apple Silicon (M1–M4) | `OpenFocus-*-macos-arm64.zip` | Portabel (entpacken) |

Die App ist nicht signiert, daher kann der erste Start einen Sicherheitshinweis zeigen: **Windows** SmartScreen → „Weitere Informationen“ → „Trotzdem ausführen“ (der erste Start kann 30–60 s dauern); **macOS** Rechtsklick → Öffnen, oder `xattr -cr /Applications/OpenFocus.app` im Terminal.

Für die Installation im Terminal prüfen Sie zuerst den Befehl für macOS oder Linux mit `--dry-run` und führen ihn dann ohne diese Option aus. Unter Linux installiert das die Headless-CLI; macOS-Builds erfordern Apple Silicon und werden nach `~/Applications` installiert.

```bash
curl -fsSL https://openfocus.top/install.sh | sh -s -- --dry-run
curl -fsSL https://openfocus.top/install.sh | sh
```

Unter Windows führen Sie den Installer aus PowerShell aus:

```powershell
irm https://openfocus.top/install.ps1 | iex
```

Das Windows-Setup direkt herunterladen:

```bash
curl -fL https://openfocus.top/download/latest/windows-setup -o OpenFocus-setup.exe
```

### 🤖 Agent-Integration

Die Desktop-App startet automatisch einen token-geschützten lokalen Agent-Kanal. Verwenden Sie `openfocus ctl` für typisierte Befehle oder generische UI-Aktionen gegen das laufende Fenster. Agenten können Stapel laden, rendern, exportieren, Dialoge öffnen und ausfüllen, Dateiauswahl-Dialoge beantworten, Maus- und Tastaturereignisse senden und Screenshots aufnehmen.

- Skill: [`skills/openfocus-control/SKILL.md`](../../skills/openfocus-control/SKILL.md)
- Protokoll und Befehlsreferenz: [`docs/AGENT_CONTROL_SPEC.md`](../AGENT_CONTROL_SPEC.md)
- End-to-End-Runner: `python tools/agent_gui_smoke.py --launch`
- Durchlauf aller Aktionen: `python tools/agent_action_sweep.py --launch`

### 💻 Kommandozeile

Alternativ die CLI unter Windows, macOS und Linux installieren – dieselbe Pipeline, ohne GUI-Stack:

```bash
pip install openfocus
openfocus -i ./stack_folder -o ./result/fused.png
```

KI-Modell, Gewichte, HEIC- und Kamera-RAW-Dekodierung stecken in dieser einen Installation. Unter Python 3.13+ entfällt `dtcwt`; dort bleibt der DTCWT-Eintrag mit Erklärung deaktiviert.

Retusche-Kontrollpunkte lassen sich headless mit dem ursprünglichen fusionierten Basisbild und den exakt verarbeiteten Quellbildern in ihrer aufgezeichneten Reihenfolge wiedergeben:

```bash
openfocus retouch replay edits.json --base fused.png --sources processed1.png processed2.png --output retouched.tif
```

Der Kontrollpunkt prüft Basis, Quellpixel, Geometrie und Gültigkeitsmasken. Exportieren Sie die verarbeiteten Bilder selbst (einschließlich Ausrichtung oder ROI-Zuschnitt), nicht den unverarbeiteten Eingabestapel. Die Wiedergabe lehnt inkompatible Kontexte ab und überschreibt ihre Eingaben nie. Die Ausgabe ist PNG oder TIFF und behält 8 oder 16 Bit bei.

Rezepte erfassen Einstellungen der Headless-Verarbeitung zur Wiederverwendung und zum Verschieben:

```bash
openfocus recipe create workflow.json -i ./stack_folder -o ./result/fused.png -m dct --cpu
openfocus recipe run workflow.json --dry-run
openfocus recipe run workflow.json
```

Das JSON-Format der Version 1 enthält `version`, ein `input`-Array, entweder `output` oder `output_dir` und ein `options`-Objekt. Pfade sind relativ zur Rezeptdatei; verschieben Sie das Rezept zusammen mit seinen Eingabedateien, um diese Beziehungen zu erhalten. Das Erstellen übernimmt die CLI-Einstellungen und die aufgelösten Algorithmusparameter. Unbekannte Felder, ungültige Werte und nicht unterstützte Versionen werden abgelehnt. Ein Probelauf prüft die Eingabe, ohne Ergebnisdateien zu schreiben. Bestehende Projektdateien bleiben unverändert.

Um Einstellungen aus einem vorhandenen `.ofproj` zu übernehmen, wählen Sie ausdrücklich die neue Eingabe und Ausgabe. Die ursprünglichen Projektquellen müssen zum Extrahieren der Einstellungen nicht vorhanden sein:

```bash
openfocus recipe create workflow.json --project session.ofproj -i ./new_stack -o ./result/fused.png
```

Das überträgt Fusion, Registrierung, Helligkeit, Reihenfolge, Schärfung und Kacheleinstellungen in das CLI-Rezept und bettet die Verarbeitungseinstellungen für die GUI ein. Bildausschlüsse, Ladeskalierung, Beschriftungen und Bearbeitungen werden nicht übertragen; bereiten Sie den neuen Eingabestapel entsprechend vor. Projekte mit Quelltransformationen, einschließlich angewendeter Flat-Field-Korrekturen, müssen zuerst ihren korrigierten Quellstapel exportieren. Rezepte verlangen Ausgaben außerhalb ihrer Eingabeordner und akzeptieren niemals eine Eingabedatei als Ausgabe. Gruppierungsoptionen lassen sich aus CLI-Argumenten übernehmen.

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

<a id="community"></a>
## Community & Support

- **Dokumentation** — [Benutzerhandbuch](../USER_MANUAL_DE.md) (zwölf Sprachen) · [Änderungsprotokoll](../../CHANGELOG.md) · [Leistung](../PERFORMANCE.md) · [Fusionsverfahren](../FUSION_METHODS.md)
- **Zitation** — [CITATION.cff](../../CITATION.cff); jede Version wird auf Zenodo archiviert ([DOI](https://doi.org/10.5281/zenodo.23050823))
- **Feedback** — eröffnen Sie ein [Issue](https://github.com/Xinzhe99/OpenFocus/issues) für Fehler, Ideen und Algorithmusarbeit
- **Danksagung** — jedes Fusions- und Registrierungsverfahren behält seine ursprüngliche Forschungslizenz; StackMFF V3 liegt unter [Xinzhe99/StackMFF-V3](https://github.com/Xinzhe99/StackMFF-V3)
- **Sicherheitsmeldungen** — [@ranas-mukminov](https://github.com/ranas-mukminov) meldete das Berechtigungsfenster des Discovery-Tokens ([PR #9](https://github.com/Xinzhe99/OpenFocus/pull/9))

<a id="license"></a>
## Lizenz

Die OpenFocus-Anwendung steht unter der [MIT-Lizenz](../../LICENSE).

**Ausnahme für die StackMFF-Serie.** Die StackMFF-Fusionsalgorithmen – derzeit StackMFF V3 und alle künftigen Ausgaben auf dem Stand der Technik – bleiben Eigentum des OpenFocus-Projekts, das alle Rechte an der StackMFF-Serie behält. Sie dürfen sie als Teil von OpenFocus für eigene Bilder und interne Arbeit verwenden. Das Kopieren, Portieren oder Migrieren dieser Algorithmen in ein anderes Open-Source-Projekt erfordert vorherige schriftliche Genehmigung.

<a id="developing"></a>
## Entwicklung

Aus dem Checkout starten:

```bash
conda create -n openfocus python=3.10
conda activate openfocus
pip install -r requirements.txt
python main.py
```

Paketierung mit PyInstaller: [docs/BUILD_COMMANDS.md](../BUILD_COMMANDS.md).

Ein Fusions- oder Registrierungsverfahren hinzufügen: [CONTRIBUTING.md](../../CONTRIBUTING.md).
Die [Erweiterungsspezifikation](../ALGORITHM_EXTENSIBILITY_SPEC.md) definiert die gemeinsamen Verträge für Registrierung, Parameter, Ausführung und Persistenz.

<a id="star-history"></a>
## ⭐ Star-Verlauf

[![Star History Chart](https://api.star-history.com/svg?repos=Xinzhe99/OpenFocus&type=Date)](https://star-history.com/#Xinzhe99/OpenFocus&Date)
