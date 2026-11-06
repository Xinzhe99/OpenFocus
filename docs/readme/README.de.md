<p align="center">
  <a href="https://github.com/Xinzhe99/OpenFocus">
    <img src="../../assets/OpenFocus.png" alt="OpenFocus" width="128">
  </a>
</p>

<h1 align="center">OpenFocus</h1>

<p align="center"><strong>Jedes Pixel aus seinem schärfsten Bild.</strong></p>

<p align="center">
  Zehn Fusionsverfahren, fünf Ausrichtungsoptionen (einschließlich keiner Ausrichtung) und eine quellengebundene Retusche für Makro-, Mikroskopie- und Industriestapel. 16 Bit durchgängig, eine Headless-CLI und eine Oberfläche in zwölf Sprachen.
</p>

<p align="center">
  <a href="https://github.com/Xinzhe99/OpenFocus/releases"><img src="https://img.shields.io/github/v/release/Xinzhe99/OpenFocus?label=release&color=2f81f7" alt="Release"></a>
  <a href="https://github.com/Xinzhe99/OpenFocus/stargazers"><img src="https://img.shields.io/github/stars/Xinzhe99/OpenFocus?color=e3b341" alt="Stars"></a>
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
> **v1.77.0 Release Candidate** — Automatischer Hintergrund und wiederherstellbare Retuschesitzungen, begrenzte JPEG/RAW-Vorschauen und Prüfung der Originalpixel, CPU-Fusion nach RAM-Budget und ein klarerer GPU-Ausführungsstatus. Unterstützte Eingaben und Einschränkungen stehen im Änderungsprotokoll.

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

StackMFF V3 kombiniert mit KI die scharfen Bereiche der Quellbilder.

[StackMFF V3 →](https://github.com/Xinzhe99/StackMFF-V3) · [Benchmark →](../../tools/synthetic_benchmark.py)

</td>
<td width="54%">

<a href="../../assets/readme/ai.mp4"><picture><source srcset="../../assets/readme/ai.gif" type="image/gif"><img src="../../assets/readme/ai_poster.jpg" alt="StackMFF V3 gegen Guided Filter auf demselben Stapel" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Wischvergleich

Vergleichen Sie Quellbild und Fusionsergebnis mit einem verschiebbaren Trenner.

[Handbuch →](../USER_MANUAL_DE.md)

</td>
<td width="54%">

<a href="../../assets/readme/wipe.mp4"><picture><source srcset="../../assets/readme/wipe.gif" type="image/gif"><img src="../../assets/readme/wipe_poster.jpg" alt="A/B-Wischvergleich zwischen Quellbild und Fusionsergebnis" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Methodenvergleich

Vergleichen Sie Verfahren nebeneinander und übernehmen Sie das Ergebnis oder seine Einstellungen.

</td>
<td width="54%">

<a href="../../assets/readme/method-comparison.mp4"><picture><source srcset="../../assets/readme/method-comparison.gif" type="image/gif"><img src="../../assets/readme/method-comparison_poster.jpg" alt="Acht Fusionsverfahren im direkten Vergleich" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Tiefenkarte &amp; 3D

Erstellen Sie Tiefenkarten und exportieren Sie texturierte 3D-Oberflächen.

</td>
<td width="54%">

<a href="../../assets/readme/depth.mp4"><picture><source srcset="../../assets/readme/depth.gif" type="image/gif"><img src="../../assets/readme/depth_poster.jpg" alt="Tiefenkarte und 3D-Oberfläche desselben Stapels" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Quellengebundene Retusche

Retuschieren Sie das Ergebnis mit Pixeln aus den Quellbildern. Der automatische Hintergrund ersetzt einen unruhigen gestackten Hintergrund durch ein echtes Einzelbild.

</td>
<td width="54%">

<a href="../../assets/readme/retouch.mp4"><picture><source srcset="../../assets/readme/retouch.gif" type="image/gif"><img src="../../assets/readme/retouch_poster.jpg" alt="Retusche mit Quellbild und Ergebnis" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Stapelverarbeitung

Verarbeiten Sie mehrere Stapel mit optionaler Gruppenfusion für begrenzten Speicherbedarf.

</td>
<td width="54%">

<a href="../../assets/readme/batch.mp4"><picture><source srcset="../../assets/readme/batch.gif" type="image/gif"><img src="../../assets/readme/batch_poster.jpg" alt="Dialog für Stapelverarbeitung" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Maßstabsbalken für Publikationen

Fügen Sie anhand von TIFF-Metadaten oder manuell kalibrierte Maßstabsbalken zu Exporten hinzu.

[Handbuch →](../USER_MANUAL_DE.md)

</td>
<td width="54%">

<a href="../../assets/readme/scalebar.mp4"><picture><source srcset="../../assets/readme/scalebar.gif" type="image/gif"><img src="../../assets/readme/scalebar_poster.jpg" alt="Maßstabsbalken-Dialog mit Live-Vorschau" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Beschriftungen und Legenden

Verwalten Sie Beschriftungen und Legenden mit eigenem Geltungsbereich und Stil.

</td>
<td width="54%">

<a href="../../assets/readme/labels.mp4"><picture><source srcset="../../assets/readme/labels.gif" type="image/gif"><img src="../../assets/readme/labels_poster.jpg" alt="Beschriftungsverwaltung mit Live-Vorschau" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Agent-steuerbar

Steuern Sie die Desktop-App über einen token-geschützten lokalen Agent-Kanal.

[Protokoll →](../AGENT_CONTROL_SPEC.md) · [Skill →](../../skills/openfocus-control/SKILL.md)

</td>
<td width="54%">

<a href="../../assets/readme/agent.mp4"><picture><source srcset="../../assets/readme/agent.gif" type="image/gif"><img src="../../assets/readme/agent_poster.jpg" alt="Ein Agent steuert die laufende GUI: laden, KI-Rendern, vergleichen, retuschieren, wischen, Tiefe, Beschriftungen" width="100%" /></picture></a>

</td>
</tr>
</table>

**Außerdem enthalten:**

- **Fokus-Stacking-Workflow** — Stapelbrowser, Schärfekurve, Schnellvorschau, ROI-Rendern und gespeicherte Einstellungen.
- **Mikroskopiebereit** — 16 Bit, TIFF-Kalibrierung, Flat-Field-Korrektur und mehrseitige Z-Stapel.
- **Farbprofile** — Einzelbild-Exporte behalten gemeinsame RGB-ICC-Profile ohne Farbkonvertierung.
- **Transparentes PNG** — kompatible Stapel mit rechteckigen Masken in voller Größe fusionieren und exportieren.
- **Headless-CLI** — `openfocus` mit Batch-Modus, `--dry-run`, `--resume` und JSON-Ausgabe.
- **Verarbeitungsvorlagen** — Einstellungen als `.oftemplate` speichern und erneut anwenden; Projekte und Rezepte werden unterstützt.
- **Projekte und Wiederherstellung** — `.ofproj`-Sitzungen, Absturzwiederherstellung und Rückgängig/Wiederholen.
- **Zwölf UI-Sprachen** — automatische Erkennung; neue Sprachen ergänzen Sie in `locales/<code>.py`.
- **Speicherschonend** — das neuronale Netz wird nur bei Bedarf geladen.

<a id="install"></a>
## ⬇️ Installation

Laden Sie den aktuellen Windows- oder macOS-Build von der [OpenFocus-Website](https://openfocus.top/#download). Dieselben Dateien und ältere Versionen gibt es auch unter [GitHub Releases](https://github.com/Xinzhe99/OpenFocus/releases).

Hinweise zur Installation und Fehlerbehebung stehen im [Handbuch](../USER_MANUAL_DE.md).

Unter Linux installiert der Terminalbefehl die Headless-CLI; macOS erfordert Apple Silicon. Die DMG-App wird nach `/Applications`, die Terminal-Version nach `~/Applications` installiert.

```bash
curl -fsSL https://openfocus.top/install.sh | sh -s -- --dry-run
curl -fsSL https://openfocus.top/install.sh | sh
```

Unter Windows führen Sie den Installer aus PowerShell aus:

```powershell
irm https://openfocus.top/install.ps1 | iex
```

Das Windows-Setup direkt herunterladen:

```powershell
curl.exe -fL https://openfocus.top/download/latest/windows-setup -o OpenFocus-setup.exe
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

Siehe das [CLI-Handbuch](../USER_MANUAL_DE.md#14-kommandozeilenschnittstelle).

<a id="community"></a>
## Community & Support

Fragen, Ideen oder Feedback? Besuchen Sie unsere [Discord-Community](https://discord.gg/afzmwgPt). Wir antworten so schnell wie möglich; wir nutzen die Pekinger Zeit (UTC+8), daher kann es durch die Zeitverschiebung zu Verzögerungen kommen.

- **Dokumentation** — [Benutzerhandbuch](../USER_MANUAL_DE.md) (zwölf Sprachen) · [Änderungsprotokoll](../../CHANGELOG.md) · [Leistung](../PERFORMANCE.md) · [Fusionsverfahren](../FUSION_METHODS.md)
- **Zitation** — [CITATION.cff](../../CITATION.cff); jede Version wird auf Zenodo archiviert ([DOI](https://doi.org/10.5281/zenodo.23050823))
- **Feedback** — eröffnen Sie ein [Issue](https://github.com/Xinzhe99/OpenFocus/issues) für Fehler, Ideen und Algorithmusarbeit
- **Danksagung** — jedes Fusions- und Registrierungsverfahren behält seine ursprüngliche Forschungslizenz; StackMFF V3 liegt unter [Xinzhe99/StackMFF-V3](https://github.com/Xinzhe99/StackMFF-V3)

<a id="license"></a>
## Lizenz

Die OpenFocus-Anwendung steht unter der [MIT-Lizenz](../../LICENSE).

**StackMFF-Lizenzen.** Die neuen V1/V2-Gewichte und Wrapper unterliegen der StackMFF V3 License 1.0. Kopieren, Weitergabe, Einbindung in andere Projekte und kommerzielle Nutzung erfordern die vorherige schriftliche Genehmigung des Rechteinhabers. Der ursprünglich eingebundene StackMFF-V3-Snapshot v2.0 steht weiterhin unter MIT. Einzelheiten: [Hinweise zu Drittanbietern](../../docs/THIRD_PARTY_NOTICES.md).

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
