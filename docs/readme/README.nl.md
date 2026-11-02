<p align="center">
  <a href="https://github.com/Xinzhe99/OpenFocus">
    <img src="../../assets/OpenFocus.png" alt="OpenFocus" width="128">
  </a>
</p>

<h1 align="center">OpenFocus</h1>

<p align="center"><strong>Elke pixel uit zijn scherpste frame.</strong></p>

<p align="center">
  Tien fusiemethoden, vijf uitlijningsopties en een bronvaste retoucheerwerkruimte voor macro-, microscopie- en industriële stapels. 16-bit van begin tot eind, een headless CLI en een interface in twaalf talen.
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
  <a href="../USER_MANUAL_NL.md">Documentatie</a> ·
  <a href="../../CHANGELOG.md">Wijzigingen</a> ·
  <a href="https://doi.org/10.5281/zenodo.23050823">DOI</a>
</p>

<p align="center">
  <a href="../../README.md">English</a> ·
  <a href="README.zh-CN.md">简体中文</a> ·
  <a href="README.ja.md">日本語</a> ·
  <a href="README.es.md">Español</a> ·
  <a href="README.de.md">Deutsch</a> ·
  <a href="README.fr.md">Français</a> ·
  <a href="README.ru.md">Русский</a> ·
  <a href="README.ko.md">한국어</a> ·
  <a href="README.pt.md">Português</a> ·
  <a href="README.it.md">Italiano</a> ·
  <strong>Nederlands</strong> ·
  <a href="README.pl.md">Polski</a>
</p>

<p align="center">
  <img src="../../assets/ui.png" alt="OpenFocus-interface" width="880">
</p>

## 📢 News

> [!NOTE]
> **v1.74.0** — De OpenFocus A/B/C-fusie en snelle uitlijning zijn bijgewerkt. De desktopapp start nu automatisch een geauthenticeerd lokaal agentkanaal, zodat `openfocus ctl` het draaiende venster kan bedienen zonder speciale startoptie.

Oudere versies staan in het [wijzigingslogboek](../../CHANGELOG.md).

📸 Als je afbeeldingen publiceert die met OpenFocus zijn gemaakt, stellen we een vermelding zoals deze op prijs:

> Gemaakt met OpenFocus — https://github.com/Xinzhe99/OpenFocus

Dit is niet verplicht, maar wel zeer gewaardeerd.

<a id="features"></a>
## Functies

<table>
<tr>
<td width="46%" valign="middle">

### AI-fusie: StackMFF V3

De standaardmethode gebruikt een neuraal netwerk om voor elke pixel het waarschijnlijkste bronframe te schatten en mengt vervolgens de voorspelde bijdragen of kiest het waarschijnlijkste frame. De synthetische benchmark vergelijkt methoden met een bekende referentie; de resultaten hangen af van de stapel en de instellingen.

[StackMFF V3 →](https://github.com/Xinzhe99/StackMFF-V3) · [Benchmark →](../../tools/synthetic_benchmark.py)

</td>
<td width="54%">

<a href="../../assets/readme/ai.mp4"><picture><source srcset="../../assets/readme/ai.gif" type="image/gif"><img src="../../assets/readme/ai_poster.jpg" alt="StackMFF V3 tegen Guided Filter op dezelfde stapel" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Veegvergelijking

Sleep de scheiding over een bronframe en het fusieresultaat om precies te zien wat de fusie veranderde: uitlijningsdrift, halo's langs randen en verloren detail zijn in één oogopslag zichtbaar, bij elke zoomstand.

[Handleiding →](../USER_MANUAL_NL.md)

</td>
<td width="54%">

<a href="../../assets/readme/wipe.mp4"><picture><source srcset="../../assets/readme/wipe.gif" type="image/gif"><img src="../../assets/readme/wipe_poster.jpg" alt="A/B-veegvergelijking tussen bronframe en fusieresultaat" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Methodenvergelijking

Render elke methode één keer, vergelijk ze naast elkaar op dezelfde uitsnede en neem de winnaar over als nieuwe uitvoer of kopieer de parameters terug naar het hoofdvenster.

</td>
<td width="54%">

<a href="../../assets/readme/method-comparison.mp4"><picture><source srcset="../../assets/readme/method-comparison.gif" type="image/gif"><img src="../../assets/readme/method-comparison_poster.jpg" alt="Acht fusiemethoden naast elkaar" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Dieptekaart en 3D

Focuspositie per pixel uit vijf focusmaten, wetenschappelijke kleurenschalen, fysieke Z-kalibratie uit TIFF-metadata en getextureerde OBJ-export van het oppervlak.

</td>
<td width="54%">

<a href="../../assets/readme/depth.mp4"><picture><source srcset="../../assets/readme/depth.gif" type="image/gif"><img src="../../assets/readme/depth_poster.jpg" alt="Dieptekaart en 3D-oppervlak van dezelfde stapel" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Bronvaste retouche

Schilder vanuit de frames die het resultaat echt hebben gemaakt: gedeelde zoom en pan, schuifregelaar voor het bronframe, zachte penseel, herstelpenseel, ongedaan maken per streek en bescherming van niet-geëxporteerde pixels.
Pas de retouche toe als nieuwe uitvoer en kies daarna in het hoofdvenster met rechtsklik op die uitvoer **TIFF met lagen exporteren** om het resultaat samen met zijn bronframes op te slaan. Live labels en schaalbalken worden niet meegenomen. GIMP 3.2.6 op Windows opent het resultaat en de bronlagen; andere editors zijn niet getest.

</td>
<td width="54%">

<a href="../../assets/readme/retouch.mp4"><picture><source srcset="../../assets/readme/retouch.gif" type="image/gif"><img src="../../assets/readme/retouch_poster.jpg" alt="Retoucheerwerkruimte met bronframe en resultaat" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Batchverwerking

Wijs een map met stapels aan, kies een methode en bestemming en laat het lopen. Alleen-uitlijnen schrijft één map per stapel; fusieregels schrijven het gefuseerde beeld en tonen precies dat. Optionele groepsfusie ruilt exacte gelijkwaardigheid met de volledige stapel in voor begrensd groepsgeheugen en tijdelijke schijfruimte.

</td>
<td width="54%">

<a href="../../assets/readme/batch.mp4"><picture><source srcset="../../assets/readme/batch.gif" type="image/gif"><img src="../../assets/readme/batch_poster.jpg" alt="Dialoog voor batchverwerking" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Schaalbalken voor publicaties

Gekalibreerd uit TIFF-metadata of handmatig ingevoerd, getekend met het echte lettertype op de echte lengte en ingebrand in elke export: losse afbeeldingen, stapels, meerbladige TIFF, GIF en batchuitvoer. Een balk die niet leesbaar past, meldt dat in plaats van een lengte af te drukken die hij niet tekent.

[Handleiding →](../USER_MANUAL_NL.md)

</td>
<td width="54%">

<a href="../../assets/readme/scalebar.mp4"><picture><source srcset="../../assets/readme/scalebar.gif" type="image/gif"><img src="../../assets/readme/scalebar_poster.jpg" alt="Schaalbalkdialoog met live voorbeeld" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Labels en bijschriften

Labels blijven een beheerde lijst; elk label heeft een eigen bereik (bron, uitgelijnde stapel of een specifieke uitvoer), framebereik en stijl. Voorbeeld en elke export gebruiken hetzelfde doel en hetzelfde framenummer, zodat `{value}` identiek leest in Opslaan, Opslaan als, slepen en Alles exporteren.

</td>
<td width="54%">

<a href="../../assets/readme/labels.mp4"><picture><source srcset="../../assets/readme/labels.gif" type="image/gif"><img src="../../assets/readme/labels_poster.jpg" alt="Labelbeheer met live voorbeeld" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Aanstuurbare agent

Elk venster, elke widget en elk menu is bereikbaar via een tokenbeveiligd lokaal kanaal: een agent laadt een stapel, rendert, retoucheert, vergelijkt, berekent diepte, exporteert en beantwoordt dialoogen — via getypte opdrachten, widgetpaden of echte muis- en toetsgebeurtenissen. Geen screen scraping of OCR nodig; het kanaal start met de desktopapp.

[Protocol →](../AGENT_CONTROL_SPEC.md) · [Skill →](../../skills/openfocus-control/SKILL.md)

</td>
<td width="54%">

<a href="../../assets/readme/agent.mp4"><picture><source srcset="../../assets/readme/agent.gif" type="image/gif"><img src="../../assets/readme/agent_poster.jpg" alt="Een agent bestuurt de live GUI: laden, AI-render, vergelijken, retoucheren, vegen, diepte, labels" width="100%" /></picture></a>

</td>
</tr>
</table>

**Ook inbegrepen:**

- **Focus-stackingworkflow** — stapelbrowser, scherptecurve, snel voorbeeld, ROI-render en per methode onthouden instellingen na een herstart.
- **Klaar voor microscopie** — 16-bit van begin tot eind, µm/px uit TIFF-metadata, flat-fieldcorrectie met een stofkaart, meerbladige TIFF-Z-stapels, getextureerde OBJ-export.
- **Kleurprofielen** — export van losse afbeeldingen als PNG, TIFF, JPEG en WebP behoudt een gedeeld RGB-ICC-profiel wanneer elk bronframe hetzelfde profiel heeft. Gemengde of ongetagde stapels krijgen geen profiel; pixels worden niet kleurgeconverteerd.
- **Transparante PNG** — stapels met overeenkomende rechthoekige maskers met constante dekking kunnen op volledige grootte worden gefuseerd en als PNG worden opgeslagen. Beperkingen staan in de handleiding.
- **Headless CLI** — dezelfde pijplijn als `openfocus`, met batchmodus, `--dry-run`, `--resume` en JSON-uitvoer voor scripts.
- **Aanstuurbare agent** — een tokenbeveiligd lokaal kanaal bestuurt de live GUI (getypte pijplijnopdrachten plus generieke widget-, menu-, muis- en toetsbesturing), met een skill, een smoke-runner met 14 controles en een volledige menuhandelingenscan.
- **Verwerkingssjablonen** — fusie-, registratie- en resource-instellingen zonder bronpaden opslaan als `.oftemplate` en op elke geladen stapel toepassen; de kiezer accepteert ook projecten en projectgebaseerde recepten.
- **Projecten en herstel** — `.ofproj`-sessies, automatisch herstel na een crash, ongedaan maken/opnieuw uitvoeren voor stapelbewerkingen en annotaties.
- **Twaalf UI-talen** — English, 中文, 日本語, Español, Deutsch, Français, Русский, 한국어, Português, Italiano, Nederlands, Polski. Automatisch gedetecteerd uit de systeemtaal; een nieuwe taal is één bestand `locales/<code>.py`.
- **Zuinig met geheugen** — het neurale netwerk wordt alleen geladen als de neurale methode wordt gebruikt; een sessie met een klassieke methode blijft rond 110 MB in plaats van 260 MB.

<a id="install"></a>
## ⬇️ Installatie

Download de nieuwste Windows- of macOS-build van de [OpenFocus-website](https://openfocus.top/#download). Dezelfde bestanden en oudere versies staan ook op [GitHub Releases](https://github.com/Xinzhe99/OpenFocus/releases).

| Platform | Bestand | Type |
|----------|---------|------|
| Windows 10/11 (64-bit) | `OpenFocus-*-setup.exe` | **Installer** (aanbevolen) |
| Windows 10/11 (64-bit) | `OpenFocus-*-windows-x64.zip` | Draagbaar (uitpakken en starten) |
| macOS Apple Silicon (M1–M4) | `OpenFocus-*-macos.dmg` | Schijfimage (naar Programma's slepen) |
| macOS Apple Silicon (M1–M4) | `OpenFocus-*-macos-arm64.zip` | Draagbaar (uitpakken) |

De app is niet ondertekend, dus de eerste start kan een beveiligingswaarschuwing tonen: **Windows** SmartScreen → 'Meer informatie' → 'Toch uitvoeren' (de eerste start kan 30–60 s duren); **macOS** rechtsklik → Open, of `xattr -cr /Applications/OpenFocus.app` in Terminal.

Voor installatie vanaf de terminal bekijk je eerst de opdracht voor macOS of Linux met `--dry-run` en voer je die daarna zonder die optie uit. Op Linux installeert dit de headless CLI; macOS-builds vereisen Apple Silicon en worden in `~/Applications` geïnstalleerd.

```bash
curl -fsSL https://openfocus.top/install.sh | sh -s -- --dry-run
curl -fsSL https://openfocus.top/install.sh | sh
```

Voer op Windows de installer uit vanuit PowerShell:

```powershell
irm https://openfocus.top/install.ps1 | iex
```

De Windows-setup rechtstreeks downloaden:

```bash
curl -fL https://openfocus.top/download/latest/windows-setup -o OpenFocus-setup.exe
```

### 🤖 Agentintegratie

De desktopapp start automatisch een tokenbeveiligd lokaal agentkanaal. Gebruik `openfocus ctl` voor getypte opdrachten of generieke UI-acties op het draaiende venster. Agenten kunnen stapels laden, renderen, exporteren, dialoogen openen en invullen, bestandskiezers beantwoorden, muis- en toetsgebeurtenissen sturen en schermafbeeldingen maken.

- Skill: [`skills/openfocus-control/SKILL.md`](../../skills/openfocus-control/SKILL.md)
- Protocol en opdrachtenreferentie: [`docs/AGENT_CONTROL_SPEC.md`](../AGENT_CONTROL_SPEC.md)
- End-to-end-runner: `python tools/agent_gui_smoke.py --launch`
- Scan van elke actie: `python tools/agent_action_sweep.py --launch`

### 💻 Opdrachtregel

Of installeer de CLI op Windows, macOS en Linux — dezelfde pijplijn, zonder GUI-stack:

```bash
pip install openfocus
openfocus -i ./stack_folder -o ./result/fused.png
```

Zie de [CLI-handleiding](../USER_MANUAL_NL.md#14-opdrachtregelinterface).

<a id="community"></a>
## Community en ondersteuning

- **Documentatie** — [Gebruikershandleiding](../USER_MANUAL_NL.md) (twaalf talen) · [Wijzigingen](../../CHANGELOG.md) · [Prestaties](../PERFORMANCE.md) · [Fusiemethoden](../FUSION_METHODS.md)
- **Citeren** — [CITATION.cff](../../CITATION.cff); elke versie wordt gearchiveerd op Zenodo ([DOI](https://doi.org/10.5281/zenodo.23050823))
- **Feedback** — open een [issue](https://github.com/Xinzhe99/OpenFocus/issues) voor bugs, ideeën en algoritmewerk
- **Credits** — elk fusie- en registratiealgoritme behoudt zijn oorspronkelijke onderzoekslicentie; StackMFF V3 staat op [Xinzhe99/StackMFF-V3](https://github.com/Xinzhe99/StackMFF-V3)

<a id="license"></a>
## Licentie

De OpenFocus-toepassing valt onder de [MIT-licentie](../../LICENSE).

**Uitzondering voor de StackMFF-serie.** De StackMFF-fusiealgoritmen — momenteel StackMFF V3 en alle toekomstige edities op de nieuwste stand van de techniek — blijven eigendom van het OpenFocus-project, dat alle rechten op de StackMFF-serie behoudt. Je mag ze als onderdeel van OpenFocus gebruiken voor je eigen afbeeldingen en intern werk. Het kopiëren, porten of migreren van deze algoritmen naar een ander opensourceproject vereist voorafgaande schriftelijke toestemming.

<a id="developing"></a>
## Ontwikkeling

Start vanuit de checkout:

```bash
conda create -n openfocus python=3.10
conda activate openfocus
pip install -r requirements.txt
python main.py
```

Verpakken met PyInstaller: [docs/BUILD_COMMANDS.md](../BUILD_COMMANDS.md).

Een fusie- of registratiealgoritme toevoegen: [CONTRIBUTING.md](../../CONTRIBUTING.md).
De [uitbreidingsspecificatie](../ALGORITHM_EXTENSIBILITY_SPEC.md) definieert de gedeelde contracten voor registratie, parameters, uitvoering en persistentie.

<a id="star-history"></a>
## ⭐ Sterrengeschiedenis

[![Star History Chart](https://api.star-history.com/svg?repos=Xinzhe99/OpenFocus&type=Date)](https://star-history.com/#Xinzhe99/OpenFocus&Date)
