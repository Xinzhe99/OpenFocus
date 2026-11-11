<p align="center">
  <a href="https://github.com/Xinzhe99/OpenFocus">
    <img src="../../assets/OpenFocus.png" alt="OpenFocus" width="128">
  </a>
</p>

<h1 align="center">OpenFocus</h1>

<p align="center"><strong>Elke pixel uit zijn scherpste frame.</strong></p>

<p align="center">
  Tien fusiemethoden, vijf uitlijningsopties (inclusief geen uitlijning) en een bronvaste retoucheerwerkruimte voor macro-, microscopie- en industriële stapels. 16-bit van begin tot eind, een headless CLI en een interface in twaalf talen.
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
> **[v1.77.0-beta.7 voorlopige release](https://github.com/Xinzhe99/OpenFocus/releases/tag/v1.77.0-beta.7)** — GPU-versnelling voor compatibele OpenCL-apparaten in OpenFocus B met terugval op de CPU, een gecorrigeerde veegvergelijking, automatische achtergrond en retouche. Zie het [wijzigingslogboek](../../CHANGELOG.md) voor details.

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

StackMFF V3 combineert met AI de scherpe delen van de bronframes.

[StackMFF V3 →](https://github.com/Xinzhe99/StackMFF-V3) · [Benchmark →](../../tools/synthetic_benchmark.py)

</td>
<td width="54%">

<a href="../../assets/readme/ai.mp4"><picture><source srcset="../../assets/readme/ai.gif" type="image/gif"><img src="../../assets/readme/ai_poster.jpg" alt="StackMFF V3 tegen Guided Filter op dezelfde stapel" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Veegvergelijking

Vergelijk een bronframe en het fusieresultaat met een versleepbare scheiding.

[Handleiding →](../USER_MANUAL_NL.md)

</td>
<td width="54%">

<a href="../../assets/readme/wipe.mp4"><picture><source srcset="../../assets/readme/wipe.gif" type="image/gif"><img src="../../assets/readme/wipe_poster.jpg" alt="A/B-veegvergelijking tussen bronframe en fusieresultaat" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Methodenvergelijking

Vergelijk methoden naast elkaar en neem het resultaat of de instellingen over.

</td>
<td width="54%">

<a href="../../assets/readme/method-comparison.mp4"><picture><source srcset="../../assets/readme/method-comparison.gif" type="image/gif"><img src="../../assets/readme/method-comparison_poster.jpg" alt="Acht fusiemethoden naast elkaar" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Dieptekaart en 3D

Maak dieptekaarten en exporteer getextureerde 3D-oppervlakken.

</td>
<td width="54%">

<a href="../../assets/readme/depth.mp4"><picture><source srcset="../../assets/readme/depth.gif" type="image/gif"><img src="../../assets/readme/depth_poster.jpg" alt="Dieptekaart en 3D-oppervlak van dezelfde stapel" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Bronvaste retouche

Retoucheer het resultaat met pixels uit de bronframes. De automatische achtergrond vervangt een onrustige gestapelde achtergrond door één echt frame.

</td>
<td width="54%">

<a href="../../assets/readme/retouch.mp4"><picture><source srcset="../../assets/readme/retouch.gif" type="image/gif"><img src="../../assets/readme/retouch_poster.jpg" alt="Retoucheerwerkruimte met bronframe en resultaat" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Batchverwerking

Verwerk meerdere stapels met optionele groepsfusie voor begrensd geheugengebruik.

</td>
<td width="54%">

<a href="../../assets/readme/batch.mp4"><picture><source srcset="../../assets/readme/batch.gif" type="image/gif"><img src="../../assets/readme/batch_poster.jpg" alt="Dialoog voor batchverwerking" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Schaalbalken voor publicaties

Voeg schaalbalken toe aan exports, gekalibreerd met TIFF-metadata of handmatig.

[Handleiding →](../USER_MANUAL_NL.md)

</td>
<td width="54%">

<a href="../../assets/readme/scalebar.mp4"><picture><source srcset="../../assets/readme/scalebar.gif" type="image/gif"><img src="../../assets/readme/scalebar_poster.jpg" alt="Schaalbalkdialoog met live voorbeeld" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Labels en bijschriften

Beheer labels en bijschriften met een eigen bereik en stijl.

</td>
<td width="54%">

<a href="../../assets/readme/labels.mp4"><picture><source srcset="../../assets/readme/labels.gif" type="image/gif"><img src="../../assets/readme/labels_poster.jpg" alt="Labelbeheer met live voorbeeld" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Aanstuurbare agent

Bestuur de desktopapp via een tokenbeveiligd lokaal agentkanaal.

[Protocol →](../AGENT_CONTROL_SPEC.md) · [Skill →](../../skills/openfocus-control/SKILL.md)

</td>
<td width="54%">

<a href="../../assets/readme/agent.mp4"><picture><source srcset="../../assets/readme/agent.gif" type="image/gif"><img src="../../assets/readme/agent_poster.jpg" alt="Een agent bestuurt de live GUI: laden, AI-render, vergelijken, retoucheren, vegen, diepte, labels" width="100%" /></picture></a>

</td>
</tr>
</table>

**Ook inbegrepen:**

- **Focus-stackingworkflow** — stapelbrowser, scherptecurve, snel voorbeeld, ROI-render en opgeslagen instellingen.
- **Klaar voor microscopie** — 16-bit, TIFF-kalibratie, flat-fieldcorrectie en meerbladige Z-stapels.
- **Kleurprofielen** — export van losse afbeeldingen behoudt gedeelde RGB-ICC-profielen zonder kleurconversie.
- **Transparante PNG** — compatibele stapels met rechthoekige maskers op volledige grootte samenvoegen en exporteren.
- **Headless CLI** — `openfocus` met batchmodus, `--dry-run`, `--resume` en JSON-uitvoer.
- **Verwerkingssjablonen** — instellingen als `.oftemplate` opslaan en hergebruiken; projecten en recepten worden ondersteund.
- **Projecten en herstel** — `.ofproj`-sessies, crashherstel en ongedaan maken/opnieuw uitvoeren.
- **Twaalf UI-talen** — automatische detectie; voeg nieuwe talen toe in `locales/<code>.py`.
- **Zuinig met geheugen** — het neurale netwerk wordt alleen geladen wanneer nodig.

<a id="install"></a>
## ⬇️ Installatie

De [OpenFocus-website](https://openfocus.top/#download) en installatiescripts bieden voor Windows en macOS de nieuwste versie, inclusief een nieuwere bèta (momenteel v1.77.0-beta.7). Downloads staan ook op [GitHub Releases](https://github.com/Xinzhe99/OpenFocus/releases). Automatische desktopupdates bieden ook nieuwere bèta’s; de Linux-CLI van PyPI blijft stabiel.

Zie de [handleiding](../USER_MANUAL_NL.md) voor installatie en probleemoplossing.

Op Linux installeert de terminalopdracht de headless CLI; macOS vereist Apple Silicon. De DMG-app komt in `/Applications`, de terminalinstallatie in `~/Applications`.

```bash
curl -fsSL https://openfocus.top/install.sh | sh -s -- --dry-run
curl -fsSL https://openfocus.top/install.sh | sh
```

Voer op Windows de installer uit vanuit PowerShell:

```powershell
irm https://openfocus.top/install.ps1 | iex
```

De Windows-setup rechtstreeks downloaden:

```powershell
curl.exe -fL https://openfocus.top/download/latest/windows-setup -o OpenFocus-setup.exe
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

Vragen, ideeën of feedback? Word lid van onze [Discord-community](https://discord.gg/afzmwgPt). We reageren zo snel mogelijk; we gebruiken de tijd van Beijing (UTC+8), dus het tijdsverschil kan de reactietijd beïnvloeden.

- **Documentatie** — [Gebruikershandleiding](../USER_MANUAL_NL.md) (twaalf talen) · [Wijzigingen](../../CHANGELOG.md) · [Prestaties](../PERFORMANCE.md) · [Fusiemethoden](../FUSION_METHODS.md)
- **Citeren** — [CITATION.cff](../../CITATION.cff); elke versie wordt gearchiveerd op Zenodo ([DOI](https://doi.org/10.5281/zenodo.23050823))
- **Feedback** — open een [issue](https://github.com/Xinzhe99/OpenFocus/issues) voor bugs, ideeën en algoritmewerk
- **Credits** — elk fusie- en registratiealgoritme behoudt zijn oorspronkelijke onderzoekslicentie; StackMFF V3 staat op [Xinzhe99/StackMFF-V3](https://github.com/Xinzhe99/StackMFF-V3)

<a id="license"></a>
## Licentie

De OpenFocus-toepassing valt onder de [MIT-licentie](../../LICENSE).

**StackMFF-licenties.** De nieuwe V1/V2-gewichten en wrappers vallen onder StackMFF V3 License 1.0. Kopiëren, verspreiden, opnemen in andere projecten en commercieel gebruik vereisen voorafgaande schriftelijke toestemming van de rechthebbende. De oorspronkelijk meegeleverde StackMFF V3-snapshot v2.0 blijft onder MIT vallen. Zie de [kennisgevingen van derden](../../docs/THIRD_PARTY_NOTICES.md).

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
