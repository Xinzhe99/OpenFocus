<p align="center">
  <a href="https://github.com/Xinzhe99/OpenFocus">
    <img src="assets/OpenFocus.png" alt="OpenFocus" width="128">
  </a>
</p>

<h1 align="center">OpenFocus</h1>

<p align="center"><strong>Elk scherptevlak. Eén volledig scherp beeld.</strong></p>

<p align="center">
  Acht fusiemethoden, drie registratiemodi en een bronvaste retoucheerwerkruimte voor macro, microscopie en industriële stapels. 16-bit van begin tot eind, een headless CLI en een interface in twaalf talen.
</p>

<p align="center">
  <a href="https://github.com/Xinzhe99/OpenFocus/releases"><img src="https://img.shields.io/github/v/release/Xinzhe99/OpenFocus?label=release&color=2f81f7" alt="Release"></a>
  <a href="https://github.com/Xinzhe99/OpenFocus/stargazers"><img src="https://img.shields.io/github/stars/Xinzhe99/OpenFocus?color=e3b341" alt="Stars"></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/license-MIT-3fb950" alt="License: MIT"></a>
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
  <a href="./docs/USER_MANUAL_NL.md">Documentatie</a> ·
  <a href="./CHANGELOG.md">Wijzigingen</a> ·
  <a href="https://doi.org/10.5281/zenodo.23050823">DOI</a>
</p>

<p align="center">
  <a href="README.md">English</a> ·
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
  <img src="assets/ui.png" alt="OpenFocus-interface" width="880">
</p>

## 📢 News

> [!NOTE]
> **v1.69.0** — de fusion-engine plant zijn werk in plaats van het te gokken: één uitvoeringsplan per render bepaalt workers, threads en geheugenbudget, fysieke cores worden van het platform gelezen in plaats van het logische aantal te halveren, en elke klassieke back-end streamt zijn tussenbuffers in plaats van tijdelijke buffers voor de hele stapel vast te houden. De uitvoer blijft byte-identiek, met gemeten winst van 1,47× op OpenFocus C en 1,33× op de neurale methode. Een nieuwe **resourcemodus** (Gebalanceerd / Responsief / Maximaal) houdt de desktop vloeiend zonder het beeld te wijzigen.

<a id="features"></a>
## Functies

<table>

<tr>
<td width="46%" valign="middle">

### AI-fusie: StackMFF V3

De standaardmethode classificeert elke pixel met een neuraal netwerk naar het meest waarschijnlijke bronframe. Op dezelfde uitgelijnde stapel blijven letterranden scherper en vlakken schoner dan bij Guided Filter (Laplaciaanse scherpte 27,1 tegen 28,5); tegen synthetische ground truth leidt het op PSNR en SSIM, met minder spookpixels dan welk klassiek algoritme ook.

[StackMFF V3 →](https://github.com/Xinzhe99/StackMFF-V3) · [Benchmark →](tools/synthetic_benchmark.py)

</td>
<td width="54%">

<a href="assets/readme/ai.mp4"><picture><source srcset="assets/readme/ai.gif" type="image/gif"><img src="assets/readme/ai_poster.jpg" alt="AI-fusie: StackMFF V3" width="100%" /></picture></a>

</td>
</tr>

<tr>
<td width="46%" valign="middle">

### A/B-wipe

Sleep de scheiding tussen een bronframe en het resultaat om te zien wat de fusie veranderde: uitlijningsdrift, halo's langs randen en verloren detail, bij elke zoomstand.

[Handleiding →](./docs/USER_MANUAL_NL.md)

</td>
<td width="54%">

<a href="assets/readme/wipe.mp4"><picture><source srcset="assets/readme/wipe.gif" type="image/gif"><img src="assets/readme/wipe_poster.jpg" alt="A/B-vergelijking tussen bronframe en resultaat" width="100%" /></picture></a>

</td>
</tr>

<tr>
<td width="46%" valign="middle">

### Methodenvergelijking

Render alle methoden één keer, vergelijk ze naast elkaar op dezelfde uitsnede en neem de winnaar over als nieuwe uitvoer of kopieer de parameters terug naar het hoofdvenster.

</td>
<td width="54%">

<a href="assets/readme/method-comparison.mp4"><picture><source srcset="assets/readme/method-comparison.gif" type="image/gif"><img src="assets/readme/method-comparison_poster.jpg" alt="Acht fusiemethoden naast elkaar" width="100%" /></picture></a>

</td>
</tr>

<tr>
<td width="46%" valign="middle">

### Dieptekaart en 3D

Focuspositie per pixel uit vijf focusmaten, wetenschappelijke kleurenschalen, fysieke Z-kalibratie uit TIFF-metadata en getextureerde OBJ-export van het oppervlak.

</td>
<td width="54%">

<a href="assets/readme/depth.mp4"><picture><source srcset="assets/readme/depth.gif" type="image/gif"><img src="assets/readme/depth_poster.jpg" alt="Dieptekaart en 3D-oppervlak van dezelfde stapel" width="100%" /></picture></a>

</td>
</tr>

<tr>
<td width="46%" valign="middle">

### Bronvaste retouche

Schilder vanuit de frames die het resultaat echt hebben gemaakt: gedeelde zoom en pan, bronframe-schuif, zachte penseel, herstelpenseel, undo per streek en bescherming van niet-geëxporteerde pixels.

</td>
<td width="54%">

<a href="assets/readme/retouch.mp4"><picture><source srcset="assets/readme/retouch.gif" type="image/gif"><img src="assets/readme/retouch_poster.jpg" alt="Retoucheerwerkruimte met bronframe en resultaat" width="100%" /></picture></a>

</td>
</tr>

<tr>
<td width="46%" valign="middle">

### Batchverwerking

Wijs een map met stapels aan, kies methode en bestemming en laat het lopen. Alleen uitlijnen schrijft één map per stapel; fusie schrijft en toont alleen het gefuseerde beeld.

</td>
<td width="54%">

<a href="assets/readme/batch.mp4"><picture><source srcset="assets/readme/batch.gif" type="image/gif"><img src="assets/readme/batch_poster.jpg" alt="Dialoog voor batchverwerking" width="100%" /></picture></a>

</td>
</tr>

<tr>
<td width="46%" valign="middle">

### Schaalbalken voor publicatie

Gekalibreerd uit TIFF-metadata of handmatig, getekend met het echte lettertype op de echte lengte en ingebrand in elke export: losse afbeelding, stapel, TIFF, GIF en batch. Past hij niet leesbaar, dan zegt de dialoog dat in plaats van een kortere balk met een verkeerd label te tekenen.

[Handleiding →](./docs/USER_MANUAL_NL.md)

</td>
<td width="54%">

<a href="assets/readme/scalebar.mp4"><picture><source srcset="assets/readme/scalebar.gif" type="image/gif"><img src="assets/readme/scalebar_poster.jpg" alt="Schaalbalkdialoog met live voorbeeld" width="100%" /></picture></a>

</td>
</tr>

<tr>
<td width="46%" valign="middle">

### Labels en bijschriften

Bijschriften blijven een beheerde lijst: elk met een eigen bereik (bron, uitgelijnde stapel of een specifieke uitvoer), framebereik en stijl. Voorbeeld en elke export gebruiken hetzelfde doel en hetzelfde framenummer, dus `{value}` leest identiek in Opslaan, Opslaan als, slepen en Alles exporteren.

</td>
<td width="54%">

<a href="assets/readme/labels.mp4"><picture><source srcset="assets/readme/labels.gif" type="image/gif"><img src="assets/readme/labels_poster.jpg" alt="Labels en bijschriften" width="100%" /></picture></a>

</td>
</tr>
</table>

**Ook inbegrepen:**

- **Focus-stackingworkflow** — stapelbrowser, scherptecurve, snelle voorvertoning, ROI-render en per methode onthouden instellingen na herstart.
- **Klaar voor microscopie** — 16-bit van begin tot eind, µm/px uit TIFF, flat-fieldcorrectie, meerbladige TIFF-Z-stacks.
- **Headless CLI** — dezelfde pijplijn als `openfocus`, met batch, `--dry-run`, `--resume` en JSON-uitvoer.
- **Projecten en herstel** — `.ofproj`-sessies, automatisch herstel na een crash, undo/redo voor stapels en annotaties.
- **Twaalf UI-talen** — English, 简体中文, 日本語, Español, Deutsch, Français, Русский, 한국어, Português, Italiano, Nederlands, Polski; automatisch gedetecteerd, een nieuwe taal is één `locales_<code>.py`-bestand.
- **Zuining met geheugen** — de neurale gewichten laden alleen bij de AI-methode, dus een klassieke sessie blijft rond 110 MB in plaats van 260 MB.

<a id="install"></a>
## ⬇️ Installatie

Geen installatie nodig — haal een build van de [Releases](https://github.com/Xinzhe99/OpenFocus/releases)-pagina:

| Platform | Bestand | Type |
|----------|---------|------|
| Windows 10/11 (64-bit) | `OpenFocus-*-setup.exe` | **Installer** (aanbevolen) |
| Windows 10/11 (64-bit) | `OpenFocus-*-windows-x64.zip` | Draagbaar (uitpakken en starten) |
| macOS Apple Silicon (M1–M4) | `OpenFocus-*-macos.dmg` | Schijfimage (naar Programma's slepen) |
| macOS Apple Silicon (M1–M4) | `OpenFocus-*-macos-arm64.zip` | Draagbaar (uitpakken) |

De app is niet ondertekend, dus de eerste start kan een waarschuwing tonen: **Windows** SmartScreen → 'Meer informatie' → 'Toch uitvoeren' (de eerste start kan 30–60 s duren); **macOS** rechtsklik → Open, of `xattr -cr /Applications/OpenFocus.app` in Terminal.

### 💻 Opdrachtregel

Of installeer de CLI op Windows, macOS en Linux — dezelfde pijplijn, zonder GUI-stack:

```bash
pip install openfocus
openfocus -i ./stack_folder -o ./result/fused.png
```

Het AI-model, de gewichten, HEIC- en camera-RAW-decodering zitten allemaal in die ene installatie. Op Python 3.13+ vervalt `dtcwt` en blijft de DTCWT-optie uitgeschakeld met uitleg.

<a id="community"></a>
## Community en ondersteuning

- **Documentatie** — [Handleiding](./docs/USER_MANUAL_NL.md) (twaalf talen) · [Wijzigingen](./CHANGELOG.md) · [Prestaties](./docs/PERFORMANCE.md) · [Algoritmepariteit](./docs/ALGORITHM_PARITY.md)
- **Citeren** — [CITATION.cff](./CITATION.cff); elke versie wordt gearchiveerd op Zenodo ([DOI](https://doi.org/10.5281/zenodo.23050823))
- **Feedback** — open een [issue](https://github.com/Xinzhe99/OpenFocus/issues) voor bugs, ideeën en algoritmewerk
- **Credits** — elk fusie- en registratiealgoritme behoudt zijn oorspronkelijke onderzoekslicentie; StackMFF V3 staat op [Xinzhe99/StackMFF-V3](https://github.com/Xinzhe99/StackMFF-V3)

<a id="developing"></a>
## Ontwikkeling

Uit de checkout starten:

```bash
conda create -n openfocus python=3.10
conda activate openfocus
pip install -r requirements.txt
python main.py
```

Verpakken met PyInstaller: [docs/BUILD_COMMANDS.md](./docs/BUILD_COMMANDS.md).

<a id="star-history"></a>
## ⭐ Sterrengeschiedenis

[![Star History Chart](https://api.star-history.com/svg?repos=Xinzhe99/OpenFocus&type=Date)](https://star-history.com/#Xinzhe99/OpenFocus&Date)
