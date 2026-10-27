<p align="center">
  <a href="https://github.com/Xinzhe99/OpenFocus">
    <img src="assets/OpenFocus.png" alt="OpenFocus" width="128">
  </a>
</p>

<h1 align="center">OpenFocus</h1>

<p align="center"><strong>Tutti i piani di fuoco. Una sola immagine completamente nitida.</strong></p>

<p align="center">
  Otto algoritmi di fusione, tre modalità di registrazione e un'area di ritocco legata alla sorgente per macro, microscopia e stack industriali. 16 bit end-to-end, CLI headless e un'interfaccia in dodici lingue.
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
  <a href="./docs/USER_MANUAL_IT.md">Documentazione</a> ·
  <a href="./CHANGELOG.md">Novità</a> ·
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
  <strong>Italiano</strong> ·
  <a href="README.nl.md">Nederlands</a> ·
  <a href="README.pl.md">Polski</a>
</p>

<p align="center">
  <img src="assets/ui.png" alt="Interfaccia di OpenFocus" width="880">
</p>

## 📢 News

> [!NOTE]
> **v1.71.0** — gli algoritmi sono ora dichiarati in un registro immutabile: un nuovo metodo di fusione o registrazione compare in rendering, batch, confronto, profondità e CLI da un solo descrittore con parametri tipizzati. Ogni avvio scrive un proprio log rotante con ID attività collegati, parametri, tempi e traceback completi; lo ZIP di feedback contiene il rapporto di crash, i dati RAM/GPU e le versioni delle dipendenze. La diagnostica parte prima degli import di Qt e delle librerie numeriche, quindi anche un errore di avvio viene registrato. Gli otto algoritmi restano identici byte per byte alla v1.70.0.

<a id="features"></a>
## Funzionalità

<table>

<tr>
<td width="46%" valign="middle">

### Fusione IA: StackMFF V3

Il metodo predefinito classifica ogni pixel nel fotogramma sorgente più probabile con una rete neurale. Sulla stessa pila registrata mantiene i bordi delle lettere più nitidi e le superfici più pulite di Guided Filter (nitidezza laplaciana 27,1 contro 28,5) e guida tutti i metodi su PSNR e SSIM rispetto a una verità sintetica, con meno pixel fantasma di qualsiasi algoritmo classico.

[StackMFF V3 →](https://github.com/Xinzhe99/StackMFF-V3) · [Benchmark →](tools/synthetic_benchmark.py)

</td>
<td width="54%">

<a href="assets/readme/ai.mp4"><picture><source srcset="assets/readme/ai.gif" type="image/gif"><img src="assets/readme/ai_poster.jpg" alt="Fusione IA: StackMFF V3" width="100%" /></picture></a>

</td>
</tr>

<tr>
<td width="46%" valign="middle">

### Confronto A/B

Trascina il divisore tra un fotogramma e il risultato per vedere cosa ha cambiato la fusione: deriva di registrazione, aloni sui bordi e dettaglio perso, a qualsiasi zoom.

[Manuale →](./docs/USER_MANUAL_IT.md)

</td>
<td width="54%">

<a href="assets/readme/wipe.mp4"><picture><source srcset="assets/readme/wipe.gif" type="image/gif"><img src="assets/readme/wipe_poster.jpg" alt="Confronto A/B tra un fotogramma e il risultato" width="100%" /></picture></a>

</td>
</tr>

<tr>
<td width="46%" valign="middle">

### Confronto metodi

Renderizza tutti i metodi una volta, confrontali affiancati sullo stesso ritaglio e adotta il vincitore come nuova uscita o copia i suoi parametri nella finestra principale.

</td>
<td width="54%">

<a href="assets/readme/method-comparison.mp4"><picture><source srcset="assets/readme/method-comparison.gif" type="image/gif"><img src="assets/readme/method-comparison_poster.jpg" alt="Otto metodi di fusione affiancati" width="100%" /></picture></a>

</td>
</tr>

<tr>
<td width="46%" valign="middle">

### Mappa di profondità e 3D

Posizione di fuoco per pixel con cinque misure, mappe colore scientifiche, calibrazione Z fisica dai metadati TIFF ed esportazione OBJ texturizzata della superficie.

</td>
<td width="54%">

<a href="assets/readme/depth.mp4"><picture><source srcset="assets/readme/depth.gif" type="image/gif"><img src="assets/readme/depth_poster.jpg" alt="Mappa di profondità e superficie 3D della stessa pila" width="100%" /></picture></a>

</td>
</tr>

<tr>
<td width="46%" valign="middle">

### Ritocco legato alla sorgente

Dipingi dai fotogrammi che hanno davvero prodotto il risultato: zoom e pan condivisi, cursore del fotogramma, pennello morbido, pennello di ripristino, annulla per tratto e protezione dei pixel non esportati.

</td>
<td width="54%">

<a href="assets/readme/retouch.mp4"><picture><source srcset="assets/readme/retouch.gif" type="image/gif"><img src="assets/readme/retouch_poster.jpg" alt="Area di ritocco con fotogramma e risultato" width="100%" /></picture></a>

</td>
</tr>

<tr>
<td width="46%" valign="middle">

### Elaborazione batch

Indica una cartella di pile, scegli metodo e destinazione, poi lascia lavorare. Solo allineamento scrive una cartella per pila; la fusione scrive e mostra solo l'immagine fusa.

</td>
<td width="54%">

<a href="assets/readme/batch.mp4"><picture><source srcset="assets/readme/batch.gif" type="image/gif"><img src="assets/readme/batch_poster.jpg" alt="Finestra di elaborazione batch" width="100%" /></picture></a>

</td>
</tr>

<tr>
<td width="46%" valign="middle">

### Barre di scala per pubblicazione

Calibrate dai metadati TIFF o inserite a mano, disegnate con font e lunghezza reali e incise in ogni esportazione: immagine, pila, TIFF, GIF e batch. Se non entra in modo leggibile lo dice, invece di disegnare una barra più corta della sua etichetta.

[Manuale →](./docs/USER_MANUAL_IT.md)

</td>
<td width="54%">

<a href="assets/readme/scalebar.mp4"><picture><source srcset="assets/readme/scalebar.gif" type="image/gif"><img src="assets/readme/scalebar_poster.jpg" alt="Finestra della barra di scala con anteprima" width="100%" /></picture></a>

</td>
</tr>

<tr>
<td width="46%" valign="middle">

### Etichette e didascalie

Le didascalie sono un elenco gestito: ognuna con il proprio ambito (sorgente, pila allineata o un output specifico), intervallo di fotogrammi e stile. Anteprima ed esportazioni usano lo stesso target e lo stesso numero di fotogramma, quindi `{value}` si legge allo stesso modo in Salva, Salva con nome, trascinamento ed Esporta tutto.

</td>
<td width="54%">

<a href="assets/readme/labels.mp4"><picture><source srcset="assets/readme/labels.gif" type="image/gif"><img src="assets/readme/labels_poster.jpg" alt="Etichette e didascalie" width="100%" /></picture></a>

</td>
</tr>
</table>

**Inoltre:**

- **Flusso di focus stacking** — browser delle pile, curva di nitidezza, anteprima rapida, render ROI e impostazioni per metodo ricordate tra sessioni.
- **Pronto per la microscopia** — 16 bit end-to-end, µm/px da TIFF, correzione di campo piatto, Z-stack TIFF multipagina.
- **CLI headless** — la stessa pipeline come `openfocus`, con batch, `--dry-run`, `--resume` e output JSON.
- **Progetti e recupero** — sessioni `.ofproj`, recupero dopo crash, annulla/ripeti su pile e annotazioni.
- **Dodici lingue dell'interfaccia** — English, 简体中文, 日本語, Español, Deutsch, Français, Русский, 한국어, Português, Italiano, Nederlands, Polski; rilevate dal sistema, una lingua nuova è un file `locales_<code>.py`.
- **Leggero in memoria** — i pesi neurali si caricano solo usando il metodo AI, così una sessione classica resta intorno a 110 MB invece di 260 MB.

<a id="install"></a>
## ⬇️ Installazione

Nessuna configurazione — scarica una build dalla pagina [Releases](https://github.com/Xinzhe99/OpenFocus/releases):

| Piattaforma | File | Tipo |
|-------------|------|------|
| Windows 10/11 (64 bit) | `OpenFocus-*-setup.exe` | **Installer** (consigliato) |
| Windows 10/11 (64 bit) | `OpenFocus-*-windows-x64.zip` | Portabile (estrarre ed eseguire) |
| macOS Apple Silicon (M1–M4) | `OpenFocus-*-macos.dmg` | Immagine disco (trascinare in Applicazioni) |
| macOS Apple Silicon (M1–M4) | `OpenFocus-*-macos-arm64.zip` | Portabile (estrarre) |

L'app non è firmata, quindi il primo avvio può mostrare un avviso: **Windows** SmartScreen → «Ulteriori informazioni» → «Esegui comunque» (il primo avvio può richiedere 30–60 s); **macOS** clic destro → Apri, oppure `xattr -cr /Applications/OpenFocus.app` nel Terminale.

### 💻 Riga di comando

Oppure installa la CLI su Windows, macOS e Linux — la stessa pipeline, senza stack grafico:

```bash
pip install openfocus
openfocus -i ./stack_folder -o ./result/fused.png
```

Modello IA, pesi, decodifica HEIC e RAW sono tutti in questa singola installazione. Su Python 3.13+ `dtcwt` viene saltato e la voce DTCWT resta disattivata con una spiegazione.

<a id="community"></a>
## Community e supporto

- **Documentazione** — [Manuale](./docs/USER_MANUAL_IT.md) (dodici lingue) · [Modifiche](./CHANGELOG.md) · [Prestazioni](./docs/PERFORMANCE.md) · [Parità algoritmi](./docs/ALGORITHM_PARITY.md)
- **Citazione** — [CITATION.cff](./CITATION.cff); ogni versione è archiviata su Zenodo ([DOI](https://doi.org/10.5281/zenodo.23050823))
- **Feedback** — apri una [issue](https://github.com/Xinzhe99/OpenFocus/issues) per bug, idee e lavoro sugli algoritmi
- **Crediti** — ogni algoritmo di fusione e registrazione mantiene la licenza di ricerca originale; StackMFF V3 è su [Xinzhe99/StackMFF-V3](https://github.com/Xinzhe99/StackMFF-V3)

<a id="developing"></a>
## Sviluppo

Eseguire dal repository:

```bash
conda create -n openfocus python=3.10
conda activate openfocus
pip install -r requirements.txt
python main.py
```

Packaging con PyInstaller: [docs/BUILD_COMMANDS.md](./docs/BUILD_COMMANDS.md).

<a id="star-history"></a>
## ⭐ Cronologia stelle

[![Star History Chart](https://api.star-history.com/svg?repos=Xinzhe99/OpenFocus&type=Date)](https://star-history.com/#Xinzhe99/OpenFocus&Date)
