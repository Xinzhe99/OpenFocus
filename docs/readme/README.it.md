<p align="center">
  <a href="https://github.com/Xinzhe99/OpenFocus">
    <img src="../../assets/OpenFocus.png" alt="OpenFocus" width="128">
  </a>
</p>

<h1 align="center">OpenFocus</h1>

<p align="center"><strong>Ogni pixel, dal suo fotogramma più nitido.</strong></p>

<p align="center">
  Dieci algoritmi di fusione, cinque opzioni di allineamento (inclusa nessuna) e uno spazio di ritocco legato alla sorgente per macro, microscopia e stack industriali. 16 bit end-to-end, una CLI senza interfaccia e un'interfaccia in dodici lingue.
</p>

<p align="center">
  <a href="https://github.com/Xinzhe99/OpenFocus/releases"><img src="https://img.shields.io/github/v/release/Xinzhe99/OpenFocus?label=release&color=2f81f7" alt="Versione"></a>
  <a href="https://github.com/Xinzhe99/OpenFocus/stargazers"><img src="https://img.shields.io/github/stars/Xinzhe99/OpenFocus?color=e3b341" alt="Stelle"></a>
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
  <a href="../USER_MANUAL_IT.md">Documentazione</a> ·
  <a href="../../CHANGELOG.md">Registro modifiche</a> ·
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
  <strong>Italiano</strong> ·
  <a href="README.nl.md">Nederlands</a> ·
  <a href="README.pl.md">Polski</a>
</p>

<p align="center">
  <img src="../../assets/ui.png" alt="Interfaccia di OpenFocus" width="880">
</p>

## 📢 Novità

> [!NOTE]
> **v1.77.0 candidata al rilascio** — Sfondo automatico e sessioni di ritocco ripristinabili, anteprime JPEG/RAW con memoria limitata e ispezione dei pixel originali, fusione CPU in base alla RAM e stato di esecuzione GPU più chiaro. Consulta il registro delle modifiche per input supportati e limitazioni.

Le versioni precedenti sono nel [registro delle modifiche](../../CHANGELOG.md).

📸 Se pubblichi immagini create con OpenFocus, valuta una nota come:

> Creato con OpenFocus — https://github.com/Xinzhe99/OpenFocus

Non è obbligatorio, ma è molto apprezzato.

<a id="features"></a>
## Funzionalità

<table>
<tr>
<td width="46%" valign="middle">

### Fusione IA: StackMFF V3

StackMFF V3 usa l'IA per combinare le parti nitide dei fotogrammi di uno stack.

[StackMFF V3 →](https://github.com/Xinzhe99/StackMFF-V3) · [Benchmark →](../../tools/synthetic_benchmark.py)

</td>
<td width="54%">

<a href="../../assets/readme/ai.mp4"><picture><source srcset="../../assets/readme/ai.gif" type="image/gif"><img src="../../assets/readme/ai_poster.jpg" alt="StackMFF V3 contro il filtro guidato sullo stesso stack" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Confronto Wipe (a tendina)

Trascina il divisore per confrontare un fotogramma sorgente con il risultato fuso.

[Manuale →](../USER_MANUAL_IT.md)

</td>
<td width="54%">

<a href="../../assets/readme/wipe.mp4"><picture><source srcset="../../assets/readme/wipe.gif" type="image/gif"><img src="../../assets/readme/wipe_poster.jpg" alt="Confronto A/B tra un fotogramma sorgente e il risultato fuso" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Confronto metodi

Confronta i metodi affiancati e riutilizza il risultato o le impostazioni che preferisci.

</td>
<td width="54%">

<a href="../../assets/readme/method-comparison.mp4"><picture><source srcset="../../assets/readme/method-comparison.gif" type="image/gif"><img src="../../assets/readme/method-comparison_poster.jpg" alt="Confronto affiancato di otto metodi di fusione" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Mappa di profondità e 3D

Crea una mappa di profondità ed esporta una superficie 3D texturizzata.

</td>
<td width="54%">

<a href="../../assets/readme/depth.mp4"><picture><source srcset="../../assets/readme/depth.gif" type="image/gif"><img src="../../assets/readme/depth_poster.jpg" alt="Mappa di profondità e superficie 3D dello stesso stack" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Ritocco legato alla sorgente

Ritocca il risultato dipingendo dai fotogrammi sorgente ed esportalo insieme a questi in un TIFF a livelli. Lo sfondo automatico sostituisce uno sfondo impilato irregolare con un fotogramma reale.

</td>
<td width="54%">

<a href="../../assets/readme/retouch.mp4"><picture><source srcset="../../assets/readme/retouch.gif" type="image/gif"><img src="../../assets/readme/retouch_poster.jpg" alt="Spazio di ritocco con fotogramma sorgente e risultato" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Elaborazione batch

Elabora più stack alla volta, con fusione raggruppata opzionale per limitare memoria e spazio temporaneo.

</td>
<td width="54%">

<a href="../../assets/readme/batch.mp4"><picture><source srcset="../../assets/readme/batch.gif" type="image/gif"><img src="../../assets/readme/batch_poster.jpg" alt="Finestra di elaborazione batch" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Barre di scala per pubblicazione

Aggiungi barre di scala calibrate dai metadati TIFF o inserite manualmente.

[Manuale →](../USER_MANUAL_IT.md)

</td>
<td width="54%">

<a href="../../assets/readme/scalebar.mp4"><picture><source srcset="../../assets/readme/scalebar.gif" type="image/gif"><img src="../../assets/readme/scalebar_poster.jpg" alt="Finestra della barra di scala con anteprima" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Didascalie ed etichette

Aggiungi didascalie ed etichette definendone ambito, intervallo di fotogrammi e stile.

</td>
<td width="54%">

<a href="../../assets/readme/labels.mp4"><picture><source srcset="../../assets/readme/labels.gif" type="image/gif"><img src="../../assets/readme/labels_poster.jpg" alt="Gestore delle etichette con anteprima" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Controllabile da un agente

Un agente può controllare l'applicazione tramite un canale locale protetto da token.

[Protocollo →](../AGENT_CONTROL_SPEC.md) · [Skill →](../../skills/openfocus-control/SKILL.md)

</td>
<td width="54%">

<a href="../../assets/readme/agent.mp4"><picture><source srcset="../../assets/readme/agent.gif" type="image/gif"><img src="../../assets/readme/agent_poster.jpg" alt="Un agente pilota l'interfaccia dal vivo" width="100%" /></picture></a>

</td>
</tr>
</table>

**In dotazione anche:**

- **Focus stacking** — browser degli stack, curva di nitidezza, anteprima, render ROI e impostazioni salvate.
- **Microscopia** — 16 bit, calibrazione TIFF, correzione di campo piatto e stack Z TIFF multipagina.
- **Profili colore** — conserva il profilo ICC RGB comune nelle esportazioni singole compatibili.
- **PNG trasparente** — fusione di stack con maschere rettangolari compatibili; restrizioni nel manuale.
- **CLI senza interfaccia** — batch, `--dry-run`, `--resume` e output JSON con `openfocus`.
- **Progetti e recupero** — sessioni `.ofproj`, recupero automatico e annulla/ripeti.
- **Dodici lingue** — rilevamento automatico e aggiunta tramite file `locales/<code>.py`.
- **Memoria** — la rete neurale viene caricata solo quando si usa il metodo IA.

<a id="install"></a>
## ⬇️ Installazione

Scarica l'ultima build per Windows o macOS dal [sito di OpenFocus](https://openfocus.top/#download). Gli stessi file e le versioni precedenti sono disponibili anche su [GitHub Releases](https://github.com/Xinzhe99/OpenFocus/releases).

Consulta il [manuale](../USER_MANUAL_IT.md) per l'installazione e la risoluzione dei problemi.

Su macOS con Apple Silicon o su Linux, installa dal terminale; su Linux viene installata la CLI senza interfaccia e su macOS l'applicazione viene salvata in `~/Applications`:

```bash
curl -fsSL https://openfocus.top/install.sh | sh -s -- --dry-run # facoltativo: anteprima
curl -fsSL https://openfocus.top/install.sh | sh
```

Su Windows, esegui il programma di installazione da PowerShell:

```powershell
irm https://openfocus.top/install.ps1 | iex
```

Download diretto del programma di installazione Windows:

```powershell
curl.exe -fL https://openfocus.top/download/latest/windows-setup -o OpenFocus-setup.exe
```

### 🤖 Integrazione con agenti

L'app desktop avvia automaticamente un canale agente locale protetto da token. Usa `openfocus ctl` per comandi tipizzati o azioni UI generiche sulla finestra in tempo reale. Gli agenti possono caricare stack, renderizzare, esportare, aprire e compilare dialoghi, rispondere ai selettori di file, inviare eventi di mouse e tastiera e acquisire screenshot.

- Skill: [`skills/openfocus-control/SKILL.md`](../../skills/openfocus-control/SKILL.md)
- Protocollo e riferimento comandi: [`docs/AGENT_CONTROL_SPEC.md`](../AGENT_CONTROL_SPEC.md)
- Esecutore end-to-end: `python tools/agent_gui_smoke.py --launch`
- Scansione di tutte le azioni: `python tools/agent_action_sweep.py --launch`

### 💻 Uso da riga di comando

Puoi anche installare la CLI su Windows, macOS e Linux — la stessa pipeline, senza lo stack grafico:

```bash
pip install openfocus
openfocus -i ./stack_folder -o ./result/fused.png
```

Consulta il [manuale della CLI](../USER_MANUAL_IT.md#14-interfaccia-a-riga-di-comando).

<a id="community"></a>
## Community e supporto

Domande, idee o suggerimenti? Unisciti alla nostra [comunità Discord](https://discord.gg/afzmwgPt). Rispondiamo appena possibile; usiamo l'ora di Pechino (UTC+8), quindi il fuso orario può influire sui tempi di risposta.

- **Documentazione** — [Manuale](../USER_MANUAL_IT.md) (dodici lingue) · [Registro modifiche](../../CHANGELOG.md) · [Prestazioni](../PERFORMANCE.md) · [Metodi di fusione](../FUSION_METHODS.md)
- **Citazione** — [CITATION.cff](../../CITATION.cff); ogni versione è archiviata su Zenodo ([DOI](https://doi.org/10.5281/zenodo.23050823))
- **Feedback** — apri una [issue](https://github.com/Xinzhe99/OpenFocus/issues) per bug, idee e lavoro sugli algoritmi
- **Crediti** — ogni algoritmo di fusione e registrazione conserva la propria licenza di ricerca originale; StackMFF V3 si trova su [Xinzhe99/StackMFF-V3](https://github.com/Xinzhe99/StackMFF-V3)

<a id="license"></a>
## Licenza

L'applicazione OpenFocus è distribuita con [licenza MIT](../../LICENSE).

**Licenze StackMFF.** I nuovi pesi e wrapper V1/V2 sono soggetti alla StackMFF V3 License 1.0. Copia, distribuzione, integrazione in altri progetti e uso commerciale richiedono l'autorizzazione scritta preventiva del titolare dei diritti. Lo snapshot v2.0 originale di StackMFF V3 incluso nel progetto resta sotto licenza MIT. Vedi le [note sui componenti di terze parti](../../docs/THIRD_PARTY_NOTICES.md).

<a id="developing"></a>
## Sviluppo

Esegui da un clone:

```bash
conda create -n openfocus python=3.10
conda activate openfocus
pip install -r requirements.txt
python main.py
```

Packaging con PyInstaller: [docs/BUILD_COMMANDS.md](../BUILD_COMMANDS.md).

Aggiungere un algoritmo di fusione o registrazione: [CONTRIBUTING.md](../../CONTRIBUTING.md).
La [specifica di estensibilità](../ALGORITHM_EXTENSIBILITY_SPEC.md) definisce i contratti condivisi di registrazione, parametri, esecuzione e persistenza.

<a id="star-history"></a>
## ⭐ Cronologia delle stelle

[![Star History Chart](https://api.star-history.com/svg?repos=Xinzhe99/OpenFocus&type=Date)](https://star-history.com/#Xinzhe99/OpenFocus&Date)
