<p align="center">
  <a href="https://github.com/Xinzhe99/OpenFocus">
    <img src="../../assets/OpenFocus.png" alt="OpenFocus" width="128">
  </a>
</p>

<h1 align="center">OpenFocus</h1>

<p align="center"><strong>Ogni pixel, dal suo fotogramma più nitido.</strong></p>

<p align="center">
  Dieci algoritmi di fusione, quattro modalità di registrazione e uno spazio di ritocco legato alla sorgente per macro, microscopia e stack industriali. 16 bit end-to-end, una CLI senza interfaccia e un'interfaccia in dodici lingue.
</p>

<p align="center">
  <a href="https://github.com/Xinzhe99/OpenFocus/releases"><img src="https://img.shields.io/github/v/release/Xinzhe99/OpenFocus?label=release&color=2f81f7" alt="Versione"></a>
  <a href="https://github.com/Xinzhe99/OpenFocus/stargazers"><img src="https://img.shields.io/github/stars/Xinzhe99/OpenFocus?color=e3b341" alt="Stelle"></a>
  <a href="../../LICENSE"><img src="https://img.shields.io/badge/license-MIT-3fb950" alt="Licenza: MIT"></a>
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
> **v1.74.0** — Fusione OpenFocus A/B/C e allineamento rapido aggiornati. L'app desktop ora avvia automaticamente il proprio canale agente locale autenticato, quindi `openfocus ctl` può controllare la finestra in esecuzione senza flag di avvio speciali.

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

Il metodo predefinito usa una rete neurale per stimare il fotogramma sorgente più probabile per ogni pixel, quindi miscela i contributi previsti oppure sceglie il fotogramma più probabile. Il benchmark sintetico confronta i metodi con un riferimento noto; i risultati dipendono dallo stack e dalle impostazioni.

[StackMFF V3 →](https://github.com/Xinzhe99/StackMFF-V3) · [Benchmark →](../../tools/synthetic_benchmark.py)

</td>
<td width="54%">

<a href="../../assets/readme/ai.mp4"><picture><source srcset="../../assets/readme/ai.gif" type="image/gif"><img src="../../assets/readme/ai_poster.jpg" alt="StackMFF V3 contro il filtro guidato sullo stesso stack" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Confronto Wipe (a tendina)

Trascina il divisore tra un fotogramma e il risultato fuso per vedere esattamente cosa ha cambiato la miscelazione: deriva di allineamento, aloni sui bordi e dettaglio perso, a qualsiasi zoom.

[Manuale →](../USER_MANUAL_IT.md)

</td>
<td width="54%">

<a href="../../assets/readme/wipe.mp4"><picture><source srcset="../../assets/readme/wipe.gif" type="image/gif"><img src="../../assets/readme/wipe_poster.jpg" alt="Confronto A/B tra un fotogramma sorgente e il risultato fuso" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Confronto metodi

Renderizza ogni metodo una volta, confrontali affiancati sullo stesso ritaglio, poi adotta il migliore come nuova uscita o copia i suoi parametri nella finestra principale.

</td>
<td width="54%">

<a href="../../assets/readme/method-comparison.mp4"><picture><source srcset="../../assets/readme/method-comparison.gif" type="image/gif"><img src="../../assets/readme/method-comparison_poster.jpg" alt="Confronto affiancato di otto metodi di fusione" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Mappa di profondità e 3D

Posizione di messa a fuoco per pixel con cinque misure, palette scientifiche, calibrazione Z fisica dai metadati TIFF ed esportazione OBJ texturizzata della superficie.

</td>
<td width="54%">

<a href="../../assets/readme/depth.mp4"><picture><source srcset="../../assets/readme/depth.gif" type="image/gif"><img src="../../assets/readme/depth_poster.jpg" alt="Mappa di profondità e superficie 3D dello stesso stack" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Ritocco legato alla sorgente

Dipingi dai fotogrammi che hanno davvero prodotto il risultato: zoom e panoramica condivisi, cursore del fotogramma sorgente, pennello morbido, pennello di ripristino, annullamento per tratto e protezione dei pixel non esportati. Applica il ritocco come nuova uscita, poi **Esporta TIFF a livelli** per salvarla con i fotogrammi sorgente. Etichette e barre di scala attive non sono incluse. GIMP 3.2.6 su Windows apre il risultato e i livelli sorgente; altri editor non sono stati testati.

</td>
<td width="54%">

<a href="../../assets/readme/retouch.mp4"><picture><source srcset="../../assets/readme/retouch.gif" type="image/gif"><img src="../../assets/readme/retouch_poster.jpg" alt="Spazio di ritocco con fotogramma sorgente e risultato" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Elaborazione batch

Indica una cartella di stack, scegli metodo e destinazione, poi lascia lavorare. Il solo allineamento scrive una cartella per stack; la fusione scrive l'immagine fusa e mostra solo quella. La fusione raggruppata opzionale scambia l'equivalenza esatta con lo stack completo con memoria e spazio temporaneo limitati.

</td>
<td width="54%">

<a href="../../assets/readme/batch.mp4"><picture><source srcset="../../assets/readme/batch.gif" type="image/gif"><img src="../../assets/readme/batch_poster.jpg" alt="Finestra di elaborazione batch" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Barre di scala per pubblicazione

Calibrate dai metadati TIFF o inserite a mano, disegnate con font e lunghezza reali e impresse in ogni esportazione: immagini singole, stack, TIFF multipagina, GIF e batch. Una barra che non può essere resa leggibile lo segnala invece di stampare una lunghezza che non disegna.

[Manuale →](../USER_MANUAL_IT.md)

</td>
<td width="54%">

<a href="../../assets/readme/scalebar.mp4"><picture><source srcset="../../assets/readme/scalebar.gif" type="image/gif"><img src="../../assets/readme/scalebar_poster.jpg" alt="Finestra della barra di scala con anteprima" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Didascalie ed etichette

Mantieni un elenco gestito di didascalie, ciascuna con il proprio ambito (stack sorgente, stack allineato o un'uscita specifica), intervallo di fotogrammi e stile. Anteprima ed esportazioni usano la stessa destinazione e lo stesso numero di fotogramma, quindi `{value}` si legge allo stesso modo in Salva, Salva con nome, trascinamento ed Esporta tutto.

</td>
<td width="54%">

<a href="../../assets/readme/labels.mp4"><picture><source srcset="../../assets/readme/labels.gif" type="image/gif"><img src="../../assets/readme/labels_poster.jpg" alt="Gestore delle etichette con anteprima" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Controllabile da un agente

Ogni finestra, widget e menu è raggiungibile tramite un canale locale protetto da token: un agente carica uno stack, renderizza, ritocca, confronta, calcola la profondità, esporta e risponde ai dialoghi tramite comandi tipizzati, percorsi di widget o eventi reali di mouse e tastiera. Non servono cattura schermo o OCR; il canale si avvia con l'app desktop.

[Protocollo →](../AGENT_CONTROL_SPEC.md) · [Skill →](../../skills/openfocus-control/SKILL.md)

</td>
<td width="54%">

<a href="../../assets/readme/agent.mp4"><picture><source srcset="../../assets/readme/agent.gif" type="image/gif"><img src="../../assets/readme/agent_poster.jpg" alt="Un agente pilota l'interfaccia dal vivo" width="100%" /></picture></a>

</td>
</tr>
</table>

**In dotazione anche:**

- **Flusso focus stacking** — browser degli stack, curva di nitidezza, anteprima rapida, render ROI e impostazioni per metodo memorizzate tra un riavvio e l'altro.
- **Pronto per la microscopia** — 16 bit end-to-end, µm/px dai metadati TIFF, correzione di campo piatto con mappa polvere, stack Z TIFF multipagina ed esportazione OBJ texturizzata.
- **Profili colore** — le esportazioni singole PNG, TIFF, JPEG e WebP conservano un profilo ICC RGB condiviso quando tutti i fotogrammi sorgente hanno lo stesso profilo. Stack misti o senza profilo vengono esportati senza profilo; i pixel non vengono convertiti tra spazi colore.
- **PNG trasparente** — gli stack con maschere rettangolari a opacità costante corrispondenti possono essere fusi a dimensione piena e salvati come PNG. Vedi le restrizioni nel manuale.
- **CLI senza interfaccia** — la stessa pipeline come `openfocus`, con modalità batch, `--dry-run`, `--resume` e output JSON per gli script.
- **Controllabile da un agente** — un canale locale protetto da token guida l'interfaccia in tempo reale (comandi pipeline tipizzati e controllo generico di widget, menu, mouse e tastiera), con una skill, un esecutore smoke a 14 controlli e una scansione completa delle azioni di menu.
- **Progetti e recupero** — sessioni `.ofproj`, recupero automatico dopo crash, annulla/ripeti su modifiche allo stack e annotazioni.
- **Dodici lingue dell'interfaccia** — English, 中文, 日本語, Español, Deutsch, Français, Русский, 한국어, Português, Italiano, Nederlands, Polski. Rilevate dalle impostazioni locali del sistema; una nuova lingua è un file `locales/<code>.py`.
- **Leggero in memoria** — la rete neurale viene caricata solo quando si usa il metodo IA, quindi una sessione con metodi classici resta intorno ai 110 MB invece di 260 MB.

<a id="install"></a>
## ⬇️ Installazione

Scarica l'ultima build per Windows o macOS dal [sito di OpenFocus](https://openfocus.top/#download). Gli stessi file e le versioni precedenti sono disponibili anche su [GitHub Releases](https://github.com/Xinzhe99/OpenFocus/releases).

| Piattaforma | File | Tipo |
|------------|---------|------|
| Windows 10/11 (64 bit) | `OpenFocus-*-setup.exe` | **Programma di installazione** (consigliato) |
| Windows 10/11 (64 bit) | `OpenFocus-*-windows-x64.zip` | Portatile (scompatta ed esegui) |
| macOS Apple Silicon (M1–M4) | `OpenFocus-*-macos.dmg` | Immagine disco (trascina in Applicazioni) |
| macOS Apple Silicon (M1–M4) | `OpenFocus-*-macos-arm64.zip` | Portatile (scompatta) |

L'app non è firmata, quindi il primo avvio può mostrare un avviso di sicurezza: **Windows** SmartScreen → «Ulteriori informazioni» → «Esegui comunque» (il primo avvio può richiedere 30–60 s); **macOS** clic destro → Apri, oppure `xattr -cr /Applications/OpenFocus.app` nel Terminale.

Per installare da terminale, visualizza prima l'anteprima del comando per macOS o Linux, poi eseguilo senza `--dry-run`. Su Linux installa la CLI senza interfaccia; le build macOS richiedono Apple Silicon e vengono installate in `~/Applications`.

```bash
curl -fsSL https://openfocus.top/install.sh | sh -s -- --dry-run
curl -fsSL https://openfocus.top/install.sh | sh
```

Su Windows, esegui il programma di installazione da PowerShell:

```powershell
irm https://openfocus.top/install.ps1 | iex
```

Il download diretto del programma di installazione Windows è disponibile anche con
`curl -fL https://openfocus.top/download/latest/windows-setup -o OpenFocus-setup.exe`.

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

- **Documentazione** — [Manuale](../USER_MANUAL_IT.md) (dodici lingue) · [Registro modifiche](../../CHANGELOG.md) · [Prestazioni](../PERFORMANCE.md) · [Metodi di fusione](../FUSION_METHODS.md)
- **Citazione** — [CITATION.cff](../../CITATION.cff); ogni versione è archiviata su Zenodo ([DOI](https://doi.org/10.5281/zenodo.23050823))
- **Feedback** — apri una [issue](https://github.com/Xinzhe99/OpenFocus/issues) per bug, idee e lavoro sugli algoritmi
- **Crediti** — ogni algoritmo di fusione e registrazione conserva la propria licenza di ricerca originale; StackMFF V3 si trova su [Xinzhe99/StackMFF-V3](https://github.com/Xinzhe99/StackMFF-V3)

<a id="license"></a>
## Licenza

L'applicazione OpenFocus è distribuita con [licenza MIT](../../LICENSE).

**Eccezione della serie StackMFF.** Gli algoritmi di fusione StackMFF — attualmente StackMFF V3 e ogni futura edizione allo stato dell'arte — restano di proprietà del progetto OpenFocus, che ne detiene tutti i diritti. Puoi usarli all'interno di OpenFocus per le tue immagini e il tuo lavoro interno. Copiare, portare o migrare questi algoritmi in un altro progetto open source richiede un'autorizzazione scritta preventiva.

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
