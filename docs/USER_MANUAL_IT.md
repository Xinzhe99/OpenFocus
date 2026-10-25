# Manuale utente di OpenFocus

## Indice

1. [Introduzione](#1-introduzione)
2. [Per iniziare](#2-per-iniziare)
3. [Panoramica dell'interfaccia principale](#3-panoramica-dellinterfaccia-principale)
4. [Importazione di immagini e video](#4-importazione-di-immagini-e-video)
5. [Visualizzazione e navigazione delle immagini](#5-visualizzazione-e-navigazione-delle-immagini)
6. [Vista di confronto Wipe](#6-vista-di-confronto-wipe)
7. [Registrazione delle immagini](#7-registrazione-delle-immagini)
8. [Metodi di fusione delle immagini](#8-metodi-di-fusione-delle-immagini)
9. [Trasformazioni delle immagini](#9-trasformazioni-delle-immagini)
10. [Ritocco, etichette e annotazioni](#10-ritocco-etichette-e-annotazioni)
11. [Impostazioni e configurazione](#11-impostazioni-e-configurazione)
12. [Elaborazione batch](#12-elaborazione-batch)
13. [Esportazione dei risultati](#13-esportazione-dei-risultati)
14. [Interfaccia a riga di comando](#14-interfaccia-a-riga-di-comando)
15. [Scorciatoie da tastiera](#15-scorciatoie-da-tastiera)
16. [Risoluzione dei problemi](#16-risoluzione-dei-problemi)

---

## 1. Introduzione

OpenFocus è un'applicazione desktop professionale per la fusione di immagini multi-fuoco, pensata per ricercatori, ingegneri e professionisti dell'elaborazione di immagini. Il software elabora sequenze di immagini riprese a punti di fuoco diversi applicando algoritmi avanzati di registrazione e fusione, e genera immagini tutto-a-fuoco con profondità di campo completa.

### Funzionalità principali

- **Fusione di immagini multi-fuoco**: combina più immagini riprese con punti di fuoco diversi in un'unica immagine completamente a fuoco
- **Molteplici algoritmi di fusione**: scegli tra Filtro guidato, DCT, DTCWT, GFG-FGF e AI (apprendimento profondo)
- **Registrazione delle immagini**: allinea sequenze disallineate con i metodi ECC o Omografia
- **Vista di confronto Wipe**: ispeziona l'allineamento e confronta i risultati con un divisore A/B
- **File di progetto**: salva e ripristina intere sessioni di lavoro
- **TIFF multipagina**: gli stack Z in un unico file vengono caricati pagina per pagina e riesportati in un solo file
- **Mappa di profondità**: mappe della posizione di fuoco per singolo pixel, con schemi di falsi colori personalizzati
- **Barra di scala**: barre di scala di livello editoriale con calibrazione automatica µm/px
- **Recupero automatico dopo un arresto anomalo**: dopo un crash viene proposta la sessione di lavoro precedente
- **Interfaccia multilingua**: English, 中文, 日本語, Español con rilevamento automatico della lingua di sistema
- **CLI senza interfaccia**: automatizza OpenFocus dalla riga di comando, per pipeline e CI
- **Esportazione flessibile**: salva i risultati come immagini singole, cartelle o animazioni GIF
- **Trasformazioni delle immagini**: ruota, capovolgi e ridimensiona gli stack di immagini
- **Aggiunta di etichette**: aggiungi etichette di testo allo stack registrato e a quello di input
- **Impostazioni persistenti**: preferenze, lingua e stack aperti di recente sopravvivono al riavvio

### Formati di input supportati

- File immagine: JPG/JPEG, PNG, BMP, TIFF, WebP e altri formati comuni
- File video: formato MP4 (decodificato automaticamente in sequenze di immagini)
- Cartelle contenenti sequenze di immagini numerate
- File RAW di fotocamera (DNG, CR2/CR3, NEF, ARW, RAF, ORF, RW2 e altri formati
  RAW comuni). La decodifica usa `rawpy` (LibRaw), installato da
  `pip install openfocus` e incluso nelle build per Windows/macOS

### Formati di output supportati

- Immagini singole: JPG, PNG, BMP, TIFF, WebP
- Sequenze di immagini: salvate come cartelle
- Animazioni: formato GIF

---

## 2. Per iniziare

### Avvio dell'applicazione

Dopo l'installazione, avvia OpenFocus in uno di questi modi:

1. **Windows (eseguibile)**: fai doppio clic sul collegamento o sul file eseguibile di OpenFocus
2. **Codice sorgente Python**: esegui `python main.py` nel tuo ambiente conda attivato

All'avvio compare la finestra principale dell'applicazione con interfaccia a tema scuro.

### Flusso di lavoro rapido

1. Importa la sequenza di immagini (File → Apri cartella, oppure trascina e rilascia)
2. Configura le impostazioni di fusione nel pannello di destra
3. Se serve, attiva la registrazione per allineare i frame
4. Regola la dimensione del kernel se necessario
5. *(facoltativo)* Seleziona **Anteprima rapida** per provare prima i parametri su una bozza ridotta
6. Fai clic su «Avvia render» per iniziare l'elaborazione
7. Esporta i risultati

---

## 3. Panoramica dell'interfaccia principale

L'interfaccia di OpenFocus è composta da tre aree principali:

### Pannello sinistro: area di visualizzazione delle immagini

La parte sinistra della finestra mostra le immagini in una vista divisa:

- **Pannello dell'immagine sorgente (in alto)**: mostra la sequenza di immagini di input
- **Pannello dell'immagine risultato (in basso)**: mostra i risultati della fusione o della registrazione
- **Cursore di navigazione**: usa il cursore sotto ciascun pannello per scorrere i frame
- **Etichette informative**: mostrano la posizione del frame corrente e le dimensioni dell'immagine

Puoi ridimensionare i pannelli trascinando il separatore tra di essi.

### Pannello destro: pannello di controllo

Il pannello di destra raccoglie tutte le opzioni di configurazione, disposte verticalmente:

1. **Impostazioni di fusione**: scelta dell'algoritmo di fusione e della dimensione del kernel
2. **Impostazioni di registrazione**: attivazione/disattivazione dei metodi di allineamento
3. **Pulsanti di azione**: Ripristina predefiniti e Avvia render
4. **Elenco delle immagini sorgente**: mostra tutte le immagini caricate con il nome del file
5. **Elenco output**: mostra i risultati generati

### Barra dei menu

La barra dei menu in alto dà accesso a tutte le funzioni:

- **File**: operazioni di importazione/esportazione e elenco dei file recenti
- **Modifica**: trasformazioni delle immagini ed etichette
- **Strumenti**: elaborazione batch, mappa di profondità e mappa polvere
- **Impostazioni**: accelerazione GPU, lingua, tasselli, registrazione e thread
- **Aiuto**: informazioni sull'ambiente, contatti, controllo aggiornamenti e cartella dei log

---

## 4. Importazione di immagini e video

### Apertura di una cartella di immagini

1. Vai su **File → Apri cartella** o premi `Ctrl+O`
2. Seleziona una cartella contenente la sequenza di immagini
3. Compare una finestra per la scala di caricamento: scegli la scala (100% = risoluzione piena)
4. Fai clic su OK per caricare le immagini

Seleziona **Non chiedere più** in quella finestra per caricare sempre con la scala
scelta senza domande; **Impostazioni → Chiedi la scala di caricamento all'importazione**
riattiva la richiesta. Uno stack già caricato può comunque essere ridotto in seguito con
**Modifica → Ridimensiona**.

Il software eseguirà automaticamente queste operazioni:
- Rileva e ordina i file immagine per nome
- Mostra la prima immagine nel pannello sorgente
- Popola l'elenco delle immagini sorgente con tutti i frame

### Apertura di file video

1. Vai su **File → Apri video** o premi `Ctrl+Shift+O`
2. Seleziona un file video (MP4, AVI, MOV, MKV, WMV, FLV, WEBM)
3. Il video viene decodificato automaticamente in frame singoli
4. Ogni frame entra a far parte dello stack di immagini

### File di progetto

File → Salva progetto memorizza l'intera sessione di lavoro in un file `.ofproj`:
l'elenco esatto dei file sorgente (nell'ordine), la scala di caricamento, il metodo
di fusione, le opzioni di registrazione, la dimensione del kernel, le impostazioni di
tasselli/thread/registrazione, le configurazioni delle etichette e il frame corrente.
File → Apri progetto ricostruisce esattamente quella sessione (le immagini sorgente
sono referenziate, non copiate: lasciale dove sono). Il titolo della finestra mostra il progetto aperto.

### Stack demo

La finestra di benvenuto offre **Carica stack demo**: sei foto di esempio incluse,
provenienti da un vero stack di messa a fuoco, per provare subito l'intero flusso di
lavoro senza immagini tue. È utile anche per verificare un'installazione.

### Curva di nitidezza

Una piccola curva sopra il cursore di navigazione della sorgente mostra la nitidezza
per frame (normalizzata). I frame fuori fuoco appaiono come avvallamenti contrassegnati
da un punto rosso. Fai clic in un punto qualsiasi della curva per passare a quel frame;
passando il mouse sopra viene mostrato il numero del frame.

### Stack TIFF multipagina

I software di microscopia (ZEN, NIS-Elements, ImageJ/Fiji) esportano spesso un intero
stack Z in UN unico file TIFF multipagina. OpenFocus legge ogni pagina di quel file
come frame singolo dello stack di immagini: trascina o apri il file e lo stack appare
completo (i frame si chiamano `<file>_page_001` e così via). I TIFF a pagina singola si
comportano esattamente come prima. Funziona anche il contrario:
**File → Salva stack → (Stack registrato / Stack di input) → Salva come TIFF multipagina**
riscrive l'intero stack in un unico `.tif`, preservando la profondità a 16 bit.

### Barra di scala (figure per pubblicazioni)

Impostazioni → Barra di scala… configura una barra di scala impressa nelle immagini
esportate. La dimensione del pixel (µm/px) viene rilevata automaticamente dai metadati
TIFF — la convenzione `unit=micron` di ImageJ, `PhysicalSizeX` di OME-TIFF oppure
un'unità di risoluzione esplicita in centimetri — così gli stack di microscopia
(compresi i TIFF multipagina) si calibrano da soli. La risoluzione di stampa
(72/300 dpi e simili) è deliberatamente *esclusa* dalla calibrazione, e un valore
inserito manualmente ha sempre la precedenza sui metadati rilevati: una barra sbagliata
ma credibile in una figura è peggio di nessuna barra. Scegli l'angolo e il colore
(Auto inverte rispetto allo sfondo); la finestra mostra un'anteprima in tempo reale.
Con la barra attiva, ogni esportazione (immagine singola, stack in cartella, TIFF
multipagina, GIF, esportazione completa, batch) include una barra in stile editoriale:
una lunghezza arrotondata 1-2-5 (~12% della larghezza dell'immagine) con testo
leggibile e contornato. La CLI la supporta tramite `--scale-bar` (calibrazione
automatica) e `--scale-um <um/px>`.

### Mappa di profondità (posizione di fuoco)

**Strumenti → Mappa di profondità…** apre un visualizzatore che calcola, per ogni pixel,
quale frame dello stack è a fuoco — il segnale di pseudo-profondità su cui la fusione
stessa decide — usando la misura di attività del metodo selezionato. Sono disponibili
solo i metodi che decidono un indice (DCT, DTCWT, GFG-FGF, StackMFF-V3, OpenFocus B/C);
le fusioni a media ponderata (Filtro guidato, OpenFocus A) non hanno una decisione per
pixel e non compaiono nell'elenco. La mappa è mostrata in falsi colori con una legenda
(frame 0 → N-1) e può essere personalizzata in tempo reale:

- **Mappa colori**: scala di grigi (standard per le pubblicazioni) più 15 palette
  scientifiche (Turbo, Viridis, Magma, …) e le tue — *Nuovo schema personalizzato…*
  costruisce un gradiente da un numero qualsiasi di stop di colore (doppio clic su uno
  stop per modificarne posizione e colore); gli schemi personalizzati vengono salvati
  e proposti nel menu a discesa.
- **Inverti / Gamma / Levigatura**: inverti l'asse di profondità, ribilancia il tono
  (gamma < 1 schiarisce gli indici bassi) e rimuovi la maculatura con un passaggio
  che tiene conto dei bordi.
- **Sovrapponi all'immagine fusa**: miscela la mappa colorata sul risultato della
  fusione con opacità regolabile, per vedere esattamente dove ha vinto ciascun frame.
- **Esportazione**: *Salva colorata…* (PNG/JPEG/TIFF/BMP) oppure *Salva grezza a 16 bit…*
  (la mappa degli indici normalizzata a piena precisione, ideale per analisi successive
  in ImageJ/Fiji/python).
- **Modello 3D**: *Esporta 3D (OBJ)…* scrive una superficie texturizzata costruita dalla
  mappa di profondità e dal risultato fuso come `<name>.obj`, `<name>.mtl` e
  `<name>_texture.png`, file che si aprono in Blender, MeshLab o Windows 3D Viewer.
  La mesh viene ridotta a non più di 600 vertici sul lato più lungo e leggermente
  ripulita dalla maculatura, dopo aver applicato il valore Levigatura indicato sopra.
  *Rilievo 3D* (1–100 %, predefinito 15 %) imposta l'altezza della superficie come
  percentuale rispetto al lato più lungo dell'immagine e viene ricordato. *Inverti asse
  di profondità* scambia quale estremità dello stack è vicina. La texture è il risultato
  fuso, oppure il primo frame dello stack se non esiste un risultato della stessa dimensione.

Il calcolo avviene in background, con avanzamento e possibilità di annullamento; il
metodo AI riutilizza la stessa tassellatura sicura per la memoria della fusione AI, così
anche gli stack profondi funzionano. L'equivalente da CLI è `--depth-map [path]` con
`--depth-colormap`.

### Recupero automatico dopo un arresto anomalo

OpenFocus mantiene continuamente un'istantanea della sessione di lavoro (stack,
impostazioni, etichette, frame corrente). Se l'applicazione si chiude in modo imprevisto
— crash, mancanza di corrente o chiusura forzata — all'avvio successivo viene rilevato e
proposto **Ripristina sessione** per tornare esattamente al punto in cui eri. Le chiusure
regolari non mostrano mai la richiesta. I dati di recupero si trovano accanto alle
impostazioni (le installazioni portatili li tengono accanto all'eseguibile, così
viaggiano con te).

### File recenti

Ogni stack aperto viene ricordato in **File → Apri recenti** (fino a 8 voci). Premi
`Alt+1`–`Alt+8` per riaprire direttamente una voce. Riaprendo lo stesso stack, questo
passa in cima; le voci i cui file non esistono più possono essere rimosse con una
richiesta di conferma, e **Cancella elenco recenti** svuota il menu.

### Trascinamento

Puoi anche trascinare una cartella direttamente sulla finestra dell'applicazione per
importare le immagini. Trascinando altri file su uno stack esistente, questi vengono
aggiunti in coda (dopo un controllo delle dimensioni).

### Orientamento EXIF

I file JPEG/TIFF delle fotocamere con tag EXIF Orientation vengono ruotati e corretti
automaticamente durante l'importazione, così le foto verticali si caricano dritte.

### Sorgenti a 16 bit

Gli input PNG/TIFF a 16 bit vengono rilevati automaticamente e conservati da un capo
all'altro: la fusione opera a profondità piena e i risultati possono essere risalvati
come PNG/TIFF a 16 bit (le esportazioni JPG/BMP vengono convertite a 8 bit, perché quei
formati non possono memorizzare 16 bit).

I file RAW di fotocamera vengono sviluppati a 16 bit (bilanciamento del bianco della
fotocamera, gamma sRGB) con la schiaritura automatica disattivata, così ogni frame dello
stack riceve la stessa mappatura di esposizione; un'esposizione automatica per frame
farebbe sfarfallare la luminosità tra un frame e l'altro. Salva come TIFF o PNG per
conservare i dati a 16 bit.

### Svuotamento dello stack

Per rimuovere tutte le immagini caricate, vai su **File → Svuota stack** o premi `Ctrl+W`.

---

## 5. Visualizzazione e navigazione delle immagini

### Navigazione nello stack di immagini

- **Cursore**: trascina il cursore sotto il pannello sorgente per scorrere i frame
- **Elenco dei file**: fai clic su un nome file nell'elenco delle immagini sorgente, nel pannello di destra
- **Tastiera**: usa i tasti freccia (sinistra/destra) per spostarti

### Zoom e spostamento

- **Zoom avanti/indietro**: rotella del mouse sull'immagine
- **Zoom fine**: tieni premuto `Ctrl` + rotella per incrementi più piccoli
- **Adatta alla finestra**: fai doppio clic sull'immagine per passare tra vista adattata e 100%
- **Spostamento**: fai clic e trascina per muovere l'immagine dentro il pannello

### Navigazione sincronizzata

Quando ti sposti nel pannello sorgente, il pannello risultato mostra il frame
corrispondente del risultato, se disponibile.

---

## 6. Vista di confronto Wipe

La vista Wipe sovrappone due immagini in un unico riquadro con un divisore verticale
trascinabile, rendendo immediatamente visibili disallineamenti e differenze di qualità
attraverso la divisione.

### Apertura della vista Wipe

1. Esegui il render di almeno un risultato (fusione o registrazione)
2. Premi l'interruttore **Wipe** nell'angolo in alto a destra della barra del titolo del pannello risultato
3. L'area del risultato passa alla vista di confronto; premi di nuovo **Wipe** per tornare indietro

### Lettura della vista

- **Lato A (a sinistra del divisore)**: un frame sorgente — per impostazione predefinita segue il cursore del pannello sorgente
- **Lato B (a destra del divisore)**: il render più recente
- **Divisore bianco + maniglia rotonda**: trascina orizzontalmente per spostare la divisione
- Trascinare bordi che attraversano il divisore rivela errori di registrazione difficili da notare in pannelli affiancati

### Scelta di cosa confrontare

La barra in fondo alla vista di confronto ha un gruppo per lato:

- **Lato A** — seleziona **Segui la sorgente** (predefinito) per seguire il cursore del pannello sorgente, oppure deselezionalo per bloccare il lato A su un frame qualsiasi con il suo cursore
- **Lato B** — seleziona **Ultimo risultato** (predefinito) per mostrare sempre il render più recente, oppure deselezionalo per scorrere la cronologia dell'output con il suo cursore

Entrambi i lati si aggiornano in tempo reale: nuovi render, spostamenti del cursore
sorgente ed eliminazioni di frame si riflettono immediatamente.

### Zoom e spostamento

La vista di confronto condivide un'unica trasformazione per entrambe le immagini:

- **Zoom**: rotella del mouse (lo zoom è centrato sul cursore)
- **Spostamento**: fai clic e trascina in un punto qualsiasi tranne il divisore
- **Reimposta**: doppio clic

La modalità Wipe e la modalità ROI condividono il pannello risultato, quindi attivare
una disattiva l'altra. Svuotare lo stack esce anche dalla modalità Wipe.

---

## 7. Registrazione delle immagini

La registrazione delle immagini corregge il disallineamento spaziale tra i frame dello
stack. È essenziale quando le immagini presentano piccoli spostamenti o variazioni di
prospettiva.

### Metodi di registrazione

#### ECC (Enhanced Correlation Coefficient)

- Usa l'ottimizzazione per trovare il miglior allineamento in base alla correlazione
- Adatto a disallineamenti contenuti
- Garantisce accuratezza sub-pixel
- Più lento dell'Omografia ma più preciso

#### Omografia

- Basata sull'abbinamento di punti caratteristici (feature SIFT/ORB)
- Gestisce trasformazioni geometriche più ampie
- Più veloce di ECC per un allineamento iniziale
- Buona per immagini con variazioni di prospettiva significative

#### OpenFocus (veloce)

- Allineamento per similarità limitata (traslazione + rotazione + scala),
  con equalizzazione della luminosità per frame e ritaglio automatico
- Meno di un secondo per frame — molto più veloce di ECC o Omografia
- I frame che si spostano più dell'**8% della dimensione del frame vengono lasciati non allineati**
  anziché corretti a metà. Puoi modificare la soglia con l'opzione CLI
  `--align-shift-limit`
- Per geometrie non rigide o estreme, scegli invece ECC o Omografia

### Attivazione della registrazione

1. Nel pannello di destra individua il gruppo **Registrazione**
2. Scegli un metodo dal menu a discesa: **Nessun allineamento**, **OpenFocus (veloce)**,
   **ECC** o **Omografia**
3. L'allineamento veloce di OpenFocus può anche essere concatenato prima di ECC o
   Omografia (prima un passaggio globale grossolano, poi il passaggio locale preciso)

### Quando usare la registrazione

- Le immagini presentano un disallineamento visibile
- Le immagini sono state scattate a mano libera
- La scena ha parallasse (gli oggetti più vicini si spostano tra i frame)
- I risultati mostrano artefatti di sdoppiamento o immagini fantasma

### Impostazioni di registrazione

Le impostazioni aggiuntive sono disponibili in **Impostazioni → Impostazioni registrazione**:
- **Larghezza di ridimensionamento**: controlla la risoluzione di pre-elaborazione (predefinito: 1024px)
- Valori più bassi = elaborazione più rapida, potenzialmente minore accuratezza
- Valori più alti = elaborazione più lenta, potenzialmente maggiore accuratezza

---

## 8. Metodi di fusione delle immagini

OpenFocus offre otto algoritmi di fusione (più una modalità di solo allineamento). Ognuno
ha caratteristiche diverse, adatte a tipi di immagine differenti.

### Filtro guidato (predefinito)

- **Algoritmo**: fusione con conservazione dei bordi tramite filtro guidato
- **Ideale per**: uso generico, buon compromesso tra velocità e qualità
- **Vantaggi**: buona conservazione dei bordi, elaborazione veloce
- **Parametro**: dimensione del kernel (regolabile da 1 a 51, solo valori dispari)
  - Kernel più piccolo: più dettaglio locale, potenzialmente più rumore
  - Kernel più grande: risultati più morbidi, migliore soppressione del rumore

### DCT (Discrete Cosine Transform)

- **Algoritmo**: fusione nel dominio della frequenza tramite coefficienti DCT
- **Ideale per**: immagini ricche di texture, imaging scientifico
- **Vantaggi**: preserva le texture fini, matematicamente solido
- **Parametro**: la dimensione del kernel controlla la finestra di elaborazione

### DTCWT (Dual-Tree Complex Wavelet Transform)

- **Algoritmo**: fusione multi-scala nel dominio delle wavelet
- **Ideale per**: scene complesse con più livelli di dettaglio
- **Vantaggi**: eccellente analisi multi-scala, buona direzionalità
- **Regola di fusione**: una decisione per pixel e per livello, valida per tutti gli
  orientamenti e i canali di colore (a N vie, con verifica di coerenza a maggioranza);
  la banda passa-basso segue la decisione più grossolana invece di mediare tutti i frame
- **Parametro**: nessuno (il cursore del kernel è disattivato)

### GFG-FGF (Gradient a quattro vicinati di tipo gaussiano + filtro guidato rapido)

- **Algoritmo**: implementazione di Fu et al. (*Laser & Optoelectronics
  Progress*, 2024): una mappa di fuoco per differenza di sfocatura |I − mean(I)| viene
  arricchita con l'operatore gradiente a quattro vicinati di tipo gaussiano (GFG),
  raffinata da un filtro guidato rapido, decisa per pixel (i pareggi sono risolti da una
  misura di piccola regione Sobel 3×3) e le mappe di decisione binarie vengono levigate
  di nuovo dal filtro guidato rapido prima della miscelazione pesata.
- **Ideale per**: microscopia e immagini con regioni a fuoco ben definite
- **Vantaggi**: veloce, confini delle regioni a fuoco nitidi; la memoria non cresce con
  il numero di frame
- **Parametro**: Kernel = dimensione del filtro medio della mappa di fuoco grossolana

### StackMFF-V3 (apprendimento profondo)

- **Algoritmo**: classificazione della sorgente pixel per pixel (richiede PyTorch). La
  rete prevede, per ogni pixel, da quale frame di input proviene, poi miscela i frame in
  base a tali probabilità oppure sceglie un solo frame
- **Output AI** (pannello di destra, solo per questo metodo):
  - **Miscelazione soft (pesata)** — predefinita; ogni pixel è una miscela dei frame
    pesata per probabilità. Transizioni morbide, nessuna giuntura netta; corrisponde
    alla configurazione riportata nell'articolo
  - **Scelta netta (frame singolo)** — ogni pixel è copiato dal frame più probabile.
    Selezione più nitida senza interpolazione tra frame; può comparire una giuntura a
    livello di pixel dove due frame sono in parità
- **Ideale per**: risultati della massima qualità quando è disponibile una GPU
- **Vantaggi**: invariante all'ordine (l'ordine dei frame non cambia mai il risultato),
  gestisce stack di qualsiasi lunghezza, nessuna taratura per metodo, ottimizzazione automatica
- **Requisiti**: installazione di PyTorch, GPU CUDA opzionale (senza torchvision)
- **Nota**: può non essere disponibile se PyTorch non è installato

### OpenFocus A / B / C

- **Algoritmo**: media ponderata per la varianza, selezione dalla mappa di profondità
  della varianza e fusione a piramide laplaciana
- **OpenFocus A**: media ponderata per la varianza con prior sull'ultimo frame —
  veloce, colori stabili su stack brevi
- **OpenFocus B**: selezione basata sulla mappa di profondità della varianza — sceglie
  il frame più nitido per ogni pixel; ideale su superfici lisce e in microscopia
- **OpenFocus C**: miscelazione per banda con piramide laplaciana — aspetto naturale su
  soggetti texturizzati, aumenta il contrasto locale
- **Parametri**: A e B usano una Finestra varianza (cursore del kernel); C non ha
  raggio. Tutti e tre usano un sigma di levigatura.

**Levigatura**: un unico campo numerico sotto il cursore del kernel, con il significato
del metodo selezionato (intervallo 0–8, passo 0,2):

| Metodo | Etichetta | Predefinito | Effetto |
|---|---|---|---|
| OpenFocus A | Levigatura miscelazione (σ) | 2.0 | Levigatura gaussiana della mappa dei pesi per frame — quanto ampiamente vengono miscelate le regioni nitide dei frame vicini |
| OpenFocus B | Levigatura profondità (σ) | 1.2 | Levigatura gaussiana della mappa di profondità — larghezza della transizione tra i frame |
| OpenFocus C | Levigatura miscelazione (σ) | 3.0 | Levigatura gaussiana della mappa dei pesi di ciascuna banda della piramide |

0 mantiene la scelta per pixel più nitida (dettaglio più fine, ma maculature sugli stack
rumorosi); valori più alti ammorbidiscono transizioni e aloni. Ogni metodo ricorda il
proprio valore tra le sessioni e l'elaborazione batch applica la stessa regola.

C non ha Raggio: il Metodo C di Helicon espone solo la Levigatura, quindi per C il
cursore del kernel viene nascosto invece di non fare nulla in silenzio.

CLI: `--depth-smoothing SIGMA` (B) e `--blend-smoothing SIGMA` (A, C).

**Le opzioni di luminosità sono per metodo.** *Uniforma l'esposizione prima della fusione* e
*Rimuovi deriva di luminosità* vengono ricordate separatamente per ciascun metodo di fusione,
perché rispondono a una domanda relativa al singolo stack e non a una preferenza globale.
Entrambe sono attive per impostazione predefinita per OpenFocus A/B/C (che non hanno una
calibrazione dell'esposizione integrata); gli altri metodi mantengono i valori predefiniti
storici (uniformatura dell'esposizione attiva, rimozione della deriva disattivata).
Cambiando metodo vengono mostrati i valori di quel metodo.

### Solo allineamento (nessuna fusione)

Scegli **Solo allineamento (nessuna fusione)** come metodo per registrare lo stack e
inviare i frame allineati all'elenco output senza fondere: comodo per ispezionare la
qualità della registrazione o esportare i frame allineati.

### Scelta di un metodo di fusione

1. Nel pannello di destra individua il gruppo **Fusione**
2. Scegli un metodo dal **menu a discesa** (uno solo è sempre selezionato;
   la prima voce registra senza fondere)
3. Regola la dimensione del kernel se applicabile — l'etichetta indica cosa
   significa il kernel per il metodo selezionato
4. Fai clic su **Avvia render** per elaborare

### Guida al confronto degli algoritmi

| Algoritmo | Velocità | Qualità | Ideale per |
|-----------|-------|---------|----------|
| Filtro guidato | Veloce | Buona | Uso generico |
| DCT | Media | Buona | Texture |
| DTCWT | Media | Molto buona | Scene complesse |
| GFG-FGF | Veloce | Buona | Regioni a fuoco |
| AI (StackMFF-V3) | Lenta (GPU) | Eccellente | Massima qualità |
| OpenFocus A | Molto veloce | Molto buona | Stack brevi, stabilità dei colori |
| OpenFocus B | Molto veloce | Molto buona | Superfici lisce, microscopia |
| OpenFocus C | Veloce | Molto buona | Soggetti texturizzati, aspetto naturale |

### Anteprima rapida (render di bozza)

Tarare i parametri su stack grandi a piena risoluzione è lento. Seleziona
**Anteprima rapida** sopra il pulsante di render e OpenFocus esegue prima il render di
una bozza ridotta (lato più lungo 1200 px) — in genere diverse volte più veloce, anche
per la registrazione. L'anteprima viene mostrata con la nota «Anteprima» nel riepilogo
della barra di stato dopo il render e **non** viene aggiunta alla cronologia
dell'output. Deseleziona la casella ed esegui di nuovo il render per ottenere il
risultato a piena risoluzione. La modalità anteprima non si applica ai render ROI e non
inquina mai la cache di allineamento a piena risoluzione.

---

## 9. Trasformazioni delle immagini

### Rotazione

1. Vai su **Modifica → Ruota**
2. Scegli il tipo di rotazione:
   - **90° in senso orario**: ruota tutte le immagini di 90 gradi verso destra
   - **90° in senso antiorario**: ruota tutte le immagini di 90 gradi verso sinistra
   - **180°**: capovolge le immagini sottosopra

### Capovolgimento

1. Vai su **Modifica → Capovolgi**
2. Scegli il tipo di capovolgimento:
   - **Capovolgi orizzontalmente**: specchia da sinistra a destra
   - **Capovolgi verticalmente**: specchia dall'alto verso il basso

### Ridimensionamento

1. Vai su **Modifica → Ridimensiona**
2. Compare una finestra con le opzioni di ridimensionamento
3. Inserisci le nuove dimensioni o la percentuale di scala
4. Tutte le immagini dello stack vengono ridimensionate in modo proporzionale

### Ordine delle trasformazioni

Le trasformazioni si applicano all'intero stack di immagini, mantenendo l'allineamento
tra i frame.

---

## 10. Ritocco, etichette e annotazioni

### Ritocco del risultato (Modifica → Ritocca risultato…, `Ctrl+T`)

La fusione può comunque lasciare un difetto locale: un alone attorno a un bordo ad alto
contrasto, una setola presa dal frame sbagliato, rumore in un'area piatta. Il pennello di
ritocco copia i pixel da un frame sorgente sul risultato:

1. Esegui il render di un risultato (oppure selezionane uno nell'elenco output; funziona
   anche il clic con il tasto destro → *Ritocca risultato…*) e premi `Ctrl+T`.
2. Dipingi sul difetto con il tasto sinistro del mouse. Con **Scelta automatica del frame
   più nitido per ogni tratto** attiva (predefinita), ogni tratto seleziona prima il frame
   con il maggior dettaglio sotto il pennello: correggere un alone richiede di solito un
   solo tratto. Disattivala per scegliere il frame manualmente con il cursore o con
   `,` / `.`.
3. Regola **raggio** (`[` / `]`), **durezza** (i bordi morbidi nascondono la giuntura) e
   **opacità**. Le passate sovrapposte all'interno di uno stesso tratto non si sommano.
4. **Mostra frame sorgente** (`Tab`) mostra in anteprima il frame da cui stai dipingendo.
   La rotella fa zoom, il trascinamento con tasto destro/centrale sposta, `F` adatta l'immagine.
5. `Ctrl+Z` / `Ctrl+Y` annullano e ripetono interi tratti. **Applica come nuovo output**
   aggiunge l'immagine ritoccata all'elenco output (suffisso `_Retouched`); l'output
   originale resta intatto.

Il pennello usa i frame allineati che hanno prodotto il risultato, quindi si sovrappone
correttamente; i risultati a 16 bit restano a 16 bit. I render ROI e gli stack
ridimensionati non possono essere ritoccati (i loro frame non corrispondono più alla
geometria del risultato).

### Aggiunta di etichette

1. Assicurati di aver eseguito il render di risultati o di aver caricato immagini
2. Vai su **Modifica → Aggiungi etichetta**
3. Inserisci il testo dell'etichetta nella finestra
4. Le etichette vengono aggiunte sia allo stack registrato sia a quello di input

### Eliminazione delle etichette

- **Elimina etichette dello stack registrato**: rimuove le etichette dalle immagini allineate
- **Elimina etichette dello stack di input**: rimuove le etichette dalle immagini originali

Accedi a queste opzioni dal sottomenu **Modifica → Elimina etichetta**.

---

## 11. Impostazioni e configurazione

### Impostazioni dei tasselli (ottimizzazione della memoria)

Per le immagini grandi, OpenFocus usa l'elaborazione a tasselli per evitare problemi di memoria.

Accesso tramite **Impostazioni → Impostazioni tasselli**:

- **Dimensione del blocco**: dimensione di ciascun tassello di elaborazione (predefinito: 1024px)
- **Sovrapposizione dei tasselli**: sovrapposizione tra tasselli per una miscelazione senza giunture (predefinito: 256px)
- **Soglia di tassellatura**: dimensione dell'immagine oltre la quale si usano i tasselli (predefinito: 2048px)

**Consigli**:
- Blocchi più piccoli: meno memoria usata, elaborazione più lenta
- Blocchi più grandi: più memoria usata, elaborazione più rapida
- Aumenta la sovrapposizione se noti giunture visibili nei risultati

### Impostazioni di registrazione

Accesso tramite **Impostazioni → Impostazioni registrazione**:

- **Larghezza di ridimensionamento**: risoluzione di pre-elaborazione per la registrazione
  - Predefinito: 1024px
  - Valori più bassi per la velocità, più alti per l'accuratezza

### Riferimenti sulle prestazioni

Vedi [PERFORMANCE.md](./PERFORMANCE.md) per i valori misurati di throughput e memoria
su stack reali di grandi dimensioni, e per indicazioni pratiche su stack profondi e
input ad alta risoluzione.

### Impostazioni del numero di thread

Controllano l'elaborazione parallela sulla CPU:

Accesso tramite **Impostazioni → Impostazioni thread**:

- `0` seleziona la pianificazione automatica ed è il valore iniziale delle nuove impostazioni.
- I valori positivi limitano i thread; i limiti salvati vengono conservati.
- Capacità della CPU, modalità risorse e memoria disponibile possono ridurre la concorrenza.

### Interruttore dell'accelerazione GPU

OpenFocus usa la GPU (CUDA/MPS) solo per il modello neurale AI (StackMFF-V3); gli algoritmi classici girano sempre sulla CPU.

- Interruttore in **Impostazioni → Accelerazione GPU**
- Deselezionandolo, tutto viene forzato sulla CPU — utile quando i driver della GPU causano problemi
- Il pannello di stato mostra `CPU*` mentre la GPU è disattivata
- La scelta ha effetto dal render successivo e viene ricordata

### Dimensione batch AI

Accesso tramite **Impostazioni → Dimensione batch AI**:

- Il limite batch salvato è 2 per impostazione predefinita; attualmente viene elaborato un tassello per chiamata su ogni dispositivo.
- Aumentare il limite non abilita batch paralleli sulla GPU.

### Temi

Sono disponibili due temi dell'interfaccia: **Scuro (predefinito)** e **Chiaro**. Si cambia
da **Impostazioni → Tema**; la scelta viene applicata subito (anche la barra del titolo di
Windows si adegua) e viene ricordata. Al primo avvio il tema si sceglie nella finestra di
benvenuto.

### Modalità portatile

Crea un file vuoto chiamato `OpenFocus.portable` accanto a `OpenFocus.exe`
(lo zip portatile ne contiene già uno) e tutti i dati utente — impostazioni, log, file
recenti — vengono salvati in una cartella `OpenFocusData` accanto all'applicazione
anziché nel profilo utente, così l'installazione può viaggiare su una chiavetta USB.
Elimina il file marcatore per tornare al profilo normale.

### Metadati EXIF

I metadati di origine (modello di fotocamera, esposizione, orientamento e così via)
vengono conservati: l'orientamento viene applicato al caricamento e le esportazioni
PNG/TIFF/WebP incorporano automaticamente gli EXIF originali. Le esportazioni JPG/BMP
non possono memorizzare gli EXIF completi e vengono scritte senza.

### Input HEIC/HEIF

Le foto di iPhone si caricano direttamente. Richiede il pacchetto opzionale `pillow-heif`
(`pip install pillow-heif`; incluso nelle build pacchettizzate).

### Aggiornamenti

I file di progetto (`.ofproj`) si comportano come un documento: la barra del titolo mostra
il nome del progetto con un `*` finché ci sono modifiche non salvate, e chiudendo
l'applicazione o aprendo un altro progetto viene chiesta conferma prima di scartarle. Il
progetto memorizza l'elenco dei frame, la scala di caricamento, tutte le opzioni di
rendering, le etichette e il frame corrente — le immagini stesse sono referenziate, non copiate.

Aiuto → Controlla aggiornamenti confronta la tua versione con le release di GitHub.
**Scarica e installa** recupera il nuovo programma di installazione dall'interno
dell'app (con avanzamento) e lo avvia; Apri pagina download porta all'elenco delle
release. Il controllo viene eseguito in silenzio al massimo una volta al giorno.

### Cache di registrazione

OpenFocus può memorizzare i risultati della registrazione in una cartella `.openfocus_cache`
accanto allo stack sorgente. Riaprendo lo stesso stack (file invariati), l'allineamento
viene saltato del tutto. Interruttore in **Impostazioni → Cache di registrazione**; le
voci obsolete vengono sostituite automaticamente.

### Impostazioni persistenti

Tutte le voci seguenti sopravvivono automaticamente al riavvio dell'applicazione: numero
di thread, parametri dei tasselli, larghezza di ridimensionamento per la registrazione,
dimensione batch della fusione AI (StackMFF-V3), interruttore dell'accelerazione GPU,
interruttore della cache di registrazione, lingua dell'interfaccia, dimensioni della
finestra e disposizione dei pannelli, formato di esportazione per trascinamento e stack
aperti di recente. Le finestre delle impostazioni salvano nel momento in cui confermi;
tutto viene comunque salvato alla chiusura dell'applicazione. Le preferenze sono
memorizzate per utente in `OpenFocus/OpenFocus.ini` (formato QSettings INI).

### Lingua dell'interfaccia

Si cambia in qualsiasi momento da **Impostazioni → Lingua → English / 中文**. La modifica
si applica immediatamente e viene ricordata per le sessioni successive.

### Controllo automatico degli aggiornamenti

Una volta al giorno, pochi secondi dopo l'avvio, OpenFocus controlla in silenzio su GitHub
se esiste una release più recente. Quando c'è, compare un banner sopra i pannelli delle
immagini: la versione, **Aggiorna ora** (scarica e riavvia in un solo passaggio),
**Novità** (note di rilascio) e un pulsante per chiudere. Chiudendo il banner, questo resta
nascosto solo per quella versione — la release successiva lo fa ricomparire, e
**Aiuto → Controlla aggiornamenti** lo riattiva subito.

Durante il download in background la barra di stato mostra l'avanzamento. Puoi anche
controllare manualmente in qualsiasi momento con **Aiuto → Controlla aggiornamenti**.

---

## 12. Elaborazione batch

Elabora più cartelle automaticamente:

1. Vai su **Strumenti → Elaborazione batch** oppure fai clic sull'opzione batch
2. Nella finestra batch:
   - **Aggiungi cartelle**: seleziona le cartelle contenenti le sequenze di immagini
   - **Rimuovi selezionati**: rimuovi le cartelle selezionate dall'elenco
   - **Rimuovi tutte**: rimuovi tutte le cartelle
3. Configura le impostazioni di output:
   - **Crea una sottocartella nella cartella sorgente**: crea i risultati in ogni cartella di input
   - **Uguale alla cartella sorgente**: sovrascrive o salva accanto all'input
   - **Specifica la cartella di output**: indica la posizione di output
4. Controlla le impostazioni correnti di fusione/registrazione
5. Fai clic su **Avvia elaborazione batch** per iniziare

La finestra batch mostra l'avanzamento in tempo reale. Puoi annullare l'elaborazione in
qualsiasi momento.

### Note sull'elaborazione batch

- Usa le stesse impostazioni di fusione e registrazione dell'elaborazione manuale
- Ogni cartella viene elaborata in modo indipendente
- I nomi dei file includono data/ora e parametri per l'identificazione

---

## 13. Esportazione dei risultati

### Salvataggio di un singolo risultato

1. Seleziona nell'elenco output il risultato che vuoi salvare
2. Vai su **File → Salva** o premi `Ctrl+S`
3. Scegli formato e destinazione
4. Fai clic su Salva

### Formato di esportazione per trascinamento

Trascinando i risultati fuori dall'elenco output vengono scritti file immagine nel formato
scelto dal menu contestuale dell'elenco output (**Formato di trascinamento**: JPG, PNG o
TIFF). La scelta viene ricordata.

### Salvataggio dello stack registrato

1. Vai su **File → Salva stack → Stack registrato**
2. Scegli il formato di salvataggio:
   - **Salva come cartella**: salva ogni frame allineato come file immagine separato
   - **Salva come GIF**: crea una GIF animata della sequenza allineata

### Salvataggio dello stack di input

1. Vai su **File → Salva stack → Stack di input**
2. Scegli il formato di salvataggio:
   - **Salva come cartella**: salva i frame originali come immagini separate
   - **Salva come GIF**: crea una GIF animata della sequenza di input

### Formati di esportazione

- **JPG**: compresso, buon rapporto qualità/dimensione
- **PNG**: senza perdita, supporta la trasparenza
- **BMP**: non compresso, qualità massima
- **TIFF**: alta qualità, supporta i livelli
- **GIF**: formato per animazioni

---

## 14. Interfaccia a riga di comando

OpenFocus può funzionare senza interfaccia grafica — utile per pipeline scriptate e CI.
Da un ambiente Python (oppure accanto all'eseguibile pacchettizzato):

```bash
# Fuse a folder with the default Guided Filter
python main.py --input ./stack_folder --output ./result/fused.png

# DTCWT with ECC registration, 8 threads
python main.py -i ./stack_folder -o ./result/fused.png -m dtcwt -a ecc -t 8

# StackMFF-V3 forced to CPU with a custom tile size
python main.py -i ./stack_folder -o ./result/fused.png -m stackmffv3 --cpu --tile-size 512

# Explicit file list; video files are also accepted as input
python main.py -i img1.jpg img2.jpg img3.jpg -o fused.png -m gfgfgf

# Single multi-page TIFF Z-stack with a metadata-calibrated scale bar
python main.py -i ./zstack.tif -o fused.png -m dtcwt --scale-bar
```

| Opzione | Descrizione |
|--------|-------------|
| `--input, -i` | Cartella di immagini, file video oppure uno o più file immagine |
| `--output, -o` | Percorso dell'immagine di output (l'estensione determina il formato) |
| `--method, -m` | `stackmffv3` (predefinito), `none`, `openfocus_a`, `openfocus_b`, `openfocus_c`, `guided_filter`, `dct`, `dtcwt`, `gfgfgf` |
| `--align, -a` | `none` (predefinito), `homography`, `ecc`, `both`, `openfocus` |
| `--align-shift-limit` | Solo per l'allineamento `openfocus`: spostamento massimo accettato come frazione della dimensione del frame (predefinito 0.08; i frame oltre vengono lasciati non allineati) |
| `--exposure-match` | Uniforma la media di ogni frame a quella dell'ultimo frame prima della fusione (predefinito: attivo; `--no-exposure-match` lo disattiva). Guadagni calcolati sull'intero stack, così l'output a tasselli resta senza giunture; si applica a ogni metodo e al riferimento di luminosità dell'allineatore OpenFocus |
| `--base-frame` | Passaggio opzionale dopo la fusione: allinea le basse frequenze dell'output al frame `last`/`first` (predefinito `none`) |
| `--sort` | Ordine dei frame per `openfocus_a/b/c`: `asc` (predefinito, ordine dei file), `desc` (inverte una scansione da lontano a vicino), `off` |
| `--dust-map WHITE` | Correzione campo piatto (mappa polvere) da uno scatto fuori fuoco di un campo bianco |
| `--kernel, -k` | Dimensione del kernel (forzata dispari; per la famiglia openfocus è l'origine del raggio della finestra di varianza) |
| `--depth-smoothing SIGMA` | Solo `openfocus_b`: sigma gaussiano della transizione di profondità tra un frame e l'altro (predefinito 1.2; 0 mantiene il dettaglio più fine) |
| `--blend-smoothing SIGMA` | Solo `openfocus_a` / `openfocus_c`: sigma gaussiano della mappa dei pesi per frame (predefinito 2.0 per A, 3.0 per C; 0 mantiene la scelta per pixel più nitida) |
| `--threads, -t` | Limite dei thread; `0` = automatico (predefinito), i valori positivi limitano la concorrenza |
| `--resource-preference` | Modalità risorse: `balanced` (predefinito), `responsive` o `maximum` |
| `--batch-size` | Limite batch dei tasselli AI (predefinito 2); attualmente un tassello per chiamata |
| `--cpu` | Forza la CPU anche se una GPU è disponibile |
| `--downscale` | Larghezza di ridimensionamento per il rilevamento delle feature di registrazione |
| `--tile-size` | Dimensione del blocco dei tasselli per immagini grandi |
| `--output-dir, -d` | Modalità batch: fonde ogni cartella di input scrivendo `<folder>.png` per stack |
| `--scale-bar` | Disegna una barra di scala (calibrazione rilevata automaticamente dai metadati TIFF) |
| `--depth-map [path]` | Salva anche una mappa colorata della posizione di fuoco (profondità) (predefinito: `<output>_depth.png`) |
| `--depth-colormap` | Mappa colori per `--depth-map`: turbo, viridis, jet, gray, … (predefinito turbo) |
| `--scale-um` | Dimensione del pixel in µm/px, con precedenza sui metadati (implica `--scale-bar`) |

Codici di uscita: `0` successo, `1` errore di elaborazione, `2` errore di utilizzo.
Eseguendo `python main.py --help` vengono stampate tutte le opzioni; avviando con un
semplice percorso (`python main.py C:\stack`) si apre comunque la GUI con quella cartella caricata.

---

## 15. Scorciatoie da tastiera

### Operazioni sui file

| Scorciatoia | Azione |
|----------|--------|
| `Ctrl+O` | Apri cartella |
| `Ctrl+Shift+O` | Apri video |
| `Ctrl+S` | Salva risultato |
| `Ctrl+Shift+S` | Salva stack registrato |
| `Ctrl+W` | Svuota stack |
| `Ctrl+Q` | Esci |
| `Ctrl+R` | Avvia render |
| `Ctrl+Shift+R` | Confronta tutti |
| `Ctrl+T` | Ritocca risultato |
| `Ctrl+M` | Mappa di profondità |
| `Ctrl+E` | Attiva/disattiva il confronto Wipe |
| `Ctrl+Shift+B` | Elaborazione batch |
| `Ctrl+Z` / `Ctrl+Y` | Annulla / Ripeti (modifiche allo stack) |

### Navigazione

| Scorciatoia | Azione |
|----------|--------|
| `Freccia sinistra` | Frame precedente |
| `Freccia destra` | Frame successivo |
| `Rotella del mouse` | Zoom avanti/indietro |
| `Ctrl + rotella` | Zoom fine |
| `Doppio clic` | Passa tra vista adattata e 100% |

### Generali

| Scorciatoia | Azione |
|----------|--------|
| `Spazio` | Attiva/disattiva la modalità spostamento |
| `Canc` | Elimina l'elemento selezionato |

---

## 16. Risoluzione dei problemi

### Problemi comuni

#### «Nessuna immagine caricata»

- Assicurati di aver prima importato una cartella di immagini o un video
- Verifica che la cartella contenga formati di immagine supportati

#### I risultati della fusione appaiono sfocati o con immagini fantasma

- Attiva la registrazione delle immagini (scegli uno dal menu a discesa: Omografia / ECC / OpenFocus veloce)
- Assicurati che le immagini siano correttamente allineate prima della fusione
- Prova un algoritmo di fusione diverso
- Aumenta la dimensione del kernel per risultati più morbidi

#### AI non disponibile

- Installa PyTorch: `pip install torch torchvision`
- Per il supporto GPU, installa PyTorch con supporto CUDA
- Verifica con **Aiuto → Informazioni sull'ambiente**

#### Errori di memoria esaurita

- Attiva l'elaborazione a tasselli in **Impostazioni → Impostazioni tasselli**
- Riduci la dimensione del blocco dei tasselli
- Usa dimensioni del kernel più piccole
- Ridimensiona le immagini a dimensioni inferiori

#### La registrazione non riesce

- Prova un solo metodo di registrazione alla volta
- Riduci la larghezza di ridimensionamento in **Impostazioni → Impostazioni registrazione**
- Assicurati che le immagini abbiano contrasto e feature sufficienti

#### Elaborazione lenta

- Aumenta il numero di thread in **Impostazioni → Impostazioni thread**
- Usa una CPU più potente
- Riduci la risoluzione delle immagini
- Usa algoritmi di fusione più semplici (Filtro guidato invece di AI)

#### Avviso di Windows SmartScreen al primo avvio

I binari distribuiti non sono firmati. Fai clic su **Ulteriori informazioni → Esegui comunque**.
Il primo avvio dopo l'estrazione può richiedere 30–60 secondi.

#### macOS segnala che l'app «non può essere aperta» o «è danneggiata»

L'applicazione non è firmata né notarizzata. Dopo aver spostato `OpenFocus.app` in
Applicazioni, esegui:

```bash
xattr -cr /Applications/OpenFocus.app
```

oppure fai clic con il tasto destro sull'app e scegli **Apri**. Le build macOS distribuite
sono per Apple Silicon (M1/M2/M3/M4); i Mac Intel possono eseguire il codice sorgente.

### Come ottenere aiuto

1. **Informazioni sull'ambiente**: vai su **Aiuto → Informazioni sull'ambiente** per verificare l'installazione
2. **Contatta l'assistenza**: vai su **Aiuto → Contattaci** per le informazioni di supporto

---

## Appendice: dettagli sugli algoritmi di fusione

### Filtro guidato

Il filtro guidato esegue una levigatura con conservazione dei bordi usando un'immagine di
guida. Calcola trasformazioni lineari locali tra l'immagine di guida e quella di input, poi
le applica per produrre l'output. Questo metodo eccelle nel preservare i bordi riducendo il rumore.

### DCT (Discrete Cosine Transform)

La DCT trasforma le immagini nel dominio della frequenza, dove le decisioni di fusione
vengono prese in base all'ampiezza dei coefficienti. Le componenti ad alta frequenza (i
dettagli) vengono preservate, mentre quelle a bassa frequenza (le regioni uniformi) vengono
unite in base all'energia locale.

### DTCWT (Dual-Tree Complex Wavelet Transform)

La DTCWT fornisce una decomposizione multi-scala e multi-direzionale delle immagini.
Cattura le caratteristiche a scale e orientamenti diversi, permettendo regole di fusione
sofisticate che conservano sia la texture sia la struttura.

### GFG-FGF

Implementa Fu et al. (2024): la mappa di fuoco grossolana |I − mean(I)| viene arricchita
con il gradiente a quattro vicinati di tipo gaussiano — il modello 3×3 T = [1 2 1; 2 4 2; 1 2 1]/16
dà P = I⊗T, e G = (|I(x±1,y) − P|)/2 + (|I(x,y±1) − P|)/2, con i valori inferiori a
T0 = 0.005 scartati — poi raffinata da un filtro guidato rapido. Ogni pixel sceglie il frame
con il valore di fuoco maggiore (in caso di parità: misura di piccola regione Sobel 3×3)
e le mappe di decisione binarie vengono levigate dal filtro guidato rapido prima della
miscelazione pesata.

### Fusione AI (StackMFF-V3)

Un approccio di apprendimento profondo basato su una rete neurale addestrata su grandi
insiemi di coppie di immagini multi-fuoco. La rete apprende automaticamente strategie di
fusione ottimali, offrendo risultati allo stato dell'arte se configurata correttamente.

---

*OpenFocus è un progetto open source. Per aggiornamenti, segnalazioni di bug e richieste di funzionalità, visita il repository del progetto.*
