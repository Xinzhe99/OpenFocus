# Manuale utente di OpenFocus

## Indice

1. [Introduzione](#1-introduzione)
2. [Per iniziare](#2-per-iniziare)
3. [Panoramica dell'interfaccia principale](#3-panoramica-dellinterfaccia-principale)
4. [Importazione di immagini e video](#4-importazione-di-immagini-e-video)
5. [Visualizzazione e navigazione delle immagini](#5-visualizzazione-e-navigazione-delle-immagini)
6. [Vista di confronto Wipe (a tendina)](#6-vista-di-confronto-wipe-a-tendina)
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

OpenFocus è un'applicazione desktop professionale per la fusione di immagini multifocus, pensata per ricercatori, ingegneri e professionisti dell'elaborazione delle immagini. Elabora sequenze di immagini messe a fuoco a piani diversi con algoritmi avanzati di registrazione e fusione, e genera immagini completamente nitide con profondità di campo totale.

### Funzionalità principali

- **Fusione di immagini multifocus**: combina più immagini con punti di fuoco diversi in un'unica immagine completamente a fuoco
- **Dieci algoritmi di fusione**: filtro guidato, DCT, DTCWT, GFG-FGF, StackMFF V1/V2/V3, OpenFocus A/B/C, più la modalità solo allineamento
- **Registrazione delle immagini**: allinea sequenze disallineate con OpenFocus (veloce), ECC, omografia o la catena omografia → ECC
- **Sfondo automatico**: sostituire uno sfondo impilato irregolare con un fotogramma reale e vedere dove nessun fotogramma ha ripreso chiaramente la scena
- **Vista di confronto Wipe (a tendina)**: controlla l'allineamento e confronta i risultati con un divisore A/B
- **File di progetto**: ripristinano riferimenti di input e impostazioni riproducibili; i pixel del risultato si esportano separatamente
- **Modelli di elaborazione**: salvano e riapplicano impostazioni di fusione, registrazione e risorse a qualsiasi stack caricato
- **TIFF multipagina**: gli stack Z in un unico file vengono caricati pagina per pagina e riesportati in un solo file
- **Mappa di profondità**: mappe di posizione di fuoco per pixel con schemi di pseudocolore personalizzati
- **Barra di scala**: barre di scala di qualità editoriale con calibrazione automatica in µm/px
- **Recupero automatico dopo crash**: al riavvio viene proposta l'ultima sessione di lavoro
- **Multilingua**: dodici lingue dell'interfaccia — English, 中文, 日本語, Español, Deutsch, Français, Русский, 한국어, Português, Italiano, Nederlands, Polski — con rilevamento automatico della lingua di sistema
- **CLI headless e per agenti**: automatizza pipeline headless per la CI o controlla una finestra in esecuzione tramite il canale di controllo locale
- **Esportazione flessibile**: salva i risultati come immagini singole, cartelle o animazioni GIF
- **Trasformazioni delle immagini**: rotazione, capovolgimento e ridimensionamento degli stack
- **Etichette**: aggiungi testo allo stack sorgente, a quello allineato o a uscite specifiche
- **Impostazioni persistenti**: preferenze, lingua e stack aperti di recente sopravvivono ai riavvii

### Formati di input supportati

- File immagine: JPG/JPEG, PNG, BMP, TIFF, WebP e altri formati comuni
- File video: MP4, AVI, MOV, MKV, WMV, FLV e WEBM (decodificati in sequenze di immagini)
- Cartelle con sequenze di immagini numerate
- File RAW di fotocamera (DNG, CR2/CR3, NEF, ARW, RAF, ORF, RW2 e altri formati RAW comuni). La decodifica usa `rawpy` (LibRaw), installato da `pip install openfocus` e incluso nelle build Windows/macOS

Gli stack PNG trasparenti possono essere fusi nell'app desktop, nell'elaborazione batch desktop o nell'interfaccia a riga di comando, senza fusione raggruppata. Ogni fotogramma deve essere un PNG a 8 o 16 bit con le stesse dimensioni e la stessa profondità di bit, con la stessa maschera alfa. La maschera deve essere zero fuori da un unico rettangolo non vuoto e avere un'opacità costante diversa da zero in tutto il rettangolo. È consentita un'opacità costante parzialmente trasparente. Il rettangolo visibile viene fuso in modo indipendente, quindi i colori nascosti non possono influire sulla misura di nitidezza né sui confini dei tasselli. L'output PNG conserva le dimensioni originali della tela e l'opacità; i colori nascosti dell'output sono zero. Le viste sorgente e risultato del desktop mostrano la trasparenza su una scacchiera.

Carica a dimensione piena, scegli nessun allineamento, disattiva l'anteprima rapida e la ROI, quindi renderizza e salva come PNG. La modalità batch richiede un metodo di fusione e un output PNG senza salvare uno stack allineato. Allineamento, ridimensionamento o riduzione, correzione di campo piatto, mappe di profondità, confronto metodi, ritocco e barre di scala non sono supportati per i dati trasparenti, e le etichette non possono essere incise in un'esportazione. Un metodo di fusione che cambia le dimensioni del rettangolo viene rifiutato. Maschere variabili o con bordi sfumati, maschere diverse tra fotogrammi, input TIFF trasparente e output non PNG, inclusi GIF e TIFF multipagina, non sono supportati. La CLI rifiuta anche la pulizia del sensore e le mappe di profondità per input trasparenti.

Ad esempio, l'equivalente CLI è `python main.py --input ./png-stack --output ./result.png --method guided_filter --align none`.

### Formati di output supportati

In **Salva con nome**, il TIFF normale salva il risultato come immagine singola; il TIFF a livelli include i fotogrammi sorgente come livelli. Linear DNG è sempre elencato, ma per salvarlo serve un risultato lineare della fotocamera senza modifiche incorporate, ottenuto con **Apri RAW nativo** e dotato della calibrazione della fotocamera.

- Immagini singole: JPG, PNG, BMP, TIFF, WebP
- Sequenze di immagini: salvate come cartelle
- Animazioni: formato GIF
- Modelli 3D: OBJ con MTL, texture PNG e metadati JSON

---

## 2. Per iniziare

### Avvio dell'applicazione

Il modo preferito per installare una build pacchettizzata è il sito del progetto, **https://openfocus.top**: collega l'installer Windows corrente, la build macOS e il pacchetto Linux, con script di installazione per il terminale. Preferiscilo alla ricompilazione: quegli artefatti sono l'output esatto della pipeline di rilascio. `pip install openfocus` installa solo l'interfaccia a riga di comando headless (la wheel non dipende deliberatamente da PyQt6); usa una build desktop o una copia del codice sorgente per la GUI.

Per installare da terminale, visualizza prima l'anteprima del comando e poi eseguilo senza `--dry-run`: su macOS o Linux, `curl -fsSL https://openfocus.top/install.sh | sh -s -- --dry-run` e poi `curl -fsSL https://openfocus.top/install.sh | sh` (su Linux installa la CLI); su Windows, `irm https://openfocus.top/install.ps1 | iex` da PowerShell.

Dopo l'installazione, avvia OpenFocus in uno dei seguenti modi:

1. **Windows (eseguibile)**: fai doppio clic sul collegamento o sull'eseguibile di OpenFocus
2. **macOS**: trascina `OpenFocus.app` in Applicazioni e aprilo da lì
3. **Sorgente Python**: esegui `pip install -r requirements.txt`, poi `python main.py` nell'ambiente attivato

Il primo avvio apre la finestra di benvenuto, dove puoi scegliere il tema e caricare lo stack demo incluso. Un profilo nuovo usa il tema chiaro per impostazione predefinita; la scelta viene ricordata.

### Flusso di lavoro rapido

1. Importa la sequenza di immagini (File → Apri cartella o trascina e rilascia)
2. Configura le impostazioni di fusione nel pannello destro
3. La registrazione parte su **ECC**; scegli **Nessun allineamento** se i fotogrammi sono già allineati
4. Regola la dimensione del kernel se necessario
5. *(facoltativo)* Apri **Impostazioni di rendering** e seleziona **Anteprima rapida** per provare prima i parametri su una bozza ridotta
6. Fai clic su **Avvia render** per iniziare l'elaborazione
7. Esporta i risultati

---

## 3. Panoramica dell'interfaccia principale

L'interfaccia di OpenFocus comprende tre aree principali:

### Pannello sinistro: area di visualizzazione

La parte sinistra della finestra mostra le immagini in una vista divisa:

- **Pannello immagine sorgente (in alto)**: mostra la sequenza di immagini di input
- **Pannello immagine risultato (in basso)**: mostra i risultati di fusione o registrazione
- **Cursore di navigazione**: usa il cursore sotto ogni pannello per scorrere i fotogrammi
- **Etichette informative**: mostrano la posizione del fotogramma corrente e le dimensioni dell'immagine

Puoi ridimensionare i pannelli trascinando il separatore tra di essi.

### Pannello destro: pannello di controllo

Il pannello destro raccoglie tutte le opzioni di configurazione, dall'alto verso il basso:

1. **Impostazioni di fusione**: selezione dell'algoritmo e dei relativi parametri
2. **Impostazioni di registrazione**: attivazione o disattivazione dei metodi di allineamento
3. **Pulsanti di azione**: Avvia render e le azioni secondarie Reimposta, Impostazioni di rendering e rendering di tutti i metodi. La finestra delle impostazioni raccoglie anteprima rapida, ordine dei fotogrammi, abbinamento dell'esposizione, rimozione della deriva di luminosità e nitidezza dell'output
4. **Elenco immagini sorgente**: tutte le immagini caricate con i nomi dei file
5. **Elenco output**: i risultati generati

### Barra dei menu

La barra dei menu superiore dà accesso a tutte le funzioni:

- **File**: operazioni di importazione ed esportazione, progetti, modelli di elaborazione ed elenco file recenti
- **Modifica**: trasformazioni delle immagini, ritocco ed etichette
- **Strumenti**: elaborazione batch, confronto metodi, mappa di profondità, correzione di campo piatto, barra di scala ed etichette
- **Impostazioni**: impostazioni avanzate (prestazioni, registrazione e diagnostica di allineamento, pulizia del sensore, nomi output, IA e avvio), accelerazione GPU, tema e lingua
- **Aiuto**: informazioni sull'ambiente, contatto, controllo aggiornamenti e cartella dei log

---

## 4. Importazione di immagini e video

### Apertura di una cartella di immagini

1. Vai su **File → Apri cartella** o premi `Ctrl+O`
2. Seleziona una cartella contenente la sequenza di immagini
3. Compare una finestra per la scala di caricamento: scegli la scala (100% = risoluzione piena)
4. Fai clic su OK per caricare le immagini

Seleziona **Non chiedere più** in quella finestra per caricare sempre alla scala scelta senza richiesta; **Impostazioni → Impostazioni avanzate → Chiedi la scala di caricamento all'importazione** riattiva la richiesta. Uno stack già caricato può essere ridotto in seguito con **Modifica → Ridimensiona**.

Il software esegue automaticamente queste operazioni:

- Rileva e ordina i file immagine per nome, in ordine naturale (ad esempio `frame2` prima di `frame10`)
- Mostra la prima immagine nel pannello sorgente
- Popola l'elenco delle immagini sorgente con tutti i fotogrammi

### Apertura di file video

1. Vai su **File → Apri video** o premi `Ctrl+Shift+O`
2. Seleziona un file video (MP4, AVI, MOV, MKV, WMV, FLV, WEBM)
3. Il video viene decodificato automaticamente in fotogrammi singoli
4. Ogni fotogramma diventa parte dello stack di immagini

### File di progetto

File → Salva progetto memorizza i riferimenti di input e le impostazioni riproducibili in un `.ofproj`: ordine esatto dei fotogrammi (incluse le pagine TIFF e i fotogrammi duplicati), esclusioni, rotazioni e capovolgimenti, scala di caricamento, parametri di rendering per metodo, etichette, stile della barra di scala e fotogramma corrente. I pixel del risultato e del ritocco devono essere esportati separatamente. Le operazioni non ricostruibili dai file sorgente richiedono di salvare prima un nuovo stack di input.

File → Apri progetto ricostruisce i fotogrammi registrati prima di sostituire lo stack corrente. Se le sorgenti sono state spostate, scegli la nuova radice e conferma la corrispondenza completa tra vecchi e nuovi percorsi. Salva di nuovo il progetto per conservare i riferimenti spostati.

Un progetto si comporta come un documento: la barra del titolo mostra il suo nome con un `*` finché ci sono modifiche non salvate, e chiudere l'applicazione o aprire un altro progetto chiede conferma prima di scartarle. Le immagini sorgente sono referenziate, non copiate.

### Modelli di elaborazione

Un modello di elaborazione contiene impostazioni, mai percorsi sorgente. **File → Salva modello di elaborazione…** scrive un `.oftemplate` con il metodo e i parametri di fusione correnti, il metodo e i parametri di registrazione, le opzioni di luminosità per metodo, il numero di thread, le impostazioni dei tasselli, la larghezza di riduzione della registrazione, la dimensione batch IA di StackMFF, la nitidezza, l'ordine inverso e l'indicatore di riduzione del rumore del sensore. Il percorso di una maschera difetti non viene deliberatamente memorizzato.

**File → Applica modello di elaborazione…** applica tali impostazioni allo stack già caricato; non sostituisce le sorgenti correnti. Il selettore accetta anche un progetto `.ofproj` (le impostazioni vengono estratte senza richiederne le sorgenti) e una ricetta `.json` che contiene impostazioni di progetto. Applicare un modello contrassegna il progetto come modificato e può cambiare parametri degli algoritmi e impostazioni delle risorse, quindi controlla il pannello destro prima di renderizzare. Le ricette restano il formato headless e includono il percorso della maschera difetti e un'impronta.

### Stack demo

La finestra di benvenuto offre **Carica stack demo**: sei foto di esempio incluse, provenienti da un vero stack di messa a fuoco, per provare subito il flusso completo senza immagini proprie. Utile anche per verificare un'installazione.

### Curva di nitidezza

Una piccola curva sopra il cursore di navigazione sorgente mostra la nitidezza di ogni fotogramma (normalizzata). I fotogrammi sfocati appaiono come valli marcate da un punto rosso. Fai clic in qualsiasi punto della curva per saltare a quel fotogramma; passando il cursore viene mostrato il numero del fotogramma.

### Stack TIFF multipagina

I software di microscopia (ZEN, NIS-Elements, ImageJ/Fiji) esportano spesso un intero stack Z come UN unico file TIFF multipagina. OpenFocus legge ogni pagina di quel file come fotogramma individuale dello stack: trascina o apri il file e lo stack appare completo (i fotogrammi sono denominati `<file>_page_001` e così via). I TIFF a pagina singola si comportano esattamente come prima. Funziona anche il contrario: **File → Salva stack → (Stack registrato / Stack di input) → Salva come TIFF multipagina** scrive l'intero stack in un unico `.tif`, conservando la profondità a 16 bit.

### Barra di scala (figure per pubblicazioni)

Se la destinazione scelta non ha metadati utilizzabili, viene selezionata la calibrazione manuale ed è richiesta una dimensione pixel valida.

Strumenti → Barra di scala… configura una barra di scala impressa nelle immagini esportate. La dimensione del pixel (µm/px) viene rilevata automaticamente dai metadati TIFF (convenzione `unit=micron` di ImageJ, `PhysicalSizeX` di OME-TIFF o unità di risoluzione esplicita in centimetri), così gli stack di microscopia (inclusi i TIFF multipagina) si calibrano da soli. La risoluzione di stampa (72/300 dpi e simili) *non* viene deliberatamente trattata come calibrazione, e un valore inserito manualmente prevale sempre sui metadati rilevati: una barra sbagliata ma credibile in una figura è peggio di nessuna barra. Scegli l'angolo e il colore (Auto inverte rispetto allo sfondo); la finestra mostra un'anteprima dal vivo. Con la barra attiva, ogni esportazione (immagine singola, stack in cartelle, TIFF multipagina, GIF, esporta tutto, batch) porta una barra in stile pubblicazione: una lunghezza arrotondata della serie 1-2-5 (~12% della larghezza dell'immagine) con testo contornato e leggibile. La CLI la supporta con `--scale-bar` (calibrazione automatica) e `--scale-um <um/px>`.

L'anteprima viene disegnata con la geometria nativa in pixel della destinazione prima di adattarla all'area visibile. Adatta, 100%, panoramica e navigazione tra fotogrammi cambiano solo la vista. Applica conferma una singola modifica annullabile; Annulla conserva la calibrazione precedente. Le uscite storiche mantengono la propria calibrazione e il proprio stile quando viene caricato un altro stack sorgente.

Per rimuoverla, usa **Strumenti → Barra di scala → Rimuovi barra di scala**: *Dallo stack sorgente* e *Dal risultato* limitano l'ambito «Applica a» della finestra alla tela rimanente, mentre *Da ovunque* disattiva la barra. Entrambe le tele e il confronto A/B vengono ridisegnati subito; rimuovere la barra da una tela che non l'ha mai mostrata lascia l'impostazione invariata.

### Mappa di profondità (posizione di fuoco)

**Strumenti → Mappa di profondità…** stima le posizioni di fuoco usando la misura di attività del metodo selezionato. Non è una superficie misurata né una mappa esatta dell'origine dei pixel: i metodi a piramide/wavelet stimano l'attività e le varianti StackMFF riportano una posizione di fotogramma stimata. Scegli i fotogrammi attualmente selezionati o la sorgente registrata dell'output selezionato. Una provenienza storica mancante non viene mai sostituita dallo stack corrente. Sorgente e texture restano legate a quell'istantanea. Le mappe di profondità non sono disponibili per stack PNG trasparenti, né nella GUI né nella CLI.

Tutti e sette i metodi compatibili con la profondità — DCT, DTCWT, GFG-FGF, OpenFocus B, OpenFocus C, StackMFF V2 e StackMFF V3 — richiedono un clic esplicito su **Calcola**. StackMFF V1 non ha una mappa di profondità. Aprire la finestra o cambiare misura, sorgente, registrazione o parametri non avvia mai il calcolo. L'editor dei parametri compare subito sotto il selettore della misura: il kernel mediano DCT e il kernel iniziale GFG usano valori dispari da 1 a 51 (7 predefinito). OpenFocus B usa un raggio di risposta di fuoco e una levigatura della profondità; OpenFocus C usa **Levigatura della miscelazione** per levigare la mappa di profondità, senza raggio. I livelli di decomposizione DTCWT sono modificabili qui come bozza locale, inizializzata dai parametri della sorgente. StackMFF V3 mostra la propria modalità di output IA (miscelazione soft predefinita o scelta netta) e una spiegazione al posto di un kernel numerico: la mappa riporta una posizione di fotogramma continua, non una distanza fisica. La miscelazione soft usa la posizione ponderata per probabilità; la scelta netta usa il fotogramma più probabile per pixel. StackMFF V2 non ha parametri e riporta lo strato focale campionato dalla rete. Le bozze partono dai parametri dell'output corrispondente o dalle impostazioni memorizzate del metodo nella finestra principale. Sono ricordate per sorgente e metodo finché la finestra resta aperta; modificarle non cambia la finestra principale, il progetto né l'output registrato.

Cambiare le impostazioni di calcolo contrassegna la mappa 2D precedente come **non aggiornata**, svuota la superficie 3D e blocca ogni esportazione, incluse quelle colorizzate, grezze, fisiche e OBJ, finché un nuovo calcolo non riesce. Un calcolo attivo viene annullato in modo cooperativo; al termine dell'annullamento premi di nuovo **Calcola**. Nessun calcolo successivo viene messo in coda. Calcola resta visibile ed è disabilitato durante calcolo, annullamento o esportazione.

- **Mappa colori**: scala di grigi (standard editoriale), 15 palette scientifiche (Turbo, Viridis, Magma, …) e le tue — *Nuovo schema personalizzato…* costruisce un gradiente da un numero qualsiasi di fermate di colore (doppio clic su una fermata per modificarne posizione e colore); gli schemi personalizzati vengono salvati e proposti nell'elenco.
- **Inverti / Gamma / Levigatura**: inverte l'asse di profondità, ribilancia il tono (gamma < 1 schiarisce gli indici bassi) e rimuove il rumore puntiforme con una passata che rispetta i bordi.
- **Sovrapposizione**: disponibile solo per l'output associato con coordinate corrispondenti.
- **Esportazione grezza**: il TIFF a 16 bit memorizza un indice normalizzato codificato da 0 a 65535; non sono dati float senza perdita né una distanza fisica. JPEG/BMP vengono rifiutati.
- **TIFF indici e TIFF immagine alternativa**: esporta la stima di profondità corrente come indici di fotogramma a 16 bit senza segno o come immagine assemblata dai pixel sorgente. I valori 1–N si riferiscono all'ordine esatto dei fotogrammi usato come input del calcolo di profondità, dopo registrazione e riordino; 0 marca i pixel non validi. Sono supportati al massimo 65.535 fotogrammi. Le posizioni frazionarie stimate vengono arrotondate al fotogramma più vicino, con i valori intermedi arrotondati per eccesso. Questi indici stimano la posizione di fuoco; non sono la vera provenienza dei pixel di un risultato fuso. L'immagine alternativa seleziona il pixel del fotogramma di input di profondità corrispondente in ogni posizione valida, con i pixel non validi impostati a zero. Quando un input è a 16 bit, i valori a 8 bit vengono promossi a 16 bit moltiplicandoli per 257. Le esportazioni usano la ROI della mappa di profondità e conservano le aree non valide a zero.
- **Profondità fisica**: inserisci inizio Z, passo positivo, unità e direzione di acquisizione, poi conferma la mappatura di cattura originale. Passare tra lo stack corrente e un output cancella la conferma, anche se nomi e indici coincidono; riconfermala prima di esportare. I fotogrammi esclusi mantengono i loro intervalli. Il TIFF Float32 memorizza micrometri, NaN per i dati mancanti e i metadati di origine. Un OBJ fisico richiede anche la calibrazione XY; usa una scala Z fisica 1:1.
- **Modello 3D / Texture 3D**: scegli il lato più lungo della griglia (600 vertici predefiniti). Ogni nuova finestra usa **Automatico (colore reale)**: usa l'output associato solo se identità e coordinate/ROI coincidono con la mappa completata; altrimenti usa il primo fotogramma partecipante di quel calcolo completato, dopo registrazione ed eventuale riordino. Viene mostrato il nome dell'output o del fotogramma risolto. Puoi scegliere esplicitamente l'output corrispondente, un fotogramma partecipante o **Pseudocolore di profondità**; le immagini non disponibili vengono segnalate, mai sostituite con output non correlati. Questa texture è condivisa da anteprima 3D e PNG di texture dell'OBJ. La mappa 2D mantiene la propria palette e sovrapposizione: cambiare palette o Gamma non ricolora una texture a colori reali. Lo pseudocolore esplicito segue palette, Gamma e inversione. L'anteprima stabilizzata e l'esportazione condividono lo stesso campo di altezza campionato, senza levigatura nascosta aggiuntiva. Il rilievo relativo è espresso in unità immagine.
- **Esportazione modello**: OBJ, MTL, texture PNG e metadati JSON formano un gruppo di file; tienili insieme. L'esportazione avviene in background. I conflitti vengono controllati per ogni membro, la preparazione è organizzata per fasi, l'annullamento conserva i file esistenti e un errore di conferma annulla le sostituzioni. Salvare un modello non esporta la sua immagine di fusione padre né rimuove la protezione delle modifiche non salvate di quell'immagine.

Il calcolo avviene in background con avanzamento e annullamento; i metodi IA riutilizzano lo stesso tiling sicuro per la memoria della fusione IA, quindi gli stack profondi funzionano. L'equivalente CLI è `--depth-map [path]` con `--depth-colormap`; scrive solo la mappa colorizzata (predefinito `<output>_depth.png`), funziona solo per i metodi compatibili con la profondità sopra e viene rifiutato per input trasparenti o raggruppati. La GUI esporta inoltre profondità grezza, indici di fotogramma, immagini alternative, profondità fisica e modelli 3D.

### Recupero automatico dopo un arresto anomalo

OpenFocus mantiene continuamente un'istantanea della sessione di lavoro (stack, impostazioni, etichette, fotogramma corrente). Se l'applicazione si chiude in modo imprevisto — crash, perdita di alimentazione o chiusura forzata — il successivo avvio lo rileva e offre **Ripristina sessione** per tornare esattamente dov'eri. Le chiusure pulite non mostrano mai la richiesta. I dati di recupero si trovano accanto alle impostazioni (le installazioni portatili li tengono vicino all'eseguibile, così viaggiano con te).

### File recenti

Ogni stack aperto viene ricordato in **File → Apri recenti** (fino a 8 voci, numerate nel sottomenu). Fai clic su una voce per riaprirla; non esiste una scorciatoia da tastiera globale per i file recenti. Riaprire lo stesso stack lo porta in cima; le voci i cui file non esistono più possono essere rimosse con una conferma, e **Svuota elenco recenti** svuota il menu.

### Trascinamento

Puoi anche trascinare una cartella direttamente sulla finestra dell'applicazione per importare immagini. Trascinare altri file su uno stack esistente li aggiunge in coda (dopo un controllo delle dimensioni).

### Orientamento EXIF

I file JPEG/TIFF di fotocamera con tag EXIF Orientation vengono ruotati o capovolti automaticamente durante l'importazione, così gli scatti verticali si caricano dritti.

### Sorgenti a 16 bit

Gli input PNG/TIFF a 16 bit vengono rilevati automaticamente e conservati end-to-end: la fusione avviene a profondità piena e i risultati possono essere salvati di nuovo come PNG/TIFF a 16 bit (le esportazioni JPG/BMP vengono convertite a 8 bit, poiché questi formati non possono memorizzare 16 bit).

I file RAW di fotocamera vengono sviluppati a 16 bit (bilanciamento del bianco della fotocamera, gamma sRGB) con lo schiarimento automatico disattivato, così ogni fotogramma dello stack riceve la stessa mappatura di esposizione; un'esposizione automatica per fotogramma farebbe sfarfallare la luminosità. Salva come TIFF o PNG per conservare i dati a 16 bit.

### Profili colore

Le esportazioni singole PNG, TIFF, JPEG e WebP conservano il profilo ICC RGB quando tutti i fotogrammi sorgente hanno lo stesso profilo incorporato. Gli stack misti o senza profilo vengono esportati senza profilo ICC. OpenFocus conserva il profilo; non converte i pixel sorgente in uno spazio colore comune né applica gestione del colore all'anteprima. Converti le immagini con profili diversi in uno spazio comune prima di caricarle. L'input RAW viene sviluppato in sRGB e non eredita il profilo della fotocamera.

Per un TIFF multipagina, tutte le pagine del file devono avere lo stesso profilo RGB valido; altrimenti i loro fotogrammi sono trattati come senza profilo. Anche i fotogrammi video sono trattati come senza profilo. I fotogrammi aggiunti e quelli selezionati per un rendering contribuiscono al suo profilo, che resta associato a quel risultato per le esportazioni successive.

### Svuotamento dello stack

Per rimuovere tutte le immagini caricate, vai su **File → Svuota stack** o premi `Ctrl+W`.

---

## 5. Visualizzazione e navigazione delle immagini

### Navigazione nello stack di immagini

- **Cursore**: trascina il cursore sotto il pannello sorgente per scorrere i fotogrammi
- **Elenco file**: fai clic su un nome file nell'elenco delle immagini sorgente del pannello destro; quando l'elenco ha il focus, Su/Giù spostano la selezione e il pannello sorgente la segue
- **Tastiera**: non esiste una scorciatoia globale fotogramma precedente/successivo; porta prima il focus sull'elenco sorgente

### Esame dei fotogrammi simili

Fai clic su **Esamina fotogrammi** sotto l'intestazione dell'elenco sorgente per trovare fotogrammi adiacenti con pixel quasi identici. Il pulsante diventa **Annulla esame** durante l'analisi. Ogni riga suggerita indica il fotogramma di riferimento e la differenza; ispeziona entrambi prima di usare la casella sorgente per escludere un fotogramma dalla fusione. I suggerimenti non modificano mai le caselle né eliminano file. Questo confronto prudente può non rilevare fotogrammi simili; i cambiamenti limitati a una piccola regione a fuoco non vengono mediati. I risultati vengono scartati se lo stack cambia. I suggerimenti sono mostrati solo quando sia il fotogramma suggerito sia il suo riferimento sono inclusi; cambiare una casella aggiorna i suggerimenti visualizzati.

### Zoom e spostamento

- **Zoom avanti/indietro**: rotella del mouse sull'immagine
- **Zoom fine**: tieni premuto `Ctrl` e usa la rotella per incrementi più piccoli
- **Adatta alla finestra**: doppio clic sull'immagine per alternare adattamento e vista al 100%
- **Spostamento**: fai clic e trascina per muovere l'immagine nel pannello

### Navigazione sincronizzata

Quando navighi nel pannello sorgente, il pannello risultato mostra il fotogramma di risultato corrispondente, se disponibile.

---

## 6. Vista di confronto Wipe (a tendina)

La vista Wipe sovrappone due immagini in un unico riquadro con un divisore verticale trascinabile, rendendo direttamente visibili disallineamenti e differenze di qualità ai due lati.

### Apertura della vista Wipe

1. Renderizza almeno un risultato (fusione o registrazione)
2. Premi l'interruttore **Wipe** nell'angolo superiore destro della barra del titolo del pannello risultato
3. L'area risultato passa alla vista di confronto; premi di nuovo **Wipe** per tornare

### Lettura della vista

- **Lato A (a sinistra del divisore)**: un fotogramma sorgente — per impostazione predefinita segue il cursore del pannello sorgente
- **Lato B (a destra del divisore)**: il rendering più recente
- **Divisore bianco + maniglia rotonda**: trascina orizzontalmente per spostare la divisione
- Trascinare i bordi che attraversano il divisore rivela errori di registrazione difficili da vedere affiancati

### Scelta di cosa confrontare

La barra in basso nella vista di confronto ha un gruppo per lato:

- **Lato A**: seleziona **Segui sorgente** (predefinito) per seguire il cursore del pannello sorgente, oppure deseleziona per bloccare il lato A su qualsiasi fotogramma con il proprio cursore
- **Lato B**: seleziona **Ultimo risultato** (predefinito) per mostrare sempre il rendering più recente, oppure deseleziona per scorrere la cronologia output con il proprio cursore

Entrambi i lati si aggiornano dal vivo: nuovi rendering, spostamenti del cursore sorgente ed eliminazioni di fotogrammi si riflettono subito.

### Zoom e spostamento

La vista di confronto condivide un'unica trasformazione per entrambe le immagini:

- **Zoom**: rotella del mouse (zoom intorno al cursore)
- **Spostamento**: clic e trascinamento ovunque tranne sul divisore
- **Reimposta**: doppio clic

La modalità Wipe e la modalità ROI condividono il pannello risultato: entrare in una esce dall'altra. Svuotare lo stack esce anche dalla modalità Wipe.

---

## 7. Registrazione delle immagini

La registrazione corregge il disallineamento spaziale tra i fotogrammi dello stack. È essenziale quando le immagini presentano piccoli spostamenti o cambi di prospettiva.

### Metodi di registrazione

#### ECC (Enhanced Correlation Coefficient)

- Usa un'ottimizzazione per trovare il miglior allineamento in base alla correlazione
- Adatto a disallineamenti sottili
- Precisione sotto il pixel
- Più lento dell'omografia, ma più preciso

#### Omografia

- Basata sulla corrispondenza di punti caratteristici (SIFT/ORB)
- Gestisce trasformazioni geometriche più ampie
- Più veloce dell'ECC per un allineamento iniziale
- Adatta a immagini con cambi di prospettiva significativi

#### OpenFocus (veloce)

- Allineamento per similarità limitata (traslazione + rotazione + scala), con equalizzazione della luminosità per fotogramma e ritaglio automatico
- Cerca trasformazioni di similarità limitate usando differenze di immagine campionate e applica warp con interpolazione Lanczos
- I fotogrammi con uno spostamento superiore all'**8% della dimensione del fotogramma restano non allineati** invece di essere corretti parzialmente. Modifica il limite con `--align-shift-limit` nella CLI
- Per geometrie non rigide o estreme, scegli ECC o omografia
- **Gestione dei fotogrammi rifiutati**: per impostazione predefinita si continua con il fotogramma non allineato. Scegli **Interrompi su fotogramma rifiutato** per far fallire il rendering o l'elemento batch prima di produrre un risultato quando la qualità del rilevatore è inferiore a 0,06 o il movimento stimato supera il limite di spostamento configurato, 8° di rotazione o un cambio di scala del 5%. I fotogrammi sono numerati da 1 nell'ordine dello stack elaborato. Il riferimento geometrico è il fotogramma 1 per impostazione predefinita; una corrispondenza diretta debole può essere ritentata sul fotogramma adiacente più vicino al riferimento. Questa politica rileva le stime rifiutate, non tutte le corrispondenze errate: bassa texture, pattern ripetuti, occlusione, movimento ampio o cambi di fuoco possono ancora ingannare la registrazione. Ispeziona il risultato.
- Nella CLI, seleziona la stessa politica con `--align openfocus --align-param 'failure_policy="fail"'` (usa le virgolette adatte alla shell). Gli elementi batch usano la politica scelta in modo indipendente. Il valore predefinito è `"continue"`.
- Imposta **Fotogramma di riferimento** su un numero basato su 1 nell'ordine dello stack incluso ed elaborato per ancorare la geometria a quel fotogramma. L'ordine viene conservato; l'equalizzazione della luminosità usa un proprio riferimento. Ogni stack batch deve contenere quel numero, altrimenti l'elemento fallisce. Esempio CLI: `--align openfocus --align-param reference_frame=3`. Scegliere un riferimento non corregge la parallasse né una corrispondenza scadente; ispeziona il risultato.

I report di allineamento identificano il riferimento e l'esito di ogni fotogramma. Un ripiego significa che il metodo non è riuscito a stabilire l'allineamento o dipende da una stima precedente fallita; la ragione riportata distingue stime deboli e movimento fuori dai limiti quando noto. I valori di qualità sono specifici del metodo, non una garanzia di accuratezza. Le trasformazioni mappano le coordinate sorgente sulla tela di riferimento della fase di registrazione prima del ritaglio comune; i metodi compositi riportano le loro fasi separatamente. Anche omografia ed ECC accettano il parametro facoltativo `failure_policy="fail"` per fermarsi quando i loro algoritmi rilevano un errore. Il valore predefinito resta `"continue"`; ispeziona i risultati anche quando l'allineamento riesce.

Per ispezionare un rendering, attiva **Impostazioni → Impostazioni avanzate → Registrazione → Raccogli diagnostica di allineamento** e renderizza di nuovo. L'allineamento viene ricalcolato invece di riutilizzare le immagini in cache. Lo stato di completamento evidenzia i ripieghi; apri **Impostazioni → Impostazioni avanzate → Registrazione → Diagnostica di allineamento** per il report per fotogramma. Mostra le ragioni di errore disponibili, i punteggi specifici del metodo, i conteggi delle corrispondenze, i fotogrammi di riferimento e le trasformazioni. Numeri e nomi di riferimento si riferiscono all'elenco sorgente originale, anche quando i fotogrammi sono esclusi. I rendering batch e raggruppati non forniscono questo report.

### Attivazione della registrazione

1. Nel pannello destro, individua il gruppo **Registrazione**
2. Scegli un metodo dall'elenco: **Nessun allineamento**, **OpenFocus (veloce)**, **Allinea (ECC)**, **Allinea (omografia)** o **Omografia → ECC** (concatenato)
3. Per OpenFocus (veloce), scegli un fotogramma di riferimento. In **Impostazioni → Impostazioni avanzate → Registrazione**, scegli **Gestione fotogrammi rifiutati**; la modifica viene applicata ai metodi di registrazione
4. **Raccogli diagnostica di allineamento** ora si trova in Impostazioni avanzate (vedi sopra)

### Quando usare la registrazione

- Le immagini presentano un disallineamento visibile
- Le immagini sono state scattate a mano libera
- La scena ha parallasse (gli oggetti vicini si spostano tra i fotogrammi)
- I risultati mostrano artefatti fantasma o sdoppiamento

### Impostazioni di registrazione

Le impostazioni aggiuntive si trovano in **Impostazioni → Impostazioni avanzate → Registrazione**:

- **Larghezza di riduzione**: risoluzione di pre-elaborazione per la registrazione (1024 px predefiniti)
- Valori più bassi = elaborazione più veloce, precisione eventualmente inferiore
- Valori più alti = elaborazione più lenta, precisione eventualmente superiore
- **Raccogli diagnostica di allineamento** e **Cache di registrazione** si trovano nella stessa sezione

---

## 8. Metodi di fusione delle immagini

OpenFocus offre dieci algoritmi di fusione (più una modalità solo allineamento). Un profilo nuovo parte su **StackMFF V3** con registrazione **ECC**; gli elenchi seguenti descrivono tutte le scelte. Ognuno ha caratteristiche diverse adatte a vari tipi di immagine.

### Filtro guidato

- **Algoritmo**: filtro guidato per una fusione che preserva i bordi
- **Ideale per**: uso generale, buon equilibrio tra velocità e qualità
- **Vantaggi**: buona conservazione dei bordi, elaborazione rapida
- **Parametro**: dimensione del kernel (regolabile da 1 a 51, solo valori dispari)
  - Kernel più piccolo: più dettaglio locale, possibile rumore maggiore
  - Kernel più grande: risultati più morbidi, migliore soppressione del rumore

### DCT (Discrete Cosine Transform)

- **Algoritmo**: fusione nel dominio della frequenza tramite coefficienti DCT
- **Ideale per**: immagini ricche di texture, imaging scientifico
- **Vantaggi**: conserva le texture fini, base matematica solida
- **Parametro**: la dimensione del kernel controlla la finestra di elaborazione

### DTCWT (Dual-Tree Complex Wavelet Transform)

- **Algoritmo**: fusione nel dominio wavelet multi-scala
- **Ideale per**: scene complesse con più livelli di dettaglio
- **Vantaggi**: eccellente analisi multi-scala, buona direzionalità
- **Regola di fusione**: una decisione per pixel e livello per tutti gli orientamenti e canali colore (N vie, coerenza a maggioranza verificata); la banda passa-basso segue la decisione più grossolana invece di mediare tutti i fotogrammi
- **Parametro**: livelli wavelet (1–6, 4 predefiniti) nel pannello destro. Il metodo non usa un kernel spaziale, quindi il cursore del kernel è nascosto.

### GFG-FGF (gradiente a quattro vicini di tipo gaussiano + filtro guidato rapido)

- **Algoritmo**: implementazione di Fu et al. (*Laser & Optoelectronics Progress*, 2024): una mappa di fuoco per differenza di sfocatura |I − mean(I)| viene arricchita con l'operatore gradiente a quattro vicini di tipo gaussiano (GFG), affinata con un filtro guidato rapido, decisa per pixel (i pareggi sono risolti da una misura di piccola regione Sobel 3×3), e le mappe di decisione binarie vengono levigate di nuovo dal filtro guidato rapido prima della miscelazione pesata.
- **Ideale per**: microscopia e immagini con regioni a fuoco nette
- **Vantaggi**: veloce, confini delle regioni a fuoco nitidi; la memoria non cresce con il numero di fotogrammi
- **Parametro**: kernel = dimensione del filtro medio della mappa di fuoco grossolana

### StackMFF (V1, apprendimento profondo)

- **Algoritmo**: una rete di attenzione a singola scala miscela la luminanza delle sorgenti con pesi softmax; Cb/Cr provengono dal fotogramma sorgente con luminanza più vicina a quella fusa
- **Ideale per**: fusione neurale di un intero stack di messa a fuoco
- **Vantaggi**: nessun parametro e colori entro la gamma; la luminanza fusa è indipendente dall'ordine dei fotogrammi, ma la crominanza può variare in caso di parità delle distanze di luminanza
- **Nota**: circa 156.172 parametri; gli stack grandi traggono vantaggio dalla GPU

### StackMFF V2 (apprendimento profondo)

- **Algoritmo**: una rete di attenzione a singola scala normalizza la profondità di fuoco attesa e campiona il fotogramma sorgente più vicino. Gli indici di profondità dipendono dall'ordine dello stack. **I fotogrammi devono mantenere il loro ordine di fuoco originale** — l'inversione è disattivata per questo metodo
- **Mappe di profondità**: supportate; riporta lo strato focale campionato
- **Ideale per**: scansioni di fuoco continue in cui conta un modello molto piccolo e veloce
- **Requisiti**: ONNX Runtime e il modello ONNX incluso.

### StackMFF V3 (apprendimento profondo)

È il metodo preselezionato in una nuova installazione.

- **Algoritmo**: classificazione della sorgente pixel per pixel (ONNX Runtime). La rete predice, per ogni pixel, da quale immagine di input proviene, poi miscela le immagini in base a quelle probabilità o ne sceglie una
- **Modalità di output IA** (pannello destro, solo per questo metodo):
  - **Miscelazione soft (pesata)** — predefinito; ogni pixel è una miscela delle immagini ponderata per probabilità. Transizioni morbide, senza cuciture nette; corrisponde alla configurazione riportata nell'articolo
  - **Scelta netta (frame singolo)** — ogni pixel è copiato dal fotogramma più probabile. Selezione più nitida senza interpolazione tra fotogrammi; può comparire una cucitura a livello di pixel dove due fotogrammi sono in parità
- **Ideale per**: massima qualità quando è disponibile una GPU
- **Vantaggi**: invariante all'ordine (l'ordine dei fotogrammi non cambia mai il risultato), gestisce stack entro le risorse disponibili, nessuna regolazione per metodo, ottimizzazione automatica
- **Requisiti**: ONNX Runtime e il modello ONNX incluso.

Ogni tile neurale usa tutti i fotogrammi. I fotogrammi e i dati temporanei di allineamento possono usare il disco; spazio libero e memoria disponibile limitano lo stack. La suddivisione spaziale V2 approssima l’inferenza sull’intera immagine.


### OpenFocus A / B / C

- **OpenFocus A**: leviga la sua mappa di risposta di fuoco multi-scala, poi seleziona i pixel a fuoco.
- **OpenFocus B**: miscela fotogrammi vicini guidato da una mappa di profondità levigata.
- **OpenFocus C**: ricostruisce bande multi-scala guidato da una mappa di profondità levigata.
- **Parametri**: A e B usano il cursore del kernel per controllare la scala di risposta di fuoco; C non ha raggio. Tutti e tre hanno un parametro di levigatura che controlla la forza delle transizioni.

**Levigatura**: un unico campo numerico sotto il cursore del kernel, il cui significato dipende dal metodo selezionato (intervallo 0–8, passo 0,2):

| Metodo | Etichetta | Predefinito | Effetto |
|---|---|---|---|
| OpenFocus A | Levigatura della miscelazione | 2,0 | Controlla la levigatura della mappa di risposta di fuoco prima della selezione dei pixel |
| OpenFocus B | Levigatura della profondità | 1,2 | Controlla la levigatura della mappa di profondità che guida la miscelazione |
| OpenFocus C | Levigatura della miscelazione | 3,0 | Controlla la levigatura della mappa di profondità che guida la ricostruzione multi-scala |

Valori più alti aumentano la levigatura delle transizioni. Ogni metodo ricorda il proprio valore tra le sessioni e l'elaborazione batch applica la stessa regola.

C non ha il parametro Raggio, quindi il suo cursore del kernel è nascosto.

CLI: `--depth-smoothing VALUE` (B) e `--blend-smoothing VALUE` (A, C).

**Le opzioni di luminosità sono per metodo.** *Abbina l'esposizione prima della fusione* e *Rimuovi la deriva di luminosità* sono ricordate separatamente per ogni metodo di fusione, perché rispondono a una domanda specifica dello stack più che a una preferenza globale. Entrambe sono attive per impostazione predefinita in OpenFocus A/B/C (che non hanno calibrazione dell'esposizione integrata); gli altri metodi mantengono i valori storici (abbinamento esposizione attivo, rimozione deriva disattivata). Cambiando metodo vengono mostrati i valori di quel metodo, che sopravvivono al riavvio. **Reimposta** cancella i valori memorizzati di luminosità, levigatura e raggio di tutti i metodi.

**Nitidezza dell'output fuso** applica una leggera maschera di contrasto dopo la fusione, prima che il risultato venga convertito nella profondità di output. È disattivata per impostazione predefinita e può essere attivata in **Impostazioni di rendering** o nelle opzioni batch; il flag CLI equivalente è `--sharpen-output`. Influisce sull'immagine fusa, non sui fotogrammi registrati né sulle mappe di profondità. Può rendere i bordi più distinti, ma può amplificare rumore e aloni. Viene salvata con il progetto e ricordata tra le sessioni.

### Solo allineamento (nessuna fusione)

Seleziona **Solo allineamento (nessuna fusione)** per registrare lo stack e aggiungere una voce stack all'elenco output. I fotogrammi restano in memoria finché non li esporti; il solo allineamento non crea una cache aggiuntiva né scrive accanto ai file sorgente. Salva la voce per creare una cartella nella destinazione scelta. I nomi file originali vengono conservati, con l'estensione del formato selezionato.

### Scelta di un metodo di fusione

1. Nel pannello destro, individua il gruppo **Fusione**
2. Scegli un metodo dall'**elenco a discesa** (ne è sempre selezionato esattamente uno; la prima voce registra senza fondere)
3. Regola la dimensione del kernel se applicabile — l'etichetta indica cosa significa il kernel per il metodo selezionato
4. Fai clic su **Avvia render** per elaborare

### Guida al confronto degli algoritmi

| Algoritmo | Velocità | Qualità | Ideale per |
|-----------|-------|---------|----------|
| Filtro guidato | Veloce | Buona | Uso generale |
| DCT | Media | Buona | Texture |
| DTCWT | Media | Molto buona | Scene complesse |
| GFG-FGF | Veloce | Buona | Regioni a fuoco |
| StackMFF (V1) | Lenta (GPU) | Non valutata qui | Luminanza indipendente dall'ordine; crominanza variabile in caso di parità delle distanze di luminanza |
| StackMFF V2 | Veloce (GPU) | Non valutata qui | Scansioni continue con un modello minuscolo |
| StackMFF V3 | Lenta (GPU) | Eccellente | Massima qualità |
| OpenFocus A | Dipende dallo stack | Dipende dallo stack | Selezione multi-scala dei pixel a fuoco |
| OpenFocus B | Dipende dallo stack | Dipende dallo stack | Miscelazione di fotogrammi vicini guidata dalla profondità |
| OpenFocus C | Dipende dallo stack | Dipende dallo stack | Ricostruzione multi-scala guidata dalla profondità |

Apri **Strumenti → Confronto metodi** per regolare più metodi in modo indipendente. Nome, etichette dei parametri e controlli di ogni metodo restano insieme quando la finestra si restringe. Le modifiche aggiornano automaticamente l'anteprima. **Anteprima ROI (veloce)** fonde l'area selezionata con il contesto circostante; **Fusione immagine intera** verifica il risultato sull'intera immagine e permette di ispezionare un'altra area senza renderizzare di nuovo. I risultati ROI possono differire da quelli a immagine intera perché le statistiche di esposizione e alcuni algoritmi dipendono dall'estensione dell'immagine.
La **fusione immagine intera** è selezionata per impostazione predefinita.

Un suggerimento visibile spiega come selezionare o deselezionare un risultato: fai clic su di esso oppure raggiungilo con `Tab` e premi `Invio` o `Spazio`.

Usa **Dimensione anteprima (px)** sopra i riquadri di destra per scegliere una dimensione quadrata da 160 a 960 pixel, trascinando il cursore o inserendo un numero. Tutti i metodi usano la stessa dimensione e si adattano alla larghezza disponibile. La dimensione viene ricordata per la prossima apertura. Cambiarla mantiene l'ingrandimento corrente e regola il quadrato di ispezione intorno al centro, limitandolo ai bordi dell'immagine.

Premi **F11** o il pulsante **Schermo intero (F11)** per usare tutto lo schermo; il testo del pulsante diventa **Esci da schermo intero (F11)**. La barra del titolo può anche massimizzare la finestra. **Esc** esce prima da schermo intero e chiude la finestra solo alla seconda pressione. Il ritorno da schermo intero ripristina la dimensione precedente e la griglia dei risultati si ridispone alla larghezza disponibile in entrambi i casi.

Trascina l'immagine sorgente con il tasto sinistro per spostare il quadrato bianco di ispezione; rilasciando il pulsante resta in posizione. I riquadri di destra mostrano esattamente quella regione dello stesso stack allineato. Usa la rotella per cambiare l'ingrandimento della regione, trascina con il tasto destro per spostare la sorgente, **Ctrl+rotella** per zoomare la sorgente e **F** per adattare. Cambiare fotogramma sorgente o ridimensionare la finestra conserva la selezione. Annulla ferma il lavoro in corso; chiudere attende che l'operazione nativa corrente termini in sicurezza.

### Anteprima rapida (render di bozza)

Regolare i parametri su stack grandi è lento a risoluzione piena. Apri **Impostazioni di rendering** e seleziona **Anteprima rapida**: OpenFocus produce prima una bozza a metà della dimensione lineare (mai sotto 320 px sul lato lungo), in genere diverse volte più veloce, registrazione inclusa. Il messaggio di completamento (e il tooltip del risultato più recente) indica **Qualità: anteprima**, e la bozza **non** viene aggiunta alla cronologia output. Deseleziona la casella e renderizza di nuovo per ottenere il risultato a risoluzione piena. La modalità anteprima non si applica ai rendering ROI e non inquina mai la cache di allineamento a risoluzione piena.

---

## 9. Trasformazioni delle immagini

### Rotazione

1. Vai su **Modifica → Ruota**
2. Scegli il tipo di rotazione:
   - **90° orario**: ruota tutte le immagini di 90 gradi a destra
   - **90° antiorario**: ruota tutte le immagini di 90 gradi a sinistra
   - **180°**: capovolge le immagini sottosopra

### Capovolgimento

1. Vai su **Modifica → Capovolgi**
2. Scegli il tipo di capovolgimento:
   - **Capovolgimento orizzontale**: specchia da sinistra a destra
   - **Capovolgimento verticale**: specchia dall'alto in basso

### Ridimensionamento

1. Vai su **Modifica → Ridimensiona**
2. Compare una finestra con le opzioni di ridimensionamento
3. Inserisci le nuove dimensioni o la percentuale di scala
4. Tutte le immagini dello stack vengono ridimensionate proporzionalmente

### Ordine delle trasformazioni

Le trasformazioni si applicano all'intero stack di immagini, mantenendo l'allineamento tra i fotogrammi.

---

## 10. Ritocco, etichette e annotazioni

### Ritocco del risultato (Modifica → Ritocca risultato…, `Ctrl+T`)

La fusione può ancora lasciare un difetto locale — un alone attorno a un bordo ad alto contrasto, una setola presa dal fotogramma sbagliato, rumore in un'area piatta. Lo spazio di lavoro copia i pixel dai fotogrammi esatti che hanno prodotto il risultato selezionato:

1. Seleziona un output nell'elenco (funziona anche clic destro → *Ritocca risultato…*) e premi `Ctrl+T`. Senza selezione, un selettore elenca tutti i risultati e indica quali hanno fotogrammi sorgente registrati.
2. Il fotogramma sorgente è a sinistra, il risultato a destra. Entrambi condividono zoom e posizione: la rotella zooma, il trascinamento con tasto destro o centrale (o Spazio+trascina) sposta, `F` adatta e `1` mostra il 100%.
3. Scegli una miniatura, trascina il cursore dei fotogrammi o premi `,` / `.`. Un tratto è bloccato su quel fotogramma e sulle impostazioni correnti del pennello; il fotogramma non può cambiare finché il tratto non termina.
4. Scegli **Pennello sorgente** (`B`) o **Pennello di ripristino** (`E`). Diametro, durezza e opacità hanno sia cursore sia campo numerico; `[` / `]` regola il diametro. Con `Ctrl` e la rotella verso l'alto aumenti il diametro, verso il basso lo riduci. La rotella senza modificatori regola lo zoom. Il pennello di ripristino riporta i pixel all'immagine che hai aperto, non a un output precedente.
5. Tieni premuto `Tab` per vedere il risultato prima di questa sessione e `Alt` per sbirciare il fotogramma sorgente. Il riferimento di nitidezza è facoltativo, copre solo la tela sorgente e usa una scala condivisa per tutti i fotogrammi.
6. `Ctrl+Z` / `Ctrl+Y` annullano e ripristinano interi tratti. **Ripristina tutto…** è un'unica operazione annullabile. **Applica come nuovo output** aggiunge una voce chiamata `<nome> · Ritoccato N` e la seleziona; l'originale resta intatto. La nuova voce è contrassegnata come *Non esportata* finché non la esporti dalla finestra principale, e OpenFocus la protegge dall'eliminazione automatica dalla cronologia e dalla chiusura senza decisione.

Applica il ritocco come nuovo output, selezionalo e apri **Salva con nome**. Scegli TIFF a livelli nella lista dei formati per salvare il risultato con i fotogrammi sorgente. Etichette e barre di scala attive non sono incluse; le modifiche alle annotazioni non salvate restano protette. Geometria sorgente e maschere di validità vengono conservate. Risultato e sorgenti devono essere immagini RGB o in scala di grigi opache con la stessa profondità a 8 o 16 bit, e ogni rettangolo sorgente deve rientrare nel risultato. L'esportazione supporta fino a 128 sorgenti; i nomi delle sorgenti sono limitati a 255 caratteri e non possono contenere NUL né caratteri fuori dal piano multilingue di base, come molte emoji. Rifiuta contesti non sicuri e lavori stimati oltre 512 MiB di memoria di lavoro o oltre il budget di RAM disponibile, segnalando l'errore invece di scrivere un'esportazione incompleta. GIMP 3.2.6 su Windows importa i livelli a 8 e 16 bit, inclusi i loro offset e il riempimento trasparente di registrazione. Il risultato è visibile e i livelli sorgente nascosti; GIMP aggiunge anche un livello Sfondo visibile. Altri editor non sono stati testati.

Per una correzione tonale locale, seleziona **Pennello esposizione**, imposta la quantità in centesimi di stop (100 raddoppia i valori dei pixel, -100 li dimezza) e dipingi sul risultato. L'anteprima si aggiorna mentre dipingi. Ogni tratto regola il risultato corrente una volta, anche dove le pennellate si sovrappongono; un altro tratto si accumula. Usa opacità e durezza del pennello per controllare la miscelazione. Le alte luci possono bruciarsi schiarendo. Tieni premuto **Prima** per confrontare con il risultato all'apertura, premi Esc durante un tratto per annullarlo, oppure usa annulla/ripristina dopo il tratto. Sono modifiche ai pixel, non livelli regolabili. Applica come nuovo output ed esporta la nuova voce per salvare i pixel modificati; un progetto non memorizza quei pixel.

**Regolazioni risultato…** modifica l'intero risultato di ritocco corrente. La sua anteprima campionata si aggiorna mentre cambi esposizione e contrasto; seleziona **Prima** per confrontare con il risultato precedente all'apertura della regolazione. Entrambi i controlli usano centesimi di stop: l'esposizione moltiplica i valori dei pixel, poi il contrasto scala la loro distanza dal grigio medio. Valori estremi possono bruciare alte luci o ombre. **OK** applica la regolazione come un'unica modifica annullabile; **Annulla** lascia il risultato invariato. Durante l'elaborazione, Annulla richiede un ripristino. L'immagine intera deve rientrare nel limite di memoria della cronologia di ritocco per questa operazione annullabile. Pubblica ed esporta il risultato come descritto sopra.

La provenienza viaggia con l'output: i fotogrammi allineati, gli offset di ritaglio/incolla ROI, il taglio dei bordi DCT e la profondità di bit vengono registrati quando il risultato viene renderizzato. File batch, risultati legacy e voci di cartella di solo allineamento non hanno una sorgente registrata, quindi sono elencati come non disponibili invece di essere dedotti dallo stack corrente. Tratti non applicati e cronologia annulla sono solo di sessione; salvare un progetto non li memorizza.

Per un file immagine esistente, scegli **Modifica → Ritocca file immagine…**, poi seleziona il risultato e una o più immagini sorgente. Le sorgenti devono essere già allineate al risultato sulla stessa griglia di pixel; dimensioni identiche non dimostrano l'allineamento. Sono accettate immagini PNG, TIFF, JPEG e BMP a pagina singola con dimensioni verticali, canali e profondità a 8 o 16 bit corrispondenti; l'orientamento EXIF viene applicato al caricamento. Immagini con alfa e multipagina non sono supportate. I fotogrammi selezionati sono forniti dall'utente, non sono provenienza di rendering registrata. Applica crea un output separato non esportato; salvalo dall'elenco output. I file scelti sono protetti dalla sovrascrittura durante questa sessione, incluse le esportazioni di mappe di profondità e mesh 3D. Come per altri risultati, i file di progetto non memorizzano questi pixel né la cronologia del pennello.

### Sostituire lo sfondo (Sfondo automatico)

Con l'impilamento lo sfondo diventa nitido insieme al soggetto. Lo sfondo di
un'immagine impilata è spesso un mosaico di sfocature diverse, con un alone
intorno al soggetto. **Sfondo automatico…** nell'area di ritocco lo sostituisce
con un fotogramma reale della stessa pila. Il soggetto resta come impilato. Lo
sfondo deriva da un fotogramma e dal risultato impilato; l'elaborazione può
ricostruire, correggere e fondere i suoi pixel.

1. Apri il risultato nell'area di ritocco e fai clic su **Sfondo automatico…**.
2. Scegli lo sfondo:
   - **Automatico** (consigliato) usa lo sfondo morbido, a meno che la pila
     abbia registrato a fuoco gran parte dello sfondo. In quel caso usa quello
     nitido.
   - **Sfondo morbido** usa il fotogramma con lo sfondo più sfocato.
   - **Sfondo nitido** usa il fotogramma con il fuoco più lontano.
3. Lascia attivo **Pulisci aloni sui bordi** per togliere gli aloni lungo i
   bordi del soggetto. Non va mai oltre ciò che i fotogrammi hanno registrato.
   Disattivalo per mantenere l'aspetto dei bordi della pila.
4. Seleziona **Mantieni le parti sfocate collegate al soggetto** quando parti del
   soggetto sono appena fuori dalla profondità di campo, come la punta
   arrotondata di una spina. Può mantenere come impilati anche oggetti di sfondo
   accanto al soggetto, quindi controlla il risultato.
5. Fai clic su **Sostituisci sfondo**. Una pila di 15 fotogrammi da 20 megapixel
   richiede circa un minuto e circa 3,5 GB di memoria libera. **Annulla** la
   interrompe.

La modifica è un solo passaggio annullabile, e i tratti già dipinti sul soggetto
restano. Usa il **Pennello di ripristino** per riportare i pixel impilati dove li
preferisci, per esempio un filo d'erba sfocato davanti al soggetto. Se il
soggetto riempie quasi tutta l'inquadratura, come in una scena di sottobosco,
l'immagine resta come impilata e compare una nota.

Dopo un'elaborazione, **Mostra dati mancanti** colora i punti che nessun
fotogramma ha visto con chiarezza. Sono i punti da controllare o ritoccare a
mano:

- magenta: sfondo che nessun fotogramma ha registrato senza il soggetto davanti
- ciano: soggetto nascosto alla fotocamera
- giallo: soggetto fuori dall'intervallo di fuoco della pila
- ardesia: zone troppo uniformi per giudicare

**Salva mappa dei dati mancanti…** salva l'anteprima colorata e una maschera in
bianco e nero per ogni colore. Lo sfondo automatico richiede i fotogrammi che
hanno prodotto il risultato e non è disponibile per i risultati di area (ROI).
I nuovi checkpoint dello sfondo ripristinano esattamente immagine, cronologia
raggiungibile e diagnostica senza ricalcolare lo sfondo.

### Gestione delle etichette

Apri **Strumenti → Etichette** dopo aver caricato immagini o prodotto un risultato. Ogni etichetta ha il proprio bersaglio: lo stack sorgente, lo stack registrato o i risultati di output. Le etichette di output possono applicarsi a tutti i risultati o al risultato selezionato per nome; rinominare o riordinare quel risultato non cambia l'associazione.

Usa **Aggiungi** per creare un'altra etichetta, seleziona una voce per modificarla e usa la casella per attivarla o disattivarla. Inserisci testo fisso o `{value}` per un numero di fotogramma. Il numero è il valore iniziale più l'indice del fotogramma (base zero) moltiplicato per l'intervallo; un output singolo usa sempre il valore iniziale.

Scegli tutti i fotogrammi, quello corrente o un intervallo personalizzato come `3-5,8`. Numeri di fotogramma non validi o non disponibili impediscono l'applicazione della bozza. Gli intervalli corrente e personalizzato seguono le identità sorgente selezionate quando lo stack viene riordinato; le etichette per tutti i fotogrammi si applicano anche ai fotogrammi aggiunti. I bersagli mancanti restano associati alle loro identità originali finché non scegli esplicitamente un nuovo bersaglio.

L'anteprima usa il bersaglio reale. Naviga con Precedente/Successivo o digita il numero del fotogramma, scegli **Adatta** o **100%**, e trascina l'immagine per spostarla al 100%. Posizione e dimensione del carattere usano i pixel dell'immagine. Un font sostitutivo o un disegno non riuscito viene segnalato. Le etichette conservano la profondità di bit originale quando esportate come PNG o TIFF.

**Applica e chiudi** conferma l'intera bozza come un'unica modifica annullabile. Annulla, Esc e la chiusura della finestra la scartano. Selezionare un'altra etichetta conserva le modifiche della bozza. **Elimina** rimuove solo la voce selezionata, inclusa l'ultima. Le azioni di rimozione del menu cancellano indipendentemente le etichette sorgente, dello stack registrato, di output o tutte, e possono essere annullate. Applicare etichette cambia la visualizzazione della sessione; esporta un'immagine per scrivere le annotazioni in un file.

---

## 11. Impostazioni e configurazione

Apri **Impostazioni → Impostazioni avanzate** per le preferenze di prestazioni, registrazione, pulizia del sensore, nomi output, IA StackMFF e avvio. Le modifiche restano bozze fino a **OK**; Annulla, Esc e la chiusura le scartano. Ridurre la dimensione dei tasselli aggiorna subito il limite di sovrapposizione visibile a metà tassello. Il limite batch salvato della sezione IA viene mantenuto, ma l'inferenza attuale invia un tassello per chiamata; la larghezza di registrazione può cambiare l'accuratezza dell'allineamento. La raccolta dei diagnostici di allineamento è una casella della sezione Registrazione.

I download automatici degli aggiornamenti avvengono mentre l'applicazione è inattiva e non si installano mai da soli. Un download completato resta pronto finché non scegli esplicitamente **Riavvia e aggiorna**. Il riavvio protegge le modifiche attive e il lavoro non salvato, poi scrive i dati di recupero prima di avviare la sostituzione. Annullare un download o una preparazione attende il limite di sicurezza corrente. Una build preparata completa e più recente può essere installata offline.

La correzione di campo piatto cambia i pixel di lavoro della sessione senza sovrascrivere i file sorgente. Seleziona il riferimento bianco e un fotogramma di anteprima, poi rigenera l'anteprima dopo aver cambiato sorgente o scala. Le immagini di riferimento e di lavoro/base devono condividere dominio di coordinate e dimensioni. Applica prepara entrambi gli stack prima della conferma; la rimozione facoltativa del fotogramma di riferimento fa parte dello stesso passo di annullamento. Esporta le immagini sorgente corrette quando devi conservare questi pixel fuori dalla sessione.

L'elaborazione batch accetta un'immagine di campo bianco facoltativa nelle sue opzioni. Scegli un riferimento sfocato alla risoluzione originale dei fotogrammi; viene scalato con lo stack di input e applicato prima della registrazione, usando la dimensione polvere predefinita. Un riferimento illeggibile o di dimensioni incompatibili fa fallire visibilmente lo stack interessato. Il riferimento è escluso dagli input di cartella, inclusa l'elaborazione raggruppata; devono restare almeno due fotogrammi sorgente per la fusione. La correzione di campo piatto non può essere combinata con input trasparente. Lascia il campo vuoto per elaborare senza correzione. La CLI accetta il riferimento anche con `--dust-map WHITE`.

La pulizia del sensore è facoltativa e viene eseguita sui fotogrammi originali in coordinate sensore prima della registrazione. Seleziona una maschera difetti esplicita per correggere pixel caldi o morti noti; OpenFocus non deduce una maschera dalle immagini del campione. La maschera deve essere un'immagine monocanale a 8 o 16 bit con le stesse dimensioni della sorgente: zero significa non interessato, mentre 1 o il valore massimo del canale marcano i difetti. Controlla la sua estensione prima di elaborare. Maschere che coprono più del 10% del fotogramma, o difetti senza un vicino non mascherato, vengono rifiutate. Mantieni la maschera alla risoluzione originale e non selezionarla come immagine sorgente. Per l'elaborazione nella GUI, carica i fotogrammi sorgente a scala piena quando usi una maschera difetti; la maschera non viene mai ridimensionata. La pulizia del sensore (inclusa la riduzione del rumore) non può essere applicata a uno stack GUI dopo rotazione, capovolgimento, correzione di campo piatto o altra modifica della sorgente: ricarica i fotogrammi originali o disattiva la pulizia. Questo vale anche per i progetti ripristinati con tali cronologie. L'elaborazione ROI preallineata non può essere combinata con la pulizia del sensore; esci dalla modalità ROI e renderizza lo stack completo. Le impostazioni di pulizia appartengono alla sessione e al progetto salvato, non alle preferenze globali. Batch e CLI elaborano i loro input originali separatamente, con la pulizia del sensore prima della correzione di campo piatto e della registrazione. Le maschere difetti richiedono file sorgente con orientamento EXIF neutro. File orientati, RAW di fotocamera e sorgenti con orientamento non determinabile vengono rifiutati perché i pixel decodificati potrebbero non corrispondere più alle coordinate del sensore. Usa immagini sorgente e maschera nello stesso orientamento di pixel noto, con metadati di orientamento neutri. La sola riduzione del rumore non richiede una maschera difetti.

La riduzione del rumore del sensore facoltativa è indipendente dalla maschera difetti ed è disattivata per impostazione predefinita. Ispeziona il risultato a dimensione piena per decidere se la sua levigatura è adatta al campione. La pulizia del sensore conserva i file sorgente e la profondità a 8 o 16 bit; è distinta dalla correzione di campo piatto. La CLI accetta `--defect-mask-path MASK` e `--sensor-denoise`; le ricette e l'API ricette Python catturano queste opzioni. Le ricette registrano un'impronta della maschera e rifiutano una maschera mancante o modificata. La pulizia CLI non può essere combinata con input trasparente. L'elaborazione raggruppata applica la pulizia ai fotogrammi originali, non ai risultati intermedi del gruppo.

Le ricette richiedono che la posizione di output sia fuori da tutte le cartelle di input e rifiutano un input selezionato esplicitamente come output. Scegli una posizione di output separata per proteggere le immagini sorgente ed evitare di includere un risultato precedente in una nuova esecuzione. Le impostazioni di elaborazione di un progetto possono essere riutilizzate come modello o ricetta anche se le sorgenti originali sono state spostate: **File → Salva modello di elaborazione…** memorizza le impostazioni senza percorsi sorgente e **File → Applica modello di elaborazione…** le applica a uno stack caricato, mentre `openfocus recipe create --project` scrive una ricetta headless che richiede ancora percorsi `--input` espliciti. I modelli trasportano l'indicatore di riduzione del rumore del sensore ma non il percorso della maschera difetti; le ricette memorizzano il percorso della maschera e la sua impronta. L'apertura del progetto stesso richiede comunque le sue sorgenti e qualsiasi riferimento di campo piatto registrato.

Quando un riferimento file viene applicato in Strumenti alla scala base dello stack, un progetto salvato registra il percorso assoluto, l'hash del contenuto e la dimensione polvere, e lo riapplica una volta alle sorgenti originali all'apertura. Mantieni il file di riferimento invariato e al suo posto: un riferimento mancante o modificato impedisce l'apertura del progetto. Un riferimento selezionato dallo stack, o una correzione a una scala di lavoro/base diversa, richiede comunque di esportare lo stack sorgente corretto prima di salvare un progetto. Se la scala di lavoro cambia dopo la correzione, ripristinala alla scala di correzione prima di salvare o esporta lo stack corretto.

**Aiuto → Invia feedback** apre i log locali. Il pulsante Copia copia il log visibile della sessione; lo ZIP di feedback contiene i log di sessione e dell'applicazione completa e le informazioni sull'ambiente. Controlla il pacchetto prima di inviarlo tu stesso. Aprire GitHub apre solo il tracker. Annullare un'esportazione lascia i file invariati; un errore conserva il file di destinazione precedente e mostra il percorso fallito.

Ogni avvio scrive il proprio log rotante, così i lavori GUI/CLI concorrenti mantengono registrazioni separate. Gli ID attività collegano parametri, tempo trascorso ed esiti completati/annullati/falliti; gli errori inattesi includono una traccia completa. Lo ZIP di feedback include anche il report di crash corrente e le versioni delle dipendenze. Usa **Aiuto → Apri cartella log** o il percorso mostrato nella pagina di feedback per trovare i file. I log ruotano a 1 MB con tre backup per processo; i log di sessioni inattive più vecchi di 14 giorni vengono rimossi. Per log di supporto dettagliati, imposta `OPENFOCUS_LOG_LEVEL=DEBUG` prima dell'avvio; `OPENFOCUS_LOG_DIR` seleziona una cartella diversa.

### Impostazioni dei tasselli (ottimizzazione della memoria)

Per immagini grandi, OpenFocus usa l'elaborazione a tasselli per evitare problemi di memoria.

Accesso: **Impostazioni → Impostazioni avanzate → Prestazioni**:

- **Dimensione blocco tassello**: dimensione di ogni tassello di elaborazione (1024 px predefiniti)
- **Sovrapposizione tasselli**: sovrapposizione tra tasselli per una miscelazione senza cuciture (256 px predefiniti)
- **Soglia tasselli**: dimensione immagine oltre la quale si usano i tasselli (2048 px predefiniti)

**Consigli**:

- Tasselli più piccoli: meno memoria, elaborazione più lenta
- Tasselli più grandi: più memoria, elaborazione più veloce
- Aumenta la sovrapposizione se noti cuciture visibili nei risultati

### Impostazioni di registrazione

Accesso: **Impostazioni → Impostazioni avanzate → Registrazione**:

- **Larghezza di riduzione**: risoluzione di pre-elaborazione per la registrazione
  - Predefinito: 1024 px
  - Più bassa per la velocità, più alta per l'accuratezza
- **Raccogli diagnostica di allineamento**: ricalcola l'allineamento al prossimo rendering e abilita **Impostazioni → Impostazioni avanzate → Registrazione → Diagnostica di allineamento**

### Riferimenti sulle prestazioni

Vedi [PERFORMANCE.md](./PERFORMANCE.md) per velocità e memoria misurate su grandi stack reali, e consigli pratici per stack profondi e input ad alta risoluzione.

Il pannello di stato mostra la memoria fisica usata e totale quando il sistema operativo fornisce entrambe le letture; altrimenti mostra `N/D`.

### Impostazioni del numero di thread

Controlla l'elaborazione parallela della CPU:

Accesso: **Impostazioni → Impostazioni avanzate → Prestazioni**:

- `0` seleziona la pianificazione automatica ed è il valore predefinito per le nuove impostazioni.
- I valori positivi sono limiti superiori; i limiti salvati sopravvivono agli aggiornamenti.
- Capacità CPU, modalità risorse e margine di memoria possono ridurre la concorrenza. Più thread non rendono necessariamente più veloce un rendering.

### Modalità risorse

**Impostazioni → Impostazioni avanzate → Prestazioni → Modalità risorse** decide quanta parte della macchina può usare un rendering. Non cambia mai l'immagine fusa.

| Modalità | Comportamento |
|---|---|
| Bilanciata (predefinita) | Lascia liberi circa il 20% dei core fisici e una riserva di memoria per il desktop |
| Reattiva | Usa circa metà dei core e una finestra di lavoro più piccola; rendering più lenti, macchina più reattiva |
| Massima | Usa tutti i core fisici consentiti dal budget di memoria |

Il pianificatore considera core fisici, affinità e quote CPU, ed esegue un solo strato parallelo alla volta. I lavori pesanti condividono l'ammissione durante preparazione e pubblicazione; l'annullamento attende la chiamata nativa attiva. Memoria sconosciuta o insufficiente produce un errore di risorse. La griglia di tasselli scelta resta invariata; scegliere tasselli più piccoli può cambiare i pixel di output.

### Accelerazione GPU

StackMFF V1/V2/V3 usano ONNX Runtime. DirectML su Windows e CoreML su macOS possono accelerare l’inferenza; altrimenti si usa la CPU. La disponibilità del provider non dimostra l’esecuzione sulla GPU. Il supporto di altre GPU richiede verifica.

- Attiva o disattiva in **Impostazioni → Accelerazione GPU**
- Deselezionandola si forzano i metodi neurali sulla CPU — utile quando i driver GPU danno problemi
- Il pannello di stato mostra `N/D` quando nessun acceleratore è usato o disponibile, e `CPU*` quando un acceleratore esiste ma il metodo selezionato è solo CPU
- La scelta si applica al rendering successivo e viene ricordata

### Dimensione batch IA

Accesso: **Impostazioni → Impostazioni avanzate → IA**:

- Il limite batch salvato è 2 per impostazione predefinita; la politica di esecuzione corrente limita l'inferenza reale a un tassello per chiamata su ogni dispositivo.
- Aumentare questo limite non abilita batch GPU non validati né riduce la dimensione spaziale dei tasselli.

### Temi

Sono disponibili due temi dell'interfaccia: **Chiaro (predefinito)** e **Scuro**. Cambia in **Impostazioni → Tema**; la scelta si applica subito (anche la barra del titolo di Windows) e viene ricordata. Al primo avvio l'utente sceglie un tema nella finestra di benvenuto; una scelta salvata prevale sempre.

### Modalità portatile

Crea un file vuoto chiamato `OpenFocus.portable` accanto a `OpenFocus.exe` (lo ZIP portatile ne include uno): tutti i dati utente — impostazioni, log, file recenti — vengono memorizzati in una cartella `OpenFocusData` accanto all'applicazione invece che nel profilo utente, così l'installazione può viaggiare su una chiavetta USB. Elimina il marcatore per tornare al profilo normale.

### Metadati EXIF

I metadati di origine (modello di fotocamera, esposizione, orientamento e così via) vengono conservati: l'orientamento viene applicato al caricamento e le esportazioni PNG/TIFF/WebP incorporano automaticamente l'EXIF originale. Le esportazioni JPG/BMP non possono memorizzare l'EXIF completo e vengono scritte senza.

### Input HEIC/HEIF

Le foto iPhone si caricano direttamente. Richiede `pillow-heif`, installato da `pip install openfocus` e incluso nelle build pacchettizzate.

### Aggiornamenti

**Aiuto → Controlla aggiornamenti** confronta la tua versione con le informazioni su `openfocus.top`. Una versione più recente viene scaricata dallo stesso sito in background mentre l'applicazione è inattiva; una volta preparata la build, il banner e l'area di stato offrono **Riavvia e aggiorna**, che la applica offline. Le build che non possono aggiornarsi da sole aprono la pagina di download del sito. Il controllo automatico avviene in silenzio al massimo una volta al giorno.

### Cache di registrazione

OpenFocus può memorizzare i risultati di registrazione in una cartella `.openfocus_cache` accanto allo stack sorgente. Riaprendo lo stesso stack (file invariati) l'allineamento viene saltato del tutto. Attiva o disattiva in **Impostazioni → Impostazioni avanzate → Registrazione → Cache di registrazione**; le voci obsolete vengono sostituite automaticamente.

### Impostazioni persistenti

Tutto quanto segue sopravvive automaticamente al riavvio: numero di thread, modalità risorse, parametri dei tasselli, larghezza di riduzione della registrazione, dimensione batch della fusione IA (StackMFF V3), accelerazione GPU, cache di registrazione e interruttori dei diagnostici di allineamento, lingua dell'interfaccia, dimensioni della finestra e disposizione dei pannelli, formato di esportazione per trascinamento, opzioni di luminosità per metodo, valori di levigatura e raggio, e stack aperti di recente. Le finestre delle impostazioni salvano al momento della conferma; tutto viene salvato anche alla chiusura. Le preferenze sono memorizzate per utente in `OpenFocus/OpenFocus.ini` (formato INI di QSettings).

### Lingua dell'interfaccia

Cambia in qualsiasi momento in **Impostazioni → Lingua**, che elenca i dodici pacchetti (English, 中文, 日本語, Español, Deutsch, Français, Русский, 한국어, Português, Italiano, Nederlands, Polski). Il cambiamento si applica subito e viene ricordato per le sessioni future.

### Controllo automatico degli aggiornamenti

Una volta al giorno, pochi secondi dopo l'avvio, OpenFocus controlla `openfocus.top` in silenzio. Anche le note di rilascio provengono dallo stesso sito. Se esiste una versione più recente, compare un banner sopra i pannelli immagine con la versione, **Aggiorna ora** (applica una build preparata o avvia il download), **Novità** (note di rilascio) e un pulsante per chiuderlo. Chiuderlo nasconde il banner solo per quella versione; la versione successiva lo ripropone, e **Aiuto → Controlla aggiornamenti** lo riattiva subito.

Durante il download in background la barra di stato mostra l'avanzamento. Puoi anche controllare manualmente in qualsiasi momento con **Aiuto → Controlla aggiornamenti**.

---

## 12. Elaborazione batch

Elabora più cartelle automaticamente:

1. Vai su **Strumenti → Elaborazione batch**
2. Nella finestra batch:
   - **Aggiungi cartelle**: seleziona cartelle contenenti sequenze di immagini
   - **Trascina cartelle**: rilascia una o più cartelle di stack in qualsiasi punto della finestra (o dell'elenco) per accodarle tutte insieme; il rilascio passa alla modalità multi-cartella e le miniature compaiono durante la decodifica
   - **Rimuovi cartelle**: rimuove le cartelle selezionate dall'elenco
3. Configura l'output:
   - **Salva in sottocartella**: crea i risultati in ogni cartella di input
   - **Salva nella stessa cartella**: salva accanto all'input conservando i file esistenti
   - **Cartella personalizzata**: specifica la posizione di output
4. Controlla le impostazioni correnti di fusione e registrazione
5. Fai clic su **Avvia elaborazione batch**

La finestra batch mostra l'avanzamento in tempo reale. Puoi annullare l'elaborazione in qualsiasi momento.

### Note sull'elaborazione batch

- Parte da un'istantanea delle impostazioni della finestra principale; i controlli batch le sostituiscono.
- Ogni cartella o gruppo suddiviso viene elaborato indipendentemente dallo stesso pipeline.
- **Elabora stack grandi in gruppi** è una casella facoltativa per la modalità multi-cartella. Quando è selezionata compaiono **Dimensione gruppo** (minimo 2, 4 predefiniti) e **Sovrapposizione** (da 0 alla dimensione gruppo meno 2); quando è deselezionata questi controlli sono nascosti e il raggruppamento è disattivato. La sovrapposizione desktop parte da 1 ed è limitata all'intervallo consentito quando cambia la dimensione. I gruppi vengono fusi a turni usando spazio temporaneo su disco; l'avanzamento identifica ogni turno. Fusione, abbinamento esposizione, correzione base e registrazione usano i fotogrammi e i riferimenti di ogni gruppo, inclusi i turni intermedi, quindi il risultato approssima l'elaborazione dell'intero stack in una volta. Il raggruppamento richiede un metodo di fusione; profondità per fotogramma originale, provenienza ed esportazioni dei fotogrammi allineati non sono disponibili. La modalità cartella suddivisa non è supportata, quindi la casella viene rifiutata in quella modalità. I gruppi registrati usano il loro primo fotogramma come riferimento di coordinate. Ogni registrazione rimuove un bordo prudente per l'interpolazione, e i turni successivi intersecano i rettangoli sopravvissuti, inclusa qualsiasi coda di un singolo fotogramma. L'immagine finale viene quindi ritagliata. Ogni gruppo registrato deve essere più grande di 8 pixel in entrambe le dimensioni; rotazioni difficili o trasformazioni proiettive possono non avere un rettangolo sicuro e produrre un errore. Usa input preallineati senza registrazione, o disattiva il raggruppamento, se accade.
- L'input raggruppato deve avere dimensioni orientate corrispondenti ed essere composto da file a fotogramma singolo con intestazioni leggibili da Pillow. RAW di fotocamera, file multipagina e file con canali alfa o trasparenza non sono supportati, inclusi gli stack PNG trasparenti. Rimuovi l'alfa o disattiva il raggruppamento per elaborare questi stack. Le prove a vuoto della CLI verificano queste restrizioni. Per la riga di comando, un esempio è `python main.py --input stack --output fused.png --group-size 8 --group-overlap 1`. L'input raggruppato CLI deve essere una cartella e non può essere combinato con `--depth-map`. Un riferimento di allineamento OpenFocus diverso per tutto lo stack non può essere combinato con il raggruppamento. Un riferimento di campo piatto viene applicato ai fotogrammi originali prima della registrazione; la nitidezza dell'output viene applicata una volta dopo il gruppo finale.
- **Cosa viene scritto e dove, per modalità.** Con **solo allineamento**, la destinazione scelta riceve una cartella `<stack>_aligned` per stack e il pannello output elenca quella cartella. Con **fusione + salva allineati**, l'immagine fusa viene scritta come `<stack>.<ext>` nella destinazione e i suoi fotogrammi allineati vanno in `<stack>_aligned` accanto; il pannello output elenca solo il risultato fuso. Con **fusione senza esportazione allineati**, viene scritta solo l'immagine fusa. Il pannello output nomina le cartelle esattamente come su disco.
- I fotogrammi allineati conservano i nomi file originali. Nomi duplicati e cartelle esistenti ricevono suffissi numerici. I file sorgente non vengono mai sostituiti.
- Le copie `*_registered*` residue delle versioni precedenti non vengono caricate come fotogrammi quando il file originale è presente nella stessa cartella; il messaggio di caricamento indica quante sono state saltate.
- PNG/TIFF conservano i dati a 16 bit; scegliere esplicitamente JPG/BMP converte a 8 bit.
- L'annullamento attende l'arresto del worker. I file confermati restano disponibili; attività parziali, fallite, annullate e non avviate vengono conteggiate separatamente.

---

## 13. Esportazione dei risultati

### Salvataggio di un singolo risultato

1. Seleziona il risultato che vuoi salvare nell'elenco output
2. Vai su **File → Salva** o premi `Ctrl+S`
3. Scegli formato e posizione
4. Fai clic su Salva

In **Impostazioni → Impostazioni avanzate → Nomi dei risultati**, un prefisso facoltativo del nome file aiuta a identificare i risultati appena renderizzati. L'esempio si aggiorna mentre digiti; usa fino a 40 lettere, numeri, underscore o trattini. Lascialo vuoto per la denominazione originale. Conferma con **OK** per ricordarlo, o annulla per scartare la modifica. Il prefisso entra nei nuovi nomi dell'elenco output e nei nomi di esportazione suggeriti; i risultati esistenti mantengono i loro nomi. Puoi comunque modificare un nome nella finestra di salvataggio. Questa preferenza non rinomina i file sorgente né cambia i nomi di output batch.

### Formato di esportazione per trascinamento

Trascinare i risultati dall'elenco output scrive file immagine nel formato scelto nel menu contestuale dell'elenco (**Formato trascinamento**: JPG, PNG, WebP o TIFF). La scelta viene ricordata.

### Salvataggio dello stack registrato

1. Vai su **File → Salva stack → Stack registrato**
2. Scegli il formato di salvataggio:
   - **Salva come cartella**: scegli una destinazione e una codifica; ogni stack riceve una nuova sottocartella con i nomi dei fotogrammi originali
   - **Salva come GIF**: crea un GIF animato della sequenza allineata

### Salvataggio dello stack di input

1. Vai su **File → Salva stack → Stack di input**
2. Scegli il formato di salvataggio:
   - **Salva come cartella**: salva i fotogrammi originali come immagini separate
   - **Salva come GIF**: crea un GIF animato della sequenza di input

### Formati di esportazione

- **JPG**: compresso, buon rapporto qualità/dimensione
- **PNG**: senza perdita, conserva dati immagine a 8 e 16 bit
- **BMP**: non compresso, converte i dati a 16 bit in 8 bit
- **TIFF**: conserva dati a 8 e 16 bit; l'esportazione multipagina mantiene l'intero stack in un unico file
- **GIF**: animazione con palette a 8 bit; l'annullamento lascia intatta una destinazione esistente

---

## 14. Interfaccia a riga di comando

`pip install openfocus` include il modello IA, i suoi pesi e la decodifica HEIC e RAW della fotocamera. `dtcwt` viene saltato da Python 3.13 in poi, dove la voce DTCWT resta disabilitata con un suggerimento.

OpenFocus può funzionare senza interfaccia grafica — utile per pipeline scriptate e integrazione continua. Lo script console `openfocus` (installato da `pip install openfocus`) e `python main.py` da una copia del codice espongono gli stessi comandi. Le opzioni di fusione funzionano con entrambe le forme; i sottocomandi sono:

```bash
openfocus version                       # stampa la versione
openfocus describe                      # JSON: metodi, parametri, capacità, comandi di controllo
openfocus gui [--control] [--socket NAME] [paths…]   # avvia l'app desktop
openfocus ctl <command> [flags]          # controlla una finestra in esecuzione (vedi AGENT_CONTROL_SPEC.md)
openfocus recipe create FILE --input … --output … [processing flags]
openfocus recipe run FILE [--dry-run]    # riproduce una ricetta headless salvata
openfocus retouch replay CHECKPOINT --base BASE --sources … --output RESULT
```

`openfocus gui` si avvia in background; `--control` attende il canale di controllo e stampa il suo JSON di connessione. `openfocus describe` e `version` non richiedono input. Il flusso di lavoro per agenti è documentato in [AGENT_CONTROL_SPEC.md](./AGENT_CONTROL_SPEC.md) e `skills/openfocus-control/SKILL.md`.

Da un ambiente Python (o accanto all'eseguibile pacchettizzato):

```bash
# Fonde una cartella con il metodo IA predefinito
python main.py --input ./stack_folder --output ./result/fused.png

# DTCWT con registrazione ECC, 8 thread
python main.py -i ./stack_folder -o ./result/fused.png -m dtcwt -a ecc -t 8

# StackMFF V3 forzato su CPU con una dimensione tassello personalizzata
python main.py -i ./stack_folder -o ./result/fused.png -m stackmffv3 --cpu --tile-size 512

# Elenco file esplicito; anche i file video sono accettati come input
python main.py -i img1.jpg img2.jpg img3.jpg -o fused.png -m gfgfgf

# Un singolo stack Z TIFF multipagina con barra di scala calibrata dai metadati
python main.py -i ./zstack.tif -o fused.png -m dtcwt --scale-bar

# Esamina fotogrammi adiacenti simili senza fondere né modificare l'input
python main.py -i ./stack_folder --suggest-duplicates
```

`--suggest-duplicates` accetta una cartella o un elenco ordinato di file immagine a fotogramma singolo. Stampa un oggetto JSON con gli array `files` e `suggestions`; `index` e `reference_index` di ogni suggerimento sono posizioni basate su zero in `files`, e `difference` è la differenza media normalizzata dei campioni. Aggiungi `--json report.json` per salvare lo stesso oggetto. Questa modalità di esame non si combina con `--output`, `--output-dir`, `--json-progress`, `--dry-run` né con le opzioni di pulizia del sensore. Il report confronta file adiacenti nell'ordine di input fornito, senza sapere quali fotogrammi potresti escludere in seguito. Se escludi un riferimento, riconsidera il suo successore o riesamina i file rimanenti nell'ordine previsto.

Ad esempio, aggiungi `--align openfocus --alignment-report alignment.json` a un singolo rendering. L'oggetto JSON contiene `method`, `input`, `frame_count` e `entries`. Il `frame_index` di ogni voce è basato su zero nell'ordine di input decodificato, prima dell'ordinamento per la fusione; le immagini multipagina si espandono in ordine di pagina e i fotogrammi video in ordine di decodifica. I metodi compositi includono voci separate per ogni fase. Il report viene scritto in modo atomico dopo un'elaborazione riuscita e non può sostituire un input o un altro output. Le modalità batch, raggruppata e prova rifiutano questa opzione.

| Opzione | Descrizione |
|--------|-------------|
| `--suggest-duplicates` | Segnala fotogrammi adiacenti quasi identici per la revisione manuale; non li fonde né li esclude |
| `--input, -i` | Cartella/e di immagini, file video, uno o più file immagine o TIFF multipagina |
| `--output, -o` | Percorso dell'immagine di output (l'estensione determina il formato) |
| `--output-dir, -d` | Modalità batch: fonde ogni cartella di input scrivendo `<cartella>.png` per stack |
| `--resume` | Modalità batch: salta le cartelle il cui output esiste già |
| `--config FILE` | Modalità batch: override JSON per cartella, ad es. `{"stackA": {"method": "dct"}}` |
| `--method, -m` | `stackmffv3` (predefinito), `stackmffv2`, `stackmff`, `openfocus_a`, `openfocus_b`, `openfocus_c`, `guided_filter`, `dct`, `dtcwt`, `gfgfgf` |
| `--param KEY=JSON_VALUE` | Imposta qualsiasi parametro di fusione; ripetibile e prevale sulle opzioni legacy |
| `--align, -a` | `none` (predefinito), `homography`, `ecc`, `both`, `openfocus` |
| `--align-param KEY=JSON_VALUE` | Imposta qualsiasi parametro di registrazione; ripetibile |
| `--align-shift-limit` | Solo allineamento `openfocus`: spostamento massimo accettato come frazione della dimensione del fotogramma (0,08 predefinito; oltre resta non allineato) |
| `--alignment-report PATH` | Salva gli esiti di allineamento per fotogramma in JSON per un rendering singolo non raggruppato con allineamento attivo; batch e gruppi non supportati |
| `--exposure-match` | Abbina la media di ogni fotogramma a quella dell'ultimo prima della fusione (attivo per impostazione predefinita; `--no-exposure-match` lo disattiva). Guadagni sull'intero stack, quindi l'output a tasselli resta senza cuciture; si applica a ogni metodo e al riferimento di luminosità dell'allineatore OpenFocus |
| `--base-frame` | Passo facoltativo post-fusione: abbina le basse frequenze dell'output al fotogramma `last`/`first` (`none` predefinito) |
| `--sort` | Ordine fotogrammi per `openfocus_a/b/c`: `asc` (predefinito, ordine file), `desc` (inverte una scansione da lontano a vicino), `off` |
| `--frame-order` | Cartella in ingresso: `auto` (predefinito) ordina i fotogrammi da vicino a lontano secondo i passi di messa a fuoco EXIF quando tutti li hanno (il bracketing di fuoco OM SYSTEM parte da metà profondità), altrimenti per nome; `name` ordina sempre per nome file. I file elencati esplicitamente mantengono il loro ordine |
| `--sharpen-output` | Applica una leggera maschera di contrasto all'output fuso (disattivata per impostazione predefinita) |
| `--dust-map WHITE` | Correzione di campo piatto (mappa polvere) da uno scatto di campo bianco sfocato |
| `--defect-mask-path MASK` | Maschera difetti in coordinate sensore riparata prima della registrazione |
| `--sensor-denoise` | Riduzione prudente del rumore del sensore prima della registrazione |
| `--kernel, -k` | Dimensione kernel (forzata dispari); controlla la scala di risposta di fuoco per `openfocus_a` e `openfocus_b`; C non ha raggio |
| `--depth-smoothing VALUE` | Solo `openfocus_b`: forza della levigatura delle transizioni di profondità (1,2 predefinito) |
| `--blend-smoothing VALUE` | Solo `openfocus_a` / `openfocus_c`: forza della levigatura delle transizioni (2,0 per A, 3,0 per C predefiniti) |
| `--ai-mode {soft,hard}` | Output StackMFF V3: miscelazione ponderata per probabilità (predefinita) o fotogramma più probabile per pixel |
| `--threads, -t` | Limite superiore dei thread di lavoro; `0` = automatico (predefinito), i valori positivi limitano la concorrenza |
| `--resource-preference` | `balanced` (predefinito), `responsive` o `maximum`; controlla la pianificazione delle risorse |
| `--batch-size` | Limite superiore del batch di tasselli IA (2 predefinito); l'inferenza attuale usa un tassello per chiamata |
| `--cpu` | Forza la CPU anche se è disponibile una GPU |
| `--downscale` | Larghezza di riduzione per il rilevamento caratteristiche della registrazione |
| `--tile-enabled` / `--no-tile-enabled` | Abilita la fusione a tasselli quando necessario (predefinito: attiva) |
| `--tile-size` | Dimensione blocco tassello per immagini grandi (predefinito: automatica) |
| `--tile-overlap` | Sovrapposizione dei tasselli in pixel (predefinito: automatica) |
| `--tile-threshold` | Dimensione minima dell'immagine che attiva i tasselli (predefinito: automatica) |
| `--group-size` | Fonde una cartella in gruppi di questo numero di fotogrammi; `0` disattiva il raggruppamento (predefinito). Solo cartelle; niente `--depth-map`, input RAW/alfa/multipagina né fotogramma di riferimento OpenFocus diverso da quello predefinito |
| `--group-overlap` | Fotogrammi condivisi da gruppi consecutivi; `0` predefinito, al massimo dimensione gruppo meno 2 |
| `--scale-bar` | Disegna una barra di scala (calibrazione rilevata automaticamente dai metadati TIFF) |
| `--scale-um` | Dimensione pixel in µm/px, che prevale sui metadati (implica `--scale-bar`) |
| `--depth-map [path]` | Salva anche una mappa colorizzata della posizione di fuoco (predefinito `<output>_depth.png`); solo metodi compatibili con la profondità, non raggruppati né trasparenti. In modalità batch è consentito solo il percorso predefinito |
| `--depth-colormap` | Mappa colori per `--depth-map`: turbo, viridis, jet, gray, … (turbo predefinito) |
| `--json FILE` | Scrive un riepilogo del risultato leggibile dalla macchina in FILE al termine |
| `--json-progress` | Stampa un oggetto JSON di avanzamento per passo su stdout (NDJSON); il testo va su stderr |
| `--dry-run` | Valida gli input e stampa il piano senza fondere |

Non esiste un valore `--method none`: la CLI fonde sempre. Usa l'applicazione desktop per un output di solo allineamento.

### Ricette di elaborazione

Una ricetta è un file JSON portatile di opzioni di elaborazione CLI con percorsi relativi alla ricetta. Creane una dalle opzioni che passeresti a un'esecuzione normale, poi riproducila:

```bash
openfocus recipe create line.json --input ./stack --output ./out/fused.tif \
  --method dtcwt --align ecc --threads 8
openfocus recipe run line.json
openfocus recipe run line.json --dry-run
```

`recipe create --project project.ofproj --input ./stack --output ./out/fused.tif` cattura le impostazioni di elaborazione di un progetto su input scelti esplicitamente; le sorgenti del progetto possono essere state spostate. Le ricette rifiutano controlli transitori come `--dry-run`, `--resume`, `--json` e `--config`, e richiedono che l'output sia fuori da tutte le cartelle di input (e diverso da un input selezionato esplicitamente). Una maschera difetti viene memorizzata per percorso relativo con un'impronta SHA-256, quindi una maschera mancante o modificata fa fallire l'esecuzione. Le modifiche alla sorgente del progetto (rotazione, capovolgimento, campo piatto) non possono essere catturate; esporta prima lo stack corretto.

La cattura del progetto trasferisce nella ricetta CLI le impostazioni di fusione, registrazione, luminosità, ordine, nitidezza e tasselli, e incorpora le impostazioni di elaborazione per la GUI. Esclusioni dei fotogrammi, scala di caricamento, etichette e modifiche non vengono trasferite; prepara di conseguenza il nuovo stack di input. Le opzioni di raggruppamento possono essere catturate dagli argomenti CLI.

Il formato JSON versione 1 contiene `version`, un array `input`, `output` oppure `output_dir`, e un oggetto `options`. I percorsi sono relativi al file della ricetta; sposta la ricetta insieme ai suoi file di input per conservare tali relazioni. La creazione cattura le impostazioni della CLI e i parametri risolti dell'algoritmo. Campi sconosciuti, valori non validi e versioni non supportate vengono rifiutati. Una prova a vuoto controlla l'input senza scrivere file di risultato. I file di progetto esistenti restano invariati.

Nella GUI, File → Salva modello di elaborazione salva le impostazioni senza percorsi sorgente. File → Applica modello di elaborazione applica un modello, un progetto o una ricetta basata su progetto allo stack corrente senza sostituirne input, trasformazioni, esclusioni o etichette. Un modello non è di per sé una ricetta eseguibile. Le impostazioni GUI incorporate in una ricetta sono indipendenti dalle sue opzioni CLI; se modifichi il JSON, mantieni coerenti entrambe.

La stessa API headless è disponibile senza importare PyQt:

```python
from core.recipes import capture_project_recipe, capture_recipe, load_recipe, save_recipe, run_recipe

capture_recipe("workflow.json", ["-i", "stack_folder", "-o", "result/fused.png", "-m", "dct"])
capture_project_recipe("from-project.json", "session.ofproj",
                       ["-i", "new_stack", "-o", "result/fused.png"])
recipe = load_recipe("workflow.json")  # dizionario validato con percorsi relativi
save_recipe("workflow.json", recipe)
exit_code = run_recipe("workflow.json", dry_run=True)  # usa False per elaborare
```

`capture_recipe` interpreta i suoi argomenti come normali percorsi CLI relativi alla directory corrente; `save_recipe` interpreta i percorsi memorizzati relativi alla ricetta. La ricetta registra impostazioni e riferimenti, inclusa un'impronta di contenuto per una maschera di difetti del sensore selezionata esplicitamente. Non registra i pixel di input, le versioni software, l'hardware né i checksum dello stack sorgente; conservali separatamente quando la provenienza è importante. Impostazioni identiche non garantiscono pixel identici tra versioni software, driver o dispositivi; le operazioni neurali su GPU possono essere non deterministiche. Usa `--cpu` quando è richiesta un'esecuzione su CPU e conserva il software installato e gli input.

### Riproduzione di un punto di controllo di ritocco

Usa i comandi di salvataggio/caricamento dei checkpoint nello spazio di ritocco con la base originale e le sorgenti elaborate. Le sessioni con sfondo automatico, comprese le analisi riuscite senza cambiamenti ai pixel, salvano istantanee ZIP v2 `.ofretouch` senza perdita con metadati JSON e dati NumPy: l'immagine di lavoro esatta, i tasselli raggiungibili di annulla/ripristina, gli stati diagnostici e la provenienza delle sorgenti. Il caricamento v2 non ricalcola lo sfondo. Le sessioni con soli pennelli/toni possono usare JSON v1 compatto; i vecchi checkpoint dello sfondo in JSON v1 restano riproducibili.

La verifica delle sorgenti avviene in background ed è annullabile. Prima di sostituire la destinazione, il salvataggio ricarica il file preparato e controlla l'uguaglianza esatta dell'immagine corrente, dei tasselli raggiungibili di annulla/ripristina e degli stati diagnostici. Sorgenti incompatibili, verifica fallita, annullamento o memoria insufficiente lasciano invariati le modifiche correnti e qualsiasi checkpoint esistente.

Se il limite di registrazione del punto di controllo impedisce un'altra modifica, applica le modifiche correnti e avvia una nuova sessione di ritocco per continuare.

Salva un punto di controllo di ritocco dallo spazio di ritocco, poi fornisci l'immagine base originale e ogni fotogramma sorgente elaborato nello stesso ordine della sessione:

```bash
openfocus retouch replay edits.ofretouch --base fused.png --sources frame1.png frame2.png --output retouched.tif
```

Entrambi i formati richiedono ancora la base originale e tutti i fotogrammi effettivamente usati per il ritocco, nell'ordine della sessione, dopo allineamento e qualsiasi ritaglio ROI o altra elaborazione, con i valori di pixel e la profondità di bit originali. Il caricamento ricostruisce gli offset dei fotogrammi e il comportamento di allineamento registrati e verifica base, fotogrammi ordinati e maschere di validità; pixel, geometria o maschere incompatibili vengono rifiutati. Ogni percorso deve contenere un'immagine. L'output deve essere PNG o TIFF e conserva la profondità di bit della base. Il punto di controllo e le immagini di input non possono essere usati come output.

Codici di uscita: `0` successo, `1` errore di elaborazione, `2` errore d'uso. `python main.py --help` mostra tutte le opzioni; avviare con un semplice percorso (`python main.py C:\stack`) apre comunque la GUI con quella cartella caricata.

---

## 15. Scorciatoie da tastiera

### Operazioni sui file

| Scorciatoia | Azione |
|----------|--------|
| `Ctrl+O` | Apri cartella |
| `Ctrl+Shift+O` | Apri video |
| `Ctrl+Alt+O` | Apri progetto |
| `Ctrl+Alt+S` | Salva progetto |
| `Ctrl+S` | Salva risultato |
| `Ctrl+Shift+S` | Salva stack registrato |
| `Ctrl+W` | Svuota stack |
| `Ctrl+Q` | Esci |
| `Ctrl+R` | Avvia render |
| `Ctrl+Shift+R` | Confronta tutti i metodi |
| `Ctrl+T` | Ritocca risultato |
| `Ctrl+M` | Mappa di profondità |
| `Ctrl+E` | Attiva/disattiva confronto Wipe |
| `Ctrl+Shift+B` | Elaborazione batch |
| `Ctrl+Z` / `Ctrl+Y` | Annulla / Ripristina (modifiche stack) |

### Navigazione

| Scorciatoia | Azione |
|----------|--------|
| `Su` / `Giù` (elenco sorgente attivo) | Fotogramma precedente / successivo |
| `Rotella mouse` | Zoom avanti/indietro |
| `Ctrl + rotella` | Zoom fine |
| `Doppio clic` | Alterna adattamento/100% |

### Generali

| Scorciatoia | Azione |
|----------|--------|
| `Spazio` (tenuto, puntatore su un'immagine) | Modalità spostamento temporanea |
| `Canc` | Elimina l'elemento di elenco attivo; cancella anche la ROI quando il puntatore è sull'immagine sorgente |

---

## 16. Risoluzione dei problemi

### Problemi comuni

#### «Nessuna immagine caricata»

- Verifica di aver prima importato una cartella di immagini o un video
- Controlla che la cartella contenga formati immagine supportati

#### I risultati della fusione appaiono sfocati o con immagini fantasma

- Attiva la registrazione (scegli omografia / ECC / OpenFocus veloce)
- Assicurati che le immagini siano allineate prima della fusione
- Prova un altro algoritmo di fusione
- Aumenta la dimensione del kernel per risultati più morbidi

#### IA non disponibile

- Controlla ONNX Runtime e il modello incluso.
- Controlla in **Aiuto → Informazioni sull'ambiente**

#### Errori di memoria esaurita

- Attiva l'elaborazione a tasselli in **Impostazioni → Impostazioni avanzate → Prestazioni**
- Riduci la dimensione del blocco tassello
- Usa kernel più piccoli
- Ridimensiona le immagini

#### La registrazione non riesce

- Usa **OpenFocus (veloce)** per movimento rigido limitato, oppure attiva **Interrompi su fotogramma rifiutato** per fallire presto invece di accettare fotogrammi non allineati
- Riduci la larghezza di riduzione in **Impostazioni → Impostazioni avanzate → Registrazione**
- Assicurati che le immagini abbiano contrasto e caratteristiche sufficienti
- Attiva **Raccogli diagnostica di allineamento**, renderizza di nuovo e leggi **Impostazioni → Impostazioni avanzate → Registrazione → Diagnostica di allineamento** per i fotogrammi in errore

#### Elaborazione lenta

- Lascia i thread su automatico (`0`); capacità CPU, modalità risorse e margine di memoria limitano già la concorrenza
- Usa la modalità risorse Massima per un batch non presidiato, oppure filtro guidato / GFG-FGF invece di un metodo IA
- Riduci la risoluzione delle immagini o usa **Anteprima rapida** durante la regolazione

#### Avviso di Windows SmartScreen al primo avvio

La firma dei binari Windows dipende dalla release specifica (vedi le note di rilascio); la firma non è garantita. Scaricali da **https://openfocus.top** invece che da un mirror di terze parti. Se SmartScreen avvisa, fai clic su **Ulteriori informazioni → Esegui comunque**. Il primo avvio dopo la decompressione può richiedere 30–60 secondi.

#### macOS segnala che l'app «non può essere aperta» o «è danneggiata»

La build macOS non è notarizzata. Scarica la build Apple Silicon da **https://openfocus.top**. Se installi dal DMG spostando `OpenFocus.app` in Applicazioni, esegui:

```bash
xattr -cr /Applications/OpenFocus.app
```

Se installi dal terminale con `install.sh`, l'app si trova in `~/Applications`; esegui `xattr -cr ~/Applications/OpenFocus.app`.

oppure fai clic destro sull'app e scegli **Apri**. Le build macOS pubblicate sono per Apple Silicon (M1/M2/M3/M4); i Mac Intel possono eseguire dai sorgenti.

### Come ottenere aiuto

1. **Informazioni sull'ambiente**: vai su **Aiuto → Informazioni sull'ambiente** per controllare l'installazione
2. **Contatta il supporto**: vai su **Aiuto → Contattaci** per le informazioni di supporto

---

## Appendice: dettagli sugli algoritmi di fusione

### Filtro guidato

Il filtro guidato esegue una levigatura che preserva i bordi usando un'immagine guida. Calcola trasformazioni lineari locali tra la guida e le immagini di input, poi le applica per produrre l'output. Questo metodo eccelle nel preservare i bordi riducendo il rumore.

### DCT (Discrete Cosine Transform)

La DCT trasforma le immagini nel dominio della frequenza, dove le decisioni di fusione si basano sull'ampiezza dei coefficienti. Le componenti ad alta frequenza (dettagli) vengono conservate mentre quelle a bassa frequenza (regioni uniformi) vengono fuse in base all'energia locale.

### DTCWT (Dual-Tree Complex Wavelet Transform)

La DTCWT fornisce una decomposizione multi-scala e multidirezionale delle immagini. Cattura caratteristiche a scale e orientamenti diversi, consentendo regole di fusione sofisticate che preservano texture e struttura.

### GFG-FGF

Implementa Fu et al. (2024): la mappa di fuoco grossolana |I − mean(I)| viene arricchita dal gradiente a quattro vicini di tipo gaussiano — il template 3×3 T = [1 2 1; 2 4 2; 1 2 1]/16 dà P = I⊗T, e G = (|I(x±1,y) − P|)/2 + (|I(x,y±1) − P|)/2, scartando i valori sotto T0 = 0,005 — poi affinata da un filtro guidato rapido. Ogni pixel sceglie il fotogramma con il valore di fuoco più alto (pareggi: misura di piccola regione Sobel 3×3), e le mappe di decisione binarie vengono levigate dal filtro guidato rapido prima della miscelazione pesata.

### Fusione IA (StackMFF V3)

Approccio di apprendimento profondo basato su una rete neurale addestrata su grandi insiemi di coppie di immagini multifocus. La rete apprende automaticamente strategie di fusione ottimali e offre risultati all'avanguardia quando configurata correttamente.

---

*OpenFocus è un progetto open source. Per aggiornamenti, segnalazioni di bug e richieste di funzionalità, visita il repository del progetto.*
