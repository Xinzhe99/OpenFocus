# OpenFocus Gebruikershandleiding

## Inhoudsopgave

1. [Inleiding](#1-inleiding)
2. [Aan de slag](#2-aan-de-slag)
3. [Overzicht van de hoofdinterface](#3-overzicht-van-de-hoofdinterface)
4. [Afbeeldingen en video importeren](#4-afbeeldingen-en-video-importeren)
5. [Afbeeldingen bekijken en navigeren](#5-afbeeldingen-bekijken-en-navigeren)
6. [Vergelijkingsweergave Vegen](#6-vergelijkingsweergave-vegen)
7. [Beeldregistratie](#7-beeldregistratie)
8. [Fusiemethoden](#8-fusiemethoden)
9. [Beeldtransformaties](#9-beeldtransformaties)
10. [Retoucheren, labels en annotaties](#10-retoucheren-labels-en-annotaties)
11. [Instellingen en configuratie](#11-instellingen-en-configuratie)
12. [Batchverwerking](#12-batchverwerking)
13. [Resultaten exporteren](#13-resultaten-exporteren)
14. [Opdrachtregelinterface](#14-opdrachtregelinterface)
15. [Sneltoetsen](#15-sneltoetsen)
16. [Problemen oplossen](#16-problemen-oplossen)

---

## 1. Inleiding

OpenFocus is een professionele desktoptoepassing voor multi-focus-beeldfusie, bedoeld voor onderzoekers, ingenieurs en professionals in beeldverwerking. De software verwerkt multi-focus-beeldreeksen met geavanceerde registratie- en fusiealgoritmen en levert alles-in-focus-beelden met volledige scherptediepte.

### Belangrijkste functies

- **Multi-focus-beeldfusie**: combineer meerdere afbeeldingen met verschillende scherpstelpunten tot één volledig scherp beeld
- **Meerdere fusiealgoritmen**: kies uit Guided Filter, DCT, DTCWT, GFG-FGF en AI-fusie (deep learning)
- **Beeldregistratie**: lijn verschoven beeldreeksen uit met ECC of Homografie
- **Vergelijkingsweergave Vegen**: controleer de uitlijning en vergelijk resultaten met een A/B-scheiding
- **Projectbestanden**: sla volledige werksessies op en herstel ze later
- **Meerbladige TIFF**: Z-stapels in één bestand worden pagina voor pagina geladen en weer als één bestand opgeslagen
- **Dieptekaart**: focuspositie per pixel met eigen pseudo-kleurschema's
- **Schaalbalk**: schaalbalken op publicatieniveau met automatische µm/px-kalibratie
- **Automatisch sessieherstel**: na een crash wordt de laatste werksessie aangeboden om te herstellen
- **Meertalig**: Engels, 中文, 日本語, Español met automatische detectie van de systeemtaal
- **CLI zonder GUI**: script OpenFocus vanaf de opdrachtregel voor automatisering en CI
- **Flexibel exporteren**: sla resultaten op als losse afbeeldingen, mappen of GIF-animaties
- **Beeldtransformaties**: draai, spiegel en wijzig het formaat van beeldstapels
- **Labels toevoegen**: voeg tekstlabels toe aan uitgelijnde en invoerstapels
- **Blijvende instellingen**: je voorkeuren, taal en recent geopende stapels blijven bewaard na een herstart

### Ondersteunde invoerformaten

- Afbeeldingsbestanden: JPG/JPEG, PNG, BMP, TIFF, WebP en andere gangbare formaten
- Videobestanden: MP4-formaat (automatisch gedecodeerd naar beeldreeksen)
- Mappen met genummerde beeldreeksen
- RAW-bestanden van camera's (DNG, CR2/CR3, NEF, ARW, RAF, ORF, RW2 en andere
  gangbare RAW-formaten). Het decoderen gebeurt met `rawpy` (LibRaw), dat
  `pip install openfocus` installeert en dat de Windows/macOS-builds meeleveren

### Ondersteunde uitvoerformaten

- Losse afbeeldingen: JPG, PNG, BMP, TIFF, WebP
- Beeldreeksen: opgeslagen als mappen
- Animaties: GIF-formaat

---

## 2. Aan de slag

### De toepassing starten

Na de installatie start je OpenFocus als volgt:

1. **Windows (uitvoerbaar bestand)**: dubbelklik op de OpenFocus-snelkoppeling of het uitvoerbare bestand
2. **Python-broncode**: voer `python main.py` uit in je geactiveerde conda-omgeving

Na het starten zie je het hoofdvenster van de toepassing met een donker thema.

### Snelstartprocedure

1. Importeer je beeldreeks (Bestand → Map openen of slepen en neerzetten)
2. Stel de fusie-instellingen in op het rechterpaneel
3. Schakel indien nodig registratie in voor uitlijning
4. Pas de kernelgrootte aan als dat nodig is
5. *(optioneel)* Vink **Snel voorbeeld** aan om de parameters eerst op een verkleind concept te testen
6. Klik op "Renderen" om de verwerking te starten
7. Exporteer je resultaten

---

## 3. Overzicht van de hoofdinterface

De interface van OpenFocus bestaat uit drie hoofdgebieden:

### Linkerpaneel: beeldweergave

Het linkerdeel van het venster toont je afbeeldingen in een gedeelde weergave:

- **Paneel bronafbeelding (boven)**: toont de ingevoerde beeldreeks
- **Paneel resultaat (onder)**: toont de fusie- of registratieresultaten
- **Navigatieschuif**: gebruik de schuif onder elk paneel om door de frames te bladeren
- **Info-labels**: tonen de huidige framepositie en de afbeeldingsafmetingen

Je kunt de panelen vergroten of verkleinen door de scheidingsbalk ertussen te verslepen.

### Rechterpaneel: bedieningspaneel

Het rechterpaneel bevat alle configuratieopties, verticaal gerangschikt:

1. **Fusie-instellingen**: kies het fusiealgoritme en de kernelgrootte
2. **Registratie-instellingen**: schakel uitlijningsmethoden in of uit
3. **Actieknoppen**: Standaardwaarden en Renderen
4. **Lijst bronafbeeldingen**: toont alle geladen afbeeldingen met bestandsnamen
5. **Uitvoerlijst**: toont de gegenereerde resultaten

### Menubalk

De menubalk bovenaan geeft toegang tot alle functies:

- **Bestand**: importeren/exporteren en de lijst met recente bestanden
- **Bewerken**: beeldtransformaties en labels
- **Extra**: batchverwerking
- **Instellingen**: GPU-versnelling, taal, tegel-, registratie- en threadinstellingen
- **Help**: omgevingsinfo, contactgegevens, updatecontrole en de logmap

---

## 4. Afbeeldingen en video importeren

### Een afbeeldingsmap openen

1. Ga naar **Bestand → Map openen** of druk op `Ctrl+O`
2. Selecteer een map met je beeldreeks
3. Er verschijnt een dialoogvenster voor de laadschaal — kies de schaal (100% = volledige resolutie)
4. Klik op OK om de afbeeldingen te laden

Vink **Niet meer vragen — altijd op deze schaal laden** in dat venster aan om altijd op de gekozen schaal te
laden zonder vraag; **Instellingen → Bij import om laadschaal vragen** zet de
vraag weer aan. Een geladen stapel kun je later alsnog verkleinen met
**Bewerken → Formaat wijzigen**.

De software doet automatisch het volgende:
- Afbeeldingsbestanden opsporen en op bestandsnaam sorteren
- De eerste afbeelding in het bronpaneel tonen
- De lijst met bronafbeeldingen vullen met alle frames

### Videobestanden openen

1. Ga naar **Bestand → Video openen** of druk op `Ctrl+Shift+O`
2. Selecteer een videobestand (MP4, AVI, MOV, MKV, WMV, FLV, WEBM)
3. De video wordt automatisch gedecodeerd naar losse frames
4. Elk frame wordt onderdeel van de beeldstapel

### Projectbestanden

Bestand → Project opslaan bewaart je hele werksessie als een `.ofproj`-bestand:
de exacte lijst met bronbestanden (in volgorde), de laadschaal, de
fusiemethode, de registratieopties, de kernelgrootte, de tegel-, thread- en
registratie-instellingen, de labelconfiguraties en het huidige frame. Bestand →
Project openen bouwt die sessie exact opnieuw op (bronafbeeldingen worden
gerefereerd, niet gekopieerd — laat ze dus staan). De venstertitel toont het
geopende project.

### Demostapel

Het welkomstvenster biedt **Demostapel laden**: zes meegeleverde voorbeeldfoto's
uit een echte focusstapel, zodat je de volledige werkwijze meteen kunt
uitproberen zonder eigen afbeeldingen. Ook handig om een installatie te
controleren.

### Scherptecurve

Een kleine curve boven de navigatieschuif van de bron toont de scherpte per
frame (genormaliseerd). Onscherpe frames verschijnen als dalen met een rode
stip. Klik ergens op de curve om naar dat frame te springen; zweef eroverheen
om het framenummer te zien.

### Meerbladige TIFF-stapels

Microscopiesoftware (ZEN, NIS-Elements, ImageJ/Fiji) exporteert een hele
Z-stapel vaak als één meerbladig TIFF-bestand. OpenFocus leest elke pagina van
zo'n bestand als een afzonderlijk frame van de beeldstapel — sleep het bestand
naar het venster of open het, en de stapel verschijnt compleet (frames krijgen
namen als `<bestand>_page_001`). Meerbladige TIFF's met één pagina werken
precies zoals voorheen. Het omgekeerde kan ook:
**Bestand → Stapel opslaan → (Uitgelijnde stapel / Invoerstapel) → Opslaan als meerbladige TIFF**
schrijft de hele stapel terug naar één `.tif` en behoudt daarbij de 16-bit-diepte.

### Schaalbalk (publicatiefiguren)

Instellingen → Schaalbalk… stelt een schaalbalk in die in geëxporteerde
afbeeldingen wordt ingebrand. De pixelgrootte (µm/px) wordt automatisch uit
TIFF-metadata gedetecteerd — de `unit=micron`-conventie van ImageJ, de
`PhysicalSizeX` van OME-TIFF of een expliciete resolutie-eenheid in centimeters —
zodat microscopiestapels (ook meerbladige TIFF's) zichzelf kalibreren.
Printerresolutie (72/300 dpi en dergelijke) wordt bewust *niet* als kalibratie
gezien, en een handmatig ingevoerde waarde wint altijd van gedetecteerde
metadata: een plausibel ogende maar verkeerde balk in een figuur is erger dan
geen balk. Kies de hoek en de kleur (Automatisch keert de kleur om tegen de
achtergrond); het venster toont een live voorbeeld. Met de balk ingeschakeld
krijgt elke export (losse afbeelding, mapstapels, meerbladige TIFF, GIF,
alles exporteren, batch) een balk in publicatiestijl: een afgeronde 1-2-5-lengte
(ongeveer 12% van de beeldbreedte) met omlijnde, leesbare tekst. De CLI
ondersteunt dit met `--scale-bar` (automatische kalibratie) en
`--scale-um <um/px>`.

### Dieptekaart (focuspositie)

**Extra → Dieptekaart…** schat focusposities met de activiteitsmaat van de gekozen
methode (DCT, DTCWT, GFG-FGF, StackMFF V3, OpenFocus B/C). Dit is geen gemeten oppervlak
of exacte herkomstkaart van de gefuseerde pixels. Kies de huidige aangevinkte frames of
de vastgelegde bron van de geselecteerde uitvoer. Ontbrekende herkomst wordt niet vervangen
door de huidige stapel. Berekening, textuur en export blijven aan dezelfde bronopname gebonden.

Alle zes methoden vereisen een expliciete klik op **Berekenen**. Openen of wijzigen van
maat, bron, registratie of parameters start geen berekening. De parametereditor staat
direct onder de maatkeuze: DCT-mediaankernel en GFG-basiskernel gebruiken oneven waarden
1–51 (standaard 7); B gebruikt een variantievenster en C een venster voor banddoorlaatenergie,
beide oneven 1–21 px (standaard 11). DTCWT-decompositieniveaus zijn hier te bewerken
als lokaal concept, geïnitialiseerd met de bronparameters. StackMFF V3 toont een eigen
AI-uitvoermodus (standaard zachte menging, of harde selectie) en uitleg in plaats van een
getalsveld: zachte menging schat een kansgewogen continue framepositie, harde selectie
neemt per pixel het waarschijnlijkste frame. Geen van beide is een fysieke afstand. Conceptwaarden beginnen met de parameters
van de overeenkomende uitvoer of de per methode onthouden waarden in het hoofdvenster.
Ze worden tijdens deze opening per bron en methode bewaard en wijzigen het hoofdvenster,
project of de vastgelegde uitvoer niet.

Gewijzigde berekeningsinstellingen markeren de vorige 2D-kaart blijvend als **niet actueel**,
wissen het 3D-oppervlak en blokkeren alle exports, ook ruwe gegevens, fysieke diepte en
OBJ, totdat een nieuwe berekening slaagt. Actief werk wordt coöperatief geannuleerd.
Druk daarna opnieuw op **Berekenen**; er wordt geen volgende berekening ingepland of
automatisch gestart. Berekenen blijft zichtbaar en is uitgeschakeld tijdens berekenen,
annuleren en modelexport.

- **Kleurenkaart**: grijswaarden (publicatiestandaard) plus 15
  wetenschappelijke paletten (Turbo, Viridis, Magma, …) en je eigen palet —
  *Nieuw aangepast schema…* bouwt een gradiënt uit willekeurig veel
  kleurstops (dubbelklik op een stop om positie en kleur te bewerken);
  aangepaste schema's worden opgeslagen en in de keuzelijst aangeboden.
- **Omkeren / Gamma / Vervaging**: draai de diepteas om, herbalanceer de toon
  (gamma < 1 maakt lage indices helderder) en verwijder spikkels met een
  randbewuste stap.
- **Overlay**: meng de 2D-kaart met instelbare dekking over de gekoppelde uitvoer,
  alleen als hun coördinatengebieden overeenkomen.
- **Exporteren**: de gekleurde versie ondersteunt PNG/JPEG/TIFF/BMP. De ruwe 16-bit TIFF
  codeert genormaliseerde focusindices als 0–65535; dit zijn geen verliesvrije drijvende-
  kommagetallen of fysieke afstanden. JPEG/BMP worden voor ruwe gegevens geweigerd.
- **3D-model / 3D-textuur**: de langste maaszijde telt standaard 600 hoekpunten.
  Elke nieuwe opening kiest **Automatisch (ware kleuren)**: de gekoppelde uitvoer wordt
  alleen gebruikt als identiteit en coördinatengebied/ROI overeenkomen met de voltooide
  kaart. Anders wordt het eerste deelnemende frame van de voltooide berekening gebruikt,
  na registratie en in de werkelijk gebruikte volgorde. De naam van de gebruikte uitvoer
  of het frame wordt getoond. Je kunt expliciet de overeenkomende uitvoer, een deelnemend
  frame of **Diepte in pseudokleuren** kiezen; ontbrekende texturen worden gemeld.
  Het 3D-voorbeeld en de OBJ-textuur-PNG gebruiken dezelfde afbeelding. De 2D-kaart behoudt
  zijn eigen kleurenkaart en overlay; palet en Gamma kleuren echte foto's niet opnieuw.
  Alleen de expliciete pseudokleurtextuur volgt palet, Gamma en omkering. Stilstaand
  voorbeeld en export delen hetzelfde bemonsterde hoogteveld zonder verborgen extra
  vervaging. Relatief reliëf is een weergave in beeldeenheden.

De berekening loopt op de achtergrond met voortgang en de mogelijkheid tot
annuleren; de AI-methode hergebruikt dezelfde geheugenveilige tegelverwerking
als AI-fusie, zodat diepe stapels werken. Het CLI-equivalent is
`--depth-map [pad]` met `--depth-colormap`.

### Automatisch sessieherstel

OpenFocus houdt continu een momentopname van je werksessie bij (stapel,
instellingen, labels, huidig frame). Als de toepassing onverwacht sluit — een
crash, stroomuitval of geforceerd afsluiten — detecteert de volgende start dat
en biedt **Sessie herstellen** aan om precies terug te keren naar waar je was.
Bij een nette afsluiting verschijnt de vraag nooit. Herstelgegevens staan naast
je instellingen (draagbare installaties bewaren ze naast het uitvoerbare
bestand, zodat ze met je meereizen).

### Recente bestanden

Elke stapel die je opent wordt onthouden onder **Bestand → Recent geopend**
(maximaal 8 items). Druk op `Alt+1`–`Alt+8` om een item direct te heropenen.
Dezelfde stapel opnieuw openen zet hem bovenaan; items waarvan de bestanden
niet meer bestaan kun je met een melding verwijderen, en **Recente lijst wissen**
maakt het menu leeg.

### Slepen en neerzetten

Je kunt ook een map rechtstreeks op het toepassingsvenster slepen om
afbeeldingen te importeren. Extra bestanden op een bestaande stapel neerzetten
voegt ze toe (na een controle op de afmetingen).

### EXIF-oriëntatie

JPEG/TIFF-bestanden van camera's met een EXIF-oriëntatietag worden tijdens het
importeren automatisch gedraaid of gecorrigeerd, zodat staande opnamen rechtop
worden geladen.

### 16-bit-bronnen

16-bit PNG/TIFF-invoer wordt automatisch gedetecteerd en van begin tot eind
behouden: de fusie draait op volledige diepte en resultaten kunnen weer als
16-bit PNG/TIFF worden opgeslagen (JPG/BMP-exports worden naar 8-bit
omgezet, omdat die formaten geen 16 bit kunnen opslaan).

RAW-bestanden van camera's worden naar 16-bit ontwikkeld (cameraswitbalans,
sRGB-gamma) met automatische helderheidsversterking uitgeschakeld, zodat elk
frame van een stapel dezelfde belichtingstoewijzing krijgt; automatische
belichting per frame zou de helderheid tussen frames laten flikkeren. Sla op
als TIFF of PNG om de 16-bit-gegevens te behouden.

### De stapel wissen

Om alle geladen afbeeldingen te verwijderen ga je naar **Bestand → Stapel wissen**
of druk je op `Ctrl+W`.

---

## 5. Afbeeldingen bekijken en navigeren

### Door de beeldstapel navigeren

- **Schuif**: versleep de schuif onder het bronpaneel om door de frames te bladeren
- **Bestandslijst**: klik op een bestandsnaam in de lijst met bronafbeeldingen rechts
- **Toetsenbord**: gebruik de pijltjestoetsen (links/rechts) om te navigeren

### Zoomen en verschuiven

- **In-/uitzoomen**: met het muiswiel boven de afbeelding
- **Fijn zoomen**: houd `Ctrl` ingedrukt en gebruik het muiswiel voor kleinere stappen
- **Passend maken**: dubbelklik op de afbeelding om te wisselen tussen passend en 100%
- **Verschuiven**: klik en sleep om de afbeelding binnen het paneel te verplaatsen

### Gesynchroniseerde navigatie

Wanneer je in het bronpaneel navigeert, toont het resultaatpaneel het
bijbehorende resultaatframe als dat beschikbaar is.

---

## 6. Vergelijkingsweergave Vegen

De weergave Vegen legt twee afbeeldingen over elkaar in één kader met een
versleepbare verticale scheiding, waardoor misuitlijningen en
kwaliteitsverschillen direct zichtbaar worden over de splitsing heen.

### De weergave Vegen openen

1. Render minstens één resultaat (fusie of registratie)
2. Druk op de schakelaar **Vegen** rechtsboven in de titelbalk van het resultaatpaneel
3. Het resultaatgebied schakelt over naar de vergelijkingsweergave; druk weer op **Vegen** om terug te keren

### De weergave lezen

- **Kant A (links van de scheiding)**: een bronframe — volgt standaard de schuif van het bronpaneel
- **Kant B (rechts van de scheiding)**: het meest recente render
- **Witte scheiding + ronde greep**: sleep horizontaal om de splitsing te verplaatsen
- Door randen over de scheiding te slepen zie je registratiefouten die in naast elkaar geplaatste panelen moeilijk zichtbaar zijn

### Kiezen wat je vergelijkt

De balk onderaan de vergelijkingsweergave heeft één groep per kant:

- **Kant A** — vink **Bron volgen** aan (standaard) om de schuif van het bronpaneel te volgen, of vink het uit om kant A met een eigen schuif op een willekeurig frame vast te zetten
- **Kant B** — vink **Nieuwste resultaat** aan (standaard) om altijd het nieuwste render te tonen, of vink het uit om met de eigen schuif door je uitvoergeschiedenis te bladeren

Beide kanten worden live bijgewerkt: nieuwe renders, verschuivingen van de
bronschuif en verwijderde frames zijn direct zichtbaar.

### Zoomen en verschuiven

De vergelijkingsweergave gebruikt één transformatie voor beide afbeeldingen:

- **Zoomen**: muiswiel (zoomt rond de cursor)
- **Verschuiven**: klik en sleep, behalve op de scheiding
- **Resetten**: dubbelklikken

De modus Vegen en de ROI-modus delen het resultaatpaneel; het openen van de
een sluit de ander. De stapel wissen beëindigt ook de modus Vegen.

---

## 7. Beeldregistratie

Beeldregistratie corrigeert ruimtelijke misuitlijning tussen frames in je
beeldstapel. Dat is essentieel wanneer afbeeldingen kleine verschuivingen of
perspectiefveranderingen vertonen.

### Registratiemethoden

#### ECC (Enhanced Correlation Coefficient)

- Gebruikt optimalisatie om de beste uitlijning op basis van correlatie te vinden
- Geschikt voor subtiele misuitlijningen
- Levert subpixelnauwkeurigheid
- Langzamer dan Homografie, maar nauwkeuriger

#### Homografie

- Gebaseerd op het matchen van kenmerken (SIFT/ORB)
- Kan grotere geometrische transformaties aan
- Sneller dan ECC voor de eerste uitlijning
- Goed voor afbeeldingen met aanzienlijke perspectiefveranderingen

#### OpenFocus (snel)

- Begrensde similariteitsuitlijning (translatie + rotatie + schaal),
  met helderheidsvereffening per frame en automatisch bijsnijden
- Minder dan een seconde per frame — veel sneller dan ECC of Homografie
- Frames die meer dan **8% van de beeldgrootte verschuiven, blijven onuitgelijnd**
  in plaats van half gecorrigeerd. Overschrijf dit met de CLI-optie
  `--align-shift-limit`
- Kies voor niet-rigide of extreme geometrie liever ECC of Homografie

### Registratie inschakelen

1. Zoek in het rechterpaneel de groep **Registratie**
2. Kies een methode in de keuzelijst: **Geen uitlijning**, **OpenFocus (snel)**,
   **ECC** of **Homografie**
3. De snelle uitlijning van OpenFocus kan ook vóór ECC of Homografie worden
   geschakeld (eerst een globale grove pass, dan de nauwkeurige lokale pass)

### Wanneer registratie gebruiken

- Je afbeeldingen zichtbaar misuitgelijnd zijn
- De opnamen uit de hand zijn gemaakt
- De scène parallax heeft (dichtere objecten verschuiven tussen frames)
- Resultaten spookbeelden of dubbele randen vertonen

### Registratie-instellingen

Extra instellingen vind je onder **Instellingen → Registratie-instellingen**:
- **Verkleiningsbreedte**: bepaalt de resolutie van de voorbewerking (standaard: 1024px)
- Lagere waarden = snellere verwerking, mogelijk lagere nauwkeurigheid
- Hogere waarden = langzamere verwerking, mogelijk hogere nauwkeurigheid

---

## 8. Fusiemethoden

OpenFocus biedt acht fusiealgoritmen (plus een modus die alleen uitlijnt).
Elk heeft andere eigenschappen en past bij andere soorten afbeeldingen.

### Guided Filter (standaard)

- **Algoritme**: gebruikt guided filtering voor randbehoudende fusie
- **Het best voor**: algemeen gebruik, een goede balans tussen snelheid en kwaliteit
- **Voordelen**: goed randbehoud, snelle verwerking
- **Parameter**: kernelgrootte (instelbaar van 1-51, alleen oneven waarden)
  - Kleinere kernel: meer lokaal detail, mogelijk meer ruis
  - Grotere kernel: gladder resultaat, betere ruisonderdrukking

### DCT (Discrete Cosine Transform)

- **Algoritme**: fusie in het frequentiedomein met DCT-coëfficiënten
- **Het best voor**: textuurrijke afbeeldingen, wetenschappelijke beeldvorming
- **Voordelen**: behoudt fijne texturen, wiskundig onderbouwd
- **Parameter**: de kernelgrootte bepaalt het verwerkingsvenster

### DTCWT (Dual-Tree Complex Wavelet Transform)

- **Algoritme**: fusie in het waveletdomein op meerdere schalen
- **Het best voor**: complexe scènes met meerdere detailniveaus
- **Voordelen**: uitstekende analyse op meerdere schalen, goede directionaliteit
- **Fusieregel**: één beslissing per pixel en niveau voor alle oriëntaties en
  kleurkanalen (N-weg, met controle op meerderheidsconsistentie); de
  laagdoorlaatband volgt de grofste beslissing in plaats van elk frame te
  middelen
- **Parameter**: waveletniveaus (1–6, standaard 4) in het rechterpaneel. De
  methode heeft geen ruimtelijke kernel, dus de kernelschuif is verborgen.

### GFG-FGF (Gaussian-like Four-neighbourhood Gradient + Fast Guided Filter)

- **Algoritme**: implementatie van Fu et al. (*Laser & Optoelectronics
  Progress*, 2024): een scherptekaart op basis van blur-verschil |I − mean(I)|
  wordt verrijkt met de Gaussian-like four-neighbourhood gradient (GFG)-operator,
  verfijnd met een snelle guided filter, per pixel beslist (gelijke standen
  worden beslecht met een 3×3 Sobel-maat voor kleine gebieden), en de binaire
  besliskaarten worden nogmaals met de snelle guided filter vervaagd vóór de
  gewogen menging.
- **Het best voor**: microscopie en afbeeldingen met duidelijke focusgebieden
- **Voordelen**: snel, scherpe grenzen tussen focusgebieden; het geheugengebruik
  groeit niet mee met het aantal frames
- **Parameter**: kernel = grootte van het mean-filter van de grove scherptekaart

### StackMFF-V3 (deep learning)

- **Algoritme**: classificatie van de bron per pixel (vereist PyTorch). Het
  netwerk voorspelt voor elke pixel uit welk invoerframe die komt en mengt de
  frames vervolgens op basis van die waarschijnlijkheden, of kiest één frame
- **AI-uitvoer** (rechterpaneel, alleen voor deze methode):
  - **Zachte menging (gewogen)** — standaard; elke pixel is een met
    waarschijnlijkheid gewogen mix van de frames. Vloeiende overgangen, geen
    harde naden; komt overeen met de configuratie uit de publicatie
  - **Harde keuze (één frame)** — elke pixel wordt gekopieerd uit het meest
    waarschijnlijke frame. Scherpere selectie zonder interpolatie tussen frames;
    er kan een naad op pixelniveau ontstaan waar twee frames gelijk staan
- **Het best voor**: de hoogste kwaliteit wanneer een GPU beschikbaar is
- **Voordelen**: volgorde-onafhankelijk (de framevolgorde verandert het
  resultaat nooit), werkt met elke stapellengte, geen afstemming per methode,
  automatische optimalisatie
- **Vereisten**: PyTorch-installatie, optioneel een CUDA-GPU (geen torchvision)
- **Let op**: mogelijk niet beschikbaar als PyTorch niet is geïnstalleerd

### OpenFocus A / B / C

- **Algoritme**: variantiegewogen gemiddelde, dieptekaartkeuze op variantie en
  Laplaciaanse-piramidefusie
- **OpenFocus A**: variantiegewogen gemiddelde met voorkeur voor het laatste
  frame — snel en kleurvast bij korte stapels
- **OpenFocus B**: dieptekaartkeuze op variantie — kiest per pixel het scherpste
  frame; het best op gladde oppervlakken en in microscopie
- **OpenFocus C**: Laplaciaanse piramide per band — natuurlijke uitstraling bij
  getextureerde onderwerpen, versterkt lokaal contrast
- **Parameters**: A en B gebruiken een variantievenster (kernelschuif); C heeft
  geen straal. Alle drie hebben ze een vervagingssigma.

**Vervaging**: één spinbox onder de kernelschuif, met de betekenis van de
gekozen methode (bereik 0–8, stap 0,2):

| Methode | Label | Standaard | Effect |
|---|---|---|---|
| OpenFocus A | Mengvervaging (σ) | 2,0 | Gaussische vervaging van de gewichtsmap per frame — hoe breed de scherpe gebieden van naburige frames worden gemengd |
| OpenFocus B | Dieptevervaging (σ) | 1,2 | Gaussische vervaging van de dieptekaart — breedte van de overgang tussen frames |
| OpenFocus C | Mengvervaging (σ) | 3,0 | Gaussische vervaging van de gewichtsmap van elke piramideband |

0 houdt de scherpste keuze per pixel (fijnste detail, maar spikkels bij ruisige
stapels); grotere waarden verzachten overgangen en halo's. Elke methode
onthoudt zijn eigen waarde tussen sessies, en batchverwerking gebruikt dezelfde
regel.

C heeft geen straal: de methode C van Helicon biedt alleen Vervaging, dus de
kernelschuif wordt bij C verborgen in plaats van stilzwijgend niets te doen.

CLI: `--depth-smoothing SIGMA` (B) en `--blend-smoothing SIGMA` (A, C).

**Helderheidsopties gelden per methode.** *Belichting afstemmen vóór fusie* en
*Helderheidsdrift verwijderen* worden per fusiemethode apart onthouden, omdat
ze een vraag per stapel beantwoorden en geen algemene voorkeur zijn. Beide staan
standaard aan voor OpenFocus A/B/C (die geen ingebouwde belichtingskalibratie
hebben); de andere methoden houden de historische standaardwaarden (belichting
afstemmen aan, drift verwijderen uit). Bij het wisselen van methode zie je de
eigen waarden van die methode.

### Alleen uitlijning (geen fusie)

Kies **Alleen uitlijning (geen fusie)** als methode om de stapel te registreren
en de uitgelijnde frames naar de uitvoerlijst te sturen zonder te fuseren —
handig om de registratiekwaliteit te controleren of uitgelijnde frames te
exporteren.

### Een fusiemethode kiezen

1. Zoek in het rechterpaneel de groep **Fusie**
2. Kies een methode in de **keuzelijst** (er is altijd precies één geselecteerd;
   de eerste optie registreert zonder te fuseren)
3. Pas indien van toepassing de kernelgrootte aan — het label toont wat de
   kernel voor de gekozen methode betekent
4. Klik op **Renderen** om te verwerken

### Vergelijkingstabel algoritmen

| Algoritme | Snelheid | Kwaliteit | Het best voor |
|-----------|-------|---------|----------|
| Guided Filter | Snel | Goed | Algemeen gebruik |
| DCT | Gemiddeld | Goed | Texture |
| DTCWT | Gemiddeld | Zeer goed | Complexe scènes |
| GFG-FGF | Snel | Goed | Focusgebieden |
| AI-fusie (StackMFF-V3) | Langzaam (GPU) | Uitstekend | Beste kwaliteit |
| OpenFocus A | Zeer snel | Zeer goed | Korte stapels, kleurstabiliteit |
| OpenFocus B | Zeer snel | Zeer goed | Gladde oppervlakken, microscopie |
| OpenFocus C | Snel | Zeer goed | Getextureerde onderwerpen, natuurlijke uitstraling |

### Snel voorbeeld (concept-render)

Parameters afstemmen op grote stapels is traag op volledige resolutie. Vink
**Snel voorbeeld** boven de rendertoets aan en OpenFocus maakt eerst een
verkleind concept (langste zijde 1200 px) — doorgaans meerdere keren sneller,
inclusief registratie. Het voorbeeld wordt na het renderen met een
"Voorbeeld"-melding in de samenvatting van de statusbalk getoond en wordt
**niet** aan de uitvoergeschiedenis toegevoegd. Vink het vakje uit en render
opnieuw voor het resultaat op volledige resolutie. De voorbeeldmodus geldt niet
voor ROI-renders en vervuilt nooit de cache van de uitlijning op volledige
resolutie.

---

## 9. Beeldtransformaties

### Draaien

1. Ga naar **Bewerken → Draaien**
2. Kies het type draaiing:
   - **90° rechtsom**: draai alle afbeeldingen 90 graden naar rechts
   - **90° linksom**: draai alle afbeeldingen 90 graden naar links
   - **180°**: draai afbeeldingen ondersteboven

### Spiegelen

1. Ga naar **Bewerken → Spiegelen**
2. Kies het type spiegeling:
   - **Horizontaal spiegelen**: van links naar rechts
   - **Verticaal spiegelen**: van boven naar beneden

### Formaat wijzigen

1. Ga naar **Bewerken → Formaat wijzigen**
2. Er verschijnt een venster met de opties voor het wijzigen van het formaat
3. Voer nieuwe afmetingen of een schaalpercentage in
4. Alle afbeeldingen in de stapel worden evenredig geschaald

### Volgorde van transformaties

Transformaties worden op de hele beeldstapel toegepast en behouden de
uitlijning tussen frames.

---

## 10. Retoucheren, labels en annotaties

### Het resultaat retoucheren (Bewerken → Resultaat retoucheren…, `Ctrl+T`)

Fusie kan toch een lokaal defect achterlaten — een halo rond een rand met hoog
contrast, een haartje uit het verkeerde frame, ruis in een vlak gebied. Het
retoucheerpenseel kopieert pixels van een bronframe naar het resultaat:

1. Render een resultaat (of selecteer er een in de uitvoerlijst; rechtsklik →
   *Resultaat retoucheren…* werkt ook) en druk op `Ctrl+T`.
2. Schilder over het defect met de linkermuisknop. Met **Scherpste frame per
   streek automatisch kiezen** aan (standaard) kiest elke streek eerst het
   frame met het meeste detail onder de kwast — een halo herstellen is meestal
   één beweging. Zet de optie uit om zelf een frame te kiezen met de schuif of
   met `,` / `.`.
3. Pas de **straal** (`[` / `]`), de **hardheid** (zachte randen verbergen de
   naad) en de **dekking** aan. Overlappende streken binnen één beweging
   versterken elkaar nooit.
4. **Bronframe tonen** (`Tab`) toont het frame waar je van schildert. Met het
   muiswiel zoom je, met rechts/midden slepen verschuif je en met `F` maak je
   het beeld passend.
5. `Ctrl+Z` / `Ctrl+Y` maken hele streken ongedaan of voeren ze opnieuw uit.
   **Toepassen als nieuwe uitvoer** voegt de geretoucheerde afbeelding aan de
   uitvoerlijst toe (achtervoegsel `_Retouched`); de oorspronkelijke uitvoer
   blijft onaangetast.

Het penseel gebruikt de uitgelijnde frames waaruit het resultaat is ontstaan,
dus het sluit daarop aan; 16-bit-resultaten blijven 16-bit. ROI-renders en
verkleinde stapels kunnen niet worden geretoucheerd (hun frames passen niet
meer bij de geometrie van het resultaat).

### Labels toevoegen

1. Zorg dat je resultaten hebt gerenderd of afbeeldingen hebt geladen
2. Ga naar **Bewerken → Label toevoegen**
3. Voer in het venster je labeltekst in
4. Labels worden toegevoegd aan zowel de uitgelijnde stapel als de invoerstapel

### Labels verwijderen

- **Labels van uitgelijnde stapel verwijderen**: verwijdert labels van uitgelijnde afbeeldingen
- **Labels van invoerstapel verwijderen**: verwijdert labels van de originele afbeeldingen

Deze opties vind je in het submenu **Bewerken → Label verwijderen**.

---

## 11. Instellingen en configuratie

### Tegelinstellingen (geheugenoptimalisatie)

Voor grote afbeeldingen gebruikt OpenFocus verwerking in tegels om
geheugenproblemen te voorkomen.

Bereikbaar via **Instellingen → Tegelinstellingen**:

- **Blokgrootte tegel**: grootte van elke verwerkingstegel (standaard: 1024px)
- **Tegeloverlap**: overlap tussen tegels voor naadloze menging (standaard: 256px)
- **Tegeldrempel**: afbeeldingsgrootte waarboven tegels worden gebruikt (standaard: 2048px)

**Aanbevelingen**:
- Kleinere blokgrootte: minder geheugengebruik, langzamere verwerking
- Grotere blokgrootte: meer geheugengebruik, snellere verwerking
- Verhoog de overlap als je zichtbare naden in resultaten ziet

### Registratie-instellingen

Bereikbaar via **Instellingen → Registratie-instellingen**:

- **Verkleiningsbreedte**: resolutie van de voorbewerking voor registratie
  - Standaard: 1024px
  - Lager voor snelheid, hoger voor nauwkeurigheid

### Prestatiegegevens

Zie [PERFORMANCE.md](./PERFORMANCE.md) voor gemeten doorvoer en
geheugengebruik bij grote, echte stapels, plus praktische richtlijnen voor diepe
stapels en invoer met hoge resolutie.

### Threadinstellingen

Bepaal de parallelle verwerking op de CPU:

Bereikbaar via **Instellingen → Threadinstellingen**:

- `0` kiest automatische planning en is de standaard voor nieuwe instellingen.
- Positieve waarden begrenzen het aantal threads; opgeslagen grenzen blijven behouden.
- CPU-capaciteit, resourcemodus en beschikbaar geheugen kunnen de gelijktijdigheid verlagen.

### GPU-versnelling

OpenFocus gebruikt de GPU (CUDA/MPS) alleen voor het neurale AI-model
(StackMFF-V3); klassieke algoritmen draaien altijd op de CPU.

- Schakel via **Instellingen → GPU-versnelling**
- Uitvinken dwingt alles naar de CPU — handig als GPU-stuurprogramma's problemen geven
- Het statuspaneel toont `CPU*` zolang de GPU is uitgeschakeld
- De keuze geldt vanaf het volgende render en wordt onthouden

### AI-batchgrootte

Bereikbaar via **Instellingen → AI-batchinstellingen**:

- De opgeslagen batchlimiet is standaard 2; momenteel wordt op elk apparaat één tegel per aanroep verwerkt.
- Een hogere limiet schakelt geen parallelle GPU-batches in.

### Thema's

Er zijn twee interfacethema's: **Donker (standaard)** en **Licht**. Wissel via
**Instellingen → Thema**; de keuze wordt direct toegepast (de Windows-titelbalk
volgt) en onthouden. Bij de eerste start kies je een thema in het
welkomstvenster.

### Draagbare modus

Maak een leeg bestand met de naam `OpenFocus.portable` naast `OpenFocus.exe`
(de draagbare zip bevat er al een) en alle gebruikersgegevens — instellingen,
logs, recente bestanden — worden opgeslagen in een map `OpenFocusData` naast de
toepassing in plaats van in je gebruikersprofiel, zodat de installatie op een
USB-stick meekan. Verwijder het markeringsbestand om terug te keren naar het
normale profiel.

### EXIF-metadata

Metadata van de bron (cameramodel, belichting, oriëntatie enzovoort) blijft
bewaard: de oriëntatie wordt bij het laden toegepast, en PNG/TIFF/WebP-exports
nemen de originele EXIF automatisch mee. JPG/BMP-exports kunnen geen volledige
EXIF opslaan en worden zonder geschreven.

### HEIC/HEIF-invoer

Foto's van een iPhone worden direct geladen. Dit vereist het optionele pakket
`pillow-heif` (`pip install pillow-heif`; zit in de gepackagede builds).

### Updates

Projectbestanden (`.ofproj`) gedragen zich als een document: de titelbalk toont
de projectnaam met een `*` zolang er niet-opgeslagen wijzigingen zijn, en bij
het sluiten van de app of het openen van een ander project wordt gevraagd of je
ze wilt verwerpen. Het project bewaart de framelijst, de laadschaal, elke
renderoptie, labels en het huidige frame — de afbeeldingen zelf worden
gerefereerd, niet gekopieerd.

Help → Controleren op updates vergelijkt je versie met de GitHub Releases.
**Downloaden en installeren** haalt de nieuwe installer binnen in de app (met
voortgang) en start die; Downloadpagina openen gaat naar de releaselijst. De
controle draait stil hooguit één keer per dag.

### Registratiecache

OpenFocus kan registratieresultaten opslaan in een map `.openfocus_cache` naast
de bronstapel. Dezelfde stapel opnieuw openen (bestanden ongewijzigd) slaat de
uitlijning dan volledig over. Schakel via **Instellingen → Registratiecache**;
verouderde items worden automatisch vervangen.

### Bewaarde instellingen

Al het volgende blijft automatisch bewaard na een herstart van de toepassing:
het aantal threads, de tegelparameters, de verkleiningsbreedte voor registratie,
de batchgrootte van AI-fusie (StackMFF-V3), de schakelaar voor GPU-versnelling,
de schakelaar voor de registratiecache, de interfacetaal, de venstergrootte en
paneelindeling, het sleepformaat voor exporteren en je recent geopende stapels.
Instellingenvensters slaan op zodra je ze bevestigt; alles wordt ook opgeslagen
wanneer de toepassing sluit. Voorkeuren worden per gebruiker bewaard in
`OpenFocus/OpenFocus.ini` (QSettings INI-formaat).

### Interfacetaal

Wissel op elk moment via **Instellingen → Taal → English / 中文**. De wijziging
wordt direct toegepast en onthouden voor volgende sessies.

### Automatische updatecontrole

Eén keer per dag, enkele seconden na het starten, controleert OpenFocus stil
GitHub op een nieuwere release. Als die bestaat, verschijnt er een banner boven
de beeldpanelen: de versie, **Nu bijwerken** (downloadt en herstart in één stap),
**Wat is er nieuw** (releaseopmerkingen) en een knop om te sluiten. Sluiten
verbergt de banner alleen voor die versie — de volgende release brengt hem
terug, en **Help → Controleren op updates** zet hem direct weer aan.

Terwijl de download op de achtergrond loopt, toont de statusbalk de voortgang.
Je kunt ook altijd handmatig controleren via **Help → Controleren op updates**.

---

## 12. Batchverwerking

Verwerk meerdere mappen automatisch:

1. Ga naar **Extra → Batchverwerking** of klik op de batchoptie
2. In het batchvenster:
   - **Mappen toevoegen**: selecteer mappen met beeldreeksen
   - **Selectie verwijderen**: verwijder geselecteerde mappen uit de lijst
   - **Alles wissen**: verwijder alle mappen
3. Stel de uitvoer in:
   - **Submap in de bronmap aanmaken**: maakt resultaten in elke invoermap
   - **Zelfde als bronmap**: overschrijft of bewaart naast de invoer
   - **Uitvoermap opgeven**: geef zelf een uitvoerlocatie op
4. Controleer de huidige fusie- en registratie-instellingen
5. Klik op **Batchverwerking starten** om te beginnen

Het batchvenster toont de voortgang in realtime. Je kunt de verwerking op elk
moment annuleren.

### Opmerkingen bij batchverwerking

- Gebruikt dezelfde fusie- en registratie-instellingen als handmatige verwerking
- Elke map wordt afzonderlijk verwerkt
- Bestandsnamen bevatten een tijdstempel en parameters ter identificatie

---

## 13. Resultaten exporteren

### Eén resultaat opslaan

1. Selecteer het resultaat dat je wilt opslaan in de uitvoerlijst
2. Ga naar **Bestand → Opslaan** of druk op `Ctrl+S`
3. Kies formaat en locatie
4. Klik op Opslaan

### Sleepformaat voor exporteren

Als je resultaten uit de uitvoerlijst sleept, worden afbeeldingsbestanden
geschreven in het formaat dat je kiest in het contextmenu van de uitvoerlijst
(**Sleepformaat**: JPG, PNG of TIFF). De keuze wordt onthouden.

### De uitgelijnde stapel opslaan

1. Ga naar **Bestand → Stapel opslaan → Uitgelijnde stapel**
2. Kies het opslagformaat:
   - **Opslaan als map**: slaat elk uitgelijnd frame op als apart afbeeldingsbestand
   - **Opslaan als GIF**: maakt een geanimeerde GIF van de uitgelijnde reeks

### De invoerstapel opslaan

1. Ga naar **Bestand → Stapel opslaan → Invoerstapel**
2. Kies het opslagformaat:
   - **Opslaan als map**: slaat de originele frames op als aparte afbeeldingen
   - **Opslaan als GIF**: maakt een geanimeerde GIF van de invoerreeks

### Exportformaten

- **JPG**: gecomprimeerd, goede verhouding tussen kwaliteit en bestandsgrootte
- **PNG**: verliesvrij, ondersteunt transparantie
- **BMP**: ongecomprimeerd, maximale kwaliteit
- **TIFF**: hoge kwaliteit, ondersteunt lagen
- **GIF**: animatieformaat

---

## 14. Opdrachtregelinterface

OpenFocus kan zonder GUI draaien — handig voor gescripte pipelines en CI.
Vanuit een Python-omgeving (of naast het gepackagede uitvoerbare bestand):

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

| Optie | Beschrijving |
|--------|-------------|
| `--input, -i` | Afbeeldingsmap, videobestand of een of meer afbeeldingsbestanden |
| `--output, -o` | Uitvoerpad van de afbeelding (de extensie bepaalt het formaat) |
| `--method, -m` | `stackmffv3` (standaard), `none`, `openfocus_a`, `openfocus_b`, `openfocus_c`, `guided_filter`, `dct`, `dtcwt`, `gfgfgf` |
| `--align, -a` | `none` (standaard), `homography`, `ecc`, `both`, `openfocus` |
| `--align-shift-limit` | Alleen bij `openfocus`-uitlijning: maximaal geaccepteerde verschuiving als deel van de beeldgrootte (standaard 0,08; frames daarboven blijven onuitgelijnd) |
| `--exposure-match` | Stem het gemiddelde van elk frame vóór de fusie af op het laatste frame (standaard: aan; `--no-exposure-match` schakelt dit uit). Versterking voor de hele stapel, zodat getegelde uitvoer naadloos blijft; geldt voor elke methode en voor de helderheidsreferentie van de OpenFocus-uitlijning |
| `--base-frame` | Optionele stap na de fusie: stem de laagfrequente delen van de uitvoer af op het `last`/`first` frame (`none` is standaard) |
| `--sort` | Framevolgorde voor `openfocus_a/b/c`: `asc` (standaard, bestandsvolgorde), `desc` (keert een veer-naar-dichtbij-scan om), `off` |
| `--dust-map WHITE` | Flat-field-correctie (stofkaart) op basis van een onscherpe witveldopname |
| `--kernel, -k` | Kernelgrootte (oneven gemaakt; voor de openfocus-familie bepaalt deze de straal van het variantievenster) |
| `--depth-smoothing SIGMA` | Alleen `openfocus_b`: Gaussische sigma van de diepteovergang tussen frames (standaard 1,2; 0 behoudt het fijnste detail) |
| `--blend-smoothing SIGMA` | Alleen `openfocus_a` / `openfocus_c`: Gaussische sigma van de gewichtsmap per frame (standaard 2,0 voor A, 3,0 voor C; 0 behoudt de scherpste keuze per pixel) |
| `--threads, -t` | Threadlimiet; `0` = automatisch (standaard), positieve waarden beperken de gelijktijdigheid |
| `--resource-preference` | Resourcemodus: `balanced` (standaard), `responsive` of `maximum` |
| `--batch-size` | Batchlimiet voor AI-tegels (standaard 2); momenteel één tegel per aanroep |
| `--cpu` | Dwing de CPU af, ook als er een GPU beschikbaar is |
| `--downscale` | Verkleiningsbreedte voor kenmerkdetectie bij registratie |
| `--tile-size` | Blokgrootte van tegels voor grote afbeeldingen |
| `--output-dir, -d` | Batchmodus: fuser elke invoermap en schrijf `<map>.png` per stapel |
| `--scale-bar` | Teken een schaalbalk (kalibratie automatisch gedetecteerd uit TIFF-metadata) |
| `--depth-map [pad]` | Sla ook een gekleurde dieptekaart (focuspositie) op (standaard: `<output>_depth.png`) |
| `--depth-colormap` | Kleurenkaart voor `--depth-map`: turbo, viridis, jet, gray, … (standaard turbo) |
| `--scale-um` | Pixelgrootte in µm/px, overschrijft metadata (impliceert `--scale-bar`) |

Exitcodes: `0` succes, `1` verwerkingsfout, `2` gebruiksfout. Met
`python main.py --help` zie je alle opties; starten met een kaal pad
(`python main.py C:\stack`) opent nog steeds de GUI met die map geladen.

---

## 15. Sneltoetsen

### Bestandsbewerkingen

| Sneltoets | Actie |
|----------|--------|
| `Ctrl+O` | Map openen |
| `Ctrl+Shift+O` | Video openen |
| `Ctrl+S` | Resultaat opslaan |
| `Ctrl+Shift+S` | Uitgelijnde stapel opslaan |
| `Ctrl+W` | Stapel wissen |
| `Ctrl+Q` | Afsluiten |
| `Ctrl+R` | Renderen starten |
| `Ctrl+Shift+R` | Alles vergelijken |
| `Ctrl+T` | Resultaat retoucheren |
| `Ctrl+M` | Dieptekaart |
| `Ctrl+E` | Vergelijking Vegen aan/uit |
| `Ctrl+Shift+B` | Batchverwerking |
| `Ctrl+Z` / `Ctrl+Y` | Ongedaan maken / opnieuw uitvoeren (stapelbewerkingen) |

### Navigatie

| Sneltoets | Actie |
|----------|--------|
| `Linkerpijl` | Vorig frame |
| `Rechterpijl` | Volgend frame |
| `Muiswiel` | In-/uitzoomen |
| `Ctrl + wiel` | Fijn zoomen |
| `Dubbelklik` | Wisselen tussen passend en 100% |

### Algemeen

| Sneltoets | Actie |
|----------|--------|
| `Spatie` | Modus verschuiven aan/uit |
| `Delete` | Geselecteerd item verwijderen |

---

## 16. Problemen oplossen

### Veelvoorkomende problemen

#### "Geen afbeeldingen geladen"

- Controleer of je eerst een afbeeldingsmap of video hebt geïmporteerd
- Controleer of de map ondersteunde afbeeldingsformaten bevat

#### Fuseresultaten zien er wazig of spookachtig uit

- Schakel beeldregistratie in (kies er één in de keuzelijst: Homografie / ECC / OpenFocus (snel))
- Zorg dat afbeeldingen goed zijn uitgelijnd vóór de fusie
- Probeer een ander fusiealgoritme
- Vergroot de kernelgrootte voor een gladder resultaat

#### AI niet beschikbaar

- Installeer PyTorch: `pip install torch torchvision`
- Installeer voor GPU-ondersteuning een CUDA-versie van PyTorch
- Controleer via **Help → Omgevingsinfo**

#### Onvoldoende geheugen

- Schakel tegelverwerking in via **Instellingen → Tegelinstellingen**
- Verklein de blokgrootte van tegels
- Gebruik kleinere kernels
- Verklein de afbeeldingen

#### Registratie mislukt

- Probeer telkens maar één registratiemethode
- Verlaag de verkleiningsbreedte in **Instellingen → Registratie-instellingen**
- Zorg dat afbeeldingen voldoende contrast en kenmerken hebben

#### Trage verwerking

- Verhoog het aantal threads in **Instellingen → Threadinstellingen**
- Gebruik een krachtigere CPU
- Verlaag de resolutie van de afbeeldingen
- Gebruik eenvoudigere fusiealgoritmen (Guided Filter in plaats van AI-fusie)

#### Windows SmartScreen-waarschuwing bij de eerste start

De uitgebrachte binaries zijn niet ondertekend. Klik op **Meer informatie → Toch uitvoeren**.
De eerste start na het uitpakken kan 30–60 seconden duren.

#### macOS meldt dat de app "niet kan worden geopend" of "beschadigd" is

De app is niet ondertekend en niet genotariseerd. Nadat je `OpenFocus.app` naar
Programma's hebt verplaatst, voer je uit:

```bash
xattr -cr /Applications/OpenFocus.app
```

of je rechtsklikt op de app en **Open** kiest. Uitgebrachte macOS-builds zijn
voor Apple Silicon (M1/M2/M3/M4); Intel-Macs kunnen vanaf de broncode draaien.

### Hulp krijgen

1. **Omgevingsinfo**: ga naar **Help → Omgevingsinfo** om je installatie te controleren
2. **Contact opnemen**: ga naar **Help → Contact opnemen** voor ondersteuningsinformatie

---

## Bijlage: details van de fusiealgoritmen

### Guided Filter

De guided filter voert randbehoudende vervaging uit met een hulpbeeld. Hij
berekent lokale lineaire transformaties tussen het hulpbeeld en het invoerbeeld
en past die toe om de uitvoer te produceren. Deze methode is sterk in het
behouden van randen terwijl ruis wordt verminderd.

### DCT (Discrete Cosine Transform)

DCT zet afbeeldingen om naar het frequentiedomein, waar fusiebeslissingen op
basis van de grootte van coëfficiënten worden genomen. Hoogfrequente
componenten (details) blijven behouden, terwijl laagfrequente componenten
(gladde gebieden) op basis van lokale energie worden samengevoegd.

### DTCWT (Dual-Tree Complex Wavelet Transform)

DTCWT levert een decompositie van afbeeldingen op meerdere schalen en in
meerdere richtingen. Kenmerken op verschillende schalen en oriëntaties worden
vastgelegd, wat geavanceerde fusieregels mogelijk maakt die zowel textuur- als
structuurinformatie behouden.

### GFG-FGF

Implementeert Fu et al. (2024): de grove scherptekaart |I − mean(I)| wordt
verrijkt met de Gaussian-like four-neighbourhood gradient — het 3×3-sjabloon
T = [1 2 1; 2 4 2; 1 2 1]/16 geeft P = I⊗T, en
G = (|I(x±1,y) − P|)/2 + (|I(x,y±1) − P|)/2, waarbij waarden onder T0 = 0.005
worden weggelaten — en daarna verfijnd met een snelle guided filter. Elke pixel
kiest het frame met de grootste scherptewaarde (bij gelijke stand: een 3×3
Sobel-maat voor kleine gebieden), en de binaire besliskaarten worden met de
snelle guided filter vervaagd vóór de gewogen menging.

### AI-fusie (StackMFF-V3)

Een deep-learningbenadering met een neuraal netwerk dat is getraind op grote
datasets van multi-focus-beeldparen. Het netwerk leert automatisch optimale
fusiestrategieën en levert toonaangevende resultaten wanneer alles goed is
geconfigureerd.

---

*OpenFocus is een open-sourceproject. Ga voor updates, bugmeldingen en
functieverzoeken naar de projectrepository.*
