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

OpenFocus is een professionele desktoptoepassing voor multifocus-beeldfusie, bedoeld voor onderzoekers, ingenieurs en professionals in beeldverwerking. De software verwerkt multifocus-beeldreeksen met geavanceerde registratie- en fusiealgorithmen en levert alles-in-focus-beelden met volledige scherptediepte.

### Belangrijkste functies

- **Multifocus-beeldfusie**: meerdere afbeeldingen met verschillende scherpstelpunten combineren tot één volledig scherp beeld
- **Tien fusiemethoden**: Guided Filter, DCT, DTCWT, GFG-FGF, StackMFF, StackMFF V2, StackMFF V3 en OpenFocus A, B en C; daarnaast een modus met alleen uitlijning
- **Beeldregistratie**: verschoven beeldreeksen uitlijnen met ECC, homografie, homografie met daaropvolgende ECC-verfijning of de snelle OpenFocus-uitlijning
- **Automatische achtergrond**: een onrustige gestapelde achtergrond vervangen door één echt frame, en zien waar geen frame de scène scherp zag
- **Bronvaste retouche**: pixels schilderen vanuit de frames die het resultaat echt hebben gemaakt, met bron- en herstelpenseel, belichtingspenseel en ongedaan maken per streek
- **Veegvergelijking**: de uitlijning controleren en resultaten vergelijken met een A/B-scheiding
- **Projectbestanden**: reproduceerbare instellingen en bronverwijzingen opslaan in `.ofproj`-bestanden; resultaat- en retouchepixels worden apart geëxporteerd
- **Verwerkingssjablonen**: fusie-, registratie- en resource-instellingen opslaan als `.oftemplate` en op elke geladen stapel toepassen
- **Meerbladige TIFF**: Z-stapels uit één bestand worden pagina voor pagina geladen en weer als één bestand opgeslagen
- **Dieptekaart**: focuspositie per pixel met wetenschappelijke kleurenschalen, index- en fysische exporten en een 3D-oppervlak
- **Schaalbalken**: schaalbalken op publicatieniveau met automatische µm/px-kalibratie uit TIFF-metadata
- **Automatisch sessieherstel**: na een crash wordt de laatste werksessie aangeboden om te herstellen
- **Twaalf interfacetalen**: English, 中文, 日本語, Español, Deutsch, Français, Русский, 한국어, Português, Italiano, Nederlands en Polski, met automatische detectie van de systeemtaal
- **Headless- en agent-CLI**: headless pijplijnen voor CI scripten of een draaiend venster via het lokale besturingskanaal bedienen, inclusief recepten en het afspelen van retouchecontrolepunten
- **Flexibel exporteren**: resultaten opslaan als losse afbeeldingen, mappen, meerbladige TIFF, TIFF met lagen of GIF-animaties
- **Beeldtransformaties**: beeldstapels draaien, spiegelen en van formaat wijzigen
- **Labels**: beheerde tekstlabels voor bron-, uitgelijnde en resultaatstapels
- **Blijvende instellingen**: voorkeuren, taal en recent geopende stapels blijven bewaard na een herstart

### Ondersteunde invoerformaten

- Afbeeldingsbestanden: JPG/JPEG, PNG, BMP, TIFF, WebP en andere gangbare formaten
- Videobestanden: MP4, AVI, MOV, MKV, WMV, FLV en WEBM (worden automatisch naar beeldreeksen gedecodeerd)
- Mappen met genummerde beeldreeksen
- RAW-bestanden van camera's (DNG, CR2/CR3, NEF, ARW, RAF, ORF, RW2 en andere gangbare RAW-formaten). Het decoderen gebeurt met `rawpy` (LibRaw), dat `pip install openfocus` installeert en dat de Windows/macOS-builds meeleveren

Transparante PNG-stapels kunnen in de desktopapp, in batchverwerking en op de opdrachtregel zonder groepsfusie worden verwerkt. Elk frame moet een 8- of 16-bit PNG met dezelfde afmetingen en bitdiepte en hetzelfde alfamasker zijn. Het masker moet buiten precies één niet-lege rechthoek nul zijn en binnen die rechthoek één constante, niet-nul dekkingsgraad hebben; gedeeltelijk transparante constante dekking is toegestaan. Alleen de zichtbare rechthoek wordt gefuseerd, zodat verborgen kleuren geen invloed hebben op de scherptemeting of tegelgrenzen. De PNG-uitvoer behoudt de oorspronkelijke canvasgrootte en dekking; verborgen uitvoerkleuren zijn nul. Bron- en resultaatweergaven tonen transparantie tegen een schaakbordpatroon.

Laad op volledige grootte, kies geen uitlijning, schakel Snel voorbeeld en ROI uit, render en sla op als PNG. Batchverwerking vereist een fusiemethode en PNG-uitvoer zonder een uitgelijnde stapel op te slaan. Uitlijning, formaatwijziging of verkleining, flat-fieldcorrectie, dieptekaarten, methodenvergelijking, retoucheren en schaalbalken zijn niet beschikbaar voor transparante gegevens; labels kunnen niet in een export worden ingebrand. Een fusiemethode die de afmetingen van de rechthoek wijzigt, wordt geweigerd. Afwijkende of zachte maskers, verschillende maskers per frame, transparante TIFF-invoer en niet-PNG-uitvoer, inclusief GIF en meerbladige TIFF, zijn niet mogelijk. De CLI weigert bovendien sensorreiniging en dieptekaarten voor transparante invoer.

Het CLI-equivalent is bijvoorbeeld `python main.py --input ./png-stack --output ./result.png --method guided_filter --align none`.

### Ondersteunde uitvoerformaten

Bij **Opslaan als** bewaart gewone TIFF het resultaat als één afbeelding; TIFF met lagen bevat de bronframes als lagen. Linear DNG staat altijd in de lijst, maar opslaan vereist een camera-lineair resultaat zonder ingebakken bewerkingen uit **Native RAW openen**, met camerakalibratie.

- Losse afbeeldingen: JPG, PNG, BMP, TIFF, WebP
- Beeldreeksen: opgeslagen als mappen
- Animaties: GIF-formaat
- Meerbladige TIFF: de hele stapel in één bestand, met 8 of 16 bit
- TIFF met lagen: retoucheringsresultaat en bronframes als lagen

---

## 2. Aan de slag

### Installatie

Download de nieuwste Windows- of macOS-build van de [OpenFocus-website](https://openfocus.top/#download). Dezelfde bestanden en oudere versies staan ook op [GitHub Releases](https://github.com/Xinzhe99/OpenFocus/releases).

| Platform | Bestand | Type |
|----------|---------|------|
| Windows 10/11 (64-bit) | `OpenFocus-*-setup.exe` | **Installer** (aanbevolen) |
| Windows 10/11 (64-bit) | `OpenFocus-*-windows-x64.zip` | Draagbaar (uitpakken en starten) |
| macOS Apple Silicon (M1–M4) | `OpenFocus-*-macos.dmg` | Schijfimage (naar Programma's slepen) |
| macOS Apple Silicon (M1–M4) | `OpenFocus-*-macos-arm64.zip` | Draagbaar (uitpakken) |

Of de Windows-builds ondertekend zijn, hangt af van de specifieke release (zie de releaseopmerkingen); een handtekening is niet gegarandeerd. Bij de eerste start kan daarom een beveiligingsvraag verschijnen: onder **Windows** SmartScreen → 'Meer informatie' → 'Toch uitvoeren' (de eerste start kan 30–60 s duren). De macOS-builds zijn niet genotariseerd; na de DMG-installatie (naar Programma's) voert u indien nodig `xattr -cr /Applications/OpenFocus.app` uit, na de terminalinstallatie (naar `~/Applications`) `xattr -cr ~/Applications/OpenFocus.app`, of u klikt met rechts op de app → **Open**.

Bekijk voor installatie vanaf de terminal eerst de opdracht voor macOS of Linux met `--dry-run` en voer die daarna zonder die optie uit. Op Linux installeert dit de headless CLI; macOS-builds vereisen Apple Silicon en worden in `~/Applications` geïnstalleerd.

```bash
curl -fsSL https://openfocus.top/install.sh | sh -s -- --dry-run
curl -fsSL https://openfocus.top/install.sh | sh
```

Voer op Windows de installer uit vanuit PowerShell:

```powershell
irm https://openfocus.top/install.ps1 | iex
```

Installeer als alternatief de CLI op Windows, macOS en Linux — dezelfde pijplijn, zonder GUI-stack (de wheel is bewust niet afhankelijk van PyQt6):

```bash
pip install openfocus
openfocus -i ./stack_folder -o ./result/fused.png
```

### De toepassing starten

Na de installatie start je OpenFocus als volgt:

1. **Windows (uitvoerbaar bestand)**: dubbelklik op de OpenFocus-snelkoppeling of het uitvoerbare bestand
2. **macOS**: open `OpenFocus.app` vanuit Programma's
3. **Python-broncode**: voer `pip install -r requirements.txt` uit en start daarna `python main.py` in je geactiveerde omgeving

De eerste start opent het welkomstvenster, waar je het thema kiest en de meegeleverde demostapel kunt laden. Een fris profiel gebruikt standaard het lichte thema; de keuze wordt onthouden.

### Snelstartprocedure

1. Importeer je beeldreeks (Bestand → Map openen of slepen en neerzetten)
2. Stel de fusie-instellingen in op het rechterpaneel
3. De registratie start met **ECC**; kies **Geen uitlijning** als de frames al zijn uitgelijnd
4. Pas de kernelgrootte of methodespecifieke parameters aan als dat nodig is
5. *(optioneel)* Open **Renderinstellingen** en vink **Snel voorbeeld** aan om de parameters eerst op een verkleind concept te testen
6. Klik op **▶ Renderen** om de verwerking te starten
7. Exporteer je resultaten

---

## 3. Overzicht van de hoofdinterface

De OpenFocus-interface bestaat uit drie hoofdgebieden:

### Linkerpaneel: beeldweergave

Het linker deel van het venster toont de afbeeldingen in een gesplitste weergave:

- **Bronpaneel (boven)**: toont de ingevoerde beeldstapel
- **Resultaatpaneel (onder)**: toont de fusie- of registratieresultaten
- **Navigatieschuif**: gebruik de schuif onder elk paneel om door de frames te bladeren
- **Beeldinfolabels**: tonen de huidige framepositie en de beeldafmetingen

Je kunt de panelen van grootte veranderen door de splitter ertussen te slepen.

### Rechterpaneel: bedieningspaneel

Het rechterpaneel bevat alle configuratieopties, verticaal gegroepeerd:

1. **Fusie-instellingen**: fusiemethode en bijbehorende parameters kiezen
2. **Registratie-instellingen**: uitlijningsmethode en bijbehorende parameters kiezen
3. **Renderinstellingen**: de dialoog bevat snel voorbeeld, framevolgorde, belichtingsafstemming, verwijdering van helderheidsdrift en uitvoerverscherping
4. **Actieknoppen**: renderen starten, herstellen, renderinstellingen openen en alle methoden renderen
5. **Lijst met bronframes**: alle geladen afbeeldingen met bestandsnamen en selectievakjes
6. **Uitvoerlijst**: de gegenereerde resultaten

### Menubalk

De menubalk bovenaan biedt toegang tot alle functies:

- **Bestand**: importeren, projecten, verwerkingssjablonen, opslaan, stapel opslaan en de lijst met recent geopende bestanden
- **Bewerken**: beeldtransformaties, retoucheren en het bewerken van losse beeldbestanden
- **Extra**: batchverwerking, methodenvergelijking, dieptekaart, stofkaart (flat-field), schaalbalk en labels
- **Instellingen**: geavanceerde instellingen (registratie en uitlijningsdiagnose), GPU-versnelling, thema en taal
- **Help**: snelstartgids, omgevingsinfo, feedback, contact, logmap en updatecontrole

---

## 4. Afbeeldingen en video importeren

### Een afbeeldingsmap openen

1. Ga naar **Bestand → Map openen** of druk op `Ctrl+O`
2. Selecteer een map met je beeldreeks
3. Er verschijnt een dialoog voor de laadschaal — kies de schaal (100 % = volledige resolutie)
4. Klik op OK om de afbeeldingen te laden

Vink **Niet meer vragen — altijd op deze schaal laden** in dat dialoog aan om voortaan zonder vraag op de gekozen schaal te laden; **Instellingen → Geavanceerde instellingen… → Opstarten en updates → Bij import om laadschaal vragen** zet de vraag weer aan. Een geladen stapel kan later nog worden verkleind via **Bewerken → Formaat wijzigen**.

De software zal automatisch:

- beeldbestanden herkennen en natuurlijk op bestandsnaam sorteren
- het eerste frame in het bronpaneel tonen
- de lijst met bronframes vullen met alle frames

### Videobestanden openen

1. Ga naar **Bestand → Video openen** of druk op `Ctrl+Shift+O`
2. Selecteer een videobestand (MP4, AVI, MOV, MKV, WMV, FLV, WEBM)
3. De video wordt automatisch in afzonderlijke frames gedecodeerd
4. Elk frame wordt onderdeel van de beeldstapel

### Projectbestanden

**Bestand → Project opslaan** slaat bronverwijzingen en reproduceerbare instellingen op in een `.ofproj`: de exacte framevolgorde (inclusief TIFF-pagina's en dubbele frames), uitsluitingen, draaiingen en spiegelingen, laadschaal, renderingparameters per methode, labels, schaalbalkstijl en het huidige frame. Resultaat- en retouchepixels moeten apart worden geëxporteerd. Bewerkingen die niet uit de bronbestanden kunnen worden gereconstrueerd, vereisen eerst het opslaan van een nieuwe invoerstapel.

**Bestand → Project openen…** bouwt de vastgelegde frames op voordat de huidige stapel wordt vervangen. Als bronnen zijn verplaatst, kies je de nieuwe bronmap en bevestig je de volledige toewijzing van oude naar nieuwe paden. Sla het project daarna opnieuw op om de nieuwe verwijzingen te bewaren. Een project gedraagt zich als een document: de titelbalk toont de naam met een `*` zolang wijzigingen niet zijn opgeslagen; bij het sluiten van de app of openen van een ander project wordt om bevestiging gevraagd. Bronafbeeldingen worden gerefereerd, niet gekopieerd.

### Verwerkingssjablonen

Een verwerkingssjabloon bevat instellingen, nooit bronpaden. **Bestand → Verwerkingssjabloon opslaan…** schrijft een `.oftemplate` met de huidige fusiemethode en parameters, registratiemethode en parameters, helderheidsopties per methode, threadlimiet, tegelinstellingen, registratiebreedte, StackMFF AI-batchgrootte, verscherping, omgekeerde volgorde en de schakelaar voor sensorruisonderdrukking. Een pad naar een defectmasker wordt bewust niet opgeslagen.

**Bestand → Verwerkingssjabloon toepassen…** past die instellingen toe op de al geladen stapel zonder de huidige bronnen te vervangen. De kiezer accepteert ook een `.ofproj` (de instellingen worden zonder de projectbronnen geëxtraheerd) en een recept `.json` met projectinstellingen. Toepassen markeert het project als gewijzigd en kan algoritmeparameters en resource-instellingen veranderen; controleer het rechterpaneel vóór het renderen. Recepten blijven het headless formaat en bevatten het pad naar het defectmasker plus de vingerafdruk.

### Demostapel

Het welkomstvenster biedt **Demostapel laden**: zes meegeleverde voorbeeldfoto's uit een echte focusstapel, zodat je de volledige workflow meteen zonder eigen afbeeldingen kunt uitproberen. Ook handig om een installatie te controleren.

### Scherptecurve

Een kleine curve boven de navigatieschuif van de bron toont de scherpte per frame (genormaliseerd). Onscherpe frames verschijnen als dips met een rode stip. Klik op de curve om naar dat frame te springen; bij hover wordt het framenummer getoond.

### Meerbladige TIFF-stapels

Microscopiesoftware (ZEN, NIS-Elements, ImageJ/Fiji) exporteert een volledige Z-stapel vaak als ÉÉN meerbladig TIFF-bestand. OpenFocus leest elke pagina van zo'n bestand als een afzonderlijk frame van de stapel — sleep of open het bestand en de stapel verschijnt compleet (frames heten `<file>_page_001` enzovoort). Enkelbladige TIFF's gedragen zich precies zoals voorheen. Het omgekeerde werkt ook: **Bestand → Stapel opslaan → Uitgelijnde stapel / Invoerstapel → Opslaan als meerbladige TIFF** schrijft de hele stapel terug naar één `.tif` en behoudt de 16-bitdiepte.

### Schaalbalk (publicatieafbeeldingen)

Als het gekozen doel geen bruikbare metadata heeft, wordt handmatige kalibratie geselecteerd en is een geldige pixelgrootte vereist.

**Extra → Schaalbalk…** configureert een schaalbalk die in geëxporteerde afbeeldingen wordt ingebrand. De pixelgrootte (µm/px) wordt automatisch uit TIFF-metadata gedetecteerd — via ImageJ's `unit=micron`-conventie, OME-TIFF's `PhysicalSizeX` of een expliciete resolutie-eenheid in centimeters — zodat microscopiestapels (inclusief meerbladige TIFF's) zichzelf kalibreren. Printerresolutie (72/300 dpi en dergelijke) wordt bewust *niet* als kalibratie behandeld, en een handmatig ingevoerde waarde wint altijd van gedetecteerde metadata: een plausibel ogende maar foute balk in een figuur is erger dan geen balk. Kies de hoek en kleur (Auto keert om tegen de achtergrond); het dialoog toont een live voorbeeld. Met de balk ingeschakeld draagt elke export (losse afbeelding, mapstapels, meerbladige TIFF, GIF, alles exporteren, batchverwerking) een publicatiestijl-balk: een afgeronde 1-2-5-lengte (ongeveer 12 % van de beeldbreedte) met omlijnde, leesbare tekst. De CLI ondersteunt dit via `--scale-bar` (automatische kalibratie) en `--scale-um <um/px>`.

Het voorbeeld tekent op de native pixelgeometrie van het doel voordat het in de viewport wordt ingepast. Inpassen, 100 %, pannen en framenavigatie wijzigen alleen de weergave. **Toepassen** legt één ongedaan-maakbare bewerking vast; **Annuleren** behoudt de vorige kalibratie. Historische uitvoer behoudt zijn eigen kalibratie en stijl wanneer een andere bronstapel wordt geladen.

Verwijder de balk via **Extra → Schaalbalk verwijderen**: *Uit de bronstapel* en *Uit het resultaat* beperken het bereik 'Toepassen op' tot het resterende canvas, terwijl *Overal* de balk uitschakelt. Beide canvassen en de A/B-veegvergelijking worden direct opnieuw getekend; het verwijderen uit een canvas dat de balk nooit toonde, laat de instelling ongewijzigd.

### Dieptekaart (focuspositie)

**Extra → Dieptekaart…** schat focusposities met de activiteitsmaat van de gekozen methode. Dit is geen gemeten oppervlak en geen exacte pixelherkomstkaart: piramide- en waveletmethoden schatten activiteit en de dieptegeschikte StackMFF-varianten melden een geschatte framepositie. Kies de huidige aangevinkte frames of de vastgelegde bron van de geselecteerde uitvoer. Ontbrekende historische herkomst wordt nooit vervangen door de huidige stapel; bron en textuur blijven aan die momentopname gebonden. Dieptekaarten zijn niet beschikbaar voor transparante PNG-stapels, zowel in de GUI als in de CLI.

Alle zeven dieptegeschikte methoden — DCT, DTCWT, GFG-FGF, OpenFocus B, OpenFocus C, StackMFF V2 en StackMFF V3 — vereisen een uitdrukkelijke klik op **Berekenen**. StackMFF V1 heeft geen dieptekaart. Het openen van het dialoog of het wijzigen van maat, bron, registratie of parameters start nooit een berekening. De parametereditor verschijnt direct onder de maatselectie: DCT-mediaan- en GFG-startkernel zijn oneven waarden van 1–51 (standaard 7). OpenFocus B gebruikt een focusstraal en diepteafvlakking; OpenFocus C gebruikt **mengingsafvlakking** zonder straal. De DTCWT-decompositieniveaus zijn hier als lokaal concept bewerkbaar en beginnen met de bronparameters. StackMFF V3 toont een eigen AI-uitvoermodus — **Zachte menging** (standaard) of **Harde keuze** — en een uitleg in plaats van een numerieke kernel: de kaart meldt een continue framepositie, geen fysieke afstand. Zachte menging gebruikt de waarschijnlijkheidsgewogen positie; harde keuze neemt per pixel het waarschijnlijkste frame. StackMFF V2 heeft geen parameters en meldt de focuslaag die het netwerk bemonstert.

Concepten beginnen met de parameters van de bijbehorende uitvoer of de onthouden methode-instellingen van het hoofdvenster. Ze worden tijdens dit dialoog per bron en methode onthouden; ze wijzigen het hoofdvenster, het project of de vastgelegde uitvoer niet.

Gewijzigde berekeningsinstellingen markeren de vorige 2D-kaart als **niet actueel**, wissen het 3D-oppervlak en blokkeren elke export — ook gekleurd, ruw, fysisch en OBJ — tot een nieuwe berekening slaagt. Een actieve berekening wordt coöperatief geannuleerd; druk na het annuleren opnieuw op **Berekenen**. Er wordt geen volgende berekening in de wachtrij gezet. **Berekenen** blijft zichtbaar en is uitgeschakeld tijdens berekenen, annuleren en exporteren.

- **Kleurenschaal**: grijstinten (publicatiestandaard) plus 15 wetenschappelijke paletten (Turbo, Viridis, Magma, …) en je eigen schema — *Nieuw eigen schema…* bouwt een gradiënt uit willekeurig veel kleurpunten (dubbelklik op een punt om positie en kleur te bewerken); eigen schema's worden opgeslagen en in de vervolgkeuzelijst aangeboden.
- **Omkeren / Gamma / Afvlakking**: de diepteas omdraaien, de toon opnieuw balanceren (gamma < 1 maakt lage indices lichter) en spikkels verwijderen met een randbewarende pass.
- **Overlay**: alleen beschikbaar voor de bijbehorende uitvoer met overeenkomende coördinaten.
- **Ruwe export**: een 16-bit TIFF slaat een genormaliseerde index op, gecodeerd van 0–65535; het zijn geen verliesvrije floatgegevens en geen fysieke afstand. JPEG/BMP worden geweigerd.
- **Index-TIFF en alternatieve afbeelding als TIFF**: exporteer de huidige diepteschatting als unsigned 16-bit frame-indices of als een afbeelding samengesteld uit bronpixels. Indexwaarden 1–N verwijzen naar de exacte framevolgorde die als invoer voor de diepteberekening is gebruikt, na registratie en eventuele herordening; 0 markeert ongeldige pixels. Maximaal 65.535 frames worden ondersteund. Gebroken geschatte posities worden op het dichtstbijzijnde frame afgerond, met halve waarden naar boven. Deze indices schatten de focuspositie; ze zijn niet de werkelijke pixelherkomst van een gefuseerd resultaat. De alternatieve afbeelding kiest op elke geldige locatie de pixel uit het overeenkomstige diepte-invoerframe en zet ongeldige pixels op nul. Als een invoer 16-bit is, worden 8-bitwaarden met 257 vermenigvuldigd naar 16-bit. Exporten gebruiken de ROI van de dieptekaart en behouden ongeldige gebieden als nul.
- **Fysische diepte**: voer Z-start, positieve stap, eenheid en opnamerichting in en bevestig de oorspronkelijke opnametoewijzing. Uitgesloten frames behouden hun gaten. Een float32-TIFF slaat micrometers op, NaN voor ontbrekende gegevens en bronmetadata. Een fysisch OBJ vereist ook XY-kalibratie; het gebruikt een fysische Z-schaal van 1:1.
  Bij wisselen van bron vervalt de bevestiging; bevestig de opnamevolgorde opnieuw.
- **3D-model / 3D-textuur**: kies de langste zijde van het raster (standaard 600 hoekpunten). Elk nieuw geopend dialoog begint met **Automatisch (echte kleuren)**: gebruik de bijbehorende uitvoer alleen als identiteit en coördinaten/ROI overeenkomen met de voltooide kaart; gebruik anders het eerste deelnemende frame uit die voltooide berekening, na registratie en eventuele frameherordening. De opgeloste uitvoer- of framenaam wordt getoond. Je kunt uitdrukkelijk de bijbehorende uitvoer, een deelnemend frame of **Dieptepseudokleuren** kiezen; niet-beschikbare afbeeldingen worden gemeld, nooit vervangen door ongerelateerde uitvoer. Deze textuur wordt gedeeld door het 3D-voorbeeld en de OBJ-textuur-PNG. De 2D-kaart behoudt zijn eigen kleurenschaal en overlay: het wijzigen van palet of gamma herkleurt een tekstuur met echte kleuren niet. Uitdrukkelijke pseudokleuren volgen palet, gamma en omkering. Het definitieve voorbeeld en de export gebruiken hetzelfde bemonsterde hoogteveld zonder verborgen extra afvlakking. Reliëf wordt in beeldeenheden uitgedrukt.
- **Modelexport**: OBJ, MTL, textuur-PNG en JSON-metadata vormen één bestandsgroep; houd ze bij elkaar. De export loopt op de achtergrond. Conflicten worden voor elk lid gecontroleerd, de voorbereiding wordt klaargezet, annuleren behoudt bestaande bestanden en fouten bij het vastleggen draaien vervangingen terug. Het opslaan van een model exporteert niet de bijbehorende fusieafbeelding en verwijdert de bescherming tegen niet-opgeslagen wijzigingen niet.

De berekening loopt op de achtergrond met voortgang en annulering; de AI-methoden hergebruiken dezelfde geheugenveilige tegelverwerking als AI-fusie, zodat diepe stapels werken. De CLI schrijft met `--depth-map [pad]` en `--depth-colormap` alleen de gekleurde kaart (standaard `<output>_depth.png`), alleen voor de hierboven genoemde dieptegeschikte methoden, en weigert transparante of gegroepeerde invoer; in batchmodus is alleen het standaardpad toegestaan. De GUI exporteert daarnaast ruwe data, frame-indices, alternatieve afbeeldingen, fysische diepte en 3D-modellen.

### Automatisch sessieherstel

OpenFocus houdt voortdurend een momentopname van je werksessie bij (stapel, instellingen, labels, huidig frame). Als de toepassing onverwacht sluit — door een crash, stroomuitval of geforceerd afsluiten — detecteert de volgende start dat en biedt **Sessie herstellen** aan om precies naar je laatste toestand terug te keren. Na een nette afsluiting verschijnt de vraag nooit. Herstelgegevens staan naast je instellingen (draagbare installaties bewaren ze naast het uitvoerbare bestand, zodat ze meereizen).

### Recente bestanden

Elke geopende stapel wordt onthouden onder **Bestand → Recent geopend** (maximaal 8 items, genummerd in het submenu). Klik op een item om het opnieuw te openen; er is geen globale sneltoets voor recente bestanden. Een opnieuw geopende stapel verhuist naar boven; items waarvan de bestanden niet meer bestaan, kunnen na een vraag worden verwijderd, en **Recente lijst wissen** maakt het menu leeg.

### Slepen en neerzetten

Je kunt ook een map rechtstreeks op het toepassingsvenster slepen om afbeeldingen te importeren. Extra bestanden op een bestaande stapel worden toegevoegd (na een groottecontrole).

### EXIF-oriëntatie

Camera-JPEG/TIFF-bestanden met een EXIF-oriëntatietag worden tijdens het importeren automatisch gedraaid of gespiegeld, zodat staande opnamen rechtop worden geladen.

### 16-bit-bronnen

16-bit PNG/TIFF-invoer wordt automatisch gedetecteerd en van begin tot eind behouden: fusie loopt op volledige bitdiepte en resultaten kunnen weer als 16-bit PNG/TIFF worden opgeslagen (JPG/BMP-exporten worden naar 8 bit omgezet, omdat die formaten geen 16 bit kunnen opslaan).

Camera-RAW-bestanden worden naar 16 bit ontwikkeld (camerawitbalans, sRGB-gamma) met automatische helderheid uitgeschakeld, zodat elk frame van een stapel dezelfde belichtingstoewijzing krijgt; automatische belichting per frame zou de helderheid tussen frames laten flikkeren. Sla op als TIFF of PNG om de 16-bitgegevens te behouden.

### Kleurprofielen

Export van losse afbeeldingen als PNG, TIFF, JPEG en WebP behoudt het RGB-ICC-profiel wanneer elk bronframe hetzelfde ingesloten profiel heeft. Gemengde of ongetagde stapels worden zonder ICC-profiel geëxporteerd. OpenFocus bewaart het profiel; het converteert de bronpixels niet naar een gemeenschappelijke kleurruimte en beheert het voorbeeld niet kleurmatig. Converteer afbeeldingen met verschillende profielen vóór het laden naar een gemeenschappelijke kleurruimte. RAW-invoer wordt naar sRGB ontwikkeld en erft geen cameraprofiel.

Voor een meerbladige TIFF moeten alle pagina's in het bestand hetzelfde geldige RGB-profiel hebben; anders worden de frames als ongetagd behandeld. Videoframes worden ook als ongetagd behandeld. Toegevoegde frames en de frames die voor een render worden geselecteerd, dragen bij aan het profiel, dat bij latere exports aan dat resultaat blijft gekoppeld.

### Stapel wissen

Om alle geladen afbeeldingen te verwijderen, ga je naar **Bestand → Stapel wissen** of druk je op `Ctrl+W`.

---

## 5. Afbeeldingen bekijken en navigeren

### Door de beeldstapel navigeren

- **Schuifregelaar**: sleep de schuif onder het bronpaneel om door de frames te bladeren
- **Bestandslijst**: klik op een bestandsnaam in de bronlijst van het rechterpaneel; zolang de lijst focus heeft, verplaatsen pijl omhoog/omlaag de selectie en volgt het bronpaneel
- **Toetsenbord**: er is geen globale sneltoets voor vorig/volgend frame; focus eerst de bronlijst

### Vergelijkbare frames bekijken

Klik onder de kop van de bronlijst op **Beelden controleren** om aangrenzende frames te vinden waarvan de pixels bijna identiek zijn. Tijdens de analyse verandert de knop in **Controle annuleren**. Elke voorgestelde rij noemt het referentieframe en het verschil; bekijk beide voordat je een frame via het selectievakje van de fusie uitsluit. Voorstellen wijzigen nooit selectievakjes of verwijderen bestanden. Deze conservatieve vergelijking kan vergelijkbare frames missen; wijzigingen die tot een klein scherp gebied beperkt blijven, worden niet weggemiddeld. Resultaten worden verworpen als de stapel verandert. Voorstellen verschijnen alleen als zowel het voorgestelde frame als zijn referentie zijn opgenomen; een wijziging van het selectievakje werkt de getoonde voorstellen bij.

### Zoomen en pannen

- **In-/uitzoomen**: muiswiel boven de afbeelding
- **Fijn zoomen**: houd `Ctrl` ingedrukt + muiswiel voor kleinere stappen
- **Aan venster aanpassen**: dubbelklik op de afbeelding om tussen inpassen en 100 % te wisselen
- **Pannen**: klik en sleep om de afbeelding binnen het paneel te verplaatsen

### Gesynchroniseerde navigatie

Wanneer je in het bronpaneel navigeert, toont het resultaatpaneel het overeenkomstige resultaatframe, indien beschikbaar.

---

## 6. Vergelijkingsweergave Vegen

De veegweergave legt twee afbeeldingen over elkaar met een versleepbare verticale scheiding, waardoor uitlijningsfouten en kwaliteitsverschillen direct zichtbaar worden langs de splitsing.

### De veegweergave openen

1. Render minstens één resultaat (fusie of registratie)
2. Druk op de schakelaar **Vegen** rechtsboven in de titelbalk van het resultaatpaneel
3. Het resultaatgebied schakelt naar de vergelijkingsweergave; druk opnieuw op **Vegen** om terug te keren

### De weergave lezen

- **Zijde A (links van de scheiding)**: een bronframe — volgt standaard de schuif van het bronpaneel
- **Zijde B (rechts van de scheiding)**: het meest recente render
- **Witte scheiding + ronde greep**: sleep horizontaal om de splitsing te verplaatsen
- Door randen over de scheiding te slepen, worden registratiefouten zichtbaar die in naast elkaar geplaatste panelen moeilijk te zien zijn

### Vergelijkingsobjecten kiezen

De balk onderaan de vergelijkingsweergave heeft één groep per zijde:

- **Zijde A** — vink **Bron volgen** aan (standaard) om de schuif van het bronpaneel te volgen, of vink uit om zijde A met een eigen schuif op een willekeurig frame vast te zetten
- **Zijde B** — vink **Nieuwste resultaat** aan (standaard) om altijd het nieuwste render te tonen, of vink uit om met de schuif door je uitvoergeschiedenis te bladeren

Beide zijden worden live bijgewerkt: nieuwe renders, bewegingen van de bronschuif en verwijderde frames verschijnen onmiddellijk.

### Zoomen en pannen

De vergelijkingsweergave gebruikt één transformatie voor beide afbeeldingen:

- **Zoom**: muiswiel (zoomt rond de cursor)
- **Pannen**: klik en sleep overal behalve op de scheiding
- **Resetten**: dubbelklik

Veegmodus en ROI-modus delen het resultaatpaneel, dus de een sluit de ander uit. De stapel wissen beëindigt ook de veegmodus.

---
## 7. Beeldregistratie

Beeldregistratie corrigeert ruimtelijke misuitlijning tussen frames in je beeldstapel. Dit is essentieel wanneer afbeeldingen kleine verschuivingen of perspectiefveranderingen vertonen.

### Registratiemethoden

#### ECC (Enhanced Correlation Coefficient)

- Gebruikt optimalisatie om de beste uitlijning op basis van correlatie te vinden
- Geschikt voor subtiele misuitlijning
- Levert subpixelprecisie
- Langzamer dan homografie, maar nauwkeuriger

#### Homografie

- Gebaseerd op het matchen van kenmerken (SIFT/ORB)
- Verwerkt grotere geometrische transformaties
- Sneller dan ECC voor de eerste uitlijning
- Geschikt bij duidelijke perspectiefveranderingen

#### Homografie → ECC

- Voert eerst een homografie en daarna een ECC-verfijning uit
- Voor grotere perspectiefveranderingen waarbij ECC alleen te instabiel zou zijn

#### OpenFocus (snel)

- Begrensde similariteitsuitlijning (translatie + rotatie + schaal), met helderheidsvereffening per frame en automatische bijsnijding
- Zoekt begrensde similariteitstransformaties met bemonsterde beeldverschillen en warpt met Lanczos-interpolatie
- Frames die meer dan **8 % van de frameafmeting verschuiven, blijven onuitgelijnd** in plaats van half gecorrigeerd. Aan te passen met de CLI-optie `--align-shift-limit`
- Kies voor niet-rigide of extreme geometrie in plaats daarvan ECC of homografie
- **Afgewezen beelden verwerken** staat standaard op „Doorgaan met niet-uitgelijnd beeld”. **Stoppen bij afgewezen beeld** laat de rendering of het batchitem mislukken voordat een resultaat wordt geproduceerd wanneer de detectorkwaliteit onder 0,06 ligt of de geschatte beweging de ingestelde verschuivingslimiet, 8° rotatie of 5 % schaalverandering overschrijdt. Frames worden vanaf 1 genummerd in de verwerkte stapelvolgorde. De geometrische referentie is standaard frame 1; een zwakke directe match kan opnieuw worden geprobeerd tegen het aangrenzende frame dat dichter bij de referentie ligt. Dit beleid detecteert afgewezen schattingen, niet alle onjuiste matches: weinig textuur, herhaalde patronen, occlusie, grote beweging of focusveranderingen kunnen registratie nog steeds misleiden. Controleer het resultaat.
- Kies in de CLI hetzelfde beleid met `--align openfocus --align-param 'failure_policy="fail"'` (gebruik shell-passende aanhalingstekens). Batchitems gebruiken het gekozen beleid onafhankelijk. De standaard is `"continue"`.
- Stel **Referentiebeeld** in op een 1-gebaseerd nummer in de opgenomen, verwerkte stapelvolgorde om de geometrie op dat frame te verankeren. De framevolgorde blijft behouden; helderheidsvereffening gebruikt zijn eigen referentie. Elke batchstapel moet dat nummer bevatten, anders mislukt het item. CLI-voorbeeld: `--align openfocus --align-param reference_frame=3`. Een referentie kan parallax of een slechte match niet corrigeren; controleer het resultaat.

Uitlijningsrapporten noemen de referentie en de uitkomst per frame. Een terugval betekent dat de methode de uitlijning niet kon vaststellen of afhankelijk is van een eerdere mislukte schatting; de gemelde reden onderscheidt zwakke schattingen en beweging buiten de limieten wanneer bekend. Kwaliteitswaarden zijn methodespecifiek en geen garantie voor nauwkeurigheid. Transformaties mappen broncoördinaten naar het referentiecanvas van de registratiefase vóór de gemeenschappelijke bijsnijding; samengestelde methoden melden hun fasen afzonderlijk. Homografie en ECC accepteren ook de optionele registratieparameter `failure_policy="fail"` om te stoppen wanneer hun algoritme een fout detecteert. De standaard blijft `"continue"`; controleer resultaten ook wanneer de uitlijning slaagt.

Om een rendering te onderzoeken, schakel je **Instellingen → Geavanceerde instellingen… → Registratie → Uitlijningsdiagnose verzamelen** in en rendert opnieuw. Dit berekent de uitlijning opnieuw in plaats van gecachte afbeeldingen te hergebruiken. De voltooiingsstatus markeert terugvallen; **Instellingen → Geavanceerde instellingen… → Registratie → Uitlijningsdiagnose** opent het rapport per frame. Het toont beschikbare foutredenen, methodespecifieke scores, matchaantallen, gerefereerde frames en transformaties. Referentienummers en -namen verwijzen naar de oorspronkelijke bronlijst, ook wanneer frames zijn uitgesloten. Batch- en gegroepeerde renders leveren dit rapport niet.

### Registratie inschakelen

1. Zoek in het rechterpaneel de groep **Registratie**
2. Kies een methode in de vervolgkeuzelijst: **Geen uitlijning**, **OpenFocus (snel)**, **ECC**, **Homografie** of **Homografie → ECC**. Zonder opgeslagen keuze start een nieuwe sessie met ECC; de laatst gekozen methode wordt bewaard.
3. Kies **Afgewezen beelden verwerken** onder **Instellingen → Geavanceerde instellingen… → Registratie**. Wijzigingen gelden voor de methoden; bij OpenFocus (snel) kun je ook het referentieframe instellen.
4. **Uitlijningsdiagnose verzamelen** staat nu onder **Instellingen → Geavanceerde instellingen… → Registratie** (zie hierboven).

### Wanneer registratie zinvol is

- De afbeeldingen zijn zichtbaar misuitgelijnd
- De opnamen zijn uit de hand gemaakt
- De scène heeft parallax (dichtere objecten verschuiven tussen frames)
- Resultaten vertonen spookbeelden of dubbele contouren

### Registratie-instellingen

Meer instellingen via **Instellingen → Geavanceerde instellingen… → Registratie**:

- **Verkleiningsbreedte**: resolutie van de voorverwerking (standaard: 1024 px)
- Kleinere waarden = snellere verwerking, mogelijk lagere nauwkeurigheid
- Grotere waarden = langzamere verwerking, mogelijk hogere nauwkeurigheid
- **Uitlijningsdiagnose verzamelen** en **Registratiecache** staan in dezelfde sectie

---

## 8. Fusiemethoden

OpenFocus biedt tien fusiealgorithmen plus een modus met alleen uitlijning. Een fris profiel start met **StackMFF V3** en **ECC**; de volgende secties beschrijven elke optie. Elke methode heeft eigen eigenschappen voor verschillende beeldtypen.

### Guided Filter

- **Algoritme**: randbewarende fusie via Guided Filtering
- **Het beste voor**: algemeen gebruik, gebalanceerde snelheid en kwaliteit
- **Voordelen**: goede randbehoud, snelle verwerking, uitsluitend CPU
- **Parameter**: kernelgrootte (instelbaar van 1–51, alleen oneven waarden)
  - Kleinere kernel: meer lokaal detail, mogelijk meer ruis
  - Grotere kernel: gladder resultaat, betere ruisonderdrukking
- **Opmerking**: Guided Filter is de klassieke standaardmethode, maar niet de standaard. Zonder opgeslagen keuze start een nieuwe sessie met StackMFF V3.

### DCT (Discrete Cosinustransformatie)

- **Algoritme**: fusie in het frequentiedomein via DCT-coëfficiënten
- **Het beste voor**: textuurrijke afbeeldingen, wetenschappelijke beeldvorming
- **Voordelen**: behoudt fijne texturen, wiskundig onderbouwd
- **Parameter**: de kernelgrootte bepaalt het verwerkingsvenster

### DTCWT (Dual-Tree Complex Wavelet Transform)

- **Algoritme**: fusie in het meerschalige waveletdomein
- **Het beste voor**: complexe scènes met meerdere detailniveaus
- **Voordelen**: uitstekende meerschalige analyse, goede richtingsgevoeligheid
- **Fusieregel**: één beslissing per pixel en niveau voor alle oriëntaties en kleurkanalen (N-voudig, met meerderheidsconsistentiecontrole); de laagdoorlaatband volgt de grofste beslissing in plaats van elk frame te middelen
- **Parameter**: waveletniveaus (1–6, standaard 4) in het rechterpaneel. De methode heeft geen ruimtelijke kernel, dus de kernelregelaar is verborgen.

### GFG-FGF (Gaussian-like Four-neighbourhood Gradient + Fast Guided Filter)

- **Algoritme**: implementatie van Fu et al. (*Laser & Optoelectronics Progress*, 2024): een onscherpteverschil-focuskaart |I − mean(I)| wordt verrijkt met de Gaussian-like vierbuurgradiënt (GFG), verfijnd door een fast guided filter, per pixel beslist (gelijke standen opgelost met een 3×3 Sobel-kleinmaat), en de binaire beslissingskaarten worden vóór de gewogen menging opnieuw met de fast guided filter afgevlakt.
- **Het beste voor**: microscopie en afbeeldingen met duidelijke focusgebieden
- **Voordelen**: snel, scherpe grenzen van focusgebieden; het geheugengebruik groeit niet met het aantal frames
- **Parameter**: kernel = grootte van het gemiddelde filter van de grove focuskaart

### StackMFF (V1, deep learning)

- **Algoritme**: een aandacht netwerk op één schaal mengt de luma van de bronframes met softmax-gewichten; Cb/Cr komen uit het bronframe waarvan de luma het dichtst bij de gefuseerde luma ligt
- **Het beste voor**: neurale fusie van een volledige focusstapel
- **Voordelen**: geen parameters, kleuren blijven binnen het gamut; de gefuseerde luma is onafhankelijk van de framevolgorde, maar chroma kan verschillen bij gelijke luma-afstanden
- **Opmerking**: ongeveer 156.172 parameters; grotere stapels profiteren van GPU-versnelling

### StackMFF V2 (deep learning)

- **Algoritme**: een aandacht netwerk op één schaal normaliseert de verwachte focusediepte en bemonstert het dichtstbijzijnde bronframe. Diepte-indices hangen af van de stapelvolgorde. **De frames moeten in hun oorspronkelijke focusvolgorde blijven** — omkeren is voor deze methode uitgeschakeld
- **Dieptekaarten**: ondersteund; meldt de bemonsterde focuslaag
- **Het beste voor**: continue focusreeksen waarbij een zeer klein, snel model belangrijk is
- **Vereisten**: ONNX Runtime en het meegeleverde ONNX-model.

### StackMFF V3 (deep learning)

- **Algoritme**: pixelgewijze bronclassificatie (ONNX Runtime). Het netwerk voorspelt voor elke pixel uit welk invoerframe die komt en mengt de frames vervolgens op basis van die waarschijnlijkheden of kiest één frame
- **AI-uitvoer** (rechterpaneel, alleen voor deze methode):
  - **Zachte menging (gewogen)** — standaard; elke pixel is een waarschijnlijkheidsgewogen mix van de frames. Vloeiende overgangen, geen harde naden; komt overeen met de in het paper gerapporteerde configuratie
  - **Harde keuze (één frame)** — elke pixel wordt gekopieerd uit het waarschijnlijkste frame. Scherpere selectie zonder interpolatie tussen frames; waar twee frames gelijk staan, kan een naad op pixelniveau verschijnen
- **Het beste voor**: hoogste kwaliteit wanneer een GPU beschikbaar is
- **Voordelen**: volgordeonafhankelijk (de framevolgorde verandert het resultaat nooit), stapellengtes binnen de beschikbare middelen, geen methode-specifieke afstelling, automatische optimalisatie
- **Vereisten**: ONNX Runtime en het meegeleverde ONNX-model.

Elke neurale tegel gebruikt alle frames. Bronframes en tijdelijke uitlijngegevens kunnen op schijf staan; vrije schijfruimte en beschikbaar geheugen begrenzen de stapelgrootte. Ruimtelijke V2-tegels benaderen inferentie op de volledige afbeelding.

- **Standaard**: StackMFF V3.

### OpenFocus A / B / C

- **OpenFocus A**: vlakt zijn meerschalige focusresponskaart af en selecteert vervolgens scherpe pixels.
- **OpenFocus B**: mengt naburige frames op basis van een afgeplatte dieptekaart.
- **OpenFocus C**: reconstrueert meerschalige banden op basis van een afgeplatte dieptekaart.
- **Parameters**: A en B gebruiken de kernelregelaar voor de schaal van de focusrespons; C heeft geen straal. Alle drie hebben een afvlakkingsparameter voor de overgangssterkte.

**Afvlakking**: één draaischijf onder de kernelregelaar, met een betekenis die afhangt van de gekozen methode (bereik 0–8, stap 0,2):

| Methode | Label | Standaard | Effect |
|---|---|---|---|
| OpenFocus A | Mengingsafvlakking | 2,0 | Regelt de afvlakking van de focusresponskaart voordat pixels worden geselecteerd |
| OpenFocus B | Diepteafvlakking | 1,2 | Regelt de afvlakking van de dieptekaart die de menging stuurt |
| OpenFocus C | Mengingsafvlakking | 3,0 | Regelt de afvlakking van de dieptekaart die de meerschalige reconstructie stuurt |

Hogere waarden versterken de overgangsafvlakking. Elke methode onthoudt zijn eigen waarde tussen sessies, en batchverwerking gebruikt dezelfde regel.

C heeft geen straalparameter, dus de kernelregelaar is verborgen.

CLI: `--depth-smoothing VALUE` (B) en `--blend-smoothing VALUE` (A, C).

**Helderheidsopties gelden per methode.** *Belichting afstemmen vóór fusie* en *Helderheidsdrift verwijderen* worden per fusiemethode afzonderlijk bewaard, omdat ze een vraag per stapel beantwoorden en geen globale voorkeur zijn. Beide staan standaard aan voor OpenFocus A/B/C (die geen ingebouwde belichtingskalibratie hebben); de andere methoden behouden de historische standaarden (belichtingsafstemming aan, driftverwijdering uit). Bij het wisselen van methode worden de waarden van die methode getoond, en ze overleven een herstart. **Herstellen** wist de onthouden helderheids-, afvlakkings- en straalwaarden van elke methode.

**Gefuseerd resultaat verscherpen** past na de fusie een milde onscherpe maskering toe, voordat het resultaat naar zijn uitvoerbitdiepte wordt omgezet. Dit staat standaard uit en kan in de hoofdfusie-opties of batchopties worden ingeschakeld; de equivalente CLI-optie is `--sharpen-output`. Dit beïnvloedt het gefuseerde beeld, niet geregistreerde frames of dieptekaarten. Het kan randen duidelijker maken, maar ook ruis en halo's versterken. De instelling wordt in het project opgeslagen en tussen sessies onthouden.

### Alleen uitlijning (geen fusie)

Kies **Alleen uitlijning (geen fusie)** om de stapel te registreren en één stapelitem aan de uitvoerlijst toe te voegen. De frames blijven in het geheugen tot je ze exporteert; alleen uitlijnen maakt geen extra cache aan en schrijft niets naast de bronbestanden. Sla het item op om één map in je gekozen bestemming te maken. De oorspronkelijke bestandsnamen blijven behouden, met de extensie van het gekozen formaat.

### Een fusiemethode kiezen

1. Zoek in het rechterpaneel de groep **Fusie**
2. Kies een methode in de **vervolgkeuzelijst** (er is altijd precies één actief; het eerste item registreert zonder fusie)
3. Pas indien nodig de kernelgrootte aan — het label toont wat de kernel voor de gekozen methode betekent
4. Klik op **▶ Renderen** om te verwerken

### Algorithmevergelijking

| Algoritme | Snelheid | Kwaliteit | Het beste voor |
|-----------|-------|---------|----------|
| Guided Filter | Snel | Goed | Algemeen gebruik |
| DCT | Gemiddeld | Goed | Texturen |
| DTCWT | Gemiddeld | Zeer goed | Complexe scènes |
| GFG-FGF | Snel | Goed | Focusgebieden |
| StackMFF (V1) | Langzaam (GPU) | Hier niet beoordeeld | Luma onafhankelijk van volgorde; chroma kan verschillen bij gelijke luma-afstanden |
| StackMFF V2 | Snel (GPU) | Hier niet beoordeeld | Continue focusreeksen met een klein model |
| StackMFF V3 | Langzaam (GPU) | Uitstekend | Beste kwaliteit |
| OpenFocus A | Afhankelijk van stapel | Afhankelijk van stapel | Meerschalige focusselectie |
| OpenFocus B | Afhankelijk van stapel | Afhankelijk van stapel | Dieptegestuurde menging van naburige frames |
| OpenFocus C | Afhankelijk van stapel | Afhankelijk van stapel | Dieptegestuurde meerschalige reconstructie |

**Extra → Methoden vergelijken…** opent een dialoog waarin meerdere methoden onafhankelijk worden ingesteld. Naam, parameterlabels en bedieningselementen van elke methode blijven bij elkaar wanneer het venster smaller wordt. Wijzigingen werken het voorbeeld automatisch bij. **ROI-voorbeeld (snel)** fuseert het geselecteerde gebied met omringende context; **Fusie van het hele beeld** controleert het resultaat van het hele beeld en laat je een ander gebied bekijken zonder opnieuw te renderen. ROI-resultaten kunnen afwijken van resultaten van het hele beeld, omdat belichtingsstatistieken en sommige algoritmen van de beeldomvang afhangen.
**Fusie van het hele beeld** is standaard geselecteerd.

Een zichtbare aanwijzing legt de resultaatkeuze uit: klik op een resultaat of geef het met `Tab` de focus en selecteer het met `Enter` of de spatiebalk.

Kies boven de rechterpanelen met **Voorbeeldgrootte (px)** een vierkante grootte van 160 tot 960 pixels, door de schuif te slepen of een getal in te voeren. Alle methoden gebruiken dezelfde grootte en wikkelen passend bij de beschikbare breedte. De grootte wordt onthouden voor de volgende opening. Een wijziging behoudt de huidige vergroting en past het inspectievierkant rond zijn midden aan, begrensd door de beeldranden.

Druk op **F11** of de knop **Volledig scherm (F11)** om het hele scherm te gebruiken; de knoptekst wisselt naar **Volledig scherm verlaten (F11)**. De titelbalk kan het venster ook maximaliseren. **Esc** verlaat eerst het volledig scherm en sluit het dialoog pas bij een tweede druk. Terugkeren uit volledig scherm herstelt de vorige venstergrootte, en het resultatenraster herschikt zich in beide gevallen naar de beschikbare breedte.

Sleep de bronafbeelding met de linkermuisknop om het witte inspectievierkant te verplaatsen; bij loslaten blijft het staan. De rechterpanelen tonen precies dat gebied uit dezelfde uitgelijnde stapel. Gebruik het muiswiel om de vergroting van het gebied te wijzigen, rechterslepen om de bron te pannen, **Ctrl+wiel** om de bron te zoomen en **F** om in te passen. Wisselen van bronframe of formaatwijziging van het venster behoudt de selectie. **Annuleren** stopt verder werk; sluiten wacht tot de huidige native bewerking veilig terugkeert.

### Snel voorbeeld (conceptrender)

Parameters afstellen op grote stapels is traag op volledige resolutie. Open **Renderinstellingen** en vink **Snel voorbeeld** aan: OpenFocus rendert dan eerst een verkleind concept (de helft van de lineaire afmeting, minimaal 320 px aan de langste zijde) — meestal meerdere keren sneller, inclusief registratie. De voltooiingsmelding (en de tooltip van het nieuwste resultaat) meldt **Kwaliteit: voorbeeld**; het concept wordt **niet** aan de uitvoergeschiedenis toegevoegd. Vink het vakje uit en render opnieuw voor het resultaat op volledige resolutie. De voorbeeldmodus geldt niet voor ROI-renders en vervuilt nooit de uitlijningscache op volledige resolutie.

---

## 9. Beeldtransformaties

### Draaien

1. Ga naar **Bewerken → Draaien**
2. Kies het draaitype:
   - **90° rechtsom**: alle afbeeldingen 90 graden naar rechts draaien
   - **90° linksom**: alle afbeeldingen 90 graden naar links draaien
   - **180°**: afbeeldingen ondersteboven draaien

### Spiegelen

1. Ga naar **Bewerken → Spiegelen**
2. Kies het spiegelingstype:
   - **Horizontaal spiegelen**: van links naar rechts spiegelen
   - **Verticaal spiegelen**: van boven naar beneden spiegelen

### Formaat wijzigen

1. Ga naar **Bewerken → Formaat wijzigen**
2. Er verschijnt een dialoog met formaatopties
3. Voer nieuwe afmetingen of een schaalpercentage in
4. Alle afbeeldingen in de stapel worden proportioneel geschaald

### Volgorde van transformaties

Transformaties worden op de hele beeldstapel toegepast en behouden de uitlijning tussen frames.

---

## 10. Retoucheren, labels en annotaties

### Het resultaat retoucheren (Bewerken → Resultaat retoucheren…, `Ctrl+T`)

Fusie kan nog steeds een lokaal defect achterlaten — een halo rond een contrastrijke rand, een haartje uit het verkeerde frame, ruis in een vlak gebied. De werkruimte kopieert pixels uit precies de frames die het geselecteerde resultaat hebben geproduceerd:

1. Selecteer één uitvoer in de uitvoerlijst (rechtsklik → *Resultaat retoucheren…* werkt ook) en druk op `Ctrl+T`. Zonder selectie toont een kiesdialoog elke uitvoer en markeert welke vastgelegde bronframes hebben.
2. Het bronframe staat links, het resultaat rechts. Beide delen zoom en positie: wiel zoomt, rechts-/middelslepen (of Space+slepen) versleept, `F` past in, `1` toont 100 %.
3. Kies een miniatuur, sleep de frameslider of druk op `,` / `.`. Een streek is vergrendeld aan dat frame en de huidige kwastinstellingen; het frame kan niet wisselen voordat de streek eindigt.
4. Kies **Bronkwast** (`B`) of **Herstelkwast** (`E`). Diameter, hardheid en dekking hebben zowel schuiven als numerieke velden; `[` / `]` past de diameter aan. Met `Ctrl` + muiswiel omhoog vergroot je de kwastdiameter en omlaag verklein je die; het muiswiel zonder `Ctrl` blijft zoomen. De herstelkwast zet pixels terug naar de afbeelding die je hebt geopend, niet naar een oudere uitvoer.
5. Houd `Tab` ingedrukt om het resultaat vóór deze sessie te zien en `Alt` om een glimp van het bronframe op te vangen. De scherptereferentie is optioneel, dekt alleen het broncanvas en gebruikt één gedeelde schaal voor elk frame.
6. `Ctrl+Z` / `Ctrl+Y` maken hele streken ongedaan en herhalen ze. **Alles herstellen…** is één ongedaan-maakbare stap. **Toepassen als nieuwe uitvoer** voegt een item met de naam `<naam> · Geretoucheerd N` toe en selecteert het; het origineel blijft onaangetast. Het nieuwe item is gemarkeerd als *Niet geëxporteerd* tot je het vanuit het hoofdvenster exporteert, en OpenFocus beschermt het tegen automatische verdringing uit de geschiedenis en tegen afsluiten zonder beslissing.

Pas de retouche toe als nieuwe uitvoer, selecteer deze en open **Opslaan als**. Kies TIFF met lagen in de indelinglijst om het resultaat met zijn bronframes op te slaan. Live labels en schaalbalken worden niet meegenomen; niet-opgeslagen annotatiewijzigingen blijven beschermd. Brongeometrie en geldigheidsmaskers blijven behouden. Het resultaat en de bronnen moeten ondoorzichtige RGB- of grijstintenafbeeldingen met overeenkomende 8- of 16-bitdiepte zijn, en elke bronrechthoek moet binnen het resultaat passen. De export ondersteunt maximaal 128 bronnen; bronnamen zijn beperkt tot 255 tekens en mogen geen NUL of tekens buiten de Basic Multilingual Plane bevatten, zoals veel emoji. Onveilige contexten en taken die naar schatting meer dan 512 MiB werkgeheugen of het beschikbare RAM-budget overschrijden, worden geweigerd en de fout wordt gemeld in plaats van een onvolledige export te schrijven. GIMP 3.2.6 op Windows importeert de 8- en 16-bitlagen, inclusief hun offsets en transparante registratievulling. Het resultaat is zichtbaar en bronlagen zijn verborgen; GIMP voegt ook een zichtbare achtergrondlaag toe. Andere editors zijn niet getest.

Voor een lokale toonaanpassing kies je **Belichtingspenseel**, stel je het bedrag in honderdsten van een stop in (100 verdubbelt de pixelwaarden, −100 halveert ze) en schilder je op het resultaat. Het voorbeeld wordt tijdens het schilderen bijgewerkt. Elke streek past het huidige resultaat één keer aan, ook waar de stippen overlappen; een volgende streek bouwt daarop voort. Penseeldekking en hardheid regelen de menging. Hooglichten kunnen clipping vertonen bij het ophogen. Houd **Voor** ingedrukt om te vergelijken met het resultaat zoals geopend, druk tijdens een streek op Escape om die te annuleren, of gebruik ongedaan maken en opnieuw uitvoeren na de streek. Dit zijn pixelbewerkingen, geen aanpasbare lagen. Pas toe als nieuwe uitvoer en exporteer het nieuwe item om de bewerkte pixels op te slaan; een project bewaart die pixels niet.

**Resultaat aanpassen…** wijzigt het hele huidige retoucheringsresultaat. Het bemonsterde voorbeeld wordt bijgewerkt terwijl je belichting en contrast wijzigt; vink **Voor** aan om te vergelijken met het resultaat vóór het openen van de aanpassing. Beide regelaars gebruiken honderdsten van een stop: belichting vermenigvuldigt de pixelwaarden, daarna schaalt contrast hun afstand tot middengrijs. Extreme waarden kunnen hooglichten of schaduwen laten clippen. **OK** past de aanpassing toe als één ongedaan-maakbare bewerking; **Annuleren** laat het resultaat ongewijzigd. Tijdens de verwerking vraagt **Annuleren** om terugdraaien. De volledige afbeelding moet binnen de geheugenlimiet van de retouchegeschiedenis passen voor deze ongedaan-maakbare bewerking. Publiceer en exporteer het resultaat zoals hierboven beschreven.

Herkomst wordt met de uitvoer opgeslagen: de uitgelijnde frames, ROI-bijsnijdings-/plakoffsets, DCT-randtrimming en bitdiepte worden vastgelegd wanneer het resultaat wordt gerenderd. Batchbestanden, oudere resultaten en items uit alleen-uitlijnen hebben geen vastgelegde bron en worden daarom als niet beschikbaar vermeld in plaats van uit de huidige stapel te worden geraden. Niet-toegepaste streken en de ongedaan-maakgeschiedenis gelden alleen voor de sessie; een project opslaan bewaart ze niet.

Voor een bestaand afbeeldingsbestand kies je **Bewerken → Afbeeldingsbestanden retoucheren…** en selecteer je het resultaat en een of meer bronafbeeldingen. De bronnen moeten al op hetzelfde pixelraster als het resultaat zijn uitgelijnd; gelijke afmetingen alleen bewijzen geen uitlijning. Enkelbladige PNG-, TIFF-, JPEG- en BMP-afbeeldingen met overeenkomende rechtopstaande afmetingen, kanalen en 8- of 16-bitdiepte worden geaccepteerd; EXIF-oriëntatie wordt bij het laden toegepast. Alpha- en meerbladige afbeeldingen worden niet ondersteund. De geselecteerde frames zijn door de gebruiker aangeleverd, geen vastgelegde renderherkomst. **Toepassen** maakt een aparte, niet-geëxporteerde uitvoer; sla die op vanuit de uitvoerlijst. De gekozen bestanden zijn tijdens deze sessie beschermd tegen overschrijven, inclusief export van dieptekaarten en 3D-meshes. Zoals bij andere resultaten bewaren projectbestanden deze pixels of penseelgeschiedenis niet.

### De achtergrond vervangen (Automatische achtergrond)

Bij stapelen wordt de achtergrond samen met het onderwerp scherp gemaakt. De
achtergrond van een gestapeld beeld is dan vaak een lappendeken van
verschillende onscherptes, met een halo rond het onderwerp.
**Automatische achtergrond…** in de retoucheerwerkruimte vervangt die door één
echt frame uit dezelfde stapel. Het onderwerp blijft precies zoals gestapeld.
De achtergrond wordt afgeleid van een frame en het gestapelde resultaat; de
verwerking kan de pixels reconstrueren, aanpassen en mengen.

1. Open het resultaat in de retoucheerwerkruimte en klik op
   **Automatische achtergrond…**.
2. Kies de achtergrond:
   - **Automatisch** (aanbevolen) gebruikt de zachte achtergrond, tenzij de
     stapel een groot deel van de achtergrond scherp heeft vastgelegd. Dan
     gebruikt het de scherpe.
   - **Zachte achtergrond** gebruikt het frame met de onscherpste achtergrond.
   - **Scherpe achtergrond** gebruikt het frame met de verste focus.
3. Laat **Randhalo's opschonen** aan om halo's langs de randen van het onderwerp
   weg te halen. Het gaat nooit verder dan wat de frames vastlegden. Zet het uit
   om de randen van de stapel te behouden.
4. Vink **Zachte delen die aan het onderwerp vastzitten behouden** aan wanneer
   delen van het onderwerp net buiten de scherptediepte vallen, zoals het
   knopje aan het eind van een stekel. Ook achtergrondobjecten naast het
   onderwerp kunnen dan zoals gestapeld blijven, dus controleer het resultaat.
5. Klik op **Achtergrond vervangen**. Een stapel van 15 frames van 20 megapixel
   duurt ongeveer een minuut en vraagt ongeveer 3,5 GB vrij geheugen.
   **Annuleren** stopt het.

De wijziging is één stap die u ongedaan kunt maken, en penseelstreken die u al
op het onderwerp hebt gezet blijven staan. Met de **Herstelkwast** haalt u
gestapelde pixels terug waar u die liever hebt, bijvoorbeeld een onscherp
grassprietje vóór het onderwerp. Vult het onderwerp bijna het hele beeld, zoals
bij een bosbodemscène, dan blijft het beeld zoals gestapeld en ziet u een
melding.

Als je Retoucheren opent op een resultaat dat al geretoucheerd is, analyseert
Automatische achtergrond het gestapelde resultaat waar het van komt en legt de
nieuwe achtergrond op je geretoucheerde afbeelding. Penseelwerk op het onderwerp
blijft behouden en een eerder vervangen achtergrond wordt nooit opnieuw
geanalyseerd, dus een tweede keer draaien bouwt geen halo's op.

Na een run kleurt **Ontbrekende gegevens tonen** de plekken die geen frame scherp
zag. Dit zijn de plekken om te controleren of met de hand te retoucheren:

- magenta: achtergrond die geen frame zonder het onderwerp ervoor vastlegde
- cyaan: onderwerp dat voor de camera verborgen was
- geel: onderwerp buiten het focusbereik van de stapel
- leisteen: te glad om te beoordelen

**Kaart met ontbrekende gegevens opslaan…** schrijft de gekleurde voorvertoning
en per kleur één zwart-witmasker. De automatische achtergrond heeft de frames
nodig waaruit het resultaat ontstond en is niet beschikbaar voor
gebiedsresultaten (ROI). Nieuwe achtergrondcontrolepunten herstellen het exacte
beeld, de bereikbare geschiedenis en diagnostiek zonder de achtergrond opnieuw te berekenen.

### Labels beheren

Open **Extra → Labels** nadat je afbeeldingen hebt geladen of een resultaat hebt geproduceerd. Elk label heeft zijn eigen doel: de bronstapel, de uitgelijnde stapel of uitvoerresultaten. Uitvoerlabels kunnen voor alle resultaten gelden of voor het geselecteerde resultaat op naam; hernoemen of herordenen van dat resultaat verandert de koppeling niet.

Gebruik **Toevoegen** om een label te maken, selecteer een item om het te bewerken en gebruik het selectievakje om het in of uit te schakelen. Voer vaste tekst in of voeg `{value}` in voor een framenummer. Het nummer is de startwaarde plus de nulgebaseerde frame-index maal het interval; een enkele uitvoer gebruikt altijd de startwaarde.

Kies alle frames, het huidige frame of een aangepast bereik zoals `3-5,8`. Ongeldige of niet-beschikbare framenummers verhinderen het toepassen van het concept. Huidige en aangepaste bereiken volgen de geselecteerde bronidentiteiten wanneer de stapel wordt herordend; labels voor alle frames gelden ook voor toegevoegde frames. Ontbrekende doelen blijven aan hun oorspronkelijke identiteiten gekoppeld tot je uitdrukkelijk een nieuw doel kiest.

Het voorbeeld gebruikt het werkelijke doel. Navigeer met **Vorige/Volgende** of voer het framenummer in, kies **Passend maken** of **100 %** en sleep de afbeelding om te pannen bij 100 %. Positie en lettergrootte gebruiken beeldpixels. Een fallback-lettertype of een mislukte tekening wordt gemeld. Labels behouden de oorspronkelijke bitdiepte bij export als PNG of TIFF.

**Toepassen en sluiten** legt het volledige concept vast als één ongedaan-maakbare wijziging. Annuleren, Esc en sluiten verwerpen het. Het selecteren van een ander label behoudt je conceptwijzigingen. **Verwijderen** verwijdert alleen het geselecteerde item, inclusief het laatste. De verwijderacties in het menu wissen bronlabels, labels van de uitgelijnde stapel, uitvoerlabels of alle labels onafhankelijk van elkaar en kunnen ook ongedaan worden gemaakt. Toegepaste labels wijzigen de sessieweergave; exporteer een afbeelding om de annotaties naar een bestand te schrijven.

---

## 11. Instellingen en configuratie

**Instellingen → Geavanceerde instellingen…** opent prestatie-, registratie-, sensorreinigings-, uitvoerbenamings-, AI- en opstartinstellingen. Wijzigingen blijven concepten tot je op **OK** drukt; Annuleren, Esc en sluiten verwerpen ze. Een kleinere tegelgrootte werkt de zichtbare overlaplimiet onmiddellijk bij naar een halve tegel. De opgeslagen AI-batchlimiet blijft behouden; de inferentie gebruikt momenteel één tegel per aanroep. De registratiebreedte kan de nauwkeurigheid van de uitlijning veranderen. Het verzamelen van de uitlijningsdiagnose is een selectievakje in de sectie Registratie.

Automatische update-downloads lopen tijdens inactiviteit en installeren nooit vanzelf. Een voltooide download blijft klaar tot je uitdrukkelijk **Bijwerken en herstarten** kiest. De herstart beschermt actieve bewerkingen en niet-opgeslagen werk en schrijft daarna herstelgegevens voordat de vervangende versie start. Annuleren van een download of voorbereiding wacht op de huidige veilige grens. Een bestaande volledige, nieuwere klaargezette build kan offline worden geïnstalleerd.

Flat-field-correctie wijzigt de werkpixels van de sessie zonder bronbestanden te overschrijven. Kies via **Extra → Stofkaart (flat-field)** het witbeeld en een voorbeeldframe en genereer het voorbeeld opnieuw na een wijziging van bron of schaal. Referentie- en werk-/basisafbeeldingen moeten hetzelfde coördinatendomein en dezelfde grootte hebben. Toepassen bereidt beide stapels voor voordat wordt vastgelegd; het optionele verwijderen van het referentieframe hoort bij dezelfde ongedaan-maakstap. Exporteer gecorrigeerde bronafbeeldingen wanneer je deze pixels buiten de sessie wilt bewaren.

Batchverwerking accepteert een optioneel witveldbeeld in de verwerkingsopties. Kies een onscherpe referentie op de oorspronkelijke frameresolutie; die wordt met de invoerstapel geschaald en vóór registratie toegepast met de standaard stofgrootte. Een onleesbare referentie of een grootteverschil laat de betrokken stapel zichtbaar mislukken. De referentie wordt uitgesloten van mapinvoer, ook bij gegroepeerde verwerking; er moeten minstens twee bronframes overblijven voor fusie. Flat-field-correctie kan niet worden gecombineerd met transparante invoer. Laat het veld leeg om zonder correctie te verwerken. De CLI accepteert de referentie met `--dust-map WHITE`.

Sensorreiniging is optioneel en loopt op de oorspronkelijke frames in sensorcoördinaten vóór registratie. In de desktopapp vind je dit onder **Instellingen → Geavanceerde instellingen… → Sensor opschonen**; het batchdialoog biedt dezelfde bediening. Een uitdrukkelijk defectmasker corrigeert bekende hot- of dode pixels; OpenFocus leidt geen masker af uit specimenafbeeldingen. Het masker moet een enkelkanaals 8- of 16-bitafbeelding met dezelfde afmetingen als de bron zijn: nul betekent onaangetast en ofwel 1 of de maximale kanaalwaarde markeert defecten. Controleer de omvang vóór verwerking. Maskers die meer dan 10 % van het frame bedekken, of defecten zonder ongemaskeerde buur, worden geweigerd. Houd het masker op de oorspronkelijke resolutie en selecteer het niet als bronafbeelding. Laad bronframes voor GUI-verwerking op volledige schaal wanneer je een defectmasker gebruikt; het masker wordt nooit geschaald. Sensorreiniging (inclusief ruisonderdrukking) kan niet op een GUI-stapel worden toegepast na een draaiing, spiegeling, flat-field-correctie of andere bronwijziging: laad de oorspronkelijke frames opnieuw of schakel reiniging uit. Dit geldt ook voor herstelde projecten met zulke brongeschiedenissen. Vooraf uitgelijnde ROI-verwerking kan niet met sensorreiniging worden gecombineerd; verlaat de ROI-modus en render de hele stapel. Reinigingsinstellingen horen bij de huidige sessie en het opgeslagen project, niet bij globale toepassingsvoorkeuren. Batch en CLI verwerken hun oorspronkelijke invoer afzonderlijk, met sensorreiniging vóór flat-field-correctie en registratie. Defectmaskers vereisen bronbestanden met neutrale EXIF-oriëntatie. Georiënteerde bestanden, camera-RAW-bestanden en bronnen waarvan de oriëntatie niet kan worden vastgesteld, worden geweigerd omdat hun gedecodeerde pixels mogelijk niet meer overeenkomen met sensorcoördinaten. Gebruik bronframes en een masker in dezelfde bekende pixeloriëntatie, met neutrale oriëntatiemetadata. Alleen ruisonderdrukking vereist geen defectmasker.

De optionele sensorruisonderdrukking staat los van het defectmasker en is standaard uit. Bekijk het resultaat op volledige grootte om te beslissen of de afvlakking geschikt is voor het specimen. Sensorreiniging behoudt de bronbestanden en de 8- of 16-bitbeelddiepte; het staat los van flat-field-correctie. De CLI accepteert `--defect-mask-path MASK` en `--sensor-denoise`; recepten en de Python-recept-API leggen deze opties vast. Recepten registreren een maskervingerafdruk en weigeren een ontbrekend of gewijzigd masker. CLI-reiniging kan niet met transparante invoer worden gecombineerd. Gegroepeerde verwerking past reiniging toe op oorspronkelijke frames, niet op tussenresultaten van groepen.

Recepten vereisen een uitvoerlocatie buiten elke invoermap en weigeren een uitdrukkelijk geselecteerde invoer als uitvoer. Kies een aparte uitvoerlocatie om de bronafbeeldingen te beschermen en te voorkomen dat een eerder resultaat bij een nieuwe run wordt meegenomen. Verwerkingsinstellingen kunnen als sjabloon of recept uit een project worden geëxtraheerd, ook als de oorspronkelijke bronnen zijn verplaatst. Een verwerkingssjabloon neemt de schakelaar voor sensorruisonderdrukking over, maar geen pad naar een defectmasker; een recept bewaart het maskerpad en zijn vingerafdruk. Het project zelf openen vereist nog steeds de bronnen en een eventueel vastgelegde flat-field-referentie.

Wanneer een bestandsreferentie in Extra op de basisschaal van de stapel wordt toegepast, legt een opgeslagen project het absolute pad, de inhoudshash en de stofgrootte vast en past die bij openen één keer toe op de oorspronkelijke bronnen. Houd het referentiebestand ongewijzigd en op zijn plaats: een ontbrekende of gewijzigde referentie verhindert het openen van het project. Een referentie die uit de stapel is gekozen, of correctie op een andere werk-/basisschaal, vereist nog steeds export van de gecorrigeerde bronstapel voordat een project wordt opgeslagen. Verandert de werkschaal na correctie, zet die dan vóór opslaan terug naar de correctieschaal of exporteer de gecorrigeerde stapel.

**Help → Feedback versturen…** opent lokale logboeken. Kopiëren kopieert het zichtbare sessielogboek; de feedback-ZIP bevat sessie- en volledige applicatielogboeken en omgevingsinformatie. Controleer het pakket voordat je het zelf indient. GitHub openen opent alleen de tracker. Annuleren van een export laat bestanden ongewijzigd; bij een fout blijft het vorige doelbestand behouden en wordt het mislukte pad getoond.

Elke start schrijft zijn eigen roterende logboek, zodat gelijktijdige GUI-/CLI-taken gescheiden registraties houden. Taak-ID's koppelen parameters, verstreken tijd en uitkomsten (voltooid/geannuleerd/mislukt); onverwachte fouten bevatten een volledige traceback. De feedback-ZIP bevat ook het huidige crashrapport en afhankelijkheidsversies. Gebruik **Help → Logmap openen** of het pad op de feedbackpagina om de bestanden te vinden. Logboeken roteren bij 1 MB met drie back-ups per proces; inactieve sessielogboeken ouder dan 14 dagen worden verwijderd. Stel voor uitgebreide supportlogboeken `OPENFOCUS_LOG_LEVEL=DEBUG` in vóór de start; `OPENFOCUS_LOG_DIR` kiest optioneel een andere logmap.

### Tegelinstellingen (geheugenoptimalisatie)

Voor grote afbeeldingen gebruikt OpenFocus tegelgebaseerde verwerking om geheugenproblemen te vermijden.

Toegang via **Instellingen → Geavanceerde instellingen… → Prestaties**:

- **Tegelgrootte (px)**: grootte van elke verwerkingstegel (standaard: 1024 px)
- **Tegeloverlap (px)**: overlap tussen tegels voor naadloze samenvoeging (standaard: 256 px)
- **Tegels boven (px)**: beeldgrootte waarboven tegels worden gebruikt (standaard: 2048 px)

**Aanbevelingen**:

- Kleinere blokgrootte: lager geheugengebruik, langzamere verwerking
- Grotere blokgrootte: hoger geheugengebruik, snellere verwerking
- Verhoog de overlap als je zichtbare naden in het resultaat ziet

Een kleinere tegel kan het resultaat enkele niveaus verschuiven; de overlap is beperkt tot de helft van de tegelgrootte.

### Registratie-instellingen

Toegang via **Instellingen → Geavanceerde instellingen… → Registratie**:

- **Verkleiningsbreedte**: voorverwerkingsresolutie voor registratie
  - Standaard: 1024 px
  - Lager voor snelheid, hoger voor nauwkeurigheid

### Prestatiereferentie

Zie [PERFORMANCE.md](./PERFORMANCE.md) voor gemeten doorvoer en geheugengebruik op grote echte stapels, en praktische richtlijnen voor diepe stapels en invoer met hoge resolutie.

Het statuspaneel toont gebruikt en totaal fysiek geheugen wanneer het besturingssysteem beide waarden kan leveren; anders toont het `N/A`.

### Threadinstellingen

Regelt parallelle CPU-verwerking:

Toegang via **Instellingen → Geavanceerde instellingen… → Prestaties**:

- `0` kiest automatische planning en is de standaard voor nieuwe instellingen.
- Positieve waarden zijn bovenlimieten; opgeslagen limieten overleven upgrades.
- CPU-capaciteit, resourcemodus en vrije geheugenruimte kunnen de gelijktijdigheid verlagen. Meer threads maken een render niet noodzakelijk sneller.

### Resourcemodus

**Instellingen → Geavanceerde instellingen… → Prestaties → Resourcemodus** bepaalt hoeveel van de machine een render mag gebruiken. Het verandert nooit het gefuseerde beeld.

| Modus | Gedrag |
|---|---|
| Gebalanceerd (standaard) | Houdt ongeveer 20 % van de fysieke cores en een geheugenreserve vrij voor de desktop |
| Responsief | Gebruikt ongeveer de helft van de cores en een kleiner in-flight-venster; langzamere renders, meest responsieve machine |
| Maximaal | Gebruikt elke fysieke core die het geheugenbudget toestaat |

De planner houdt rekening met fysieke cores, CPU-affiniteit en quota en voert één parallelle laag tegelijk uit. Zware taken delen toelating via voorbereiding en publicatie; annuleren wacht tot de actieve native aanroep is voltooid. Onbekend of onvoldoende geheugen geeft een resourcefout. Het gekozen tegelraster blijft ongewijzigd; zelf kleinere tegels kiezen kan de uitvoerpixels veranderen.

### GPU-versnelling

StackMFF V1/V2/V3 gebruiken ONNX Runtime. DirectML op Windows en CoreML op macOS kunnen inferentie versnellen; anders wordt de CPU gebruikt. Beschikbaarheid van een provider bewijst geen GPU-uitvoering. Ondersteuning voor andere GPU’s vereist validatie.

- Schakel via **Instellingen → GPU-versnelling**
- Uitschakelen dwingt de neurale methoden naar de CPU — nuttig wanneer GPU-stuurprogramma's problemen geven
- Het statuspaneel toont `N/A` wanneer geen versneller wordt gebruikt of beschikbaar is, en `CPU*` wanneer er een versneller is maar de gekozen methode alleen op de CPU draait
- De keuze werkt bij de volgende render en wordt onthouden

### AI-batchgrootte

Toegang via **Instellingen → Geavanceerde instellingen… → AI**:

- De opgeslagen batchlimiet is standaard 2; het huidige uitvoeringsbeleid beperkt de werkelijke inferentie op elk apparaat tot één tegel per aanroep.
- Een hogere limiet schakelt geen ongevalideerde GPU-batching in en verkleint de ruimtelijke tegelgrootte niet.

### Thema's

Er zijn twee interfacethema's: **Licht (standaard)** en **Donker**. Wissel via **Instellingen → Thema**; de keuze wordt onmiddellijk toegepast (de Windows-titelbalk volgt) en onthouden. Bij de eerste start kies je een thema in het welkomstvenster; een opgeslagen keuze heeft voorrang.

### Draagbare modus

Maak een leeg bestand met de naam `OpenFocus.portable` naast `OpenFocus.exe` (de draagbare ZIP bevat er al een) en alle gebruikersgegevens — instellingen, logboeken, recente bestanden — worden opgeslagen in een map `OpenFocusData` naast de toepassing in plaats van in je gebruikersprofiel, zodat de installatie op een USB-stick kan meereizen. Verwijder het markeringsbestand om terug te schakelen naar het normale profiel.

### EXIF-metadata

Bronmetadata (cameramodel, belichting, oriëntatie enzovoort) blijft behouden: oriëntatie wordt bij het laden toegepast en PNG/TIFF/WebP-exporten sluiten de oorspronkelijke EXIF automatisch in. JPG/BMP-exporten kunnen geen volledige EXIF opslaan en worden zonder geschreven.

### HEIC/HEIF-invoer

iPhone-foto's laden rechtstreeks. Vereist `pillow-heif`; `pip install openfocus` installeert het en de gepakte builds bevatten het.

### Updates

**Help → Controleren op updates** vergelijkt je versie met de release-informatie op `openfocus.top`. Een nieuwere versie wordt van dezelfde website op de achtergrond gedownload terwijl de app inactief is; zodra de build klaarstaat, bieden de banner en statusbalk **Bijwerken en herstarten**, dat hem offline toepast. Builds die zichzelf niet kunnen bijwerken, openen in plaats daarvan de downloadpagina van de website. De automatische controle loopt hoogstens één keer per dag stil op de achtergrond.

### Registratiecache

OpenFocus kan registratieresultaten opslaan in een map `.openfocus_cache` naast de bronstapel. Dezelfde stapel opnieuw openen (bestanden ongewijzigd) slaat de uitlijning dan volledig over. Schakel via **Instellingen → Geavanceerde instellingen… → Registratie → Registratiecache**; verouderde items worden automatisch vervangen. De schakelaar staat standaard aan.

### Bewaarde instellingen

Alles hieronder overleeft automatisch een herstart van de toepassing: threadlimiet, resourcemodus, tegelparameters, verkleiningsbreedte voor registratie, AI-batchgrootte (StackMFF V3), GPU-versnelling, registratiecache, uitlijningsdiagnose, interfacetaal, venstergrootte/paneelindeling, sleepformaat, de helderheidsopties per methode, afvlakkings- en straalwaarden en je recent geopende stapels. Instellingsdialogen slaan op zodra je bevestigt; alles wordt ook bij het afsluiten van de toepassing opgeslagen. Voorkeuren worden per gebruiker opgeslagen in `OpenFocus/OpenFocus.ini` (QSettings INI-formaat).

### Interfacetaal

Wissel op elk moment via **Instellingen → Taal**; het menu toont alle beschikbare taalpakketten — English, 中文, 日本語, Español, Deutsch, Français, Русский, 한국어, Português, Italiano, Nederlands en Polski. De wijziging werkt onmiddellijk en wordt voor volgende sessies onthouden. Een nieuwe taal is één bestand `locales/<code>.py`.

### Automatische updatecontrole

Eén keer per dag, enkele seconden na de start, controleert OpenFocus stil op `openfocus.top` op een nieuwere versie. Ook de release-opmerkingen komen van die website. Als er een nieuwere versie is, verschijnt een banner boven de beeldpanelen: de versie, **Nu bijwerken** (past een klaargezette build toe of start de download), **Wat is er nieuw** (release-opmerkingen) en een knop om te sluiten. Sluiten verbergt de banner alleen voor die versie — de volgende release brengt hem terug, en **Help → Controleren op updates** activeert hem onmiddellijk opnieuw.

Terwijl de download op de achtergrond loopt, toont de statusbalk de voortgang. Je kunt ook op elk moment handmatig controleren via **Help → Controleren op updates**.

---
## 12. Batchverwerking

Meerdere mappen automatisch verwerken:

1. Ga naar **Extra → Batchverwerking**
2. In het batchdialoog:
   - **Mappen toevoegen**: mappen met beeldreeksen selecteren
   - **Stapelmappen hierheen slepen**: sleep een of meer stapelmappen naar het dialoog of de mappenlijst om ze in één keer in de wachtrij te zetten; neerzetten schakelt naar de modus met meerdere mappen en de miniaturen verschijnen terwijl ze worden gedecodeerd
   - **Selectie verwijderen**: geselecteerde mappen uit de lijst verwijderen
3. Voor één map kan OpenFocus die automatisch in meerdere stapels splitsen: kies **Eén map (automatisch splitsen in meerdere stapels)**, stel de splitsingsmethode **Vast aantal** of **Tijdsdrempel** in, bekijk het voorbeeld en klik op **Map kiezen en splitsen**.
4. Uitvoerinstellingen configureren:
   - **Submap in de bronmap aanmaken**: maakt resultaten in elke invoermap
   - **Zelfde als bronmap**: slaat naast de invoer op en behoudt bestaande bestanden
   - **Uitvoermap opgeven**: een uitvoerlocatie kiezen
5. Controleer de huidige fusie- en registratie-instellingen
6. Klik op **Batchverwerking starten** om te beginnen

Het batchdialoog toont voortgang in realtime. Je kunt de verwerking op elk moment annuleren.

### Opmerkingen bij batchverwerking

- Start met een momentopname van de instellingen van het hoofdvenster; bedieningselementen in het batchdialoog overschrijven die.
- Elke map of gesplitste groep wordt onafhankelijk door dezelfde pijplijn verwerkt.
- **Groepsfusie** is een optionele manier om lange stapels in de modus met meerdere mappen te verwerken. Vink **Grote stapels in groepen verwerken** aan; groepsgrootte en overlap worden pas dan getoond. De groepsgrootte moet minstens 2 zijn en de overlap tussen 0 en de groepsgrootte min 2 liggen. Staat het vakje uit, dan zijn beide velden verborgen en is groepering uitgeschakeld. De overlap in de desktop begint bij 1 en wordt bij het wijzigen van de grootte tot het toegestane bereik beperkt. Groepen worden in rondes met tijdelijke schijfruimte gefuseerd; de voortgangsweergave markeert elke ronde. Fusie, belichtingsafstemming, basiscorrectie en registratie gebruiken de frames en referenties van elke groep, ook in tussenrondes, zodat het resultaat de verwerking van de volledige stapel benadert. Groepering vereist een fusiemethode; diepte, herkomst en export van uitgelijnde frames zijn niet beschikbaar. De gesplitste-mapmodus wordt niet ondersteund; daar wordt het groeperingsvakje geweigerd. Geregistreerde groepen gebruiken hun eerste frame als coördinatenreferentie. Elke registratie verwijdert een conservatieve rand voor interpolatie, en latere rondes snijden de resterende rechthoeken inclusief een eventueel enkel restframe bij. Het uiteindelijke beeld wordt daardoor bijgesneden. Elke te registreren groep moet in beide richtingen groter dan 8 pixels zijn; moeilijke rotaties of projectieve transformaties kunnen geen veilig bruikbare rechthoek opleveren en een fout geven. Gebruik in dat geval vooraf uitgelijnde invoer zonder registratie, of schakel groepering uit.
- Gegroepeerde invoer moet overeenkomende georiënteerde afmetingen hebben en bestaan uit enkelbladige bestanden met door Pillow leesbare headers. Camera-RAW, meerbladige bestanden en bestanden met alfakanalen of transparantie worden niet ondersteund, inclusief transparante PNG-stapels. Verwijder alpha of schakel groepering uit om zulke stapels te verwerken. CLI-proefruns controleren deze beperkingen. Een CLI-voorbeeld is `python main.py --input stack --output fused.png --group-size 8 --group-overlap 1`. Gegroepeerde CLI-invoer moet een map zijn en kan niet met `--depth-map` worden gecombineerd. Een andere stapelbrede OpenFocus-uitlijningsreferentie kan niet met groepering worden gecombineerd. Een flat-field-referentie wordt vóór registratie op oorspronkelijke frames toegepast; uitvoerverscherping wordt één keer na de laatste groep toegepast.
- **Wat waar terechtkomt, hangt van de modus af.** Bij **Alleen uitlijnen** krijgt de gekozen bestemming één map `<stapel>_aligned` per stapel, en het uitvoerpaneel toont die map. Bij **Fusie + uitgelijnde frames opslaan** wordt het gefuseerde beeld als `<stapel>.<ext>` in de bestemming geschreven en gaan de uitgelijnde frames ernaast naar `<stapel>_aligned`; het uitvoerpaneel toont alleen het gefuseerde resultaat. Bij **Fusie zonder export van uitlijning** wordt alleen het gefuseerde beeld geschreven. Het uitvoerpaneel noemt mappen precies zoals ze op schijf heten.
- Uitgelijnde frames behouden de oorspronkelijke bestandsnamen. Dubbele namen en bestaande mappen krijgen numerieke suffixen. Bronbestanden worden nooit vervangen.
- Achtergebleven `*_registered*`-kopieën van oudere builds worden niet als frames geladen wanneer het oorspronkelijke bestand in dezelfde map aanwezig is; het laadbericht meldt hoeveel er zijn overgeslagen.
- PNG/TIFF behouden 16-bitgegevens; expliciet JPG/BMP kiezen zet ze om naar 8 bit.
- Annuleren wacht tot de worker stopt. Vastgelegde bestanden blijven beschikbaar; gedeeltelijke, mislukte, geannuleerde en niet-gestarte taken worden afzonderlijk geteld.

---

## 13. Resultaten exporteren

### Eén resultaat opslaan

1. Selecteer het gewenste resultaat in de uitvoerlijst
2. Ga naar **Bestand → Opslaan** of druk op `Ctrl+S`
3. Kies formaat en locatie
4. Klik op Opslaan

Onder **Instellingen → Geavanceerde instellingen… → Uitvoerbenaming** helpt een optioneel bestandsnaamvoorvoegsel om nieuw gerenderde resultaten te herkennen. Het voorbeeld werkt bij terwijl je typt; gebruik maximaal 40 letters, cijfers, underscores of koppeltekens. Laat het leeg voor de oorspronkelijke benaming. Bevestig met **OK** om het te bewaren, of annuleer om de wijziging te verwerpen. Het voorvoegsel wordt onderdeel van nieuwe namen in de uitvoerlijst en hun voorgestelde exportnamen; bestaande resultaten behouden hun namen. Je kunt een naam in het opslaandialoog nog steeds bewerken. Deze voorkeur hernoemt geen bronbestanden en wijzigt geen batchnamen.

### Sleepformaat voor export

Wanneer je resultaten uit de uitvoerlijst sleept, worden beeldbestanden geschreven in het formaat dat in het contextmenu van de uitvoerlijst onder **Sleepformaat** is gekozen (JPG, PNG, WebP of TIFF). De keuze wordt onthouden.

### De uitgelijnde stapel opslaan

1. Ga naar **Bestand → Stapel opslaan → Uitgelijnde stapel**
2. Kies het opslagformaat:
   - **Opslaan als map**: kies een bestemming en codering; elke stapel krijgt een nieuwe submap met zijn oorspronkelijke bestandsnamen
   - **Opslaan als GIF**: maakt een geanimeerde GIF van de uitgelijnde reeks

### De invoerstapel opslaan

1. Ga naar **Bestand → Stapel opslaan → Invoerstapel**
2. Kies het opslagformaat:
   - **Opslaan als map**: slaat de oorspronkelijke frames als afzonderlijke afbeeldingen op
   - **Opslaan als GIF**: maakt een geanimeerde GIF van de invoerreeks

### Exportformaten

- **JPG**: gecomprimeerd, goede verhouding tussen kwaliteit en bestandsgrootte
- **PNG**: verliesvrij, behoudt 8- en 16-bitbeeldgegevens
- **BMP**: ongecomprimeerd, zet 16-bitbeeldgegevens om naar 8 bit
- **TIFF**: behoudt 8- en 16-bitgegevens; meerbladige export bewaart de hele stapel in één bestand
- **GIF**: animatie met 8-bitpalet; annuleren laat een bestaand doel ongewijzigd
- **TIFF met lagen**: via de indelinglijst van **Opslaan als**; bewaart resultaat en bronframes met geometrie en geldigheidsmaskers

---

## 14. Opdrachtregelinterface

`pip install openfocus` installeert het AI-model, de gewichten en de HEIC- en camera-RAW-decodering. Op Python 3.13+ vervalt `dtcwt`; daar blijft de DTCWT-optie uitgeschakeld met uitleg.

OpenFocus kan zonder GUI draaien — nuttig voor gescripte pijplijnen en CI. Het consolescript `openfocus` (geïnstalleerd door `pip install openfocus`) en `python main.py` uit een checkout bieden dezelfde opdrachten. Fusieopties werken in beide vormen; de subopdrachten zijn:

```bash
openfocus version                       # versie tonen
openfocus describe                      # JSON: methoden, parameters, mogelijkheden en besturingsopdrachten
openfocus gui [--control] [--socket NAAM] [paden…]   # desktopapp starten
openfocus ctl <command> [flags]          # een draaiend venster besturen (zie AGENT_CONTROL_SPEC.md)
openfocus recipe create FILE --input … --output … [verwerkingsopties]
openfocus recipe run FILE [--dry-run]    # een opgeslagen headless recept afspelen
openfocus retouch replay CHECKPOINT --base BASE --sources … --output RESULT
```

`openfocus gui` start de app losgekoppeld; `--control` wacht op het besturingskanaal en drukt de verbindings-JSON af. `openfocus describe` en `version` hebben geen invoer nodig. De agentworkflow staat in [AGENT_CONTROL_SPEC.md](./AGENT_CONTROL_SPEC.md) en `skills/openfocus-control/SKILL.md`.

Vanuit een Python-omgeving (of naast het gepakte uitvoerbare bestand):

```bash
# Een map fuseren met de standaard AI-methode
python main.py --input ./stack_folder --output ./result/fused.png

# DTCWT met ECC-registratie, 8 threads
python main.py -i ./stack_folder -o ./result/fused.png -m dtcwt -a ecc -t 8

# StackMFF V3 geforceerd op CPU, met aangepaste tegelgrootte
python main.py -i ./stack_folder -o ./result/fused.png -m stackmffv3 --cpu --tile-size 512

# Expliciete bestandslijst; videobestanden worden ook als invoer geaccepteerd
python main.py -i img1.jpg img2.jpg img3.jpg -o fused.png -m gfgfgf

# Eén meerbladige TIFF-Z-stapel met een uit metadata gekalibreerde schaalbalk
python main.py -i ./zstack.tif -o fused.png -m dtcwt --scale-bar

# Aangrenzende vergelijkbare frames bekijken zonder te fuseren of de invoer te wijzigen
python main.py -i ./stack_folder --suggest-duplicates
```

`--suggest-duplicates` accepteert een map of een geordende lijst enkelbladige beeldbestanden. Het drukt één JSON-object af met de arrays `files` en `suggestions`; `index` en `reference_index` van elke suggestie zijn nulgebaseerde posities in `files`, en `difference` is het gemiddelde genormaliseerde steekproefverschil. Met `--json report.json` wordt hetzelfde object opgeslagen. Deze controlemode kan niet worden gecombineerd met `--output`, `--output-dir`, `--json-progress`, `--dry-run` of sensorreinigingsopties. Het rapport vergelijkt aangrenzende bestanden in de opgegeven invoervolgorde, zonder te weten welke frames je later uitsluit. Als je een referentie uitsluit, heroverweeg dan zijn opvolger of bekijk de resterende bestanden opnieuw in de bedoelde volgorde.

Voeg voor één rendering bijvoorbeeld `--align openfocus --alignment-report alignment.json` toe. Het JSON-object bevat `method`, `input`, `frame_count` en `entries`. `frame_index` van elke entry is nulgebaseerd in de gedecodeerde invoervolgorde, vóór de fusiesortering; meerbladige afbeeldingen worden in paginavolgorde uitgebreid en videoframes in decodeervolgorde. Samengestelde methoden bevatten afzonderlijke entries per fase. Het rapport wordt na succesvolle verwerking atomair geschreven en mag geen invoer of andere uitvoer vervangen. Batch-, gegroepeerde en proefrunmodi weigeren deze optie.

| Optie | Beschrijving |
|--------|-------------|
| `--input, -i` | Beeldmap of -mappen, videobestand, een of meer beeldbestanden of een meerbladige TIFF |
| `--output, -o` | Pad van het uitvoerbeeld (de extensie bepaalt het formaat) |
| `--output-dir, -d` | Batchmodus: elke invoermap fuseren en per stapel `<map>.png` schrijven |
| `--method, -m` | `stackmffv3` (standaard), `openfocus_a`, `openfocus_b`, `openfocus_c`, `guided_filter`, `dct`, `dtcwt`, `gfgfgf`, `stackmff`, `stackmffv2` |
| `--align, -a` | `none` (standaard), `homography`, `ecc`, `both`, `openfocus` |
| `--param KEY=JSON_VALUE` | Fusieparameter; herhaal voor meerdere sleutels; overschrijft oudere flags |
| `--align-param KEY=JSON_VALUE` | Registratieparameter; herhaal voor meerdere sleutels |
| `--align-shift-limit` | Alleen `openfocus`: maximaal geaccepteerde verschuiving als fractie van de frameafmeting (standaard 0,08; frames erboven blijven onuitgelijnd) |
| `--alignment-report PATH` | Uitlijningsresultaten per frame als JSON opslaan voor één niet-gegroepeerde rendering met uitlijning; batch- en gegroepeerde renders worden niet ondersteund |
| `--exposure-match` | Het gemiddelde van elk frame vóór fusie op het laatste frame afstemmen (standaard: aan; `--no-exposure-match` schakelt uit). Versterkingen gelden voor de hele stapel, zodat getegelde uitvoer naadloos blijft; geldt voor elke methode en voor de helderheidsreferentie van de OpenFocus-uitlijner |
| `--base-frame` | Optionele stap na fusie: de laagfrequentie van de uitvoer op het `last`/`first`-frame afstemmen (standaard `none`) |
| `--sort` | Framevolgorde voor `openfocus_a/b/c`: `asc` (standaard, bestandsvolgorde), `desc` (keert een ver-naar-dichtbij-reeks om), `off` |
| `--frame-order` | Map als invoer: `auto` (standaard) sorteert de beelden van dichtbij naar veraf op EXIF-focusstappen als alle beelden die hebben (OM SYSTEM-focusbracketing begint halverwege de diepte), anders op naam; `name` altijd op bestandsnaam. Expliciet opgegeven bestanden behouden hun volgorde |
| `--dust-map WHITE` | Flat-field-correctie (stofkaart) uit een onscherpe witveldsopname |
| `--defect-mask-path MASK` | Sensordefectmasker; waarden die niet nul zijn worden vóór registratie gerepareerd |
| `--sensor-denoise` | Voorzichtige sensorruisonderdrukking vóór registratie |
| `--kernel, -k` | Kernelgrootte (oneven; regelt bij OpenFocus A/B de schaal van de focusrespons; C heeft geen straal) |
| `--depth-smoothing VALUE` | Alleen `openfocus_b`: sterkte van diepteafvlakking (standaard 1,2) |
| `--blend-smoothing VALUE` | Alleen `openfocus_a` / `openfocus_c`: sterkte van mengingsafvlakking (standaard 2,0 voor A, 3,0 voor C) |
| `--sharpen-output` | Milde onscherpe maskering op het gefuseerde resultaat toepassen (standaard uit) |
| `--threads, -t` | Bovengrens voor workerthreads; `0` = automatisch (standaard), positieve waarden beperken de gelijktijdigheid |
| `--resource-preference` | Resourcemodus: `balanced` (standaard), `responsive` of `maximum` |
| `--batch-size` | Bovengrens voor AI-tegelbatch (standaard 2); de werkelijke inferentie gebruikt momenteel één tegel per aanroep |
| `--ai-mode` | StackMFF V3: `soft` (standaard, waarschijnlijkheidsgewogen menging) of `hard` (waarschijnlijkste frame per pixel) |
| `--cpu` | CPU forceren, ook als een GPU beschikbaar is |
| `--downscale` | Verkleiningsbreedte voor kenmerkdetectie bij registratie |
| `--tile-size` | Tegelblokgrootte voor grote afbeeldingen (standaard: automatisch) |
| `--tile-enabled` / `--no-tile-enabled` | Getegelde fusie in- of uitschakelen (standaard: aan) |
| `--tile-overlap` | Tegeloverlap in pixels (standaard: automatisch) |
| `--tile-threshold` | Minimale beeldgrootte voor tegels (standaard: automatisch) |
| `--group-size` | Een map in groepen van deze grootte fuseren; `0` schakelt groepering uit (standaard). Alleen mappen; geen `--depth-map`, RAW/alpha/meerbladige invoer of niet-standaard OpenFocus-referentie |
| `--group-overlap` | Frames die opeenvolgende groepen delen; standaard `0`, maximaal groepsgrootte min 2 |
| `--scale-bar` | Een schaalbalk tekenen (kalibratie automatisch uit TIFF-metadata gedetecteerd) |
| `--scale-um` | Pixelgrootte in µm/px, overschrijft metadata (impliceert `--scale-bar`) |
| `--depth-map [pad]` | Ook een gekleurde dieptekaart (focuspositie) opslaan (standaard `<output>_depth.png`); alleen dieptegeschikte methoden, niet gegroepeerd of transparant. In batchmodus is alleen het standaardpad toegestaan |
| `--depth-colormap` | Kleurenschaal voor `--depth-map`: turbo, viridis, jet, gray, … (standaard turbo) |
| `--auto-background [preset]` | De gestapelde achtergrond vervangen door die van een uitgelijnd beeld, zoals Automatische achtergrond bij retoucheren: `auto` (standaard), `soft` (meest onscherpe achtergrond) of `sharp` (verst scherpgestelde beeld). Heeft de uitgelijnde beelden nodig, dus niet met gegroepeerde fusie |
| `--background-mode` | Automatische achtergrond: `clean` (standaard) verwijdert ook de randhalo's van de stapel, binnen wat de beelden hebben vastgelegd; `faithful` behoudt ze |
| `--keep-soft-parts` | Automatische achtergrond: zachte delen aan het onderwerp behouden waarop een beeld heeft scherpgesteld |
| `--save-stacked [pad]` | Met `--auto-background` ook het resultaat vóór de vervanging opslaan (standaard `<output>_stacked<ext>`). In batchmodus alleen het standaardpad |
| `--missing-data-map [pad]` | Met `--auto-background` ook de kaart met ontbrekende gegevens en per klasse een 8-bit masker opslaan (standaard `<output>_missing.png`). In batchmodus alleen het standaardpad |
| `--dry-run` | Invoer controleren en het plan tonen zonder te fuseren |
| `--resume` | Batchmodus: mappen overslaan waarvan het uitvoerbestand al bestaat |
| `--json FILE` | Een machineleesbare resultaatsamenvatting naar FILE schrijven |
| `--json-progress` | Eén JSON-voortgangsobject per stap naar stdout schrijven (NDJSON); proza gaat naar stderr |
| `--config FILE` | Batchmodus: JSON-bestand met overschrijvingen per map, bijvoorbeeld `{"stackA": {"method": "dct", "kernel": 9}}` |
| `--suggest-duplicates` | Bijna identieke aangrenzende frames voor handmatige controle melden; fuseert of sluit ze niet uit |

Er is geen `--method none`-waarde: de CLI fuseert altijd. Gebruik de desktopapp voor uitvoer met alleen uitlijning.

### Recepten

Recepten leggen headless verwerkingsinstellingen vast voor hergebruik en verplaatsing:

```bash
openfocus recipe create workflow.json -i ./stack_folder -o ./result/fused.png -m dct --cpu
openfocus recipe run workflow.json --dry-run
openfocus recipe run workflow.json
```

Het JSON-formaat van versie 1 bevat `version`, een `input`-array, ofwel `output` of `output_dir`, en een `options`-object. Paden zijn relatief ten opzichte van het receptbestand; verplaats het recept samen met zijn invoerbestanden. Bij het maken worden de CLI-instellingen en opgeloste algoritmeparameters vastgelegd. Onbekende velden, ongeldige waarden en niet-ondersteunde versies worden geweigerd. Een proefrun controleert de invoer zonder resultaatbestanden te schrijven. Bestaande projectbestanden blijven ongewijzigd.

Instellingen uit een bestaand `.ofproj` overnemen:

```bash
openfocus recipe create workflow.json --project session.ofproj -i ./new_stack -o ./result/fused.png
```

Dit neemt fusie, registratie, helderheid, volgorde, verscherping en tegelinstellingen over in het CLI-recept en sluit de verwerkingsinstellingen voor de GUI in. Frame-uitsluitingen, laadschaal, labels en bewerkingen worden niet overgedragen. Projecten met brontransformaties, inclusief flat-fieldcorrecties, moeten eerst hun gecorrigeerde bronstapel exporteren. Recepten vereisen uitvoer buiten hun invoermappen en accepteren nooit een invoerbestand als uitvoer. Ze weigeren tijdelijke besturingsopties zoals `--dry-run`, `--resume`, `--json` en `--config`. Een defectmasker wordt als relatief pad met een SHA-256-vingerafdruk opgeslagen; een ontbrekend of gewijzigd masker laat de run mislukken.

Groeperingsopties kunnen uit CLI-argumenten worden overgenomen.

In de GUI slaat **Bestand → Verwerkingssjabloon opslaan** instellingen op zonder bronpaden. **Bestand → Verwerkingssjabloon toepassen** past een sjabloon, project of projectsjabloon toe op de huidige stapel zonder de invoer, transformaties, uitsluitingen of labels te vervangen. Een sjabloon is zelf geen uitvoerbaar recept. De GUI-instellingen in een recept staan los van de CLI-opties; houd beide consistent als je de JSON bewerkt.

Dezelfde headless API is beschikbaar zonder PyQt te importeren:

```python
from core.recipes import capture_project_recipe, capture_recipe, load_recipe, save_recipe, run_recipe

capture_recipe("workflow.json", ["-i", "stack_folder", "-o", "result/fused.png", "-m", "dct"])
capture_project_recipe("from-project.json", "session.ofproj",
                       ["-i", "new_stack", "-o", "result/fused.png"])
recipe = load_recipe("workflow.json")  # gevalideerd dictionary met relatieve paden
save_recipe("workflow.json", recipe)
exit_code = run_recipe("workflow.json", dry_run=True)  # False om te verwerken
```

`capture_recipe` interpreteert zijn argumenten als gewone CLI-paden relatief ten opzichte van de huidige map; `save_recipe` interpreteert opgeslagen paden relatief ten opzichte van het recept. Het recept legt instellingen en verwijzingen vast, inclusief een inhoudsvingerafdruk voor een uitdrukkelijk gekozen sensordefectmasker. Het legt niet de invoerpixels, softwareversies, hardware of controlesommen van de bronstapel vast; bewaar die apart wanneer herkomst belangrijk is. Identieke instellingen garanderen geen identieke pixels over softwareversies, stuurprogramma's of apparaten heen; GPU-neurale bewerkingen kunnen niet-deterministisch zijn. Gebruik `--cpu` wanneer uitvoering op de CPU vereist is, en bewaar de geïnstalleerde software en invoer.

### Retouchecontrolepunten afspelen

Gebruik de knoppen voor het opslaan en laden van controlepunten in de retouchewerkruimte met de oorspronkelijke basis en verwerkte bronnen. Sessies met automatische achtergrond, ook geslaagde analyses zonder pixelwijzigingen, bewaren verliesloze v2-`.ofretouch`-ZIP-momentopnamen met JSON-metadata en NumPy-gegevens: het exacte werkbeeld, bereikbare tegels voor ongedaan maken/opnieuw uitvoeren, diagnostische toestanden en bronherkomst. Laden van v2 berekent de achtergrond niet opnieuw. Sessies met alleen penseel-/toonbewerkingen kunnen compact JSON v1 gebruiken; oudere achtergrondcontrolepunten in JSON v1 blijven afspeelbaar.

Broncontrole verloopt in een annuleerbare achtergrondbewerking. Voor het vervangen van de bestemming laadt opslaan het voorbereide bestand opnieuw en controleert het huidige beeld, bereikbare tegels voor ongedaan maken/opnieuw uitvoeren en diagnostische toestanden op exacte gelijkheid. Niet-passende bronnen, een mislukte controle, annulering of onvoldoende geheugen laten je huidige bewerkingen en een bestaand controlepunt ongewijzigd.

Als de opnamelimiet van het controlepunt een volgende bewerking verhindert, pas je de huidige bewerkingen toe en begin je een nieuwe retouchesessie om verder te gaan.

Sla een retouchecontrolepunt op vanuit de retouchewerkruimte en geef daarna het oorspronkelijke basisbeeld en elk verwerkt bronframe in dezelfde volgorde als de sessie op:

```bash
openfocus retouch replay edits.ofretouch --base fused.png --sources processed1.png processed2.png --output retouched.tif
```

Beide formaten vereisen nog steeds het oorspronkelijke basisbeeld en alle frames die daadwerkelijk voor het retoucheren zijn gebruikt, in sessievolgorde, na uitlijning en een eventuele ROI-bijsnijding of andere verwerking, met hun oorspronkelijke pixelwaarden en bitdiepte. Laden reconstrueert de vastgelegde frame-offsets en het uitlijningsgedrag en controleert de basis, de geordende frames en geldigheidsmaskers; onverenigbare pixels, geometrie of maskers worden geweigerd. Elk pad moet één afbeelding bevatten. De uitvoer moet PNG of TIFF zijn en behoudt de bitdiepte van de basis. Het controlepunt en de invoerbeelden mogen niet als uitvoer worden gebruikt.

Exitcodes: `0` succes, `1` verwerkingsfout, `2` gebruiksfout. `python main.py --help` toont alle opties; starten met een gewoon pad (`python main.py C:\stack`) opent nog steeds de GUI met die map geladen.

---

## 15. Sneltoetsen

### Bestandsbewerkingen

| Sneltoets | Actie |
|----------|--------|
| `Ctrl+O` | Map openen |
| `Ctrl+Shift+O` | Video openen |
| `Ctrl+Alt+O` | Project openen |
| `Ctrl+Alt+S` | Project opslaan |
| `Ctrl+S` | Resultaat opslaan |
| `Ctrl+Shift+S` | Uitgelijnde stapel opslaan |
| `Ctrl+W` | Stapel wissen |
| `Ctrl+Q` | Afsluiten |
| `Ctrl+R` | Renderen starten |
| `Ctrl+Shift+R` | Alle methoden vergelijken |
| `Ctrl+T` | Resultaat retoucheren |
| `Ctrl+M` | Dieptekaart |
| `Ctrl+E` | Veegvergelijking in-/uitschakelen |
| `Ctrl+Shift+B` | Batchverwerking |
| `Ctrl+Z` / `Ctrl+Y` | Ongedaan maken / opnieuw uitvoeren (stapelbewerkingen) |

### Navigatie

| Sneltoets | Actie |
|----------|--------|
| `Pijl omhoog` / `Pijl omlaag` (bronlijst gefocust) | Vorig / volgend frame |
| `Muiswiel` | In-/uitzoomen |
| `Ctrl + wiel` | Fijn zoomen |
| `Dubbelklik` | Tussen inpassen en 100 % wisselen |

### Algemeen

| Sneltoets | Actie |
|----------|--------|
| `Spatie` (ingedrukt houden, cursor boven een afbeelding) | Tijdelijke panmodus |
| `Delete` | Het gefocuste lijstitem verwijderen; wist ook de ROI wanneer de cursor boven de bronafbeelding staat |

---

## 16. Problemen oplossen

### Veelvoorkomende problemen

#### „Geen afbeeldingen geladen”

- Zorg dat je eerst een beeldmap of video hebt geïmporteerd
- Controleer of de map ondersteunde beeldformaten bevat

#### Fusieresultaten zien er wazig of dubbel uit

- Schakel beeldregistratie in (kies een methode in de vervolgkeuzelijst: homografie / ECC / OpenFocus snel)
- Zorg dat de afbeeldingen vóór fusie goed zijn uitgelijnd
- Probeer een andere fusiemethode
- Verhoog de kernelgrootte voor een gladder resultaat

#### AI niet beschikbaar

- Controleer ONNX Runtime en het meegeleverde model.
- Controleer via **Help → Omgevingsinfo**

#### Fouten door te weinig geheugen

- Schakel tegelverwerking in via **Instellingen → Geavanceerde instellingen… → Prestaties**
- Verklein de tegelblokgrootte
- Gebruik kleinere kernelgroottes
- Schaal de afbeeldingen naar kleinere afmetingen

#### Registratie mislukt

- Kies **OpenFocus (snel)** voor begrensde starre beweging, of schakel **Stoppen bij afgewezen beeld** in om vroeg te falen in plaats van onuitgelijnde frames te accepteren
- Verklein de verkleiningsbreedte in **Instellingen → Geavanceerde instellingen… → Registratie**
- Zorg dat de afbeeldingen voldoende contrast en kenmerken hebben
- Schakel onder **Instellingen → Geavanceerde instellingen… → Registratie** de optie **Uitlijningsdiagnose verzamelen** in, render opnieuw en lees **Instellingen → Geavanceerde instellingen… → Registratie → Uitlijningsdiagnose** voor de betrokken frames

#### Trage verwerking

- Laat threads op automatisch (`0`); CPU-capaciteit, resourcemodus en geheugenreserve beperken de gelijktijdigheid al
- Gebruik voor een onbeheerde batch de modus **Maximaal**, of Guided Filter / GFG-FGF in plaats van een AI-methode
- Verlaag de beeldresolutie of gebruik **Snel voorbeeld** tijdens het afstellen

#### Windows SmartScreen-waarschuwing bij de eerste start

Of de Windows-binaries ondertekend zijn, hangt af van de specifieke release (zie de releaseopmerkingen); een handtekening is niet gegarandeerd. Download ze van **https://openfocus.top** in plaats van een externe mirror. Als SmartScreen waarschuwt, klik op **Meer informatie → Toch uitvoeren**. De eerste start na uitpakken kan 30–60 seconden duren.

#### macOS meldt dat de app „niet kan worden geopend” of „beschadigd is”

De macOS-builds zijn niet genotariseerd. Download de Apple Silicon-build van **https://openfocus.top**. Als u de DMG gebruikt en `OpenFocus.app` naar Programma's verplaatst, voer uit:

```bash
xattr -cr /Applications/OpenFocus.app
```

Bij een terminalinstallatie met `install.sh` staat de app in `~/Applications`; voer dan `xattr -cr ~/Applications/OpenFocus.app` uit.

of klik met rechts op de app en kies **Open**. De vrijgegeven macOS-builds zijn voor Apple Silicon (M1/M2/M3/M4); Intel-Macs kunnen vanuit de broncode draaien.

### Hulp krijgen

1. **Omgevingsinfo**: controleer je installatie via **Help → Omgevingsinfo**
2. **Contact opnemen**: ga naar **Help → Contact opnemen** voor supportinformatie
3. **Feedback versturen**: via **Help → Feedback versturen…** kopieer je logboeken of exporteer je een pakket om handmatig in te dienen

---

## Bijlage: Details van de fusiealgorithmen

### Guided Filter

De guided filter voert randbewarende afvlakking uit met een geleidingsbeeld. Hij berekent lokale lineaire transformaties tussen het geleidings- en invoerbeeld en past die toe op de uitvoer. Deze methode blinkt uit in het behouden van randen terwijl ruis wordt verminderd.

### DCT (Discrete Cosinustransformatie)

DCT brengt afbeeldingen naar het frequentiedomein, waar fusiebeslissingen op basis van coëfficiëntgroottes worden genomen. Hoogfrequente componenten (details) blijven behouden, terwijl laagfrequente componenten (egale gebieden) op basis van lokale energie worden samengevoegd.

### DTCWT (Dual-Tree Complex Wavelet Transform)

DTCWT levert een meerschalige, meervoudig gerichte decompositie van afbeeldingen. Het legt kenmerken op verschillende schalen en in verschillende richtingen vast, waardoor verfijnde fusieregels mogelijk zijn die zowel textuur- als structuurinformatie behouden.

### GFG-FGF

Implementeert Fu et al. (2024): de grove focuskaart |I − mean(I)| wordt verrijkt met de Gaussian-like vierbuurgradiënt — het 3×3-sjabloon T = [1 2 1; 2 4 2; 1 2 1]/16 geeft P = I⊗T, en G = (|I(x±1,y) − P|)/2 + (|I(x,y±1) − P|)/2, waarbij waarden onder T0 = 0,005 worden verworpen — en daarna verfijnd door een fast guided filter. Elke pixel kiest het frame met de grootste focuswaarde (bij gelijke stand: 3×3 Sobel-kleinmaat), en de binaire beslissingskaarten worden vóór de gewogen menging met de fast guided filter afgevlakt.

### AI-fusie (StackMFF V3)

Een deep-learningbenadering met een neuraal netwerk dat is getraind op grote datasets van multifocus-beeldparen. Het netwerk leert automatisch optimale fusiestrategieën en levert bij juiste configuratie resultaten op het hoogste niveau.

---

*OpenFocus is een opensourceproject. Bezoek voor updates, bugmeldingen en functieverzoeken de projectrepository of https://openfocus.top.*
