# Podręcznik użytkownika OpenFocus

## Spis treści

1. [Wprowadzenie](#1-wprowadzenie)
2. [Pierwsze kroki](#2-pierwsze-kroki)
3. [Przegląd głównego interfejsu](#3-przegląd-głównego-interfejsu)
4. [Importowanie obrazów i wideo](#4-importowanie-obrazów-i-wideo)
5. [Przeglądanie i nawigacja po obrazach](#5-przeglądanie-i-nawigacja-po-obrazach)
6. [Widok porównania suwakiem](#6-widok-porównania-suwakiem)
7. [Rejestracja obrazów](#7-rejestracja-obrazów)
8. [Metody fuzji obrazów](#8-metody-fuzji-obrazów)
9. [Przekształcenia obrazów](#9-przekształcenia-obrazów)
10. [Retusz, etykiety i adnotacje](#10-retusz-etykiety-i-adnotacje)
11. [Ustawienia i konfiguracja](#11-ustawienia-i-konfiguracja)
12. [Przetwarzanie wsadowe](#12-przetwarzanie-wsadowe)
13. [Eksport wyników](#13-eksport-wyników)
14. [Interfejs wiersza poleceń](#14-interfejs-wiersza-poleceń)
15. [Skróty klawiaturowe](#15-skróty-klawiaturowe)
16. [Rozwiązywanie problemów](#16-rozwiązywanie-problemów)

---

## 1. Wprowadzenie

OpenFocus to profesjonalna aplikacja desktopowa do fuzji obrazów z wieloma punktami ostrości, przeznaczona dla badaczy, inżynierów i specjalistów od przetwarzania obrazów. Program przetwarza sekwencje obrazów o różnych punktach ostrości za pomocą zaawansowanych algorytmów rejestracji i fuzji, tworząc obrazy w pełni ostre, o pełnej głębi ostrości.

### Najważniejsze funkcje

- **Fuzja obrazów o wielu punktach ostrości**: łączenie kilku zdjęć o różnych punktach ostrości w jeden całkowicie ostry obraz
- **Wiele algorytmów fuzji**: do wyboru filtr sterowany, DCT, DTCWT, GFG-FGF oraz fuzja AI (uczenie głębokie)
- **Rejestracja obrazów**: wyrównywanie przesuniętych sekwencji metodami ECC lub homografii
- **Widok porównania suwakiem**: kontrola wyrównania i porównywanie wyników w widoku A/B z dzielnikiem
- **Pliki projektu**: zapisywanie i przywracanie całych sesji pracy
- **Wielostronicowy TIFF**: stosy Z z jednego pliku wczytywane strona po stronie i eksportowane z powrotem do jednego pliku
- **Mapa głębi**: mapy położenia ostrości dla każdego piksela z własnymi paletami pseudo-kolorów
- **Pasek skali**: paski skali w jakości publikacyjnej z automatyczną kalibracją µm/px
- **Automatyczne odzyskiwanie sesji**: po awarii program proponuje przywrócenie ostatniej sesji
- **Wiele języków**: angielski, 中文, 日本語, Español z automatycznym wykrywaniem języka systemu
- **Interfejs wiersza poleceń**: skryptowe uruchamianie OpenFocus do automatyzacji i CI
- **Elastyczny eksport**: zapis wyników jako pojedynczych obrazów, folderów lub animacji GIF
- **Przekształcenia obrazów**: obracanie, odbijanie i zmiana rozmiaru stosów obrazów
- **Dodawanie etykiet**: tekstowe etykiety na stosach zarejestrowanych i wejściowych
- **Trwałe ustawienia**: preferencje, język i ostatnio otwierane stosy są zachowywane po restarcie

### Obsługiwane formaty wejściowe

- Pliki obrazów: JPG/JPEG, PNG, BMP, TIFF, WebP i inne popularne formaty
- Pliki wideo: format MP4 (automatycznie dekodowany do sekwencji obrazów)
- Foldery z numerowanymi sekwencjami obrazów
- Pliki RAW z aparatów (DNG, CR2/CR3, NEF, ARW, RAF, ORF, RW2 i inne popularne
  formaty RAW). Dekodowanie korzysta z `rawpy` (LibRaw), który instaluje
  `pip install openfocus`, a kompilacje dla Windows/macOS zawierają go w pakiecie

### Obsługiwane formaty wyjściowe

- Pojedyncze obrazy: JPG, PNG, BMP, TIFF, WebP
- Sekwencje obrazów: zapisywane jako foldery
- Animacje: format GIF

---

## 2. Pierwsze kroki

### Uruchamianie aplikacji

Po instalacji uruchom OpenFocus w jeden z następujących sposobów:

1. **Windows (plik wykonywalny)**: kliknij dwukrotnie skrót lub plik wykonywalny OpenFocus
2. **Kod źródłowy Pythona**: uruchom `python main.py` w aktywnym środowisku conda

Po uruchomieniu zobaczysz główne okno aplikacji z ciemnym motywem interfejsu.

### Szybki start

1. Zaimportuj sekwencję obrazów (Plik → Otwórz folder lub przeciągnij i upuść)
2. Skonfiguruj ustawienia fuzji w prawym panelu
3. Opcjonalnie włącz rejestrację, aby wyrównać klatki
4. W razie potrzeby dostosuj rozmiar jądra
5. *(opcjonalnie)* Zaznacz **Szybki podgląd**, aby najpierw przetestować parametry na zmniejszonym szkicu
6. Kliknij „Rozpocznij renderowanie”, aby rozpocząć przetwarzanie
7. Wyeksportuj wyniki

---

## 3. Przegląd głównego interfejsu

Interfejs OpenFocus składa się z trzech głównych obszarów:

### Lewy panel: obszar wyświetlania obrazów

Lewa część okna pokazuje obrazy w widoku dzielonym:

- **Panel obrazu źródłowego (góra)**: wyświetla wejściową sekwencję obrazów
- **Panel obrazu wynikowego (dół)**: wyświetla wyniki fuzji lub rejestracji
- **Suwak nawigacji**: suwak pod każdym panelem służy do przewijania klatek obrazu
- **Etykiety informacji o obrazie**: pokazują bieżącą pozycję klatki i wymiary obrazu

Panele można zmieniać w rozmiarze, przeciągając uchwyt dzielnika między nimi.

### Prawy panel: panel sterowania

Prawy panel zawiera wszystkie opcje konfiguracyjne ułożone pionowo:

1. **Ustawienia fuzji**: wybór algorytmu fuzji i rozmiaru jądra
2. **Ustawienia rejestracji**: włączanie i wyłączanie metod wyrównania
3. **Przyciski akcji**: Przywróć domyślne i Rozpocznij renderowanie
4. **Lista obrazów źródłowych**: pokazuje wszystkie wczytane obrazy z nazwami plików
5. **Lista wyników**: pokazuje wygenerowane wyniki

### Pasek menu

Górny pasek menu daje dostęp do wszystkich funkcji:

- **Plik**: operacje importu i eksportu oraz lista ostatnich plików
- **Edycja**: przekształcenia obrazów i etykiety
- **Narzędzia**: przetwarzanie wsadowe
- **Ustawienia**: akceleracja GPU, język, kafelki, rejestracja i liczba wątków
- **Pomoc**: informacje o środowisku, dane kontaktowe, sprawdzanie aktualizacji i folder logów

---

## 4. Importowanie obrazów i wideo

### Otwieranie folderu z obrazami

1. Wybierz **Plik → Otwórz folder** lub naciśnij `Ctrl+O`
2. Wskaż folder zawierający sekwencję obrazów
3. Pojawi się okno skali wczytywania — wybierz skalę (100% = pełna rozdzielczość)
4. Kliknij OK, aby wczytać obrazy

Zaznacz w tym oknie **Nie pytaj ponownie**, aby zawsze wczytywać w wybranej
skali bez pytania; **Ustawienia → Pytaj o skalę wczytywania przy imporcie**
przywraca pytanie. Wczytany stos nadal można później zmniejszyć przez
**Edycja → Zmień rozmiar**.

Program automatycznie:
- wykryje i posortuje pliki obrazów według nazw
- wyświetli pierwszy obraz w panelu źródłowym
- wypełni listę obrazów źródłowych wszystkimi klatkami

### Otwieranie plików wideo

1. Wybierz **Plik → Otwórz wideo** lub naciśnij `Ctrl+Shift+O`
2. Wskaż plik wideo (MP4, AVI, MOV, MKV, WMV, FLV, WEBM)
3. Wideo zostanie automatycznie zdekodowane na pojedyncze klatki
4. Każda klatka staje się częścią stosu obrazów

### Pliki projektu

Plik → Zapisz projekt zapisuje całą sesję pracy jako plik `.ofproj`:
dokładną listę plików źródłowych (w kolejności), skalę wczytywania, metodę
fuzji, opcje rejestracji, rozmiar jądra, ustawienia kafelków, wątków i
rejestracji, konfiguracje etykiet oraz bieżącą klatkę. Plik → Otwórz projekt
odtwarza tę sesję dokładnie (obrazy źródłowe są tylko wskazywane, nie kopiowane — trzymaj je na miejscu). Tytuł okna pokazuje otwarty projekt.

### Stos demonstracyjny

Okno powitalne oferuje **Wczytaj stos demonstracyjny**: sześć dołączonych
zdjęć przykładowych z prawdziwego stosu ostrości, dzięki czemu możesz od razu
wypróbować cały proces bez własnych obrazów. Przydaje się też do sprawdzenia
instalacji.

### Krzywa ostrości

Niewielka krzywa nad suwakiem nawigacji źródła pokazuje ostrość kolejnych
klatek (znormalizowaną). Klatki nieostre pojawiają się jako obniżenia
oznaczone czerwoną kropką. Kliknięcie w dowolnym miejscu krzywej przenosi
do danej klatki; najechanie kursorem pokazuje numer klatki.

### Stosy wielostronicowych plików TIFF

Oprogramowanie mikroskopowe (ZEN, NIS-Elements, ImageJ/Fiji) często eksportuje
cały stos Z jako JEDEN wielostronicowy plik TIFF. OpenFocus odczytuje każdą
stronę takiego pliku jako osobną klatkę stosu — upuść lub otwórz plik, a stos
pojawi się w całości (klatki nazywają się `<file>_page_001` itd.).
Jednostronicowe pliki TIFF działają dokładnie jak dotąd. Działa też odwrotnie:
**Plik → Zapisz stos → (Stos zarejestrowany / Stos wejściowy) → Zapisz jako wielostronicowy TIFF**
zapisuje cały stos do jednego pliku `.tif`, zachowując głębię 16-bitową.

### Pasek skali (ryciny publikacyjne)

Ustawienia → Pasek skali… konfiguruje pasek skali wypalany w eksportowanych obrazach.
Rozmiar piksela (µm/px) jest wykrywany automatycznie z metadanych TIFF — konwencja
ImageJ `unit=micron`, `PhysicalSizeX` w OME-TIFF lub jawna jednostka rozdzielczości
w centymetrach — dzięki czemu stosy mikroskopowe (w tym wielostronicowe TIFF)
kalibrują się same. Rozdzielczość drukarki (72/300 dpi i podobne) celowo *nie*
jest traktowana jako kalibracja, a wartość wpisana ręcznie zawsze wygrywa z
wykrytymi metadanymi: wiarygodnie wyglądający, ale błędny pasek na rycinie jest
gorszy niż brak paska. Wybierz narożnik i kolor (Auto odwraca kolor względem tła);
okno pokazuje podgląd na żywo. Po włączeniu paska każdy eksport (pojedynczy obraz,
stosy w folderach, wielostronicowy TIFF, GIF, eksport wszystkich wyników,
przetwarzanie wsadowe) zawiera pasek w stylu publikacyjnym: zaokrągloną długość
1-2-5 (około 12% szerokości obrazu) z obrysowanym, czytelnym tekstem. CLI obsługuje
to przez `--scale-bar` (automatyczna kalibracja) oraz `--scale-um <um/px>`.

### Mapa głębi (położenie ostrości)

**Narzędzia → Mapa głębi…** szacuje położenie ostrości miarą aktywności wybranej metody
(DCT, DTCWT, GFG-FGF, StackMFF V3, OpenFocus B/C). Nie jest to zmierzona powierzchnia
ani dokładna mapa pochodzenia pikseli wyniku fuzji. Wybierz obecnie zaznaczone klatki
lub zapisane źródło wybranego wyniku. Brakujących danych pochodzenia nie zastępuje się
bieżącym stosem. Obliczenia, tekstura i eksport pozostają związane z tą samą migawką źródła.

Wszystkie sześć metod wymaga jawnego kliknięcia **Oblicz**. Otwarcie okna oraz zmiana
miary, źródła, rejestracji lub parametrów nie rozpoczynają obliczeń. Edytor parametrów
znajduje się tuż pod wyborem miary: jądra medianowe DCT i początkowe GFG przyjmują wartości
nieparzyste 1–51 (domyślnie 7); B używa okna wariancji, a C okna energii pasmowej,
nieparzystych 1–21 px (domyślnie 11). DTCWT nie udostępnia tu poziomów: używa głębokości
dekompozycji z okna głównego, tak jak zwykłe renderowanie. StackMFF V3 zamiast pola
liczbowego pokazuje własny tryb wyjścia AI (domyślnie miękkie mieszanie lub twardy wybór)
i objaśnienie: miękkie mieszanie szacuje ważoną prawdopodobieństwem ciągłą pozycję klatki,
twardy wybór bierze najbardziej prawdopodobną klatkę dla każdego piksela. Żadne nie jest
odległością fizyczną. Wartości robocze zaczynają się od parametrów
pasującego wyniku lub zapamiętanych ustawień metody w oknie głównym. Są zachowywane
osobno dla źródła i metody podczas tego otwarcia, bez zmiany okna głównego, projektu
lub zapisanego wyniku.

Zmiana ustawień obliczeń trwale oznacza poprzednią mapę 2D jako **nieaktualną**, czyści
powierzchnię 3D i blokuje każdy eksport, w tym dane surowe, głębię fizyczną i OBJ, aż do
udanego ponownego obliczenia. Bieżące zadanie jest anulowane kooperacyjnie. Po zakończeniu
trzeba ponownie nacisnąć **Oblicz**; nic nie jest kolejkowane ani wznawiane automatycznie.
Oblicz pozostaje widoczne i jest wyłączone podczas obliczania, anulowania i eksportu modelu.

- **Mapa kolorów**: skala szarości (standard publikacyjny) oraz 15 palet
  naukowych (Turbo, Viridis, Magma, …) i własne — *Nowa paleta własna…*
  buduje gradient z dowolnej liczby punktów koloru (dwuklik na punkcie
  edytuje jego pozycję i kolor); własne palety są zapisywane i pojawiają się
  na liście rozwijanej.
- **Odwrócenie / Gamma / Wygładzanie**: odwrócenie osi głębi, ponowne
  zrównoważenie tonu (gamma < 1 rozjaśnia niskie indeksy) i usuwanie plamek
  przebiegiem świadomym krawędzi.
- **Nakładka**: miesza mapę 2D z powiązanym wynikiem przy regulowanym kryciu, tylko gdy
  ich obszary współrzędnych są zgodne.
- **Eksport**: wersja kolorowa obsługuje PNG/JPEG/TIFF/BMP. Surowy TIFF 16-bit koduje
  znormalizowane indeksy ostrości jako 0–65535; nie są to bezstratne dane zmiennoprzecinkowe
  ani odległości fizyczne. JPEG/BMP są odrzucane przy zapisie danych surowych.
- **Model 3D / Tekstura 3D**: dłuższy bok siatki ma domyślnie 600 wierzchołków.
  Każde nowe otwarcie wybiera **Automatyczna (rzeczywiste kolory)**: powiązany wynik jest
  używany tylko wtedy, gdy jego tożsamość i obszar współrzędnych/ROI pasują do ukończonej
  mapy. W przeciwnym razie używana jest pierwsza klatka uczestnicząca w zakończonych
  obliczeniach, po rejestracji i w faktycznej kolejności. Widoczna jest nazwa użytego wyniku
  lub klatki. Można jawnie wybrać pasujący wynik, klatkę uczestniczącą lub **Pseudokolory
  głębi**; brak dostępnej tekstury jest sygnalizowany. Podgląd 3D i tekstura PNG modelu OBJ
  korzystają z tego samego obrazu. Mapa 2D zachowuje własną paletę i nakładkę; zmiana palety
  lub Gammy nie zmienia kolorów zdjęcia. Tylko jawnie wybrana tekstura pseudokolorowa
  reaguje na paletę, Gammę i odwrócenie. Nieruchomy podgląd i eksport korzystają z tego
  samego próbkowanego pola wysokości bez dodatkowego ukrytego wygładzania. Względny relief
  to wizualizacja w jednostkach obrazu.

Obliczenia działają w tle z postępem i możliwością anulowania; metoda AI
wykorzystuje to samo bezpieczne pamięciowo kafelkowanie co fuzja AI, więc
głębokie stosy też działają. Odpowiednik w CLI to `--depth-map [path]` z
`--depth-colormap`.

### Automatyczne odzyskiwanie sesji

OpenFocus stale utrzymuje migawkę sesji roboczej (stos, ustawienia, etykiety,
bieżąca klatka). Jeśli aplikacja zakończy się nieoczekiwanie — awaria, utrata
zasilania lub wymuszone zamknięcie — następne uruchomienie to wykryje i
zaproponuje **Przywróć sesję**, aby wrócić dokładnie do miejsca, w którym
byłeś. Czyste zamknięcia nigdy nie pokazują tego pytania. Dane odzyskiwania
leżą obok ustawień (instalacje przenośne trzymają je przy pliku wykonywalnym,
więc podróżują razem z programem).

### Ostatnie pliki

Każdy otwarty stos jest zapamiętywany w **Plik → Otwórz ostatnie** (do 8 pozycji). Naciśnij `Alt+1`–`Alt+8`, aby otworzyć pozycję bezpośrednio. Ponowne otwarcie tego samego stosu przenosi go na górę; pozycje, których pliki już nie istnieją, można usunąć po potwierdzeniu, a **Wyczyść listę ostatnich** opróżnia menu.

### Przeciąganie i upuszczanie

Możesz też przeciągnąć folder bezpośrednio na okno aplikacji, aby zaimportować obrazy. Upuszczenie kolejnych plików na istniejący stos dołącza je (po sprawdzeniu rozmiaru).

### Orientacja EXIF

Pliki JPEG/TIFF z aparatów ze znacznikiem EXIF Orientation są automatycznie
obracane i odbijane podczas importu, więc zdjęcia pionowe wczytują się
prawidłowo.

### Źródła 16-bitowe

Wejściowe pliki PNG/TIFF 16-bitowe są wykrywane automatycznie i zachowywane
na całej ścieżce: fuzja działa w pełnej głębi, a wyniki można zapisać z powrotem
jako 16-bitowe PNG/TIFF (eksporty JPG/BMP są konwertowane do 8 bitów, ponieważ
te formaty nie potrafią zapisać 16 bitów).

Pliki RAW z aparatów są wywoływane do 16 bitów (balans bieli aparatu, gamma
sRGB) z wyłączonym automatycznym rozjaśnianiem, dzięki czemu każda klatka
stosu otrzymuje to samo mapowanie ekspozycji; automatyczna ekspozycja dla
każdej klatki powodowałaby migotanie jasności między klatkami. Zapisz jako
TIFF lub PNG, aby zachować dane 16-bitowe.

### Czyszczenie stosu

Aby usunąć wszystkie wczytane obrazy, wybierz **Plik → Wyczyść stos** lub naciśnij `Ctrl+W`.

---

## 5. Przeglądanie i nawigacja po obrazach

### Nawigacja po stosie obrazów

- **Suwak**: przeciągnij suwak pod panelem źródłowym, aby przewijać klatki
- **Lista plików**: kliknij dowolną nazwę na liście obrazów źródłowych w prawym panelu
- **Klawiatura**: używaj klawiszy strzałek (w lewo/w prawo) do nawigacji

### Powiększanie i przesuwanie

- **Powiększanie/zmniejszanie**: kółko myszy nad obrazem
- **Dokładne powiększanie**: przytrzymaj `Ctrl` + kółko myszy, aby zmieniać w mniejszych krokach
- **Dopasuj do okna**: kliknij dwukrotnie obraz, aby przełączać między dopasowaniem a widokiem 100%
- **Przesuwanie**: kliknij i przeciągnij, aby przesunąć obraz w panelu

### Nawigacja zsynchronizowana

Gdy poruszasz się po panelu źródłowym, panel wyników pokazuje odpowiednią klatkę wyniku, jeśli jest dostępna.

---

## 6. Widok porównania suwakiem

Widok Suwak nakłada dwa obrazy w jednej ramce z przeciąganym pionowym
dzielnikiem, dzięki czemu przesunięcia i różnice jakości są bezpośrednio
widoczne po obu stronach podziału.

### Otwieranie widoku Suwak

1. Wyrenderuj co najmniej jeden wynik (fuzji lub rejestracji)
2. Kliknij przełącznik **Suwak** w prawym górnym rogu paska tytułu panelu wyników
3. Obszar wyników przełączy się na widok porównania; kliknij **Suwak** ponownie, aby wrócić

### Interpretacja widoku

- **Strona A (na lewo od dzielnika)**: klatka źródłowa — domyślnie podąża za suwakiem panelu źródłowego
- **Strona B (na prawo od dzielnika)**: najnowszy render
- **Biały dzielnik + okrągły uchwyt**: przeciągnij poziomo, aby przesunąć podział
- Przeciąganie krawędzi, które przeskakują przez dzielnik, ujawnia błędy rejestracji trudne do zauważenia w panelach obok siebie

### Wybór porównywanych obrazów

Pasek na dole widoku porównania ma po jednej grupie na stronę:

- **Strona A** — zaznacz **Podążaj za źródłem** (domyślnie), aby śledzić suwak panelu źródłowego, albo odznacz, aby przypiąć stronę A do dowolnej klatki jej własnym suwakiem
- **Strona B** — zaznacz **Najnowszy wynik** (domyślnie), aby zawsze pokazywać najnowszy render, albo odznacz, aby przeglądać historię wyników jej suwakiem

Obie strony aktualizują się na żywo: nowe rendery, ruchy suwaka źródła i usunięcia klatek są odzwierciedlane natychmiast.

### Powiększanie i przesuwanie

Widok porównania używa jednego przekształcenia dla obu obrazów:

- **Powiększanie**: kółko myszy (powiększa wokół kursora)
- **Przesuwanie**: kliknij i przeciągnij w dowolnym miejscu poza dzielnikiem
- **Reset**: dwuklik

Tryb Suwak i tryb ROI dzielą panel wyników, więc wejście w jeden kończy drugi. Wyczyszczenie stosu również zamyka tryb Suwak.

---

## 7. Rejestracja obrazów

Rejestracja obrazów koryguje przestrzenne przesunięcia między klatkami stosu. Jest niezbędna, gdy obrazy mają niewielkie przesunięcia lub zmiany perspektywy.

### Metody rejestracji

#### ECC (ulepszony współczynnik korelacji)

- Wykorzystuje optymalizację do znalezienia najlepszego wyrównania na podstawie korelacji
- Odpowiednia do subtelnych przesunięć
- Zapewnia dokładność podpikselową
- Wolniejsza niż homografia, ale dokładniejsza

#### Homografia

- Oparta na dopasowywaniu punktów charakterystycznych (cechy SIFT/ORB)
- Radzi sobie z większymi przekształceniami geometrycznymi
- Szybsza niż ECC przy wstępnym wyrównaniu
- Dobra do obrazów ze znaczącymi zmianami perspektywy

#### OpenFocus (szybko)

- Ograniczone wyrównanie podobieństwa (przesunięcie + obrót + skala),
  z wyrównaniem jasności każdej klatki i automatycznym przycięciem
- Poniżej sekundy na klatkę — znacznie szybciej niż ECC czy homografia
- Klatki przesunięte o więcej niż **8% rozmiaru ramki pozostają niewyrównane**
  zamiast być częściowo poprawione. Próg można zmienić w CLI przez
  `--align-shift-limit`
- Do geometrii nieodkształcalnej lub ekstremalnej wybierz ECC albo homografię

### Włączanie rejestracji

1. W prawym panelu znajdź grupę **Rejestracja**
2. Wybierz jedną metodę z listy rozwijanej: **Bez wyrównania**, **OpenFocus (szybko)**,
   **ECC** lub **Homografia**
3. Szybkie wyrównanie OpenFocus można też łączyć przed ECC lub
   homografią (najpierw zgrubny przebieg globalny, potem precyzyjny przebieg lokalny)

### Kiedy stosować rejestrację

- Obrazy mają widoczne przesunięcia
- Zdjęcia wykonano z ręki
- Scena ma paralaksę (bliższe obiekty przesuwają się między klatkami)
- Wyniki pokazują artefakty duchów lub podwajania

### Ustawienia rejestracji

Dodatkowe ustawienia znajdziesz w **Ustawienia → Ustawienia rejestracji**:
- **Szerokość zmniejszania**: kontrola rozdzielczości wstępnego przetwarzania (domyślnie: 1024px)
- Niższe wartości = szybsze przetwarzanie, potencjalnie niższa dokładność
- Wyższe wartości = wolniejsze przetwarzanie, potencjalnie wyższa dokładność

---

## 8. Metody fuzji obrazów

OpenFocus oferuje osiem algorytmów fuzji (oraz tryb tylko wyrównania). Każdy ma inne właściwości, odpowiednie do różnych typów obrazów.

### Filtr sterowany (domyślnie)

- **Algorytm**: fuzja z zachowaniem krawędzi za pomocą filtrowania sterowanego
- **Najlepszy do**: zastosowań ogólnych, zrównoważona szybkość i jakość
- **Zalety**: dobre zachowanie krawędzi, szybkie przetwarzanie
- **Parametr**: rozmiar jądra (regulowany od 1 do 51, tylko wartości nieparzyste)
  - Mniejsze jądro: więcej szczegółów lokalnych, potencjalnie więcej szumu
  - Większe jądro: gładsze wyniki, lepsze tłumienie szumu

### DCT (dyskretna transformata kosinusowa)

- **Algorytm**: fuzja w dziedzinie częstotliwości z użyciem współczynników DCT
- **Najlepszy do**: obrazów bogatych w tekstury, obrazowania naukowego
- **Zalety**: zachowuje drobne tekstury, matematycznie uzasadniony
- **Parametr**: rozmiar jądra określa okno przetwarzania

### DTCWT (dual-tree complex wavelet transform)

- **Algorytm**: fuzja w dziedzinie falkowej w wielu skalach
- **Najlepszy do**: złożonych scen z wieloma poziomami szczegółów
- **Zalety**: doskonała analiza wieloskalowa, dobra kierunkowość
- **Reguła fuzji**: jedna decyzja na piksel i poziom dla wszystkich orientacji i
  kanałów koloru (N-kierunkowa, ze sprawdzeniem spójności większości); pasmo
  dolnoprzepustowe podąża za najgrubszą decyzją zamiast uśredniać wszystkie klatki
- **Parametr**: poziomy falkowe (1–6, domyślnie 4) w panelu po prawej. Metoda nie
  ma jądra przestrzennego, więc suwak jądra jest dla niej ukryty.

### GFG-FGF (gradient quasi-gaussowski w czterech sąsiedztwach + szybki filtr sterowany)

- **Algorytm**: implementacja Fu i in. (*Laser & Optoelectronics
  Progress*, 2024): mapa ostrości z różnicy rozmyć |I − mean(I)| jest wzbogacana
  operatorem gradientu quasi-gaussowskiego w czterech sąsiedztwach (GFG),
  udokładniana szybkim filtrem sterowanym, decyzja zapada na piksel (remisy
  rozstrzyga miara małego obszaru 3×3 Sobela), a binarne mapy decyzji są
  wygładzane ponownie szybkim filtrem sterowanym przed ważonym mieszaniem.
- **Najlepszy do**: mikroskopii i obrazów z wyraźnymi obszarami ostrości
- **Zalety**: szybki, ostre granice obszarów ostrości; pamięć nie rośnie
  wraz z liczbą klatek
- **Parametr**: jądro = rozmiar filtra średniej zgrubnej mapy ostrości

### StackMFF-V3 (uczenie głębokie)

- **Algorytm**: klasyfikacja źródła dla każdego piksela (wymaga PyTorch).
  Sieć przewiduje dla każdego piksela, z której klatki wejściowej pochodzi, a
  następnie albo miesza klatki według tych prawdopodobieństw, albo wybiera jedną klatkę
- **Wyjście AI** (prawy panel, tylko dla tej metody):
  - **Miękkie mieszanie (ważone)** — domyślnie; każdy piksel to mieszanka
    klatek ważona prawdopodobieństwem. Gładkie przejścia, bez ostrych szwów;
    zgodne z konfiguracją opisaną w artykule
  - **Twardy wybór (jedna klatka)** — każdy piksel jest kopiowany z najbardziej
    prawdopodobnej klatki. Ostrzejsza selekcja bez interpolacji między klatkami;
    szew na poziomie pikseli może pojawić się tam, gdzie dwie klatki remisują
- **Najlepszy do**: najwyższej jakości wyników, gdy dostępny jest GPU
- **Zalety**: niezależność od kolejności (kolejność klatek nigdy nie zmienia
  wyniku), dowolna długość stosu, brak strojenia dla poszczególnych metod,
  automatyczna optymalizacja
- **Wymagania**: instalacja PyTorch, opcjonalny GPU CUDA (bez torchvision)
- **Uwaga**: może być niedostępna, jeśli PyTorch nie jest zainstalowany

### OpenFocus A / B / C

- **Algorytm**: średnia ważona wariancją, wybór z mapy wariancji głębi oraz
  fuzja piramidą laplasjanową
- **OpenFocus A**: średnia ważona wariancją z priorem ostatniej klatki —
  szybka, stabilna kolorystycznie na krótkich stosach
- **OpenFocus B**: wybór z mapy wariancji głębi — wybiera najostrzejszą klatkę
  na piksel; najlepsza na gładkich powierzchniach i w mikroskopii
- **OpenFocus C**: mieszanie pasm piramidą laplasjanową — naturalny wygląd na
  obiektach teksturowanych, wzmacnia kontrast lokalny
- **Parametry**: A i B mają okno wariancji (suwak jądra); C nie ma
  promienia. Wszystkie trzy mają parametr sigma wygładzania.

**Wygładzanie**: jedno pole liczbowe pod suwakiem jądra, którego znaczenie zależy od
wybranej metody (zakres 0–8, krok 0,2):

| Metoda | Etykieta | Domyślnie | Efekt |
|---|---|---|---|
| OpenFocus A | Wygładzanie mieszania (σ) | 2,0 | Wygładzanie gaussowskie mapy wag każdej klatki — jak szeroko mieszane są ostre obszary sąsiednich klatek |
| OpenFocus B | Wygładzanie głębi (σ) | 1,2 | Wygładzanie gaussowskie mapy głębi — szerokość przejścia między klatkami |
| OpenFocus C | Wygładzanie mieszania (σ) | 3,0 | Wygładzanie gaussowskie mapy wag każdego pasma piramidy |

0 zachowuje najostrzejszy wybór na piksel (najdrobniejsze szczegóły, ale plamki na
zaszumionych stosach); większe wartości łagodzą przejścia i aureole. Każda metoda
pamięta własną wartość między sesjami, a przetwarzanie wsadowe stosuje tę samą zasadę.

C nie ma promienia: metoda C w Heliconie udostępnia tylko wygładzanie, więc suwak
jądra jest dla C ukryty, zamiast po cichu nic nie robić.

CLI: `--depth-smoothing SIGMA` (B) oraz `--blend-smoothing SIGMA` (A, C).

**Opcje jasności są ustawiane osobno dla każdej metody.** *Dopasuj ekspozycję przed
fuzją* i *Usuń dryf jasności* są zapamiętywane oddzielnie dla każdej metody fuzji,
ponieważ odpowiadają na pytanie dotyczące konkretnego stosu, a nie na preferencję
globalną. Obie są domyślnie włączone dla OpenFocus A/B/C (które nie mają wbudowanej
kalibracji ekspozycji); pozostałe metody zachowują historyczne wartości domyślne
(dopasowanie ekspozycji włączone, usuwanie dryfu wyłączone). Przełączenie metody
pokazuje jej własne wartości.

### Tylko wyrównanie (bez fuzji)

Wybierz **Tylko wyrównanie (bez fuzji)** jako metodę, aby zarejestrować stos
i wysłać wyrównane klatki na listę wyników bez fuzji — przydatne do sprawdzenia
jakości rejestracji lub wyeksportowania wyrównanych klatek.

### Wybór metody fuzji

1. W prawym panelu znajdź grupę **Fuzja**
2. Wybierz metodę z **listy rozwijanej** (zawsze wybrana jest dokładnie jedna;
   pierwsza pozycja rejestruje bez fuzji)
3. W razie potrzeby dostosuj rozmiar jądra — etykieta pokazuje, co oznacza
   jądro dla wybranej metody
4. Kliknij **Rozpocznij renderowanie**, aby przetworzyć

### Przewodnik porównania algorytmów

| Algorytm | Szybkość | Jakość | Najlepszy do |
|-----------|-------|---------|----------|
| Filtr sterowany | Szybki | Dobra | Zastosowania ogólne |
| DCT | Średnia | Dobra | Tekstury |
| DTCWT | Średnia | Bardzo dobra | Złożone sceny |
| GFG-FGF | Szybki | Dobra | Obszary ostrości |
| Fuzja AI (StackMFF-V3) | Wolna (GPU) | Doskonała | Najwyższa jakość |
| OpenFocus A | Bardzo szybki | Bardzo dobra | Krótkie stosy, stabilność kolorów |
| OpenFocus B | Bardzo szybki | Bardzo dobra | Gładkie powierzchnie, mikroskopia |
| OpenFocus C | Szybki | Bardzo dobra | Obiekty teksturowane, naturalny wygląd |

### Szybki podgląd (szkic renderu)

Strojenie parametrów na dużych stosach jest wolne w pełnej rozdzielczości.
Zaznacz **Szybki podgląd** nad przyciskiem renderowania, a OpenFocus najpierw
wyrenderuje zmniejszony szkic (najdłuższy bok 1200 px) — zwykle kilkakrotnie
szybciej, łącznie z rejestracją. Podgląd jest oznaczony uwagą „Podgląd” w
podsumowaniu paska stanu po renderowaniu i **nie** trafia do historii wyników.
Odznacz pole i wyrenderuj ponownie, aby uzyskać wynik w pełnej rozdzielczości.
Tryb podglądu nie dotyczy renderów ROI i nigdy nie zanieczyszcza pamięci
wyrównania w pełnej rozdzielczości.

---

## 9. Przekształcenia obrazów

### Obrót

1. Wybierz **Edycja → Obróć**
2. Wybierz typ obrotu:
   - **90° w prawo**: obraca wszystkie obrazy o 90 stopni w prawo
   - **90° w lewo**: obraca wszystkie obrazy o 90 stopni w lewo
   - **180°**: obraca obrazy do góry nogami

### Odbijanie

1. Wybierz **Edycja → Odbij**
2. Wybierz typ odbicia:
   - **Odbicie poziome**: lustrzane odbicie od lewej do prawej
   - **Odbicie pionowe**: lustrzane odbicie od góry do dołu

### Zmiana rozmiaru

1. Wybierz **Edycja → Zmień rozmiar**
2. Pojawi się okno z opcjami zmiany rozmiaru
3. Wpisz nowe wymiary lub procent skali
4. Wszystkie obrazy w stosie zostaną proporcjonalnie przeskalowane

### Kolejność przekształceń

Przekształcenia są stosowane do całego stosu obrazów, zachowując wyrównanie między klatkami.

---

## 10. Retusz, etykiety i adnotacje

### Retusz wyniku (Edycja → Retusz wyniku…, `Ctrl+T`)

Fuzja może mimo wszystko zostawić lokalny defekt — aureolę wokół krawędzi o
wysokim kontraście, włos pobrany z niewłaściwej klatki, szum na płaskim obszarze.
Pędzel retuszu kopiuje piksele z klatki źródłowej na wynik:

1. Wyrenderuj wynik (albo wybierz go na liście wyników; działa też kliknięcie
   prawym przyciskiem → *Retusz wyniku…*) i naciśnij `Ctrl+T`.
2. Zamaluj defekt lewym przyciskiem myszy. Przy włączonym **Auto-wyborze
   najostrzejszej klatki na pociągnięcie** (domyślnie) każde pociągnięcie najpierw
   wybiera klatkę o największej szczegółowości pod pędzlem — naprawa aureoli to
   zwykle jedno przeciągnięcie. Wyłącz tę opcję, aby wybierać klatkę samodzielnie
   suwakiem lub klawiszami `,` / `.`.
3. Dostosuj **promień** (`[` / `]`), **twardość** (miękkie krawędzie ukrywają szew)
   oraz **krycie**. Nakładające się pociągnięcia w ramach jednego ruchu nigdy się
   nie kumulują.
4. **Pokaż klatkę źródłową** (`Tab`) pokazuje podgląd klatki, z której malujesz.
   Kółko myszy powiększa, przeciągnięcie prawym/środkowym przyciskiem przesuwa,
   `F` dopasowuje obraz.
5. `Ctrl+Z` / `Ctrl+Y` cofa i ponawia całe pociągnięcia. **Zastosuj jako nowy wynik**
   dodaje retuszowany obraz na listę wyników (z sufiksem `_Retouched`); oryginalny
   wynik pozostaje nietknięty.

Pędzel korzysta z wyrównanych klatek, z których powstał wynik, więc pasuje do
niego; wyniki 16-bitowe pozostają 16-bitowe. Renderów ROI i stosów o zmienionym
rozmiarze nie można retuszować (ich klatki nie odpowiadają już geometrii wyniku).

### Dodawanie etykiet

1. Upewnij się, że masz wyrenderowane wyniki lub wczytane obrazy
2. Wybierz **Edycja → Dodaj etykietę**
3. Wpisz tekst etykiety w oknie dialogowym
4. Etykiety są dodawane zarówno do stosu zarejestrowanego, jak i do stosu wejściowego

### Usuwanie etykiet

- **Usuń etykiety stosu zarejestrowanego**: usuwa etykiety z wyrównanych obrazów
- **Usuń etykiety stosu wejściowego**: usuwa etykiety z obrazów oryginalnych

Opcje te znajdziesz w podmenu **Edycja → Usuń etykietę**.

---

## 11. Ustawienia i konfiguracja

### Ustawienia kafelków (optymalizacja pamięci)

W przypadku dużych obrazów OpenFocus używa przetwarzania kafelkowego, aby uniknąć problemów z pamięcią.

Dostęp przez **Ustawienia → Ustawienia kafelków**:

- **Rozmiar bloku kafelka**: rozmiar każdego kafelka przetwarzania (domyślnie: 1024px)
- **Nakładanie kafelków**: nakładanie się kafelków dla bezszwowego mieszania (domyślnie: 256px)
- **Próg kafelkowania**: rozmiar obrazu, powyżej którego używane są kafelki (domyślnie: 2048px)

**Zalecenia**:
- Mniejszy rozmiar bloku: mniejsze zużycie pamięci, wolniejsze przetwarzanie
- Większy rozmiar bloku: większe zużycie pamięci, szybsze przetwarzanie
- Zwiększ nakładanie, jeśli w wynikach widzisz szwy

### Ustawienia rejestracji

Dostęp przez **Ustawienia → Ustawienia rejestracji**:

- **Szerokość zmniejszania**: rozdzielczość wstępnego przetwarzania dla rejestracji
  - Domyślnie: 1024px
  - Niższa dla szybkości, wyższa dla dokładności

### Informacje o wydajności

Zobacz [PERFORMANCE.md](./PERFORMANCE.md), aby poznać zmierzoną przepustowość i
zużycie pamięci na dużych rzeczywistych stosach oraz praktyczne wskazówki dla
głębokich stosów i danych wejściowych o wysokiej rozdzielczości.

### Ustawienia liczby wątków

Kontrola równoległego przetwarzania na CPU:

Dostęp przez **Ustawienia → Ustawienia wątków**:

- `0` wybiera automatyczne planowanie i jest wartością domyślną nowych ustawień.
- Wartości dodatnie ograniczają liczbę wątków; zapisane limity pozostają zachowane.
- Dostępna moc CPU, tryb zasobów i wolna pamięć mogą zmniejszyć współbieżność.

### Przełącznik akceleracji GPU

OpenFocus używa GPU (CUDA/MPS) tylko dla neuronowego modelu AI (StackMFF-V3); algorytmy klasyczne zawsze działają na CPU.

- Przełącznik w **Ustawienia → Akceleracja GPU**
- Odznaczenie wymusza wszystko na CPU — przydatne, gdy sterowniki GPU sprawiają problemy
- Panel stanu pokazuje `CPU*`, gdy GPU jest wyłączony
- Wybór działa od następnego renderowania i jest zapamiętywany

### Rozmiar partii AI

Dostęp przez **Ustawienia → Rozmiar partii AI**:

- Zapisany limit partii wynosi domyślnie 2; obecnie każde urządzenie przetwarza jeden kafelek na wywołanie.
- Zwiększenie limitu nie włącza równoległych partii GPU.

### Motywy

Dostępne są dwa motywy interfejsu: **Ciemny (domyślnie)** i **Jasny**. Przełączaj w
**Ustawienia → Motyw**; wybór jest stosowany natychmiast (pasek tytułu Windows
też się zmienia) i zapamiętywany. Użytkownicy uruchamiający program po raz pierwszy
wybierają motyw w oknie powitalnym.

### Tryb przenośny

Utwórz pusty plik o nazwie `OpenFocus.portable` obok `OpenFocus.exe`
(wersja przenośna w ZIP zawiera go) — a wszystkie dane użytkownika (ustawienia,
logi, ostatnie pliki) będą przechowywane w folderze `OpenFocusData` obok
aplikacji, a nie w profilu użytkownika, więc instalację można zabrać na
pendrive. Usuń plik znacznika, aby wrócić do normalnego profilu.

### Metadane EXIF

Metadane źródłowe (model aparatu, ekspozycja, orientacja itd.) są zachowywane:
orientacja jest stosowana przy wczytywaniu, a eksporty PNG/TIFF/WebP
automatycznie osadzają oryginalny EXIF. Eksporty JPG/BMP nie mogą zapisać pełnego
EXIF i są zapisywane bez niego.

### Wejście HEIC/HEIF

Zdjęcia z iPhone'a wczytują się bezpośrednio. Wymagany jest opcjonalny pakiet
`pillow-heif` (`pip install pillow-heif`; dołączony do gotowych kompilacji).

### Aktualizacje

Pliki projektu (`.ofproj`) zachowują się jak dokument: pasek tytułu pokazuje
nazwę projektu z `*`, dopóki istnieją niezapisane zmiany, a zamknięcie aplikacji
lub otwarcie innego projektu pyta przed ich odrzuceniem. Projekt przechowuje
listę klatek, skalę wczytywania, wszystkie opcje renderowania, etykiety i bieżącą
klatkę — same obrazy są tylko wskazywane, nie kopiowane.

Pomoc → Sprawdź aktualizacje porównuje Twoją wersję z wydaniami na GitHubie.
**Pobierz i zainstaluj** pobiera nowy instalator w aplikacji (z postępem) i
uruchamia go; Otwórz stronę pobierania prowadzi do listy wydań. Sprawdzanie
odbywa się po cichu najwyżej raz dziennie.

### Pamięć rejestracji

OpenFocus może przechowywać wyniki rejestracji w folderze `.openfocus_cache`
obok stosu źródłowego. Ponowne otwarcie tego samego stosu (przy niezmienionych
plikach) całkowicie pomija wyrównywanie. Przełącznik w **Ustawienia → Pamięć
rejestracji**; nieaktualne wpisy są automatycznie zastępowane.

### Zapamiętywane ustawienia

Wszystkie poniższe ustawienia przetrwają automatycznie restart aplikacji:
liczba wątków, parametry kafelków, szerokość zmniejszania przy rejestracji, rozmiar
partii fuzji AI (StackMFF-V3), przełącznik akceleracji GPU, przełącznik pamięci
rejestracji, język interfejsu, rozmiar okna i układ paneli, format eksportu przez
przeciąganie oraz ostatnio otwierane stosy. Okna ustawień zapisują zmiany w chwili
potwierdzenia; wszystko jest też zapisywane przy zamknięciu aplikacji. Preferencje
są przechowywane osobno dla każdego użytkownika w `OpenFocus/OpenFocus.ini`
(format INI QSettings).

### Język interfejsu

Przełączaj w dowolnym momencie przez **Ustawienia → Język → English / 中文**. Zmiana
działa natychmiast i jest pamiętana w kolejnych sesjach.

### Automatyczne sprawdzanie aktualizacji

Raz dziennie, kilka sekund po uruchomieniu, OpenFocus po cichu sprawdza w GitHubie,
czy jest nowsze wydanie. Jeśli istnieje, nad panelami obrazów pojawia się baner:
wersja, **Zaktualizuj teraz** (pobiera i uruchamia ponownie w jednym kroku),
**Co nowego** (informacje o wydaniu) i przycisk odrzucenia. Odrzucenie ukrywa baner
tylko dla tej wersji — następne wydanie go przywróci, a **Pomoc → Sprawdź
aktualizacje** natychmiast go przywraca.

Gdy pobieranie działa w tle, pasek stanu pokazuje postęp.
Możesz też sprawdzić ręcznie w dowolnym momencie przez **Pomoc → Sprawdź aktualizacje**.

---

## 12. Przetwarzanie wsadowe

Automatyczne przetwarzanie wielu folderów:

1. Wybierz **Narzędzia → Przetwarzanie wsadowe** lub kliknij opcję przetwarzania wsadowego
2. W oknie przetwarzania wsadowego:
   - **Dodaj foldery**: wybierz foldery zawierające sekwencje obrazów
   - **Usuń wybrane**: usuń zaznaczone foldery z listy
   - **Wyczyść wszystko**: usuń wszystkie foldery
3. Skonfiguruj ustawienia wyjściowe:
   - **Utwórz podfolder w folderze źródłowym**: tworzy wyniki w każdym folderze wejściowym
   - **Tak jak folder źródłowy**: nadpisuje lub zapisuje obok danych wejściowych
   - **Wskaż folder wyjściowy**: określa lokalizację wyników
4. Sprawdź bieżące ustawienia fuzji i rejestracji
5. Kliknij **Rozpocznij przetwarzanie wsadowe**, aby zacząć

Okno przetwarzania wsadowego pokazuje postęp w czasie rzeczywistym. Przetwarzanie można anulować w dowolnym momencie.

### Uwagi dotyczące przetwarzania wsadowego

- Używa tych samych ustawień fuzji i rejestracji co przetwarzanie ręczne
- Każdy folder jest przetwarzany niezależnie
- Nazwy plików zawierają znacznik czasu i parametry ułatwiające identyfikację

---

## 13. Eksport wyników

### Zapisywanie pojedynczego wyniku

1. Na liście wyników wybierz wynik, który chcesz zapisać
2. Wybierz **Plik → Zapisz** lub naciśnij `Ctrl+S`
3. Wybierz format i lokalizację
4. Kliknij Zapisz

### Format eksportu przez przeciąganie

Przeciąganie wyników z listy wyników zapisuje pliki obrazów w formacie wybranym
w menu kontekstowym listy wyników (**Format przeciągania**: JPG, PNG lub TIFF).
Wybór jest zapamiętywany.

### Zapisywanie stosu zarejestrowanego

1. Wybierz **Plik → Zapisz stos → Stos zarejestrowany**
2. Wybierz format zapisu:
   - **Zapisz jako folder**: zapisuje każdą wyrównaną klatkę jako osobny plik obrazu
   - **Zapisz jako GIF**: tworzy animowany GIF z wyrównanej sekwencji

### Zapisywanie stosu wejściowego

1. Wybierz **Plik → Zapisz stos → Stos wejściowy**
2. Wybierz format zapisu:
   - **Zapisz jako folder**: zapisuje oryginalne klatki jako osobne obrazy
   - **Zapisz jako GIF**: tworzy animowany GIF z sekwencji wejściowej

### Formaty eksportu

- **JPG**: skompresowany, dobry stosunek jakości do rozmiaru
- **PNG**: bezstratny, obsługuje przezroczystość
- **BMP**: nieskompresowany, maksymalna jakość
- **TIFF**: wysoka jakość, obsługuje warstwy
- **GIF**: format animacji

---

## 14. Interfejs wiersza poleceń

OpenFocus może działać bez GUI — przydatne w skryptowych potokach i CI.
W środowisku Pythona (lub obok zapakowanego pliku wykonywalnego):

```bash
# Fuzja folderu domyślnym filtrem sterowanym
python main.py --input ./stack_folder --output ./result/fused.png

# DTCWT z rejestracją ECC, 8 wątków
python main.py -i ./stack_folder -o ./result/fused.png -m dtcwt -a ecc -t 8

# StackMFF-V3 wymuszony na CPU z własnym rozmiarem kafelka
python main.py -i ./stack_folder -o ./result/fused.png -m stackmffv3 --cpu --tile-size 512

# Jawna lista plików; pliki wideo są również akceptowane jako wejście
python main.py -i img1.jpg img2.jpg img3.jpg -o fused.png -m gfgfgf

# Pojedynczy wielostronicowy stos Z TIFF z paskiem skali kalibrowanym z metadanych
python main.py -i ./zstack.tif -o fused.png -m dtcwt --scale-bar
```

| Opcja | Opis |
|--------|-------------|
| `--input, -i` | Folder z obrazami, plik wideo lub jeden bądź więcej plików obrazów |
| `--output, -o` | Ścieżka obrazu wyjściowego (rozszerzenie ustala format) |
| `--method, -m` | `stackmffv3` (domyślnie), `none`, `openfocus_a`, `openfocus_b`, `openfocus_c`, `guided_filter`, `dct`, `dtcwt`, `gfgfgf` |
| `--align, -a` | `none` (domyślnie), `homography`, `ecc`, `both`, `openfocus` |
| `--align-shift-limit` | tylko wyrównanie `openfocus`: maksymalne akceptowane przesunięcie jako ułamek rozmiaru ramki (domyślnie 0.08; klatki powyżej pozostają niewyrównane) |
| `--exposure-match` | dopasowanie średniej każdej klatki do ostatniej klatki przed fuzją (domyślnie włączone; `--no-exposure-match` wyłącza). Wzmocnienia dla całego stosu, więc wynik kafelkowy pozostaje bezszwowy; dotyczy każdej metody i referencji jasności wyrównywacza OpenFocus |
| `--base-frame` | opcjonalny etap po fuzji: dopasowanie niskich częstotliwości wyniku do klatki `last`/`first` (domyślnie `none`) |
| `--sort` | kolejność klatek dla `openfocus_a/b/c`: `asc` (domyślnie, kolejność plików), `desc` (odwrócenie przemiatania od dalekich do bliskich), `off` |
| `--dust-map WHITE` | korekcja pola płaskiego (mapa pyłu) na podstawie rozogniskowanego zdjęcia białego pola |
| `--kernel, -k` | rozmiar jądra (wymuszany na nieparzysty; dla rodziny openfocus jest źródłem promienia okna wariancji) |
| `--depth-smoothing SIGMA` | tylko `openfocus_b`: sigma gaussowska przejścia głębi między klatkami (domyślnie 1.2; 0 zachowuje najdrobniejsze szczegóły) |
| `--blend-smoothing SIGMA` | tylko `openfocus_a` / `openfocus_c`: sigma gaussowska mapy wag każdej klatki (domyślnie 2.0 dla A, 3.0 dla C; 0 zachowuje najostrzejszy wybór na piksel) |
| `--threads, -t` | Limit wątków; `0` = automatycznie (domyślnie), wartości dodatnie ograniczają współbieżność |
| `--resource-preference` | Tryb zasobów: `balanced` (domyślnie), `responsive` lub `maximum` |
| `--batch-size` | Limit partii kafelków AI (domyślnie 2); obecnie jeden kafelek na wywołanie |
| `--cpu` | wymusza CPU nawet jeśli GPU jest dostępny |
| `--downscale` | szerokość zmniejszania przy detekcji cech rejestracji |
| `--tile-size` | rozmiar bloku kafelka dla dużych obrazów |
| `--output-dir, -d` | tryb wsadowy: fuzja każdego folderu wejściowego z zapisem `<folder>.png` na stos |
| `--scale-bar` | rysuje pasek skali (kalibracja wykrywana automatycznie z metadanych TIFF) |
| `--depth-map [path]` | dodatkowo zapisuje kolorową mapę położenia ostrości (głębi) (domyślnie: `<output>_depth.png`) |
| `--depth-colormap` | mapa kolorów dla `--depth-map`: turbo, viridis, jet, gray, … (domyślnie turbo) |
| `--scale-um` | rozmiar piksela w µm/px, zastępujący metadane (implikuje `--scale-bar`) |

Kody wyjścia: `0` sukces, `1` błąd przetwarzania, `2` błąd użycia. Uruchomienie
`python main.py --help` wypisuje wszystkie opcje; uruchomienie ze zwykłą ścieżką
(`python main.py C:\stack`) nadal otwiera GUI z tym folderem.

---

## 15. Skróty klawiaturowe

### Operacje na plikach

| Skrót | Działanie |
|----------|--------|
| `Ctrl+O` | Otwórz folder |
| `Ctrl+Shift+O` | Otwórz wideo |
| `Ctrl+S` | Zapisz wynik |
| `Ctrl+Shift+S` | Zapisz stos zarejestrowany |
| `Ctrl+W` | Wyczyść stos |
| `Ctrl+Q` | Zakończ |
| `Ctrl+R` | Rozpocznij renderowanie |
| `Ctrl+Shift+R` | Porównaj wszystkie metody |
| `Ctrl+T` | Retusz wyniku |
| `Ctrl+M` | Mapa głębi |
| `Ctrl+E` | Przełącz porównanie suwakiem |
| `Ctrl+Shift+B` | Przetwarzanie wsadowe |
| `Ctrl+Z` / `Ctrl+Y` | Cofnij / Ponów (edycje stosu) |

### Nawigacja

| Skrót | Działanie |
|----------|--------|
| `Left Arrow` | Poprzednia klatka |
| `Right Arrow` | Następna klatka |
| `Mouse Wheel` | Powiększ/zmniejsz |
| `Ctrl + Wheel` | Dokładne powiększanie |
| `Double-click` | Przełącz widok dopasowany / 100% |

### Ogólne

| Skrót | Działanie |
|----------|--------|
| `Space` | Przełącz tryb przesuwania |
| `Delete` | Usuń zaznaczony element |

---

## 16. Rozwiązywanie problemów

### Typowe problemy

#### „Brak wczytanych obrazów”

- Upewnij się, że najpierw zaimportowano folder z obrazami lub plik wideo
- Sprawdź, czy folder zawiera obsługiwane formaty obrazów

#### Wyniki fuzji są rozmyte lub mają duchy

- Włącz rejestrację obrazów (wybierz jedną z listy: Homografia / ECC / OpenFocus szybko)
- Upewnij się, że obrazy są dobrze wyrównane przed fuzją
- Wypróbuj inny algorytm fuzji
- Zwiększ rozmiar jądra, aby uzyskać gładsze wyniki

#### AI niedostępne

- Zainstaluj PyTorch: `pip install torch torchvision`
- Aby uzyskać obsługę GPU, zainstaluj PyTorch z CUDA
- Sprawdź w **Pomoc → Informacje o środowisku**

#### Błędy braku pamięci

- Włącz przetwarzanie kafelkowe w **Ustawienia → Ustawienia kafelków**
- Zmniejsz rozmiar bloku kafelka
- Użyj mniejszych rozmiarów jąder
- Zmniejsz rozdzielczość obrazów

#### Rejestracja nie udaje się

- Używaj tylko jednej metody rejestracji naraz
- Zmniejsz szerokość zmniejszania w **Ustawienia → Ustawienia rejestracji**
- Upewnij się, że obrazy mają wystarczający kontrast i cechy

#### Wolne przetwarzanie

- Zwiększ liczbę wątków w **Ustawienia → Ustawienia wątków**
- Użyj mocniejszego CPU
- Zmniejsz rozdzielczość obrazów
- Użyj prostszych algorytmów fuzji (filtr sterowany zamiast fuzji AI)

#### Ostrzeżenie Windows SmartScreen przy pierwszym uruchomieniu

Wydane pliki binarne nie są podpisane. Kliknij **Więcej informacji → Uruchom mimo to**.
Pierwsze uruchomienie po rozpakowaniu może potrwać 30–60 sekund.

#### macOS zgłasza, że aplikacji „nie można otworzyć” lub że „jest uszkodzona”

Aplikacja nie jest podpisana ani notaryzowana. Po przeniesieniu `OpenFocus.app` do
folderu Aplikacje uruchom:

```bash
xattr -cr /Applications/OpenFocus.app
```

albo kliknij aplikację prawym przyciskiem i wybierz **Otwórz**. Wydane kompilacje
dla macOS są przeznaczone na Apple Silicon (M1/M2/M3/M4); komputery Mac z Intelem
mogą uruchamiać program ze źródeł.

### Uzyskiwanie pomocy

1. **Informacje o środowisku**: przejdź do **Pomoc → Informacje o środowisku**, aby sprawdzić instalację
2. **Kontakt z pomocą techniczną**: przejdź do **Pomoc → Kontakt**, aby uzyskać informacje o wsparciu

---

## Dodatek: szczegóły algorytmów fuzji

### Filtr sterowany

Filtr sterowany wykonuje wygładzanie z zachowaniem krawędzi z użyciem obrazu
wskazującego. Oblicza lokalne przekształcenia liniowe między obrazem wskazującym
a wejściowym, a następnie stosuje je do utworzenia wyniku. Metoda ta świetnie
zachowuje krawędzie, jednocześnie redukując szum.

### DCT (dyskretna transformata kosinusowa)

DCT przekształca obrazy do dziedziny częstotliwości, w której decyzje fuzji są
podejmowane na podstawie amplitud współczynników. Składowe wysokoczęstotliwościowe
(szczegóły) są zachowywane, a składowe niskoczęstotliwościowe (obszary gładkie)
są łączone na podstawie energii lokalnej.

### DTCWT (dual-tree complex wavelet transform)

DTCWT zapewnia wieloskalowy i wielokierunkowy rozkład obrazów. Wychwytuje cechy w
różnych skalach i orientacjach, umożliwiając zaawansowane reguły fuzji, które
zachowują zarówno informacje o teksturze, jak i o strukturze.

### GFG-FGF

Implementuje Fu i in. (2024): zgrubna mapa ostrości |I − mean(I)| jest wzbogacana
gradientem quasi-gaussowskim w czterech sąsiedztwach — szablon 3×3 T = [1 2 1; 2 4 2; 1 2 1]/16 daje P = I⊗T, a G = (|I(x±1,y) − P|)/2 + (|I(x,y±1) − P|)/2, przy czym wartości poniżej T0 = 0.005 są odrzucane — a następnie udokładniana szybkim filtrem sterowanym. Piksele wybierają klatkę o największej wartości ostrości (remisy: miara małego obszaru 3×3 Sobela), a binarne mapy decyzji są wygładzane szybkim filtrem sterowanym przed ważonym mieszaniem.

### Fuzja AI (StackMFF-V3)

Podejście oparte na uczeniu głębokim, wykorzystujące sieć neuronową trenowaną na
dużych zbiorach par obrazów o wielu punktach ostrości. Sieć sama uczy się optymalnych
strategii fuzji, dając wyniki najnowszej generacji przy prawidłowej konfiguracji.

---

*OpenFocus to projekt open source. Aktualizacje, zgłoszenia błędów i propozycje funkcji znajdziesz w repozytorium projektu.*
