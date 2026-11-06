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

OpenFocus to profesjonalna aplikacja desktopowa do wieloogniskowej fuzji obrazów, przeznaczona dla badaczy, inżynierów i specjalistów od przetwarzania obrazów. Program przetwarza sekwencje zdjęć o różnych płaszczyznach ostrości, stosując nowoczesne algorytmy rejestracji i fuzji, i tworzy w pełni ostry obraz o dużej głębi ostrości.

### Najważniejsze funkcje

- **Wieloogniskowa fuzja obrazów**: wiele zdjęć o różnych punktach ostrości łączy się w jeden w pełni ostry obraz
- **Dziesięć algorytmów fuzji**: StackMFF V3 (domyślnie), StackMFF V2, StackMFF, Guided Filter, DCT, DTCWT, GFG-FGF oraz OpenFocus A/B/C, a także tryb tylko wyrównania
- **Rejestracja obrazów**: ECC, homografia, homografia → ECC i szybkie wyrównanie OpenFocus
- **Automatyczne tło**: zastąpienie niespokojnego tła ze stacku jedną prawdziwą klatką oraz pokazanie, gdzie żadna klatka nie uchwyciła sceny wyraźnie
- **Porównanie suwakiem**: kontrola wyrównania i wyniku za pomocą przeciąganego dzielnika A/B
- **Porównanie metod**: kilka algorytmów renderowanych obok siebie na jednym fragmencie; zwycięzcę można przyjąć jako nowy wynik
- **Retusz powiązany ze źródłami**: pędzel kopiuje piksele z klatek, które naprawdę utworzyły wynik; eksport TIFF z warstwami zapisuje wynik razem ze źródłami
- **Pliki projektów**: odwołania do źródeł i odtwarzalne ustawienia; piksele wyniku eksportuje się osobno
- **Szablony przetwarzania**: zapisywanie i ponowne stosowanie ustawień fuzji, rejestracji i zasobów do dowolnego wczytanego stosu
- **Wielostronicowy TIFF**: stosy Z z jednego pliku wczytują się strona po stronie i są zapisywane z powrotem do jednego pliku
- **Mapa głębi**: mapy położenia ostrości dla każdego piksela z własnymi schematami pseudokolorów i eksportem 3D
- **Pasek skali**: paski jakości publikacyjnej z automatyczną kalibracją µm/px
- **Odzyskiwanie po awarii**: po awarii proponowane jest przywrócenie ostatniej sesji roboczej
- **Dwanaście języków interfejsu**: English, 中文, 日本語, Español, Deutsch, Français, Русский, 한국어, Português, Italiano, Nederlands, Polski — z automatycznym wykrywaniem języka systemu
- **Headless CLI**: sterowanie OpenFocus z wiersza poleceń do automatyzacji i CI, w tym przepisy i tryb wsadowy
- **Elastyczny eksport**: pojedyncze obrazy, foldery, GIF, wielostronicowy TIFF
- **Przekształcenia obrazów**: obrót, odbicie i zmiana rozmiaru stosów
- **Etykiety i adnotacje**: zarządzana lista etykiet z obszarami docelowymi i zakresami klatek
- **Zapamiętywane ustawienia**: preferencje, język i ostatnie stosy przetrwają restart
- **Sterowanie przez agenta**: lokalny kanał chroniony tokenem pozwala agentowi sterować działającym interfejsem

### Obsługiwane formaty wejściowe

- Pliki obrazów: JPG/JPEG, PNG, BMP, TIFF, WebP i inne popularne formaty
- Pliki wideo: MP4, AVI, MOV, MKV, WMV, FLV, WEBM (automatycznie dzielone na klatki)
- Foldery z numerowanymi sekwencjami obrazów
- Pliki RAW aparatów (DNG, CR2/CR3, NEF, ARW, RAF, ORF, RW2 i inne popularne formaty). Dekodowanie używa `rawpy` (LibRaw); `pip install openfocus` instaluje go, a kompilacje dla Windows i macOS dołączają go do zestawu

Stosy przezroczystych PNG można łączyć w aplikacji desktopowej, przetwarzaniu wsadowym lub wierszu poleceń, ale nie w trybie grupowym. Każda klatka musi być 8- lub 16-bitowym PNG o tych samych wymiarach i głębi, z tą samą maską alfa. Maska musi być zerowa poza jednym niepustym prostokątem i mieć stałą, niezerową nieprzezroczystość wewnątrz niego. Dopuszczalna jest częściowa przezroczystość o stałej nieprzezroczystości. Widoczny prostokąt jest łączony niezależnie, więc ukryte kolory nie wpływają na pomiar ostrości ani granice kafelków. Wyjściowy PNG zachowuje rozmiar płótna i nieprzezroczystość; ukryte kolory wyniku są zerowane. Widoki źródła i wyniku w aplikacji pokazują przezroczystość na szachownicy.

Wczytaj w pełnym rozmiarze, wybierz brak wyrównania, wyłącz szybki podgląd i ROI, a następnie wyrenderuj i zapisz jako PNG. Przetwarzanie wsadowe wymaga metody fuzji i wyjścia PNG bez zapisywania wyrównanego stosu. Wyrównanie, zmiana rozmiaru i skalowanie w dół, korekcja pyłu, mapy głębi, porównanie, retusz, etykiety i paski skali nie są obsługiwane dla danych przezroczystych. Metoda, która zmienia wymiary prostokąta, jest odrzucana. Zmienne lub rozmyte maski, różne maski klatek, przezroczysty TIFF na wejściu i wyjście inne niż PNG, w tym GIF i wielostronicowy TIFF, nie są obsługiwane. CLI odrzuca również czyszczenie matrycy i mapy głębi dla przezroczystego wejścia.

Odpowiednik w CLI: `python main.py --input ./png-stack --output ./result.png --method guided_filter --align none`.

### Obsługiwane formaty wyjściowe

W oknie **Zapisz jako** zwykły TIFF zapisuje wynik jako jeden obraz, a TIFF z warstwami zawiera klatki źródłowe jako warstwy. Linear DNG jest zawsze widoczny, ale jego zapis wymaga liniowego wyniku kamery bez utrwalonych zmian, pochodzącego z **Otwórz natywny RAW** i mającego kalibrację kamery.

- Pojedyncze obrazy: JPG, PNG, BMP, TIFF, WebP
- Sekwencje obrazów: zapisywane w folderach
- Animacje: GIF
- Wielostronicowy TIFF: cały stos w jednym `.tif`
- TIFF z warstwami: wynik retuszu razem z klatkami źródłowymi (wybierany w oknie Zapisz jako)

---

## 2. Pierwsze kroki

### Uruchamianie aplikacji

Pobierz najnowszą kompilację ze [strony OpenFocus](https://openfocus.top/#download) lub z [GitHub Releases](https://github.com/Xinzhe99/OpenFocus/releases). Strona jest preferowana: publikuje bieżący instalator Windows, kompilację macOS i pakiet Linux ze skryptami instalacyjnymi; te artefakty są dokładnym wynikiem potoku wydania, więc wybieraj je zamiast ponownej kompilacji. `pip install openfocus` instaluje tylko headless CLI (koło celowo nie zależy od PyQt6); do interfejsu graficznego użyj gotowej kompilacji albo kodu źródłowego. Po instalacji uruchom OpenFocus na jeden z sposobów:

1. **Windows (instalator lub wersja przenośna)**: kliknij dwukrotnie skrót lub `OpenFocus.exe`
2. **macOS**: przeciągnij `OpenFocus.app` do Programów i uruchom; kompilacje są przeznaczone dla Apple Silicon
3. **Kod źródłowy Pythona**: zainstaluj zależności poleceniem `pip install -r requirements.txt`, a następnie uruchom `python main.py` w aktywowanym środowisku

Instalacja z terminala:

```bash
# macOS lub Linux: najpierw sprawdź plan, potem uruchom bez --dry-run
curl -fsSL https://openfocus.top/install.sh | sh -s -- --dry-run
curl -fsSL https://openfocus.top/install.sh | sh
```

```powershell
# Windows: instalator z PowerShell
irm https://openfocus.top/install.ps1 | iex
```

Skrypt `install.sh` instaluje kompilację macOS w `~/Applications`, a w systemie Linux — headless CLI. Podpis kompilacji Windows zależy od konkretnego wydania (zobacz informacje o wydaniu); podpis nie jest gwarantowany. W systemie Windows przy ostrzeżeniu SmartScreen wybierz „Więcej informacji → Uruchom mimo to”. Kompilacje macOS nie są notaryzowane: po instalacji z DMG (do Programów) w razie potrzeby wykonaj `xattr -cr /Applications/OpenFocus.app`, po instalacji z terminala (do `~/Applications`) — `xattr -cr ~/Applications/OpenFocus.app`, albo kliknij aplikację prawym przyciskiem i wybierz „Otwórz”.

Przy pierwszym uruchomieniu otwiera się okno powitalne, w którym można wybrać motyw i wczytać stos demonstracyjny. Domyślnie używany jest jasny motyw; ciemny wybiera się w oknie powitalnym lub później w **Ustawienia → Motyw**. Wybór jest pamiętany.

### Szybki start

1. Zaimportuj sekwencję obrazów (**Plik → Otwórz folder** lub przeciągnij)
2. Wybierz metodę fuzji na prawym panelu; domyślnie wybrany jest **StackMFF V3**
3. Wybierz metodę rejestracji; domyślnie wybrany jest **ECC**. Jeśli stos jest już wyrównany, wybierz **Bez wyrównania**
4. W razie potrzeby zmień rozmiar jądra lub parametry wybranej metody
5. *(opcjonalnie)* Otwórz **Ustawienia renderowania** i zaznacz **Szybki podgląd**, aby najpierw dobrać parametry na zmniejszonym szkicu
6. Kliknij **Rozpocznij renderowanie**
7. Wyeksportuj wynik

---

## 3. Przegląd głównego interfejsu

Interfejs OpenFocus składa się z trzech głównych obszarów.

### Lewy panel: obszar wyświetlania obrazów

W lewej części okna obrazy są pokazane w podzielonym widoku:

- **Panel obrazu źródłowego (góra)**: wejściowa sekwencja klatek
- **Panel wyniku (dół)**: wynik fuzji, rejestracji lub retuszu
- **Suwak nawigacji**: przewijanie klatek pod każdym panelem
- **Etykiety informacyjne**: numer bieżącej klatki i wymiary obrazu

Rozmiar paneli zmienia się przez przeciąganie dzielnika między nimi.

### Prawy panel: panel sterowania

Prawy panel zawiera parametry i listy:

1. **Rejestracja**: lista rozwijana metod i parametry wybranej metody
2. **Fuzja**: lista rozwijana metod, parametry i tryb wyjścia AI
3. **Ustawienia renderowania**: okno zawiera szybki podgląd, odwróconą kolejność klatek, dopasowanie ekspozycji, usuwanie dryfu jasności i wyostrzanie wyniku
4. **Przyciski akcji**: rozpoczęcie renderowania oraz przywrócenie domyślnych, ustawienia renderowania i renderowanie wszystkimi metodami
5. **Lista obrazów źródłowych**: wczytane klatki, pola wyboru i przycisk **Sprawdź klatki**
6. **Lista wyników**: utworzone wyniki i ich menu kontekstowe

### Pasek menu

- **Plik**: otwieranie folderu, wideo i projektu, zapisywanie projektu i szablonów przetwarzania, ostatnie pliki i projekty, zapisywanie wyniku i stosu, czyszczenie stosu, wyjście
- **Edycja**: cofanie i ponawianie, obrót, odbicie, zmiana rozmiaru, retusz wyniku i retusz plików obrazów
- **Narzędzia**: przetwarzanie wsadowe, porównanie metod, mapa głębi, mapa pyłu, pasek skali, etykiety
- **Ustawienia**: ustawienia zaawansowane (rejestracja i diagnostyka wyrównania), akceleracja GPU, motyw, język
- **Pomoc**: informacje o środowisku, opinia, kontakt, przewodnik szybkiego startu, sprawdzanie aktualizacji, instalacja gotowej aktualizacji i folder logów

---

## 4. Importowanie obrazów i wideo

### Otwieranie folderu z obrazami

1. Wybierz **Plik → Otwórz folder** lub naciśnij `Ctrl+O`
2. Wskaż folder z sekwencją obrazów
3. W oknie skali wczytywania wybierz skalę (100 % — pełna rozdzielczość)
4. Kliknij OK, aby wczytać obrazy

Zaznacz w tym oknie **Nie pytaj ponownie**, aby w przyszłości wczytywać stosy w wybranej skali bez pytania; **Ustawienia → Ustawienia zaawansowane → Uruchamianie i aktualizacje → Pytaj o skalę wczytywania przy imporcie** przywraca pytanie. Wczytany stos można później zmniejszyć poleceniem **Edycja → Zmień rozmiar**.

Program automatycznie:
- Wykryje i posortuje pliki obrazów według nazw
- Pokaże pierwszą klatkę w panelu źródła
- Wypełni listę obrazów źródłowych wszystkimi klatkami

### Otwieranie plików wideo

1. Wybierz **Plik → Otwórz wideo** lub naciśnij `Ctrl+Shift+O`
2. Wskaż plik wideo (MP4, AVI, MOV, MKV, WMV, FLV, WEBM)
3. Wideo zostanie automatycznie podzielone na osobne klatki
4. Każda klatka staje się częścią stosu obrazów

### Pliki projektów

**Plik → Zapisz projekt** zapisuje w `.ofproj` odwołania do źródeł i odtwarzalne ustawienia: dokładną kolejność klatek (w tym strony TIFF i powtórzone klatki), wykluczenia, obroty i odbicia, skalę wczytywania, parametry renderowania dla każdej metody, etykiety, styl paska skali i bieżącą klatkę. Piksele wyniku i retuszu trzeba wyeksportować osobno. Operacje, których nie można odtworzyć z plików źródłowych, wymagają wcześniejszego zapisania nowego stosu wejściowego.

**Plik → Otwórz projekt** odtwarza zapisane klatki, zanim zastąpi bieżący stos. Jeśli źródła zostały przeniesione, wybierz nowy katalog główny i potwierdź pełne mapowanie starych i nowych ścieżek. Zapisz projekt ponownie, aby utrwalić nowe odwołania.

**Plik → Zapisz projekt jako** zapisuje bieżącą sesję w nowym pliku. Pasek tytułu pokazuje nazwę projektu z symbolem `*`, dopóki są nie zapisane zmiany; przy zamykaniu aplikacji lub otwieraniu innego projektu program pyta o potwierdzenie. Projekt nie przechowuje pikseli wyników i nie zdejmuje z nich ochrony przed nadpisaniem. Obrazy źródłowe są przywoływane, a nie kopiowane.

### Szablony przetwarzania

Szablon przetwarzania przenosi wyłącznie ustawienia, nigdy ścieżki do źródeł. **Plik → Zapisz szablon przetwarzania…** zapisuje `.oftemplate` z bieżącą metodą i parametrami fuzji, metodą i parametrami rejestracji, opcjami jasności dla każdej metody, liczbą wątków, ustawieniami kafelków, szerokością skalowania rejestracji, rozmiarem partii AI StackMFF, wyostrzaniem, odwrotną kolejnością klatek i flagą odszumiania matrycy. Ścieżka maski defektów celowo nie jest przechowywana.

**Plik → Zastosuj szablon przetwarzania…** stosuje takie ustawienia do już wczytanego stosu; nie zastępuje źródeł. Wybór przyjmuje też projekt `.ofproj` (ustawienia są wyodrębniane bez jego źródeł) oraz przepis `.json` zawierający ustawienia projektu. Zastosowanie szablonu oznacza projekt jako zmieniony i może zmienić parametry algorytmów oraz zasobów, więc sprawdź prawy panel przed renderowaniem. Przepisy pozostają formatem headless i rzeczywiście zawierają ścieżkę maski defektów oraz jej odcisk.

### Stos demonstracyjny

W oknie powitalnym jest przycisk **Wczytaj stos demonstracyjny**: sześć dołączonych zdjęć z prawdziwego stosu ostrości, dzięki czemu można od razu wypróbować cały przepływ bez własnych obrazów. Przydaje się też do sprawdzenia instalacji.

### Krzywa ostrości

Niewielka krzywa nad suwakiem stosu źródłowego pokazuje ostrość każdej klatki (znormalizowaną). Klatki nieostre widać jako doliny oznaczone czerwoną kropką. Kliknij w dowolnym miejscu krzywej, aby przejść do tej klatki; po najechaniu pokazuje się numer klatki.

### Stosy wielostronicowych plików TIFF

Programy mikroskopowe (ZEN, NIS-Elements, ImageJ/Fiji) często eksportują cały stos Z jako JEDEN wielostronicowy plik TIFF. OpenFocus czyta każdą stronę takiego pliku jako osobną klatkę stosu — przeciągnij lub otwórz plik, a stos pojawi się w całości (klatki nazywają się `<file>_page_001` itd.). Jednostronicowe pliki TIFF działają jak wcześniej. Odwrotnie również: **Plik → Zapisz stos → (Zarejestrowany / Wejściowy) → Zapisz jako wielostronicowy TIFF** zapisuje cały stos z powrotem do jednego `.tif` z zachowaniem głębi 16 bitów.

### Pasek skali (ryciny publikacyjne)

Jeśli wybrany cel nie ma użytecznych metadanych, domyślnie wybierana jest kalibracja ręczna i wymagany jest prawidłowy rozmiar piksela.

**Narzędzia → Pasek skali…** konfiguruje pasek skali wpalany w eksportowane obrazy. Rozmiar piksela (µm/px) jest wykrywany automatycznie z metadanych TIFF — konwencja ImageJ `unit=micron`, `PhysicalSizeX` w OME-TIFF lub jawnie podana jednostka rozdzielczości w centymetrach — więc stosy mikroskopowe (w tym wielostronicowe TIFF) kalibrują się same. Rozdzielczość drukarki (72/300 dpi i podobne) celowo *nie* jest traktowana jako kalibracja, a wartość wpisana ręcznie zawsze wygrywa z wykrytymi metadanymi: prawdopodobnie wyglądający, ale błędny pasek na rycinie jest gorszy niż jego brak. Wybierz narożnik i kolor (tryb „Auto” odwraca kolor względem tła); okno pokazuje podgląd na żywo. Przy włączonym pasku każdy eksport (pojedynczy obraz, stosy do folderu, wielostronicowy TIFF, GIF, eksport wszystkich wyników, przetwarzanie wsadowe) otrzymuje pasek w stylu publikacyjnym: zaokrągloną długość z szeregu 1-2-5 (około 12 % szerokości obrazu) i obrysowany, czytelny tekst. W CLI jest dostępny przez `--scale-bar` (automatyczna kalibracja) i `--scale-um <um/px>`.

Podgląd jest rysowany w natywnej geometrii pikseli celu, a następnie dopasowywany do okna. „Dopasuj”, 100 %, przesuwanie i przewijanie klatek zmieniają tylko widok. **Zastosuj** dodaje jedną operację do cofnięcia; **Anuluj** zachowuje poprzednią kalibrację. Historyczne wyniki zachowują własną kalibrację i styl po wczytaniu innego stosu źródłowego.

Pasek można usunąć przez **Narzędzia → Pasek skali → Usuń pasek skali**: pozycje *Ze stosu źródłowego* i *Z wyniku* zawężają zakres „Zastosuj do” do pozostałego płótna, a *Ze wszystkich* wyłącza pasek. Oba płótna i porównanie suwakiem odświeżają się natychmiast; usunięcie z płótna, które nigdy nie miało paska, nie zmienia ustawienia.

### Mapa głębi (położenie ostrości)

**Narzędzia → Mapa głębi…** szacuje położenia ostrości według miary aktywności wybranej metody. To nie jest zmierzona powierzchnia ani dokładna mapa pochodzenia pikseli: metody piramidalne i falkowe szacują aktywność, a AI podaje oczekiwane położenie klatki. Wybierz bieżące zaznaczone klatki albo zapisane źródło wybranego wyniku. Brakujące informacje o pochodzeniu nigdy nie są zastępowane bieżącym stosem. Źródło i tekstura pozostają powiązane z tym migawkowym stanem. Mapy głębi są niedostępne dla stosów przezroczystych PNG — zarówno w GUI, jak i w CLI.

Wszystkie siedem metod — DCT, DTCWT, GFG-FGF, OpenFocus B, OpenFocus C, StackMFF V2 i StackMFF V3 — wymagają jawnego kliknięcia **Oblicz**. StackMFF V1 nie ma mapy głębi. Otwarcie okna lub zmiana miary, źródła, rejestracji bądź parametrów nigdy nie uruchamia obliczeń. Edytor parametrów znajduje się bezpośrednio pod wyborem miary: jądro medianowe DCT i początkowe jądro GFG przyjmują nieparzyste wartości 1–51 (domyślnie 7). OpenFocus B używa promienia odpowiedzi ostrości i wygładzania głębi; OpenFocus C używa **Wygładzania mieszania** do wygładzania mapy głębi, bez promienia. Poziomy rozkładu DTCWT są edytowane tutaj jako lokalny szkic inicjowany parametrami źródła. StackMFF V3 pokazuje własny tryb wyjścia AI (domyślnie miękkie mieszanie lub twardy wybór) i wyjaśnienie zamiast numerycznego jądra: mapa podaje ciągłe położenie klatki, a nie fizyczną odległość. Miękkie mieszanie używa położenia ważonego prawdopodobieństwami; twardy wybór — najbardziej prawdopodobnej klatki dla każdego piksela. StackMFF V2 nie ma parametrów i zgłasza warstwę ogniskową próbkowaną przez swoją sieć. Wartości szkicu pochodzą z parametrów pasującego wyniku lub zapamiętanych ustawień metody w głównym oknie. Są pamiętane osobno dla źródła i metody, dopóki okno jest otwarte; ich zmiana nie zmienia głównego okna, projektu ani zapisanego wyniku.

Zmiana ustawień obliczeń oznacza poprzednią mapę 2D jako **nieaktualną**, czyści powierzchnię 3D i blokuje każdy eksport, w tym kolorowy, surowy, fizyczny i OBJ, do czasu udanych nowych obliczeń. Aktywne obliczenia są anulowane kooperacyjnie; po zakończeniu anulowania kliknij ponownie **Oblicz**. Następne obliczenia nie są kolejkowane. Przycisk pozostaje widoczny i jest wyłączony podczas obliczania, anulowania i eksportu.

- **Mapa kolorów**: odcienie szarości (standard publikacyjny) plus 15 palet naukowych (Turbo, Viridis, Magma, …) i własna — *Nowy schemat niestandardowy…* buduje gradient z dowolnej liczby punktów koloru (dwuklik na punkcie zmienia położenie i kolor); własne schematy są zapisywane i oferowane na liście.
- **Odwróć / Gamma / Wygładzanie**: odwraca oś głębi, wyrównuje ton (gamma < 1 rozjaśnia niskie indeksy) i usuwa plamki przejściem świadomym krawędzi.
- **Nakładka**: dostępna tylko dla powiązanego wyniku o pasujących współrzędnych.
- **Eksport surowy**: 16-bitowy TIFF koduje znormalizowane indeksy wartościami 0–65535; to nie są bezstratne dane zmiennoprzecinkowe ani fizyczna odległość. JPEG/BMP są odrzucane.
- **Indeksowy TIFF i alternatywny obraz TIFF**: eksport bieżącego oszacowania głębi jako 16-bitowych indeksów klatek lub obrazu złożonego z pikseli źródeł. Wartości indeksów 1–N odnoszą się do dokładnej kolejności klatek wejściowych obliczenia głębi, po rejestracji i zmianie kolejności; 0 oznacza nieprawidłowe piksele. Obsługiwane jest maksymalnie 65 535 klatek. Ułamkowe oszacowania są zaokrąglane do najbliższej klatki, a wartości w połowie — w górę. Indeksy szacują położenie ostrości; nie są prawdziwym pochodzeniem pikseli wyniku fuzji. Alternatywny obraz wybiera piksel odpowiedniej wejściowej klatki głębi w każdym prawidłowym miejscu, a nieprawidłowe piksele zeruje. Jeśli którykolwiek wejściowy obraz jest 16-bitowy, wartości 8-bitowe są promowane do 16 bitów przez pomnożenie przez 257. Eksport używa ROI mapy głębi i zachowuje nieprawidłowe obszary jako zero.
- **Głębia fizyczna**: wprowadź początek Z, dodatni krok, jednostkę i kierunek akwizycji, a następnie potwierdź odwzorowanie oryginalnej rejestracji. Przełączenie między bieżącym stosem a wynikiem czyści potwierdzenie kolejności rejestracji, nawet jeśli nazwy i indeksy się zgadzają; po zmianie źródła potwierdź je ponownie. Wykluczone klatki zachowują przerwy. Float32 TIFF przechowuje mikrometry, NaN dla brakujących danych i metadane źródła. Fizyczny OBJ wymaga też kalibracji XY; używa fizycznej skali Z 1:1.
- **Model 3D / tekstura 3D**: wybierz najdłuższy bok siatki (domyślnie 600 wierzchołków). Każde nowo otwarte okno domyślnie używa **Automatycznie (prawdziwe kolory)**: powiązany wynik jest używany tylko wtedy, gdy jego tożsamość i współrzędne/ROI pasują do ukończonej mapy; w przeciwnym razie używana jest pierwsza uczestnicząca klatka z ukończonych obliczeń, po rejestracji i zmianie kolejności. Pokazywana jest nazwa wybranego wyniku lub klatki. Można jawnie wybrać pasujący wynik, uczestniczącą klatkę lub **Pseudokolor głębi**; niedostępne obrazy są zgłaszane i nigdy nie zastępowane niepowiązanymi wynikami. Ta tekstura jest wspólna dla podglądu 3D i PNG tekstury OBJ. Mapa 2D zachowuje własną paletę i nakładkę: zmiana palety lub gammy nie przebarwia tekstury prawdziwych kolorów. Jawne pseudokolory podążają za paletą, gammą i odwróceniem. Ustalony podgląd i eksport używają tego samego próbkowanego pola wysokości bez ukrytego dodatkowego wygładzania. Względna rzeźba jest podana w jednostkach obrazu.
- **Eksport modelu**: OBJ, MTL, PNG tekstury i metadane JSON tworzą jedną grupę plików; przechowuj je razem. Eksport działa w tle. Konflikty są sprawdzane dla każdego pliku, przygotowanie jest etapowe, anulowanie zachowuje istniejące pliki, a błąd zatwierdzenia wycofuje zamiany. Zapis modelu nie eksportuje nadrzędnego obrazu fuzji i nie zdejmuje z niego ochrony niezapisanych zmian.

Obliczenia działają w tle z postępem i możliwością anulowania; metoda AI używa tego samego oszczędnego podziału na kafelki co fuzja AI, więc głębokie stosy są obsługiwane. Odpowiednik w CLI — `--depth-map [path]` razem z `--depth-colormap`; zapisuje tylko kolorową mapę (domyślnie `<output>_depth.png`), działa wyłącznie dla wymienionych wyżej metod i jest odrzucany dla wejścia przezroczystego lub grupowego. GUI dodatkowo eksportuje surową głębię, indeksy klatek, obrazy alternatywne, głębię fizyczną i modele 3D.

### Automatyczne odzyskiwanie sesji

OpenFocus stale przechowuje migawkę sesji roboczej (stos, ustawienia, etykiety, bieżąca klatka). Jeśli aplikacja zakończyła się nieoczekiwanie — z powodu awarii, utraty zasilania lub wymuszonego zamknięcia — przy następnym uruchomieniu zostanie to wykryte i zaproponowane **Przywróć sesję**, aby wrócić dokładnie tam, gdzie przerwano. Przy poprawnym wyjściu pytanie się nie pojawia. Dane odzyskiwania są przechowywane obok ustawień (w instalacji przenośnej obok pliku wykonywalnego, więc przenoszą się razem z nią).

### Ostatnie pliki

Każdy otwarty stos jest zapamiętywany w **Plik → Otwórz ostatnie** (do 8 pozycji); wybierz pozycję z menu. Ponowne otwarcie tego samego stosu przenosi go na górę; pozycje, których pliki już nie istnieją, są usuwane z potwierdzeniem, a **Wyczyść listę ostatnich** opróżnia menu. Ostatnie projekty są prowadzone osobno w **Plik → Otwórz ostatni projekt**.

### Przeciąganie i upuszczanie

Folder można przeciągnąć bezpośrednio do okna aplikacji, aby zaimportować obrazy. Jeśli przeciągniesz pliki do już wczytanego stosu, zostaną do niego dodane (po sprawdzeniu rozmiarów).

### Orientacja EXIF

Pliki JPEG/TIFF aparatów ze znacznikiem EXIF Orientation są automatycznie obracane i odbijane przy imporcie, więc zdjęcia portretowe wczytują się prawidłowo.

### Źródła 16-bitowe

16-bitowe PNG/TIFF są wykrywane automatycznie i zachowywane na całej ścieżce: fuzja działa w pełnej głębi, a wyniki można zapisywać z powrotem jako 16-bitowe PNG/TIFF (eksport JPG/BMP jest konwertowany do 8 bitów, ponieważ te formaty nie przechowują 16 bitów).

Pliki RAW aparatów są wywoływane do 16 bitów (balans bieli aparatu, gamma sRGB) z wyłączonym automatycznym rozjaśnianiem, dzięki czemu każda klatka stosu otrzymuje to samo odwzorowanie ekspozycji; automatyczna ekspozycja dla każdej klatki powodowałaby migotanie jasności między klatkami. Zapisuj w TIFF lub PNG, aby zachować dane 16-bitowe.

### Profile kolorów

Eksport pojedynczych obrazów PNG, TIFF, JPEG i WebP zachowuje profil RGB ICC, jeśli wszystkie klatki źródłowe mają ten sam osadzony profil. Stosy z mieszanymi profilami lub bez profilu są eksportowane bez profilu ICC. OpenFocus zachowuje profil, ale nie konwertuje pikseli źródłowych do wspólnej przestrzeni barw ani nie zarządza kolorem podglądu. Obrazy z mieszanymi profilami przekonwertuj do wspólnej przestrzeni przed wczytaniem. Wejście RAW jest wywoływane do sRGB i nie dziedziczy profilu aparatu.

Dla wielostronicowego TIFF wszystkie strony pliku muszą mieć ten sam prawidłowy profil RGB; w przeciwnym razie jego klatki są traktowane jako bez profilu. Klatki wideo również są traktowane jako bez profilu. Dodane klatki i klatki wybrane do renderowania uczestniczą w wyborze profilu, który pozostaje z wynikiem do późniejszego eksportu.

### Czyszczenie stosu

Aby usunąć wszystkie wczytane obrazy, wybierz **Plik → Wyczyść stos** lub naciśnij `Ctrl+W`.

---

## 5. Przeglądanie i nawigacja po obrazach

### Nawigacja po stosie obrazów

- **Suwak**: przeciągaj suwak pod panelem źródła, aby przewijać klatki
- **Lista plików**: kliknij nazwę pliku na liście obrazów źródłowych w prawym panelu; gdy lista ma fokus, strzałki w górę/w dół przesuwają zaznaczenie, a panel źródła podąża za nim
- **Klawiatura**: nie ma globalnego skrótu poprzedniej/następnej klatki; najpierw przenieś fokus na listę źródeł

### Przeglądanie podobnych klatek

Kliknij **Sprawdź klatki** pod nagłówkiem listy źródeł, aby znaleźć sąsiednie klatki o prawie identycznych pikselach. Podczas analizy przycisk zmienia się na **Anuluj sprawdzanie**. Każdy wiersz propozycji podaje klatkę odniesienia i różnicę; obejrzyj obie, zanim użyjesz pola wyboru, aby wykluczyć klatkę z fuzji. Propozycje nigdy nie zmieniają pól wyboru ani nie usuwają plików. To ostrożne porównanie może pominąć podobne klatki; zmiany ograniczone do małego ostrego obszaru nie są uśredniane. Wyniki są odrzucane, jeśli stos się zmieni. Propozycje są pokazywane tylko wtedy, gdy zarówno proponowana klatka, jak i jej odniesienie są uwzględnione; zmiana pola wyboru aktualizuje wyświetlane propozycje.

### Powiększanie i przesuwanie

- **Powiększanie/zmniejszanie**: kółko myszy nad obrazem
- **Dokładne powiększanie**: przytrzymaj `Ctrl` i kręć kółkiem, aby uzyskać mniejszy krok
- **Dopasowanie do okna**: dwuklik na obrazie przełącza między dopasowaniem a widokiem 100 %
- **Przesuwanie**: kliknij i przeciągnij, aby przesuwać obraz w panelu

### Nawigacja zsynchronizowana

Przy przechodzeniu po klatkach w panelu źródła panel wyniku pokazuje odpowiednią klatkę wyniku, jeśli istnieje.

---

## 6. Widok porównania suwakiem

Widok suwakiem nakłada dwa obrazy w jednej ramce z przeciąganym pionowym dzielnikiem, dzięki czemu przesunięcia i różnice jakości są od razu widoczne po obu stronach podziału.

### Otwieranie widoku suwakiem

1. Uzyskaj co najmniej jeden wynik (fuzja lub rejestracja)
2. Naciśnij przełącznik **Suwak** w prawym górnym rogu paska tytułu panelu wyniku
3. Obszar wyniku przełączy się w widok porównania; naciśnij **Suwak** ponownie, aby wrócić

### Interpretacja widoku

- **Strona A (lewo od dzielnika)**: klatka źródłowa — domyślnie podąża za suwakiem stosu źródłowego
- **Strona B (prawo od dzielnika)**: najnowszy render
- **Biały dzielnik z okrągłym uchwytem**: przeciągaj poziomo, aby przesuwać granicę
- Przeciąganie przez dzielnik krawędzi, które „skaczą”, ujawnia błędy rejestracji słabo widoczne przy porównaniu paneli obok siebie

### Wybór porównywanych obrazów

Pasek na dole widoku porównania ma po jednej grupie dla każdej strony:

- **Strona A** — zaznacz **Podążaj za źródłem** (domyślnie), aby śledzić suwak stosu źródłowego, lub odznacz, aby zablokować stronę A na dowolnej klatce własnym suwakiem
- **Strona B** — zaznacz **Najnowszy wynik** (domyślnie), aby zawsze pokazywać najświeższy render, lub odznacz, aby przeglądać historię wyników własnym suwakiem

Obie strony aktualizują się na żywo: nowe rendery, ruchy suwaka źródła i usuwanie klatek są odzwierciedlane natychmiast.

### Powiększanie i przesuwanie

Widok porównania stosuje jedno przekształcenie do obu obrazów:

- **Powiększenie**: kółko myszy (powiększa wokół kursora)
- **Przesuwanie**: kliknij i przeciągnij w dowolnym miejscu poza dzielnikiem
- **Reset**: dwuklik

Tryb suwakiem i tryb ROI współdzielą panel wyniku, więc wejście w jeden wychodzi z drugiego. Wyczyszczenie stosu również wyłącza tryb suwakiem.

---

## 7. Rejestracja obrazów

Rejestracja usuwa przestrzenne przesunięcie klatek w stosie. Jest niezbędna, gdy zdjęcia są lekko przesunięte lub mają inną perspektywę.

### Metody rejestracji

#### ECC (ulepszony współczynnik korelacji)

- Szuka najlepszego wyrównania przez optymalizację korelacji
- Nadaje się do niewielkich przesunięć
- Daje dokładność subpikselową
- Wolniejszy od homografii, ale dokładniejszy

#### Homografia

- Oparta na dopasowaniu punktów charakterystycznych (cechy SIFT/ORB)
- Radzi sobie z większymi przekształceniami geometrycznymi
- Szybsza od ECC przy wstępnym wyrównaniu
- Dobra dla zdjęć o znacznej zmianie perspektywy

#### Homografia → ECC

- Najpierw homografia, potem udokładnienie ECC
- Najwolniejszy i najbardziej odporny na trudną geometrię wariant

#### OpenFocus (szybko)

- Ograniczone wyrównanie podobieństwa (przesunięcie + obrót + skala) z wyrównaniem jasności klatek i automatycznym przycięciem
- Wyszukiwanie ocenia różnice próbkowanych pikseli; przekształcenie używa interpolacji Lanczosa
- Klatki przesunięte o **więcej niż 8 % rozmiaru klatki pozostają niewyrównane**, a nie częściowo poprawione. Próg zmienia parametr CLI `--align-shift-limit`
- Dla geometrii nieodkształcalnej lub ekstremalnej wybierz ECC albo homografię
- **Obsługa odrzuconych klatek** domyślnie kontynuuje z niewyrównaną klatką. Wybierz **Zatrzymaj po odrzuceniu klatki**, aby render lub element wsadu zakończył się błędem przed utworzeniem wyniku, gdy jakość detektora jest niższa niż 0,06 albo oszacowany ruch przekracza skonfigurowany limit przesunięcia, obrót 8° lub zmianę skali 5 %. Klatki są numerowane od 1 w kolejności przetwarzanego stosu. Geometryczne odniesienie domyślnie to klatka 1; słabe bezpośrednie dopasowanie może zostać ponowione z sąsiednią klatką bliższą odniesieniu. Ta polityka wykrywa odrzucone oszacowania, ale nie wszystkie błędne dopasowania: słaba tekstura, powtarzające się wzory, przesłonięcia, duży ruch lub zmiany ostrości nadal mogą pokonać rejestrację. Sprawdź wynik.
- W CLI tę samą politykę wybiera `--align openfocus --align-param 'failure_policy="fail"'` (cytowanie zależne od powłoki). Elementy wsadu stosują wybraną politykę niezależnie. Domyślnie jest `"continue"`.
- Ustaw **Klatkę odniesienia** jako liczbę od 1 w kolejności uwzględnionych, przetworzonych klatek, aby zakotwiczyć geometrię w tej klatce. Kolejność klatek jest zachowana; wyrównanie jasności używa własnego odniesienia. Każdy stos wsadu musi zawierać ten numer, inaczej element kończy się błędem. Przykład CLI: `--align openfocus --align-param reference_frame=3`. Wybór odniesienia nie poprawia paralaksy ani słabego dopasowania; sprawdź wynik.

Raporty wyrównania wskazują odniesienie i wynik każdej klatki. „Wynik zastępczy” oznacza, że metoda nie mogła ustalić wyrównania lub zależy od wcześniejszego nieudanego oszacowania; podana przyczyna odróżnia słabe oszacowania od ruchu poza limitami, gdy jest to znane. Wartości jakości zależą od metody i nie gwarantują dokładności. Przekształcenia mapują współrzędne źródła na płótno odniesienia etapu rejestracji przed wspólnym przycięciem; metody złożone raportują etapy osobno. Homografia i ECC również przyjmują opcjonalny parametr rejestracji `failure_policy="fail"`, aby zatrzymać się po wykryciu błędu. Domyślnie pozostaje `"continue"`; sprawdzaj wyniki nawet przy udanym wyrównaniu.

Aby zbadać render, włącz **Ustawienia → Ustawienia zaawansowane → Rejestracja → Zbieraj diagnostykę wyrównania** i wyrenderuj ponownie. Powoduje to ponowne obliczenie wyrównania zamiast użycia pamięci podręcznej. Podsumowanie zakończenia wyróżnia wyniki zastępcze; otwórz **Ustawienia → Ustawienia zaawansowane → Rejestracja → Diagnostyka wyrównania**, aby zobaczyć raport dla klatek. Pokazuje dostępne przyczyny niepowodzenia, oceny metody, liczbę dopasowań, klatki odniesienia i przekształcenia. Numery i nazwy odniesień dotyczą oryginalnej listy źródeł, także gdy klatki są wykluczone. Renderowania wsadowe i grupowe nie udostępniają tego raportu.

### Włączanie rejestracji

1. Na prawym panelu znajdź grupę **Rejestracja**
2. Wybierz jedną metodę z listy rozwijanej: **Bez wyrównania**, **OpenFocus (szybko)**, **ECC**, **Homografia** lub **Homografia → ECC**
3. Wybierz **Obsługę odrzuconych klatek** w **Ustawienia → Ustawienia zaawansowane → Rejestracja**; zmiana jest stosowana do metod. Dla OpenFocus (szybko) ustaw także klatkę odniesienia.
4. **Zbieraj diagnostykę wyrównania** znajduje się teraz w ustawieniach zaawansowanych (patrz wyżej)

Domyślnie wybrane jest **ECC**. Jeśli stos jest już wyrównany, na przykład pochodzi ze sztywnego stanowiska, wybierz **Bez wyrównania**: jest szybciej i unika zbędnych przekształceń.

### Kiedy stosować rejestrację

- Na obrazach widać wyraźne przesunięcie
- Zdjęcia wykonano z ręki
- W scenie występuje paralaksa (bliższe obiekty przesuwają się między klatkami)
- W wynikach widać duchu lub podwojenia

### Ustawienia rejestracji

Dodatkowe parametry otwiera **Ustawienia → Ustawienia zaawansowane → Rejestracja**:

- **Szerokość skalowania rejestracji**: rozdzielczość wstępnego przetwarzania (domyślnie 1024 px)
- Mniejsze wartości — szybciej, ale dokładność może spaść
- Większe wartości — wolniej, ale dokładność może wzrosnąć
- **Pamięć rejestracji**: przechowuje wyniki obok stosu, aby ponowne otwarcie pomijało wyrównanie
- **Zbieraj diagnostykę wyrównania**: ponownie oblicza wyrównanie na potrzeby raportu dla klatek

---

## 8. Metody fuzji obrazów

OpenFocus oferuje dziesięć algorytmów fuzji plus tryb tylko wyrównania. W nowym profilu wybrana jest **StackMFF V3** z rejestracją **ECC**; poniżej opisano wszystkie opcje. Każdy ma inne właściwości odpowiednie dla różnych typów obrazów.

### StackMFF V3 (domyślnie, uczenie głębokie)

- **Algorytm**: ONNX Runtime i dołączony model ONNX.
- **Tryb wyjścia AI** (prawy panel, tylko dla tej metody):
  - **Miękkie mieszanie (ważone)** — domyślnie; każdy piksel to ważona prawdopodobieństwami mieszanka klatek. Płynne przejścia, bez twardych szwów; odpowiada konfiguracji z artykułu
  - **Twardy wybór (jedna klatka)** — każdy piksel jest kopiowany z najbardziej prawdopodobnej klatki. Ostrzejszy wybór bez interpolacji między klatkami; na granicy dwóch równie prawdopodobnych klatek może pojawić się szew
- **Najlepsza do**: najwyższej jakości, gdy dostępny jest GPU
- **Zalety**: niezależna od kolejności klatek, obsługuje stos w granicach dostępnych zasobów, nie wymaga strojenia parametrów, optymalizuje się automatycznie
- **Wymagania**: ONNX Runtime i dołączony model ONNX.

Każdy kafelek sieci używa wszystkich klatek. Klatki źródłowe i dane tymczasowe wyrównania mogą być przechowywane na dysku; wolne miejsce i dostępna pamięć ograniczają wielkość stosu. Kafelkowanie przestrzenne V2 przybliża wnioskowanie dla całego obrazu.


### StackMFF V2 (uczenie głębokie)

- **Algorytm**: jednoskalowa sieć uwagi normalizuje oczekiwaną głębię ostrości i próbkuje najbliższą klatkę źródłową. Indeksy głębi zależą od kolejności stosu. **Klatki muszą zachować oryginalną kolejność ogniskowania** — odwrócenie jest dla tej metody wyłączone
- **Mapy głębi**: obsługiwane; podają próbkowaną warstwę ogniskową
- **Najlepsza do**: ciągłych serii ogniskowania, gdzie liczy się bardzo mały, szybki model
- **Wymagania**: ONNX Runtime i dołączony model ONNX.

### StackMFF (V1, uczenie głębokie)

- **Algorytm**: jednoskalowa sieć uwagi miesza luminancję klatek źródłowych przy użyciu wag softmax; Cb/Cr pochodzą z klatki o luminancji najbliższej wynikowej
- **Najlepsza do**: neuronowej fuzji całego stosu ostrości
- **Zalety**: bez parametrów, kolor pozostaje w gamucie; wynikowa luminancja nie zależy od kolejności klatek, ale chrominancja może się różnić przy równych odległościach luminancji
- **Uwaga**: około 156 172 parametrów; większym stosom pomaga GPU

### Filtr sterowany

- **Algorytm**: filtrowanie sterowane z zachowaniem krawędzi
- **Najlepsza do**: zastosowań ogólnych, rozsądny kompromis między szybkością a jakością
- **Zalety**: dobrze zachowuje krawędzie, działa szybko
- **Parametr**: rozmiar jądra od 1 do 51, tylko wartości nieparzyste (domyślnie 31)
  - Mniejsze jądro: więcej lokalnych szczegółów, możliwy wzrost szumu
  - Większe jądro: gładszy wynik, lepsze tłumienie szumu

### DCT (dyskretna transformata kosinusowa)

- **Algorytm**: fuzja w dziedzinie częstotliwości według współczynników DCT
- **Najlepsza do**: obrazów bogatych w tekstury, obrazowania naukowego
- **Zalety**: zachowuje drobne tekstury, matematycznie uzasadniona
- **Parametr**: rozmiar jądra określa okno przetwarzania (1–51, domyślnie 7)

### DTCWT (dual-tree complex wavelet transform)

- **Algorytm**: fuzja w wieloskalowej dziedzinie falkowej
- **Najlepsza do**: złożonych scen z detalami w różnych skalach
- **Zalety**: doskonała analiza wieloskalowa, dobra kierunkowość
- **Reguła fuzji**: jedna decyzja na piksel i poziom dla wszystkich orientacji i kanałów koloru (wybór N-kierunkowy z kontrolą spójności większości); pasmo dolnoprzepustowe podąża za najgrubszą decyzją zamiast uśredniania wszystkich klatek
- **Parametr**: poziomy falkowe (1–6, domyślnie 4) w prawym panelu. Metoda nie ma jądra przestrzennego, więc suwak jądra jest dla niej ukryty

### GFG-FGF (gradient quasi-gaussowski w czterech sąsiedztwach + szybki filtr sterowany)

- **Algorytm**: implementacja metody Fu i in. (*Laser & Optoelectronics Progress*, 2024): mapa ostrości z różnicy rozmyć |I − mean(I)| jest wzbogacana operatorem gradientu quasi-gaussowskiego w czterech sąsiedztwach (GFG), udokładniana szybkim filtrem sterowanym, decyzja podejmowana jest dla każdego piksela (remisy rozstrzyga miara małego obszaru Sobela 3×3), a binarne mapy decyzji są ponownie wygładzane szybkim filtrem sterowanym przed ważonym mieszaniem
- **Najlepsza do**: mikroskopii i zdjęć z wyraźnymi obszarami ostrości
- **Zalety**: szybka, ostre granice obszarów ostrości; pamięć nie rośnie z liczbą klatek
- **Parametr**: jądro = rozmiar filtra średniego zgrubnej mapy ostrości (1–51, domyślnie 7)

### OpenFocus A / B / C

- **OpenFocus A**: wygładza wieloskalową mapę odpowiedzi ostrości, a następnie wybiera ostre piksele
- **OpenFocus B**: miesza sąsiednie klatki według wygładzonej mapy głębi
- **OpenFocus C**: rekonstruuje wieloskalowe pasma według wygładzonej mapy głębi
- **Parametry**: A i B używają suwaka jądra do skali odpowiedzi ostrości (1–21, domyślnie 11); C nie ma promienia. Wszystkie trzy mają parametr wygładzania kontrolujący siłę przejść

**Wygładzanie**: jedno pole pod suwakiem jądra, znaczenie zależy od wybranej metody (zakres 0–8, krok 0,2):

| Metoda | Etykieta | Domyślnie | Efekt |
|---|---|---|---|
| OpenFocus A | Wygładzanie mieszania | 2,0 | Wygładza mapę odpowiedzi ostrości przed wyborem pikseli |
| OpenFocus B | Wygładzanie głębi | 1,2 | Kontroluje wygładzanie mapy głębi kierującej mieszaniem |
| OpenFocus C | Wygładzanie mieszania | 3,0 | Kontroluje wygładzanie mapy głębi dla rekonstrukcji wieloskalowej |

Większe wartości zwiększają wygładzanie przejść. Każda metoda pamięta własną wartość między sesjami, a przetwarzanie wsadowe stosuje tę samą regułę.

C nie ma parametru promienia, więc suwak jądra jest dla niej ukryty.

CLI: `--depth-smoothing VALUE` (B) i `--blend-smoothing VALUE` (A, C).

**Opcje jasności są osobne dla każdej metody.** *Dopasuj ekspozycję przed fuzją* i *Usuń dryf jasności* są pamiętane osobno dla każdej metody fuzji, ponieważ odpowiadają na pytanie konkretnego stosu, a nie na ogólne preferencje. Obie są domyślnie włączone dla OpenFocus A/B/C (które nie mają wbudowanej kalibracji ekspozycji); pozostałe metody zachowują historyczne wartości domyślne (dopasowanie ekspozycji włączone, usuwanie dryfu wyłączone). Przełączenie metody pokazuje jej własne wartości, a one przetrwają restart. **Przywróć domyślne** czyści zapamiętane wartości jasności, wygładzania i promienia wszystkich metod.

**Wyostrz wynik łączenia** stosuje łagodną maskę nieostrą po fuzji, przed konwersją do głębi wyjściowej. Jest domyślnie wyłączone i można je włączyć w głównych lub wsadowych opcjach fuzji; odpowiednik w CLI to `--sharpen-output`. Wpływa na obraz po fuzji, a nie na wyrównane klatki ani mapy głębi. Może uwydatnić krawędzie, ale też wzmocnić szum i obwódki. Ustawienie jest zapisywane w projekcie i pamiętane między sesjami.

### Tylko wyrównanie (bez fuzji)

Wybierz **Tylko wyrównanie (bez fuzji)**, aby zarejestrować stos i dodać do listy wyników jedną pozycję stosu. Klatki pozostają w pamięci do eksportu; tryb nie tworzy dodatkowej pamięci podręcznej ani nie zapisuje obok źródeł. Zapis pozycji tworzy jeden folder w wybranym miejscu docelowym. Oryginalne nazwy plików są zachowane, a rozszerzenie zależy od wybranego formatu.

### Wybór metody fuzji

1. W prawym panelu znajdź grupę **Fuzja**
2. Wybierz metodę z **listy rozwijanej** (zawsze wybrana jest dokładnie jedna; pierwsza pozycja rejestruje bez fuzji)
3. W razie potrzeby dostosuj rozmiar jądra lub parametry — etykieta pokazuje, co oznacza jądro dla wybranej metody
4. Kliknij **Rozpocznij renderowanie**, aby uruchomić przetwarzanie

### Przewodnik porównania algorytmów

| Algorytm | Szybkość | Jakość | Najlepszy do |
|-----------|-------|---------|----------|
| Filtr sterowany | Szybko | Dobra | Zastosowań ogólnych |
| DCT | Średnio | Dobra | Tekstur |
| DTCWT | Średnio | Bardzo dobra | Złożonych scen |
| GFG-FGF | Szybko | Dobra | Obszarów ostrości |
| StackMFF (V1) | Wolno (GPU) | Nie badano tutaj | Luminancja niezależna od kolejności; chrominancja może się różnić przy równych odległościach luminancji |
| StackMFF V2 | Szybko (GPU) | Nie badano tutaj | Ciągłych serii z bardzo małym modelem |
| StackMFF V3 | Wolno (GPU) | Doskonała | Najwyższej jakości |
| OpenFocus A | Zależy od stosu | Zależy od stosu | Wyboru ostrych pikseli według odpowiedzi wieloskalowej |
| OpenFocus B | Zależy od stosu | Zależy od stosu | Mieszania sąsiednich klatek według mapy głębi |
| OpenFocus C | Zależy od stosu | Zależy od stosu | Rekonstrukcji wieloskalowej według mapy głębi |

**Narzędzia → Porównaj metody…** pozwala niezależnie ustawić kilka metod. Nazwa metody, etykiety parametrów i kontrolki pozostają razem przy zwężaniu okna. Zmiany automatycznie aktualizują podgląd. **Podgląd ROI (szybko)** łączy wybrany obszar z otaczającym kontekstem; **Fuzja całego obrazu** sprawdza wynik całego obrazu i pozwala zbadać inny obszar bez ponownego renderowania. Wyniki ROI mogą różnić się od pełnego obrazu, ponieważ statystyki ekspozycji i niektóre algorytmy zależą od rozmiaru obrazu.
Domyślnie wybrana jest **fuzja całego obrazu**.

Widoczna wskazówka wyjaśnia wybór wyniku: kliknij wynik albo przejdź do niego klawiszem `Tab` i wybierz go klawiszem `Enter` lub `Spacja`.

**Rozmiar podglądu (px)** nad prawymi panelami ustawia kwadrat od 160 do 960 pikseli przez przeciągnięcie suwaka lub wpisanie liczby. Wszystkie metody używają tego samego rozmiaru i zawijają się do dostępnej szerokości. Rozmiar jest pamiętany przy następnym otwarciu. Zmiana zachowuje bieżące powiększenie i dopasowuje kwadrat inspekcji wokół środka z ograniczeniem do granic obrazu.

Naciśnij **F11** lub przycisk **Pełny ekran (F11)**, aby wykorzystać cały ekran; tekst przycisku zmienia się na **Wyjdź z pełnego ekranu (F11)**. Pasek tytułu również pozwala zmaksymalizować okno. **Esc** najpierw wychodzi z pełnego ekranu i zamyka okno dopiero przy drugim naciśnięciu. Powrót z pełnego ekranu przywraca poprzedni rozmiar okna, a siatka wyników w obu przypadkach dopasowuje się do dostępnej szerokości.

Przeciągaj obraz źródłowy lewym przyciskiem, aby przesuwać biały kwadrat inspekcji; po zwolnieniu przycisku pozostaje na miejscu. Prawe panele pokazują dokładnie ten obszar z tego samego wyrównanego stosu. Kółko zmienia powiększenie obszaru, przeciąganie prawym przyciskiem przesuwa źródło, **Ctrl+kółko** powiększa źródło, a **F** dopasowuje. Zmiana klatki źródłowej lub rozmiaru okna zachowuje zaznaczenie. **Anuluj** zatrzymuje dalszą pracę; zamknięcie czeka na bezpieczny powrót bieżącej operacji natywnej.

### Szybki podgląd (szkic renderu)

Dostrajanie parametrów na dużych stosach w pełnej rozdzielczości jest wolne. Otwórz **Ustawienia renderowania** i zaznacz **Szybki podgląd**, a OpenFocus najpierw wyrenderuje szkic w połowie liniowego rozmiaru (lecz nie mniejszy niż 320 px na dłuższym boku) — zwykle kilka razy szybciej, łącznie z rejestracją. Komunikat o zakończeniu (oraz podpowiedź najnowszego wyniku) pokazuje **Jakość: Podgląd**, a szkic **nie jest dodawany** do historii wyników. Odznacz pole i wyrenderuj ponownie, aby uzyskać wynik w pełnej rozdzielczości. Tryb podglądu nie dotyczy renderów ROI i nigdy nie zanieczyszcza pamięci podręcznej wyrównania w pełnej rozdzielczości.

---

## 9. Przekształcenia obrazów

### Obrót

1. Wybierz **Edycja → Obrót**
2. Wybierz typ obrotu:
   - **90° zgodnie z ruchem wskazówek zegara**: obrót wszystkich obrazów o 90 stopni w prawo
   - **90° przeciwnie do ruchu wskazówek zegara**: obrót wszystkich obrazów o 90 stopni w lewo
   - **180°**: obrót obrazów do góry nogami

### Odbijanie

1. Wybierz **Edycja → Odbicie**
2. Wybierz typ odbicia:
   - **Odbicie poziome**: lustrzane odbicie od lewej do prawej
   - **Odbicie pionowe**: lustrzane odbicie od góry do dołu

### Zmiana rozmiaru

1. Wybierz **Edycja → Zmień rozmiar**
2. Otworzy się okno opcji zmiany rozmiaru
3. Wpisz nowe wymiary lub procent skalowania
4. Wszystkie obrazy w stosie są skalowane proporcjonalnie

### Kolejność przekształceń

Przekształcenia są stosowane do całego stosu obrazów i zachowują wyrównanie między klatkami. Po obrocie, odbiciu lub zmianie rozmiaru czyszczenie matrycy jest niedostępne: wczytaj ponownie oryginalne klatki lub wyłącz czyszczenie.

---

## 10. Retusz, etykiety i adnotacje

### Retusz wyniku (Edycja → Retusz wyniku…, `Ctrl+T`)

Fuzja może pozostawić lokalną wadę — obwódkę przy kontrastowej krawędzi, włos pochodzący z niewłaściwej klatki, szum na płaskim obszarze. Przestrzeń robocza kopiuje piksele z dokładnych klatek, które utworzyły wybrany wynik:

1. Wybierz jeden wynik na liście wyników (kliknięcie prawym przyciskiem → *Retusz wyniku…* również działa) i naciśnij `Ctrl+T`. Bez zaznaczenia otwiera się lista wszystkich wyników z oznaczeniem tych, które mają zapisane klatki źródłowe.
2. Klatka źródłowa jest po lewej, wynik po prawej. Oba współdzielą powiększenie i położenie: kółko powiększa, przeciąganie prawym/środkowym przyciskiem (lub `Spacja`+przeciąganie) przesuwa, `F` dopasowuje, `1` pokazuje 100 %.
3. Wybierz miniaturę, przeciągnij suwak klatki lub naciśnij `,` / `.`. Pociągnięcie jest zablokowane na tej klatce i bieżących ustawieniach pędzla; klatki nie można zmienić do zakończenia pociągnięcia.
4. Wybierz **Pędzel źródłowy** (`B`) lub **Pędzel przywracania** (`E`). Średnica, twardość i krycie mają suwaki i pola liczbowe; `[` / `]` zmienia średnicę. `Ctrl` + kółko w górę zwiększa średnicę pędzla, a w dół ją zmniejsza; samo kółko nadal zmienia powiększenie. Pędzel przywracania przywraca piksele do obrazu otwartego na początku, a nie do starszego wyniku.
5. Przytrzymaj `Tab`, aby zobaczyć wynik sprzed tej sesji, i `Alt`, aby podejrzeć klatkę źródłową. Opcjonalne odniesienie ostrości obejmuje tylko płótno źródła i używa jednej wspólnej skali dla wszystkich klatek.
6. `Ctrl+Z` / `Ctrl+Y` cofają i ponawiają całe pociągnięcia. **Przywróć wszystko…** to jedna operacja do cofnięcia. **Zastosuj jako nowy wynik** dodaje pozycję `<nazwa> · Retusz N` i wybiera ją; oryginalny wynik pozostaje nietknięty. Nowa pozycja jest oznaczona jako *Nie wyeksportowano*, dopóki nie wyeksportujesz jej z głównego okna; OpenFocus chroni ją przed automatycznym usunięciem z historii i przed wyjściem bez decyzji.

Zastosuj retusz jako nowy wynik, zaznacz go i otwórz **Zapisz jako**. Wybierz TIFF z warstwami z listy formatów, aby zapisać wynik razem z klatkami źródłowymi. Etykiety na żywo i paski skali nie są dołączane; niezapisane zmiany adnotacji pozostają chronione. Geometria źródeł i maski ważności są zachowane. Wynik i źródła muszą być nieprzezroczystymi obrazami RGB lub w odcieniach szarości o tej samej głębi 8 lub 16 bitów, a każdy prostokąt źródła musi mieścić się w wyniku. Eksport obsługuje do 128 źródeł; nazwy źródeł są ograniczone do 255 znaków i nie mogą zawierać NUL ani znaków spoza Basic Multilingual Plane, na przykład wielu emoji. Niebezpieczne konteksty i zadania szacowane na więcej niż 512 MiB pamięci roboczej lub dostępnego budżetu RAM są odrzucane z komunikatem o błędzie zamiast zapisywania niekompletnego eksportu. GIMP 3.2.6 w systemie Windows importuje warstwy 8- i 16-bitowe, w tym ich przesunięcia i przezroczyste wypełnienie rejestracji. Wynik jest widoczny, warstwy źródeł ukryte; GIMP dodaje też widoczną warstwę Background. Inne edytory nie były testowane.

Do lokalnej korekty tonalnej wybierz **Pędzel ekspozycji**, ustaw wartość w setnych stopnia (100 podwaja wartości pikseli, −100 zmniejsza je o połowę) i maluj po wyniku. Podgląd aktualizuje się podczas malowania. Każde pociągnięcie dostosowuje bieżący wynik raz, nawet gdy ślady pędzla się nakładają; kolejne pociągnięcie buduje na nim. Krycie i twardość pędzla kontrolują mieszanie. Przy rozjaśnianiu światła mogą się obciąć. Przytrzymaj **Przed**, aby porównać z wynikiem z chwili otwarcia; naciśnij Escape podczas pociągnięcia, aby je anulować, albo użyj cofania i ponawiania po pociągnięciu. To edycje pikseli, a nie regulowane warstwy. Zastosuj jako nowy wynik i wyeksportuj nową pozycję, aby zapisać edytowane piksele; projekt nie przechowuje tych pikseli.

**Dostosuj wynik…** zmienia cały bieżący wynik retuszu. Podgląd próbkowany aktualizuje się przy zmianie ekspozycji i kontrastu; zaznacz **Przed**, aby porównać z wynikiem sprzed otwarcia korekty. Oba elementy używają setnych stopnia: ekspozycja mnoży wartości pikseli, a następnie kontrast skaluje ich odległość od średniej szarości. Ekstremalne wartości mogą obciąć światła lub cienie. **OK** stosuje korektę jako jedną operację do cofnięcia; **Anuluj** pozostawia wynik bez zmian. Podczas przetwarzania „Anuluj” żąda wycofania. Cały obraz musi mieścić się w limicie pamięci historii retuszu dla tej operacji do cofnięcia. Opublikuj i wyeksportuj wynik zgodnie z opisem powyżej.

Pochodzenie podróżuje z wynikiem: wyrównane klatki, przesunięcia przycięcia/wklejenia ROI, przycięcie krawędzi DCT i głębia są zapisywane przy renderowaniu. Pliki wsadowe, starsze wyniki i pozycje folderów tylko wyrównania nie mają zapisanego źródła, więc są oznaczone jako niedostępne, a nie zgadywane z bieżącego stosu. Niezastosowane pociągnięcia i historia cofania istnieją tylko w sesji; zapis projektu ich nie przechowuje.

Dla istniejącego pliku obrazu wybierz **Edycja → Retuszuj pliki obrazów…**, a następnie wskaż wynik i jedną lub więcej klatek źródłowych. Źródła muszą być już wyrównane do wyniku na tej samej siatce pikseli; same pasujące wymiary nie dowodzą wyrównania. Akceptowane są jednostronicowe PNG, TIFF, JPEG i BMP o pasujących wymiarach pionowych, kanałach i głębi 8 lub 16 bitów; orientacja EXIF jest stosowana przy wczytywaniu. Kanały alfa i obrazy wielostronicowe nie są obsługiwane. Wybrane klatki są dostarczone przez użytkownika, a nie zapisanym pochodzeniem renderu. Zastosowanie tworzy osobny niewyeksportowany wynik; zapisz go z listy wyników. Wybrane pliki są chronione przed nadpisaniem w tej sesji, w tym podczas eksportu mapy głębi i siatki 3D. Jak przy innych wynikach, pliki projektów nie przechowują tych pikseli ani historii pędzla.

### Zastępowanie tła (Automatyczne tło)

Podczas stackowania tło wyostrza się razem z obiektem. Tło obrazu ze stacku
bywa więc mozaiką różnych rozmyć, z aureolą wokół obiektu.
**Automatyczne tło…** w obszarze retuszu zastępuje je jedną prawdziwą klatką z
tego samego stacku. Obiekt pozostaje dokładnie taki jak po stackowaniu. Tło
powstaje z klatki i wyniku stacku; przetwarzanie może rekonstruować, korygować
i mieszać jego piksele.

1. Otwórz wynik w obszarze retuszu i kliknij **Automatyczne tło…**.
2. Wybierz tło:
   - **Automatycznie** (zalecane) używa miękkiego tła, chyba że stack
     zarejestrował ostro dużą część tła. Wtedy używa ostrego.
   - **Miękkie tło** używa klatki z najbardziej rozmytym tłem.
   - **Ostre tło** używa klatki z najdalszym ogniskowaniem.
3. Zostaw włączone **Usuń aureole na krawędziach**, aby usunąć aureole wzdłuż
   krawędzi obiektu. Nigdy nie wychodzi poza to, co zarejestrowały klatki.
   Wyłącz, aby zachować krawędzie ze stacku.
4. Zaznacz **Zachowaj miękkie części połączone z obiektem**, gdy części obiektu
   leżą tuż poza głębią ostrości, na przykład główka na końcu kolca. Obiekty tła
   obok obiektu też mogą wtedy zostać jak po stackowaniu, więc sprawdź wynik.
5. Kliknij **Zastąp tło**. Stack 15 klatek po 20 megapikseli trwa około minuty i
   potrzebuje około 3,5 GB wolnej pamięci. **Anuluj** przerywa.

Zmiana to jeden krok, który można cofnąć, a pociągnięcia namalowane wcześniej na
obiekcie zostają. **Pędzlem przywracania** przywrócisz piksele ze stacku tam,
gdzie wolisz, na przykład nieostre źdźbło trawy przed obiektem. Gdy obiekt
wypełnia prawie cały kadr, jak w scenie z dna lasu, obraz zostaje jak po
stackowaniu i pojawia się komunikat.

Po przebiegu **Pokaż brakujące dane** zabarwia miejsca, których żadna klatka nie
uchwyciła wyraźnie. To miejsca do sprawdzenia lub ręcznego retuszu:

- magenta: tło, którego żadna klatka nie zarejestrowała bez obiektu z przodu
- cyjan: obiekt zasłonięty przed aparatem
- żółty: obiekt poza zakresem ostrości stacku
- łupkowy: miejsca zbyt gładkie, by ocenić

**Zapisz mapę brakujących danych…** zapisuje zabarwiony podgląd i jedną
czarno-białą maskę na kolor. Automatyczne tło potrzebuje klatek, z których
powstał wynik, i nie jest dostępne dla wyników obszaru (ROI). Nowe punkty kontrolne
tła dokładnie przywracają obraz, dostępną historię i diagnostykę bez ponownego obliczania tła.

### Zarządzanie etykietami

Otwórz **Narzędzia → Etykiety** po wczytaniu obrazów lub utworzeniu wyniku. Każda etykieta ma własny cel: stos źródłowy, stos zarejestrowany lub wyniki wyjściowe. Etykiety wyników mogą dotyczyć wszystkich wyników albo wybranego wyniku według nazwy; zmiana nazwy lub kolejności tego wyniku nie zmienia powiązania.

Użyj **Dodaj**, aby utworzyć kolejną etykietę, wybierz pozycję do edycji i włączaj lub wyłączaj ją polem wyboru. Wpisz stały tekst albo wstaw `{value}` dla numeru klatki. Numer to wartość początkowa plus indeks klatki od zera pomnożony przez interwał; pojedynczy wynik zawsze używa wartości początkowej.

Wybierz wszystkie klatki, bieżącą klatkę lub własny zakres, np. `3-5,8`. Nieprawidłowe lub niedostępne numery klatek uniemożliwiają zastosowanie szkicu. Bieżący i własny zakres podążają za wybranymi tożsamościami źródeł przy zmianie kolejności stosu; etykiety wszystkich klatek dotyczą też dodanych klatek. Brakujące cele pozostają powiązane z oryginalnymi tożsamościami, dopóki nie wybierzesz jawnie nowego celu.

Podgląd używa rzeczywistego celu. Przechodź przyciskami Poprzednia/Następna lub wpisz numer klatki, wybierz **Dopasuj** albo **100 %**, przeciągaj obraz, aby przesuwać przy 100 %. Położenie i rozmiar czcionki są w pikselach obrazu. O zastępczej czcionce lub nieudanym rysowaniu pojawia się komunikat. Etykiety zachowują oryginalną głębię przy eksporcie do PNG lub TIFF.

**Zastosuj i zamknij** zatwierdza cały szkic jako jedną zmianę do cofnięcia. Anuluj, Esc i zamknięcie okna odrzucają go. Wybór innej etykiety zachowuje zmiany szkicu. **Usuń** usuwa tylko wybraną pozycję, także ostatnią. Pozycje usuwania w menu niezależnie czyszczą etykiety źródeł, stosu zarejestrowanego, wyników lub wszystkie etykiety i również można je cofnąć. Zastosowanie etykiet zmienia wyświetlanie w sesji; aby zapisać adnotacje w pliku, wyeksportuj obraz.

---

## 11. Ustawienia i konfiguracja

Otwórz **Ustawienia → Ustawienia zaawansowane**, aby zmienić wydajność, rejestrację, czyszczenie matrycy, nazwy wyników, AI StackMFF i ustawienia uruchamiania. Zmiany pozostają szkicem do naciśnięcia **OK**; Anuluj, Esc i zamknięcie okna je odrzucają. Zmniejszenie rozmiaru kafelka natychmiast aktualizuje widoczny limit nakładki do połowy kafelka. Zapisany limit partii AI pozostaje, ale wnioskowanie obecnie wysyła jeden kafelek na wywołanie; szerokość skalowania rejestracji może zmienić dokładność wyrównania. Zbieranie diagnostyki wyrównania to pole wyboru w sekcji Rejestracja.

Automatyczne pobieranie aktualizacji działa w bezczynności i nigdy nie instaluje aktualizacji samodzielnie. Ukończone pobieranie pozostaje gotowe, dopóki nie wybierzesz jawnie **Uruchom ponownie i zaktualizuj**. Ponowne uruchomienie chroni aktywne edycje i niezapisaną pracę, a następnie zapisuje dane odzyskiwania przed uruchomieniem zamiennika. Anulowanie pobierania lub przygotowania czeka na bieżącą bezpieczną granicę. Istniejąca kompletna, nowsza przygotowana kompilacja może zostać zainstalowana bez sieci.

Korekcja pola płaskiego zmienia piksele robocze sesji bez nadpisywania plików źródłowych. Wybierz białe odniesienie i klatkę podglądu, a następnie ponownie zbuduj podgląd po zmianie źródła lub skali. Odniesienie i obrazy robocze/bazowe muszą mieć tę samą dziedzinę współrzędnych i rozmiar. Zastosowanie przygotowuje oba stosy przed zatwierdzeniem; opcjonalne usunięcie klatki odniesienia należy do tego samego kroku cofania. Wyeksportuj poprawione obrazy źródłowe, jeśli chcesz zachować te piksele poza sesją.

Przetwarzanie wsadowe przyjmuje opcjonalny obraz białego pola w opcjach przetwarzania. Wybierz rozogniskowane odniesienie w oryginalnej rozdzielczości klatek; jest skalowane razem ze stosem wejściowym i stosowane przed rejestracją, z domyślnym rozmiarem pyłu. Nieczytelne odniesienie lub niezgodny rozmiar jawnie powoduje błąd danego stosu. Odniesienie jest wykluczane z folderów wejściowych, także w przetwarzaniu grupowym; do fuzji muszą pozostać co najmniej dwie klatki źródłowe. Korekcji pola płaskiego nie można łączyć z przezroczystym wejściem. Pozostaw pole puste, aby przetwarzać bez korekcji. CLI przyjmuje odniesienie przez `--dust-map WHITE`.

Czyszczenie matrycy jest opcjonalne i działa na oryginalnych klatkach we współrzędnych matrycy przed rejestracją. W aplikacji desktopowej znajduje się w **Ustawienia → Ustawienia zaawansowane → Czyszczenie matrycy**; okno przetwarzania wsadowego ma te same elementy sterujące. Wybierz jawną maskę defektów, aby poprawić znane gorące lub martwe piksele; OpenFocus nie wnioskuje maski z obrazów preparatu. Maska musi być jednokanałowym obrazem 8- lub 16-bitowym o wymiarach źródła: zero oznacza brak wpływu, a 1 lub maksymalna wartość kanału oznacza defekty. Sprawdź jej zasięg przed przetwarzaniem. Maski pokrywające więcej niż 10 % kadru oraz defekty bez niezamaskowanego sąsiada są odrzucane. Przechowuj maskę w oryginalnej rozdzielczości i nie wybieraj jej jako obrazu źródłowego. W GUI przy używaniu maski wczytuj klatki źródłowe w pełnej skali; maska nigdy nie jest skalowana. Czyszczenia matrycy (w tym odszumiania) nie można zastosować do stosu GUI po obrocie, odbiciu, korekcji pola płaskiego lub innej zmianie źródła: wczytaj ponownie oryginalne klatki albo wyłącz czyszczenie. Dotyczy to także przywróconych projektów z taką historią źródeł. Wstępnie wyrównanego przetwarzania ROI nie można łączyć z czyszczeniem matrycy; wyjdź z trybu ROI i wyrenderuj cały stos. Ustawienia czyszczenia należą do bieżącej sesji i zapisanego projektu, a nie do globalnych preferencji aplikacji. Przetwarzanie wsadowe i CLI przetwarzają swoje oryginalne dane osobno, wykonując czyszczenie matrycy przed korekcją pola płaskiego i rejestracją. Maski defektów wymagają plików źródłowych o neutralnej orientacji EXIF. Pliki z orientacją, RAW aparatów i źródła, których orientacji nie można ustalić, są odrzucane, ponieważ ich zdekodowane piksele mogą nie odpowiadać współrzędnym matrycy. Używaj obrazów źródłowych i maski w tej samej znanej orientacji pikseli, z neutralnymi metadanymi orientacji. Samo odszumianie nie wymaga maski defektów.

Opcjonalne odszumianie matrycy jest niezależne od maski defektów i domyślnie wyłączone. Obejrzyj wynik w pełnym rozmiarze, aby ocenić, czy jego wygładzanie pasuje do preparatu. Czyszczenie matrycy zachowuje pliki źródłowe i głębię 8 lub 16 bitów; jest oddzielone od korekcji pola płaskiego. CLI przyjmuje `--defect-mask-path MASK` i `--sensor-denoise`; przepisy i Python API przepisów zapisują te opcje. Przepisy zapisują odcisk maski i odrzucają brakującą lub zmienioną maskę. Czyszczenia w CLI nie można łączyć z przezroczystym wejściem. Przetwarzanie grupowe stosuje czyszczenie do oryginalnych klatek, a nie do pośrednich wyników grup.

Przepisy wymagają, aby miejsce wyjściowe znajdowało się poza każdym folderem wejściowym, i odrzucają jawnie wybrane wejście jako wyjście. Wybierz osobne miejsce wyjściowe, aby chronić obrazy źródłowe i nie dołączyć wcześniejszego wyniku przy ponownym uruchomieniu. Ustawienia przetwarzania projektu można ponownie wykorzystać jako szablon lub przepis, nawet jeśli jego źródła zostały przeniesione: **Plik → Zapisz szablon przetwarzania…** zapisuje ustawienia bez ścieżek źródeł, **Plik → Zastosuj szablon przetwarzania…** stosuje je do wczytanego stosu, a `openfocus recipe create --project` zapisuje przepis headless, który nadal wymaga jawnych ścieżek `--input`. Szablony przenoszą flagę odszumiania matrycy, ale nie ścieżkę maski defektów; przepisy przechowują ścieżkę maski i jej odcisk. Otwarcie samego projektu nadal wymaga jego źródeł i zapisanego odniesienia pola płaskiego.

Gdy odniesienie plikowe jest stosowane w **Narzędzia → Mapa pyłu (korekcja pola płaskiego)** w skali bazowej stosu, zapisany projekt przechowuje jego ścieżkę bezwzględną, hash zawartości i rozmiar pyłu, a przy otwarciu stosuje je raz do oryginalnych źródeł. Utrzymuj plik odniesienia niezmieniony i na miejscu: brakujące lub zmienione odniesienie uniemożliwia otwarcie projektu. Odniesienie wybrane ze stosu lub korekcja w innej skali roboczej/bazowej nadal wymaga wyeksportowania poprawionego stosu źródłowego przed zapisaniem projektu. Jeśli skala robocza zmieni się po korekcji, przywróć ją do skali korekcji przed zapisaniem albo wyeksportuj poprawiony stos.

**Pomoc → Wyślij opinię** otwiera lokalne logi. Kopiuj kopiuje widoczny log sesji; ZIP opinii zawiera log sesji i pełny log aplikacji oraz informacje o środowisku. Przejrzyj pakiet przed wysłaniem. Otwarcie GitHub tylko otwiera tracker. Anulowanie eksportu pozostawia pliki bez zmian; przy błędzie poprzedni plik docelowy zostaje zachowany, a ścieżka błędu jest pokazywana.

Każde uruchomienie zapisuje własny rotowany log, więc równoległe zadania GUI i CLI prowadzą osobne zapisy. Identyfikatory zadań łączą parametry, czas i wyniki „ukończone/anulowane/błąd”; nieoczekiwane błędy zawierają pełny ślad stosu. ZIP opinii zawiera też bieżący raport awarii i wersje zależności. Użyj **Pomoc → Otwórz folder logów** lub ścieżki na stronie opinii. Logi rotują przy 1 MB z trzema kopiami zapasowymi na proces; nieaktywne logi sesji starsze niż 14 dni są usuwane. Dla szczegółowych logów wsparcia ustaw `OPENFOCUS_LOG_LEVEL=DEBUG` przed uruchomieniem; `OPENFOCUS_LOG_DIR` opcjonalnie wybiera inny folder logów.

### Ustawienia kafelków (optymalizacja pamięci)

Dla dużych obrazów OpenFocus przetwarza kafelki, aby uniknąć problemów z pamięcią.

Otwierane przez **Ustawienia → Ustawienia zaawansowane → Wydajność**:

- **Fuzja kafelkowa dla dużych obrazów**: włącza przetwarzanie kafelkowe; po wyłączeniu parametry poniżej są ukryte
- **Rozmiar kafelka (px)**: rozmiar każdego przetwarzanego kafelka (domyślnie 1024 px)
- **Nakładka kafelków (px)**: nakładka między kafelkami dla bezszwowego mieszania (domyślnie 256 px, nie więcej niż połowa kafelka)
- **Kafelkuj powyżej (px)**: rozmiar obrazu, powyżej którego włączane są kafelki (domyślnie 2048 px)

**Zalecenia**:
- Mniejszy kafelek: mniejsze zużycie pamięci, wolniejsze przetwarzanie
- Większy kafelek: większe zużycie pamięci, szybsze przetwarzanie
- Zwiększ nakładkę, jeśli w wynikach widać szwy

### Ustawienia rejestracji

Otwierane przez **Ustawienia → Ustawienia zaawansowane → Rejestracja**:

- **Szerokość skalowania rejestracji (px)**: rozdzielczość wstępnego przetwarzania rejestracji
  - Domyślnie: 1024 px
  - Mniej — szybciej, więcej — dokładniej
- **Pamięć rejestracji**: przechowuje wyniki obok stosu, aby ponowne otwarcie pomijało wyrównanie
- **Zbieraj diagnostykę wyrównania**: ponownie oblicza wyrównanie na potrzeby raportu dla klatek

### Informacje o wydajności

Zobacz [PERFORMANCE.md](./PERFORMANCE.md): zmierzona przepustowość i zużycie pamięci na dużych rzeczywistych stosach oraz praktyczne wskazówki dla głębokich stosów i danych o wysokiej rozdzielczości.

Panel stanu pokazuje używaną i całkowitą pamięć fizyczną, gdy system operacyjny udostępnia oba odczyty; w przeciwnym razie pokazuje `N/A`.

### Ustawienia liczby wątków

Sterowanie równoległym przetwarzaniem CPU:

Otwierane przez **Ustawienia → Ustawienia zaawansowane → Wydajność**:

- `0` wybiera automatyczne planowanie i jest domyślne dla nowych ustawień.
- Wartości dodatnie są górnymi limitami; zapisane limity przetrwają aktualizacje.
- Możliwości CPU, tryb zasobów i zapas pamięci mogą obniżyć współbieżność. Większa liczba wątków nie zawsze przyspiesza renderowanie.

### Tryb zasobów

**Ustawienia → Ustawienia zaawansowane → Wydajność → Tryb zasobów** decyduje, jaką część maszyny może wykorzystać renderowanie. Nigdy nie zmienia obrazu po fuzji.

| Tryb | Zachowanie |
|---|---|
| Zrównoważony (domyślnie) | Zostawia około 20 % fizycznych rdzeni i zapas pamięci dla pulpitu |
| Responsywny | Używa około połowy rdzeni i mniejszego okna zadań; wolniejsze renderowanie, najbardziej responsywna maszyna |
| Maksymalny | Używa wszystkich fizycznych rdzeni, na ile pozwala budżet pamięci |

Planista uwzględnia fizyczne rdzenie, powinowactwo CPU i limity oraz uruchamia jedną warstwę równoległą naraz. Ciężkie zadania wspólnie przechodzą dopuszczenie przez przygotowanie i publikację; anulowanie czeka na zakończenie aktywnego wywołania natywnego. Nieznana lub niewystarczająca pamięć daje błąd zasobów. Wybrana siatka kafelków pozostaje bez zmian; ręczne zmniejszenie kafelków może zmienić piksele wyniku.

### Akceleracja GPU

StackMFF V1/V2/V3 używają ONNX Runtime. DirectML w Windows i CoreML w macOS mogą przyspieszać wnioskowanie; w przeciwnym razie używany jest CPU. Dostępność dostawcy nie dowodzi wykonania na GPU. Obsługa innych GPU wymaga weryfikacji.

- Przełącznik znajduje się w **Ustawienia → Akceleracja GPU**
- Odznaczenie wymusza metody neuronowe na CPU — przydatne, gdy sterowniki GPU działają niepoprawnie
- Panel stanu pokazuje `N/A`, gdy akcelerator nie jest używany lub niedostępny, oraz `CPU*`, gdy akcelerator istnieje, ale wybrana metoda działa tylko na CPU
- Wybór działa przy następnym renderowaniu i jest pamiętany

### Rozmiar partii AI

Otwierany przez **Ustawienia → Ustawienia zaawansowane → AI**:

- Zapisany limit partii domyślnie wynosi 2; bieżąca polityka wykonania ogranicza faktyczne wnioskowanie do jednego kafelka na wywołanie na każdym urządzeniu.
- Zwiększenie limitu nie włącza niezweryfikowanego przetwarzania wsadowego na GPU ani nie zmniejsza przestrzennego kafelka.

### Motywy

Dostępne są dwa motywy interfejsu: **Jasny (domyślnie)** i **Ciemny**. Przełączanie — **Ustawienia → Motyw**; wybór stosuje się natychmiast (pasek tytułu Windows też się zmienia) i jest pamiętany. Przy pierwszym uruchomieniu motyw wybiera się w oknie powitalnym; zapisany wybór zawsze wygrywa.

### Tryb przenośny

Utwórz pusty plik o nazwie `OpenFocus.portable` obok `OpenFocus.exe` (w przenośnym ZIP już jest) — a wszystkie dane użytkownika (ustawienia, logi, ostatnie pliki) będą przechowywane w folderze `OpenFocusData` obok aplikacji zamiast w profilu użytkownika, więc instalację można przenieść na pendrive. Usuń plik znacznika, aby wrócić do zwykłego profilu.

### Metadane EXIF

Metadane źródła (model aparatu, ekspozycja, orientacja itd.) są zachowywane: orientacja jest stosowana przy wczytywaniu, a eksport PNG/TIFF/WebP automatycznie osadza oryginalny EXIF. Eksport JPG/BMP nie może przechować pełnego EXIF i jest zapisywany bez niego.

### Wejście HEIC/HEIF

Zdjęcia iPhone'a wczytują się bezpośrednio. Wymagany jest pakiet `pillow-heif`: `pip install openfocus` instaluje go, a gotowe kompilacje dołączają do zestawu.

### Aktualizacje

**Pomoc → Sprawdź aktualizacje** porównuje wersję z informacjami o wydaniach na `openfocus.top`. Nowsze wydanie jest pobierane z tej samej strony w tle, gdy aplikacja jest bezczynna; gdy kompilacja jest przygotowana, baner i obszar stanu oferują **Uruchom ponownie i zaktualizuj**, co stosuje ją bez sieci. Kompilacje, które nie potrafią się samodzielnie zaktualizować, otwierają stronę pobierania na witrynie. Automatyczne sprawdzanie odbywa się po cichu najwyżej raz dziennie.

### Pamięć rejestracji

OpenFocus może przechowywać wyniki rejestracji w folderze `.openfocus_cache` obok stosu źródłowego. Ponowne otwarcie tego samego stosu (bez zmian plików) całkowicie pomija wyrównanie. Przełącznik — **Ustawienia → Ustawienia zaawansowane → Rejestracja → Pamięć rejestracji**; nieaktualne wpisy są automatycznie zastępowane.

### Zapamiętywane ustawienia

Wszystko poniższe automatycznie przetrwa restart aplikacji: liczba wątków, tryb zasobów, parametry kafelków, szerokość skalowania rejestracji, rozmiar partii fuzji AI (StackMFF V3), przełącznik akceleracji GPU, przełączniki pamięci rejestracji i zbierania diagnostyki wyrównania, motyw, język interfejsu, rozmiar okna i układ paneli, format eksportu przez przeciąganie, opcje jasności, wygładzania i promienia dla każdej metody oraz ostatnio otwarte stosy. Okna ustawień zapisują się w chwili potwierdzenia; wszystko jest też zapisywane przy zamykaniu aplikacji. Preferencje są przechowywane dla użytkownika w `OpenFocus/OpenFocus.ini` (format INI QSettings).

### Język interfejsu

Przełącz w dowolnym momencie: **Ustawienia → Język** — dostępnych jest dwanaście języków. Zmiana stosuje się natychmiast i jest pamiętana w kolejnych sesjach.

### Automatyczne sprawdzanie aktualizacji

Raz dziennie, kilka sekund po uruchomieniu, OpenFocus po cichu sprawdza `openfocus.top` pod kątem nowszego wydania. Informacje o wydaniu także pochodzą z tej strony. Gdy istnieje nowsza wersja, nad panelami obrazów pojawia się baner: wersja, **Zaktualizuj teraz** (stosuje przygotowaną kompilację lub rozpoczyna pobieranie), **Co nowego** (informacje o wydaniu) i przycisk zamknięcia. Zamknięcie ukrywa baner tylko dla tej wersji — następne wydanie go przywróci, a **Pomoc → Sprawdź aktualizacje** natychmiast go przywraca.

Podczas pobierania w tle pasek stanu pokazuje postęp. Możesz też sprawdzić aktualizacje ręcznie w dowolnym momencie przez **Pomoc → Sprawdź aktualizacje**.

---

## 12. Przetwarzanie wsadowe

Automatyczne przetwarzanie wielu folderów:

1. Wybierz **Narzędzia → Przetwarzanie wsadowe** (`Ctrl+Shift+B`)
2. W oknie przetwarzania wsadowego:
   - **Tryb importu**: „Wiele folderów (jeden stos na folder)” lub „Pojedynczy folder (automatyczny podział na stosy)”
   - **Dodaj foldery**: wybierz foldery z sekwencjami obrazów
   - **Przeciągnij foldery**: upuść jeden lub więcej folderów stosów w dowolnym miejscu okna (lub na liście folderów), aby dodać je do kolejki naraz; upuszczenie włącza tryb wielu folderów, a miniatury pojawiają się podczas dekodowania
   - **Usuń zaznaczone**: usuń wybrane foldery z listy
3. Skonfiguruj wyjście:
   - **Utwórz podfolder w folderze źródłowym**: wyniki powstają w każdym folderze wejściowym
   - **Jak w folderze źródłowym**: zapisuje obok wejścia bez zastępowania istniejących plików
   - **Wskaż folder wyjściowy**: ustaw wspólne miejsce zapisu
4. Sprawdź bieżące parametry fuzji i rejestracji
5. Kliknij **Rozpocznij przetwarzanie wsadowe**

Okno pokazuje postęp w czasie rzeczywistym. Przetwarzanie można anulować w dowolnym momencie.

### Uwagi dotyczące przetwarzania wsadowego

- Przetwarzanie zaczyna się od migawki ustawień głównego okna; kontrolki wsadu je nadpisują.
- Każdy folder lub grupa jest przetwarzany niezależnie przez ten sam potok.
- **Fuzja grupowa** to opcjonalny sposób przetwarzania długich stosów w trybie wielu folderów. Zaznacz **Przetwarzaj duże stosy w grupach**; pola **Rozmiar grupy** (domyślnie 4, co najmniej 2) i **Nakładanie** (domyślnie 1, od 0 do rozmiaru grupy minus 2) są pokazywane tylko przy włączonym polu. Grupy są łączone w rundach z użyciem miejsca tymczasowego na dysku; wskaźnik postępu nazywa każdą rundę. Fuzja, dopasowanie ekspozycji, korekcja bazowa i rejestracja używają własnych klatek i odniesień każdej grupy, w tym rund pośrednich, więc wynik przybliża przetwarzanie całego stosu naraz. Grupowanie wymaga metody fuzji; mapa głębi z oryginalnych klatek, pochodzenie i eksport wyrównanych klatek są niedostępne. Tryb podziału jednego folderu nie jest obsługiwany.
  Wyrównane grupy używają pierwszej klatki jako odniesienia współrzędnych. Każda rejestracja usuwa ostrożny margines na interpolację, a kolejne rundy przecinają zachowane prostokąty, w tym jedno-klatkowy ogon. Ostateczny obraz jest więc przycięty. Każda rejestrowana grupa musi być większa niż 8 pikseli w obu wymiarach; trudne obroty lub przekształcenia rzutowe mogą nie pozostawić bezpiecznie użytecznego prostokąta i spowodować błąd. W takim przypadku użyj wstępnie wyrównanych danych bez rejestracji albo wyłącz grupowanie.
- Wejście grupowe musi mieć pasujące wymiary z uwzględnieniem orientacji i składać się z jednoklatkowych plików z nagłówkami czytelnymi przez Pillow. RAW aparatów, pliki wielostronicowe oraz pliki z kanałami alfa lub przezroczystością nie są obsługiwane, w tym stosy przezroczystych PNG. Usuń kanał alfa lub wyłącz grupowanie dla takich stosów. Próby na sucho CLI sprawdzają te ograniczenia. Przykład dla wiersza poleceń: `python main.py --input stack --output fused.png --group-size 8 --group-overlap 1`. Wejście grupowe CLI musi być folderem i nie można go łączyć z `--depth-map`. Inne odniesienie wyrównania OpenFocus wspólne dla całego stosu nie łączy się z grupowaniem. Odniesienie pola płaskiego jest stosowane do oryginalnych klatek przed rejestracją; wyostrzanie wyniku jest stosowane raz po ostatniej grupie.
- **Co i gdzie trafia, według trybu.** Przy **tylko wyrównaniu** wybrane miejsce docelowe otrzymuje jeden folder `<stack>_aligned` na stos, a panel wyników pokazuje ten folder. Przy **fuzji z zapisem wyrównanych klatek** obraz po fuzji jest zapisywany jako `<stack>.<ext>` w miejscu docelowym, a wyrównane klatki trafiają do `<stack>_aligned` obok; panel wyników pokazuje tylko wynik fuzji. Przy **fuzji bez eksportu wyrównania** zapisywany jest tylko obraz po fuzji. Panel wyników nazywa foldery dokładnie tak, jak na dysku.
- Wyrównane klatki zachowują oryginalne nazwy plików. Zduplikowane nazwy i istniejące foldery otrzymują numeryczne sufiksy. Pliki źródłowe nigdy nie są zastępowane.
- Pozostałe kopie `*_registered*` ze starszych kompilacji nie są wczytywane jako klatki, gdy oryginalny plik jest obecny w tym samym folderze; komunikat wczytywania podaje liczbę pominiętych plików.
- PNG/TIFF zachowują dane 16-bitowe; jawne wybranie JPG/BMP konwertuje je do 8 bitów.
- Anulowanie czeka na zatrzymanie wątku roboczego. Zatwierdzone pliki pozostają dostępne; zadania częściowe, nieudane, anulowane i nieuruchomione są liczone osobno.

---

## 13. Eksport wyników

### Zapisywanie pojedynczego wyniku

1. Wybierz wynik do zapisania na liście wyników
2. Wybierz **Plik → Zapisz** lub naciśnij `Ctrl+S`
3. Wybierz format i miejsce
4. Kliknij Zapisz

W **Ustawienia → Ustawienia zaawansowane → Nazwy wyników** opcjonalny prefiks nazwy pomaga identyfikować nowe wyniki. Przykład aktualizuje się podczas wpisywania; dozwolone jest do 40 liter, cyfr, podkreśleń lub łączników. Pozostaw puste, aby zachować oryginalne nazewnictwo. Potwierdź **OK**, aby zapamiętać prefiks, albo anuluj, aby odrzucić zmianę. Prefiks staje się częścią nowych nazw na liście wyników i sugerowanych nazw eksportu; istniejące wyniki zachowują swoje nazwy. Nazwę nadal można zmienić w oknie zapisu. Ustawienie nie zmienia nazw plików źródłowych ani nazw wyjściowych wsadu.

### Format eksportu przez przeciąganie

Przeciąganie wyników z listy wyników zapisuje pliki obrazów w formacie wybranym w menu kontekstowym listy wyników (**Format przeciągania**: JPG, PNG, WebP lub TIFF). Wybór jest pamiętany.

### Zapisywanie stosu zarejestrowanego

1. Wybierz **Plik → Zapisz stos → Stos zarejestrowany**
2. Wybierz format zapisu:
   - **Zapisz jako folder**: wybierz miejsce i kodowanie; każdy stos otrzymuje nowy podfolder z oryginalnymi nazwami klatek
   - **Zapisz jako GIF**: tworzy animowany GIF wyrównanej sekwencji
   - **Zapisz jako wielostronicowy TIFF**: cały stos trafia do jednego `.tif` z zachowaniem głębi 16 bitów

### Zapisywanie stosu wejściowego

1. Wybierz **Plik → Zapisz stos → Stos wejściowy**
2. Wybierz format zapisu:
   - **Zapisz jako folder**: oryginalne klatki są zapisywane jako osobne obrazy
   - **Zapisz jako GIF**: tworzy animowany GIF sekwencji wejściowej
   - **Zapisz jako wielostronicowy TIFF**: stos wejściowy trafia do jednego `.tif`

### Formaty eksportu

- **JPG**: kompresja, dobry stosunek jakości do rozmiaru
- **PNG**: bezstratny, zachowuje dane 8- i 16-bitowe oraz przezroczystość
- **BMP**: bez kompresji, konwertuje dane 16-bitowe do 8 bitów
- **TIFF**: zachowuje dane 8- i 16-bitowe; eksport wielostronicowy trzyma cały stos w jednym pliku
- **GIF**: animacja z 8-bitową paletą; anulowanie pozostawia istniejący plik docelowy nietknięty

Dla wyniku retuszu wybierz TIFF z warstwami z listy formatów **Zapisz jako**, aby dołączyć klatki źródłowe (patrz rozdział 10).

---

## 14. Interfejs wiersza poleceń

`pip install openfocus` obejmuje model AI, jego wagi oraz dekodowanie HEIC i RAW aparatów.
Na Pythonie 3.13+ pakiet `dtcwt` jest pomijany, a pozycja DTCWT pozostaje wyłączona z podpowiedzią.

OpenFocus może działać bez interfejsu graficznego — przydatne w potokach skryptowych i CI. Skrypt konsolowy `openfocus` (instalowany przez `pip install openfocus`) i `python main.py` z kopii roboczej udostępniają te same polecenia. Opcje fuzji działają w obu formach; podpolecenia:

```bash
openfocus version                       # wyświetl wersję
openfocus describe                      # JSON: metody, parametry, możliwości, polecenia sterujące
openfocus gui [--control] [--socket NAME] [paths…]   # uruchom aplikację desktopową
openfocus ctl <command> [flags]          # steruj działającym oknem (patrz AGENT_CONTROL_SPEC.md)
openfocus recipe create FILE --input … --output … [processing flags]
openfocus recipe run FILE [--dry-run]    # odtwórz zapisany przepis headless
openfocus retouch replay CHECKPOINT --base BASE --sources … --output RESULT
```

`openfocus gui` uruchamia aplikację w tle; `--control` czeka na kanał sterowania i wypisuje JSON połączenia. `openfocus describe` i `version` nie wymagają danych wejściowych. Przepływ pracy dla agenta opisano w [AGENT_CONTROL_SPEC.md](./AGENT_CONTROL_SPEC.md) oraz `skills/openfocus-control/SKILL.md`.

Ze środowiska Pythona (lub obok gotowego pliku wykonywalnego):

```bash
# Fuzja folderu metodą domyślną (StackMFF V3)
python main.py --input ./stack_folder --output ./result/fused.png

# DTCWT z rejestracją ECC, 8 wątków
python main.py -i ./stack_folder -o ./result/fused.png -m dtcwt -a ecc -t 8

# StackMFF V3 wymuszony na CPU z niestandardowym rozmiarem kafelka
python main.py -i ./stack_folder -o ./result/fused.png -m stackmffv3 --cpu --tile-size 512

# Jawna lista plików; pliki wideo są również akceptowane jako wejście
python main.py -i img1.jpg img2.jpg img3.jpg -o fused.png -m gfgfgf

# Jeden wielostronicowy stos Z TIFF z paskiem skali kalibrowanym z metadanych
python main.py -i ./zstack.tif -o fused.png -m dtcwt --scale-bar

# Sprawdź sąsiednie podobne klatki bez fuzji i zmiany wejścia
python main.py -i ./stack_folder --suggest-duplicates
```

`--suggest-duplicates` przyjmuje folder lub uporządkowaną listę jednoklatkowych plików obrazów. Drukuje jeden obiekt JSON z tablicami `files` i `suggestions`; `index` i `reference_index` każdej propozycji to pozycje w `files` liczone od zera, a `difference` to średnia znormalizowana różnica próbki. Dodaj `--json report.json`, aby zapisać ten sam obiekt. Ten tryb przeglądu nie łączy się z `--output`, `--output-dir`, `--json-progress`, `--dry-run` ani opcjami czyszczenia matrycy. Raport porównuje sąsiednie pliki w podanej kolejności, nie wiedząc, które klatki później wykluczysz. Jeśli wykluczysz odniesienie, rozważ ponownie jego następcę albo przejrzyj pozostałe pliki ponownie w zamierzonej kolejności.

Na przykład dodaj `--align openfocus --alignment-report alignment.json` do pojedynczego renderu. Obiekt JSON zawiera `method`, `input`, `frame_count` i `entries`. `frame_index` każdej pozycji jest liczony od zera w kolejności zdekodowanego wejścia, przed sortowaniem do fuzji; obrazy wielostronicowe rozwijają się w kolejności stron, a klatki wideo w kolejności dekodowania. Metody złożone zawierają osobne pozycje dla każdego etapu. Raport jest zapisywany atomowo po udanym przetworzeniu i nie może zastąpić wejścia ani innego wyjścia. Tryb wsadowy, grupowy i próba na sucho odrzucają tę opcję.

| Opcja | Opis |
|--------|-------------|
| `--suggest-duplicates` | Zgłasza prawie identyczne sąsiednie klatki do ręcznego przeglądu; nie łączy ich ani nie wyklucza |
| `--input, -i` | Folder (foldery) obrazów, plik wideo, jeden lub kilka plików obrazów albo wielostronicowy TIFF |
| `--output, -o` | Ścieżka obrazu wyjściowego (rozszerzenie określa format) |
| `--output-dir, -d` | Tryb wsadowy: łączy każdy folder wejściowy, zapisując `<folder>.png` dla każdego stosu |
| `--resume` | Tryb wsadowy: pomija foldery, których plik wyjściowy już istnieje |
| `--config FILE` | Tryb wsadowy: JSON z nadpisaniami dla każdego folderu, np. `{"stackA": {"method": "dct"}}` |
| `--method, -m` | `stackmffv3` (domyślnie), `stackmffv2`, `stackmff`, `openfocus_a`, `openfocus_b`, `openfocus_c`, `guided_filter`, `dct`, `dtcwt`, `gfgfgf` |
| `--param KEY=JSON_VALUE` | Ustawia dowolny parametr fuzji; można powtarzać, nadpisuje starsze flagi |
| `--align, -a` | `none` (domyślnie), `homography`, `ecc`, `both`, `openfocus` |
| `--align-param KEY=JSON_VALUE` | Ustawia dowolny parametr rejestracji; można powtarzać |
| `--align-shift-limit` | Tylko `openfocus`: maksymalne akceptowane przesunięcie jako ułamek rozmiaru klatki (domyślnie 0,08; większe klatki pozostają niewyrównane) |
| `--alignment-report PATH` | Zapisuje wyniki wyrównania klatek w JSON dla pojedynczego, niegrupowego renderu z włączonym wyrównaniem; renderowania wsadowe i grupowe nie są obsługiwane |
| `--exposure-match` | Dopasowuje średnią każdej klatki do ostatniej klatki przed fuzją (domyślnie włączone; `--no-exposure-match` wyłącza). Wzmocnienia wspólne dla całego stosu, więc wynik kafelkowy pozostaje bezszwowy; dotyczy każdej metody i odniesienia jasności wyrównywacza OpenFocus |
| `--base-frame` | Opcjonalny krok po fuzji: dopasowuje niskie częstotliwości wyniku do klatki `last`/`first` (domyślnie `none`) |
| `--sort` | Kolejność klatek dla `openfocus_a/b/c`: `asc` (domyślnie, kolejność plików), `desc` (odwrócenie przemiatania od dalekich do bliskich), `off` |
| `--sharpen-output` | Stosuje łagodne wyostrzanie do wyniku fuzji (domyślnie wyłączone) |
| `--dust-map WHITE` | Korekcja pola płaskiego (mapa pyłu) z rozogniskowanego zdjęcia białego pola |
| `--defect-mask-path MASK` | Maska defektów we współrzędnych matrycy, naprawiana przed rejestracją |
| `--sensor-denoise` | Ostrożne odszumianie matrycy przed rejestracją |
| `--kernel, -k` | Rozmiar jądra (wymuszony nieparzysty); skala odpowiedzi ostrości dla `openfocus_a` i `openfocus_b`; C nie ma promienia |
| `--depth-smoothing VALUE` | Tylko `openfocus_b`: siła wygładzania przejść głębi (domyślnie 1,2) |
| `--blend-smoothing VALUE` | Tylko `openfocus_a` / `openfocus_c`: siła wygładzania przejść (domyślnie 2,0 dla A, 3,0 dla C) |
| `--ai-mode {soft,hard}` | Wyjście StackMFF V3: mieszanie ważone prawdopodobieństwami (domyślnie) lub najbardziej prawdopodobna klatka dla każdego piksela |
| `--threads, -t` | Górny limit wątków roboczych; `0` = automatycznie (domyślnie), wartości dodatnie ograniczają współbieżność |
| `--resource-preference` | `balanced` (domyślnie), `responsive` lub `maximum`; steruje planowaniem zasobów |
| `--batch-size` | Górny limit partii kafelków AI (domyślnie 2); faktyczne wnioskowanie używa obecnie jednego kafelka na wywołanie |
| `--cpu` | Wymusza CPU, nawet gdy dostępny jest GPU |
| `--downscale` | Szerokość skalowania przy wykrywaniu cech rejestracji |
| `--tile-enabled` / `--no-tile-enabled` | Włącza przetwarzanie kafelkowe, gdy jest potrzebne (domyślnie włączone) |
| `--tile-size` | Rozmiar kafelka dla dużych obrazów (domyślnie automatycznie) |
| `--tile-overlap` | Nakładka kafelków w pikselach (domyślnie automatycznie) |
| `--tile-threshold` | Minimalny rozmiar obrazu włączający kafelki (domyślnie automatycznie) |
| `--group-size` | Łączy folder w grupy po tyle klatek; `0` wyłącza grupowanie (domyślnie). Tylko foldery; nie łączy się z `--depth-map`, wejściem RAW/z kanałem alfa/wielostronicowym ani niestandardową klatką odniesienia OpenFocus |
| `--group-overlap` | Klatki wspólne dla kolejnych grup; domyślnie `0`, najwyżej rozmiar grupy minus 2 |
| `--scale-bar` | Rysuje pasek skali (kalibracja automatycznie z metadanych TIFF) |
| `--scale-um` | Rozmiar piksela w µm/px, ważniejszy od metadanych (implikuje `--scale-bar`) |
| `--depth-map [path]` | Dodatkowo zapisuje kolorową mapę położenia ostrości (domyślnie `<output>_depth.png`); tylko dla metod z mapą głębi, nie dla wejścia grupowego lub przezroczystego. W trybie wsadowym dozwolona jest tylko ścieżka domyślna |
| `--depth-colormap` | Mapa kolorów dla `--depth-map`: turbo, viridis, jet, gray, … (domyślnie turbo) |
| `--json FILE` | Zapisuje czytelny maszynowo raport wyników do FILE po zakończeniu |
| `--json-progress` | Wypisuje jeden obiekt JSON postępu na krok do stdout (NDJSON); komunikaty tekstowe trafiają do stderr |
| `--dry-run` | Sprawdza wejście i wypisuje plan bez wykonywania fuzji |

Wartość `--method none` nie istnieje: CLI zawsze wykonuje fuzję. Do wyjścia tylko z wyrównaniem użyj aplikacji desktopowej.

### Przepisy przetwarzania

Przepis to przenośny plik JSON opcji przetwarzania CLI ze ścieżkami względnymi wobec przepisu. Utwórz go z opcji zwykłego uruchomienia, a następnie odtwórz:

```bash
openfocus recipe create line.json --input ./stack --output ./out/fused.tif \
  --method dtcwt --align ecc --threads 8
openfocus recipe run line.json
openfocus recipe run line.json --dry-run
```

`recipe create --project project.ofproj --input ./stack --output ./out/fused.tif` przechwytuje ustawienia przetwarzania projektu dla jawnie wybranych wejść; źródła projektu mogą być przeniesione. Przepisy odrzucają tymczasowe opcje, takie jak `--dry-run`, `--resume`, `--json`, `--json-progress` i `--config`, oraz wymagają, aby wyjście znajdowało się poza każdym folderem wejściowym (i różniło się od jawnie wybranego wejścia). Maska defektów jest przechowywana ścieżką względną z odciskiem SHA-256, więc brakująca lub zmieniona maska powoduje błąd uruchomienia. Edycji źródeł projektu (obrót, odbicie, pole płaskie) nie można przechwycić; najpierw wyeksportuj poprawiony stos.

Przechwycenie projektu przenosi do przepisu CLI ustawienia fuzji, rejestracji,
jasności, kolejności, wyostrzania i kafelków oraz osadza ustawienia dla GUI.
Wykluczenia klatek, skala ładowania, etykiety i edycje nie są przenoszone —
przygotuj nowy stos wejściowy odpowiednio. Opcje grupowania można przechwycić
z argumentów CLI.

Format JSON wersji 1 zawiera `version`, tablicę `input` oraz `output` albo
`output_dir`, a także obiekt `options`. Ścieżki są względne wobec pliku przepisu;
przenoś przepis razem z plikami wejściowymi, aby zachować powiązania. Utworzenie
zapisuje ustawienia CLI i rozwiązane parametry algorytmu. Nieznane pola, nieprawidłowe
wartości i nieobsługiwane wersje są odrzucane. Próba na sucho sprawdza dane wejściowe
bez zapisywania wyników. Istniejące pliki projektów pozostają bez zmian.

W interfejsie **Plik → Zapisz szablon przetwarzania** zapisuje ustawienia bez ścieżek
źródeł. **Plik → Zastosuj szablon przetwarzania** stosuje szablon, projekt lub
przepis oparty na projekcie do bieżącego stosu, nie zastępując jego wejść,
przekształceń, wykluczeń ani etykiet. Sam szablon nie jest wykonywalnym przepisem.
Ustawienia GUI osadzone w przepisie są niezależne od jego opcji CLI; edytując JSON,
utrzymuj spójność obu zestawów.

Ten sam interfejs headless jest dostępny bez importowania PyQt:

```python
from core.recipes import capture_project_recipe, capture_recipe, load_recipe, save_recipe, run_recipe

capture_recipe("workflow.json", ["-i", "stack_folder", "-o", "result/fused.png", "-m", "dct"])
capture_project_recipe("from-project.json", "session.ofproj",
                       ["-i", "new_stack", "-o", "result/fused.png"])
recipe = load_recipe("workflow.json")  # zweryfikowany słownik ze ścieżkami względnymi
save_recipe("workflow.json", recipe)
exit_code = run_recipe("workflow.json", dry_run=True)  # False — wykonaj przetwarzanie
```

`capture_recipe` traktuje swoje argumenty jak zwykłe ścieżki CLI względem bieżącego
katalogu; `save_recipe` traktuje zapisane ścieżki względem przepisu. Przepis
zapisuje ustawienia i odwołania, w tym odcisk zawartości jawnie wybranej maski
defektów sensora. Nie zapisuje pikseli wejściowych, wersji oprogramowania,
sprzętu ani sum kontrolnych stosu źródłowego; przechowuj je osobno, gdy liczy się
odtwarzalność. Identyczne ustawienia nie gwarantują identycznych pikseli między
wersjami oprogramowania, sterownikami i urządzeniami; operacje neuronowe na GPU
mogą być niedeterministyczne. Użyj `--cpu`, gdy wymagane jest wykonanie na CPU, i
zachowaj zainstalowane oprogramowanie oraz dane wejściowe.

### Odtwarzanie punktu kontrolnego retuszu

Użyj przycisków zapisu/wczytywania punktów kontrolnych w przestrzeni retuszu z oryginalną bazą i przetworzonymi źródłami. Sesje z automatycznym tłem, w tym udane analizy bez zmian pikseli, zapisują bezstratne migawki ZIP v2 `.ofretouch` z metadanymi JSON i danymi NumPy: dokładny obraz roboczy, dostępne kafelki cofania/ponawiania, stany diagnostyczne i pochodzenie źródeł. Wczytanie v2 nie oblicza ponownie tła. Sesje obejmujące tylko pędzel/tony mogą używać zwartego JSON v1; starsze punkty kontrolne tła w JSON v1 nadal można odtwarzać.

Weryfikacja źródeł odbywa się w anulowalnej operacji w tle. Przed zastąpieniem pliku docelowego zapis ponownie wczytuje przygotowany plik i sprawdza dokładną równość bieżącego obrazu, dostępnych kafelków cofania/ponawiania i stanów diagnostycznych. Niezgodne źródło, nieudana weryfikacja, anulowanie lub brak pamięci pozostawiają bieżące edycje i istniejący punkt kontrolny bez zmian.

Jeśli limit zapisu punktów kontrolnych uniemożliwia kolejną edycję, zastosuj bieżące edycje i rozpocznij nową sesję retuszu.

Zapisz punkt kontrolny retuszu w przestrzeni retuszu, a następnie podaj oryginalny obraz bazowy i każdą przetworzoną klatkę źródłową w tej samej kolejności co w sesji:

```bash
openfocus retouch replay edits.ofretouch --base fused.png --sources frame1.png frame2.png --output retouched.tif
```

Oba formaty nadal wymagają oryginalnej bazy i wszystkich klatek rzeczywiście użytych do retuszu, w kolejności sesji, po wyrównaniu i dowolnym przycięciu ROI lub innym przetwarzaniu, z oryginalnymi wartościami pikseli i głębią bitową. Wczytanie rekonstruuje zapisane przesunięcia klatek i zachowanie wyrównania oraz weryfikuje bazę, uporządkowane klatki i maski ważności; niezgodne piksele, geometria lub maski są odrzucane. Każda ścieżka musi zawierać jeden obraz. Wyjście musi być PNG lub TIFF i zachowuje głębię bazy. Punkt kontrolny i obrazy wejściowe nie mogą być użyte jako wyjście.

Kody wyjścia: `0` — sukces, `1` — błąd przetwarzania, `2` — błąd użycia. Uruchomienie `python main.py --help` wypisuje wszystkie opcje; uruchomienie z prostą ścieżką (`python main.py C:\stack`) nadal otwiera interfejs graficzny z wczytanym folderem.

---

## 15. Skróty klawiaturowe

### Operacje na plikach

| Skrót | Działanie |
|----------|--------|
| `Ctrl+O` | Otwórz folder |
| `Ctrl+Shift+O` | Otwórz wideo |
| `Ctrl+Alt+O` | Otwórz projekt |
| `Ctrl+Alt+S` | Zapisz projekt |
| `Ctrl+S` | Zapisz wynik |
| `Ctrl+Shift+S` | Zapisz stos zarejestrowany |
| `Ctrl+W` | Wyczyść stos |
| `Ctrl+Q` | Wyjście |
| `Ctrl+R` | Rozpocznij renderowanie |
| `Ctrl+Shift+R` | Renderuj wszystkimi metodami |
| `Ctrl+T` | Retusz wyniku |
| `Ctrl+M` | Mapa głębi |
| `Ctrl+E` | Przełącz porównanie suwakiem |
| `Ctrl+Shift+B` | Przetwarzanie wsadowe |
| `Ctrl+Z` / `Ctrl+Y` | Cofnij / ponów (zmiany stosu) |

### Nawigacja

| Skrót | Działanie |
|----------|--------|
| `Up` / `Down` (fokus na liście źródeł) | Poprzednia / następna klatka |
| `Mouse Wheel` | Powiększ/zmniejsz |
| `Ctrl + Wheel` | Dokładne powiększanie |
| `Double-click` | Przełącz widok dopasowany / 100 % |

### Ogólne

| Skrót | Działanie |
|----------|--------|
| `Space` (przytrzymaj, wskaźnik nad obrazem) | Tymczasowy tryb przesuwania |
| `Delete` | Usuń pozycję listy w fokusie; czyści też ROI, gdy wskaźnik jest nad obrazem źródłowym |

---

## 16. Rozwiązywanie problemów

### Typowe problemy

#### „Brak wczytanych obrazów”

- Upewnij się, że najpierw zaimportowano folder z obrazami lub wideo
- Sprawdź, czy folder zawiera obsługiwane formaty obrazów

#### Wyniki fuzji są rozmyte lub mają duchy

- Włącz rejestrację (wybierz ECC, homografię, homografię → ECC lub OpenFocus)
- Upewnij się, że obrazy są prawidłowo wyrównane przed fuzją
- Wypróbuj inną metodę fuzji
- Zwiększ rozmiar jądra dla gładszego wyniku
- Sprawdź **Ustawienia → Ustawienia zaawansowane → Rejestracja → Diagnostyka wyrównania**, włączając zbieranie w **Ustawienia → Ustawienia zaawansowane → Rejestracja**

#### AI niedostępne

- Sprawdź ONNX Runtime i dołączony model.
- Sprawdź przez **Pomoc → Informacje o środowisku**
- StackMFF V3 nie wymaga torchvision

#### Błędy braku pamięci

- Włącz przetwarzanie kafelkowe w **Ustawienia → Ustawienia zaawansowane → Wydajność**
- Zmniejsz rozmiar kafelka
- Używaj mniejszych jąder
- Zmniejsz rozdzielczość obrazów
- Wybierz bardziej responsywny tryb zasobów

#### Rejestracja nie udaje się

- Użyj **OpenFocus (szybko)** dla ograniczonego ruchu sztywnego albo włącz **Zatrzymaj przy odrzuconej klatce**, aby zakończyć się błędem wcześniej, zamiast akceptować niewyrównane klatki
- Zmniejsz szerokość skalowania w **Ustawienia → Ustawienia zaawansowane → Rejestracja**
- Upewnij się, że obrazy mają wystarczający kontrast i charakterystyczne punkty
- Włącz **Zbieraj diagnostykę wyrównania**, wyrenderuj ponownie i otwórz **Ustawienia → Ustawienia zaawansowane → Rejestracja → Diagnostyka wyrównania**, aby zobaczyć problematyczne klatki

#### Wolne przetwarzanie

- Pozostaw wątki w trybie automatycznym (`0`): dostępna moc CPU, tryb zasobów i zapas pamięci już ograniczają współbieżność
- Do nienadzorowanego przetwarzania wsadowego wybierz tryb zasobów „Maksymalny” albo użyj filtra sterowanego / GFG-FGF zamiast metody AI
- Zmniejsz rozdzielczość obrazów lub używaj **Szybkiego podglądu**, gdy dobierasz parametry

#### Ostrzeżenie Windows SmartScreen przy pierwszym uruchomieniu

Podpis kompilacji Windows zależy od konkretnego wydania (zobacz informacje o wydaniu); podpis nie jest gwarantowany. Pobieraj je z **https://openfocus.top**, a nie z lustra strony trzeciej. Jeśli SmartScreen ostrzega, kliknij **Więcej informacji → Uruchom mimo to**. Pierwsze uruchomienie po rozpakowaniu może potrwać 30–60 sekund.

#### macOS zgłasza, że aplikacji „nie można otworzyć” lub że „jest uszkodzona”

Kompilacja macOS nie jest notaryzowana. Pobierz kompilację dla Apple Silicon z **https://openfocus.top**. Jeśli instalujesz z DMG, przenosząc `OpenFocus.app` do Programów, wykonaj:

```bash
xattr -cr /Applications/OpenFocus.app
```

Jeśli instalujesz z terminala za pomocą `install.sh`, aplikacja znajduje się w `~/Applications`; wykonaj `xattr -cr ~/Applications/OpenFocus.app`.

albo kliknij aplikację prawym przyciskiem i wybierz **Otwórz**. Opublikowane kompilacje macOS są przeznaczone dla Apple Silicon (M1/M2/M3/M4); na Macu z Intel można uruchamiać ze źródeł.

### Uzyskiwanie pomocy

1. **Informacje o środowisku**: otwórz **Pomoc → Informacje o środowisku**, aby sprawdzić instalację
2. **Wyślij opinię**: **Pomoc → Wyślij opinię…** otwiera lokalne logi; skopiuj log lub wyeksportuj ZIP i dołącz go do zgłoszenia
3. **Kontakt**: otwórz **Pomoc → Kontakt**, aby uzyskać informacje o wsparciu
4. **Logi**: **Pomoc → Otwórz folder logów**

---

## Dodatek: szczegóły algorytmów fuzji

### Filtr sterowany

Filtr sterowany wykonuje wygładzanie z zachowaniem krawędzi, używając obrazu sterującego. Oblicza lokalne przekształcenia liniowe między obrazem sterującym a wejściowym, a następnie stosuje je do wyniku. Metoda dobrze zachowuje krawędzie, jednocześnie tłumiąc szum.

### DCT (dyskretna transformata kosinusowa)

DCT przekształca obrazy do dziedziny częstotliwości, gdzie decyzje o fuzji są podejmowane na podstawie amplitud współczynników. Składowe wysokoczęstotliwościowe (szczegóły) są zachowywane, a niskoczęstotliwościowe (gładkie obszary) łączone według lokalnej energii.

### DTCWT (dual-tree complex wavelet transform)

DTCWT daje wieloskalowy i wielokierunkowy rozkład obrazów. Rejestruje cechy w różnych skalach i orientacjach, co pozwala stosować zaawansowane reguły fuzji zachowujące zarówno teksturę, jak i strukturę.

### GFG-FGF

Implementuje metodę Fu i in. (2024): zgrubna mapa ostrości |I − mean(I)| jest wzbogacana gradientem quasi-gaussowskim w czterech sąsiedztwach — szablon 3×3 T = [1 2 1; 2 4 2; 1 2 1]/16 daje P = I⊗T, a G = (|I(x±1,y) − P|)/2 + (|I(x,y±1) − P|)/2, przy czym wartości poniżej T0 = 0,005 są odrzucane — a następnie udokładniana szybkim filtrem sterowanym. Piksele wybierają klatkę o największej wartości ostrości (remisy rozstrzyga miara małego obszaru Sobela 3×3), a binarne mapy decyzji są wygładzane szybkim filtrem sterowanym przed ważonym mieszaniem.

### Fuzja AI (StackMFF V3)

Podejście uczenia głębokiego z siecią neuronową trenowaną na dużych zbiorach par obrazów wieloogniskowych. Sieć sama uczy się optymalnych strategii fuzji i przy prawidłowej konfiguracji daje wyniki na poziomie najlepszych współczesnych metod.

---

*OpenFocus to projekt open source. Aktualizacje, zgłoszenia błędów i propozycje nowych funkcji — w repozytorium projektu.*
