<p align="center">
  <a href="https://github.com/Xinzhe99/OpenFocus">
    <img src="../../assets/OpenFocus.png" alt="OpenFocus" width="128">
  </a>
</p>

<h1 align="center">OpenFocus</h1>

<p align="center"><strong>Każdy piksel z najostrzejszej klatki.</strong></p>

<p align="center">
  Dziesięć algorytmów fuzji, tryb tylko wyrównania, cztery metody rejestracji i przestrzeń retuszu powiązana ze źródłami dla makrofotografii, mikroskopii i stosów przemysłowych. 16 bitów na całej ścieżce, headless CLI i interfejs w dwunastu językach.
</p>

<p align="center">
  <a href="https://github.com/Xinzhe99/OpenFocus/releases"><img src="https://img.shields.io/github/v/release/Xinzhe99/OpenFocus?label=release&color=2f81f7" alt="Release"></a>
  <a href="https://github.com/Xinzhe99/OpenFocus/stargazers"><img src="https://img.shields.io/github/stars/Xinzhe99/OpenFocus?color=e3b341" alt="Stars"></a>
  <a href="../../LICENSE"><img src="https://img.shields.io/badge/license-MIT-3fb950" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Windows-10%2F11-0078D7?logo=windows&logoColor=white" alt="Windows 10/11">
  <img src="https://img.shields.io/badge/macOS-Apple%20Silicon-000000?logo=apple&logoColor=white" alt="macOS Apple Silicon">
  <img src="https://img.shields.io/badge/Linux-CLI-FCC624?logo=linux&logoColor=black" alt="Linux CLI">
  <a href="https://pypi.org/project/openfocus/"><img src="https://img.shields.io/badge/PyPI-openfocus-3775A9?logo=pypi&logoColor=white" alt="PyPI"></a>
</p>

<p align="center">
  <a href="#install">Pobierz</a> ·
  <a href="../USER_MANUAL_PL.md">Dokumentacja</a> ·
  <a href="../../CHANGELOG.md">Historia zmian</a> ·
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
  <a href="README.it.md">Italiano</a> ·
  <a href="README.nl.md">Nederlands</a> ·
  <strong>Polski</strong>
</p>

<p align="center">
  <img src="../../assets/ui.png" alt="Interfejs OpenFocus" width="880">
</p>

## 📢 Aktualności

> [!NOTE]
> **v1.74.0** — Zaktualizowano fuzję OpenFocus A/B/C i szybkie wyrównanie.
> Aplikacja desktopowa uruchamia teraz automatycznie uwierzytelniony lokalny
> kanał agenta, więc `openfocus ctl` może sterować działającym oknem bez
> specjalnej flagi uruchomienia.

Wcześniejsze wydania opisuje [historia zmian](../../CHANGELOG.md).

📸 Jeśli publikujesz obrazy utworzone w OpenFocus, rozważ dodanie notki:

> Utworzono w OpenFocus — https://openfocus.top

Nie jest to obowiązkowe, ale bardzo mile widziane.

<a id="features"></a>
## Funkcje

<table>
<tr>
<td width="46%" valign="middle">

### Fuzja AI: StackMFF V3

Domyślna metoda wykorzystuje sieć neuronową do oszacowania najbardziej
prawdopodobnej klatki źródłowej dla każdego piksela, a następnie miesza wkłady
klatek lub wybiera najbardziej prawdopodobną klatkę. Benchmark syntetyczny
porównuje metody ze znanym wzorcem; wyniki zależą od stosu i ustawień.

[StackMFF V3 →](https://github.com/Xinzhe99/StackMFF-V3) · [Benchmark →](../../tools/synthetic_benchmark.py)

</td>
<td width="54%">

<a href="../../assets/readme/ai.mp4"><picture><source srcset="../../assets/readme/ai.gif" type="image/gif"><img src="../../assets/readme/ai_poster.jpg" alt="StackMFF V3 i Guided Filter na tym samym stosie" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Porównanie suwakiem

Przeciągnij dzielnik po klatce źródłowej i wyniku fuzji, aby dokładnie zobaczyć,
co zmieniło mieszanie: przesunięcie wyrównania, obwódki na krawędziach i utracone
szczegóły są widoczne od razu, przy dowolnym powiększeniu.

[Podręcznik →](../USER_MANUAL_PL.md)

</td>
<td width="54%">

<a href="../../assets/readme/wipe.mp4"><picture><source srcset="../../assets/readme/wipe.gif" type="image/gif"><img src="../../assets/readme/wipe_poster.jpg" alt="Porównanie suwakiem klatki źródłowej i wyniku" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Porównanie metod

Wyrenderuj wszystkie metody za jednym razem, obejrzyj je obok siebie na wspólnym
fragmencie, a następnie przyjmij zwycięzcę jako nowy wynik albo przenieś jego
parametry z powrotem do głównego okna.

</td>
<td width="54%">

<a href="../../assets/readme/method-comparison.mp4"><picture><source srcset="../../assets/readme/method-comparison.gif" type="image/gif"><img src="../../assets/readme/method-comparison_poster.jpg" alt="Osiem metod fuzji obok siebie" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Mapa głębi i 3D

Położenie ostrości dla każdego piksela według siedmiu miar, naukowe palety barw,
kalibracja fizycznej osi Z z metadanych TIFF oraz eksport teksturowanej siatki OBJ.

</td>
<td width="54%">

<a href="../../assets/readme/depth.mp4"><picture><source srcset="../../assets/readme/depth.gif" type="image/gif"><img src="../../assets/readme/depth_poster.jpg" alt="Mapa głębi i powierzchnia 3D tego samego stosu" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Retusz powiązany ze źródłami

Maluj pikselami z klatek, które naprawdę utworzyły wynik: wspólne powiększenie
i przesuwanie, suwak klatki źródłowej, miękki pędzel, pędzel przywracania,
cofanie całych pociągnięć i ochrona niewyeksportowanych pikseli.
Zastosuj retusz jako nowy wynik, a następnie kliknij go prawym przyciskiem i
wybierz **Eksportuj TIFF z warstwami…**, aby zapisać go razem z klatkami
źródłowymi. Etykiety na żywo i paski skali nie są dołączane. GIMP 3.2.6 w systemie Windows
otwiera wynik i warstwy źródeł; inne edytory nie były testowane.

</td>
<td width="54%">

<a href="../../assets/readme/retouch.mp4"><picture><source srcset="../../assets/readme/retouch.gif" type="image/gif"><img src="../../assets/readme/retouch_poster.jpg" alt="Przestrzeń retuszu z klatką źródłową i wynikiem" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Przetwarzanie wsadowe

Wskaż folder ze stosami, wybierz metodę i miejsce docelowe, a potem zostaw
program. Tryb tylko wyrównania zapisuje jeden folder na stos; fuzja zapisuje
obraz wynikowy i pokazuje w liście wyłącznie jego. Opcjonalna fuzja grupowa
wymienia dokładną równoważność z przetwarzaniem całego stosu na ograniczone
zużycie pamięci i miejsca tymczasowego na dysku.

</td>
<td width="54%">

<a href="../../assets/readme/batch.mp4"><picture><source srcset="../../assets/readme/batch.gif" type="image/gif"><img src="../../assets/readme/batch_poster.jpg" alt="Okno przetwarzania wsadowego" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Paski skali publikacyjnej

Kalibrowane z metadanych TIFF lub wpisywane ręcznie, rysowane prawdziwą czcionką
w rzeczywistej długości i wpalane w każdy eksport — pojedyncze obrazy, stosy,
wielostronicowy TIFF, GIF i wyniki wsadowe. Jeśli pasek nie mieści się czytelnie,
program to zgłasza zamiast drukować długość, której nie rysuje.

[Podręcznik →](../USER_MANUAL_PL.md)

</td>
<td width="54%">

<a href="../../assets/readme/scalebar.mp4"><picture><source srcset="../../assets/readme/scalebar.gif" type="image/gif"><img src="../../assets/readme/scalebar_poster.jpg" alt="Okno paska skali z podglądem na żywo" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Etykiety i adnotacje

Prowadź zarządzaną listę etykiet; każda ma własny zakres (stos źródłowy, stos
wyrównany lub konkretny wynik), zakres klatek i styl. Podgląd i każdy eksport
używają tego samego celu i numeru klatki, więc `{value}` czyta się tak samo w
„Zapisz”, „Zapisz jako”, przeciąganiu i „Eksportuj wszystko”.

</td>
<td width="54%">

<a href="../../assets/readme/labels.mp4"><picture><source srcset="../../assets/readme/labels.gif" type="image/gif"><img src="../../assets/readme/labels_poster.jpg" alt="Menedżer etykiet z podglądem na żywo" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Sterowanie przez agenta

Każde okno, widżet i menu są osiągalne przez lokalny kanał chroniony tokenem:
agent ładuje stos, renderuje, retuszuje, porównuje, oblicza głębię, eksportuje i
odpowiada na okna dialogowe — poleceniem typowanym, ścieżką widżetu albo
rzeczywistymi zdarzeniami myszy i klawiatury. Nie potrzeba odczytywania ekranu
ani OCR; kanał startuje razem z aplikacją desktopową.

[Protokół →](../AGENT_CONTROL_SPEC.md) · [Umiejętność →](../../skills/openfocus-control/SKILL.md)

</td>
<td width="54%">

<a href="../../assets/readme/agent.mp4"><picture><source srcset="../../assets/readme/agent.gif" type="image/gif"><img src="../../assets/readme/agent_poster.jpg" alt="Agent steruje działającym interfejsem: ładowanie, render AI, porównanie, retusz, suwak, głębia, etykiety" width="100%" /></picture></a>

</td>
</tr>
</table>

**W zestawie także:**

- **Przepływ pracy ze stosem ostrości** — przeglądarka stosu, krzywa ostrości,
  szybki podgląd, render ROI i zapamiętywane ustawienia każdej metody między
  uruchomieniami.
- **Gotowość do mikroskopii** — 16 bitów na całej ścieżce, µm/px z metadanych
  TIFF, korekcja pola płaskiego z mapy pyłu, wielostronicowe stosy Z TIFF, eksport
  teksturowanego OBJ.
- **Profile kolorów** — eksport pojedynczych obrazów PNG, TIFF, JPEG i WebP
  zachowuje wspólny profil RGB ICC, gdy wszystkie klatki źródłowe mają ten sam
  profil. Stosy mieszane lub bez profilu są zapisywane bez profilu ICC; piksele nie
  są konwertowane do wspólnej przestrzeni barw.
- **Przezroczysty PNG** — stosy z identycznymi prostokątnymi maskami stałej
  nieprzezroczystości można połączyć w pełnym rozmiarze i zapisać jako PNG.
  Ograniczenia opisuje podręcznik.
- **Headless CLI** — ten sam potok jako `openfocus`, z trybem wsadowym,
  `--dry-run`, `--resume` i wyjściem JSON dla skryptów.
- **Sterowanie przez agenta** — lokalny kanał chroniony tokenem steruje działającym
  interfejsem (typowe polecenia potoku oraz uniwersalna kontrola widżetów, menu,
  myszy i klawiatury), z umiejętnością, runnerem smoke z 14 sprawdzeniami i pełnym
  przemarszem po akcjach menu.
- **Projekty i odzyskiwanie** — sesje `.ofproj`, automatyczne odzyskiwanie po
  awarii, cofanie i ponawianie zmian stosu oraz adnotacji.
- **Dwanaście języków interfejsu** — English, 中文, 日本語, Español, Deutsch,
  Français, Русский, 한국어, Português, Italiano, Nederlands, Polski.
  Wykrywane z ustawień systemowych; nowy język to jeden plik `locales/<code>.py`.
- **Oszczędność pamięci** — sieć neuronowa jest ładowana tylko przy użyciu metody
  neuronowej, więc sesja z metodą klasyczną zajmuje około 110 MB zamiast 260 MB.

<a id="install"></a>
## ⬇️ Instalacja

Pobierz najnowszą kompilację dla Windows lub macOS ze [strony OpenFocus](https://openfocus.top/#download). Te same pliki i starsze wersje są dostępne na [GitHub Releases](https://github.com/Xinzhe99/OpenFocus/releases).

| Platforma | Plik | Typ |
|-----------|------|-----|
| Windows 10/11 (64 bity) | `OpenFocus-*-setup.exe` | **Instalator** (zalecany) |
| Windows 10/11 (64 bity) | `OpenFocus-*-windows-x64.zip` | Wersja przenośna (rozpakuj i uruchom) |
| macOS Apple Silicon (M1–M4) | `OpenFocus-*-macos.dmg` | Obraz dysku (przeciągnij do Programów) |
| macOS Apple Silicon (M1–M4) | `OpenFocus-*-macos-arm64.zip` | Wersja przenośna (rozpakuj) |

Aplikacja nie jest podpisana, więc przy pierwszym uruchomieniu może pojawić się
ostrzeżenie: **Windows** SmartScreen → „Więcej informacji” → „Uruchom mimo to”
(pierwsze uruchomienie może potrwać 30–60 s); **macOS** kliknij prawym przyciskiem →
„Otwórz” albo wykonaj `xattr -cr /Applications/OpenFocus.app` w Terminalu.

Aby zainstalować z terminala, najpierw obejrzyj polecenie dla macOS lub Linux z
flagą `--dry-run`, a potem uruchom je bez niej. W systemie Linux instaluje się w ten sposób
headless CLI; kompilacje macOS wymagają Apple Silicon i są instalowane w
`~/Applications`.

```bash
curl -fsSL https://openfocus.top/install.sh | sh -s -- --dry-run
curl -fsSL https://openfocus.top/install.sh | sh
```

W systemie Windows uruchom instalator z PowerShell:

```powershell
irm https://openfocus.top/install.ps1 | iex
```

Bezpośrednie pobranie instalatora Windows jest też dostępne przez
`curl -fL https://openfocus.top/download/latest/windows-setup -o OpenFocus-setup.exe`.

### 🤖 Integracja z agentami

Aplikacja desktopowa automatycznie uruchamia lokalny kanał agenta chroniony
tokenem. Użyj `openfocus ctl` do poleceń typowanych lub ogólnych działań na
działającym oknie. Agenci mogą ładować stosy, renderować, eksportować, otwierać i
wypełniać okna dialogowe, odpowiadać na wybór plików, wysyłać zdarzenia myszy i
klawiatury oraz robić zrzuty ekranu.

- Umiejętność: [`skills/openfocus-control/SKILL.md`](../../skills/openfocus-control/SKILL.md)
- Protokół i opis poleceń: [`docs/AGENT_CONTROL_SPEC.md`](../AGENT_CONTROL_SPEC.md)
- Runner end-to-end: `python tools/agent_gui_smoke.py --launch`
- Przemarsz po wszystkich akcjach: `python tools/agent_action_sweep.py --launch`

### 💻 Korzystanie z wiersza poleceń

Możesz też zainstalować CLI w systemach Windows, macOS i Linux — ten sam potok bez warstwy GUI:

```bash
pip install openfocus
openfocus -i ./stack_folder -o ./result/fused.png
```

Zobacz [podręcznik CLI](../USER_MANUAL_PL.md#14-interfejs-wiersza-poleceń).

<a id="community"></a>
## Społeczność i wsparcie

- **Dokumentacja** — [podręcznik użytkownika](../USER_MANUAL_PL.md) (dwanaście języków) · [historia zmian](../../CHANGELOG.md) · [wydajność](../PERFORMANCE.md) · [metody fuzji](../FUSION_METHODS.md)
- **Cytowanie** — [CITATION.cff](../../CITATION.cff); każde wydanie jest archiwizowane na Zenodo ([DOI](https://doi.org/10.5281/zenodo.23050823))
- **Opinie** — [issue](https://github.com/Xinzhe99/OpenFocus/issues) dla błędów, pomysłów i prac nad algorytmami
- **Podziękowania** — każdy algorytm fuzji i rejestracji zachowuje swoją oryginalną licencję badawczą; StackMFF V3 mieszka w [Xinzhe99/StackMFF-V3](https://github.com/Xinzhe99/StackMFF-V3)

<a id="license"></a>
## Licencja

Aplikacja OpenFocus jest udostępniana na [licencji MIT](../../LICENSE).

**Wyjątek dla serii StackMFF.** Algorytmy fuzji StackMFF — obecnie StackMFF V3 i
wszystkie przyszłe wydania klasy state-of-the-art — pozostają własnością projektu
OpenFocus, który zachowuje wszelkie prawa do serii StackMFF. Możesz używać ich w
ramach OpenFocus do własnych obrazów i pracy wewnętrznej. Kopiowanie, portowanie
lub przenoszenie tych algorytmów do innego projektu open source wymaga uprzedniej
pisemnej zgody.

<a id="developing"></a>
## Rozwój

Uruchomienie z repozytorium:

```bash
conda create -n openfocus python=3.10
conda activate openfocus
pip install -r requirements.txt
python main.py
```

Pakowanie przez PyInstaller: [docs/BUILD_COMMANDS.md](../BUILD_COMMANDS.md).

Dodawanie algorytmu fuzji lub rejestracji: [CONTRIBUTING.md](../../CONTRIBUTING.md).
[Specyfikacja rozszerzalności](../ALGORITHM_EXTENSIBILITY_SPEC.md) określa wspólne
kontrakty rejestracji, parametrów, wykonania i zapisu.

<a id="star-history"></a>
## ⭐ Historia gwiazdek

[![Star History Chart](https://api.star-history.com/svg?repos=Xinzhe99/OpenFocus&type=Date)](https://star-history.com/#Xinzhe99/OpenFocus&Date)
