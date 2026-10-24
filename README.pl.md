<p align="center">
  <a href="https://github.com/Xinzhe99/OpenFocus">
    <img src="assets/OpenFocus.png" alt="OpenFocus" width="128">
  </a>
</p>

<h1 align="center">OpenFocus</h1>

<p align="center"><strong>Każda płaszczyzna ostrości. Jeden w pełni ostry obraz.</strong></p>

<p align="center">
  Osiem algorytmów fuzji, trzy tryby rejestracji i obszar retuszu powiązany ze źródłem dla makro, mikroskopii i stosów przemysłowych. 16 bitów na całej ścieżce, bezgłowy CLI i interfejs w dwunastu językach.
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
  <a href="#download">Pobierz</a> ·
  <a href="./docs/USER_MANUAL_PL.md">Dokumentacja</a> ·
  <a href="#command-line-usage">CLI</a> ·
  <a href="./CHANGELOG.md">Zmiany</a> ·
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
  <a href="README.it.md">Italiano</a> ·
  <a href="README.nl.md">Nederlands</a> ·
  <strong>Polski</strong>
</p>

<p align="center">
  <img src="assets/ui.png" alt="Interfejs OpenFocus" width="880">
</p>

## 📢 News

> [!NOTE]
> **v1.68.1** — eksport wsadowy działa według wybranego trybu: samo wyrównanie tworzy folder `<stos>_aligned` i pokazuje go na liście, fuzja pokazuje tylko wynik, a bez eksportu wyrównania zapisywany jest wyłącznie obraz wynikowy. Pozostałe w folderze źródłowym kopie `*_registered*` nie są już wczytywane jako klatki. Przepracowano też etykiety, paski skali, porównanie metod, eksport głębi/modelu, aktualizacje i retusz zgodnie z audytem w `docs/ux-specs`.

<a id="download"></a>
## Pobieranie i instalacja

Pobierz wersję ze strony [Releases](https://github.com/Xinzhe99/OpenFocus/releases):

| Platforma | Plik | Typ |
|-----------|------|-----|
| Windows 10/11 (64-bit) | `OpenFocus-*-setup.exe` | **Instalator** (zalecany) |
| Windows 10/11 (64-bit) | `OpenFocus-*-windows-x64.zip` | Przenośna (rozpakuj i uruchom) |
| macOS Apple Silicon (M1–M4) | `OpenFocus-*-macos.dmg` | Obraz dysku (przeciągnij do Programów) |
| macOS Apple Silicon (M1–M4) | `OpenFocus-*-macos-arm64.zip` | Przenośna (rozpakuj) |

Aplikacja nie jest podpisana. Podczas pierwszego uruchomienia Windows może pokazać „Więcej informacji → Uruchom mimo to”; w macOS kliknij prawym przyciskiem → Otwórz lub użyj `xattr -cr /Applications/OpenFocus.app`.

## Najważniejsze funkcje

<table>

<tr>
<td width="46%" valign="middle">

### Fuzja AI: StackMFF V3

Domyślna metoda klasyfikuje każdy piksel do najbardziej prawdopodobnej klatki źródłowej za pomocą sieci neuronowej. Na tym samym wyrównanym stosie krawędzie liter pozostają ostrzejsze, a powierzchnie czystsze niż w Guided Filter (ostrość laplasjanowa 27,1 vs 28,5); wobec syntetycznej prawdy prowadzi w PSNR i SSIM i zostawia mniej duchów niż jakikolwiek klasyczny algorytm.

[Algorytmy →](#algorithms) · [Benchmark →](tools/synthetic_benchmark.py)

</td>
<td width="54%">

<img src="assets/comparison_ai_vs_gff.png" alt="Fuzja AI: StackMFF V3" width="100%" />

</td>
</tr>

<tr>
<td width="46%" valign="middle">

### Porównanie A/B

Przeciągnij podziałkę po klatce źródłowej i wyniku, aby zobaczyć, co zmieniła fuzja: dryf wyrównania, obwódki na krawędziach i utracony detal, przy dowolnym powiększeniu.

[Podręcznik →](./docs/USER_MANUAL_PL.md)

</td>
<td width="54%">

<img src="assets/readme/wipe.png" alt="Porównanie A/B klatki i wyniku" width="100%" />

</td>
</tr>

<tr>
<td width="46%" valign="middle">

### Porównanie metod

Wyrenderuj wszystkie metody raz, porównaj je obok siebie na tym samym kadrze i przyjmij zwycięzcę jako nowy wynik lub skopiuj jego parametry do okna głównego.

</td>
<td width="54%">

<img src="assets/readme/compare.png" alt="Osiem metod fuzji obok siebie" width="100%" />

</td>
</tr>

<tr>
<td width="46%" valign="middle">

### Mapa głębi i 3D

Pozycja ostrości na piksel z pięciu miar, naukowe palety, fizyczna kalibracja Z z metadanych TIFF i eksport teksturowanego OBJ powierzchni.

</td>
<td width="54%">

<img src="assets/readme/depth.png" alt="Mapa głębi i powierzchnia 3D tego samego stosu" width="100%" />

</td>
</tr>

<tr>
<td width="46%" valign="middle">

### Retusz powiązany ze źródłem

Maluj z klatek, które naprawdę utworzyły wynik: wspólny zoom i przesuwanie, suwak klatki źródłowej, miękki pędzel, pędzel przywracania, cofanie na pociągnięcie i ochrona niewyeksportowanych pikseli.

</td>
<td width="54%">

<img src="assets/readme/retouch.png" alt="Obszar retuszu z klatką i wynikiem" width="100%" />

</td>
</tr>

<tr>
<td width="46%" valign="middle">

### Przetwarzanie wsadowe

Wskaż folder ze stosami, wybierz metodę i miejsce docelowe, a potem zostaw. Samo wyrównanie zapisuje folder na stos; fuzja zapisuje i pokazuje wyłącznie obraz wynikowy.

</td>
<td width="54%">

<img src="assets/readme/batch.png" alt="Okno przetwarzania wsadowego" width="100%" />

</td>
</tr>

<tr>
<td width="46%" valign="middle">

### Paski skali do publikacji

Kalibrowane z metadanych TIFF lub ręcznie, rysowane prawdziwą czcionką w rzeczywistej długości i wypalane w każdym eksporcie: obraz, stos, TIFF, GIF i wsad. Jeśli pasek nie mieści się czytelnie, okno to zgłasza zamiast rysować krótszy pasek z niepasującym opisem.

[Podręcznik →](./docs/USER_MANUAL_PL.md)

</td>
<td width="54%">

<img src="assets/readme/scalebar.png" alt="Okno paska skali z podglądem" width="100%" />

</td>
</tr>

<tr>
<td width="46%" valign="middle">

### Etykiety i podpisy

Podpisy to zarządzana lista: każdy ma własny zakres (źródło, wyrównany stos lub konkretny wynik), zakres klatek i styl. Podgląd i każdy eksport używają tego samego celu i numeru klatki, więc `{value}` czyta się tak samo w Zapisz, Zapisz jako, przeciąganiu i Eksportuj wszystko.

</td>
<td width="54%">

<img src="assets/readme/labels.png" alt="Etykiety i podpisy" width="100%" />

</td>
</tr>
</table>

**Dodatkowo w zestawie:**

- **Przepływ focus stackingu** — przeglądarka stosów, krzywa ostrości, szybki podgląd, render ROI i ustawienia zapamiętywane dla każdej metody.
- **Gotowe do mikroskopii** — 16 bitów na całej ścieżce, µm/px z TIFF, korekcja pola płaskiego, wielostronicowe stosy Z w TIFF.
- **Bezgłowy CLI** — ten sam potok jako `openfocus`, z trybem wsadowym, `--dry-run`, `--resume` i wyjściem JSON.
- **Projekty i odzyskiwanie** — sesje `.ofproj`, automatyczne odzyskiwanie po awarii, cofanie/ponawianie dla stosów i adnotacji.
- **Dwanaście języków interfejsu** — English, 简体中文, 日本語, Español, Deutsch, Français, Русский, 한국어, Português, Italiano, Nederlands, Polski; wykrywane z systemu, nowy język to jeden plik `locales_<code>.py`.
- **Oszczędny w pamięci** — wagi neuronowe ładują się tylko przy metodzie AI, więc sesja klasyczna zajmuje około 110 MB zamiast 260 MB.

## Szybki start

```bash
# Pełne CLI: wszystkie metody, model AI, HEIC i RAW z aparatów
pip install openfocus

# Fuzja folderu obrazów
openfocus -i ./stack_folder -o ./result/fused.png

# DTCWT z ECC i 8 wątkami
openfocus -i ./stack_folder -o ./result/fused.png -m dtcwt -a ecc -t 8
```

Uruchomienie GUI ze źródeł:

```bash
conda create -n openfocus python=3.10
conda activate openfocus
pip install -r requirements.txt
python main.py
```

## Dokumentacja

- [Podręcznik użytkownika (polski)](./docs/USER_MANUAL_PL.md)
- [Historia zmian](./CHANGELOG.md)
- [Wydajność i duże stosy](./docs/PERFORMANCE.md)
- [Polecenia budowania](./docs/BUILD_COMMANDS.md)
- [Zgodność algorytmów](./docs/ALGORITHM_PARITY.md)

Pełna dokumentacja angielska znajduje się w [README.md](./README.md).

## Cytowanie i licencja

Jeśli OpenFocus pomaga w Twojej pracy, zacytuj projekt zgodnie z [CITATION.cff](./CITATION.cff).

Projekt jest udostępniany na [licencji MIT](./LICENSE). Każdy algorytm fuzji/wyrównywania zachowuje również swoją oryginalną licencję.
