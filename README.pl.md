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
  <a href="#install">Pobierz</a> ·
  <a href="./docs/USER_MANUAL_PL.md">Dokumentacja</a> ·
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
> **v1.69.0** — silnik fuzji planuje teraz pracę zamiast zgadywać: jeden plan wykonania na render decyduje o liczbie workerów, wątkach i budżecie pamięci, rdzenie fizyczne są odczytywane z platformy zamiast połowienia liczby logicznej, a każdy klasyczny backend strumieniuje bufory pośrednie zamiast trzymać tymczasowe dla całego stosu. Wynik pozostaje identyczny bajt w bajt, z zmierzonym przyspieszeniem 1,47× dla OpenFocus C i 1,33× dla metody neuronowej. Nowy **tryb zasobów** (Zrównoważony / Responsywny / Maksymalny) utrzymuje płynność pulpitu bez zmiany obrazu.

<a id="features"></a>
## Najważniejsze funkcje

<table>

<tr>
<td width="46%" valign="middle">

### Fuzja AI: StackMFF V3

Domyślna metoda klasyfikuje każdy piksel do najbardziej prawdopodobnej klatki źródłowej za pomocą sieci neuronowej. Na tym samym wyrównanym stosie krawędzie liter pozostają ostrzejsze, a powierzchnie czystsze niż w Guided Filter (ostrość laplasjanowa 27,1 vs 28,5); wobec syntetycznej prawdy prowadzi w PSNR i SSIM i zostawia mniej duchów niż jakikolwiek klasyczny algorytm.

[StackMFF V3 →](https://github.com/Xinzhe99/StackMFF-V3) · [Benchmark →](tools/synthetic_benchmark.py)

</td>
<td width="54%">

<a href="assets/readme/ai.mp4"><picture><source srcset="assets/readme/ai.gif" type="image/gif"><img src="assets/readme/ai_poster.jpg" alt="Fuzja AI: StackMFF V3" width="100%" /></picture></a>

</td>
</tr>

<tr>
<td width="46%" valign="middle">

### Porównanie A/B

Przeciągnij podziałkę po klatce źródłowej i wyniku, aby zobaczyć, co zmieniła fuzja: dryf wyrównania, obwódki na krawędziach i utracony detal, przy dowolnym powiększeniu.

[Podręcznik →](./docs/USER_MANUAL_PL.md)

</td>
<td width="54%">

<a href="assets/readme/wipe.mp4"><picture><source srcset="assets/readme/wipe.gif" type="image/gif"><img src="assets/readme/wipe_poster.jpg" alt="Porównanie A/B klatki i wyniku" width="100%" /></picture></a>

</td>
</tr>

<tr>
<td width="46%" valign="middle">

### Porównanie metod

Wyrenderuj wszystkie metody raz, porównaj je obok siebie na tym samym kadrze i przyjmij zwycięzcę jako nowy wynik lub skopiuj jego parametry do okna głównego.

</td>
<td width="54%">

<a href="assets/readme/method-comparison.mp4"><picture><source srcset="assets/readme/method-comparison.gif" type="image/gif"><img src="assets/readme/method-comparison_poster.jpg" alt="Osiem metod fuzji obok siebie" width="100%" /></picture></a>

</td>
</tr>

<tr>
<td width="46%" valign="middle">

### Mapa głębi i 3D

Pozycja ostrości na piksel z pięciu miar, naukowe palety, fizyczna kalibracja Z z metadanych TIFF i eksport teksturowanego OBJ powierzchni.

</td>
<td width="54%">

<a href="assets/readme/depth.mp4"><picture><source srcset="assets/readme/depth.gif" type="image/gif"><img src="assets/readme/depth_poster.jpg" alt="Mapa głębi i powierzchnia 3D tego samego stosu" width="100%" /></picture></a>

</td>
</tr>

<tr>
<td width="46%" valign="middle">

### Retusz powiązany ze źródłem

Maluj z klatek, które naprawdę utworzyły wynik: wspólny zoom i przesuwanie, suwak klatki źródłowej, miękki pędzel, pędzel przywracania, cofanie na pociągnięcie i ochrona niewyeksportowanych pikseli.

</td>
<td width="54%">

<a href="assets/readme/retouch.mp4"><picture><source srcset="assets/readme/retouch.gif" type="image/gif"><img src="assets/readme/retouch_poster.jpg" alt="Obszar retuszu z klatką i wynikiem" width="100%" /></picture></a>

</td>
</tr>

<tr>
<td width="46%" valign="middle">

### Przetwarzanie wsadowe

Wskaż folder ze stosami, wybierz metodę i miejsce docelowe, a potem zostaw. Samo wyrównanie zapisuje folder na stos; fuzja zapisuje i pokazuje wyłącznie obraz wynikowy.

</td>
<td width="54%">

<a href="assets/readme/batch.mp4"><picture><source srcset="assets/readme/batch.gif" type="image/gif"><img src="assets/readme/batch_poster.jpg" alt="Okno przetwarzania wsadowego" width="100%" /></picture></a>

</td>
</tr>

<tr>
<td width="46%" valign="middle">

### Paski skali do publikacji

Kalibrowane z metadanych TIFF lub ręcznie, rysowane prawdziwą czcionką w rzeczywistej długości i wypalane w każdym eksporcie: obraz, stos, TIFF, GIF i wsad. Jeśli pasek nie mieści się czytelnie, okno to zgłasza zamiast rysować krótszy pasek z niepasującym opisem.

[Podręcznik →](./docs/USER_MANUAL_PL.md)

</td>
<td width="54%">

<a href="assets/readme/scalebar.mp4"><picture><source srcset="assets/readme/scalebar.gif" type="image/gif"><img src="assets/readme/scalebar_poster.jpg" alt="Okno paska skali z podglądem" width="100%" /></picture></a>

</td>
</tr>

<tr>
<td width="46%" valign="middle">

### Etykiety i podpisy

Podpisy to zarządzana lista: każdy ma własny zakres (źródło, wyrównany stos lub konkretny wynik), zakres klatek i styl. Podgląd i każdy eksport używają tego samego celu i numeru klatki, więc `{value}` czyta się tak samo w Zapisz, Zapisz jako, przeciąganiu i Eksportuj wszystko.

</td>
<td width="54%">

<a href="assets/readme/labels.mp4"><picture><source srcset="assets/readme/labels.gif" type="image/gif"><img src="assets/readme/labels_poster.jpg" alt="Etykiety i podpisy" width="100%" /></picture></a>

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

<a id="install"></a>
## ⬇️ Instalacja

Bez konfiguracji — pobierz kompilację ze strony [Releases](https://github.com/Xinzhe99/OpenFocus/releases):

| Platforma | Plik | Typ |
|-----------|------|-----|
| Windows 10/11 (64-bit) | `OpenFocus-*-setup.exe` | **Instalator** (zalecany) |
| Windows 10/11 (64-bit) | `OpenFocus-*-windows-x64.zip` | Przenośna (rozpakuj i uruchom) |
| macOS Apple Silicon (M1–M4) | `OpenFocus-*-macos.dmg` | Obraz dysku (przeciągnij do Programów) |
| macOS Apple Silicon (M1–M4) | `OpenFocus-*-macos-arm64.zip` | Przenośna (rozpakuj) |

Aplikacja nie jest podpisana, więc przy pierwszym uruchomieniu może pojawić się ostrzeżenie: **Windows** SmartScreen → „Więcej informacji” → „Uruchom mimo to” (pierwsze uruchomienie może potrwać 30–60 s); **macOS** kliknij prawym → Otwórz lub `xattr -cr /Applications/OpenFocus.app` w Terminalu.

### 💻 Wiersz poleceń

Możesz też zainstalować CLI na Windows, macOS i Linux — ten sam potok bez stosu graficznego:

```bash
pip install openfocus
openfocus -i ./stack_folder -o ./result/fused.png
```

Model AI, wagi, dekodowanie HEIC i RAW są w tej jednej instalacji. Na Pythonie 3.13+ `dtcwt` jest pomijany, a pozycja DTCWT pozostaje wyłączona z wyjaśnieniem.

<a id="community"></a>
## Społeczność i wsparcie

- **Dokumentacja** — [Podręcznik](./docs/USER_MANUAL_PL.md) (dwanaście języków) · [Zmiany](./CHANGELOG.md) · [Wydajność](./docs/PERFORMANCE.md) · [Zgodność algorytmów](./docs/ALGORITHM_PARITY.md)
- **Cytowanie** — [CITATION.cff](./CITATION.cff); każda wersja jest archiwizowana w Zenodo ([DOI](https://doi.org/10.5281/zenodo.23050823))
- **Opinie** — otwórz [issue](https://github.com/Xinzhe99/OpenFocus/issues) dla błędów, pomysłów i prac nad algorytmami
- **Podziękowania** — każdy algorytm fuzji i rejestracji zachowuje swoją pierwotną licencję badawczą; StackMFF V3 jest na [Xinzhe99/StackMFF-V3](https://github.com/Xinzhe99/StackMFF-V3)

<a id="developing"></a>
## Rozwój

Uruchamianie z repozytorium:

```bash
conda create -n openfocus python=3.10
conda activate openfocus
pip install -r requirements.txt
python main.py
```

Pakowanie przez PyInstaller: [docs/BUILD_COMMANDS.md](./docs/BUILD_COMMANDS.md).

<a id="star-history"></a>
## ⭐ Historia gwiazdek

[![Star History Chart](https://api.star-history.com/svg?repos=Xinzhe99/OpenFocus&type=Date)](https://star-history.com/#Xinzhe99/OpenFocus&Date)
