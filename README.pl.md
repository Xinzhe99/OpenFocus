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

- **Osiem metod fuzji** — Guided Filter, DCT, DTCWT, GFG-FGF, StackMFF-V3 (sieć neuronowa) oraz OpenFocus A/B/C.
- **Trzy tryby wyrównywania** — szybka transformacja podobieństwa, ECC i homografia; można też fuzować bez wyrównywania.
- **Przepływ focus stackingu** — przeglądarka stosu, porównanie A/B, mapy głębi, porównanie metod, przetwarzanie wsadowe i szybki podgląd.
- **Retusz ze źródłem** — piksele z klatek, które faktycznie utworzyły wynik; wspólny zoom/pozycja, suwak klatek źródłowych, miękki pędzel, pędzel przywracania, cofanie/ponawianie pociągnięć i ochrona nieeksportowanych wyników.
- **Gotowe do mikroskopii i publikacji** — 16 bitów od początku do końca, µm/px z TIFF, korekcja flat-field, paski skali, wielostronicowe stosy Z TIFF i eksport teksturowanego OBJ.
- **CLI bez interfejsu** — ten sam potok przez `openfocus`, z trybem wsadowym, `--dry-run`, `--resume` i wyjściem JSON.
- **Dwanaście języków interfejsu** — English, 简体中文, 日本語, Español, Deutsch, Français, Русский, 한국어, Português, Italiano, Nederlands, Polski.

<a id="command-line-usage"></a>
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
