<p align="center">
  <a href="https://github.com/Xinzhe99/OpenFocus">
    <img src="../../assets/OpenFocus.png" alt="OpenFocus" width="128">
  </a>
</p>

<h1 align="center">OpenFocus</h1>

<p align="center"><strong>Każdy piksel z najostrzejszej klatki.</strong></p>

<p align="center">
  Dziesięć algorytmów fuzji, pięć opcji wyrównania (w tym brak wyrównania) i przestrzeń retuszu powiązana ze źródłami dla makrofotografii, mikroskopii i stosów przemysłowych. 16 bitów na całej ścieżce, headless CLI i interfejs w dwunastu językach.
</p>

<p align="center">
  <a href="https://github.com/Xinzhe99/OpenFocus/releases"><img src="https://img.shields.io/github/v/release/Xinzhe99/OpenFocus?label=release&color=2f81f7" alt="Release"></a>
  <a href="https://github.com/Xinzhe99/OpenFocus/stargazers"><img src="https://img.shields.io/github/stars/Xinzhe99/OpenFocus?color=e3b341" alt="Stars"></a>
  <a href="../../docs/THIRD_PARTY_NOTICES.md"><img src="https://img.shields.io/badge/license-MIT%20%2B%20model%20terms-3fb950" alt="License: MIT + model terms"></a>
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
> **[v1.77.0-beta.7 — wydanie wstępne](https://github.com/Xinzhe99/OpenFocus/releases/tag/v1.77.0-beta.7)** — Przyspieszenie GPU dla zgodnych urządzeń OpenCL w OpenFocus B z przełączeniem na CPU, poprawione porównanie suwakiem oraz automatyczne tło i retusz. Szczegóły w [historii zmian](../../CHANGELOG.md).

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

StackMFF V3 wykorzystuje AI do łączenia ostrych fragmentów klatek źródłowych.

[StackMFF V3 →](https://github.com/Xinzhe99/StackMFF-V3) · [Benchmark →](../../tools/synthetic_benchmark.py)

</td>
<td width="54%">

<a href="../../assets/readme/ai.mp4"><picture><source srcset="../../assets/readme/ai.gif" type="image/gif"><img src="../../assets/readme/ai_poster.jpg" alt="StackMFF V3 i Guided Filter na tym samym stosie" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Porównanie suwakiem

Porównuj klatkę źródłową z wynikiem fuzji za pomocą przesuwanego podziału.

[Podręcznik →](../USER_MANUAL_PL.md)

</td>
<td width="54%">

<a href="../../assets/readme/wipe.mp4"><picture><source srcset="../../assets/readme/wipe.gif" type="image/gif"><img src="../../assets/readme/wipe_poster.jpg" alt="Porównanie suwakiem klatki źródłowej i wyniku" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Porównanie metod

Porównuj metody obok siebie i wykorzystaj wybrany wynik lub jego ustawienia.

</td>
<td width="54%">

<a href="../../assets/readme/method-comparison.mp4"><picture><source srcset="../../assets/readme/method-comparison.gif" type="image/gif"><img src="../../assets/readme/method-comparison_poster.jpg" alt="Osiem metod fuzji obok siebie" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Mapa głębi i 3D

Twórz mapy głębi i eksportuj teksturowane powierzchnie 3D.

</td>
<td width="54%">

<a href="../../assets/readme/depth.mp4"><picture><source srcset="../../assets/readme/depth.gif" type="image/gif"><img src="../../assets/readme/depth_poster.jpg" alt="Mapa głębi i powierzchnia 3D tego samego stosu" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Retusz powiązany ze źródłami

Retuszuj wynik pikselami z klatek źródłowych. Automatyczne tło zastępuje niespokojne tło ze stacku jedną prawdziwą klatką.

</td>
<td width="54%">

<a href="../../assets/readme/retouch.mp4"><picture><source srcset="../../assets/readme/retouch.gif" type="image/gif"><img src="../../assets/readme/retouch_poster.jpg" alt="Przestrzeń retuszu z klatką źródłową i wynikiem" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Przetwarzanie wsadowe

Przetwarzaj wiele stosów z opcjonalną fuzją grupową ograniczającą zużycie pamięci.

</td>
<td width="54%">

<a href="../../assets/readme/batch.mp4"><picture><source srcset="../../assets/readme/batch.gif" type="image/gif"><img src="../../assets/readme/batch_poster.jpg" alt="Okno przetwarzania wsadowego" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Paski skali publikacyjnej

Dodawaj do eksportów paski skali kalibrowane ręcznie lub na podstawie metadanych TIFF.

[Podręcznik →](../USER_MANUAL_PL.md)

</td>
<td width="54%">

<a href="../../assets/readme/scalebar.mp4"><picture><source srcset="../../assets/readme/scalebar.gif" type="image/gif"><img src="../../assets/readme/scalebar_poster.jpg" alt="Okno paska skali z podglądem na żywo" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Etykiety i adnotacje

Zarządzaj etykietami i adnotacjami o własnym zakresie i stylu.

</td>
<td width="54%">

<a href="../../assets/readme/labels.mp4"><picture><source srcset="../../assets/readme/labels.gif" type="image/gif"><img src="../../assets/readme/labels_poster.jpg" alt="Menedżer etykiet z podglądem na żywo" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Sterowanie przez agenta

Steruj aplikacją przez lokalny kanał agenta chroniony tokenem.

[Protokół →](../AGENT_CONTROL_SPEC.md) · [Umiejętność →](../../skills/openfocus-control/SKILL.md)

</td>
<td width="54%">

<a href="../../assets/readme/agent.mp4"><picture><source srcset="../../assets/readme/agent.gif" type="image/gif"><img src="../../assets/readme/agent_poster.jpg" alt="Agent steruje działającym interfejsem: ładowanie, render AI, porównanie, retusz, suwak, głębia, etykiety" width="100%" /></picture></a>

</td>
</tr>
</table>

**W zestawie także:**

- **Przepływ pracy ze stosem ostrości** — przeglądarka stosu, krzywa ostrości, szybki podgląd, render ROI i zapamiętane ustawienia.
- **Gotowość do mikroskopii** — 16 bitów, kalibracja TIFF, korekcja pola płaskiego i wielostronicowe stosy Z.
- **Profile kolorów** — eksport pojedynczych obrazów zachowuje wspólny profil RGB ICC bez konwersji kolorów.
- **Przezroczysty PNG** — łączenie i eksport zgodnych stosów z prostokątnymi maskami w pełnym rozmiarze.
- **Headless CLI** — `openfocus` z trybem wsadowym, `--dry-run`, `--resume` i wyjściem JSON.
- **Projekty i odzyskiwanie** — sesje `.ofproj`, odzyskiwanie po awarii oraz cofanie i ponawianie zmian.
- **Dwanaście języków interfejsu** — automatyczne wykrywanie; nowe języki można dodać w `locales/<code>.py`.
- **Oszczędność pamięci** — sieć neuronowa jest ładowana tylko w razie potrzeby.

<a id="install"></a>
## ⬇️ Instalacja

[stronie OpenFocus](https://openfocus.top/#download) i skrypty instalacyjne oferują dla Windows i macOS najnowszą wersję, także nowszą wersję beta (obecnie v1.77.0-beta.7). Wydania są również dostępne na [GitHub Releases](https://github.com/Xinzhe99/OpenFocus/releases). Automatyczne aktualizacje aplikacji oferują także nowsze wersje beta; linuksowe CLI z PyPI pozostaje stabilne.

Informacje o instalacji i rozwiązywaniu problemów zawiera [podręcznik](../USER_MANUAL_PL.md).

W systemie Linux polecenie terminala instaluje headless CLI; macOS wymaga Apple Silicon. Aplikacja z DMG trafia do `/Applications`, a instalacja z terminala do `~/Applications`.

```bash
curl -fsSL https://openfocus.top/install.sh | sh -s -- --dry-run
curl -fsSL https://openfocus.top/install.sh | sh
```

W systemie Windows uruchom instalator z PowerShell:

```powershell
irm https://openfocus.top/install.ps1 | iex
```

Bezpośrednie pobranie instalatora Windows:

```powershell
curl.exe -fL https://openfocus.top/download/latest/windows-setup -o OpenFocus-setup.exe
```

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

Masz pytania, pomysły lub uwagi? Dołącz do naszej [społeczności Discord](https://discord.gg/afzmwgPt). Odpowiadamy tak szybko, jak to możliwe; pracujemy według czasu pekińskiego (UTC+8), więc różnica czasu może opóźnić odpowiedź.

- **Dokumentacja** — [podręcznik użytkownika](../USER_MANUAL_PL.md) (dwanaście języków) · [historia zmian](../../CHANGELOG.md) · [wydajność](../PERFORMANCE.md) · [metody fuzji](../FUSION_METHODS.md)
- **Cytowanie** — [CITATION.cff](../../CITATION.cff); każde wydanie jest archiwizowane na Zenodo ([DOI](https://doi.org/10.5281/zenodo.23050823))
- **Opinie** — [issue](https://github.com/Xinzhe99/OpenFocus/issues) dla błędów, pomysłów i prac nad algorytmami
- **Podziękowania** — każdy algorytm fuzji i rejestracji zachowuje swoją oryginalną licencję badawczą; StackMFF V3 mieszka w [Xinzhe99/StackMFF-V3](https://github.com/Xinzhe99/StackMFF-V3)

<a id="license"></a>
## Licencja

Aplikacja OpenFocus jest udostępniana na [licencji MIT](../../LICENSE).

**Licencje StackMFF.** Nowe wagi i wrappery V1/V2 podlegają StackMFF V3 License 1.0. Kopiowanie, rozpowszechnianie, włączanie do innych projektów i użycie komercyjne wymagają uprzedniej pisemnej zgody posiadacza praw. Oryginalna dołączona wersja migawkowa StackMFF V3 v2.0 pozostaje na licencji MIT. Zobacz [informacje o oprogramowaniu stron trzecich](../../docs/THIRD_PARTY_NOTICES.md).

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
