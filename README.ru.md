<p align="center">
  <a href="https://github.com/Xinzhe99/OpenFocus">
    <img src="assets/OpenFocus.png" alt="OpenFocus" width="128">
  </a>
</p>

<h1 align="center">OpenFocus</h1>

<p align="center"><strong>Все плоскости фокуса — в одном полностью резком изображении.</strong></p>

<p align="center">
  Восемь алгоритмов фузии, три режима совмещения и рабочее пространство ретуши с привязкой к источникам для макро, микроскопии и промышленных стеков. 16 бит по всей цепочке, headless-CLI и интерфейс на двенадцати языках.
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
  <a href="#install">Скачать</a> ·
  <a href="./docs/USER_MANUAL_RU.md">Документация</a> ·
  <a href="./CHANGELOG.md">История</a> ·
  <a href="https://doi.org/10.5281/zenodo.23050823">DOI</a>
</p>

<p align="center">
  <a href="README.md">English</a> ·
  <a href="README.zh-CN.md">简体中文</a> ·
  <a href="README.ja.md">日本語</a> ·
  <a href="README.es.md">Español</a> ·
  <a href="README.de.md">Deutsch</a> ·
  <a href="README.fr.md">Français</a> ·
  <strong>Русский</strong> ·
  <a href="README.ko.md">한국어</a> ·
  <a href="README.pt.md">Português</a> ·
  <a href="README.it.md">Italiano</a> ·
  <a href="README.nl.md">Nederlands</a> ·
  <a href="README.pl.md">Polski</a>
</p>

<p align="center">
  <img src="assets/ui.png" alt="Интерфейс OpenFocus" width="880">
</p>

## 📢 News

> [!NOTE]
> **v1.70.0** — каждая тяжёлая стадия теперь использует одно разрешение на выполнение от подготовки до публикации: отмена ждёт фактического завершения текущего нативного вызова, а не запускает поверх него второй рендер. Память проверяется до выделения (нечитаемая или недостаточная — явная ошибка, а не тихо уменьшенные тайлы), кэш выравнивания фиксируется атомарно и не перекодируется при попадании, 16-битный нейросетевой вывод сохраняет младшие биты, а новый **режим ресурсов** с автоматическими потоками (`0`) оставляет запас рабочему столу. Восемь алгоритмов остаются побайтово идентичными v1.69.0.

<a id="features"></a>
## Возможности

<table>

<tr>
<td width="46%" valign="middle">

### ИИ-фузия: StackMFF V3

Метод по умолчанию относит каждый пиксель к наиболее вероятному исходному кадру с помощью нейросети. На том же выровненном стеке края букв остаются резче, а поверхности чище, чем у Guided Filter (резкость по Лапласу 27,1 против 28,5); на синтетических данных с известной истиной он лидирует по PSNR и SSIM и оставляет меньше артефактов-призраков, чем любой классический алгоритм.

[StackMFF V3 →](https://github.com/Xinzhe99/StackMFF-V3) · [Бенчмарк →](tools/synthetic_benchmark.py)

</td>
<td width="54%">

<a href="assets/readme/ai.mp4"><picture><source srcset="assets/readme/ai.gif" type="image/gif"><img src="assets/readme/ai_poster.jpg" alt="ИИ-фузия: StackMFF V3" width="100%" /></picture></a>

</td>
</tr>

<tr>
<td width="46%" valign="middle">

### A/B-сравнение

Проведите разделитель по исходному кадру и результату: сразу видно, что изменила фузия — сдвиг совмещения, ореолы по краям и потерянные детали при любом масштабе.

[Руководство →](./docs/USER_MANUAL_RU.md)

</td>
<td width="54%">

<a href="assets/readme/wipe.mp4"><picture><source srcset="assets/readme/wipe.gif" type="image/gif"><img src="assets/readme/wipe_poster.jpg" alt="A/B-сравнение кадра и результата" width="100%" /></picture></a>

</td>
</tr>

<tr>
<td width="46%" valign="middle">

### Сравнение методов

Отрисуйте все методы за один проход, сравните их рядом на одном фрагменте и примите победителя как новый результат или перенесите его параметры в главное окно.

</td>
<td width="54%">

<a href="assets/readme/method-comparison.mp4"><picture><source srcset="assets/readme/method-comparison.gif" type="image/gif"><img src="assets/readme/method-comparison_poster.jpg" alt="Восемь методов фузии рядом" width="100%" /></picture></a>

</td>
</tr>

<tr>
<td width="46%" valign="middle">

### Карта глубины и 3D

Позиция фокуса по пикселям по пяти мерам, научные палитры, физическая калибровка Z из метаданных TIFF и экспорт текстурированного OBJ поверхности.

</td>
<td width="54%">

<a href="assets/readme/depth.mp4"><picture><source srcset="assets/readme/depth.gif" type="image/gif"><img src="assets/readme/depth_poster.jpg" alt="Карта глубины и 3D-поверхность одного стека" width="100%" /></picture></a>

</td>
</tr>

<tr>
<td width="46%" valign="middle">

### Ретушь с привязкой к источникам

Рисуйте из кадров, которые действительно создали результат: общий зум и панорамирование, ползунок исходного кадра, мягкая кисть, кисть восстановления, отмена по штриху и защита неэкспортированных пикселей.

</td>
<td width="54%">

<a href="assets/readme/retouch.mp4"><picture><source srcset="assets/readme/retouch.gif" type="image/gif"><img src="assets/readme/retouch_poster.jpg" alt="Рабочее пространство ретуши с кадром и результатом" width="100%" /></picture></a>

</td>
</tr>

<tr>
<td width="46%" valign="middle">

### Пакетная обработка

Укажите папку со стеками, выберите метод и назначение и оставьте работать. Только выравнивание пишет одну папку на стек; фузия пишет и показывает только fused-изображение.

</td>
<td width="54%">

<a href="assets/readme/batch.mp4"><picture><source srcset="assets/readme/batch.gif" type="image/gif"><img src="assets/readme/batch_poster.jpg" alt="Диалог пакетной обработки" width="100%" /></picture></a>

</td>
</tr>

<tr>
<td width="46%" valign="middle">

### Масштабные линейки для публикаций

Калибруются из метаданных TIFF или вручную, рисуются реальным шрифтом в реальной длине и вписываются в каждый экспорт: изображение, стек, TIFF, GIF и пакет. Если линейка не помещается разборчиво, диалог сообщает об этом, а не рисует более короткую полосу с неверной подписью.

[Руководство →](./docs/USER_MANUAL_RU.md)

</td>
<td width="54%">

<a href="assets/readme/scalebar.mp4"><picture><source srcset="assets/readme/scalebar.gif" type="image/gif"><img src="assets/readme/scalebar_poster.jpg" alt="Диалог масштабной линейки с предпросмотром" width="100%" /></picture></a>

</td>
</tr>

<tr>
<td width="46%" valign="middle">

### Подписи и легенды

Подписи хранятся управляемым списком: у каждой своя область (источник, выровненный стек или конкретный результат), диапазон кадров и стиль. Предпросмотр и любой экспорт используют одну и ту же цель и номер кадра, поэтому `{value}` читается одинаково в «Сохранить», «Сохранить как», перетаскивании и «Экспортировать всё».

</td>
<td width="54%">

<a href="assets/readme/labels.mp4"><picture><source srcset="assets/readme/labels.gif" type="image/gif"><img src="assets/readme/labels_poster.jpg" alt="Подписи и легенды" width="100%" /></picture></a>

</td>
</tr>
</table>

**Также в комплекте:**

- **Рабочий процесс фокус-стекинга** — просмотр стека, кривая резкости, быстрый предпросмотр, ROI-рендер и запоминание настроек для каждого метода между запусками.
- **Готово к микроскопии** — 16 бит по всей цепочке, µm/px из TIFF, коррекция плоского поля, многостраничные TIFF Z-стеки.
- **Headless-CLI** — тот же конвейер как `openfocus`, с пакетным режимом, `--dry-run`, `--resume` и JSON-выводом.
- **Проекты и восстановление** — сессии `.ofproj`, автосохранение при сбое, отмена/повтор для стека и аннотаций.
- **Двенадцать языков интерфейса** — English, 简体中文, 日本語, Español, Deutsch, Français, Русский, 한국어, Português, Italiano, Nederlands, Polski; определяются по системе, новый язык — один файл `locales_<code>.py`.
- **Экономия памяти** — нейросетевые веса загружаются только при использовании AI-метода, классическая сессия занимает около 110 МБ вместо 260 МБ.

<a id="install"></a>
## ⬇️ Установка

Ничего настраивать не нужно — сборка на странице [Releases](https://github.com/Xinzhe99/OpenFocus/releases):

| Платформа | Файл | Тип |
|-----------|------|-----|
| Windows 10/11 (64 бита) | `OpenFocus-*-setup.exe` | **Установщик** (рекомендуется) |
| Windows 10/11 (64 бита) | `OpenFocus-*-windows-x64.zip` | Портативная версия (распаковать) |
| macOS Apple Silicon (M1–M4) | `OpenFocus-*-macos.dmg` | Образ диска (перетащить в «Программы») |
| macOS Apple Silicon (M1–M4) | `OpenFocus-*-macos-arm64.zip` | Портативная версия (распаковать) |

Приложение не подписано, поэтому при первом запуске возможно предупреждение: **Windows** SmartScreen → «Подробнее» → «Выполнить всё равно» (первый запуск может занять 30–60 с); **macOS** правый клик → «Открыть» или `xattr -cr /Applications/OpenFocus.app` в терминале.

### 💻 Командная строка

Либо установите CLI на Windows, macOS и Linux — тот же конвейер без графического стека:

```bash
pip install openfocus
openfocus -i ./stack_folder -o ./result/fused.png
```

Модель ИИ, веса, декодирование HEIC и RAW входят в эту установку. На Python 3.13+ `dtcwt` пропускается, и пункт DTCWT остаётся отключённым с пояснением.

<a id="community"></a>
## Сообщество и поддержка

- **Документация** — [Руководство](./docs/USER_MANUAL_RU.md) (двенадцать языков) · [Изменения](./CHANGELOG.md) · [Производительность](./docs/PERFORMANCE.md) · [Соответствие алгоритмов](./docs/ALGORITHM_PARITY.md)
- **Цитирование** — [CITATION.cff](./CITATION.cff); каждый релиз архивируется на Zenodo ([DOI](https://doi.org/10.5281/zenodo.23050823))
- **Обратная связь** — [issue](https://github.com/Xinzhe99/OpenFocus/issues) для ошибок, идей и работы над алгоритмами
- **Благодарности** — каждый алгоритм фузии и совмещения сохраняет свою исходную исследовательскую лицензию; StackMFF V3 — [Xinzhe99/StackMFF-V3](https://github.com/Xinzhe99/StackMFF-V3)

<a id="developing"></a>
## Разработка

Запуск из репозитория:

```bash
conda create -n openfocus python=3.10
conda activate openfocus
pip install -r requirements.txt
python main.py
```

Упаковка через PyInstaller: [docs/BUILD_COMMANDS.md](./docs/BUILD_COMMANDS.md).

<a id="star-history"></a>
## ⭐ История звёзд

[![Star History Chart](https://api.star-history.com/svg?repos=Xinzhe99/OpenFocus&type=Date)](https://star-history.com/#Xinzhe99/OpenFocus&Date)
