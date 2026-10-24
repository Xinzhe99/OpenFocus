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
  <a href="#download">Скачать</a> ·
  <a href="./docs/USER_MANUAL_RU.md">Документация</a> ·
  <a href="#command-line-usage">CLI</a> ·
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
> **v1.68.1** — пакетный экспорт следует выбранному режиму: только выравнивание создаёт папку `<стек>_aligned` и показывает её в списке, фузия показывает только результат, а без сохранения выравнивания пишется лишь fused-изображение. Оставшиеся в исходной папке копии `*_registered*` больше не загружаются как кадры. Также переработаны метки, масштабные линейки, сравнение методов, экспорт глубины/модели, обновления и ретушь по аудиту `docs/ux-specs`.

<a id="download"></a>
## Скачивание и установка

Готовые сборки — на странице [Releases](https://github.com/Xinzhe99/OpenFocus/releases):

| Платформа | Файл | Тип |
|-----------|------|-----|
| Windows 10/11 (64 бита) | `OpenFocus-*-setup.exe` | **Установщик** (рекомендуется) |
| Windows 10/11 (64 бита) | `OpenFocus-*-windows-x64.zip` | Портативная версия (распаковать) |
| macOS Apple Silicon (M1–M4) | `OpenFocus-*-macos.dmg` | Образ диска (перетащить в «Программы») |
| macOS Apple Silicon (M1–M4) | `OpenFocus-*-macos-arm64.zip` | Портативная версия (распаковать) |

Приложение не подписано. При первом запуске Windows может предложить «Подробнее → Выполнить всё равно»; в macOS — правый клик → «Открыть» или `xattr -cr /Applications/OpenFocus.app`.

## Возможности

- **Восемь методов объединения** — Guided Filter, DCT, DTCWT, GFG-FGF, StackMFF-V3 (нейросеть) и OpenFocus A/B/C.
- **Три режима совмещения** — быстрая трансформация подобия, ECC и гомография; можно объединять и без совмещения.
- **Рабочий процесс фокус-стекинга** — просмотр стека, A/B-сравнение, карты глубины, сравнение методов, пакетная обработка и быстрый предпросмотр.
- **Ретушь с привязкой к источнику** — пиксели берутся из кадров, реально участвовавших в рендеринге; общий масштаб/позиция, ползунок кадров, мягкая кисть, кисть восстановления, отмена/повтор по штрихам и защита неэкспортированных результатов.
- **Для микроскопии и публикаций** — 16 бит сквозным образом, µm/px из TIFF, коррекция плоского поля, масштабные линейки, многостраничные TIFF Z-стеки и текстурированный экспорт OBJ.
- **CLI без интерфейса** — тот же конвейер через `openfocus`: пакетный режим, `--dry-run`, `--resume` и JSON-вывод.
- **Двенадцать языков интерфейса** — English, 简体中文, 日本語, Español, Deutsch, Français, Русский, 한국어, Português, Italiano, Nederlands, Polski.

<a id="command-line-usage"></a>
## Быстрый старт

```bash
# Полная CLI: все методы, ИИ-модель, HEIC и RAW камер
pip install openfocus

# Объединить папку изображений
openfocus -i ./stack_folder -o ./result/fused.png

# DTCWT с ECC и 8 потоками
openfocus -i ./stack_folder -o ./result/fused.png -m dtcwt -a ecc -t 8
```

Запуск GUI из исходников:

```bash
conda create -n openfocus python=3.10
conda activate openfocus
pip install -r requirements.txt
python main.py
```

## Документация

- [Руководство пользователя (русский)](./docs/USER_MANUAL_RU.md)
- [История изменений](./CHANGELOG.md)
- [Производительность и большие стеки](./docs/PERFORMANCE.md)
- [Команды сборки](./docs/BUILD_COMMANDS.md)
- [Соответствие алгоритмам](./docs/ALGORITHM_PARITY.md)

Полная английская документация — в [README.md](./README.md).

## Цитирование и лицензия

Если OpenFocus помогает вашей работе, пожалуйста, цитируйте проект по [CITATION.cff](./CITATION.cff).

Проект распространяется по [лицензии MIT](./LICENSE). Для каждого алгоритма объединения/совмещения также действуют условия его исходной лицензии.
