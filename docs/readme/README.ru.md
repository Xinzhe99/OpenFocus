<p align="center">
  <a href="https://github.com/Xinzhe99/OpenFocus">
    <img src="../../assets/OpenFocus.png" alt="OpenFocus" width="128">
  </a>
</p>

<h1 align="center">OpenFocus</h1>

<p align="center"><strong>Каждый пиксель — из самого резкого кадра.</strong></p>

<p align="center">
  Десять алгоритмов объединения, пять вариантов совмещения (включая отсутствие совмещения) и рабочее пространство ретуши с привязкой к исходным кадрам для макро-, микроскопической и промышленной съёмки. 16 бит по всей цепочке, headless-CLI и интерфейс на двенадцати языках.
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
  <a href="#install">Скачать</a> ·
  <a href="../USER_MANUAL_RU.md">Документация</a> ·
  <a href="../../CHANGELOG.md">История изменений</a> ·
  <a href="https://doi.org/10.5281/zenodo.23050823">DOI</a>
</p>

<p align="center">
  <a href="../../README.md">English</a> ·
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
  <img src="../../assets/ui.png" alt="Интерфейс OpenFocus" width="880">
</p>

## 📢 Новости

> [!NOTE]
> **v1.75.0** — Групповое объединение, переиспользуемые рабочие процессы обработки,
> прозрачные PNG-стеки, диагностика совмещения, расширенный экспорт глубины
> и обновлённый рабочий процесс настольного приложения. Поддерживаемые входные
> данные и ограничения см. в истории изменений.

Предыдущие выпуски — в [истории изменений](../../CHANGELOG.md).

📸 Если вы публикуете изображения, созданные в OpenFocus, по возможности добавьте примечание:

> Создано в OpenFocus — https://openfocus.top

Это не обязательно, но мы будем очень признательны.

<a id="features"></a>
## Возможности

<table>
<tr>
<td width="46%" valign="middle">

### ИИ-объединение: StackMFF V3

Метод по умолчанию использует нейросеть для оценки наиболее вероятного исходного
кадра каждого пикселя, а затем смешивает вклады кадров или выбирает наиболее
вероятный кадр. Синтетический бенчмарк сравнивает методы с известным эталоном;
результат зависит от стека и настроек.

[StackMFF V3 →](https://github.com/Xinzhe99/StackMFF-V3) · [Бенчмарк →](../../tools/synthetic_benchmark.py)

</td>
<td width="54%">

<a href="../../assets/readme/ai.mp4"><picture><source srcset="../../assets/readme/ai.gif" type="image/gif"><img src="../../assets/readme/ai_poster.jpg" alt="StackMFF V3 и Guided Filter на одном стеке" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Сравнение «Шторка»

Проведите разделитель по исходному кадру и результату объединения, чтобы точно
увидеть, что изменило смешивание: сдвиг совмещения, ореолы по краям и потерянные
детали заметны с первого взгляда при любом масштабе.

[Руководство →](../USER_MANUAL_RU.md)

</td>
<td width="54%">

<a href="../../assets/readme/wipe.mp4"><picture><source srcset="../../assets/readme/wipe.gif" type="image/gif"><img src="../../assets/readme/wipe_poster.jpg" alt="Сравнение «Шторка» между исходным кадром и результатом" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Сравнение методов

По умолчанию каждый метод объединяет изображение целиком; квадрат для проверки
источника остаётся интерактивным без повторных вычислений полного изображения.
Затем сравните результаты рядом на общем фрагменте и примите победителя как новый
результат или перенесите его параметры обратно в главное окно.

</td>
<td width="54%">

<a href="../../assets/readme/method-comparison.mp4"><picture><source srcset="../../assets/readme/method-comparison.gif" type="image/gif"><img src="../../assets/readme/method-comparison_poster.jpg" alt="Восемь методов объединения рядом" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Карта глубины и 3D

Положение фокуса для каждого пикселя по семи методам, поддерживающим карту
глубины, научные цветовые палитры, ручная калибровка физической оси Z (начало и
шаг; метаданные TIFF дают только калибровку XY) и экспорт текстурированного OBJ
поверхности.

</td>
<td width="54%">

<a href="../../assets/readme/depth.mp4"><picture><source srcset="../../assets/readme/depth.gif" type="image/gif"><img src="../../assets/readme/depth_poster.jpg" alt="Карта глубины и 3D-поверхность одного стека" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Ретушь с привязкой к источникам

Рисуйте пикселями из кадров, которые действительно создали результат: общий
масштаб и панорамирование, ползунок исходного кадра, мягкая кисть, кисть
восстановления, отмена целыми штрихами и защита неэкспортированных пикселей.
Примените ретушь как новый результат, затем щёлкните его правой кнопкой и
выберите **Экспортировать TIFF со слоями…**, чтобы сохранить его вместе с исходными
кадрами. Живые подписи и масштабные линейки в файл не входят. GIMP 3.2.6 в Windows
открывает результат и слои источников; другие редакторы не проверялись.

</td>
<td width="54%">

<a href="../../assets/readme/retouch.mp4"><picture><source srcset="../../assets/readme/retouch.gif" type="image/gif"><img src="../../assets/readme/retouch_poster.jpg" alt="Рабочее пространство ретуши с исходным кадром и результатом" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Пакетная обработка

Укажите папку со стеками, выберите метод и назначение — и оставьте работать.
Режим «только выравнивание» записывает по одной папке на стек; объединение
записывает объединённое изображение и показывает в списке именно его. Необязательное
групповое объединение обменивает точную эквивалентность обработке всего стека
на ограниченный расход памяти и временное место на диске.

</td>
<td width="54%">

<a href="../../assets/readme/batch.mp4"><picture><source srcset="../../assets/readme/batch.gif" type="image/gif"><img src="../../assets/readme/batch_poster.jpg" alt="Диалог пакетной обработки" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Масштабные линейки для публикаций

Калибруются по метаданным TIFF или вводятся вручную, рисуются настоящим шрифтом
в реальной длине и впечатываются в каждый экспорт — отдельные изображения, стеки,
многостраничный TIFF, GIF и результаты пакетной обработки. Если линейка не
помещается разборчиво, приложение сообщает об этом, а не печатает длину, которую
не рисует.

[Руководство →](../USER_MANUAL_RU.md)

</td>
<td width="54%">

<a href="../../assets/readme/scalebar.mp4"><picture><source srcset="../../assets/readme/scalebar.gif" type="image/gif"><img src="../../assets/readme/scalebar_poster.jpg" alt="Диалог масштабной линейки с живым предпросмотром" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Подписи и аннотации

Ведите управляемый список подписей; у каждой своя область действия (исходный
стек, выровненный стек или конкретный результат), диапазон кадров и стиль.
Предпросмотр и любой экспорт используют одну и ту же цель и один и тот же номер
кадра, поэтому `{value}` читается одинаково в «Сохранить», «Сохранить как»,
перетаскивании и «Экспортировать всё».

</td>
<td width="54%">

<a href="../../assets/readme/labels.mp4"><picture><source srcset="../../assets/readme/labels.gif" type="image/gif"><img src="../../assets/readme/labels_poster.jpg" alt="Менеджер подписей с живым предпросмотром" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Управление агентом

К каждому окну, виджету и меню можно обратиться через локальный канал,
защищённый токеном: агент загружает стек, рендерит, ретуширует, сравнивает,
строит карту глубины, экспортирует и отвечает на диалоги — типизированной
командой, путём к виджету или настоящими событиями мыши и клавиатуры. Распознавание
экрана и OCR не нужны; канал запускается вместе с настольным приложением.

[Протокол →](../AGENT_CONTROL_SPEC.md) · [Навык →](../../skills/openfocus-control/SKILL.md)

</td>
<td width="54%">

<a href="../../assets/readme/agent.mp4"><picture><source srcset="../../assets/readme/agent.gif" type="image/gif"><img src="../../assets/readme/agent_poster.jpg" alt="Агент управляет живым интерфейсом: загрузка, ИИ-рендер, сравнение, ретушь, «Шторка», глубина, подписи" width="100%" /></picture></a>

</td>
</tr>
</table>

**Также в комплекте:**

- **Рабочий процесс фокус-стекинга** — обзор стека, кривая резкости, быстрый
  предпросмотр, ROI-рендер и запоминание настроек каждого метода между запусками.
- **Готовность к микроскопии** — 16 бит по всей цепочке, мкм/пикс из метаданных
  TIFF, коррекция плоского поля по карте пыли, многостраничные Z-стеки TIFF,
  экспорт текстурированного OBJ.
- **Цветовые профили** — экспорт отдельных изображений в PNG, TIFF, JPEG и WebP
  сохраняет общий RGB-профиль ICC, если все исходные кадры имеют одинаковый
  профиль. Для смешанных или неразмеченных стеков профиль не добавляется;
  пиксели не преобразуются в другое цветовое пространство.
- **Прозрачный PNG** — стеки с одинаковыми прямоугольными масками постоянной
  непрозрачности можно объединить в полном размере и сохранить в PNG. Ограничения
  описаны в руководстве.
- **Headless-CLI** — тот же конвейер как `openfocus`, с пакетным режимом,
  `--dry-run`, `--resume` и JSON-выводом для скриптов.
- **Управление агентом** — локальный канал с токеном управляет живым интерфейсом
  (типизированные команды конвейера плюс универсальное управление виджетами,
  меню, мышью и клавиатурой); в комплекте навык, smoke-раннер из 14 проверок и
  полный обход пунктов меню.
- **Проекты и восстановление** — сеансы `.ofproj`, автовосстановление после сбоя,
  отмена и повтор изменений стека и аннотаций.
- **Двенадцать языков интерфейса** — English, 中文, 日本語, Español, Deutsch,
  Français, Русский, 한국어, Português, Italiano, Nederlands, Polski.
  Определяются по системной локали; новый язык — один файл `locales/<code>.py`.
- **Экономия памяти** — нейросеть загружается только при использовании
  нейрометода, поэтому сеанс с классическим алгоритмом занимает около 110 МБ
  вместо 260 МБ.

<a id="install"></a>
## ⬇️ Установка

Скачайте последнюю сборку для Windows или macOS на [сайте OpenFocus](https://openfocus.top/#download). Те же файлы и прежние версии доступны в [GitHub Releases](https://github.com/Xinzhe99/OpenFocus/releases).

| Платформа | Файл | Тип |
|----------|------|-----|
| Windows 10/11 (64 бита) | `OpenFocus-*-setup.exe` | **Установщик** (рекомендуется) |
| Windows 10/11 (64 бита) | `OpenFocus-*-windows-x64.zip` | Портативная версия (распаковать и запустить) |
| macOS Apple Silicon (M1–M4) | `OpenFocus-*-macos.dmg` | Образ диска (перетащить в «Программы») |
| macOS Apple Silicon (M1–M4) | `OpenFocus-*-macos-arm64.zip` | Портативная версия (распаковать) |

Статус подписи сборок Windows указан в примечаниях к выпуску. Сборки macOS не
нотаризованы, поэтому при первом запуске может появиться предупреждение:
**Windows** SmartScreen → «Подробнее» → «Выполнить всё равно»
(первый запуск может занять 30–60 с); **macOS** выполните
`xattr -cr /Applications/OpenFocus.app` в Терминале либо попробуйте открыть
приложение один раз и разрешите его в **Системные настройки → Конфиденциальность
и безопасность → Открыть всё равно** (начиная с macOS 15 щелчок правой кнопкой →
«Открыть» больше не обходит Gatekeeper).

Чтобы установить из терминала, сначала проверьте команду для macOS или Linux с
`--dry-run`, затем выполните её без этого флага. В Linux так устанавливается
headless-CLI; сборки macOS требуют Apple Silicon; приложение из DMG устанавливается в
`/Applications`, а установка из терминала — в `~/Applications`.

```bash
curl -fsSL https://openfocus.top/install.sh | sh -s -- --dry-run
curl -fsSL https://openfocus.top/install.sh | sh
```

В системе Windows запустите установщик из PowerShell:

```powershell
irm https://openfocus.top/install.ps1 | iex
```

Прямая загрузка установщика Windows также доступна командой
`curl -fL https://openfocus.top/download/latest/windows-setup -o OpenFocus-setup.exe`.

### 🤖 Интеграция с агентами

Настольное приложение автоматически запускает локальный канал для агентов,
защищённый токеном. Используйте `openfocus ctl` для типизированных команд или
универсальных действий с интерфейсом работающего окна. Агенты могут загружать
стеки, рендерить, экспортировать, открывать и заполнять диалоги, отвечать на
выбор файлов, отправлять события мыши и клавиатуры и делать снимки экрана.

- Навык: [`skills/openfocus-control/SKILL.md`](../../skills/openfocus-control/SKILL.md)
- Протокол и справочник команд: [`docs/AGENT_CONTROL_SPEC.md`](../AGENT_CONTROL_SPEC.md)
- Сквозной раннер: `python tools/agent_gui_smoke.py --launch`
- Обход всех действий: `python tools/agent_action_sweep.py --launch`

### 💻 Использование командной строки

Либо установите CLI в Windows, macOS и Linux — тот же конвейер без графического стека:

```bash
pip install openfocus
openfocus -i ./stack_folder -o ./result/fused.png
```

См. [руководство по CLI](../USER_MANUAL_RU.md#14-интерфейс-командной-строки).

<a id="community"></a>
## Сообщество и поддержка

- **Документация** — [руководство пользователя](../USER_MANUAL_RU.md) (двенадцать языков) · [история изменений](../../CHANGELOG.md) · [производительность](../PERFORMANCE.md) · [методы объединения](../FUSION_METHODS.md)
- **Цитирование** — [CITATION.cff](../../CITATION.cff); каждый выпуск архивируется на Zenodo ([DOI](https://doi.org/10.5281/zenodo.23050823))
- **Обратная связь** — [issue](https://github.com/Xinzhe99/OpenFocus/issues) для ошибок, идей и работы над алгоритмами
- **Благодарности** — каждый алгоритм объединения и совмещения сохраняет свою исходную исследовательскую лицензию; StackMFF V3 живёт в [Xinzhe99/StackMFF-V3](https://github.com/Xinzhe99/StackMFF-V3)

<a id="license"></a>
## Лицензия

Приложение OpenFocus распространяется по [лицензии MIT](../../LICENSE).

**Исключение для серии StackMFF.** Алгоритмы объединения StackMFF — сейчас
StackMFF V3 и любые будущие издания уровня state-of-the-art — остаются
собственностью проекта OpenFocus, который сохраняет все права на серию StackMFF.
Вы можете использовать их в составе OpenFocus для собственных изображений и
внутренней работы. Копирование, портирование или перенос этих алгоритмов в другой
проект с открытым исходным кодом требует предварительного письменного разрешения.

<a id="developing"></a>
## Разработка

Запуск из репозитория:

```bash
conda create -n openfocus python=3.10
conda activate openfocus
pip install -r requirements.txt
python main.py
```

Упаковка через PyInstaller: [docs/BUILD_COMMANDS.md](../BUILD_COMMANDS.md).

Добавление алгоритма объединения или совмещения: [CONTRIBUTING.md](../../CONTRIBUTING.md).
[Спецификация расширяемости](../ALGORITHM_EXTENSIBILITY_SPEC.md) определяет общие
контракты совмещения, параметров, выполнения и сохранения.

<a id="star-history"></a>
## ⭐ История звёзд

[![Star History Chart](https://api.star-history.com/svg?repos=Xinzhe99/OpenFocus&type=Date)](https://star-history.com/#Xinzhe99/OpenFocus&Date)
