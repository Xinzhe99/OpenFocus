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

StackMFF V3 использует ИИ для объединения резких участков исходных кадров.

[StackMFF V3 →](https://github.com/Xinzhe99/StackMFF-V3) · [Бенчмарк →](../../tools/synthetic_benchmark.py)

</td>
<td width="54%">

<a href="../../assets/readme/ai.mp4"><picture><source srcset="../../assets/readme/ai.gif" type="image/gif"><img src="../../assets/readme/ai_poster.jpg" alt="StackMFF V3 и Guided Filter на одном стеке" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Сравнение «Шторка»

Сравнивайте исходный кадр и результат объединения с помощью подвижного разделителя.

[Руководство →](../USER_MANUAL_RU.md)

</td>
<td width="54%">

<a href="../../assets/readme/wipe.mp4"><picture><source srcset="../../assets/readme/wipe.gif" type="image/gif"><img src="../../assets/readme/wipe_poster.jpg" alt="Сравнение «Шторка» между исходным кадром и результатом" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Сравнение методов

Сравнивайте методы рядом и используйте выбранный результат или его настройки.

</td>
<td width="54%">

<a href="../../assets/readme/method-comparison.mp4"><picture><source srcset="../../assets/readme/method-comparison.gif" type="image/gif"><img src="../../assets/readme/method-comparison_poster.jpg" alt="Восемь методов объединения рядом" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Карта глубины и 3D

Создавайте карты глубины и экспортируйте текстурированные 3D-поверхности.

</td>
<td width="54%">

<a href="../../assets/readme/depth.mp4"><picture><source srcset="../../assets/readme/depth.gif" type="image/gif"><img src="../../assets/readme/depth_poster.jpg" alt="Карта глубины и 3D-поверхность одного стека" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Ретушь с привязкой к источникам

Ретушируйте результат пикселями из исходных кадров.

</td>
<td width="54%">

<a href="../../assets/readme/retouch.mp4"><picture><source srcset="../../assets/readme/retouch.gif" type="image/gif"><img src="../../assets/readme/retouch_poster.jpg" alt="Рабочее пространство ретуши с исходным кадром и результатом" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Пакетная обработка

Обрабатывайте несколько стеков с возможностью группового объединения для ограничения расхода памяти.

</td>
<td width="54%">

<a href="../../assets/readme/batch.mp4"><picture><source srcset="../../assets/readme/batch.gif" type="image/gif"><img src="../../assets/readme/batch_poster.jpg" alt="Диалог пакетной обработки" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Масштабные линейки для публикаций

Добавляйте к экспортируемым изображениям масштабные линейки с калибровкой вручную или по метаданным TIFF.

[Руководство →](../USER_MANUAL_RU.md)

</td>
<td width="54%">

<a href="../../assets/readme/scalebar.mp4"><picture><source srcset="../../assets/readme/scalebar.gif" type="image/gif"><img src="../../assets/readme/scalebar_poster.jpg" alt="Диалог масштабной линейки с живым предпросмотром" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Подписи и аннотации

Управляйте подписями и аннотациями с собственными областью действия и стилем.

</td>
<td width="54%">

<a href="../../assets/readme/labels.mp4"><picture><source srcset="../../assets/readme/labels.gif" type="image/gif"><img src="../../assets/readme/labels_poster.jpg" alt="Менеджер подписей с живым предпросмотром" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Управление агентом

Управляйте приложением через локальный канал агента, защищённый токеном.

[Протокол →](../AGENT_CONTROL_SPEC.md) · [Навык →](../../skills/openfocus-control/SKILL.md)

</td>
<td width="54%">

<a href="../../assets/readme/agent.mp4"><picture><source srcset="../../assets/readme/agent.gif" type="image/gif"><img src="../../assets/readme/agent_poster.jpg" alt="Агент управляет живым интерфейсом: загрузка, ИИ-рендер, сравнение, ретушь, «Шторка», глубина, подписи" width="100%" /></picture></a>

</td>
</tr>
</table>

**Также в комплекте:**

- **Рабочий процесс фокус-стекинга** — обзор стека, кривая резкости, быстрый предпросмотр, ROI-рендер и сохранённые настройки.
- **Готовность к микроскопии** — 16 бит, калибровка TIFF, коррекция плоского поля и многостраничные Z-стеки.
- **Цветовые профили** — экспорт отдельных изображений сохраняет общий профиль RGB ICC без преобразования цветов.
- **Прозрачный PNG** — объединение и экспорт совместимых стеков с прямоугольными масками в полном размере.
- **Headless-CLI** — `openfocus` с пакетным режимом, `--dry-run`, `--resume` и выводом JSON.
- **Проекты и восстановление** — сеансы `.ofproj`, восстановление после сбоя, отмена и повтор изменений.
- **Двенадцать языков интерфейса** — автоматическое определение; новые языки добавляются в `locales/<code>.py`.
- **Экономия памяти** — нейросеть загружается только при необходимости.

<a id="install"></a>
## ⬇️ Установка

Скачайте последнюю сборку для Windows или macOS на [сайте OpenFocus](https://openfocus.top/#download). Те же файлы и прежние версии доступны в [GitHub Releases](https://github.com/Xinzhe99/OpenFocus/releases).

Инструкции по установке и устранению неполадок приведены в [руководстве](../USER_MANUAL_RU.md).

В Linux команда терминала устанавливает headless-CLI; для macOS требуется Apple Silicon. Приложение из DMG устанавливается в `/Applications`, а из терминала — в `~/Applications`.

```bash
curl -fsSL https://openfocus.top/install.sh | sh -s -- --dry-run
curl -fsSL https://openfocus.top/install.sh | sh
```

В системе Windows запустите установщик из PowerShell:

```powershell
irm https://openfocus.top/install.ps1 | iex
```

Прямая загрузка установщика Windows:

```powershell
curl.exe -fL https://openfocus.top/download/latest/windows-setup -o OpenFocus-setup.exe
```

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

Вопросы, идеи или отзывы? Присоединяйтесь к нашему [сообществу Discord](https://discord.gg/afzmwgPt). Мы отвечаем как можно скорее; мы работаем по пекинскому времени (UTC+8), поэтому разница во времени может повлиять на скорость ответа.

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
