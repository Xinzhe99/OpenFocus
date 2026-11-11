<p align="center">
  <a href="https://github.com/Xinzhe99/OpenFocus">
    <img src="../../assets/OpenFocus.png" alt="OpenFocus" width="128">
  </a>
</p>

<h1 align="center">OpenFocus</h1>

<p align="center"><strong>Cada píxel, desde su fotograma más nítido.</strong></p>

<p align="center">
  Diez algoritmos de fusión, cinco opciones de alineación (incluida ninguna) y un espacio de retoque ligado a la fuente para macro, microscopía e imágenes industriales. 16 bits de extremo a extremo, una CLI sin interfaz gráfica y una interfaz en doce idiomas.
</p>

<p align="center">
  <a href="https://github.com/Xinzhe99/OpenFocus/releases"><img src="https://img.shields.io/github/v/release/Xinzhe99/OpenFocus?label=release&color=2f81f7" alt="Versión"></a>
  <a href="https://github.com/Xinzhe99/OpenFocus/stargazers"><img src="https://img.shields.io/github/stars/Xinzhe99/OpenFocus?color=e3b341" alt="Estrellas"></a>
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
  <a href="#install">Descargar</a> ·
  <a href="../USER_MANUAL_ES.md">Documentación</a> ·
  <a href="../../CHANGELOG.md">Cambios</a> ·
  <a href="https://doi.org/10.5281/zenodo.23050823">DOI</a>
</p>

<p align="center">
  <a href="../../README.md">English</a> ·
  <a href="README.zh-CN.md">简体中文</a> ·
  <a href="README.ja.md">日本語</a> ·
  <strong>Español</strong> ·
  <a href="README.de.md">Deutsch</a> ·
  <a href="README.fr.md">Français</a> ·
  <a href="README.ru.md">Русский</a> ·
  <a href="README.ko.md">한국어</a> ·
  <a href="README.pt.md">Português</a> ·
  <a href="README.it.md">Italiano</a> ·
  <a href="README.nl.md">Nederlands</a> ·
  <a href="README.pl.md">Polski</a>
</p>

<p align="center">
  <img src="../../assets/ui.png" alt="Interfaz de OpenFocus" width="880">
</p>

## 📢 Novedades

> [!NOTE]
> **[v1.77.0-beta.7, versión preliminar](https://github.com/Xinzhe99/OpenFocus/releases/tag/v1.77.0-beta.7)** — Aceleración GPU para dispositivos OpenCL compatibles en OpenFocus B, con alternativa CPU; comparación por deslizamiento corregida; fondo automático y retoque. Más detalles en el [registro de cambios](../../CHANGELOG.md).

Las versiones anteriores están en el [registro de cambios](../../CHANGELOG.md).

📸 Si publicas imágenes creadas con OpenFocus, considera añadir una nota como:

> Creado con OpenFocus — https://github.com/Xinzhe99/OpenFocus

No es obligatorio, pero se agradece mucho.

<a id="features"></a>
## Funciones

<table>
<tr>
<td width="46%" valign="middle">

### Fusión con IA: StackMFF V3

StackMFF V3 usa IA para combinar las partes nítidas de los fotogramas de una pila.

[StackMFF V3 →](https://github.com/Xinzhe99/StackMFF-V3) · [Banco de pruebas →](../../tools/synthetic_benchmark.py)

</td>
<td width="54%">

<a href="../../assets/readme/ai.mp4"><picture><source srcset="../../assets/readme/ai.gif" type="image/gif"><img src="../../assets/readme/ai_poster.jpg" alt="StackMFF V3 frente al filtro guiado en la misma pila" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Comparación Wipe (cortina)

Arrastra el divisor para comparar un fotograma de origen con el resultado fusionado.

[Manual →](../USER_MANUAL_ES.md)

</td>
<td width="54%">

<a href="../../assets/readme/wipe.mp4"><picture><source srcset="../../assets/readme/wipe.gif" type="image/gif"><img src="../../assets/readme/wipe_poster.jpg" alt="Comparación A/B entre un fotograma y el resultado fusionado" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Comparación de métodos

Compara los métodos lado a lado y reutiliza el resultado o los ajustes que prefieras.

</td>
<td width="54%">

<a href="../../assets/readme/method-comparison.mp4"><picture><source srcset="../../assets/readme/method-comparison.gif" type="image/gif"><img src="../../assets/readme/method-comparison_poster.jpg" alt="Comparación lado a lado de ocho métodos de fusión" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Mapa de profundidad y 3D

Crea un mapa de profundidad y exporta una superficie 3D texturizada.

</td>
<td width="54%">

<a href="../../assets/readme/depth.mp4"><picture><source srcset="../../assets/readme/depth.gif" type="image/gif"><img src="../../assets/readme/depth_poster.jpg" alt="Mapa de profundidad y superficie 3D de la misma pila" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Retoque ligado a la fuente

Retoca el resultado pintando desde los fotogramas de origen y expórtalo con ellos en un TIFF por capas. El fondo automático sustituye un fondo apilado irregular por un fotograma real.

</td>
<td width="54%">

<a href="../../assets/readme/retouch.mp4"><picture><source srcset="../../assets/readme/retouch.gif" type="image/gif"><img src="../../assets/readme/retouch_poster.jpg" alt="Espacio de retoque con fotograma de origen y resultado" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Procesamiento por lotes

Procesa varias pilas a la vez, con fusión agrupada opcional para limitar la memoria y el espacio temporal.

</td>
<td width="54%">

<a href="../../assets/readme/batch.mp4"><picture><source srcset="../../assets/readme/batch.gif" type="image/gif"><img src="../../assets/readme/batch_poster.jpg" alt="Diálogo de procesamiento por lotes" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Barras de escala para publicación

Añade barras de escala calibradas mediante metadatos TIFF o introducidas manualmente.

[Manual →](../USER_MANUAL_ES.md)

</td>
<td width="54%">

<a href="../../assets/readme/scalebar.mp4"><picture><source srcset="../../assets/readme/scalebar.gif" type="image/gif"><img src="../../assets/readme/scalebar_poster.jpg" alt="Diálogo de barra de escala con vista previa" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Etiquetas y leyendas

Añade etiquetas y leyendas con ámbito, intervalo de fotogramas y estilo definidos.

</td>
<td width="54%">

<a href="../../assets/readme/labels.mp4"><picture><source srcset="../../assets/readme/labels.gif" type="image/gif"><img src="../../assets/readme/labels_poster.jpg" alt="Gestor de etiquetas con vista previa" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Controlable por agentes

Un agente puede controlar la aplicación mediante un canal local protegido por token.

[Protocolo →](../AGENT_CONTROL_SPEC.md) · [Skill →](../../skills/openfocus-control/SKILL.md)

</td>
<td width="54%">

<a href="../../assets/readme/agent.mp4"><picture><source srcset="../../assets/readme/agent.gif" type="image/gif"><img src="../../assets/readme/agent_poster.jpg" alt="Un agente controla la interfaz en vivo" width="100%" /></picture></a>

</td>
</tr>
</table>

**También incluye:**

- **Focus stacking** — explorador de pilas, curva de nitidez, vista previa, render por ROI y ajustes guardados.
- **Microscopía** — 16 bits, calibración TIFF, corrección de campo plano y pilas Z TIFF multipágina.
- **Perfiles de color** — conserva el perfil ICC RGB común en las exportaciones individuales compatibles.
- **PNG transparente** — fusión de pilas con máscaras rectangulares compatibles; consulta las restricciones en el manual.
- **CLI sin interfaz** — lotes, `--dry-run`, `--resume` y salida JSON con `openfocus`.
- **Proyectos y recuperación** — sesiones `.ofproj`, recuperación automática y deshacer/rehacer.
- **Doce idiomas** — detección automática y ampliación mediante archivos `locales/<code>.py`.
- **Memoria** — la red neuronal solo se carga al usar el método de IA.

<a id="install"></a>
## ⬇️ Instalación

El [sitio web de OpenFocus](https://openfocus.top/#download) y sus scripts de instalación ofrecen para Windows y macOS la versión más reciente, incluida una beta más nueva (actualmente v1.77.0-beta.7). También puedes descargar versiones desde [GitHub Releases](https://github.com/Xinzhe99/OpenFocus/releases). Las actualizaciones automáticas de escritorio también ofrecen betas más nuevas; la CLI de Linux de PyPI permanece estable.

Consulta el [manual](../USER_MANUAL_ES.md) para la instalación y la resolución de problemas.

En macOS con Apple Silicon o Linux, instala desde la terminal; en Linux se instala la CLI sin interfaz y en macOS la aplicación se guarda en `~/Applications`:

```bash
curl -fsSL https://openfocus.top/install.sh | sh -s -- --dry-run # opcional: vista previa
curl -fsSL https://openfocus.top/install.sh | sh
```

En Windows, ejecuta el instalador desde PowerShell:

```powershell
irm https://openfocus.top/install.ps1 | iex
```

Descarga directa del instalador de Windows:

```powershell
curl.exe -fL https://openfocus.top/download/latest/windows-setup -o OpenFocus-setup.exe
```

### 🤖 Integración con agentes

La aplicación de escritorio inicia automáticamente un canal local de agente protegido por token. Usa `openfocus ctl` para enviar comandos tipados o acciones genéricas de interfaz a la ventana en vivo. Los agentes pueden cargar pilas, renderizar, exportar, abrir y rellenar diálogos, responder a selectores de archivos, enviar eventos de ratón y teclado y tomar capturas de pantalla.

- Skill: [`skills/openfocus-control/SKILL.md`](../../skills/openfocus-control/SKILL.md)
- Protocolo y referencia de comandos: [`docs/AGENT_CONTROL_SPEC.md`](../AGENT_CONTROL_SPEC.md)
- Ejecutor de extremo a extremo: `python tools/agent_gui_smoke.py --launch`
- Barrido de todas las acciones: `python tools/agent_action_sweep.py --launch`

### 💻 Uso desde la línea de comandos

También puedes instalar la CLI en Windows, macOS y Linux: el mismo flujo de procesamiento, sin la pila gráfica:

```bash
pip install openfocus
openfocus -i ./stack_folder -o ./result/fused.png
```

Consulta el [manual de la CLI](../USER_MANUAL_ES.md#14-interfaz-de-línea-de-comandos).

<a id="community"></a>
## Comunidad y soporte

¿Preguntas, ideas o comentarios? Únete a nuestra [comunidad de Discord](https://discord.gg/afzmwgPt). Respondemos lo antes posible; usamos la hora de Pekín (UTC+8), por lo que la diferencia horaria puede afectar al tiempo de respuesta.

- **Documentación** — [Manual](../USER_MANUAL_ES.md) (doce idiomas) · [Cambios](../../CHANGELOG.md) · [Rendimiento](../PERFORMANCE.md) · [Métodos de fusión](../FUSION_METHODS.md)
- **Citación** — [CITATION.cff](../../CITATION.cff); cada versión se archiva en Zenodo ([DOI](https://doi.org/10.5281/zenodo.23050823))
- **Comentarios** — abre un [issue](https://github.com/Xinzhe99/OpenFocus/issues) para fallos, ideas y mejoras de algoritmos
- **Créditos** — cada algoritmo de fusión y registro conserva su licencia de investigación original; StackMFF V3 está en [Xinzhe99/StackMFF-V3](https://github.com/Xinzhe99/StackMFF-V3)

<a id="license"></a>
## Licencia

La aplicación OpenFocus se distribuye bajo la [licencia MIT](../../LICENSE).

**Licencias de StackMFF.** Los nuevos pesos y wrappers de V1/V2 están sujetos a StackMFF V3 License 1.0. Copiarlos, distribuirlos, incorporarlos a otros proyectos o usarlos comercialmente requiere permiso previo por escrito del titular de los derechos. La instantánea v2.0 original de StackMFF V3 incluida en el proyecto sigue bajo licencia MIT. Consulta los [avisos de terceros](../../docs/THIRD_PARTY_NOTICES.md).

<a id="developing"></a>
## Desarrollo

Ejecutar desde un clon:

```bash
conda create -n openfocus python=3.10
conda activate openfocus
pip install -r requirements.txt
python main.py
```

Empaquetado con PyInstaller: [docs/BUILD_COMMANDS.md](../BUILD_COMMANDS.md).

Añadir un algoritmo de fusión o registro: [CONTRIBUTING.md](../../CONTRIBUTING.md).
La [especificación de extensibilidad](../ALGORITHM_EXTENSIBILITY_SPEC.md) define los contratos compartidos de registro, parámetros, ejecución y persistencia.

<a id="star-history"></a>
## ⭐ Historial de estrellas

[![Star History Chart](https://api.star-history.com/svg?repos=Xinzhe99/OpenFocus&type=Date)](https://star-history.com/#Xinzhe99/OpenFocus&Date)
