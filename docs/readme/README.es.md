<p align="center">
  <a href="https://github.com/Xinzhe99/OpenFocus">
    <img src="../../assets/OpenFocus.png" alt="OpenFocus" width="128">
  </a>
</p>

<h1 align="center">OpenFocus</h1>

<p align="center"><strong>Cada píxel, desde su fotograma más nítido.</strong></p>

<p align="center">
  Diez algoritmos de fusión, cuatro modos de registro y un espacio de retoque ligado a la fuente para macro, microscopía e imágenes industriales. 16 bits de extremo a extremo, una CLI sin interfaz gráfica y una interfaz en doce idiomas.
</p>

<p align="center">
  <a href="https://github.com/Xinzhe99/OpenFocus/releases"><img src="https://img.shields.io/github/v/release/Xinzhe99/OpenFocus?label=release&color=2f81f7" alt="Versión"></a>
  <a href="https://github.com/Xinzhe99/OpenFocus/stargazers"><img src="https://img.shields.io/github/stars/Xinzhe99/OpenFocus?color=e3b341" alt="Estrellas"></a>
  <a href="../../LICENSE"><img src="https://img.shields.io/badge/license-MIT-3fb950" alt="Licencia: MIT"></a>
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
> **v1.74.0** — Renovación de la fusión OpenFocus A/B/C y del alineamiento rápido. La aplicación de escritorio ahora inicia automáticamente su canal local de agente autenticado, por lo que `openfocus ctl` puede controlar la ventana en ejecución sin ningún indicador de arranque especial.

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

El método predeterminado usa una red neuronal para estimar el fotograma de origen más probable de cada píxel y después mezcla las contribuciones previstas o elige el fotograma más probable. El banco de pruebas sintético compara los métodos con una referencia conocida; los resultados dependen de la pila y de los ajustes.

[StackMFF V3 →](https://github.com/Xinzhe99/StackMFF-V3) · [Banco de pruebas →](../../tools/synthetic_benchmark.py)

</td>
<td width="54%">

<a href="../../assets/readme/ai.mp4"><picture><source srcset="../../assets/readme/ai.gif" type="image/gif"><img src="../../assets/readme/ai_poster.jpg" alt="StackMFF V3 frente al filtro guiado en la misma pila" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Comparación Wipe (cortina)

Arrastra el divisor entre un fotograma y el resultado fusionado para ver exactamente qué cambió la mezcla: deriva de alineación, halos en los bordes y detalle perdido, con cualquier zoom.

[Manual →](../USER_MANUAL_ES.md)

</td>
<td width="54%">

<a href="../../assets/readme/wipe.mp4"><picture><source srcset="../../assets/readme/wipe.gif" type="image/gif"><img src="../../assets/readme/wipe_poster.jpg" alt="Comparación A/B entre un fotograma y el resultado fusionado" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Comparación de métodos

Renderiza todos los métodos una vez, compáralos lado a lado en el mismo recorte y adopta el ganador como nueva salida o copia sus parámetros a la ventana principal.

</td>
<td width="54%">

<a href="../../assets/readme/method-comparison.mp4"><picture><source srcset="../../assets/readme/method-comparison.gif" type="image/gif"><img src="../../assets/readme/method-comparison_poster.jpg" alt="Comparación lado a lado de ocho métodos de fusión" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Mapa de profundidad y 3D

Posición de enfoque por píxel con cinco medidas, mapas de color científicos, calibración Z física a partir de metadatos TIFF y exportación OBJ texturizada de la superficie.

</td>
<td width="54%">

<a href="../../assets/readme/depth.mp4"><picture><source srcset="../../assets/readme/depth.gif" type="image/gif"><img src="../../assets/readme/depth_poster.jpg" alt="Mapa de profundidad y superficie 3D de la misma pila" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Retoque ligado a la fuente

Pinta desde los fotogramas que produjeron el resultado: zoom y desplazamiento compartidos, control de fotograma, pincel suave, pincel de restauración, deshacer por trazo y protección de píxeles sin exportar. Aplica el retoque como nueva salida y usa **Exportar TIFF por capas** para guardarla con sus fotogramas de origen. Las etiquetas y barras de escala activas no se incluyen. GIMP 3.2.6 en Windows abre el resultado y las capas de origen; otros editores no se han probado.

</td>
<td width="54%">

<a href="../../assets/readme/retouch.mp4"><picture><source srcset="../../assets/readme/retouch.gif" type="image/gif"><img src="../../assets/readme/retouch_poster.jpg" alt="Espacio de retoque con fotograma de origen y resultado" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Procesamiento por lotes

Apunta a una carpeta de pilas, elige un método y un destino, y déjalo trabajar. El modo de solo alineación escribe una carpeta por pila; la fusión escribe la imagen fusionada y lista exactamente eso. La fusión agrupada opcional cambia la equivalencia exacta con la pila completa por un uso de memoria y espacio temporal acotados.

</td>
<td width="54%">

<a href="../../assets/readme/batch.mp4"><picture><source srcset="../../assets/readme/batch.gif" type="image/gif"><img src="../../assets/readme/batch_poster.jpg" alt="Diálogo de procesamiento por lotes" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Barras de escala para publicación

Calibradas desde metadatos TIFF o introducidas a mano, dibujadas con la fuente y la longitud reales e incrustadas en todas las exportaciones: imágenes individuales, pilas, TIFF multipágina, GIF y lotes. Si una barra no cabe de forma legible, lo indica en lugar de imprimir una longitud que no dibuja.

[Manual →](../USER_MANUAL_ES.md)

</td>
<td width="54%">

<a href="../../assets/readme/scalebar.mp4"><picture><source srcset="../../assets/readme/scalebar.gif" type="image/gif"><img src="../../assets/readme/scalebar_poster.jpg" alt="Diálogo de barra de escala con vista previa" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Etiquetas y leyendas

Mantén una lista gestionada de leyendas, cada una con su ámbito (pila de origen, pila alineada o una salida concreta), rango de fotogramas y estilo. La vista previa y todas las exportaciones usan el mismo destino y número de fotograma, así que `{value}` se lee igual en Guardar, Guardar como, arrastrar y Exportar todo.

</td>
<td width="54%">

<a href="../../assets/readme/labels.mp4"><picture><source srcset="../../assets/readme/labels.gif" type="image/gif"><img src="../../assets/readme/labels_poster.jpg" alt="Gestor de etiquetas con vista previa" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Controlable por agentes

Cada ventana, widget y menú se puede controlar mediante un canal local protegido por token: un agente carga una pila, renderiza, retoca, compara, calcula profundidad, exporta y responde a diálogos mediante comandos tipados, rutas de widgets o eventos reales de ratón y teclado. No hace falta capturar la pantalla ni usar OCR; el canal se inicia con la aplicación de escritorio.

[Protocolo →](../AGENT_CONTROL_SPEC.md) · [Skill →](../../skills/openfocus-control/SKILL.md)

</td>
<td width="54%">

<a href="../../assets/readme/agent.mp4"><picture><source srcset="../../assets/readme/agent.gif" type="image/gif"><img src="../../assets/readme/agent_poster.jpg" alt="Un agente controla la interfaz en vivo" width="100%" /></picture></a>

</td>
</tr>
</table>

**También incluye:**

- **Flujo de focus stacking** — explorador de pilas, curva de nitidez, vista previa rápida, render por ROI y ajustes por método recordados entre reinicios.
- **Listo para microscopía** — 16 bits de extremo a extremo, µm/px desde metadatos TIFF, corrección de campo plano mediante mapa de polvo, pilas Z TIFF multipágina y exportación OBJ texturizada.
- **Perfiles de color** — las exportaciones individuales a PNG, TIFF, JPEG y WebP conservan un perfil ICC RGB compartido cuando todos los fotogramas de origen tienen el mismo perfil. Las pilas mixtas o sin perfil se exportan sin perfil; los píxeles no se convierten entre espacios de color.
- **PNG transparente** — las pilas con máscaras rectangulares de opacidad constante coincidentes se pueden fusionar a tamaño completo y guardar como PNG. Consulta las restricciones en el manual.
- **CLI sin interfaz** — el mismo flujo de procesamiento como `openfocus`, con modo lote, `--dry-run`, `--resume` y salida JSON para scripts.
- **Controlable por agentes** — un canal local protegido por token controla la interfaz en vivo (comandos tipados de procesamiento y control genérico de widgets, menús, ratón y teclado), con una skill, un ejecutor de humo de 14 comprobaciones y un barrido completo de acciones de menú.
- **Proyectos y recuperación** — sesiones `.ofproj`, recuperación automática tras fallos y deshacer/rehacer en ediciones de pilas y anotaciones.
- **Doce idiomas de interfaz** — English, 中文, 日本語, Español, Deutsch, Français, Русский, 한국어, Português, Italiano, Nederlands, Polski. Se detectan desde la configuración regional del sistema; un idioma nuevo es un archivo `locales/<code>.py`.
- **Ligero en memoria** — la red neuronal solo se carga cuando se usa el método de IA, así que una sesión con métodos clásicos ronda los 110 MB en lugar de 260 MB.

<a id="install"></a>
## ⬇️ Instalación

Descarga la última compilación para Windows o macOS desde el [sitio web de OpenFocus](https://openfocus.top/#download). Los mismos archivos y las versiones anteriores también están en [GitHub Releases](https://github.com/Xinzhe99/OpenFocus/releases).

| Plataforma | Archivo | Tipo |
|------------|---------|------|
| Windows 10/11 (64 bits) | `OpenFocus-*-setup.exe` | **Instalador** (recomendado) |
| Windows 10/11 (64 bits) | `OpenFocus-*-windows-x64.zip` | Portable (descomprimir y ejecutar) |
| macOS Apple Silicon (M1–M4) | `OpenFocus-*-macos.dmg` | Imagen de disco (arrastrar a Aplicaciones) |
| macOS Apple Silicon (M1–M4) | `OpenFocus-*-macos-arm64.zip` | Portable (descomprimir) |

La aplicación no está firmada, así que el primer inicio puede mostrar un aviso de seguridad: en **Windows**, SmartScreen → «Más información» → «Ejecutar de todas formas» (el primer arranque puede tardar 30–60 s); en **macOS**, clic derecho → Abrir, o ejecuta `xattr -cr /Applications/OpenFocus.app` en Terminal.

Para instalar desde una terminal, previsualiza primero el comando para macOS o Linux y ejecútalo después sin `--dry-run`. En Linux instala la CLI sin interfaz; las compilaciones para macOS requieren Apple Silicon y se instalan en `~/Applications`.

```bash
curl -fsSL https://openfocus.top/install.sh | sh -s -- --dry-run
curl -fsSL https://openfocus.top/install.sh | sh
```

En Windows, ejecuta el instalador desde PowerShell:

```powershell
irm https://openfocus.top/install.ps1 | iex
```

La descarga directa del instalador de Windows también está disponible con
`curl -fL https://openfocus.top/download/latest/windows-setup -o OpenFocus-setup.exe`.

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

- **Documentación** — [Manual](../USER_MANUAL_ES.md) (doce idiomas) · [Cambios](../../CHANGELOG.md) · [Rendimiento](../PERFORMANCE.md) · [Métodos de fusión](../FUSION_METHODS.md)
- **Citación** — [CITATION.cff](../../CITATION.cff); cada versión se archiva en Zenodo ([DOI](https://doi.org/10.5281/zenodo.23050823))
- **Comentarios** — abre un [issue](https://github.com/Xinzhe99/OpenFocus/issues) para fallos, ideas y mejoras de algoritmos
- **Créditos** — cada algoritmo de fusión y registro conserva su licencia de investigación original; StackMFF V3 está en [Xinzhe99/StackMFF-V3](https://github.com/Xinzhe99/StackMFF-V3)

<a id="license"></a>
## Licencia

La aplicación OpenFocus se distribuye bajo la [licencia MIT](../../LICENSE).

**Excepción de la serie StackMFF.** Los algoritmos de fusión StackMFF —actualmente StackMFF V3 y cualquier futura edición de última generación— siguen siendo propiedad del proyecto OpenFocus, que conserva todos los derechos sobre la serie StackMFF. Puedes usarlos como parte de OpenFocus para tus propias imágenes y trabajo interno. Copiar, portar o migrar estos algoritmos a otro proyecto de código abierto requiere permiso previo por escrito.

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
