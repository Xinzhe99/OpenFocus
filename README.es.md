<p align="center">
  <a href="https://github.com/Xinzhe99/OpenFocus">
    <img src="assets/OpenFocus.png" alt="OpenFocus" width="128">
  </a>
</p>

<h1 align="center">OpenFocus</h1>

<p align="center"><strong>Todos los planos de enfoque. Una imagen totalmente nítida.</strong></p>

<p align="center">
  Ocho algoritmos de fusión, tres modos de registro y un espacio de retoque ligado a la fuente para macro, microscopía e imágenes industriales. 16 bits de extremo a extremo, CLI sin interfaz y una UI en doce idiomas.
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
  <a href="#download">Descargar</a> ·
  <a href="./docs/USER_MANUAL_ES.md">Documentación</a> ·
  <a href="#command-line-usage">CLI</a> ·
  <a href="./CHANGELOG.md">Cambios</a> ·
  <a href="https://doi.org/10.5281/zenodo.23050823">DOI</a>
</p>

<p align="center">
  <a href="README.md">English</a> ·
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
  <img src="assets/ui.png" alt="Interfaz de OpenFocus" width="880">
</p>

## 📢 News

> [!NOTE]
> **v1.68.1** — la exportación por lotes sigue el modo elegido: solo alineación crea una carpeta `<pila>_aligned` y la lista, la fusión lista solo la imagen fusionada, y sin exportar la alineación solo se escribe esa imagen. Las copias `*_registered*` que quedan en la carpeta de origen ya no se cargan como fotogramas. También se rehicieron etiquetas, barras de escala, comparación de métodos, exportación de profundidad/modelo, actualizaciones y retoque según la auditoría de `docs/ux-specs`.

<a id="download"></a>
## Descarga e instalación

Descarga una compilación desde [Releases](https://github.com/Xinzhe99/OpenFocus/releases):

| Plataforma | Archivo | Tipo |
|------------|---------|------|
| Windows 10/11 (64 bits) | `OpenFocus-*-setup.exe` | **Instalador** (recomendado) |
| Windows 10/11 (64 bits) | `OpenFocus-*-windows-x64.zip` | Portable (descomprimir y ejecutar) |
| macOS Apple Silicon (M1–M4) | `OpenFocus-*-macos.dmg` | Imagen de disco (arrastrar a Aplicaciones) |
| macOS Apple Silicon (M1–M4) | `OpenFocus-*-macos-arm64.zip` | Portable (descomprimir) |

La aplicación no está firmada, así que el primer inicio puede mostrar un aviso: en Windows, «Más información → Ejecutar de todos modos»; en macOS, clic derecho → Abrir, o `xattr -cr /Applications/OpenFocus.app`.

## Funciones

<table>

<tr>
<td width="46%" valign="middle">

### Ocho algoritmos de fusión

Cuatro clásicos más la familia OpenFocus A/B/C y el modelo neuronal StackMFF-V3, todos sobre la misma pila registrada: compara y elige por motivo en vez de confiar en un único valor por defecto.

[Algoritmos →](#algorithms)

</td>
<td width="54%">

<img src="assets/readme/result.png" alt="Resultado de fusión en la ventana de OpenFocus" width="100%" />

</td>
</tr>

<tr>
<td width="46%" valign="middle">

### Comparación A/B

Arrastra el divisor entre un fotograma y el resultado fusionado para ver qué cambió la mezcla: deriva de registro, halos en los bordes y detalle perdido, a cualquier zoom.

[Manual →](./docs/USER_MANUAL_ES.md)

</td>
<td width="54%">

<img src="assets/readme/wipe.png" alt="Comparación A/B entre un fotograma y el resultado" width="100%" />

</td>
</tr>

<tr>
<td width="46%" valign="middle">

### Comparación de métodos

Renderiza todos los métodos una vez, compáralos lado a lado en el mismo recorte y adopta el ganador como nueva salida o copia sus parámetros a la ventana principal.

</td>
<td width="54%">

<img src="assets/readme/compare.png" alt="Ocho métodos de fusión en paralelo" width="100%" />

</td>
</tr>

<tr>
<td width="46%" valign="middle">

### Mapa de profundidad y 3D

Posición de foco por píxel con cinco medidas, mapas de color científicos, calibración Z física desde metadatos TIFF y exportación OBJ texturizada.

</td>
<td width="54%">

<img src="assets/readme/depth.png" alt="Mapa de profundidad y superficie 3D de la misma pila" width="100%" />

</td>
</tr>

<tr>
<td width="46%" valign="middle">

### Retoque ligado a la fuente

Pinta desde los fotogramas que produjeron el resultado: zoom y desplazamiento compartidos, control de fotograma, pincel suave, pincel de restauración, deshacer por trazo y protección de píxeles sin exportar.

</td>
<td width="54%">

<img src="assets/readme/retouch.png" alt="Espacio de retoque con fotograma y resultado" width="100%" />

</td>
</tr>

<tr>
<td width="46%" valign="middle">

### Procesamiento por lotes

Apunta a una carpeta de pilas, elige método y destino, y déjalo trabajar. Solo alineación escribe una carpeta por pila; la fusión escribe y lista solo la imagen fusionada.

</td>
<td width="54%">

<img src="assets/readme/batch.png" alt="Diálogo de procesamiento por lotes" width="100%" />

</td>
</tr>

<tr>
<td width="46%" valign="middle">

### Barras de escala de publicación

Calibradas desde metadatos TIFF o a mano, dibujadas con la fuente y longitud reales, e incrustadas en cada exportación: imagen, pila, TIFF, GIF y lote.

</td>
<td width="54%">

<img src="assets/readme/scalebar.png" alt="Diálogo de barra de escala con vista previa" width="100%" />

</td>
</tr>

</table>

**También incluye:**

- **Flujo de focus stacking**——explorador de pilas, curva de nitidez, vista previa rápida, render ROI y ajustes por método recordados entre sesiones.
- **Listo para microscopía**——16 bits de extremo a extremo, µm/px desde TIFF, corrección de campo plano, Z-stacks TIFF multipágina.
- **CLI sin interfaz**——la misma tubería como `openfocus`, con lotes, `--dry-run`, `--resume` y salida JSON.
- **Proyectos y recuperación**——sesiones `.ofproj`, recuperación tras fallos y deshacer/rehacer de ediciones y anotaciones.
- **Doce idiomas de interfaz**——English, 简体中文, 日本語, Español, Deutsch, Français, Русский, 한국어, Português, Italiano, Nederlands, Polski; detectados del sistema, un idioma nuevo es un archivo `locales_<code>.py`.
- **Ligero en memoria**——los pesos neuronales se cargan solo al usar el método AI, así que una sesión clásica ronda 110 MB en vez de 260 MB.

## Inicio rápido

```bash
# CLI completa: todos los algoritmos, modelo de IA, HEIC y RAW de cámara
pip install openfocus

# Fusionar una carpeta de imágenes
openfocus -i ./stack_folder -o ./result/fused.png

# DTCWT con registro ECC y 8 hilos
openfocus -i ./stack_folder -o ./result/fused.png -m dtcwt -a ecc -t 8
```

Iniciar la GUI desde el código fuente:

```bash
conda create -n openfocus python=3.10
conda activate openfocus
pip install -r requirements.txt
python main.py
```

## Documentación

- [Manual de usuario (español)](./docs/USER_MANUAL_ES.md)
- [Historial de cambios](./CHANGELOG.md)
- [Rendimiento y pilas grandes](./docs/PERFORMANCE.md)
- [Comandos de compilación](./docs/BUILD_COMMANDS.md)
- [Paridad de algoritmos](./docs/ALGORITHM_PARITY.md)

La referencia completa en inglés está en [README.md](./README.md).

## Cita y licencia

Si OpenFocus contribuye a tu trabajo, cita el proyecto usando [CITATION.cff](./CITATION.cff).

El proyecto se distribuye bajo la [licencia MIT](./LICENSE). Cada algoritmo de fusión/registro conserva además su licencia original.
