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

# <img src="assets/OpenFocus.png" alt="OpenFocus Logo" width="120"> OpenFocus

OpenFocus es una estación de trabajo de código abierto para el registro y la fusión de imágenes de enfoque múltiple, con una calidad comparable al software comercial y fácil de ampliar.

<p align="left">
  <a href="https://www.python.org/downloads/release/python-3100/"><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+" /></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?logo=open-source-initiative&logoColor=white" alt="License: MIT" /></a>
  <a href="https://github.com/Xinzhe99/OpenFocus"><img src="https://img.shields.io/badge/GitHub-Repository-181717?logo=github&logoColor=white" alt="GitHub Repository" /></a>
  <a href="https://github.com/Xinzhe99/OpenFocus/releases"><img src="https://img.shields.io/badge/Windows%20%7C%20macOS-Download-0078D7?logo=github&logoColor=white" alt="Download" /></a>
  <a href="https://doi.org/10.5281/zenodo.23050823"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.23050823-0078D7?logo=doi&logoColor=white" alt="DOI" /></a>
</p>

<p align="center">
  <img src="assets/ui.png" alt="Interfaz de OpenFocus" width="820">
</p>

> **v1.68.0** — la exportación por lotes sigue el modo elegido: solo alineación crea una carpeta `<pila>_aligned` y la lista, la fusión lista solo la imagen fusionada, y sin exportar la alineación solo se escribe esa imagen. Las copias `*_registered*` que quedan en la carpeta de origen ya no se cargan como fotogramas. También se rehicieron etiquetas, barras de escala, comparación de métodos, exportación de profundidad/modelo, actualizaciones y retoque según la auditoría de `docs/ux-specs`.

## Descarga e instalación

Descarga una compilación desde [Releases](https://github.com/Xinzhe99/OpenFocus/releases):

| Plataforma | Archivo | Tipo |
|------------|---------|------|
| Windows 10/11 (64 bits) | `OpenFocus-*-setup.exe` | **Instalador** (recomendado) |
| Windows 10/11 (64 bits) | `OpenFocus-*-windows-x64.zip` | Portable (descomprimir y ejecutar) |
| macOS Apple Silicon (M1–M4) | `OpenFocus-*-macos.dmg` | Imagen de disco (arrastrar a Aplicaciones) |
| macOS Apple Silicon (M1–M4) | `OpenFocus-*-macos-arm64.zip` | Portable (descomprimir) |

La aplicación no está firmada, así que el primer inicio puede mostrar un aviso: en Windows, «Más información → Ejecutar de todos modos»; en macOS, clic derecho → Abrir, o `xattr -cr /Applications/OpenFocus.app`.

## Características principales

- **Ocho métodos de fusión** — Guided Filter, DCT, DTCWT, GFG-FGF, StackMFF-V3 (red neuronal), OpenFocus A/B/C.
- **Tres modos de registro** — transformación de similitud rápida, ECC y homografía; también se puede fusionar sin registro.
- **Flujo de trabajo de focus stacking** — explorador de pila, comparación A/B, mapas de profundidad, comparación de métodos, procesamiento por lotes y vista previa rápida.
- **Retoque con procedencia** — toma píxeles de los fotogramas que produjeron el resultado, con zoom/posición compartidos, control deslizante de origen, pincel suave, pincel de restauración, deshacer/rehacer por trazo y protección de salidas sin exportar.
- **Listo para microscopía y publicación** — 16 bits de extremo a extremo, µm/px desde TIFF, corrección de campo plano, barras de escala, Z-stack TIFF multipágina y exportación OBJ con textura.
- **CLI sin interfaz** — el mismo pipeline que la GUI, con lotes, `--dry-run`, `--resume` y salida JSON.
- **Doce idiomas de interfaz** — English, 简体中文, 日本語, Español, Deutsch, Français, Русский, 한국어, Português, Italiano, Nederlands, Polski.

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
