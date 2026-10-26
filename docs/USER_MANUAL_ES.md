# Manual de usuario de OpenFocus

## Índice

1. [Introducción](#1-introduction)
2. [Primeros pasos](#2-getting-started)
3. [Vista general de la interfaz principal](#3-main-interface-overview)
4. [Importar imágenes y vídeo](#4-importing-images-and-video)
5. [Ver y navegar por las imágenes](#5-viewing-and-navigating-images)
6. [Vista de comparación Wipe](#6-wipe-compare-view)
7. [Registro de imágenes](#7-image-registration)
8. [Métodos de fusión de imágenes](#8-image-fusion-methods)
9. [Transformaciones de imagen](#9-image-transformations)
10. [Retoque, etiquetas y anotaciones](#10-retouching-labels-and-annotations)
11. [Ajustes y configuración](#11-settings-and-configuration)
12. [Procesamiento por lotes](#12-batch-processing)
13. [Exportar resultados](#13-exporting-results)
14. [Interfaz de línea de comandos](#14-command-line-interface)
15. [Atajos de teclado](#15-keyboard-shortcuts)
16. [Solución de problemas](#16-troubleshooting)

---

## 1. Introducción

OpenFocus es una aplicación de escritorio profesional de fusión de imágenes multifoco, pensada para investigadores, ingenieros y profesionales del procesamiento de imágenes. El programa procesa secuencias de imágenes con distintos planos de enfoque mediante algoritmos avanzados de registro y fusión, y genera imágenes completamente nítidas con profundidad de campo total.

### Funciones principales

- **Fusión de imágenes multifoco**: combina varias imágenes con puntos de enfoque distintos en una sola imagen totalmente enfocada
- **Varios algoritmos de fusión**: elige entre Filtro guiado, DCT, DTCWT, GFG-FGF y fusión por IA (aprendizaje profundo)
- **Registro de imágenes**: alinea secuencias desalineadas con los métodos ECC u Homografía
- **Vista de comparación Wipe**: comprueba la alineación y compara resultados con una vista dividida A/B
- **Archivos de proyecto**: guarda y restaura sesiones de trabajo completas
- **TIFF multipágina**: las pilas Z de un solo archivo se cargan página a página y se exportan de vuelta a un único archivo
- **Mapa de profundidad**: mapas de posición de enfoque por píxel, con esquemas de pseudocolor personalizados
- **Barra de escala**: barras de escala con calidad de publicación y calibración automática en µm/px
- **Recuperación automática tras fallos**: tras un cierre inesperado se ofrece restaurar la última sesión de trabajo
- **Multilenguaje**: English, 中文, 日本語, Español, con detección automática del idioma del sistema
- **CLI sin interfaz**: automatiza OpenFocus desde la línea de comandos para scripts e integración continua
- **Exportación flexible**: guarda los resultados como imágenes individuales, carpetas o animaciones GIF
- **Transformaciones de imagen**: rota, voltea y redimensiona pilas de imágenes
- **Adición de etiquetas**: añade etiquetas de texto a las pilas registrada y de entrada
- **Ajustes persistentes**: tus preferencias, el idioma y las pilas abiertas recientemente se conservan entre sesiones

### Formatos de entrada compatibles

- Archivos de imagen: JPG/JPEG, PNG, BMP, TIFF, WebP y otros formatos habituales
- Archivos de vídeo: formato MP4 (se decodifica automáticamente en secuencias de imágenes)
- Carpetas con secuencias de imágenes numeradas
- Archivos RAW de cámara (DNG, CR2/CR3, NEF, ARW, RAF, ORF, RW2 y otros
  formatos RAW habituales). La decodificación usa `rawpy` (LibRaw), que
  `pip install openfocus` instala y las compilaciones para Windows/macOS
  incluyen

### Formatos de salida compatibles

- Imágenes individuales: JPG, PNG, BMP, TIFF, WebP
- Secuencias de imágenes: guardadas como carpetas
- Animaciones: formato GIF

---

## 2. Primeros pasos

### Iniciar la aplicación

Después de la instalación, abre OpenFocus de una de estas formas:

1. **Windows (ejecutable)**: haz doble clic en el acceso directo o en el ejecutable de OpenFocus
2. **Código fuente de Python**: ejecuta `python main.py` en tu entorno conda activado

Al arrancar verás la ventana principal de la aplicación con una interfaz de tema oscuro.

### Flujo de inicio rápido

1. Importa tu secuencia de imágenes (Archivo → Abrir carpeta o arrastrar y soltar)
2. Configura los ajustes de fusión en el panel derecho
3. Si lo necesitas, activa el registro para alinear las imágenes
4. Ajusta el tamaño de núcleo si hace falta
5. *(opcional)* Marca **Vista previa rápida** para probar los parámetros primero sobre un borrador reducido
6. Pulsa «Iniciar renderizado» para empezar el procesamiento
7. Exporta los resultados

---

## 3. Vista general de la interfaz principal

La interfaz de OpenFocus consta de tres zonas principales:

### Panel izquierdo: área de visualización

La parte izquierda de la ventana muestra tus imágenes en una vista dividida:

- **Panel de imagen de origen (arriba)**: muestra la secuencia de imágenes de entrada
- **Panel de imagen de resultado (abajo)**: muestra los resultados de fusión o registro
- **Deslizador de navegación**: usa el deslizador bajo cada panel para recorrer los fotogramas
- **Etiquetas informativas**: muestran la posición del fotograma actual y las dimensiones de la imagen

Puedes cambiar el tamaño de los paneles arrastrando el separador que hay entre ellos.

### Panel derecho: panel de control

El panel derecho reúne todas las opciones de configuración, organizadas de arriba abajo:

1. **Ajustes de fusión**: selecciona el algoritmo de fusión y el tamaño de núcleo
2. **Ajustes de registro**: activa o desactiva los métodos de alineación
3. **Botones de acción**: Restablecer y Iniciar renderizado
4. **Lista de imágenes de origen**: muestra todas las imágenes cargadas con sus nombres de archivo
5. **Lista de salida**: muestra los resultados generados

### Barra de menús

La barra de menús superior da acceso a todas las funciones:

- **Archivo**: operaciones de importación y exportación, y la lista de archivos recientes
- **Editar**: transformaciones de imagen y etiquetas
- **Herramientas**: procesamiento por lotes
- **Ajustes**: aceleración por GPU, idioma, mosaicos, registro y configuración de hilos
- **Ayuda**: información del entorno, datos de contacto, búsqueda de actualizaciones y la carpeta de registros

---

## 4. Importar imágenes y vídeo

### Abrir una carpeta de imágenes

1. Ve a **Archivo → Abrir carpeta** o pulsa `Ctrl+O`
2. Selecciona una carpeta que contenga tu secuencia de imágenes
3. Aparece un diálogo de escala de carga: elige la escala (100 % = resolución completa)
4. Pulsa OK para cargar las imágenes

Marca **No volver a preguntar** en ese diálogo para cargar siempre con la escala
elegida sin preguntar; **Ajustes → Preguntar la escala de carga al importar**
reactiva el aviso. Una pila ya cargada se puede reducir más adelante con
**Editar → Redimensionar**.

El programa hará lo siguiente automáticamente:
- Detectar y ordenar los archivos de imagen por nombre
- Mostrar la primera imagen en el panel de origen
- Rellenar la lista de imágenes de origen con todos los fotogramas

### Abrir archivos de vídeo

1. Ve a **Archivo → Abrir vídeo** o pulsa `Ctrl+Shift+O`
2. Selecciona un archivo de vídeo (MP4, AVI, MOV, MKV, WMV, FLV, WEBM)
3. El vídeo se decodifica automáticamente en fotogramas individuales
4. Cada fotograma pasa a formar parte de la pila de imágenes

### Archivos de proyecto

Archivo → Guardar proyecto guarda toda tu sesión de trabajo en un archivo
`.ofproj`: la lista exacta de archivos de origen (en orden), la escala de
carga, el método de fusión, las opciones de registro, el tamaño de núcleo,
los ajustes de mosaicos, hilos y registro, la configuración de etiquetas y el
fotograma actual. Archivo → Abrir proyecto reconstruye esa sesión tal cual
(las imágenes de origen se referencian, no se copian: no las muevas de sitio). El título de la ventana muestra el proyecto abierto.

### Pila de ejemplo

El diálogo de bienvenida ofrece **Cargar pila de ejemplo**: seis fotos de
muestra incluidas, procedentes de una pila de enfoque real, para que puedas
probar el flujo de trabajo completo de inmediato sin necesidad de tener
imágenes propias. También resulta útil para comprobar una instalación.

### Curva de nitidez

Una pequeña curva sobre el deslizador de navegación de origen muestra la
nitidez de cada fotograma (normalizada). Los fotogramas desenfocados
aparecen como valles marcados con un punto rojo. Haz clic en cualquier punto
de la curva para saltar a ese fotograma; al pasar el cursor se muestra el
número de fotograma.

### Pilas TIFF multipágina

El software de microscopía (ZEN, NIS-Elements, ImageJ/Fiji) suele exportar
toda una pila Z como UN único archivo TIFF multipágina. OpenFocus lee cada
página de ese archivo como un fotograma individual de la pila de imágenes:
suelta o abre el archivo y la pila aparecerá completa (los fotogramas se
nombran `<file>_page_001`, etc.). Los TIFF de una sola página se comportan
exactamente como antes. También funciona en sentido inverso:
**Archivo → Guardar pila → (Pila registrada / Pila de entrada) → Guardar como TIFF multipágina**
escribe toda la pila en un único `.tif`, conservando la profundidad de 16 bits.

### Barra de escala (figuras para publicación)

Ajustes → Barra de escala… configura una barra de escala incrustada en las
imágenes exportadas. El tamaño de píxel (µm/px) se detecta automáticamente
en los metadatos TIFF (la convención `unit=micron` de ImageJ, `PhysicalSizeX`
de OME-TIFF o una unidad de resolución explícita en centímetros), de modo que
las pilas de microscopía (incluidas las TIFF multipágina) se calibran solas.
La resolución de impresora (72/300 dpi y similares) *no* se toma
deliberadamente como calibración, y un valor manual que introduzcas siempre
prevalece sobre los metadatos detectados: una barra equivocada pero creíble
en una figura es peor que no poner ninguna. Elige la esquina y el color
(Auto invierte el color según el fondo); el diálogo muestra una vista previa
en vivo. Con la barra activada, todas las exportaciones (imagen individual,
pilas en carpetas, TIFF multipágina, GIF, exportar todo, lotes) incluyen una
barra con estilo de publicación: una longitud redondeada de la serie 1-2-5
(~12 % del ancho de la imagen) con texto perfilado y legible. La CLI lo
admite mediante `--scale-bar` (calibración automática) y
`--scale-um <um/px>`.

### Mapa de profundidad (posición de enfoque)

**Herramientas → Mapa de profundidad…** estima posiciones de enfoque mediante la medida
de actividad del método elegido (DCT, DTCWT, GFG-FGF, StackMFF V3, OpenFocus B/C).
No es una superficie medida ni un mapa exacto del origen de los píxeles fusionados.
Elija los fotogramas marcados actuales o el origen registrado de la salida seleccionada;
si faltan datos de procedencia, no se sustituyen por la pila actual. Cálculo, textura
y exportación permanecen vinculados a la misma instantánea del origen.

Los seis métodos requieren pulsar **Calcular** expresamente. Abrir el diálogo o cambiar
la medida, el origen, el registro o los parámetros no inicia el cálculo. El editor aparece
justo debajo del selector de medida: los núcleos de mediana DCT e inicial GFG aceptan
impares de 1 a 51 (7 por defecto); B usa una ventana de varianza y C una de energía de paso
de banda, impares de 1 a 21 px (11 por defecto). Los niveles de descomposición DTCWT
se editan aquí como borrador local, inicializado con los parámetros del origen. StackMFF V3
muestra su propio modo de salida de IA (mezcla suave por defecto o selección dura) y una
explicación, sin núcleo numérico: la mezcla suave estima una posición continua ponderada
por probabilidad; la dura toma el fotograma más probable por píxel. Ninguna es una
distancia física.
Los borradores se inicializan con los parámetros de la salida coincidente o los valores
recordados por método en la ventana principal. Se conservan por origen y método mientras
el diálogo está abierto, sin modificar la ventana principal, el proyecto ni la salida.

Cambiar ajustes de cálculo marca el mapa 2D anterior como **no actualizado**, vacía la
superficie 3D y bloquea todas las exportaciones, incluidas las de datos, profundidad física
y OBJ, hasta que un nuevo cálculo termine correctamente. El trabajo activo se cancela de
forma cooperativa. Después hay que pulsar **Calcular** de nuevo; no se encola ni se reinicia
ningún cálculo. Calcular permanece visible y se desactiva durante el cálculo, la cancelación
y la exportación del modelo.

- **Mapa de color**: escala de grises (el estándar de publicación), 15
  paletas científicas (Turbo, Viridis, Magma, …) y las tuyas propias:
  *Nuevo esquema personalizado…* genera un degradado a partir de cualquier
  número de paradas de color (haz doble clic en una parada para editar su
  posición y color); los esquemas personalizados se guardan y aparecen en la
  lista desplegable.
- **Invertir / Gamma / Suavizado**: invierte el eje de profundidad,
  reequilibra el tono (gamma < 1 aclara los índices bajos) y elimina las
  motas con una pasada que respeta los bordes.
- **Superposición**: mezcla el mapa 2D sobre la salida asociada con opacidad ajustable,
  solo cuando coinciden sus dominios de coordenadas.
- **Exportar**: *Guardar en color…* ofrece PNG/JPEG/TIFF/BMP. El TIFF de datos de 16 bits
  codifica índices de enfoque normalizados de 0 a 65535; no son valores flotantes sin
  pérdida ni distancias físicas. JPEG/BMP se rechazan para estos datos.
- **Modelo 3D / Textura 3D**: el lado mayor de la malla tiene 600 vértices por defecto.
  Cada diálogo nuevo empieza en **Automática (color real)**: usa la salida asociada solo
  si su identidad y dominio de coordenadas/ROI coinciden con el mapa terminado. En caso
  contrario usa el primer fotograma participante del cálculo completado, tras el registro
  y en el orden real utilizado. Se muestra el nombre de la salida o del fotograma empleado.
  También puede elegir la salida coincidente, un fotograma participante o **Pseudocolor de
  profundidad**; si no hay textura disponible, se indica. La vista 3D y el PNG de textura
  del OBJ comparten la misma imagen. El mapa 2D conserva su propia paleta y superposición;
  sus cambios de paleta o Gamma no recolorean fotos. Solo el pseudocolor explícito sigue
  la paleta, Gamma e inversión. La vista en reposo y la exportación comparten el mismo
  campo de alturas muestreado, sin suavizado oculto adicional. El relieve relativo se
  representa en unidades de imagen.

El cálculo se ejecuta en segundo plano con progreso y opción de cancelación;
el método de IA reutiliza el mismo procesamiento por mosaicos seguro para la
memoria que la fusión por IA, por lo que las pilas profundas funcionan. El
equivalente en la CLI es `--depth-map [path]` junto con `--depth-colormap`.

### Recuperación automática tras fallos

OpenFocus guarda continuamente una instantánea de tu sesión de trabajo (pila,
ajustes, etiquetas, fotograma actual). Si la aplicación se cierra de forma
inesperada (un fallo, un corte de corriente o un cierre forzado), el
siguiente arranque lo detecta y ofrece **Restaurar sesión** para volver
exactamente a donde estabas. Si el cierre fue limpio, el aviso no aparece
nunca. Los datos de recuperación se guardan junto a tus ajustes (las
instalaciones portables los mantienen junto al ejecutable, así que viajan
contigo).

### Archivos recientes

Cada pila que abres queda registrada en **Archivo → Abrir recientes** (hasta 8 entradas). Pulsa `Alt+1`–`Alt+8` para reabrir una entrada directamente. Al reabrir la misma pila, esta pasa a la primera posición; las entradas cuyos archivos ya no existen se pueden eliminar mediante un aviso, y **Vaciar lista de recientes** deja el menú limpio.

### Arrastrar y soltar

También puedes arrastrar una carpeta directamente sobre la ventana de la aplicación para importar imágenes. Si sueltas más archivos sobre una pila existente, se añaden al final (tras comprobar el tamaño).

### Orientación EXIF

Los archivos JPEG/TIFF de cámara con etiqueta EXIF Orientation se rotan o
voltean automáticamente durante la importación, de modo que las fotos en
vertical se cargan derechas.

### Orígenes de 16 bits

Las entradas PNG/TIFF de 16 bits se detectan automáticamente y se conservan de
extremo a extremo: la fusión se ejecuta a profundidad completa y los
resultados pueden guardarse de nuevo como PNG/TIFF de 16 bits (las
exportaciones JPG/BMP se convierten a 8 bits, ya que esos formatos no pueden
almacenar 16 bits).

Los archivos RAW de cámara se revelan a 16 bits (balance de blancos de la
cámara, gamma sRGB) con el aclarado automático desactivado, para que cada
fotograma de la pila reciba la misma curva de exposición; una exposición
automática por fotograma haría parpadear el brillo entre fotogramas. Guarda
como TIFF o PNG para conservar los datos de 16 bits.

### Vaciar la pila

Para eliminar todas las imágenes cargadas, ve a **Archivo → Vaciar pila** o pulsa `Ctrl+W`.

---

## 5. Ver y navegar por las imágenes

### Navegar por la pila de imágenes

- **Deslizador**: arrastra el deslizador bajo el panel de origen para recorrer los fotogramas
- **Lista de archivos**: haz clic en cualquier nombre de archivo de la lista de imágenes de origen del panel derecho
- **Teclado**: usa las flechas (izquierda/derecha) para navegar

### Zoom y desplazamiento

- **Acercar/Alejar**: rueda del ratón sobre la imagen
- **Zoom fino**: mantén `Ctrl` y usa la rueda del ratón para incrementos más pequeños
- **Ajustar a la ventana**: haz doble clic en la imagen para alternar entre ajustar y la vista al 100 %
- **Desplazar**: haz clic y arrastra para mover la imagen dentro del panel

### Navegación sincronizada

Cuando navegas en el panel de origen, el panel de resultados muestra el fotograma de resultado correspondiente, si existe.

---

## 6. Vista de comparación Wipe

La vista Wipe superpone dos imágenes en un mismo marco con un divisor vertical arrastrable, lo que hace visibles directamente las desalineaciones y las diferencias de calidad a cada lado de la división.

### Abrir la vista Wipe

1. Renderiza al menos un resultado (fusión o registro)
2. Pulsa el conmutador **Wipe** en la esquina superior derecha de la barra de título del panel de resultados
3. El área de resultados pasa a la vista de comparación; pulsa **Wipe** otra vez para volver

### Interpretar la vista

- **Lado A (a la izquierda del divisor)**: un fotograma de origen; por defecto sigue al deslizador del panel de origen
- **Lado B (a la derecha del divisor)**: el renderizado más reciente
- **Divisor blanco + tirador redondo**: arrastra horizontalmente para mover la división
- Arrastrar los bordes que cruzan el divisor revela errores de registro difíciles de ver en paneles lado a lado

### Elegir qué comparar

La barra inferior de la vista de comparación tiene un grupo por lado:

- **Lado A**: marca **Seguir origen** (por defecto) para seguir al deslizador del panel de origen, o desmárcalo para fijar el lado A a cualquier fotograma con su propio deslizador
- **Lado B**: marca **Último resultado** (por defecto) para mostrar siempre el renderizado más reciente, o desmárcalo para recorrer tu historial de resultados con su deslizador

Ambos lados se actualizan en vivo: los nuevos renderizados, los movimientos del deslizador de origen y la eliminación de fotogramas se reflejan al instante.

### Zoom y desplazamiento

La vista de comparación comparte una única transformación para ambas imágenes:

- **Zoom**: rueda del ratón (hace zoom alrededor del cursor)
- **Desplazar**: haz clic y arrastra en cualquier punto excepto el divisor
- **Restablecer**: doble clic

El modo Wipe y el modo ROI comparten el panel de resultados, así que al entrar en uno se sale del otro. Vaciar la pila también sale del modo Wipe.

---

## 7. Registro de imágenes

El registro de imágenes corrige la desalineación espacial entre los fotogramas de tu pila. Resulta esencial cuando las imágenes presentan pequeños desplazamientos o cambios de perspectiva.

### Métodos de registro

#### ECC (Enhanced Correlation Coefficient)

- Usa optimización para encontrar la mejor alineación basándose en la correlación
- Adecuado para desalineaciones sutiles
- Ofrece precisión subpíxel
- Más lento que la homografía, pero más preciso

#### Homografía

- Se basa en la correspondencia de puntos característicos (características SIFT/ORB)
- Maneja transformaciones geométricas mayores
- Más rápida que ECC para una alineación inicial
- Buena para imágenes con cambios de perspectiva importantes

#### OpenFocus (rápida)

- Alineación por similitud acotada (traslación + rotación + escala),
  con ecualización de brillo por fotograma y recorte automático
- Menos de un segundo por fotograma: mucho más rápida que ECC u Homografía
- Los fotogramas con un desplazamiento superior al **8 % del tamaño del
  fotograma se dejan sin alinear** en lugar de corregirlos parcialmente. Puedes
  cambiar el límite con `--align-shift-limit` en la CLI
- Para geometrías no rígidas o extremas, elige mejor ECC u Homografía

### Activar el registro

1. En el panel derecho, localiza el grupo **Registro**
2. Elige un método en la lista desplegable: **Sin alinear**, **OpenFocus (rápida)**,
   **ECC** u **Homografía**
3. La alineación rápida de OpenFocus también se puede encadenar delante de ECC
   u Homografía (primero una pasada global gruesa y después la pasada local precisa)

### Cuándo usar el registro

- Tus imágenes presentan una desalineación visible
- Las imágenes se tomaron a pulso
- La escena tiene paralaje (los objetos cercanos se desplazan entre fotogramas)
- Los resultados muestran artefactos de fantasma o duplicación

### Ajustes de registro

Accede a los ajustes adicionales en **Ajustes → Ajustes de registro**:
- **Ancho de reducción**: controla la resolución de preprocesamiento (1024 px por defecto)
- Valores menores = procesamiento más rápido, posiblemente menos precisión
- Valores mayores = procesamiento más lento, posiblemente más precisión

---

## 8. Métodos de fusión de imágenes

OpenFocus ofrece ocho algoritmos de fusión (además de un modo de solo alineación). Cada uno tiene características distintas que lo hacen adecuado para diferentes tipos de imágenes.

### Filtro guiado (predeterminado)

- **Algoritmo**: usa filtrado guiado para una fusión que preserva los bordes
- **Ideal para**: uso general, equilibrio entre velocidad y calidad
- **Ventajas**: buena preservación de bordes, procesamiento rápido
- **Parámetro**: tamaño de núcleo (ajustable de 1 a 51, solo valores impares)
  - Núcleo menor: más detalle local, posiblemente más ruido
  - Núcleo mayor: resultados más suaves, mejor supresión de ruido

### DCT (transformada discreta del coseno)

- **Algoritmo**: fusión en el dominio de la frecuencia mediante coeficientes DCT
- **Ideal para**: imágenes con mucha textura, imagen científica
- **Ventajas**: conserva texturas finas, base matemática sólida
- **Parámetro**: el tamaño de núcleo controla la ventana de procesamiento

### DTCWT (transformada wavelet compleja de árbol dual)

- **Algoritmo**: fusión en el dominio wavelet multiescala
- **Ideal para**: escenas complejas con varios niveles de detalle
- **Ventajas**: excelente análisis multiescala, buena direccionalidad
- **Regla de fusión**: una sola decisión por píxel y nivel para todas las
  orientaciones y canales de color (N vías, con comprobación de consistencia
  mayoritaria); la banda de paso bajo sigue la decisión más gruesa en lugar de
  promediar todos los fotogramas
- **Parámetro**: niveles de ondícula (1–6, 4 por defecto) en el panel derecho.
  El método no usa núcleo espacial, así que el deslizador de núcleo se oculta.

### GFG-FGF (gradiente de cuatro vecinos tipo gaussiano + filtro guiado rápido)

- **Algoritmo**: implementación de Fu et al. (*Laser & Optoelectronics
  Progress*, 2024): un mapa de enfoque por diferencia de desenfoque
  |I − mean(I)| se enriquece con el operador de gradiente de cuatro vecinos
  tipo gaussiano (GFG), se refina con un filtro guiado rápido, se decide por
  píxel (los empates se resuelven con una medida de región pequeña de Sobel
  3×3) y los mapas de decisión binarios se suavizan de nuevo con el filtro
  guiado rápido antes de la mezcla ponderada.
- **Ideal para**: microscopía e imágenes con regiones de enfoque claras
- **Ventajas**: rápido, límites de región de enfoque nítidos; la memoria no
  crece con el número de fotogramas
- **Parámetro**: Núcleo = tamaño del filtro de media del mapa de enfoque grueso

### StackMFF-V3 (aprendizaje profundo)

- **Algoritmo**: clasificación de origen píxel a píxel (requiere PyTorch). La red
  predice, para cada píxel, de qué imagen de entrada proviene, y después mezcla
  las imágenes según esas probabilidades o elige una sola imagen
- **Salida de IA** (panel derecho, solo para este método):
  - **Mezcla suave (ponderada)**: opción predeterminada; cada píxel es una
    mezcla de las imágenes ponderada por probabilidad. Transiciones suaves y
    sin costuras duras; coincide con la configuración publicada en el artículo
  - **Selección dura (una imagen)**: cada píxel se copia de la imagen más
    probable. Selección más nítida y sin interpolación entre imágenes; puede
    aparecer una costura a nivel de píxel donde dos imágenes empatan
- **Ideal para**: máxima calidad cuando hay GPU disponible
- **Ventajas**: invariante al orden (el orden de los fotogramas nunca cambia el
  resultado), admite pilas de cualquier longitud, sin ajustes por método,
  optimización automática
- **Requisitos**: PyTorch instalado, GPU CUDA opcional (sin torchvision)
- **Nota**: puede no estar disponible si PyTorch no está instalado

### OpenFocus A / B / C

- **Algoritmo**: media ponderada por varianza, selección por mapa de
  profundidad de varianza y fusión por pirámide laplaciana
- **OpenFocus A**: media ponderada por varianza con prior del último fotograma;
  rápido y estable en color en pilas cortas
- **OpenFocus B**: selección por mapa de profundidad de varianza; elige el
  fotograma más nítido por píxel; ideal para superficies lisas y microscopía
- **OpenFocus C**: mezcla por banda con pirámide laplaciana; aspecto natural en
  sujetos texturados y realza el contraste local
- **Parámetros**: A y B usan una ventana de varianza (deslizador de núcleo); C
  no tiene radio. Los tres usan una sigma de suavizado.

**Suavizado**: un único campo numérico bajo el deslizador de núcleo, cuyo
significado depende del método seleccionado (rango 0–8, paso 0,2):

| Método | Etiqueta | Predeterminado | Efecto |
|---|---|---|---|
| OpenFocus A | Suavizado de mezcla (σ) | 2,0 | Suavizado gaussiano del mapa de pesos por fotograma: con qué amplitud se mezclan las regiones nítidas de fotogramas vecinos |
| OpenFocus B | Suavizado de profundidad (σ) | 1,2 | Suavizado gaussiano del mapa de profundidad: anchura de la transición entre fotogramas |
| OpenFocus C | Suavizado de mezcla (σ) | 3,0 | Suavizado gaussiano del mapa de pesos de cada banda de la pirámide |

0 conserva la elección por píxel más nítida (máximo detalle, pero con motas en
pilas ruidosas); los valores mayores suavizan transiciones y halos. Cada método
recuerda su propio valor entre sesiones, y el procesamiento por lotes aplica la
misma regla.

C no tiene Radio: el método C de Helicon solo expone Suavizado, así que el
deslizador de núcleo se oculta para C en lugar de no hacer nada en silencio.

CLI: `--depth-smoothing SIGMA` (B) y `--blend-smoothing SIGMA` (A, C).

**Las opciones de brillo son por método.** *Igualar exposición antes de
fusionar* y *Eliminar deriva de brillo* se recuerdan por separado para cada
método de fusión, porque responden a una cuestión de cada pila y no a una
preferencia global. Ambas están activadas por defecto en OpenFocus A/B/C (que
no tienen calibración de exposición integrada); los demás métodos mantienen
los valores históricos (igualado de exposición activado, eliminación de deriva
desactivada). Al cambiar de método se muestran los valores propios de ese método.

### Solo alineación (sin fusión)

Selecciona **Solo alineación (sin fusión)** como método para registrar la pila
y enviar los fotogramas alineados a la lista de salida sin fusionarlos; resulta
útil para inspeccionar la calidad del registro o exportar los fotogramas
alineados.

### Seleccionar un método de fusión

1. En el panel derecho, localiza el grupo **Fusión**
2. Elige un método en la **lista desplegable** (siempre hay exactamente uno
   seleccionado; la primera entrada registra sin fusionar)
3. Ajusta el tamaño de núcleo si corresponde: la etiqueta indica qué significa
   el núcleo para el método seleccionado
4. Pulsa **Iniciar renderizado** para procesar

### Guía de comparación de algoritmos

| Algoritmo | Velocidad | Calidad | Ideal para |
|-----------|-------|---------|----------|
| Filtro guiado | Rápida | Buena | Uso general |
| DCT | Media | Buena | Texturas |
| DTCWT | Media | Muy buena | Escenas complejas |
| GFG-FGF | Rápida | Buena | Regiones de enfoque |
| Fusión por IA (StackMFF-V3) | Lenta (GPU) | Excelente | Máxima calidad |
| OpenFocus A | Muy rápida | Muy buena | Pilas cortas, estabilidad de color |
| OpenFocus B | Muy rápida | Muy buena | Superficies lisas, microscopía |
| OpenFocus C | Rápida | Muy buena | Sujetos texturados, aspecto natural |

### Vista previa rápida (renderizado de borrador)

Ajustar parámetros en pilas grandes es lento a resolución completa. Marca
**Vista previa rápida** encima del botón de renderizado y OpenFocus renderizará
primero un borrador reducido (lado más largo de 1200 px), normalmente varias
vez más rápido, incluido el registro. La vista previa se indica con una nota
«vista previa» en el resumen de la barra de estado tras el renderizado y **no** se
añade al historial de salida. Desmarca la casilla y vuelve a renderizar para
obtener el resultado a resolución completa. El modo de vista previa no se
aplica a los renderizados ROI y nunca contamina la caché de alineación a
resolución completa.

---
## 9. Transformaciones de imagen

### Rotación

1. Ve a **Editar → Rotar**
2. Elige el tipo de rotación:
   - **90° horario**: gira todas las imágenes 90 grados a la derecha
   - **90° antihorario**: gira todas las imágenes 90 grados a la izquierda
   - **180°**: voltea las imágenes boca abajo

### Volteo

1. Ve a **Editar → Voltear**
2. Elige el tipo de volteo:
   - **Volteo horizontal**: refleja de izquierda a derecha
   - **Volteo vertical**: refleja de arriba abajo

### Redimensionar

1. Ve a **Editar → Redimensionar**
2. Aparecerá un diálogo con las opciones de redimensionado
3. Introduce las nuevas dimensiones o el porcentaje de escala
4. Todas las imágenes de la pila se redimensionarán proporcionalmente

### Orden de las transformaciones

Las transformaciones se aplican a toda la pila de imágenes, manteniendo la alineación entre fotogramas.

---

## 10. Retoque, etiquetas y anotaciones

### Retocar el resultado (Editar → Retocar resultado…, `Ctrl+T`)

La fusión puede dejar todavía algún defecto local: un halo alrededor de un borde de alto contraste, una cerda tomada del fotograma equivocado, ruido en una zona plana. El pincel de retoque copia píxeles de un fotograma de origen sobre el resultado:

1. Renderiza un resultado (o selecciona uno en la lista de salida; también
   funciona con clic derecho → *Retocar resultado…*) y pulsa `Ctrl+T`.
2. Pinta sobre el defecto con el botón izquierdo. Con **Elegir automáticamente
   el fotograma más nítido por trazo** activado (opción predeterminada), cada
   trazo selecciona primero el fotograma con más detalle bajo el pincel;
   corregir un halo suele ser un solo arrastre. Desactívalo para elegir el
   fotograma tú mismo con el deslizador o con `,` / `.`.
3. Ajusta el **radio** (`[` / `]`), la **dureza** (los bordes suaves ocultan la
   costura) y la **opacidad**. Las pasadas que se solapan dentro de un mismo
   trazo nunca se acumulan.
4. **Mostrar fotograma de origen** (`Tab`) muestra una vista previa del
   fotograma desde el que pintas. La rueda hace zoom, arrastrar con el botón
   derecho o central desplaza y `F` ajusta la imagen.
5. `Ctrl+Z` / `Ctrl+Y` deshacen y rehacen trazos completos. **Aplicar como
   nueva salida** añade la imagen retocada a la lista de salida (con el sufijo
   `_Retouched`); la salida original queda intacta.

El pincel usa los fotogramas alineados que produjeron el resultado, así que
encaja con él; los resultados de 16 bits siguen siendo de 16 bits. Los
renderizados ROI y las pilas redimensionadas no se pueden retocar (sus
fotogramas ya no coinciden con la geometría del resultado).

### Añadir etiquetas

1. Asegúrate de tener resultados renderizados o imágenes cargadas
2. Ve a **Editar → Añadir etiqueta**
3. Introduce el texto de la etiqueta en el diálogo
4. Las etiquetas se añaden tanto a la pila registrada como a la pila de entrada

### Eliminar etiquetas

- **Eliminar etiquetas de la pila registrada**: quita las etiquetas de las imágenes alineadas
- **Eliminar etiquetas de la pila de entrada**: quita las etiquetas de las imágenes originales

Accede a estas opciones desde el submenú **Editar → Eliminar etiqueta**.

---

## 11. Ajustes y configuración

### Ajustes de mosaicos (optimización de memoria)

Para imágenes grandes, OpenFocus procesa por mosaicos para evitar problemas de memoria.

Accede en **Ajustes → Ajustes de mosaicos**:

- **Tamaño de mosaico**: tamaño de cada bloque de procesamiento (1024 px por defecto)
- **Solape de mosaico**: solape entre mosaicos para una mezcla sin costuras (256 px por defecto)
- **Umbral de mosaico**: tamaño de imagen a partir del cual se usan mosaicos (2048 px por defecto)

**Recomendaciones**:
- Mosaicos más pequeños: menos uso de memoria, procesamiento más lento
- Mosaicos más grandes: más uso de memoria, procesamiento más rápido
- Aumenta el solape si ves costuras visibles en los resultados

### Ajustes de registro

Accede en **Ajustes → Ajustes de registro**:

- **Ancho de reducción**: resolución de preprocesamiento para el registro
  - Predeterminado: 1024 px
  - Menor para ganar velocidad, mayor para ganar precisión

### Referencia de rendimiento

Consulta [PERFORMANCE.md](./PERFORMANCE.md) para ver cifras medidas de
rendimiento y memoria con pilas reales grandes, así como recomendaciones
prácticas para pilas profundas y entradas de alta resolución.

### Ajustes de hilos

Controla el procesamiento paralelo de la CPU:

Accede en **Ajustes → Ajustes de hilos**:

- `0` selecciona la planificación automática y es el valor inicial para ajustes nuevos.
- Los valores positivos limitan los hilos; se conservan los límites guardados.
- La capacidad de CPU, el modo de recursos y la memoria disponible pueden reducir la concurrencia.

### Activación de la aceleración por GPU

OpenFocus usa la GPU (CUDA/MPS) solo para el modelo neuronal de IA (StackMFF-V3); los algoritmos clásicos siempre se ejecutan en CPU.

- Actívala o desactívala en **Ajustes → Aceleración por GPU**
- Al desmarcarla, todo se fuerza a la CPU; resulta útil cuando los controladores de la GPU dan problemas
- El panel de estado muestra `CPU*` mientras la GPU está desactivada
- La elección se aplica en el siguiente renderizado y se recuerda

### Tamaño de lote de IA

Accede en **Ajustes → Tamaño de lote de IA**:

- El límite de lote guardado es 2 por defecto; actualmente se procesa un mosaico por llamada en todos los dispositivos.
- Aumentarlo no activa lotes paralelos en la GPU.

### Temas

Hay dos temas de interfaz disponibles: **Oscuro (predeterminado)** y **Claro**. Cambia en
**Ajustes → Tema**; la elección se aplica al instante (la barra de título de
Windows también cambia) y se recuerda. En el primer arranque, el usuario elige
un tema en el diálogo de bienvenida.

### Modo portable

Crea un archivo vacío llamado `OpenFocus.portable` junto a `OpenFocus.exe` (el
ZIP portable ya incluye uno) y todos los datos de usuario (ajustes, registros,
archivos recientes) se guardarán en una carpeta `OpenFocusData` junto a la
aplicación en lugar de tu perfil de usuario, de modo que la instalación puede
viajar en un lápiz USB. Borra el archivo marcador para volver al perfil normal.

### Metadatos EXIF

Los metadatos de origen (modelo de cámara, exposición, orientación, etc.) se
conservan: la orientación se aplica al cargar y las exportaciones PNG/TIFF/WebP
incluyen el EXIF original automáticamente. Las exportaciones JPG/BMP no pueden
almacenar el EXIF completo y se escriben sin él.

### Entrada HEIC/HEIF

Las fotos de iPhone se cargan directamente. Requiere el paquete opcional
`pillow-heif` (`pip install pillow-heif`; incluido en las compilaciones
empaquetadas).

### Actualizaciones

Los archivos de proyecto (`.ofproj`) se comportan como un documento: la barra
de título muestra el nombre del proyecto con un `*` mientras haya cambios sin
guardar, y al cerrar la aplicación o abrir otro proyecto se pregunta antes de
descartarlos. El proyecto guarda la lista de fotogramas, la escala de carga,
todas las opciones de renderizado, las etiquetas y el fotograma actual; las
imágenes en sí se referencian, no se copian.

Ayuda → Buscar actualizaciones compara tu versión con las versiones de GitHub.
**Descargar e instalar** obtiene el nuevo instalador dentro de la propia
aplicación (con progreso) y lo ejecuta; Abrir página de descargas lleva a la
lista de versiones. La búsqueda se ejecuta en silencio como máximo una vez al día.

### Caché de registro

OpenFocus puede guardar los resultados de registro en una carpeta
`.openfocus_cache` junto a la pila de origen. Al reabrir la misma pila (con los
archivos sin cambios) se omite por completo la alineación. Actívala o
desactívala en **Ajustes → Caché de registro**; las entradas obsoletas se
sustituyen automáticamente.

### Ajustes persistentes

Todo lo siguiente sobrevive automáticamente al reinicio de la aplicación:
número de hilos, parámetros de mosaicos, ancho de reducción del registro,
tamaño de lote de la fusión por IA (StackMFF-V3), la aceleración por GPU, la
caché de registro, el idioma de la interfaz, el tamaño de la ventana y la
distribución de paneles, el formato de exportación al arrastrar y tus pilas
abiertas recientemente. Los diálogos de ajustes guardan en el momento en que
confirmas; todo se guarda también al cerrar la aplicación. Las preferencias se
almacenan por usuario en `OpenFocus/OpenFocus.ini` (formato INI de QSettings).

### Idioma de la interfaz

Cámbialo en cualquier momento en **Ajustes → Idioma → English / 中文**. El cambio
se aplica al instante y se recuerda para futuras sesiones.

### Comprobación automática de actualizaciones

Una vez al día, unos segundos después de arrancar, OpenFocus consulta GitHub en
silencio en busca de una versión más reciente. Si existe, aparece un aviso
sobre los paneles de imagen con la versión, **Actualizar ahora** (descarga y
reinicia en un solo paso), **Novedades** (notas de la versión) y un botón para
descartarlo. Al descartarlo, el aviso se oculta solo para esa versión; la
siguiente versión lo vuelve a mostrar, y **Ayuda → Buscar actualizaciones** lo
reactiva de inmediato.

Mientras la descarga se ejecuta en segundo plano, la barra de estado muestra el
progreso. También puedes buscar actualizaciones manualmente cuando quieras en
**Ayuda → Buscar actualizaciones**.

---

## 12. Procesamiento por lotes

Procesa varias carpetas automáticamente:

1. Ve a **Herramientas → Procesamiento por lotes** o pulsa la opción de lote
2. En el diálogo de lotes:
   - **Añadir carpetas**: selecciona carpetas que contengan secuencias de imágenes
   - **Quitar seleccionadas**: elimina las carpetas seleccionadas de la lista
   - **Vaciar lista**: elimina todas las carpetas
3. Configura los ajustes de salida:
   - **Crear subcarpeta en la carpeta de origen**: crea los resultados dentro de cada carpeta de entrada
   - **Misma carpeta de origen**: sobrescribe o guarda junto a la entrada
   - **Especificar carpeta de salida**: indica la ubicación de salida
4. Revisa los ajustes actuales de fusión y registro
5. Pulsa **Iniciar procesamiento por lotes** para empezar

El diálogo de lotes muestra el progreso en tiempo real. Puedes cancelar el procesamiento en cualquier momento.

### Notas sobre el procesamiento por lotes

- Usa los mismos ajustes de fusión y registro que el procesamiento manual
- Cada carpeta se procesa de forma independiente
- Los nombres de archivo incluyen la marca de tiempo y los parámetros para su identificación

---

## 13. Exportar resultados

### Guardar un resultado individual

1. Selecciona en la lista de salida el resultado que quieres guardar
2. Ve a **Archivo → Guardar** o pulsa `Ctrl+S`
3. Elige el formato y la ubicación
4. Pulsa Guardar

### Formato de exportación al arrastrar

Si arrastras resultados desde la lista de salida, se escriben archivos de
imagen con el formato elegido en el menú contextual de la lista de salida
(**Formato al arrastrar**: JPG, PNG o TIFF). La elección se recuerda.

### Guardar la pila registrada

1. Ve a **Archivo → Guardar pila → Pila registrada**
2. Elige el formato de guardado:
   - **Guardar como carpeta**: guarda cada fotograma alineado como archivo de imagen independiente
   - **Guardar como GIF**: crea un GIF animado con la secuencia alineada

### Guardar la pila de entrada

1. Ve a **Archivo → Guardar pila → Pila de entrada**
2. Elige el formato de guardado:
   - **Guardar como carpeta**: guarda los fotogramas originales como imágenes independientes
   - **Guardar como GIF**: crea un GIF animado con la secuencia de entrada

### Formatos de exportación

- **JPG**: con compresión, buena relación calidad/tamaño
- **PNG**: sin pérdidas, admite transparencia
- **BMP**: sin compresión, máxima calidad
- **TIFF**: alta calidad, admite capas
- **GIF**: formato de animación

---

## 14. Interfaz de línea de comandos

OpenFocus puede funcionar sin la interfaz gráfica, lo que resulta útil para
canalizaciones automatizadas y para integración continua. Desde un entorno
Python (o junto al ejecutable empaquetado):

```bash
# Fusiona una carpeta con el Filtro guiado predeterminado
python main.py --input ./stack_folder --output ./result/fused.png

# DTCWT con registro ECC, 8 hilos
python main.py -i ./stack_folder -o ./result/fused.png -m dtcwt -a ecc -t 8

# StackMFF-V3 forzado a CPU con un tamaño de mosaico personalizado
python main.py -i ./stack_folder -o ./result/fused.png -m stackmffv3 --cpu --tile-size 512

# Lista explícita de archivos; también se aceptan archivos de vídeo como entrada
python main.py -i img1.jpg img2.jpg img3.jpg -o fused.png -m gfgfgf

# Una única pila Z TIFF multipágina con barra de escala calibrada por metadatos
python main.py -i ./zstack.tif -o fused.png -m dtcwt --scale-bar
```

| Opción | Descripción |
|--------|-------------|
| `--input, -i` | Carpeta de imágenes, archivo de vídeo o uno o varios archivos de imagen |
| `--output, -o` | Ruta de la imagen de salida (la extensión determina el formato) |
| `--method, -m` | `stackmffv3` (predeterminado), `none`, `openfocus_a`, `openfocus_b`, `openfocus_c`, `guided_filter`, `dct`, `dtcwt`, `gfgfgf` |
| `--align, -a` | `none` (predeterminado), `homography`, `ecc`, `both`, `openfocus` |
| `--align-shift-limit` | Solo para la alineación `openfocus`: desplazamiento máximo aceptado como fracción del tamaño del fotograma (0,08 por defecto; los fotogramas que lo superan se dejan sin alinear) |
| `--exposure-match` | Iguala la media de cada fotograma a la del último antes de la fusión (activado por defecto; `--no-exposure-match` lo desactiva). Las ganancias se calculan sobre toda la pila, así que la salida por mosaicos sigue sin costuras; se aplica a todos los métodos y a la referencia de brillo del alineador de OpenFocus |
| `--base-frame` | Paso opcional posterior a la fusión: iguala las bajas frecuencias de la salida al fotograma `last`/`first` (`none` por defecto) |
| `--sort` | Orden de los fotogramas para `openfocus_a/b/c`: `asc` (predeterminado, orden de archivo), `desc` (invierte un barrido de lejos a cerca), `off` |
| `--dust-map WHITE` | Corrección de campo plano (mapa de polvo) a partir de una toma blanca desenfocada |
| `--kernel, -k` | Tamaño de núcleo (se fuerza a impar; en la familia openfocus es el origen del radio de la ventana de varianza) |
| `--depth-smoothing SIGMA` | Solo `openfocus_b`: sigma gaussiana de la transición de profundidad entre fotogramas (1,2 por defecto; 0 conserva el detalle más fino) |
| `--blend-smoothing SIGMA` | Solo `openfocus_a` / `openfocus_c`: sigma gaussiana del mapa de pesos por fotograma (2,0 para A y 3,0 para C por defecto; 0 conserva la elección por píxel más nítida) |
| `--threads, -t` | Límite de hilos; `0` = automático (por defecto), los valores positivos limitan la concurrencia |
| `--resource-preference` | Modo de recursos: `balanced` (por defecto), `responsive` o `maximum` |
| `--batch-size` | Límite de lote de mosaicos de IA (2 por defecto); actualmente un mosaico por llamada |
| `--cpu` | Fuerza la CPU aunque haya una GPU disponible |
| `--downscale` | Ancho de reducción para la detección de características del registro |
| `--tile-size` | Tamaño de bloque de mosaico para imágenes grandes |
| `--output-dir, -d` | Modo lote: fusiona todas las carpetas de entrada y escribe `<folder>.png` por pila |
| `--scale-bar` | Dibuja una barra de escala (calibración detectada automáticamente en los metadatos TIFF) |
| `--depth-map [path]` | Guarda además un mapa de posición de enfoque (profundidad) en color (`<output>_depth.png` por defecto) |
| `--depth-colormap` | Mapa de color para `--depth-map`: turbo, viridis, jet, gray, … (turbo por defecto) |
| `--scale-um` | Tamaño de píxel en µm/px, que tiene prioridad sobre los metadatos (implica `--scale-bar`) |

Códigos de salida: `0` éxito, `1` error de procesamiento, `2` error de uso. Al
ejecutar `python main.py --help` se muestran todas las opciones; si lo lanzas
con una ruta simple (`python main.py C:\stack`) se abre la interfaz gráfica con
esa carpeta ya cargada.

---

## 15. Atajos de teclado

### Operaciones de archivo

| Atajo | Acción |
|----------|--------|
| `Ctrl+O` | Abrir carpeta |
| `Ctrl+Shift+O` | Abrir vídeo |
| `Ctrl+S` | Guardar resultado |
| `Ctrl+Shift+S` | Guardar pila registrada |
| `Ctrl+W` | Vaciar pila |
| `Ctrl+Q` | Salir |
| `Ctrl+R` | Iniciar renderizado |
| `Ctrl+Shift+R` | Comparar todos |
| `Ctrl+T` | Retocar resultado |
| `Ctrl+M` | Mapa de profundidad |
| `Ctrl+E` | Activar o desactivar la comparación Wipe |
| `Ctrl+Shift+B` | Procesamiento por lotes |
| `Ctrl+Z` / `Ctrl+Y` | Deshacer / Rehacer (ediciones de la pila) |

### Navegación

| Atajo | Acción |
|----------|--------|
| `Left Arrow` | Fotograma anterior |
| `Right Arrow` | Fotograma siguiente |
| `Mouse Wheel` | Acercar o alejar |
| `Ctrl + Wheel` | Zoom fino |
| `Double-click` | Alternar entre ajustar a la ventana y vista al 100 % |

### General

| Atajo | Acción |
|----------|--------|
| `Space` | Activar o desactivar el modo de desplazamiento |
| `Delete` | Eliminar el elemento seleccionado |

---

## 16. Solución de problemas

### Problemas frecuentes

#### «Sin imágenes cargadas»

- Asegúrate de haber importado antes una carpeta de imágenes o un vídeo
- Comprueba que la carpeta contenga formatos de imagen compatibles

#### Los resultados de fusión se ven borrosos o con fantasmas

- Activa el registro de imágenes (elige uno en la lista desplegable: Homografía / ECC / OpenFocus rápida)
- Asegúrate de que las imágenes están bien alineadas antes de la fusión
- Prueba con otro algoritmo de fusión
- Aumenta el tamaño de núcleo para obtener resultados más suaves

#### IA no disponible

- Instala PyTorch: `pip install torch torchvision`
- Para usar la GPU, instala la versión de PyTorch con CUDA
- Compruébalo en **Ayuda → Información del entorno**

#### Errores de memoria insuficiente

- Activa el procesamiento por mosaicos en **Ajustes → Ajustes de mosaicos**
- Reduce el tamaño de mosaico
- Usa tamaños de núcleo menores
- Redimensiona las imágenes a dimensiones más pequeñas

#### El registro falla

- Prueba con un solo método de registro a la vez
- Reduce el ancho de reducción en **Ajustes → Ajustes de registro**
- Asegúrate de que las imágenes tienen suficiente contraste y características

#### Procesamiento lento

- Aumenta el número de hilos en **Ajustes → Ajustes de hilos**
- Usa una CPU más potente
- Reduce la resolución de las imágenes
- Usa algoritmos de fusión más simples (Filtro guiado en lugar de fusión por IA)

#### Aviso de SmartScreen de Windows en el primer arranque

Los binarios publicados no están firmados. Pulsa **Más información → Ejecutar de todas formas**.
El primer arranque tras descomprimir el archivo puede tardar de 30 a 60 segundos.

#### macOS indica que la aplicación «no se puede abrir» o «está dañada»

La aplicación no está firmada ni notarizada. Después de mover `OpenFocus.app` a
Aplicaciones, ejecuta:

```bash
xattr -cr /Applications/OpenFocus.app
```

o haz clic con el botón derecho en la aplicación y elige **Abrir**. Las
compilaciones publicadas para macOS son para Apple Silicon (M1/M2/M3/M4); los
Mac con Intel pueden ejecutarla desde el código fuente.

### Obtener ayuda

1. **Información del entorno**: ve a **Ayuda → Información del entorno** para comprobar tu instalación
2. **Contactar con soporte**: ve a **Ayuda → Contacto** para obtener información de soporte

---

## Apéndice: detalles de los algoritmos de fusión

### Filtro guiado

El filtro guiado realiza un suavizado que preserva los bordes usando una imagen
guía. Calcula transformaciones lineales locales entre la imagen guía y la
imagen de entrada, y las aplica para producir la salida. Este método destaca
por conservar los bordes mientras reduce el ruido.

### DCT (transformada discreta del coseno)

La DCT transforma las imágenes al dominio de la frecuencia, donde las
decisiones de fusión se toman según la magnitud de los coeficientes. Los
componentes de alta frecuencia (el detalle) se conservan, mientras que los de
baja frecuencia (las regiones suaves) se combinan según la energía local.

### DTCWT (transformada wavelet compleja de árbol dual)

La DTCWT proporciona una descomposición multiescala y multidireccional de las
imágenes. Capta características a distintas escalas y orientaciones, lo que
permite reglas de fusión sofisticadas que conservan tanto la textura como la
estructura.

### GFG-FGF

Implementa Fu et al. (2024): el mapa de enfoque grueso |I − mean(I)| se
enriquece con el gradiente de cuatro vecinos tipo gaussiano (la plantilla 3×3
T = [1 2 1; 2 4 2; 1 2 1]/16 da P = I⊗T, y G = (|I(x±1,y) − P|)/2 +
(|I(x,y±1) − P|)/2, descartando los valores por debajo de T0 = 0,005) y luego
se refina con un filtro guiado rápido. Cada píxel elige el fotograma con el
mayor valor de enfoque (en caso de empate, una medida de región pequeña de
Sobel 3×3), y los mapas de decisión binarios se suavizan con el filtro guiado
rápido antes de la mezcla ponderada.

### Fusión por IA (StackMFF-V3)

Un enfoque de aprendizaje profundo que usa una red neuronal entrenada con
grandes conjuntos de pares de imágenes multifoco. La red aprende de forma
automática estrategias de fusión óptimas y ofrece resultados de última
generación cuando está bien configurada.

---

*OpenFocus es un proyecto de código abierto. Para actualizaciones, informes de errores y solicitudes de funciones, visita el repositorio del proyecto.*
