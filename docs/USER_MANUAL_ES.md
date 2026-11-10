# Manual de usuario de OpenFocus

## Índice

1. [Introducción](#1-introducción)
2. [Primeros pasos](#2-primeros-pasos)
3. [Vista general de la interfaz principal](#3-vista-general-de-la-interfaz-principal)
4. [Importar imágenes y vídeo](#4-importar-imágenes-y-vídeo)
5. [Ver y navegar por las imágenes](#5-ver-y-navegar-por-las-imágenes)
6. [Vista de comparación Wipe (cortina)](#6-vista-de-comparación-wipe-cortina)
7. [Registro de imágenes](#7-registro-de-imágenes)
8. [Métodos de fusión de imágenes](#8-métodos-de-fusión-de-imágenes)
9. [Transformaciones de imagen](#9-transformaciones-de-imagen)
10. [Retoque, etiquetas y anotaciones](#10-retoque-etiquetas-y-anotaciones)
11. [Ajustes y configuración](#11-ajustes-y-configuración)
12. [Procesamiento por lotes](#12-procesamiento-por-lotes)
13. [Exportar resultados](#13-exportar-resultados)
14. [Interfaz de línea de comandos](#14-interfaz-de-línea-de-comandos)
15. [Atajos de teclado](#15-atajos-de-teclado)
16. [Solución de problemas](#16-solución-de-problemas)

---

## 1. Introducción

OpenFocus es una aplicación de escritorio profesional de fusión de imágenes multifoco, pensada para investigadores, ingenieros y profesionales del procesamiento de imágenes. El programa procesa secuencias de imágenes con distintos planos de enfoque mediante algoritmos avanzados de registro y fusión, y genera imágenes completamente nítidas con profundidad de campo total.

### Funciones principales

- **Fusión de imágenes multifoco**: combina varias imágenes con puntos de enfoque distintos en una sola imagen totalmente enfocada
- **Diez algoritmos de fusión**: filtro guiado, DCT, DTCWT, GFG-FGF, StackMFF V1/V2/V3, OpenFocus A/B/C, además del modo de solo alineación
- **Registro de imágenes**: alinea secuencias desalineadas con OpenFocus (rápido), ECC, homografía o la cadena homografía → ECC
- **Fondo automático**: sustituir un fondo apilado irregular por un fotograma real y ver dónde ningún fotograma captó la escena con claridad
- **Vista de comparación Wipe (cortina)**: comprueba la alineación y compara resultados con un divisor A/B
- **Archivos de proyecto**: restauran las referencias de entrada y los ajustes reproducibles; los píxeles del resultado se exportan por separado
- **Plantillas de procesamiento**: guardan y reaplican ajustes de fusión, registro y recursos a cualquier pila cargada
- **TIFF multipágina**: las pilas Z de un solo archivo se cargan página a página y se exportan de vuelta a un único archivo
- **Mapa de profundidad**: mapas de posición de enfoque por píxel con esquemas de pseudocolor personalizados
- **Barra de escala**: barras de escala con calidad de publicación y calibración automática en µm/px
- **Recuperación automática tras fallos**: tras un cierre inesperado se ofrece restaurar la última sesión de trabajo
- **Multilenguaje**: doce idiomas de interfaz —English, 中文, 日本語, Español, Deutsch, Français, Русский, 한국어, Português, Italiano, Nederlands, Polski— con detección automática del idioma del sistema
- **CLI sin interfaz y para agentes**: automatiza canalizaciones sin interfaz para CI o controla una ventana en ejecución mediante el canal de control local
- **Exportación flexible**: guarda los resultados como imágenes individuales, carpetas o animaciones GIF
- **Transformaciones de imagen**: rota, voltea y redimensiona pilas de imágenes
- **Etiquetas**: añade texto a las pilas de entrada, alineada o a resultados concretos
- **Ajustes persistentes**: tus preferencias, el idioma y las pilas abiertas recientemente se conservan entre sesiones

### Formatos de entrada compatibles

- Archivos de imagen: JPG/JPEG, PNG, BMP, TIFF, WebP y otros formatos habituales
- Archivos de vídeo: MP4, AVI, MOV, MKV, WMV, FLV y WEBM (se decodifican en secuencias de imágenes)
- Carpetas con secuencias de imágenes numeradas
- Archivos RAW de cámara (DNG, CR2/CR3, NEF, ARW, RAF, ORF, RW2 y otros formatos RAW habituales). La decodificación usa `rawpy` (LibRaw), que `pip install openfocus` instala y las compilaciones para Windows/macOS incluyen

Las pilas de PNG transparentes se pueden fusionar en la aplicación de escritorio, en el procesamiento por lotes de escritorio o en la interfaz de línea de comandos, sin fusión agrupada. Todos los fotogramas deben ser PNG de 8 o 16 bits, de las mismas dimensiones y profundidad de bits, con la misma máscara alfa. La máscara debe ser cero fuera de un único rectángulo no vacío y tener una opacidad constante distinta de cero en todo el rectángulo. Se permite una opacidad constante parcialmente transparente. El rectángulo visible se fusiona de forma independiente, por lo que los colores ocultos no pueden influir en la medición de enfoque ni en los límites de los mosaicos. La salida PNG conserva el tamaño original del lienzo y la opacidad; los colores ocultos de la salida son cero. Las vistas de origen y resultado del escritorio muestran la transparencia sobre un tablero de ajedrez.

Carga a tamaño completo, elige «sin alineación», desactiva la vista previa rápida y el ROI, y renderiza y guarda como PNG. El modo por lotes exige un método de fusión y salida PNG sin guardar una pila alineada. No se admiten alineación, redimensionado o reducción, corrección de campo plano, mapas de profundidad, comparación de métodos, retoque ni barras de escala con datos transparentes, y las etiquetas no se pueden incrustar en una exportación. Se rechaza un método de fusión que cambie las dimensiones del rectángulo. No se admiten máscaras variables o con bordes suavizados, máscaras distintas entre fotogramas, TIFF transparente de entrada ni salidas que no sean PNG, incluidos GIF y TIFF multipágina. La CLI también rechaza la limpieza del sensor y los mapas de profundidad para entradas transparentes.

Por ejemplo, el equivalente en la CLI es `python main.py --input ./png-stack --output ./result.png --method guided_filter --align none`.

### Formatos de salida compatibles

En **Guardar como**, TIFF normal guarda el resultado como una sola imagen; TIFF por capas incluye los fotogramas de origen como capas. Linear DNG siempre aparece, pero para guardarlo se requiere un resultado lineal de cámara sin ajustes incorporados y con calibración de cámara.

- Imágenes individuales: JPG, PNG, BMP, TIFF, WebP
- Secuencias de imágenes: guardadas como carpetas
- Animaciones: formato GIF
- Modelos 3D: OBJ con MTL, textura PNG y metadatos JSON

---

## 2. Primeros pasos

### Iniciar la aplicación

La forma preferida de instalar una compilación empaquetada es el sitio web del proyecto, **https://openfocus.top**: enlaza el instalador actual de Windows, la compilación de macOS y el paquete de Linux, con scripts de instalación para terminal. Prefiérelo a recompilar: esos artefactos son la salida exacta del flujo de publicación. `pip install openfocus` instala solo la interfaz de línea de comandos sin interfaz gráfica (el paquete wheel no depende deliberadamente de PyQt6); usa una compilación de escritorio o una copia del código fuente para la interfaz gráfica.

Para instalar desde una terminal, previsualiza primero el comando y ejecútalo después sin `--dry-run`: en macOS o Linux, `curl -fsSL https://openfocus.top/install.sh | sh -s -- --dry-run` y luego `curl -fsSL https://openfocus.top/install.sh | sh` (en Linux instala la CLI); en Windows, `irm https://openfocus.top/install.ps1 | iex` desde PowerShell.

Después de la instalación, abre OpenFocus de una de estas formas:

1. **Windows (ejecutable)**: haz doble clic en el acceso directo o en el ejecutable de OpenFocus
2. **macOS**: arrastra `OpenFocus.app` a Aplicaciones y ábrelo desde allí
3. **Código fuente de Python**: ejecuta `pip install -r requirements.txt` y luego `python main.py` en tu entorno activado

El primer arranque abre el diálogo de bienvenida, donde puedes elegir el tema y cargar la pila de demostración incluida. Un perfil nuevo usa el tema claro por defecto; la elección se recuerda.

### Flujo de inicio rápido

1. Importa tu secuencia de imágenes (Archivo → Abrir carpeta o arrastrar y soltar)
2. Configura los ajustes de fusión en el panel derecho
3. El registro empieza en **ECC**; elige **Sin alineación** si los fotogramas ya están alineados
4. Ajusta el tamaño de núcleo si hace falta
5. *(opcional)* Abre **Ajustes de renderizado** y marca **Vista previa rápida** para probar primero los parámetros sobre un borrador reducido
6. Pulsa **Iniciar renderizado** para empezar el procesamiento
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

1. **Ajustes de fusión**: selecciona el algoritmo de fusión y sus parámetros
2. **Ajustes de registro**: activa o desactiva los métodos de alineación
3. **Botones de acción**: Iniciar renderizado y las acciones secundarias Restablecer, Ajustes de renderizado y renderizar todos los métodos. El diálogo de ajustes reúne la vista previa rápida, el orden de los fotogramas, el igualado de exposición, la eliminación de deriva de brillo y la nitidez de salida
4. **Lista de imágenes de origen**: muestra todas las imágenes cargadas con sus nombres de archivo
5. **Lista de salida**: muestra los resultados generados

### Barra de menús

La barra de menús superior da acceso a todas las funciones:

- **Archivo**: operaciones de importación y exportación, proyectos, plantillas de procesamiento y la lista de archivos recientes
- **Editar**: transformaciones de imagen, retoque y etiquetas
- **Herramientas**: unión de imágenes, procesamiento por lotes, comparación de métodos, mapa de profundidad, corrección de campo plano, barra de escala y etiquetas
- **Ajustes**: ajustes avanzados (rendimiento, registro y diagnósticos de alineación, limpieza del sensor, nombres de salida, IA y arranque), aceleración por GPU, tema e idioma
- **Ayuda**: información del entorno, contacto, búsqueda de actualizaciones y carpeta de registros

**Herramientas → Unir imágenes…** abre un espacio de trabajo independiente para regiones de microscopio ya fusionadas y panoramas de cámara; no modifica la pila actual. La pestaña **Asistente de proyecto** reúne importación, alineación y puntos de control; **Imágenes de origen** contiene la lista, la cuadrícula y la colocación manual; **Editor de panoramas** contiene los ajustes de costuras, mezcla, exposición/color, salida, proyección, cámara y HDR, además de las superposiciones de vista previa. Use **Añadir archivos…**, **Abrir carpeta** o **Añadir resultados…** para copiar resultados elegidos de la sesión actual, o arrastre archivos o carpetas a la lista o al lienzo. La carpeta solo importa los archivos compatibles directamente dentro de ella, no subcarpetas. Las importaciones usan el orden natural del nombre, ignoran rutas duplicadas y añaden cada página de un TIFF multipágina como imagen independiente. Seleccione una imagen y pulse Suprimir o Retroceso para quitarla del espacio de trabajo sin borrar el archivo.

**Alinear automáticamente** busca características compartidas en miniaturas acotadas, ignorando píxeles enmascarados y transparentes, y ajusta cada solape conectado. El modo plano estima rotación, escala y traslación; el modo panorama también ajusta el modelo de cámara y objetivo. Las imágenes sin suficientes características comunes quedan marcadas como no coincidentes. En modo plano, arrástrelas o introduzca X/Y en píxeles, escala en porcentaje y rotación en grados, incluidas coordenadas derivadas de una platina conocida. El contenido de la imagen no puede inferir la posición de una región sin rasgos y sin solape, y la coincidencia automática no es una calibración física. Para una cuadrícula de platina declarada, defina filas, columnas, orden de captura, dirección, esquina inicial y solape estimado, y pulse **Aplicar posiciones de cuadrícula**. El número de celdas debe coincidir con el de imágenes y la cuadrícula solo se aplica al modo plano. Cambiar los controles no mueve las posiciones actuales hasta Aplicar, que las sustituye y borra las conexiones de coincidencia. Revise todos los solapes y ajuste X/Y, escala y rotación antes de guardar.

**Puntos de control** abre un editor con dos vistas previas. Elija las imágenes A y B, haga clic en la misma característica en cada vista y pulse **Añadir pareja de puntos**; las coordenadas guardadas son píxeles nativos de origen aunque las vistas y los marcadores estén escalados. La tabla muestra el estado habilitado, las imágenes, las coordenadas A/B, el origen manual o automático y el residuo en píxeles. Corrija coordenadas, desactive **Habilitado** en puntos dudosos o use **Eliminar selección**, **Eliminar el peor 10%** y **Eliminar parejas no adyacentes**. **Optimizar** ajusta todas las parejas habilitadas e informa la calidad de alineación: puntos activos, parejas débiles, imágenes desconectadas y residuo p95 en píxeles. Una pareja necesita al menos cuatro puntos habilitados para restringir el ajuste. Los puntos automáticos sobreviven a las ediciones manuales; compruebe texturas repetidas, paralaje y movimiento de la muestra. Editar puntos de control no crea una nueva vista previa completa: pulse **Vista previa de mezcla** tras editar u optimizar.

**Máscara de origen** edita la máscara de resolución nativa de una imagen. Pinte con un radio expresado en píxeles de origen; **Excluir** marca en rojo e **Incluir** restaura; **Borrar máscara** elimina la base; **Importar máscara…** acepta un PNG o TIFF cuyas dimensiones coincidan exactamente con la fuente (PNG: alfa, escala de grises o BGR; TIFF: escala de grises o BGR sin alfa). **Aplicar** confirma los trazos y **Cancelar** conserva la máscara anterior. Los píxeles excluidos no contribuyen a costuras, coincidencia de características ni píxeles de salida, y una máscara o fuente RGBA permite transparencia en la salida.

El **Editor de panoramas** controla la mezcla real. En **Buscar costuras óptimas**, **Costura según contenido (DP)** busca costuras de contenido; **Corte de grafos** solo aparece si la compilación de OpenCV lo ofrece; **Difuminado** omite la búsqueda de costuras de contenido. La precisión de costura 1/2/4/8, el semiancho de transición, el coste de color o color y gradiente y el refinamiento de ciclos de solapamiento ajustan esa búsqueda; con Difuminado se desactivan precisión, coste y refinamiento. **Mezcla** es una pirámide multibanda real (1–7 niveles, 5 por defecto), difuminado o selección dura sin solape. **Exposición y color** ofrece desactivado, compensación de exposición, balance de color o campo de ganancia local, con **Mezclar en luz lineal** opcional. El modo panorama habilita campos de objetivo por imagen: guiñada, inclinación, alabeo, campo de visión horizontal, distorsión radial a/b/c y desplazamientos del centro óptico. El viñeteado k1–k3 y el tiempo de exposición en segundos (0 conserva los metadatos del archivo cuando están disponibles) se aplican a cualquier imagen seleccionada. Son valores del usuario o estimados, no una calibración física inferida.

**Proyección** elige **Mosaico plano de microscopio**, que usa X/Y/escala/rotación afines y admite la cuadrícula, o un panorama **Rectilíneo**, **Cilíndrico** o **Equirectangular** que asigna cada fuente mediante la guiñada, inclinación, alabeo, campo de visión y modelo de objetivo de su cámara; la escala del panorama es píxeles por radián y **Optimizar campo de visión** puede ajustar el FOV. La proyección rectilínea falla si cruza su horizonte y la cilíndrica falla en un polo; elija equirectangular en esos casos. En **HDR / fusión de exposición**, **Fusión de exposición** combina cada grupo completo de exposiciones consecutivas diferentes, mientras **HDR (Debevec + Reinhard)** calibra una respuesta y la mapea tonalmente. Defina el número de exposiciones consecutivas por grupo, los tiempos de exposición (0 usa los metadatos de cada archivo cuando están disponibles) y la anchura/objetivo de la fusión o la escala/el punto blanco HDR. Debevec exige un tiempo de exposición positivo para cada fuente y la compensación de exposición desactivada; HDR no puede combinarse con mezcla en luz lineal, y los grupos incompletos o las combinaciones fotométricas incompatibles producen un error.

**Vista previa de mezcla** captura el plan global actual y renderiza una vista previa acotada (hasta 1024 píxeles en el lado mayor). **Mostrar costuras**, **Mostrar números de imagen** y **Mostrar zonas excluidas** superponen las etiquetas y la máscara elegidas, y los puntos de control seleccionados aparecen como marcadores. La opacidad de la fuente solo afecta a la vista previa del espacio de trabajo y nunca entra en la mezcla exportada; X/Y, escala y rotación definen la colocación real en modo plano. Para colocar una imagen panorámica no coincidente, ajuste la guiñada, inclinación, alabeo y campo de visión de su cámara. El valor de opacidad de la vista previa se conserva entre importaciones y cambios de disposición. Cualquier cambio de fuente, colocación, máscara, punto de control o ajuste del editor invalida el plan, así que pulse **Vista previa de mezcla** de nuevo tras editar u optimizar. Al guardar solo se reutiliza el plan capturado si las fuentes y los ajustes no han cambiado; de lo contrario se planifica y renderiza de nuevo. Los controles de salida eligen profundidad Automática, 8 o 16 bits (Automática es 16 bits si hay alguna fuente de 16 bits o HDR; si no, 8 bits), tamaño de salida 0,01–10, recorte normalizado, transparencia Automática/Activada/Desactivada (Automática la activa para proyección panorámica, fuentes con máscara o RGBA y lienzo sin cubrir) y una política de perfil de color: Automática conserva un perfil común y convierte los perfiles mixtos a sRGB, sRGB convierte, conservar el primero copia la primera etiqueta sin convertir píxeles y sin perfil omite la etiqueta. Este motor no puede convertir perfiles mixtos de 16 bits, así que elija conservar el primero o sin perfil; no hay una reducción silenciosa a 8 bits. **Conservar EXIF de origen** incrusta los EXIF saneados de cámara/objetivo/exposición/fecha/GPS de la primera fuente, descartando las etiquetas de disposición de píxeles y restableciendo la orientación. Guarde como TIFF o PNG. TIFF usa composición por bloques respaldada en disco, escritura por franjas (objetivo de 4 MiB por franja, mínimo una fila completa de salida) y BigTIFF cuando hace falta; los PNG grandes se codifican por bandas de filas. Deje suficiente espacio temporal, de salida y RAM: la salida a tamaño completo se transmite al disco, pero la planificación y las vistas previas usan memoria. Se necesitan suficientes características comunes o puntos manuales, y ningún método basado solo en imágenes infiere la escala física. Las fuentes y los ajustes quedan congelados mientras hay una operación; use **Cancelar** y espere a que termine.

---

## 4. Importar imágenes y vídeo

### Abrir una carpeta de imágenes

1. Ve a **Archivo → Abrir carpeta** o pulsa `Ctrl+O`
2. Selecciona una carpeta que contenga tu secuencia de imágenes
3. Aparece un diálogo de escala de carga: elige la escala (100 % = resolución completa)
4. Pulsa OK para cargar las imágenes

Marca **No volver a preguntar** en ese diálogo para cargar siempre con la escala elegida sin preguntar; **Ajustes → Ajustes avanzados → Preguntar la escala de carga al importar** reactiva el aviso. Una pila ya cargada se puede reducir más adelante con **Editar → Redimensionar**.

El programa hará lo siguiente automáticamente:

- Detectar y ordenar los archivos de imagen por nombre, con orden natural (por ejemplo, `frame2` antes que `frame10`)
- Mostrar la primera imagen en el panel de origen
- Rellenar la lista de imágenes de origen con todos los fotogramas

### Abrir archivos de vídeo

1. Ve a **Archivo → Abrir vídeo** o pulsa `Ctrl+Shift+O`
2. Selecciona un archivo de vídeo (MP4, AVI, MOV, MKV, WMV, FLV, WEBM)
3. El vídeo se decodifica automáticamente en fotogramas individuales
4. Cada fotograma pasa a formar parte de la pila de imágenes

### Archivos de proyecto

Archivo → Guardar proyecto almacena las referencias de entrada y los ajustes reproducibles en un `.ofproj`: el orden exacto de los fotogramas (incluidas las páginas TIFF y los fotogramas duplicados), las exclusiones, las rotaciones y volteos, la escala de carga, los parámetros de renderizado por método, las etiquetas, el estilo de la barra de escala y el fotograma actual. Los píxeles del resultado y del retoque deben exportarse por separado. Las operaciones que no se pueden reconstruir a partir de los archivos de origen exigen guardar antes una nueva pila de entrada.

Archivo → Abrir proyecto reconstruye los fotogramas registrados antes de sustituir la pila actual. Si las fuentes se han movido, elige la nueva raíz de origen y confirma la correspondencia completa entre rutas antiguas y nuevas. Vuelve a guardar el proyecto para conservar las referencias trasladadas.

Un proyecto se comporta como un documento: la barra de título muestra su nombre con un `*` mientras haya cambios sin guardar, y al cerrar la aplicación o abrir otro proyecto se pregunta antes de descartarlos. Las imágenes de origen se referencian, no se copian.

### Plantillas de procesamiento

Una plantilla de procesamiento contiene ajustes, nunca rutas de origen. **Archivo → Guardar plantilla de procesamiento…** escribe un `.oftemplate` con el método y los parámetros de fusión actuales, el método y los parámetros de registro, las opciones de brillo por método, el número de hilos, los ajustes de mosaico, el ancho de reducción del registro, el tamaño de lote de IA de StackMFF, la nitidez, el orden inverso y el indicador de reducción de ruido del sensor. La ruta de una máscara de defectos deliberadamente no se guarda.

**Archivo → Aplicar plantilla de procesamiento…** aplica esos ajustes a la pila que ya está cargada; no sustituye las fuentes actuales. El selector también acepta un proyecto `.ofproj` (los ajustes se extraen sin necesitar sus fuentes) y una receta `.json` que contenga ajustes de proyecto. Aplicar una plantilla marca el proyecto como modificado y puede cambiar parámetros de algoritmos y ajustes de recursos, así que revisa el panel derecho antes de renderizar. Las recetas siguen siendo el formato sin interfaz e incluyen la ruta de la máscara de defectos y una huella.

### Pila de ejemplo

El diálogo de bienvenida ofrece **Cargar pila de ejemplo**: seis fotos de muestra incluidas, procedentes de una pila de enfoque real, para que puedas probar el flujo de trabajo completo de inmediato sin necesidad de tener imágenes propias. También resulta útil para comprobar una instalación.

### Curva de nitidez

Una pequeña curva sobre el deslizador de navegación de origen muestra la nitidez de cada fotograma (normalizada). Los fotogramas desenfocados aparecen como valles marcados con un punto rojo. Haz clic en cualquier punto de la curva para saltar a ese fotograma; al pasar el cursor se muestra el número de fotograma.

### Pilas TIFF multipágina

El software de microscopía (ZEN, NIS-Elements, ImageJ/Fiji) suele exportar toda una pila Z como UN único archivo TIFF multipágina. OpenFocus lee cada página de ese archivo como un fotograma individual de la pila de imágenes: suelta o abre el archivo y la pila aparecerá completa (los fotogramas se nombran `<file>_page_001`, etc.). Los TIFF de una sola página se comportan exactamente como antes. También funciona en sentido inverso: **Archivo → Guardar pila → (Pila registrada / Pila de entrada) → Guardar como TIFF multipágina** escribe toda la pila en un único `.tif`, conservando la profundidad de 16 bits.

### Barra de escala (figuras para publicación)

Si el destino elegido no tiene metadatos utilizables, se selecciona la calibración manual y se requiere un tamaño de píxel válido.

Herramientas → Barra de escala… configura una barra de escala incrustada en las imágenes exportadas. El tamaño de píxel (µm/px) se detecta automáticamente en los metadatos TIFF (la convención `unit=micron` de ImageJ, `PhysicalSizeX` de OME-TIFF o una unidad de resolución explícita en centímetros), de modo que las pilas de microscopía (incluidas las TIFF multipágina) se calibran solas. La resolución de impresora (72/300 dpi y similares) *no* se toma deliberadamente como calibración, y un valor manual que introduzcas siempre prevalece sobre los metadatos detectados: una barra equivocada pero creíble en una figura es peor que no poner ninguna. Elige la esquina y el color (Auto invierte el color según el fondo); el diálogo muestra una vista previa en vivo. Con la barra activada, todas las exportaciones (imagen individual, pilas en carpetas, TIFF multipágina, GIF, exportar todo, lotes) incluyen una barra con estilo de publicación: una longitud redondeada de la serie 1-2-5 (~12 % del ancho de la imagen) con texto perfilado y legible. La CLI lo admite mediante `--scale-bar` (calibración automática) y `--scale-um <um/px>`.

La vista previa se dibuja con la geometría nativa en píxeles del destino antes de ajustarla al área visible. Ajustar, 100 %, desplazar y navegar entre fotogramas solo cambian la vista. Aplicar confirma un único cambio deshacible; Cancelar conserva la calibración anterior. Las salidas anteriores mantienen su propia calibración y estilo cuando se carga otra pila de origen.

Para quitarla de nuevo, usa **Herramientas → Barra de escala → Quitar barra de escala**: *De la pila de origen* y *Del resultado* limitan el alcance «Aplicar a» del diálogo al lienzo restante, mientras que *De todas partes* desactiva la barra. Ambos lienzos y la cortina A/B se redibujan de inmediato; quitar la barra de un lienzo que nunca la mostró deja el ajuste intacto.

### Mapa de profundidad (posición de enfoque)

**Herramientas → Mapa de profundidad…** estima posiciones de enfoque usando la medida de actividad del método seleccionado. No es una superficie medida ni un mapa exacto del origen de los píxeles: los métodos de pirámide/ondícula estiman actividad y las variantes de StackMFF informan de una posición de fotograma estimada. Elige los fotogramas marcados actuales o el origen registrado de la salida seleccionada. La procedencia histórica ausente nunca se sustituye por la pila actual. El origen y la textura permanecen vinculados a esa instantánea. Los mapas de profundidad no están disponibles para pilas PNG transparentes, ni en la interfaz ni en la CLI.

Los siete métodos con capacidad de profundidad —DCT, DTCWT, GFG-FGF, OpenFocus B, OpenFocus C, StackMFF V2 y StackMFF V3— requieren pulsar **Calcular** expresamente. StackMFF V1 no tiene mapa de profundidad. Abrir el diálogo o cambiar la medida, el origen, el registro o los parámetros nunca inicia el cálculo. El editor de parámetros aparece justo debajo del selector de medida: el núcleo de mediana de DCT y el núcleo inicial de GFG usan valores impares de 1 a 51 (7 por defecto). OpenFocus B usa un radio de respuesta de enfoque y un suavizado de profundidad; OpenFocus C usa **Suavizado de mezcla** para suavizar su mapa de profundidad, sin radio. Los niveles de descomposición de DTCWT se editan aquí como borrador local, inicializado con los parámetros del origen. StackMFF V3 muestra su propio modo de salida de IA (mezcla suave por defecto o selección dura) y una explicación en lugar de un núcleo numérico: el mapa informa de una posición de fotograma continua, no de una distancia física. La mezcla suave usa la posición ponderada por probabilidad; la dura, el fotograma más probable por píxel. StackMFF V2 no tiene parámetros e informa de la capa focal que muestrea su red. Los borradores parten de los parámetros de la salida coincidente o de los ajustes recordados del método en la ventana principal. Se recuerdan por origen y método mientras el diálogo está abierto; editarlos no cambia la ventana principal, el proyecto ni la salida registrada.

Cambiar los ajustes de cálculo marca el mapa 2D anterior como **no actual**, vacía la superficie 3D y bloquea todas las exportaciones, incluidas las coloreadas, sin procesar, físicas y OBJ, hasta que un nuevo cálculo termine correctamente. Un cálculo activo se cancela de forma cooperativa; cuando termine la cancelación, pulsa **Calcular** de nuevo. No se encola ningún cálculo siguiente. Calcular permanece visible y se desactiva mientras se calcula, se cancela o se exporta.

- **Mapa de color**: escala de grises (el estándar de publicación), 15 paletas científicas (Turbo, Viridis, Magma, …) y las tuyas propias: *Nuevo esquema personalizado…* genera un degradado a partir de cualquier número de paradas de color (haz doble clic en una parada para editar su posición y color); los esquemas personalizados se guardan y aparecen en la lista desplegable.
- **Invertir / Gamma / Suavizado**: invierte el eje de profundidad, reequilibra el tono (gamma < 1 aclara los índices bajos) y elimina las motas con una pasada que respeta los bordes.
- **Superposición**: disponible solo para la salida asociada con coordenadas coincidentes.
- **Exportación de datos**: el TIFF de 16 bits almacena un índice normalizado codificado de 0 a 65535; no son datos flotantes sin pérdida ni una distancia física. Se rechazan JPEG y BMP.
- **TIFF de índices y TIFF de imagen alternativa**: exporta la estimación de profundidad actual como índices de fotograma sin signo de 16 bits o como una imagen ensamblada a partir de píxeles de origen. Los valores de índice 1–N se refieren al orden exacto de fotogramas usado como entrada del cálculo de profundidad, tras cualquier registro y reordenación; 0 marca píxeles no válidos. Se admiten como máximo 65 535 fotogramas. Las posiciones fraccionarias estimadas se redondean al fotograma más cercano, con los valores intermedios hacia arriba. Estos índices estiman la posición de enfoque; no son la procedencia real de los píxeles de un resultado fusionado. La imagen alternativa selecciona el píxel del fotograma de entrada de profundidad correspondiente en cada ubicación válida, con los píxeles no válidos a cero. Cuando alguna entrada es de 16 bits, los valores de 8 bits se promocionan a 16 bits multiplicándolos por 257. Las exportaciones usan el ROI del mapa de profundidad y conservan las zonas no válidas a cero.
- **Profundidad física**: introduce el inicio Z, el paso positivo, la unidad y la dirección de adquisición, y confirma la correspondencia original de captura. Cambiar entre la pila actual y una salida borra la confirmación, aunque coincidan los nombres y los índices; vuelve a confirmarla antes de exportar. Los fotogramas excluidos conservan sus huecos. El TIFF Float32 almacena micrómetros, NaN para datos ausentes y metadatos de origen. Un OBJ físico también necesita calibración XY; usa una escala Z física 1:1.
- **Modelo 3D / Textura 3D**: elige el lado más largo de la malla (600 vértices por defecto). Cada diálogo recién abierto usa **Automática (color real)**: usa la salida asociada solo cuando su identidad y sus coordenadas/ROI coinciden con el mapa terminado; en caso contrario usa el primer fotograma participante de ese cálculo completado, tras el registro y cualquier reordenación. Se muestra el nombre de la salida o del fotograma resuelto. Puedes elegir explícitamente la salida coincidente, un fotograma participante o **Pseudocolor de profundidad**; las imágenes no disponibles se indican, nunca se sustituyen por salidas no relacionadas. Esta textura se comparte entre la vista previa 3D y el PNG de textura del OBJ. El mapa 2D conserva su propia paleta y superposición: cambiar su paleta o Gamma no recolorea una textura en color real. El pseudocolor explícito sigue la paleta, Gamma e inversión. La vista previa y la exportación comparten el mismo campo de alturas muestreado, sin suavizado oculto adicional. El relieve relativo se expresa en unidades de imagen.
- **Exportación del modelo**: OBJ, MTL, textura PNG y metadatos JSON forman un grupo de archivos; mantenlos juntos. La exportación se ejecuta en segundo plano. Se comprueban conflictos en todos los miembros, la preparación se organiza por etapas, la cancelación conserva los archivos existentes y los fallos de confirmación revierten las sustituciones. Guardar un modelo no exporta su imagen de fusión matriz ni elimina la protección de cambios sin guardar de esa imagen.

El cálculo se ejecuta en segundo plano con progreso y cancelación; los métodos de IA reutilizan el mismo procesamiento por mosaicos seguro para la memoria que la fusión por IA, por lo que las pilas profundas funcionan. El equivalente en la CLI es `--depth-map [path]` junto con `--depth-colormap`; escribe solo el mapa coloreado (por defecto `<output>_depth.png`), funciona únicamente con los métodos con capacidad de profundidad anteriores y se rechaza para entradas transparentes o agrupadas. La interfaz también exporta profundidad sin procesar, índices de fotograma, imágenes alternativas, profundidad física y modelos 3D.

### Recuperación automática tras fallos

OpenFocus guarda continuamente una instantánea de tu sesión de trabajo (pila, ajustes, etiquetas, fotograma actual). Si la aplicación se cierra de forma inesperada (un fallo, un corte de corriente o un cierre forzado), el siguiente arranque lo detecta y ofrece **Restaurar sesión** para volver exactamente a donde estabas. Si el cierre fue limpio, el aviso no aparece nunca. Los datos de recuperación se guardan junto a tus ajustes (las instalaciones portables los mantienen junto al ejecutable, así que viajan contigo).

### Archivos recientes

Cada pila que abres queda registrada en **Archivo → Abrir recientes** (hasta 8 entradas, numeradas en el submenú). Haz clic en una entrada para reabrirla; no hay atajo de teclado global para archivos recientes. Al reabrir la misma pila, esta pasa a la primera posición; las entradas cuyos archivos ya no existen se pueden eliminar mediante un aviso, y **Vaciar lista de recientes** deja el menú limpio.

### Arrastrar y soltar

También puedes arrastrar una carpeta directamente sobre la ventana de la aplicación para importar imágenes. Si sueltas más archivos sobre una pila existente, se añaden al final (tras comprobar el tamaño).

### Orientación EXIF

Los archivos JPEG/TIFF de cámara con etiqueta EXIF Orientation se rotan o voltean automáticamente durante la importación, de modo que las fotos en vertical se cargan derechas.

### Orígenes de 16 bits

Las entradas PNG/TIFF de 16 bits se detectan automáticamente y se conservan de extremo a extremo: la fusión se ejecuta a profundidad completa y los resultados pueden guardarse de nuevo como PNG/TIFF de 16 bits (las exportaciones JPG/BMP se convierten a 8 bits, ya que esos formatos no pueden almacenar 16 bits).

Los archivos RAW de cámara se revelan a 16 bits (balance de blancos de la cámara, gamma sRGB) con el aclarado automático desactivado, para que cada fotograma de la pila reciba la misma curva de exposición; una exposición automática por fotograma haría parpadear el brillo entre fotogramas. Guarda como TIFF o PNG para conservar los datos de 16 bits.

### Perfiles de color

Las exportaciones individuales a PNG, TIFF, JPEG y WebP conservan el perfil ICC RGB cuando todos los fotogramas de origen tienen el mismo perfil incrustado. Las pilas mixtas o sin perfil se exportan sin perfil ICC. OpenFocus conserva el perfil; no convierte los píxeles de origen a un espacio de color común ni aplica gestión de color a la vista previa. Convierte las imágenes con perfiles distintos a un espacio común antes de cargarlas. La entrada RAW se revela a sRGB y no hereda el perfil de la cámara.

En un TIFF multipágina, todas las páginas del archivo deben tener el mismo perfil RGB válido; de lo contrario, sus fotogramas se tratan como sin perfil. Los fotogramas de vídeo también se tratan como sin perfil. Los fotogramas añadidos y los seleccionados para un renderizado contribuyen a su perfil, que permanece con ese resultado para exportaciones posteriores.

### Vaciar la pila

Para eliminar todas las imágenes cargadas, ve a **Archivo → Vaciar pila** o pulsa `Ctrl+W`.

---

## 5. Ver y navegar por las imágenes

### Navegar por la pila de imágenes

- **Deslizador**: arrastra el deslizador bajo el panel de origen para recorrer los fotogramas
- **Lista de archivos**: haz clic en cualquier nombre de archivo de la lista de imágenes de origen del panel derecho; mientras la lista tiene el foco, Arriba/Abajo mueven la selección y el panel de origen la sigue
- **Teclado**: no hay un atajo global de fotograma anterior/siguiente; enfoca primero la lista de origen

### Revisar fotogramas similares

Haz clic en **Revisar fotogramas** bajo el encabezado de la lista de origen para encontrar fotogramas adyacentes cuyos píxeles son casi idénticos. El botón cambia a **Cancelar revisión** mientras se ejecuta el análisis. Cada fila sugerida indica su fotograma de referencia y la diferencia; inspecciona ambos antes de usar la casilla de origen para excluir un fotograma de la fusión. Las sugerencias nunca cambian las casillas ni eliminan archivos. Esta comparación conservadora puede pasar por alto fotogramas similares; los cambios limitados a una pequeña región enfocada no se promedian. Los resultados se descartan si la pila cambia. Las sugerencias solo se muestran cuando tanto el fotograma sugerido como su referencia están incluidos; cambiar una casilla actualiza las sugerencias mostradas.

### Zoom y desplazamiento

- **Acercar/Alejar**: rueda sobre la imagen de origen o resultado, también manteniendo `Ctrl`
- **Lupa**: mantén pulsado el botón izquierdo; la rueda cambia el aumento y `Ctrl` + rueda cambia el tamaño del cuadrado. Cada panel recuerda su tamaño
- **Ajustar a la ventana**: clic con el botón central
- **Desplazar**: arrastra con el botón derecho o mantén Espacio y arrastra con el izquierdo

En imágenes de origen grandes, la lupa empieza al 100 % nativo; el porcentaje indica píxeles del dispositivo por píxel original. La vista principal alcanza el 800 %. Las pilas JPEG y RAW se abren con vistas previas limitadas cuando están disponibles. Para inspeccionar detalles, un trabajador carga los píxeles originales y entrega solo la región visible. El original seleccionado se conserva para inspecciones sucesivas y se libera al cambiar o vaciar la fuente. La primera decodificación sigue necesitando RAM suficiente y puede tardar más que abrir la vista previa.

### Navegación sincronizada

Cuando navegas en el panel de origen, el panel de resultados muestra el fotograma de resultado correspondiente, si existe.

---

## 6. Vista de comparación Wipe (cortina)

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

- Alineación por similitud acotada (traslación + rotación + escala) con recorte automático
- Busca transformaciones de similitud acotadas mediante diferencias de imagen muestreadas y transforma las imágenes con interpolación Lanczos
- El dominio de búsqueda del desplazamiento es por defecto el 3 % del tamaño del fotograma y se cambia con `--align-shift-limit`. La búsqueda usa el fotograma alineado anterior y parte del movimiento acumulado; el desplazamiento total puede superar el 3 %.
- Para geometrías no rígidas o extremas, elige mejor ECC u homografía
- Ante un error real de estimación, `failure_policy=continue` (predeterminado) conserva el fotograma sin alinear; `fail` interrumpe el proceso. La falta de textura, los patrones repetidos y los cambios de enfoque aún pueden producir coincidencias imprecisas.
- En la CLI, selecciona la misma política con `--align openfocus --align-param 'failure_policy="fail"'` (usa las comillas adecuadas para tu shell). Los elementos de lote usan la política seleccionada de forma independiente. El valor predeterminado es `"continue"`.
- Ajusta **Fotograma de referencia** a un número basado en 1 dentro del orden de la pila incluida y procesada para anclar la geometría a ese fotograma. El orden se conserva; la ecualización de brillo usa su propia referencia. Cada pila por lotes debe contener ese número o el elemento falla. Ejemplo de CLI: `--align openfocus --align-param reference_frame=3`. Elegir una referencia no corrige el paralaje ni una mala coincidencia; inspecciona el resultado.

Los informes de alineación identifican la referencia y el resultado de cada fotograma. Un respaldo significa que el método no pudo establecer la alineación o depende de una estimación anterior fallida; el motivo indicado distingue las estimaciones débiles y el movimiento fuera de los límites cuando se conoce. Los valores de calidad son específicos de cada método, no una garantía de precisión. Las transformaciones asignan coordenadas de origen al lienzo de referencia de la etapa de registro antes del recorte común; los métodos compuestos informan de sus etapas por separado. Homografía y ECC también aceptan el parámetro opcional `failure_policy="fail"` para detenerse cuando sus algoritmos detectan un fallo. El valor predeterminado sigue siendo `"continue"`; inspecciona los resultados incluso cuando la alineación tenga éxito.

Para inspeccionar un renderizado, activa **Ajustes → Ajustes avanzados → Registro → Recopilar diagnósticos de alineación** y vuelve a renderizar. Esto recalcula la alineación en lugar de reutilizar imágenes en caché. El estado de finalización resalta los respaldos; abre **Ajustes → Ajustes avanzados → Registro → Diagnósticos de alineación** para ver el informe por fotograma. Muestra los motivos de fallo disponibles, las puntuaciones específicas del método, los recuentos de coincidencias, los fotogramas referenciados y las transformaciones. Los números y nombres de referencia se refieren a la lista de origen original, incluso cuando se excluyen fotogramas. Los renderizados por lotes y agrupados no ofrecen este informe.

### Activar el registro

1. En el panel derecho, localiza el grupo **Registro**
2. Elige un método en la lista desplegable: **Sin alineación**, **OpenFocus (rápida)**, **Alinear (ECC)**, **Alinear (homografía)** o **Homografía → ECC** (encadenado)
3. Para OpenFocus (rápida), elige un fotograma de referencia. En **Ajustes → Ajustes avanzados → Registro**, selecciona **Gestión de fotogramas rechazados**; al cambiarla se aplica a los métodos de registro
4. **Recopilar diagnósticos de alineación** ahora está en Ajustes avanzados (ver arriba)

### Cuándo usar el registro

- Tus imágenes presentan una desalineación visible
- Las imágenes se tomaron a pulso
- La escena tiene paralaje (los objetos cercanos se desplazan entre fotogramas)
- Los resultados muestran artefactos de fantasma o duplicación

### Ajustes de registro

Accede a los ajustes adicionales en **Ajustes → Ajustes avanzados → Registro**:

- **Ancho de reducción**: controla la resolución de preprocesamiento (1024 px por defecto)
- Valores menores = procesamiento más rápido, posiblemente menos precisión
- Valores mayores = procesamiento más lento, posiblemente más precisión
- **Recopilar diagnósticos de alineación** y **Caché de registro** están en la misma sección

---

## 8. Métodos de fusión de imágenes

OpenFocus ofrece diez algoritmos de fusión (además de un modo de solo alineación). Un perfil nuevo empieza en **StackMFF V3** con registro **ECC**; las listas siguientes describen todas las opciones. Cada uno tiene características distintas que lo hacen adecuado para diferentes tipos de imágenes.

### Filtro guiado

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
- **Regla de fusión**: una sola decisión por píxel y nivel para todas las orientaciones y canales de color (N vías, con comprobación de consistencia mayoritaria); la banda de paso bajo sigue la decisión más gruesa en lugar de promediar todos los fotogramas
- **Parámetro**: niveles de ondícula (1–6, 4 por defecto) en el panel derecho. El método no usa núcleo espacial, así que el deslizador de núcleo se oculta.

### GFG-FGF (gradiente de cuatro vecinos tipo gaussiano + filtro guiado rápido)

- **Algoritmo**: implementación de Fu et al. (*Laser & Optoelectronics Progress*, 2024): un mapa de enfoque por diferencia de desenfoque |I − mean(I)| se enriquece con el operador de gradiente de cuatro vecinos tipo gaussiano (GFG), se refina con un filtro guiado rápido, se decide por píxel (los empates se resuelven con una medida de región pequeña de Sobel 3×3) y los mapas de decisión binarios se suavizan de nuevo con el filtro guiado rápido antes de la mezcla ponderada.
- **Ideal para**: microscopía e imágenes con regiones de enfoque claras
- **Ventajas**: rápido, límites de región de enfoque nítidos; la memoria no crece con el número de fotogramas
- **Parámetro**: núcleo = tamaño del filtro de media del mapa de enfoque grueso

### StackMFF (V1, aprendizaje profundo)

- **Algoritmo**: una red de atención de una sola escala mezcla la luminancia de origen con pesos softmax; Cb/Cr se toman del fotograma cuya luminancia más se aproxima a la fusionada
- **Ideal para**: fusión neuronal de una pila de enfoque completa
- **Ventajas**: sin parámetros y color dentro de gama; la luminancia fusionada es independiente del orden de los fotogramas, pero la crominancia puede variar si hay empate en la distancia de luminancia
- **Nota**: unos 156.172 parámetros; las pilas grandes se benefician de la aceleración por GPU

### StackMFF V2 (aprendizaje profundo)

- **Algoritmo**: una red de atención de una sola escala normaliza la profundidad de enfoque esperada y muestrea el fotograma de origen más cercano. Los índices de profundidad dependen del orden de la pila. **Los fotogramas deben mantener su orden de enfoque original** — invertir está desactivado para este método
- **Mapas de profundidad**: compatibles; informa de la capa focal muestreada
- **Ideal para**: barridos de enfoque continuos en los que importa un modelo muy pequeño y rápido
- **Requisitos**: ONNX Runtime y el modelo ONNX incluido.

### StackMFF V3 (aprendizaje profundo)

Es el método preseleccionado en una instalación nueva.

- **Algoritmo**: clasificación de origen píxel a píxel (ONNX Runtime). La red predice, para cada píxel, de qué imagen de entrada proviene, y después mezcla las imágenes según esas probabilidades o elige una sola imagen
- **Modo de salida de IA** (panel derecho, solo para este método):
  - **Mezcla suave (ponderada)**: opción predeterminada; cada píxel es una mezcla de las imágenes ponderada por probabilidad. Transiciones suaves y sin costuras duras; coincide con la configuración publicada en el artículo
  - **Selección dura (una imagen)**: cada píxel se copia de la imagen más probable. Selección más nítida y sin interpolación entre imágenes; puede aparecer una costura a nivel de píxel donde dos imágenes empatan
- **Ideal para**: máxima calidad cuando hay GPU disponible
- **Ventajas**: invariante al orden (el orden de los fotogramas nunca cambia el resultado), admite pilas dentro de los recursos disponibles, sin ajustes por método, optimización automática
- **Requisitos**: ONNX Runtime y el modelo ONNX incluido.

Cada mosaico neuronal usa todos los fotogramas. Los fotogramas y los datos temporales de alineación pueden guardarse en disco; el espacio libre y la memoria disponible limitan el tamaño de la pila. Los mosaicos espaciales de V2 aproximan la inferencia de la imagen completa.

Si no se especifica un directorio temporal, OpenFocus estima el espacio necesario según las entradas y los datos temporales; si falta espacio, puede considerar discos locales fijos con permiso de escritura. Si el directorio elegido no está disponible, avisa y usa temporalmente el directorio temporal del sistema; cada nuevo trabajo vuelve a intentar la opción guardada. Si el disco elegido está lleno, no cambia automáticamente. La comprobación es una estimación puntual, no una reserva. Ante un error de almacenamiento, elige un directorio escribible en los ajustes avanzados.


### OpenFocus A / B / C

A usa una respuesta de enfoque multiescala (radio 8, suavizado 4); su suavizado está inactivo actualmente.

B usa un mapa de profundidad (radio 8, suavizado 4, feather 3). C usa una pirámide multinivel ponderada (suavizado 4) y no tiene radio.

CLI: `--method openfocus_b --param radius=8 --param smoothing=4 --param feather=3`.

En la interfaz, la igualación de exposición está activada por defecto en A/B/C y usa el primer fotograma procesado; en los demás métodos está desactivada y usa el último. Se conservan las opciones guardadas explícitamente; al activarla en A/B/C se usa el primero. La corrección de deriva de brillo permanece desactivada por defecto. Restablecer recupera los valores de cada método. En la CLI está activada por defecto para todos los métodos; `--no-exposure-match` la desactiva.

Al seleccionar A/B/C se usa la alineación OpenFocus salvo que se haya elegido explícitamente otra alineación o restaurado un proyecto. Sus valores predeterminados son desplazamiento 0,03 y aumento 5 %.

**Nitidear la salida fusionada** aplica una máscara de enfoque suave después de la fusión, antes de convertir el resultado a su profundidad de salida. Está desactivado por defecto y se puede activar en **Ajustes de renderizado** o en las opciones del lote; la opción equivalente de la CLI es `--sharpen-output`. Afecta a la imagen fusionada, no a los fotogramas registrados ni a los mapas de profundidad. Puede hacer más distintos los bordes, pero también amplificar el ruido y los halos. Se guarda con el proyecto y se recuerda entre sesiones.

### Solo alineación (sin fusión)

Selecciona **Solo alineación (sin fusión)** para registrar la pila y añadir una entrada de pila a la lista de salida. Los fotogramas permanecen en memoria hasta que los exportes; la solo alineación no crea una caché adicional ni escribe junto a los archivos de origen. Guarda la entrada para crear una carpeta dentro del destino elegido. Se conservan los nombres de archivo originales, con la extensión del formato que elijas.

### Seleccionar un método de fusión

1. En el panel derecho, localiza el grupo **Fusión**
2. Elige un método en la **lista desplegable** (siempre hay exactamente uno seleccionado; la primera entrada registra sin fusionar)
3. Ajusta el tamaño de núcleo si corresponde: la etiqueta indica qué significa el núcleo para el método seleccionado
4. Pulsa **Iniciar renderizado** para procesar

### Guía de comparación de algoritmos

| Algoritmo | Velocidad | Calidad | Ideal para |
|-----------|-------|---------|----------|
| Filtro guiado | Rápida | Buena | Uso general |
| DCT | Media | Buena | Texturas |
| DTCWT | Media | Muy buena | Escenas complejas |
| GFG-FGF | Rápida | Buena | Regiones de enfoque |
| StackMFF (V1) | Lenta (GPU) | No evaluada aquí | Luminancia independiente del orden; crominancia variable si hay empate en la distancia de luminancia |
| StackMFF V2 | Rápida (GPU) | No evaluada aquí | Barridos continuos con un modelo minúsculo |
| StackMFF V3 | Lenta (GPU) | Excelente | Máxima calidad |
| OpenFocus A | Depende de la pila | Depende de la pila | Respuesta multiescala y mezcla de zonas poco detalladas |
| OpenFocus B | Depende de la pila | Depende de la pila | Mezcla de fotogramas vecinos guiada por profundidad |
| OpenFocus C | — | — | Pirámide multinivel ponderada |

Abre **Herramientas → Comparación de métodos** para ajustar varios métodos de forma independiente. El nombre, las etiquetas de parámetros y los controles de cada método permanecen juntos cuando la ventana se estrecha. Los cambios actualizan la vista previa automáticamente. **Vista previa ROI (rápida)** fusiona el área seleccionada con el contexto circundante; **Fusión de imagen completa** comprueba el resultado de toda la imagen y permite inspeccionar otra área sin volver a renderizar. Los resultados ROI pueden diferir de los de imagen completa porque las estadísticas de exposición y algunos algoritmos dependen de la extensión de la imagen.
La **fusión de imagen completa** está seleccionada por defecto.

Una indicación visible explica cómo seleccionar o deseleccionar un resultado: haz clic en él o usa `Tab` para enfocarlo y pulsa `Enter` o `Espacio`.

Usa **Tamaño de vista previa (px)** sobre los paneles derechos para elegir un tamaño cuadrado de 160 a 960 píxeles, arrastrando el deslizador o introduciendo un número. Todos los métodos usan el mismo tamaño y se ajustan al ancho disponible. El tamaño se recuerda para la próxima apertura. Cambiarlo mantiene la ampliación actual y ajusta el cuadrado de inspección alrededor de su centro, limitado a los bordes de la imagen.

Pulsa **F11** o el botón **Pantalla completa (F11)** para usar toda la pantalla; el texto del botón cambia a **Salir de pantalla completa (F11)**. La barra de título también puede maximizar la ventana. **Esc** sale primero de pantalla completa y solo cierra el diálogo en una segunda pulsación. Al volver de pantalla completa se restaura el tamaño anterior de la ventana, y la cuadrícula de resultados se redistribuye al ancho disponible en ambos casos.

Arrastra con el botón izquierdo sobre la imagen de origen para mover el cuadrado blanco de inspección; al soltar, permanece en su sitio. Los paneles derechos muestran exactamente esa región de la misma pila alineada. Usa la rueda para cambiar la ampliación de la región, arrastra con el botón derecho para desplazar el origen, **Ctrl+rueda** para ampliar el origen y **F** para ajustar. Cambiar de fotograma de origen o redimensionar la ventana conserva la selección. Cancelar detiene el trabajo pendiente; cerrar espera a que la operación nativa actual termine de forma segura.

### Vista previa rápida (renderizado de borrador)

Ajustar parámetros en pilas grandes es lento a resolución completa. Abre **Ajustes de renderizado** y marca **Vista previa rápida**; OpenFocus renderizará primero un borrador a la mitad del tamaño lineal (nunca por debajo de 320 px en el lado largo), normalmente varias veces más rápido, incluido el registro. El mensaje de finalización (y la información emergente del resultado más reciente) indica **Calidad: vista previa**, y el borrador **no** se añade al historial de salida. Desmarca la casilla y vuelve a renderizar para obtener el resultado a resolución completa. El modo de vista previa no se aplica a los renderizados ROI y nunca contamina la caché de alineación a resolución completa.

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

La fusión puede dejar todavía algún defecto local: un halo alrededor de un borde de alto contraste, una cerda tomada del fotograma equivocado, ruido en una zona plana. El espacio de trabajo copia píxeles de los fotogramas exactos que produjeron el resultado seleccionado:

1. Selecciona una salida en la lista de salida (también funciona con clic derecho → *Retocar resultado…*) y pulsa `Ctrl+T`. Si no hay selección, un selector lista todos los resultados e indica cuáles tienen fotogramas de origen registrados.
2. El fotograma de origen está a la izquierda y el resultado a la derecha. Ambos comparten zoom y posición: la rueda amplía, arrastrar con el botón derecho o central (o Espacio+arrastrar) desplaza, `F` ajusta y `1` muestra el 100 %.
3. Elige una miniatura, arrastra el deslizador de fotogramas o pulsa `,` / `.`. Un trazo queda bloqueado a ese fotograma y a los ajustes de pincel actuales; el fotograma no puede cambiar hasta que termine el trazo.
4. Elige **Pincel de origen** (`B`) o **Pincel de restauración** (`E`). El diámetro, la dureza y la opacidad tienen deslizador y campo numérico; `[` / `]` ajusta el diámetro. Con `Ctrl` y la rueda hacia arriba aumentas el diámetro; hacia abajo lo reduces. La rueda sin modificadores cambia el zoom. El pincel de restauración devuelve los píxeles a la imagen que abriste, no a una salida anterior.
5. Mantén `Tab` para ver el resultado antes de esta sesión y `Alt` para asomarte al fotograma de origen. La referencia de nitidez es opcional, cubre solo el lienzo de origen y usa una escala compartida para todos los fotogramas.
6. `Ctrl+Z` / `Ctrl+Y` deshacen y rehacen trazos completos. **Restaurar todo…** es un único paso deshacible. **Aplicar como nueva salida** añade una entrada llamada `<nombre> · Retocado N` y la selecciona; el original queda intacto. La nueva entrada se marca como *Sin exportar* hasta que la exportes desde la ventana principal, y OpenFocus la protege de la expulsión automática del historial y de cerrar sin tomar una decisión.

Aplica el retoque como nueva salida, selecciónala y abre **Guardar como**. Elige TIFF por capas en la lista de formatos para guardar el resultado con sus fotogramas de origen. Las etiquetas y barras de escala activas no se incluyen; los cambios de anotaciones sin guardar siguen protegidos. Se conservan la geometría de origen y las máscaras de validez. El resultado y las fuentes deben ser imágenes RGB o de escala de grises opacas con la misma profundidad de 8 o 16 bits, y cada rectángulo de origen debe caber dentro del resultado. La exportación admite hasta 128 fuentes; los nombres de origen se limitan a 255 caracteres y no pueden contener NUL ni caracteres fuera del Plano Básico Multilingüe, como muchos emoji. Rechaza contextos inseguros y trabajos que se estima necesitan más de 512 MiB de memoria de trabajo o superan el presupuesto de RAM disponible, e informa del fallo en lugar de escribir una exportación incompleta. GIMP 3.2.6 en Windows importa las capas de 8 y 16 bits, incluidos sus desplazamientos y el relleno transparente de registro. El resultado es visible y las capas de origen están ocultas; GIMP también añade una capa de fondo visible. Otros editores no se han probado.

Para una corrección tonal local, selecciona **Pincel de exposición**, ajusta su cantidad en centésimas de paso (100 duplica los valores de píxel, -100 los reduce a la mitad) y pinta sobre el resultado. La vista previa se actualiza mientras pintas. Cada trazo ajusta el resultado actual una vez, incluso donde sus toques se solapan; otro trazo se acumula. Usa la opacidad y dureza del pincel para controlar la mezcla. Las luces pueden recortarse al aclarar. Mantén **Antes** para comparar con el resultado tal como se abrió, pulsa Escape durante un trazo para cancelarlo o usa deshacer y rehacer después. Son ediciones de píxeles, no capas ajustables. Aplica como nueva salida y exporta la nueva entrada para guardar los píxeles editados; un proyecto no almacena esos píxeles.

**Ajustes del resultado…** cambia todo el resultado de retoque actual. Su vista previa muestreada se actualiza al cambiar exposición y contraste; marca **Antes** para comparar con el resultado previo a abrir el ajuste. Ambos controles usan centésimas de paso: la exposición multiplica los valores de píxel y después el contraste escala su distancia al gris medio. Los valores extremos pueden recortar luces o sombras. **OK** aplica el ajuste como una única edición deshacible; **Cancelar** deja el resultado intacto. Durante el procesamiento, Cancelar solicita una reversión. La imagen completa debe caber en el límite de memoria del historial de retoque para esta operación deshacible. Publica y exporta el resultado como se describe arriba.

La procedencia viaja con la salida: los fotogramas alineados, los desplazamientos de recorte y pegado del ROI, el recorte de bordes de DCT y la profundidad de bits se registran cuando se renderiza el resultado. Los archivos de lote, los resultados heredados y las entradas de carpeta de solo alineación no tienen origen registrado, así que se listan como no disponibles en lugar de suponerlos a partir de la pila actual. Los trazos no aplicados y el historial de deshacer son solo de sesión; guardar un proyecto no los almacena.

Para un archivo de imagen existente, elige **Editar → Retocar archivos de imagen…** y después selecciona el resultado y una o varias imágenes de origen. Las fuentes ya deben estar alineadas con el resultado en la misma cuadrícula de píxeles; que las dimensiones coincidan no demuestra la alineación. Se aceptan imágenes PNG, TIFF, JPEG y BMP de una sola página con dimensiones verticales, canales y profundidad de 8 o 16 bits coincidentes; la orientación EXIF se aplica al cargar. No se admiten imágenes con alfa ni multipágina. Los fotogramas seleccionados los aporta el usuario, no son procedencia registrada del renderizado. Aplicar crea una salida separada sin exportar; guárdala desde la lista de salida. Los archivos elegidos se protegen contra sobrescritura durante esta sesión, incluidas las exportaciones de mapas de profundidad y mallas 3D. Como en otros resultados, los archivos de proyecto no almacenan estos píxeles ni el historial de pincel.

### Sustituir el fondo (Fondo automático)

Al apilar, el fondo se enfoca junto con el sujeto. El fondo de una imagen
apilada suele quedar como un mosaico de desenfoques distintos, con un halo
alrededor del sujeto. **Fondo automático…** en el espacio de retoque lo
sustituye por un fotograma real de la misma pila. El sujeto queda tal como se
apiló y el fondo se obtiene a partir de un fotograma y del resultado apilado. El
procesamiento puede reconstruir, ajustar y mezclar los píxeles del fondo.

1. Abre el resultado en el espacio de retoque y pulsa **Fondo automático…**.
2. Elige el fondo:
   - **Automático** (recomendado) usa el fondo suave, salvo que la pila haya
     registrado enfocada gran parte del fondo. Entonces usa el nítido.
   - **Fondo suave** usa el fotograma con el fondo más desenfocado.
   - **Fondo nítido** usa el fotograma enfocado más lejos.
3. Deja activado **Limpiar halos en los bordes** para quitar los halos de los
   bordes del sujeto. Nunca va más allá de lo que registraron los fotogramas.
   Desactívalo para conservar el borde propio de la pila.
4. Marca **Conservar las partes suaves unidas al sujeto** cuando partes del
   sujeto quedan justo fuera de la profundidad de campo, como el extremo
   redondeado de una espina. También puede conservar objetos del fondo junto al
   sujeto tal como se apilaron, así que revisa el resultado.
5. Pulsa **Sustituir fondo**. Una pila de 15 fotogramas de 20 megapíxeles tarda
   alrededor de un minuto y necesita unos 3,5 GB de memoria libre. **Cancelar**
   la detiene.

El cambio es un solo paso que se puede deshacer, y los trazos pintados antes
sobre el sujeto se conservan. Usa el **Pincel de restauración** para recuperar
los píxeles apilados donde los prefieras, por ejemplo una brizna de hierba
desenfocada delante del sujeto. Si el sujeto ocupa casi todo el encuadre, como
en una escena del suelo del bosque, la imagen queda tal como se apiló y aparece
un aviso.

Si abres el retoque sobre un resultado que ya se retocó, el fondo automático
analiza el resultado apilado del que procede y coloca el nuevo fondo sobre tu
imagen retocada. Los retoques con pincel sobre el sujeto se conservan y un fondo
ya reemplazado nunca se vuelve a analizar, así que ejecutarlo dos veces no
acumula halos.

Tras una pasada, **Mostrar datos faltantes** colorea los lugares que ningún
fotograma vio con claridad. Son los lugares que conviene revisar o retocar a
mano:

- magenta: fondo que ningún fotograma registró sin el sujeto delante
- cian: sujeto oculto a la cámara
- amarillo: sujeto fuera del rango de enfoque de la pila
- pizarra: zonas demasiado lisas para juzgar

**Guardar mapa de datos faltantes…** guarda la vista coloreada y una máscara en
blanco y negro por color. El fondo automático necesita los fotogramas que
produjeron el resultado y no está disponible para resultados de región (ROI).
Los nuevos puntos de control de fondo restauran exactamente la imagen, el
historial accesible y los diagnósticos sin volver a calcular el fondo.

### Gestión de etiquetas

Abre **Herramientas → Etiquetas** después de cargar imágenes o producir un resultado. Cada etiqueta tiene su propio destino: la pila de origen, la pila alineada o los resultados de salida. Las etiquetas de salida pueden aplicarse a todos los resultados o al resultado seleccionado por su nombre; renombrar o reordenar ese resultado no cambia la asociación.

Usa **Añadir** para crear otra etiqueta, selecciona una entrada para editarla y usa su casilla para activarla o desactivarla. Introduce texto fijo o inserta `{value}` para un número de fotograma. El número es el valor inicial más el índice de fotograma basado en cero multiplicado por el intervalo; una salida única siempre usa el valor inicial.

Elige todos los fotogramas, el actual o un rango personalizado como `3-5,8`. Los números de fotograma no válidos o no disponibles impiden aplicar el borrador. Los rangos actuales y personalizados siguen las identidades de origen seleccionadas cuando se reordena la pila; las etiquetas de todos los fotogramas también se aplican a los fotogramas añadidos. Los destinos ausentes permanecen asociados a sus identidades originales hasta que elijas explícitamente un destino nuevo.

La vista previa usa el destino real. Navega con Anterior/Siguiente o escribe su número de fotograma, elige **Ajustar** o **100 %**, y arrastra la imagen para desplazarla al 100 %. La posición y el tamaño de fuente usan píxeles de imagen. Se informa de una fuente alternativa o de un dibujo fallido. Las etiquetas conservan la profundidad de bits original al exportarse como PNG o TIFF.

**Aplicar y cerrar** confirma todo el borrador como un único cambio deshacible. Cancelar, Esc y cerrar el diálogo lo descartan. Seleccionar otra etiqueta conserva las ediciones del borrador. **Eliminar** quita solo la entrada seleccionada, incluida la última. Las acciones de eliminación del menú borran las etiquetas de origen, de la pila alineada, de la salida o todas de forma independiente, y también se pueden deshacer. Aplicar etiquetas cambia la visualización de la sesión; exporta una imagen para escribir las anotaciones en un archivo.

---

## 11. Ajustes y configuración

Abre **Ajustes → Ajustes avanzados** para las preferencias de rendimiento, registro, limpieza del sensor, nombres de salida, IA de StackMFF y arranque. Los cambios siguen siendo borradores hasta **OK**; Cancelar, Esc y cerrar los descartan. Reducir el tamaño de mosaico actualiza de inmediato el límite de solape visible a la mitad de un mosaico. El límite de lote guardado de la sección de IA se conserva, pero la inferencia actualmente envía un mosaico por llamada; el ancho de registro puede cambiar la precisión de la alineación. La recopilación de diagnósticos de alineación es una casilla de la sección de Registro.

Las descargas automáticas de actualizaciones se ejecutan mientras la aplicación está inactiva y nunca se instalan solas. Una descarga completada permanece lista hasta que elijas explícitamente **Reiniciar y actualizar**. El reinicio protege las ediciones activas y el trabajo sin guardar, y después escribe los datos de recuperación antes de lanzar el reemplazo. Cancelar una descarga o preparación espera al límite seguro actual. Una compilación preparada completa y más reciente puede instalarse sin conexión.

La corrección de campo plano cambia los píxeles de trabajo de la sesión sin sobrescribir los archivos de origen. Selecciona la referencia blanca y un fotograma de vista previa, y vuelve a generar la vista previa después de cambiar el origen o la escala. Las imágenes de referencia y de trabajo/base deben compartir dominio de coordenadas y tamaño. Aplicar prepara ambas pilas antes de confirmar; la eliminación opcional del fotograma de referencia pertenece al mismo paso de deshacer. Exporta las imágenes de origen corregidas cuando necesites conservar estos píxeles fuera de la sesión.

El procesamiento por lotes acepta una imagen de campo blanco opcional en sus opciones de procesamiento. Elige una referencia desenfocada a la resolución original de los fotogramas; se escala con la pila de entrada y se aplica antes del registro, usando el tamaño de polvo predeterminado. Una referencia ilegible o un tamaño incompatible hacen fallar visiblemente la pila afectada. La referencia se excluye de las entradas de carpeta, incluido el procesamiento agrupado; deben quedar al menos dos fotogramas de origen para fusionar. La corrección de campo plano no se puede combinar con entrada transparente. Deja el campo vacío para procesar sin corrección. La CLI también acepta la referencia con `--dust-map WHITE`.

La limpieza del sensor es opcional y se ejecuta sobre los fotogramas originales en coordenadas del sensor antes del registro. Selecciona una máscara de defectos explícita para corregir píxeles calientes o muertos conocidos; OpenFocus no infiere una máscara a partir de imágenes de la muestra. La máscara debe ser una imagen de un solo canal de 8 o 16 bits con las mismas dimensiones que el origen: cero significa no afectado y 1 o el valor máximo del canal marcan defectos. Revisa su extensión antes de procesar. Se rechazan las máscaras que cubren más del 10 % del fotograma o los defectos sin un vecino no enmascarado. Mantén la máscara a la resolución original y no la selecciones como imagen de origen. Para el procesamiento en la interfaz, carga los fotogramas de origen a escala completa cuando uses una máscara de defectos; la máscara nunca se redimensiona. La limpieza del sensor (incluida la reducción de ruido) no se puede aplicar a una pila de la interfaz después de una rotación, volteo, corrección de campo plano u otro cambio de origen: vuelve a cargar los fotogramas originales o desactiva la limpieza. Esto también se aplica a proyectos restaurados con esos historiales de origen. El procesamiento ROI prealineado no se puede combinar con la limpieza del sensor; sal del modo ROI y renderiza la pila completa. Los ajustes de limpieza pertenecen a la sesión actual y al proyecto guardado, no a las preferencias globales de la aplicación. El lote y la CLI procesan sus entradas originales por separado, con la limpieza del sensor antes de la corrección de campo plano y del registro. Las máscaras de defectos exigen archivos de origen con orientación EXIF neutra. Los archivos orientados, los RAW de cámara y las fuentes cuya orientación no se puede establecer se rechazan porque sus píxeles decodificados pueden ya no coincidir con las coordenadas del sensor. Usa imágenes de origen y una máscara en la misma orientación de píxeles conocida, con metadatos de orientación neutra. La reducción de ruido por sí sola no requiere máscara de defectos.

La reducción de ruido del sensor opcional es independiente de la máscara de defectos y está desactivada por defecto. Inspecciona el resultado a tamaño completo para decidir si su suavizado es adecuado para la muestra. La limpieza del sensor conserva los archivos de origen y la profundidad de imagen de 8 o 16 bits; es independiente de la corrección de campo plano. La CLI acepta `--defect-mask-path MASK` y `--sensor-denoise`; las recetas y la API de recetas de Python capturan estas opciones. Las recetas registran una huella de la máscara y rechazan una máscara ausente o modificada. La limpieza de la CLI no se puede combinar con entrada transparente. El procesamiento agrupado aplica la limpieza a los fotogramas originales, no a los resultados intermedios del grupo.

Las recetas exigen que su ubicación de salida esté fuera de todas las carpetas de entrada y rechazan una entrada seleccionada explícitamente como salida. Elige una ubicación de salida separada para proteger las imágenes de origen y evitar incluir un resultado anterior en una nueva ejecución. Los ajustes de procesamiento de un proyecto se pueden reutilizar como plantilla o receta aunque sus fuentes originales se hayan movido: **Archivo → Guardar plantilla de procesamiento…** guarda los ajustes sin rutas de origen y **Archivo → Aplicar plantilla de procesamiento…** los aplica a una pila cargada, mientras que `openfocus recipe create --project` escribe una receta sin interfaz que aún necesita rutas `--input` explícitas. Las plantillas llevan el indicador de reducción de ruido del sensor pero no la ruta de la máscara de defectos; las recetas guardan la ruta de la máscara y su huella. Abrir el proyecto en sí sigue exigiendo sus fuentes y cualquier referencia de campo plano registrada.

Cuando una referencia de archivo se aplica en Herramientas a la escala base de la pila, un proyecto guardado registra su ruta absoluta, el hash de contenido y el tamaño de polvo, y la reaplica una vez a las fuentes originales al abrir. Mantén el archivo de referencia sin cambios y en su sitio: una referencia ausente o modificada impide abrir el proyecto. Una referencia seleccionada de la pila, o una corrección a otra escala de trabajo/base, todavía exige exportar la pila de origen corregida antes de guardar un proyecto. Si la escala de trabajo cambia después de la corrección, restáurala a la escala de corrección antes de guardar o exporta la pila corregida.

**Ayuda → Enviar comentarios** abre los registros locales. Copiar copia el registro visible de la sesión; el ZIP de comentarios contiene los registros de sesión y de la aplicación completa, además de información del entorno. Revisa el paquete antes de enviarlo tú mismo. Abrir GitHub solo abre el rastreador. Cancelar una exportación deja los archivos intactos; un error conserva el archivo de destino anterior y muestra la ruta fallida.

Cada inicio escribe su propio registro rotativo, de modo que los trabajos concurrentes de interfaz y CLI mantienen registros separados. Los ID de tarea vinculan parámetros, tiempo transcurrido y resultados completados/cancelados/fallidos; los errores inesperados incluyen una traza completa. El ZIP de comentarios también incluye el informe de fallos actual y las versiones de las dependencias. Usa **Ayuda → Abrir carpeta de registros** o la ruta mostrada en la página de comentarios para encontrar los archivos. Los registros rotan a 1 MB con tres copias de seguridad por proceso; los registros de sesiones inactivas de más de 14 días se eliminan. Para registros de soporte detallados, define `OPENFOCUS_LOG_LEVEL=DEBUG` antes de iniciar; `OPENFOCUS_LOG_DIR` permite elegir otra carpeta de registros.

### Ajustes de mosaicos (optimización de memoria)

Para imágenes grandes, OpenFocus procesa por mosaicos para evitar problemas de memoria.

Accede en **Ajustes → Ajustes avanzados → Rendimiento**:

- **Tamaño de bloque de mosaico**: tamaño de cada bloque de procesamiento (1024 px por defecto)
- **Solape de mosaico**: solape entre mosaicos para una mezcla sin costuras (256 px por defecto)
- **Umbral de mosaico**: tamaño de imagen a partir del cual se usan mosaicos (2048 px por defecto)

**Recomendaciones**:

- Mosaicos más pequeños: menos uso de memoria, procesamiento más lento
- Mosaicos más grandes: más uso de memoria, procesamiento más rápido
- Aumenta el solape si ves costuras visibles en los resultados

### Ajustes de registro

Accede en **Ajustes → Ajustes avanzados → Registro**:

- **Ancho de reducción**: resolución de preprocesamiento para el registro
  - Predeterminado: 1024 px
  - Menor para ganar velocidad, mayor para ganar precisión
- **Recopilar diagnósticos de alineación**: recalcula la alineación en el siguiente renderizado y habilita **Ajustes → Ajustes avanzados → Registro → Diagnósticos de alineación**

### Referencia de rendimiento

Consulta [PERFORMANCE.md](./PERFORMANCE.md) para ver cifras medidas de rendimiento y memoria con pilas reales grandes, así como recomendaciones prácticas para pilas profundas y entradas de alta resolución.

El panel de estado muestra la memoria física usada y total cuando el sistema operativo puede proporcionar ambas lecturas; de lo contrario, muestra `N/D`.

### Ajustes de hilos

Controla el procesamiento paralelo de la CPU:

Accede en **Ajustes → Ajustes avanzados → Rendimiento**:

- `0` selecciona la planificación automática y es el valor predeterminado para ajustes nuevos.
- Los valores positivos son límites superiores; los límites guardados sobreviven a las actualizaciones.
- La capacidad de CPU, el modo de recursos y el margen de memoria pueden reducir la concurrencia. Más hilos no hacen necesariamente más rápido un renderizado.

### Modo de recursos

**Ajustes → Ajustes avanzados → Rendimiento → Modo de recursos** decide cuánto de la máquina puede usar un renderizado. Nunca cambia la imagen fusionada.

| Modo | Comportamiento |
|---|---|
| Equilibrado (predeterminado) | Mantiene libres aproximadamente el 20 % de los núcleos físicos y una reserva de memoria para el escritorio |
| Con capacidad de respuesta | Usa cerca de la mitad de los núcleos y una ventana de trabajo menor; renderizados más lentos, máquina más ágil |
| Máximo | Usa todos los núcleos físicos que permite el presupuesto de memoria |

El planificador considera los núcleos físicos, la afinidad y las cuotas de CPU, y ejecuta una sola capa paralela a la vez. Los trabajos pesados comparten la admisión durante la preparación y la publicación; cancelar espera a que termine la llamada nativa activa. Una memoria desconocida o insuficiente produce un error de recursos. La cuadrícula de mosaicos seleccionada no cambia; elegir mosaicos más pequeños puede cambiar los píxeles de salida.

### Aceleración por GPU

StackMFF V1/V2/V3 usan ONNX Runtime. DirectML en Windows y CoreML en macOS pueden acelerar la inferencia; de lo contrario se usa la CPU. La disponibilidad del proveedor no demuestra ejecución en GPU. El soporte de otras GPU requiere validación.

- Actívala o desactívala en **Ajustes → Aceleración por GPU**
- Al desmarcarla, los métodos neuronales se fuerzan a la CPU; resulta útil cuando los controladores de la GPU dan problemas
- El panel de estado muestra `N/D` cuando no se usa o no hay acelerador, y `CPU*` cuando existe un acelerador pero el método seleccionado es solo de CPU
- La elección se aplica en el siguiente renderizado y se recuerda

### Tamaño de lote de IA

Accede en **Ajustes → Ajustes avanzados → IA**:

- El límite de lote guardado es 2 por defecto; la política de ejecución actual limita la inferencia real a un mosaico por llamada en todos los dispositivos.
- Aumentar este límite no activa lotes de GPU sin validar ni reduce el tamaño de los mosaicos espaciales.

### Temas

Hay dos temas de interfaz disponibles: **Claro (predeterminado)** y **Oscuro**. Cambia en **Ajustes → Tema**; la elección se aplica al instante (la barra de título de Windows también cambia) y se recuerda. En el primer arranque, el usuario elige un tema en el diálogo de bienvenida; una elección guardada siempre prevalece.

### Modo portable

Crea un archivo vacío llamado `OpenFocus.portable` junto a `OpenFocus.exe` (el ZIP portable ya incluye uno) y todos los datos de usuario (ajustes, registros, archivos recientes) se guardarán en una carpeta `OpenFocusData` junto a la aplicación en lugar de tu perfil de usuario, de modo que la instalación puede viajar en un lápiz USB. Borra el archivo marcador para volver al perfil normal.

### Metadatos EXIF

Las exportaciones PNG/TIFF/JPEG/WebP conservan los metadatos saneados de cámara, objetivo, exposición, fecha y GPS, y establecen Orientation en 1. BMP no incorpora EXIF. Si los datos EXIF o ICC de un JPEG superan el límite del formato, OpenFocus muestra un error claro; elige PNG o TIFF. Si el EXIF saneado de una imagen de origen supera 16 MiB, OpenFocus se niega a cargarla; elimina o reduce primero el EXIF de ese archivo.

### Entrada HEIC/HEIF

Las fotos de iPhone se cargan directamente. Requiere `pillow-heif`, que `pip install openfocus` instala y las compilaciones empaquetadas incluyen.

### Actualizaciones

**Ayuda → Buscar actualizaciones** compara tu versión con la información de `openfocus.top`. Una versión más reciente se descarga desde el mismo sitio en segundo plano mientras la aplicación está inactiva; una vez preparada la compilación, el aviso y el área de estado ofrecen **Reiniciar y actualizar**, que la aplica sin conexión. Las compilaciones que no pueden actualizarse solas abren la página de descargas del sitio web. La comprobación automática se ejecuta en silencio como máximo una vez al día.

### Caché de registro

OpenFocus puede guardar los resultados de registro en una carpeta `.openfocus_cache` junto a la pila de origen. Al reabrir la misma pila (con los archivos sin cambios) se omite por completo la alineación. Actívala o desactívala en **Ajustes → Ajustes avanzados → Registro → Caché de registro**; las entradas obsoletas se sustituyen automáticamente.

### Ajustes persistidos

Todo lo siguiente sobrevive automáticamente al reinicio de la aplicación: número de hilos, modo de recursos, parámetros de mosaicos, ancho de reducción del registro, tamaño de lote de la fusión por IA (StackMFF V3), la aceleración por GPU, la caché de registro y los interruptores de diagnósticos de alineación, el idioma de la interfaz, el tamaño de la ventana y la distribución de paneles, el formato de exportación al arrastrar, las opciones de brillo por método, los valores de suavizado y radio, y tus pilas abiertas recientemente. Los diálogos de ajustes guardan en el momento en que confirmas; todo se guarda también al cerrar la aplicación. Las preferencias se almacenan por usuario en `OpenFocus/OpenFocus.ini` (formato INI de QSettings).

### Idioma de la interfaz

Cámbialo en cualquier momento en **Ajustes → Idioma**, que enumera los doce paquetes (English, 中文, 日本語, Español, Deutsch, Français, Русский, 한국어, Português, Italiano, Nederlands, Polski). El cambio se aplica al instante y se recuerda para futuras sesiones.

### Comprobación automática de actualizaciones

Una vez al día, unos segundos después de arrancar, OpenFocus consulta `openfocus.top` en silencio en busca de una versión más reciente; las notas de la versión también proceden de ese sitio. Si existe, aparece un aviso sobre los paneles de imagen con la versión, **Actualizar ahora** (aplica una compilación preparada o inicia la descarga), **Novedades** (notas de la versión) y un botón para descartarlo. Descartarlo oculta el aviso solo para esa versión: la siguiente versión lo vuelve a mostrar, y **Ayuda → Buscar actualizaciones** lo reactiva de inmediato.

Mientras la descarga se ejecuta en segundo plano, la barra de estado muestra el progreso. También puedes comprobar manualmente en cualquier momento con **Ayuda → Buscar actualizaciones**.

---

## 12. Procesamiento por lotes

Procesa varias carpetas automáticamente:

1. Ve a **Herramientas → Procesamiento por lotes**
2. En el diálogo de lote:
   - **Añadir carpetas**: selecciona carpetas que contengan secuencias de imágenes
   - **Arrastrar carpetas**: suelta una o varias carpetas de pilas en cualquier punto del diálogo (o de la lista de carpetas) para encolarlas a la vez; al soltar se cambia al modo multicarpeta y las miniaturas aparecen a medida que se decodifican
   - **Quitar carpetas**: elimina las carpetas seleccionadas de la lista
3. Configura la salida:
   - **Guardar en subcarpeta**: crea los resultados en cada carpeta de entrada
   - **Guardar en la misma carpeta**: guarda junto a la entrada conservando los archivos existentes
   - **Carpeta personalizada**: especifica la ubicación de salida
4. Revisa los ajustes actuales de fusión y registro
5. Pulsa **Iniciar procesamiento por lotes** para empezar

El diálogo de lote muestra el progreso en tiempo real. Puedes cancelar el procesamiento en cualquier momento.

### Notas sobre el procesamiento por lotes

- Parte de una instantánea de los ajustes de la ventana principal; los controles del lote los sustituyen.
- Cada carpeta o grupo dividido se procesa de forma independiente mediante el mismo flujo de procesamiento.
- **Procesar pilas grandes por grupos** es una casilla opcional para el modo multicarpeta. Al activarla aparecen **Tamaño de grupo** (mínimo 2, 4 por defecto) y **Solape** (de 0 al tamaño de grupo menos 2); al desactivarla, ambos controles se ocultan y la agrupación queda deshabilitada. El solape de escritorio empieza en 1 y se limita al rango permitido cuando cambia el tamaño. Los grupos se fusionan por rondas usando espacio temporal en disco; la pantalla de progreso identifica cada ronda. La fusión, el igualado de exposición, la corrección base y el registro usan los fotogramas y referencias de cada grupo, incluidas las rondas intermedias, por lo que el resultado se aproxima al procesamiento de la pila completa de una vez. La agrupación requiere un método de fusión; la profundidad de fotograma original, la procedencia y las exportaciones de fotogramas alineados no están disponibles. El modo de carpeta dividida no es compatible, por lo que la casilla se rechaza allí. Los grupos registrados usan su primer fotograma como referencia de coordenadas. Cada registro elimina un borde conservador por interpolación, y las rondas posteriores intersecan los rectángulos supervivientes, incluida cualquier cola de un solo fotograma. La imagen final queda por tanto recortada. Cada grupo que se registre debe medir más de 8 píxeles en ambas dimensiones; las rotaciones difíciles o las transformaciones proyectivas pueden no tener un rectángulo seguro y producir un error. Usa entradas prealineadas sin registro o desactiva la agrupación si ocurre.
- La entrada agrupada debe tener dimensiones orientadas coincidentes y consistir en archivos de un solo fotograma con cabeceras legibles por Pillow. No se admiten RAW de cámara, archivos multipágina ni archivos con canales alfa o transparencia, incluidas las pilas PNG transparentes. Elimina el alfa o desactiva la agrupación para procesar esas pilas. Las ejecuciones de prueba de la CLI comprueban estas restricciones. En la línea de comandos, un ejemplo es `python main.py --input stack --output fused.png --group-size 8 --group-overlap 1`. La entrada agrupada de la CLI debe ser una carpeta y no se puede combinar con `--depth-map`. Una referencia de alineación OpenFocus distinta para toda la pila no se puede combinar con la agrupación. Una referencia de campo plano se aplica a los fotogramas originales antes del registro; la nitidez de salida se aplica una vez después del grupo final.
- **Qué se escribe y dónde, según el modo.** Con **solo alineación**, el destino seleccionado recibe una carpeta `<pila>_aligned` por pila y el panel de salida lista esa carpeta. Con **fusión + guardar alineados**, la imagen fusionada se escribe como `<pila>.<ext>` en el destino y sus fotogramas alineados van a `<pila>_aligned` junto a ella; el panel de salida lista solo el resultado fusionado. Con **fusión y sin exportar alineación**, solo se escribe la imagen fusionada. El panel de salida nombra las carpetas exactamente como están en disco.
- Los fotogramas alineados conservan los nombres originales. Los nombres duplicados y las carpetas existentes reciben sufijos numéricos. Los archivos de origen nunca se sustituyen.
- Las copias sobrantes `*_registered*` de versiones anteriores no se cargan como fotogramas cuando su archivo original está presente en la misma carpeta; el mensaje de carga indica cuántas se omitieron.
- PNG/TIFF conservan los datos de 16 bits; elegir explícitamente JPG/BMP convierte a 8 bits.
- La cancelación espera a que el trabajador se detenga. Los archivos confirmados siguen disponibles; las tareas parciales, fallidas, canceladas y no iniciadas se contabilizan por separado.

---

## 13. Exportar resultados

### Guardar un resultado individual

1. Selecciona el resultado que quieres guardar en la lista de salida
2. Ve a **Archivo → Guardar** o pulsa `Ctrl+S`
3. Elige formato y ubicación
4. Pulsa Guardar

En **Ajustes → Ajustes avanzados → Nombres de salida**, un prefijo de nombre de archivo opcional ayuda a identificar los resultados recién renderizados. El ejemplo se actualiza mientras escribes; usa hasta 40 letras, números, guiones bajos o guiones. Déjalo vacío para conservar la nomenclatura original. Confirma con **OK** para recordarlo o cancela para descartar el cambio. El prefijo forma parte de los nuevos nombres de la lista de salida y de sus nombres de exportación sugeridos; los resultados existentes conservan sus nombres. Aún puedes editar un nombre en el diálogo de guardado. Esta preferencia no renombra archivos de origen ni cambia los nombres de salida por lotes.

### Compartir un resultado

Haz clic derecho en una salida y abre **Compartir**. **Copiar imagen de vista previa (JPEG)** coloca el resultado a resolución completa, incluidas etiquetas y barra de escala, en el portapapeles como JPEG real y mapa de bits decodificado. **Copiar rutas de archivo** copia las rutas locales como texto y URL de archivo. Los resultados guardados sin cambios usan el archivo existente; los no guardados, modificados o cuyo archivo falta se escriben automáticamente como PNG en la carpeta temporal del sistema, sin marcarlos como guardados. El sistema puede eliminar los archivos temporales; usa **Guardar como** para conservarlos. **Mostrar en la carpeta** abre el Explorador, Finder o el gestor de archivos con el archivo seleccionado, sin pedir una carpeta. **Crear enlace compartido privado** permanece desactivado hasta que exista alojamiento compartido.

### Formato de exportación al arrastrar

Arrastrar resultados desde la lista de salida escribe archivos de imagen con el formato elegido en el menú contextual de la lista (**Formato al arrastrar**: JPG, PNG, WebP o TIFF). La elección se recuerda.

### Guardar la pila registrada

1. Ve a **Archivo → Guardar pila → Pila registrada**
2. Elige el formato de guardado:
   - **Guardar como carpeta**: elige un destino y una codificación; cada pila recibe una nueva subcarpeta con sus nombres de fotograma originales
   - **Guardar como GIF**: crea un GIF animado de la secuencia alineada

### Guardar la pila de entrada

1. Ve a **Archivo → Guardar pila → Pila de entrada**
2. Elige el formato de guardado:
   - **Guardar como carpeta**: guarda los fotogramas originales como imágenes separadas
   - **Guardar como GIF**: crea un GIF animado de la secuencia de entrada

### Formatos de exportación

- **JPG**: comprimido, buena relación calidad-tamaño
- **PNG**: sin pérdida, conserva datos de imagen de 8 y 16 bits
- **BMP**: sin compresión, convierte los datos de 16 bits a 8 bits
- **TIFF**: conserva datos de 8 y 16 bits; la exportación multipágina mantiene toda la pila en un único archivo
- **GIF**: animación con paleta de 8 bits; la cancelación deja intacto el destino existente

---

## 14. Interfaz de línea de comandos

`pip install openfocus` incluye el modelo de IA, sus pesos y la decodificación HEIC y RAW. Para DTCWT en Python 3.13+, ejecuta después `python -m pip install --no-deps dtcwt==0.14.0`. OpenFocus aporta el adaptador para NumPy 2; `pip check` seguirá señalando la restricción de NumPy del paquete original.

OpenFocus puede ejecutarse sin la interfaz gráfica, lo que resulta útil para flujos automatizados e integración continua. El script de consola `openfocus` (instalado por `pip install openfocus`) y `python main.py` desde una copia del código exponen los mismos comandos. Las opciones de fusión funcionan con ambas formas; los subcomandos son:

```bash
openfocus version                       # imprime la versión
openfocus describe                      # JSON: métodos, parámetros, capacidades, comandos de control
openfocus gui [--control] [--socket NAME] [paths…]   # inicia la aplicación de escritorio
openfocus ctl <command> [flags]          # controla una ventana en ejecución (ver AGENT_CONTROL_SPEC.md)
openfocus recipe create FILE --input … --output … [processing flags]
openfocus recipe run FILE [--dry-run]    # reproduce una receta guardada sin interfaz
openfocus retouch replay CHECKPOINT --base BASE --sources … --output RESULT
```

`openfocus gui` se inicia en segundo plano; `--control` espera el canal de control e imprime su JSON de conexión. `openfocus describe` y `version` no necesitan entrada. El flujo de trabajo para agentes está documentado en [AGENT_CONTROL_SPEC.md](./AGENT_CONTROL_SPEC.md) y `skills/openfocus-control/SKILL.md`.

Desde un entorno de Python (o junto al ejecutable empaquetado):

```bash
# Fusiona una carpeta con el método de IA predeterminado
python main.py --input ./stack_folder --output ./result/fused.png

# DTCWT con registro ECC, 8 hilos
python main.py -i ./stack_folder -o ./result/fused.png -m dtcwt -a ecc -t 8

# StackMFF V3 forzado a CPU con un tamaño de mosaico personalizado
python main.py -i ./stack_folder -o ./result/fused.png -m stackmffv3 --cpu --tile-size 512

# Lista explícita de archivos; también se aceptan archivos de vídeo como entrada
python main.py -i img1.jpg img2.jpg img3.jpg -o fused.png -m gfgfgf

# Una única pila Z TIFF multipágina con barra de escala calibrada por metadatos
python main.py -i ./zstack.tif -o fused.png -m dtcwt --scale-bar

# Inspecciona fotogramas adyacentes similares sin fusionar ni cambiar la entrada
python main.py -i ./stack_folder --suggest-duplicates
```

`--suggest-duplicates` acepta una carpeta o una lista ordenada de archivos de imagen de un solo fotograma. Imprime un objeto JSON con los arreglos `files` y `suggestions`; `index` y `reference_index` de cada sugerencia son posiciones basadas en cero dentro de `files`, y `difference` es la diferencia media normalizada de las muestras. Añade `--json report.json` para guardar el mismo objeto. Este modo de revisión no se combina con `--output`, `--output-dir`, `--json-progress`, `--dry-run` ni las opciones de limpieza del sensor. El informe compara archivos adyacentes en el orden de entrada suministrado, sin saber qué fotogramas puedes excluir después. Si excluyes una referencia, reconsidera su sucesor o vuelve a revisar los archivos restantes en su orden previsto.

Por ejemplo, añade `--align openfocus --alignment-report alignment.json` a un renderizado único. El objeto JSON contiene `method`, `input`, `frame_count` y `entries`. El `frame_index` de cada entrada está basado en cero en el orden de entrada decodificado, antes de la ordenación para la fusión; las imágenes multipágina se expanden en orden de página y los fotogramas de vídeo en orden de decodificación. Los métodos compuestos incluyen entradas separadas para cada etapa. El informe se escribe de forma atómica tras un procesamiento correcto y no puede sustituir una entrada ni otra salida. Los modos por lotes, agrupado y de prueba rechazan esta opción.

| Opción | Descripción |
|--------|-------------|
| `--suggest-duplicates` | Informa de fotogramas adyacentes casi idénticos para revisión manual; no los fusiona ni los excluye |
| `--input, -i` | Carpeta(s) de imágenes, archivo de vídeo, uno o varios archivos de imagen, o TIFF multipágina |
| `--output, -o` | Ruta de la imagen de salida (la extensión determina el formato) |
| `--output-dir, -d` | Modo por lotes: fusiona cada carpeta de entrada y escribe `<carpeta>.png` por pila |
| `--resume` | Modo por lotes: omite las carpetas cuya salida ya existe |
| `--config FILE` | Modo por lotes: anulaciones JSON por carpeta, p. ej. `{"stackA": {"method": "dct"}}` |
| `--method, -m` | `stackmffv3` (predeterminado), `stackmffv2`, `stackmff`, `openfocus_a`, `openfocus_b`, `openfocus_c`, `guided_filter`, `dct`, `dtcwt`, `gfgfgf` |
| `--param KEY=JSON_VALUE` | Define cualquier parámetro de fusión; repetible y anula las opciones heredadas |
| `--align, -a` | `none` (predeterminado), `homography`, `ecc`, `both`, `openfocus` |
| `--align-param KEY=JSON_VALUE` | Define cualquier parámetro de registro; repetible |
| `--align-shift-limit` | Dominio de búsqueda del desplazamiento como fracción del tamaño del fotograma (0,03 por defecto); el desplazamiento acumulado puede ser mayor. |
| `--alignment-report PATH` | Guarda los resultados de alineación por fotograma como JSON para un renderizado único no agrupado con alineación activada; no se admiten lotes ni grupos |
| `--exposure-match` | En la CLI está activada por defecto para todos los métodos; A/B/C usan el primer fotograma procesado y los demás el último. `--no-exposure-match` la desactiva. |
| `--base-frame` | Paso opcional posterior a la fusión: iguala las frecuencias bajas de la salida al fotograma `last`/`first` (`none` por defecto) |
| `--sort` | Orden de fotogramas para `openfocus_a/b/c`: `asc` (predeterminado, orden de archivo), `desc` (invierte un barrido de lejos a cerca), `off` |
| `--frame-order` | Carpeta de entrada: `auto` (predeterminado) ordena los fotogramas de cerca a lejos según los pasos de enfoque EXIF cuando todos los tienen (el bracketing de enfoque de OM SYSTEM empieza a media profundidad); si no, por nombre; `name` siempre por nombre de archivo. Los archivos indicados explícitamente conservan su orden |
| `--sharpen-output` | Aplica una máscara de enfoque suave a la salida fusionada (desactivado por defecto) |
| `--dust-map WHITE` | Corrección de campo plano (mapa de polvo) a partir de una toma de campo blanco desenfocada |
| `--defect-mask-path MASK` | Máscara de defectos en coordenadas del sensor que se repara antes del registro |
| `--sensor-denoise` | Reducción de ruido conservadora del sensor antes del registro |
| `--kernel, -k` | Solo para métodos que declaran `kernel_size` |
| `--ai-mode {soft,hard}` | Salida de StackMFF V3: mezcla ponderada por probabilidad (predeterminada) o fotograma más probable por píxel |
| `--threads, -t` | Límite superior de hilos de trabajo; `0` = automático (predeterminado), los valores positivos limitan la concurrencia |
| `--resource-preference` | `balanced` (predeterminado), `responsive` o `maximum`; controla la planificación de recursos |
| `--batch-size` | Límite superior del lote de mosaicos de IA (2 por defecto); la inferencia actual usa un mosaico por llamada |
| `--cpu` | Fuerza la CPU aunque haya una GPU disponible |
| `--downscale` | Ancho de reducción para la detección de características del registro |
| `--tile-enabled` / `--no-tile-enabled` | Activa la fusión por mosaicos cuando sea necesario (predeterminado: activada) |
| `--tile-size` | Tamaño de bloque de mosaico para imágenes grandes (predeterminado: automático) |
| `--tile-overlap` | Solape de mosaicos en píxeles (predeterminado: automático) |
| `--tile-threshold` | Tamaño mínimo de imagen que activa los mosaicos (predeterminado: automático) |
| `--group-size` | Fusiona una carpeta en grupos de este número de fotogramas; `0` desactiva la agrupación (predeterminado). Solo carpetas; sin `--depth-map`, entradas RAW/alfa/multipágina ni fotograma de referencia OpenFocus distinto del predeterminado |
| `--group-overlap` | Fotogramas compartidos por grupos consecutivos; `0` por defecto, como máximo el tamaño de grupo menos 2 |
| `--scale-bar` | Dibuja una barra de escala (calibración detectada automáticamente en metadatos TIFF) |
| `--scale-um` | Tamaño de píxel en µm/px, que anula los metadatos (implica `--scale-bar`) |
| `--depth-map [path]` | Guarda además un mapa de posición de enfoque coloreado (predeterminado `<output>_depth.png`); solo métodos con capacidad de profundidad, no agrupados ni transparentes. En modo por lotes solo se permite la ruta predeterminada |
| `--depth-colormap` | Mapa de color para `--depth-map`: turbo, viridis, jet, gray, … (turbo por defecto) |
| `--auto-background [preset]` | Sustituye el fondo apilado por el de un fotograma alineado, como Fondo automático en el retoque: `auto` (predeterminado), `soft` (fondo más desenfocado) o `sharp` (fotograma enfocado más lejos). Necesita los fotogramas alineados, así que no funciona con la fusión por grupos |
| `--background-mode` | Fondo automático: `clean` (predeterminado) también elimina los halos de borde del apilado, dentro de lo que registraron los fotogramas; `faithful` los conserva |
| `--keep-soft-parts` | Fondo automático: conserva las partes suaves unidas al sujeto que algún fotograma enfocó |
| `--save-stacked [path]` | Con `--auto-background`, guarda también el resultado antes de la sustitución (predeterminado `<output>_stacked<ext>`). En modo por lotes solo se admite la ruta predeterminada |
| `--missing-data-map [path]` | Con `--auto-background`, guarda también el mapa de datos faltantes y una máscara de 8 bits por clase (predeterminado `<output>_missing.png`). En modo por lotes solo se admite la ruta predeterminada |
| `--json FILE` | Escribe un resumen de resultado legible por máquina en FILE al terminar |
| `--json-progress` | Imprime un objeto JSON de progreso por paso en stdout (NDJSON); la prosa va a stderr |
| `--dry-run` | Valida las entradas e imprime el plan sin fusionar |

No existe un valor `--method none`: la CLI siempre fusiona. Usa la aplicación de escritorio para obtener una salida de solo alineación.

### Recetas de procesamiento

Una receta es un archivo JSON portátil con opciones de procesamiento de la CLI y rutas relativas a la receta. Crea una a partir de las opciones que pasarías a una ejecución normal y reprodúcela después:

```bash
openfocus recipe create line.json --input ./stack --output ./out/fused.tif \
  --method dtcwt --align ecc --threads 8
openfocus recipe run line.json
openfocus recipe run line.json --dry-run
```

`recipe create --project project.ofproj --input ./stack --output ./out/fused.tif` captura los ajustes de procesamiento de un proyecto con entradas elegidas explícitamente; las fuentes del proyecto pueden haberse movido. Las recetas rechazan controles transitorios como `--dry-run`, `--resume`, `--json` y `--config`, y exigen que la salida esté fuera de todas las carpetas de entrada (y sea distinta de una entrada seleccionada explícitamente). Una máscara de defectos se guarda por ruta relativa con una huella SHA-256, de modo que una máscara ausente o modificada hace fallar la ejecución. Las ediciones de origen del proyecto (rotación, volteo, campo plano) no se pueden capturar; exporta primero la pila corregida.

La captura del proyecto transfiere a la receta los ajustes de fusión, registro, brillo, orden, nitidez y mosaicos, e incrusta los ajustes de procesamiento para la interfaz. Las exclusiones de fotogramas, la escala de carga, las etiquetas y las ediciones no se transfieren; prepara la nueva pila de entrada en consecuencia. Las opciones de agrupación se pueden capturar desde argumentos de la CLI.

El formato JSON de la versión 1 contiene `version`, un arreglo `input`, `output` u `output_dir`, y un objeto `options`. Las rutas son relativas al archivo de receta; mueve la receta junto con sus archivos de entrada para conservar esas relaciones. La creación captura los ajustes de la CLI y los parámetros resueltos del algoritmo. Se rechazan campos desconocidos, valores inválidos y versiones no compatibles. Una ejecución de prueba comprueba la entrada sin escribir archivos de resultado. Los archivos de proyecto existentes no se modifican.

En la interfaz, Archivo → Guardar plantilla de procesamiento guarda los ajustes sin rutas de origen. Archivo → Aplicar plantilla de procesamiento aplica una plantilla, un proyecto o una receta respaldada por un proyecto a la pila actual sin sustituir sus entradas, transformaciones, exclusiones ni etiquetas. Una plantilla no es en sí misma una receta ejecutable. Los ajustes de interfaz incrustados en una receta son independientes de sus opciones de CLI; si editas el JSON, mantén ambos coherentes.

La misma API sin interfaz está disponible sin importar PyQt:

```python
from core.recipes import capture_project_recipe, capture_recipe, load_recipe, save_recipe, run_recipe

capture_recipe("workflow.json", ["-i", "stack_folder", "-o", "result/fused.png", "-m", "dct"])
capture_project_recipe("from-project.json", "session.ofproj",
                       ["-i", "new_stack", "-o", "result/fused.png"])
recipe = load_recipe("workflow.json")  # diccionario validado con rutas relativas
save_recipe("workflow.json", recipe)
exit_code = run_recipe("workflow.json", dry_run=True)  # usa False para procesar
```

`capture_recipe` interpreta sus argumentos como rutas de CLI normales relativas al directorio actual; `save_recipe` interpreta las rutas guardadas relativas a la receta. La receta registra ajustes y referencias, incluida una huella de contenido para una máscara de defectos de sensor seleccionada explícitamente. No registra los píxeles de entrada, las versiones del software, el hardware ni las sumas de comprobación de la pila de origen; guárdalos por separado cuando la procedencia sea importante. Ajustes idénticos no garantizan píxeles idénticos entre versiones de software, controladores o dispositivos; las operaciones neuronales en GPU pueden no ser deterministas. Usa `--cpu` cuando necesites una ejecución en CPU y conserva el software instalado y las entradas.

### Reproducir un punto de control de retoque

Usa los controles de guardar/cargar puntos de control del espacio de retoque con la base original y las fuentes procesadas. Las sesiones con fondo automático, incluidos los análisis correctos sin cambios de píxeles, guardan instantáneas ZIP v2 `.ofretouch` sin pérdidas con metadatos JSON y datos NumPy: la imagen de trabajo exacta, los mosaicos accesibles de deshacer/rehacer, los estados de diagnóstico y la procedencia de las fuentes. Cargar v2 no vuelve a calcular el fondo. Las sesiones solo de pincel/tonos pueden usar JSON v1 compacto; los puntos de control de fondo antiguos en JSON v1 siguen siendo reproducibles.

La verificación de fuentes se ejecuta en segundo plano y se puede cancelar. Antes de sustituir el destino, el guardado vuelve a cargar el archivo preparado y comprueba la igualdad exacta de la imagen actual, los mosaicos accesibles de deshacer/rehacer y los estados de diagnóstico. Una fuente incompatible, una verificación fallida, la cancelación o la falta de memoria dejan intactos las ediciones actuales y cualquier punto de control existente.

Si el límite de registro del punto de control impide otra edición, aplica las ediciones actuales y comienza una nueva sesión de retoque para continuar.

Guarda un punto de control de retoque desde el espacio de retoque y después proporciona la imagen base original y cada fotograma de origen procesado en el mismo orden que la sesión:

```bash
openfocus retouch replay edits.ofretouch --base fused.png --sources frame1.png frame2.png --output retouched.tif
```

Ambos formatos siguen necesitando la base original y todos los fotogramas realmente usados para el retoque, en el orden de la sesión, tras la alineación y cualquier recorte de ROI u otro procesamiento, con sus valores de píxel y profundidad de bits originales. La carga reconstruye los desplazamientos de fotograma y el comportamiento de alineación registrados y verifica la base, los fotogramas ordenados y las máscaras de validez; se rechazan píxeles, geometría o máscaras incompatibles. Cada ruta debe contener una imagen. La salida debe ser PNG o TIFF y conserva la profundidad de bits de la base. El punto de control y las imágenes de entrada no pueden usarse como salida.

Códigos de salida: `0` éxito, `1` error de procesamiento, `2` error de uso. Ejecutar `python main.py --help` muestra todas las opciones; iniciar con una ruta simple (`python main.py C:\stack`) sigue abriendo la interfaz con esa carpeta cargada.

---

## 15. Atajos de teclado

### Operaciones de archivo

| Atajo | Acción |
|----------|--------|
| `Ctrl+O` | Abrir carpeta |
| `Ctrl+Shift+O` | Abrir vídeo |
| `Ctrl+Alt+O` | Abrir proyecto |
| `Ctrl+Alt+S` | Guardar proyecto |
| `Ctrl+S` | Guardar resultado |
| `Ctrl+Shift+S` | Guardar pila registrada |
| `Ctrl+W` | Vaciar pila |
| `Ctrl+Q` | Salir |
| `Ctrl+R` | Iniciar renderizado |
| `Ctrl+Shift+R` | Comparar todos los métodos |
| `Ctrl+T` | Retocar resultado |
| `Ctrl+M` | Mapa de profundidad |
| `Ctrl+E` | Alternar comparación Wipe |
| `Ctrl+Shift+B` | Procesamiento por lotes |
| `Ctrl+Z` / `Ctrl+Y` | Deshacer / Rehacer (ediciones de pila) |

### Navegación

| Atajo | Acción |
|----------|--------|
| `Arriba` / `Abajo` (lista de origen enfocada) | Fotograma anterior / siguiente |
| `Rueda del ratón` | Acercar/alejar |
| `Ctrl + rueda` con el botón izquierdo pulsado | Cambiar el tamaño de la lupa |
| `Botón central` | Ajustar la imagen a la ventana |

### General

| Atajo | Acción |
|----------|--------|
| `Espacio` (mantener, puntero sobre una imagen) | Modo de desplazamiento temporal |
| `Supr` | Elimina el elemento de lista enfocado; también limpia el ROI cuando el puntero está sobre la imagen de origen |

---

## 16. Solución de problemas

### Problemas frecuentes

#### «Sin imágenes cargadas»

- Asegúrate de haber importado antes una carpeta de imágenes o un vídeo
- Comprueba que la carpeta contiene formatos de imagen compatibles

#### Los resultados de fusión se ven borrosos o con fantasmas

- Activa el registro de imágenes (elige uno en la lista: homografía / ECC / OpenFocus rápida)
- Asegúrate de que las imágenes están bien alineadas antes de la fusión
- Prueba otro algoritmo de fusión
- Aumenta el tamaño de núcleo para obtener resultados más suaves

#### IA no disponible

- Comprueba ONNX Runtime y el modelo incluido.
- Compruébalo en **Ayuda → Información del entorno**

#### Errores de memoria insuficiente

- Activa el procesamiento por mosaicos en **Ajustes → Ajustes avanzados → Rendimiento**
- Reduce el tamaño de bloque de mosaico
- Usa tamaños de núcleo menores
- Redimensiona las imágenes a dimensiones más pequeñas

#### El registro falla

- Usa **OpenFocus (rápida)** para movimiento rígido acotado, o activa **Detener en fotograma rechazado** para fallar pronto en lugar de aceptar fotogramas sin alinear
- Reduce el ancho de reducción en **Ajustes → Ajustes avanzados → Registro**
- Asegúrate de que las imágenes tienen suficiente contraste y características
- Activa **Recopilar diagnósticos de alineación**, vuelve a renderizar y consulta **Ajustes → Ajustes avanzados → Registro → Diagnósticos de alineación** para los fotogramas que fallan

#### Procesamiento lento

- Deja los hilos en automático (`0`); la capacidad de CPU, el modo de recursos y el margen de memoria ya limitan la concurrencia
- Usa el modo de recursos Máximo para un lote desatendido, o el filtro guiado / GFG-FGF en lugar de un método de IA
- Reduce la resolución de las imágenes o usa **Vista previa rápida** mientras ajustas

#### Aviso de SmartScreen de Windows en el primer arranque

Que los binarios de Windows estén firmados depende de la versión concreta (consulta las notas de la versión); no se garantiza la firma. Descárgalos de **https://openfocus.top** en lugar de un espejo de terceros. Si SmartScreen avisa, pulsa **Más información → Ejecutar de todas formas**. El primer arranque tras descomprimir puede tardar 30–60 segundos.

#### macOS indica que la aplicación «no se puede abrir» o «está dañada»

La compilación de macOS no está notarizada. Descarga la compilación para Apple Silicon desde **https://openfocus.top**. Si instalaste desde el DMG moviendo `OpenFocus.app` a Aplicaciones, ejecuta:

```bash
xattr -cr /Applications/OpenFocus.app
```

Si instalaste desde el terminal con `install.sh`, la app está en `~/Applications`; ejecuta `xattr -cr ~/Applications/OpenFocus.app`.

o haz clic derecho en la aplicación y elige **Abrir**. Las compilaciones publicadas para macOS son para Apple Silicon (M1/M2/M3/M4); los Mac Intel pueden ejecutarse desde el código fuente.

### Obtener ayuda

1. **Información del entorno**: ve a **Ayuda → Información del entorno** para comprobar tu instalación
2. **Contactar con soporte**: ve a **Ayuda → Contacto** para obtener información de soporte

---

## Apéndice: detalles de los algoritmos de fusión

### Filtro guiado

El filtro guiado realiza un suavizado que preserva los bordes usando una imagen de guía. Calcula transformaciones lineales locales entre la guía y las imágenes de entrada, y las aplica para producir la salida. Este método destaca por preservar los bordes mientras reduce el ruido.

### DCT (transformada discreta del coseno)

La DCT transforma las imágenes al dominio de la frecuencia, donde las decisiones de fusión se basan en la magnitud de los coeficientes. Los componentes de alta frecuencia (detalle) se conservan mientras que los de baja frecuencia (zonas suaves) se combinan según la energía local.

### DTCWT (transformada wavelet compleja de árbol dual)

La DTCWT proporciona una descomposición multiescala y multidireccional de las imágenes. Captura características a distintas escalas y orientaciones, lo que permite reglas de fusión sofisticadas que preservan tanto la textura como la información estructural.

### GFG-FGF

Implementa Fu et al. (2024): el mapa de enfoque grueso |I − mean(I)| se enriquece con el gradiente de cuatro vecinos tipo gaussiano —la plantilla 3×3 T = [1 2 1; 2 4 2; 1 2 1]/16 da P = I⊗T, y G = (|I(x±1,y) − P|)/2 + (|I(x,y±1) − P|)/2, descartando los valores por debajo de T0 = 0,005— y después se refina con un filtro guiado rápido. Cada píxel elige el fotograma con el mayor valor de enfoque (empates: medida de región pequeña de Sobel 3×3), y los mapas de decisión binarios se suavizan con el filtro guiado rápido antes de la mezcla ponderada.

### Fusión por IA (StackMFF V3)

Enfoque de aprendizaje profundo que usa una red neuronal entrenada con grandes conjuntos de pares de imágenes multifoco. La red aprende estrategias de fusión óptimas automáticamente y ofrece resultados de última generación cuando está bien configurada.

---

*OpenFocus es un proyecto de código abierto. Para actualizaciones, informes de errores y solicitudes de funciones, visita el repositorio del proyecto.*
