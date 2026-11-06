"""Curated in-app release summaries, translated before a release is published.

Keep each version's twelve bodies together. The release workflow refuses an
incomplete entry; the same catalog supplies offline notes in desktop bundles.
Older GitHub releases without translations keep their original source text.
"""
import re

LANGUAGES = ("en", "zh", "ja", "es", "de", "fr", "ru", "ko", "pt", "it", "nl", "pl")

# Summaries of the shipped CHANGELOG section, not installation boilerplate.
RELEASE_NOTES = {
    "1.76.0": {
        "en": """- Choose a temporary data directory for imported frames and processing scratch. Large stacks now open with bounded previews; original pixels load on workers only when needed. Rendering, alignment and exports use disk-backed storage and check available memory and temporary disk space.
- Tools → Batch Image Converter supports Bayer/X-Trans CFA DNG, camera-linear LinearRaw, unchanged DNG copies and numbered TIFF pages, with parallel jobs, cancellation and collision-safe output names.
- Try the experimental microscope mosaic workspace: automatic feature alignment, manual coordinates, overlap preview and applied grid placement. Export 8/16-bit TIFF or PNG; large TIFF mosaics are composed and written in tiles.
- Open/save dialogs share an editable address bar. Drop local files or folders into converter and mosaic queues. Delete/Backspace affects the active queue, and import counts and errors remain visible.
- StackMFF V1/V2/V3 now use ONNX Runtime with three bundled ONNX models. V1 and V2 were retrained; their separate license terms are documented. Available providers and memory budgets determine acceleration and tile sizes; CPU fallback remains available. V2 spatial tiling approximates whole-image inference.
- Magnifiers stay centred at image edges and across zoom, pan and high-DPI displays. Hold the left mouse button and use Ctrl+wheel to resize them; the wheel alone changes magnification. Original pixels load asynchronously and stale requests are ignored.
- Improve RAW/DNG color profiles, orientation and metadata handling, grayscale/16-bit alignment, fractional mosaic edges, streamed GIF/TIFF exports and non-ASCII directory paths. Preserve original pixels for rendering and export even when a safe preview is unavailable.
- Settings → Advanced Settings includes default-on pseudonymous usage statistics, with an opt-out. Reports contain only a random installation ID, app version and OS; reporting stops at shutdown. Feedback archives contain bounded recent logs and a diagnostic index. Add PayPal support QR codes to the English and Chinese READMEs.""",
        "zh": """- 可为导入帧和处理临时数据选择存储目录。大图堆栈以有内存上限的预览打开，工作线程仅在需要时读取原始像素。渲染、对齐和导出可使用磁盘存储，并检查可用内存和临时磁盘空间。
- “工具 → 批量图像转换器”支持 Bayer/X-Trans CFA DNG、相机线性 LinearRaw、原样复制 DNG 和编号 TIFF 页面，提供并行任务、取消操作和避免重名覆盖的输出文件名。
- 新增实验性显微镜拼接工作区：支持基于特征的自动对齐、手动坐标、重叠预览和应用网格布局。可导出 8/16 位 TIFF 或 PNG；大型 TIFF 拼接结果采用分块合成和写入。
- 打开和保存对话框共用可编辑地址栏。可将本地文件或文件夹拖入转换器和拼接队列。Delete/Backspace 仅作用于当前队列，导入数量和错误信息清晰可见。
- StackMFF V1/V2/V3 改用 ONNX Runtime，随附三个 ONNX 模型。V1 和 V2 已重新训练，单独的许可条款已有说明。加速和分块大小取决于可用执行提供程序及内存预算，并保留 CPU 回退。V2 空间分块是整图推理的近似。
- 放大镜在图像边缘、缩放、平移和高 DPI 显示器上保持以鼠标为中心。按住鼠标左键并使用 Ctrl+滚轮可调整放大镜尺寸，单独滚动则改变倍率。原始像素异步读取，过期请求会被忽略。
- 改进 RAW/DNG 色彩配置、方向和元数据处理、灰度/16 位对齐、小数坐标拼接边缘、流式 GIF/TIFF 导出及非 ASCII 目录路径。即使无法安全生成预览，渲染和导出仍使用原始像素。
- “设置 → 高级设置”新增默认启用、可关闭的化名使用统计。报告仅包含随机安装 ID、应用版本和操作系统，退出时停止上报。反馈压缩包仅保留有大小限制的近期日志和诊断索引。英文和中文 README 新增 PayPal 支持二维码。""",
        "ja": """- 読み込んだフレームと処理用の一時データの保存先を選べます。大きなスタックはメモリ使用量を制限したプレビューで開き、元の画素は必要なときだけワーカーで読み込みます。レンダリング、位置合わせ、書き出しではディスク上の保存領域を使い、空きメモリと一時ディスク容量を確認します。
- 「ツール → 画像の一括変換」は Bayer/X-Trans CFA DNG、カメラリニアの LinearRaw、DNG の無変更コピー、番号付き TIFF ページに対応。並列処理、キャンセル、名前の衝突を避ける出力も利用できます。
- 実験的な顕微鏡モザイク作業画面を追加しました。特徴点による自動位置合わせ、座標の手動入力、重なりのプレビュー、グリッド配置の適用に対応します。8/16 ビット TIFF または PNG に書き出せ、大きな TIFF はタイル単位で合成・保存します。
- 開く・保存ダイアログに編集可能なアドレスバーを共通化しました。ローカルのファイルやフォルダーを変換・モザイクのキューへドロップできます。Delete/Backspace は選択中のキューだけに作用し、読み込み件数とエラーを表示します。
- StackMFF V1/V2/V3 は同梱の 3 つの ONNX モデルを ONNX Runtime で実行します。V1 と V2 は再学習済みで、個別のライセンス条件を記載しています。実行プロバイダーとメモリ予算に応じて高速化とタイルサイズを決め、CPU への切り替えも維持します。V2 の空間分割は画像全体の推論の近似です。
- 拡大鏡は画像の端、ズーム、移動、高 DPI 表示でもマウスを中心に保ちます。左ボタンを押しながら Ctrl+ホイールで大きさを変更し、ホイールだけで倍率を変更できます。元の画素は非同期で読み込み、古い要求は無視します。
- RAW/DNG の色プロファイル・向き・メタデータ、グレースケール/16 ビットの位置合わせ、モザイク端の小数座標、GIF/TIFF の逐次書き出し、非 ASCII のフォルダーパスを改善。安全なプレビューを作れなくても、処理と書き出しには元の画素を使います。
- 「設定 → 詳細設定」に、既定で有効な仮名の利用統計を追加しました。無効化できます。送信内容はランダムなインストール ID、アプリ版、OS のみで、終了時に停止します。フィードバックには容量を制限した最近のログと診断索引を含めます。英語・中国語 README に PayPal 支援 QR コードを追加しました。""",
        "es": """- Elige un directorio temporal para los fotogramas importados y los datos de procesamiento. Las pilas grandes se abren con vistas previas de memoria limitada; los píxeles originales se cargan en segundo plano cuando hacen falta. El renderizado, la alineación y la exportación usan almacenamiento en disco y comprueban la memoria y el espacio temporal disponibles.
- Herramientas → Conversor de imágenes por lotes admite DNG CFA Bayer/X-Trans, LinearRaw lineal de cámara, copias DNG sin cambios y páginas TIFF numeradas, con tareas paralelas, cancelación y nombres de salida sin colisiones.
- Prueba el espacio experimental de mosaicos de microscopio: alineación automática por características, coordenadas manuales, vista de solapamiento y aplicación de cuadrículas. Exporta TIFF o PNG de 8/16 bits; los mosaicos TIFF grandes se componen y escriben por teselas.
- Los diálogos de apertura y guardado comparten una barra de direcciones editable. Arrastra archivos o carpetas locales a las colas del conversor y de mosaicos. Supr/Retroceso solo actúa sobre la cola activa; se muestran los recuentos y errores de importación.
- StackMFF V1/V2/V3 usa ONNX Runtime con tres modelos ONNX incluidos. V1 y V2 se han reentrenado y sus licencias específicas están documentadas. Los proveedores disponibles y el presupuesto de memoria determinan la aceleración y las teselas, con alternativa por CPU. El teselado espacial de V2 aproxima la inferencia de imagen completa.
- Las lupas se mantienen centradas en el ratón junto a los bordes, al ampliar, desplazar y usar pantallas de alta densidad. Mantén el botón izquierdo y usa Ctrl+rueda para cambiar su tamaño; la rueda sola cambia el aumento. Los píxeles originales se cargan de forma asíncrona y se descartan solicitudes obsoletas.
- Mejoran los perfiles de color, orientación y metadatos RAW/DNG, la alineación en gris/16 bits, los bordes fraccionarios del mosaico, la exportación secuencial GIF/TIFF y las rutas con caracteres no ASCII. El renderizado y la exportación conservan los originales aunque no haya una vista previa segura.
- Ajustes → Ajustes avanzados incluye estadísticas de uso seudónimas, activadas inicialmente y desactivables. Solo se envían un ID de instalación aleatorio, la versión y el SO; los envíos paran al cerrar. Los archivos de comentarios incluyen registros recientes acotados y un índice de diagnóstico. Los README inglés y chino incorporan códigos QR de apoyo por PayPal.""",
        "de": """- Wählen Sie einen temporären Ordner für importierte Bilder und Verarbeitungsdaten. Große Stapel öffnen sich mit speicherbegrenzten Vorschauen; Originalpixel werden bei Bedarf in Hintergrundprozessen geladen. Rendern, Ausrichten und Exportieren nutzen Plattenspeicher und prüfen verfügbaren RAM sowie temporären Speicherplatz.
- Werkzeuge → Stapel-Bildkonverter unterstützt Bayer-/X-Trans-CFA-DNG, kamera-lineares LinearRaw, unveränderte DNG-Kopien und nummerierte TIFF-Seiten, mit parallelen Aufträgen, Abbruch und kollisionsfreien Ausgabenamen.
- Der experimentelle Mikroskop-Mosaikbereich bietet automatische merkmalsbasierte Ausrichtung, manuelle Koordinaten, Überlappungsvorschau und anwendbare Rasteranordnung. Exportieren Sie 8-/16-Bit-TIFF oder PNG; große TIFF-Mosaike werden kachelweise zusammengesetzt und geschrieben.
- Öffnen- und Speichern-Dialoge haben eine gemeinsame bearbeitbare Adressleiste. Lokale Dateien und Ordner lassen sich in Konverter- und Mosaiklisten ziehen. Entf/Rücktaste wirkt nur auf die aktive Liste; Importzahlen und Fehler bleiben sichtbar.
- StackMFF V1/V2/V3 nutzt ONNX Runtime mit drei mitgelieferten ONNX-Modellen. V1 und V2 wurden neu trainiert; ihre gesonderten Lizenzbedingungen sind dokumentiert. Verfügbare Ausführungsanbieter und Speicherbudgets bestimmen Beschleunigung und Kachelgröße; CPU-Fallback bleibt möglich. Die räumliche Kachelung von V2 nähert die Ganzbild-Inferenz an.
- Lupen bleiben an Bildrändern sowie bei Zoom, Verschieben und hoher Pixeldichte am Mauszeiger zentriert. Mit gedrückter linker Maustaste und Strg+Mausrad ändern Sie die Lupengröße; das Rad allein ändert die Vergrößerung. Originalpixel laden asynchron; veraltete Anfragen werden verworfen.
- Verbessert wurden RAW-/DNG-Farbprofile, Orientierung und Metadaten, Graustufen-/16-Bit-Ausrichtung, gebrochene Mosaikkanten, fortlaufender GIF-/TIFF-Export und nicht-ASCII-Ordnerpfade. Auch ohne sichere Vorschau verwenden Rendern und Export die Originalpixel.
- Einstellungen → Erweiterte Einstellungen enthält standardmäßig aktivierte, abschaltbare pseudonyme Nutzungsstatistik. Berichte enthalten nur zufällige Installations-ID, App-Version und Betriebssystem; beim Beenden stoppt die Übermittlung. Feedbackarchive enthalten begrenzte aktuelle Protokolle und einen Diagnoseindex. Englische und chinesische READMEs erhalten PayPal-Unterstützungs-QR-Codes.""",
        "fr": """- Choisissez un répertoire temporaire pour les images importées et les données de travail. Les grandes piles s’ouvrent avec des aperçus à mémoire limitée ; les pixels originaux ne sont chargés en arrière-plan qu’au besoin. Le rendu, l’alignement et l’export utilisent le disque et vérifient la mémoire et l’espace temporaire disponibles.
- Outils → Convertisseur d’images par lots prend en charge les DNG CFA Bayer/X-Trans, LinearRaw linéaire de caméra, les copies DNG inchangées et les pages TIFF numérotées, avec tâches parallèles, annulation et noms de sortie sans collision.
- Essayez l’espace expérimental de mosaïque de microscope : alignement automatique par caractéristiques, coordonnées manuelles, aperçu du chevauchement et application d’une grille. Exportez en TIFF ou PNG 8/16 bits ; les grandes mosaïques TIFF sont assemblées et écrites par tuiles.
- Les fenêtres d’ouverture et d’enregistrement partagent une barre d’adresse modifiable. Déposez des fichiers ou dossiers locaux dans les files du convertisseur et de la mosaïque. Suppr/Retour arrière agit uniquement sur la file active ; les nombres d’images importées et les erreurs restent visibles.
- StackMFF V1/V2/V3 utilise ONNX Runtime avec trois modèles ONNX fournis. V1 et V2 ont été réentraînés ; leurs conditions de licence distinctes sont documentées. Les fournisseurs disponibles et les budgets mémoire déterminent l’accélération et les tuiles, avec repli sur le CPU. Le découpage spatial de V2 est une approximation de l’inférence sur l’image entière.
- Les loupes restent centrées sur la souris aux bords de l’image, pendant le zoom, le déplacement et sur les écrans haute densité. Maintenez le bouton gauche et utilisez Ctrl+molette pour les redimensionner ; la molette seule change le grossissement. Les pixels originaux se chargent de façon asynchrone et les demandes périmées sont ignorées.
- Amélioration des profils couleur, de l’orientation et des métadonnées RAW/DNG, de l’alignement gris/16 bits, des bords fractionnaires de mosaïque, de l’export progressif GIF/TIFF et des chemins non ASCII. Le rendu et l’export gardent les pixels originaux même sans aperçu sûr.
- Paramètres → Paramètres avancés propose des statistiques pseudonymes activées par défaut, avec possibilité de refus. Seuls un identifiant d’installation aléatoire, la version et le système sont transmis ; l’envoi s’arrête à la fermeture. Les archives de retour contiennent des journaux récents de taille limitée et un index de diagnostic. Les README anglais et chinois ajoutent des QR codes de soutien PayPal.""",
        "ru": """- Выбирайте временную папку для импортированных кадров и рабочих данных. Большие стеки открываются с ограниченными по памяти превью; исходные пиксели загружаются фоновыми потоками только по мере необходимости. Рендеринг, выравнивание и экспорт используют дисковое хранилище и проверяют доступную память и временное место на диске.
- «Инструменты → Пакетный конвертер изображений» поддерживает Bayer/X-Trans CFA DNG, камерно-линейный LinearRaw, неизменённые копии DNG и нумерованные страницы TIFF, параллельные задания, отмену и имена файлов без конфликтов.
- Добавлена экспериментальная рабочая область микроскопических мозаик: автоматическое выравнивание по признакам, ручные координаты, просмотр перекрытий и применение сетки. Экспортируйте TIFF или PNG 8/16 бит; большие мозаики TIFF собираются и записываются плитками.
- Диалоги открытия и сохранения получили общую редактируемую адресную строку. Перетаскивайте локальные файлы и папки в очереди конвертера и мозаики. Delete/Backspace действует только на активную очередь; видны количество импортов и ошибки.
- StackMFF V1/V2/V3 работает через ONNX Runtime с тремя включёнными ONNX-моделями. V1 и V2 переобучены, отдельные условия их лицензий описаны в документации. Ускорение и размер плиток зависят от доступных провайдеров и бюджета памяти; сохраняется переход на CPU. Пространственное разбиение V2 приближает вывод для целого изображения.
- Лупы остаются под курсором у краёв изображения, при масштабировании, перемещении и на экранах высокого разрешения. Удерживая левую кнопку, нажмите Ctrl и вращайте колесо для изменения размера; одно колесо меняет увеличение. Исходные пиксели загружаются асинхронно, устаревшие запросы отбрасываются.
- Улучшены цветовые профили, ориентация и метаданные RAW/DNG, выравнивание серых/16-битных кадров, дробные границы мозаик, потоковый экспорт GIF/TIFF и пути с не-ASCII символами. Рендеринг и экспорт используют исходные пиксели даже без безопасного превью.
- В «Настройки → Дополнительные настройки» добавлена включённая по умолчанию псевдонимная статистика с возможностью отказа. Передаются только случайный ID установки, версия и ОС; при выходе отправка прекращается. Архивы обратной связи содержат недавние журналы ограниченного размера и диагностический индекс. В английском и китайском README добавлены QR-коды поддержки через PayPal.""",
        "ko": """- 가져온 프레임과 처리용 임시 데이터의 저장 폴더를 선택할 수 있습니다. 큰 스택은 메모리 사용량을 제한한 미리보기로 열리며, 원본 픽셀은 필요할 때 작업 스레드에서 불러옵니다. 렌더링, 정렬, 내보내기는 디스크 저장소를 사용하고 가용 메모리와 임시 디스크 공간을 확인합니다.
- 도구 → 일괄 이미지 변환기는 Bayer/X-Trans CFA DNG, 카메라 선형 LinearRaw, 변경 없는 DNG 복사, 번호가 붙은 TIFF 페이지를 지원합니다. 병렬 작업, 취소, 이름 충돌을 피하는 출력도 제공합니다.
- 실험적인 현미경 모자이크 작업 공간을 추가했습니다. 특징 기반 자동 정렬, 수동 좌표, 겹침 미리보기, 격자 배치 적용을 지원합니다. 8/16비트 TIFF 또는 PNG로 내보낼 수 있으며 큰 TIFF 모자이크는 타일 단위로 합성하고 저장합니다.
- 열기와 저장 대화상자에 편집 가능한 주소 표시줄을 공통 적용했습니다. 로컬 파일이나 폴더를 변환기 및 모자이크 대기열로 끌어 놓을 수 있습니다. Delete/Backspace는 활성 대기열에만 적용되며 가져온 개수와 오류를 표시합니다.
- StackMFF V1/V2/V3는 함께 제공되는 ONNX 모델 세 개를 ONNX Runtime으로 실행합니다. V1과 V2를 재학습했으며 별도 라이선스 조건을 문서화했습니다. 가용 실행 공급자와 메모리 예산에 따라 가속과 타일 크기를 결정하고 CPU 대체 실행을 유지합니다. V2 공간 타일링은 전체 이미지 추론의 근사입니다.
- 돋보기는 이미지 가장자리, 확대·축소, 이동, 고해상도 화면에서도 마우스를 중심으로 유지됩니다. 왼쪽 버튼을 누른 채 Ctrl+휠로 크기를 조절하고 휠만 돌리면 배율이 바뀝니다. 원본 픽셀은 비동기로 불러오며 오래된 요청은 무시합니다.
- RAW/DNG 색상 프로필·방향·메타데이터, 회색조/16비트 정렬, 모자이크의 소수 좌표 가장자리, GIF/TIFF 순차 내보내기, 비 ASCII 폴더 경로를 개선했습니다. 안전한 미리보기가 없어도 렌더링과 내보내기에는 원본 픽셀을 사용합니다.
- 설정 → 고급 설정에 기본으로 켜지는 가명 사용 통계를 추가했으며 끌 수 있습니다. 무작위 설치 ID, 앱 버전, 운영체제만 전송하고 종료 시 보고를 중단합니다. 피드백 압축 파일에는 크기를 제한한 최근 로그와 진단 색인이 포함됩니다. 영어 및 중국어 README에 PayPal 후원 QR 코드를 추가했습니다.""",
        "pt": """- Escolha uma pasta temporária para os quadros importados e os dados de processamento. Pilhas grandes abrem com prévias de memória limitada; os pixels originais são carregados em segundo plano apenas quando necessários. Renderização, alinhamento e exportação usam armazenamento em disco e verificam a memória e o espaço temporário disponíveis.
- Ferramentas → Conversor de imagens em lote aceita DNG CFA Bayer/X-Trans, LinearRaw linear da câmera, cópias DNG sem alterações e páginas TIFF numeradas, com tarefas paralelas, cancelamento e nomes de saída sem conflitos.
- Experimente o espaço de mosaico de microscópio: alinhamento automático por características, coordenadas manuais, prévia de sobreposição e aplicação de grade. Exporte TIFF ou PNG de 8/16 bits; mosaicos TIFF grandes são compostos e gravados por blocos. O recurso ainda é experimental.
- As janelas de abrir e salvar compartilham uma barra de endereços editável. Arraste arquivos ou pastas locais para as filas do conversor e do mosaico. Delete/Backspace atua somente na fila ativa; contagens e erros de importação ficam visíveis.
- StackMFF V1/V2/V3 usa ONNX Runtime com três modelos ONNX incluídos. V1 e V2 foram retreinados; suas licenças específicas estão documentadas. Os provedores disponíveis e o orçamento de memória determinam a aceleração e o tamanho dos blocos, com alternativa por CPU. O processamento espacial em blocos de V2 aproxima a inferência da imagem inteira.
- As lupas permanecem centradas no mouse nas bordas, ao ampliar, mover e usar telas de alta densidade. Segure o botão esquerdo e use Ctrl+roda para redimensioná-las; apenas a roda altera a ampliação. Os pixels originais carregam de forma assíncrona e pedidos antigos são ignorados.
- Melhorias nos perfis de cor, orientação e metadados RAW/DNG, alinhamento em cinza/16 bits, bordas fracionárias de mosaico, exportação sequencial GIF/TIFF e caminhos não ASCII. Renderização e exportação preservam os pixels originais mesmo sem uma prévia segura.
- Configurações → Configurações avançadas inclui estatísticas de uso pseudônimas ativadas por padrão, com opção de desativar. Os relatórios contêm apenas um ID aleatório da instalação, versão e sistema operacional; o envio para ao encerrar. Arquivos de feedback incluem logs recentes de tamanho limitado e um índice de diagnóstico. Os README em inglês e chinês recebem códigos QR de apoio via PayPal.""",
        "it": """- Scegli una cartella temporanea per i fotogrammi importati e i dati di elaborazione. Le pile grandi si aprono con anteprime a memoria limitata; i pixel originali vengono caricati in background solo quando servono. Rendering, allineamento ed esportazione usano il disco e verificano memoria e spazio temporaneo disponibili.
- Strumenti → Convertitore immagini in serie supporta DNG CFA Bayer/X-Trans, LinearRaw lineare della fotocamera, copie DNG inalterate e pagine TIFF numerate, con attività parallele, annullamento e nomi di uscita senza conflitti.
- Prova l’area sperimentale per mosaici al microscopio: allineamento automatico tramite caratteristiche, coordinate manuali, anteprima della sovrapposizione e applicazione di una griglia. Esporta TIFF o PNG a 8/16 bit; i mosaici TIFF grandi sono composti e scritti a tasselli.
- Le finestre di apertura e salvataggio condividono una barra degli indirizzi modificabile. Trascina file o cartelle locali nelle code del convertitore e del mosaico. Canc/Backspace agisce solo sulla coda attiva; conteggi ed errori di importazione restano visibili.
- StackMFF V1/V2/V3 usa ONNX Runtime con tre modelli ONNX inclusi. V1 e V2 sono stati riaddestrati e le loro licenze specifiche sono documentate. Provider disponibili e budget di memoria determinano accelerazione e dimensione dei tasselli, con ripiego sulla CPU. La suddivisione spaziale di V2 approssima l’inferenza sull’intera immagine.
- Le lenti restano centrate sul mouse ai bordi, durante zoom e spostamento e sugli schermi ad alta densità. Tieni premuto il pulsante sinistro e usa Ctrl+rotella per ridimensionarle; la sola rotella cambia l’ingrandimento. I pixel originali vengono caricati in modo asincrono e le richieste superate sono ignorate.
- Migliorati profili colore, orientamento e metadati RAW/DNG, allineamento in grigio/16 bit, bordi frazionari dei mosaici, esportazione sequenziale GIF/TIFF e percorsi non ASCII. Rendering ed esportazione mantengono i pixel originali anche senza un’anteprima sicura.
- Impostazioni → Impostazioni avanzate include statistiche d’uso pseudonime attive per impostazione iniziale, disattivabili. I rapporti contengono solo ID casuale dell’installazione, versione e sistema operativo; l’invio termina alla chiusura. Gli archivi di feedback contengono log recenti di dimensione limitata e un indice diagnostico. I README inglese e cinese aggiungono codici QR per il sostegno tramite PayPal.""",
        "nl": """- Kies een tijdelijke map voor geïmporteerde beelden en werkgegevens. Grote stapels openen met voorvertoningen met begrensd geheugengebruik; achtergrondtaken laden oorspronkelijke pixels alleen wanneer nodig. Renderen, uitlijnen en exporteren gebruiken schijfopslag en controleren beschikbaar geheugen en tijdelijke schijfruimte.
- Gereedschap → Afbeeldingen in batches converteren ondersteunt Bayer/X-Trans-CFA-DNG, camera-lineaire LinearRaw, ongewijzigde DNG-kopieën en genummerde TIFF-pagina’s, met parallelle taken, annuleren en uitvoernamen zonder conflicten.
- Probeer de experimentele werkruimte voor microscoopmozaïeken: automatische uitlijning op kenmerken, handmatige coördinaten, overlapvoorbeeld en toepasbare rasterplaatsing. Exporteer 8/16-bits TIFF of PNG; grote TIFF-mozaïeken worden per tegel samengesteld en geschreven.
- Open- en opslaandialogen delen een bewerkbare adresbalk. Sleep lokale bestanden of mappen naar de wachtrijen van de converter en het mozaïek. Delete/Backspace werkt alleen op de actieve wachtrij; importaantallen en fouten blijven zichtbaar.
- StackMFF V1/V2/V3 gebruikt ONNX Runtime met drie meegeleverde ONNX-modellen. V1 en V2 zijn opnieuw getraind; de afzonderlijke licentievoorwaarden zijn vastgelegd. Beschikbare uitvoerproviders en geheugenbudgetten bepalen versnelling en tegelgrootte, met terugval op de CPU. Ruimtelijke tegels in V2 benaderen inferentie op het hele beeld.
- Vergrootglazen blijven rond de muis gecentreerd aan beeldranden, bij zoomen, verschuiven en op schermen met hoge pixeldichtheid. Houd de linkermuisknop vast en gebruik Ctrl+wiel om de grootte te wijzigen; het wiel alleen verandert de vergroting. Oorspronkelijke pixels laden asynchroon en verouderde verzoeken worden genegeerd.
- Verbeteringen aan RAW/DNG-kleurprofielen, oriëntatie en metadata, grijswaarden-/16-bits uitlijning, gebroken mozaïekranden, stapsgewijze GIF/TIFF-export en niet-ASCII-mappaden. Renderen en exporteren behouden oorspronkelijke pixels, ook zonder veilige voorvertoning.
- Instellingen → Geavanceerde instellingen bevat standaard ingeschakelde, uitschakelbare pseudonieme gebruiksstatistieken. Rapporten bevatten alleen een willekeurige installatie-ID, appversie en besturingssysteem; verzenden stopt bij afsluiten. Feedbackarchieven bevatten begrensde recente logboeken en een diagnostische index. Engelse en Chinese README’s krijgen PayPal-steun-QR-codes.""",
        "pl": """- Wybierz katalog tymczasowy dla importowanych klatek i danych roboczych. Duże stosy otwierają się z podglądami o ograniczonym zużyciu pamięci; oryginalne piksele są wczytywane w tle tylko w razie potrzeby. Renderowanie, wyrównywanie i eksport korzystają z dysku oraz sprawdzają dostępną pamięć i miejsce tymczasowe.
- Narzędzia → Konwerter obrazów wsadowych obsługuje DNG CFA Bayer/X-Trans, liniowy zapis z aparatu LinearRaw, niezmienione kopie DNG i numerowane strony TIFF, z zadaniami równoległymi, anulowaniem oraz nazwami wyjściowymi bez kolizji.
- Wypróbuj eksperymentalny obszar mozaik mikroskopowych: automatyczne wyrównywanie według cech, ręczne współrzędne, podgląd nakładania i stosowanie siatki. Eksportuj TIFF lub PNG 8/16-bitowy; duże mozaiki TIFF są składane i zapisywane kafelkami.
- Okna otwierania i zapisu mają wspólny edytowalny pasek adresu. Przeciągaj lokalne pliki lub foldery do kolejek konwertera i mozaiki. Delete/Backspace działa wyłącznie na aktywną kolejkę; liczby importów i błędy pozostają widoczne.
- StackMFF V1/V2/V3 używa ONNX Runtime z trzema dołączonymi modelami ONNX. V1 i V2 wytrenowano ponownie, a ich osobne warunki licencyjne opisano w dokumentacji. Dostępni dostawcy wykonania i budżet pamięci określają przyspieszenie i wielkość kafelków; zachowano możliwość przejścia na CPU. Kafelkowanie przestrzenne V2 przybliża wnioskowanie dla całego obrazu.
- Lupy pozostają wyśrodkowane na myszy przy krawędziach obrazu, podczas powiększania, przesuwania i na ekranach o dużej gęstości pikseli. Przytrzymaj lewy przycisk i użyj Ctrl+kółka, aby zmienić rozmiar; samo kółko zmienia powiększenie. Oryginalne piksele są ładowane asynchronicznie, a nieaktualne żądania pomijane.
- Ulepszono profile kolorów, orientację i metadane RAW/DNG, wyrównywanie szarych/16-bitowych klatek, ułamkowe krawędzie mozaik, strumieniowy eksport GIF/TIFF i ścieżki ze znakami spoza ASCII. Renderowanie i eksport zachowują oryginalne piksele nawet bez bezpiecznego podglądu.
- Ustawienia → Ustawienia zaawansowane zawierają domyślnie włączone pseudonimowe statystyki z możliwością wyłączenia. Raporty obejmują tylko losowy ID instalacji, wersję i system operacyjny; wysyłanie kończy się przy zamykaniu. Archiwa opinii zawierają ograniczone ostatnie logi i indeks diagnostyczny. Angielski i chiński README otrzymały kody QR wsparcia przez PayPal.""",
    },
}

_BLOCK = re.compile(
    r"<!-- openfocus-notes:([a-z]{2}) -->\s*(.*?)\s*<!-- /openfocus-notes -->",
    re.S,
)


def localized_bodies(body):
    """Extract explicit language blocks from a GitHub release body."""
    return {lang: text.strip() for lang, text in _BLOCK.findall(body or "")
            if lang in LANGUAGES and text.strip()}


def release_notes_markdown(tag):
    """Generate publishable language blocks, refusing incomplete translations."""
    version = tag.lstrip("vV")
    bodies = RELEASE_NOTES.get(version, {})
    missing = [lang for lang in LANGUAGES if not bodies.get(lang, "").strip()]
    if missing:
        raise ValueError(f"Release {tag} needs translated notes: {', '.join(missing)}")
    return "\n\n".join(
        f"### {lang}\n\n<!-- openfocus-notes:{lang} -->\n{bodies[lang].strip()}\n<!-- /openfocus-notes -->"
        for lang in LANGUAGES
    )
