"""Help-dialog bodies for ja/es.

Split out of locales_extra so multi-line HTML never has to survive a
single-line insertion into the big table (that is how an earlier attempt
produced an unterminated string). Imported by locales_extra at load time.
"""

JA = {
    "help_render_body": """<h3>融合アルゴリズム</h3>
<p><b>ガイデッドフィルター</b><br>エッジを保つ融合。半径スライダーでシャープさと滑らかさを調整します。</p>
<p><b>DCT</b><br>ブロック単位の DCT 分散＋一貫性検証。最速で CPU のみ、エッジが鮮明。</p>
<p><b>DTCWT</b><br>双樹複素ウェーブレットによる多スケール・多方向融合。高周波の細部に最適。</p>
<p><b>GFG-FGF</b><br>ガウス類似の四近傍勾配＋高速ガイデッドフィルター（Fu ら, 2024）。顕微鏡画像に好適。</p>
<p><b>StackMFF-V4</b><br>一般的なフォーカススタックで学習したニューラルネット。調整が少なく高品質、GPU があると最速。ドメインが大きく異なる場合は非推奨。</p>
<p><b>OpenFocus A / B / C</b><br><b>A</b> 分散加重平均＋最終フレーム事前分布（高速・色安定）。<b>B</b> 分散深度マップ選択（画素ごとに最も鮮明なフレーム；平滑面・顕微鏡向け）。<b>C</b> ラプラシアンピラミッドの帯域別合成（テクスチャに自然）。A・B は分散ウィンドウ、C は半径なしでスムージングのみ。</p>""",
    "help_registration_body": """<h3>位置合わせ方法</h3>
<p>位置合わせは単一選択です。1 回のレンダーにつき 1 つ、または「位置合わせなし」を選びます。</p>
<p><b>位置合わせなし</b><br>登録を省略。最速で、既に整列済みのスタック（固定撮影）に適します。</p>
<p><b>ホモグラフィー</b><br>特徴量（SIFT + RANSAC）による射影変換。全体的な幾何補正に最適。</p>
<p><b>ECC</b><br>拡張相関係数によるサブピクセル精調整。細かな補正や特徴量が不安定な場合に有効。</p>
<p><b>OpenFocus（高速）</b><br>有界相似変換（並進＋回転＋拡大縮小）＋明度均等化＋自動裁ち落とし。1 フレーム 1 秒未満。フレームサイズの 8% を超える移動のフレームは中途半端な補正をせず位置合わせを中止します。非剛体・極端な幾何には ECC かホモグラフィーを使用してください。</p>""",
    "help_tile_body": """<h3>タイル設定</h3>
<p><b>タイル処理を有効化</b>：大きな画像をブロックに分けて処理し、ピークメモリを抑えます。</p>
<p><b>ブロックサイズ</b>：各タイルの一辺（通常 512〜2048）。</p>
<p><b>オーバーラップ</b>：隣接タイルで共有する画素数。継ぎ目を目立たなくします。</p>
<p><b>しきい値</b>：長辺がこの値を超える画像のみタイル処理し、小さい画像は全体を融合します。</p>""",
}

ES = {
    "help_render_body": """<h3>Métodos de fusión</h3>
<p><b>Filtro guiado</b><br>Fusión que preserva bordes; ajusta el radio para equilibrar nitidez y suavidad.</p>
<p><b>DCT</b><br>Varianza DCT por bloques con verificación de consistencia; el más rápido, solo CPU, bordes nítidos.</p>
<p><b>DTCWT</b><br>Fusión con wavelet compleja de doble árbol en escalas y orientaciones; ideal para contenido de alta frecuencia.</p>
<p><b>GFG-FGF</b><br>Gradiente de cuatro vecinos tipo gaussiano con filtro guiado rápido (Fu et al., 2024); muy adecuado para microscopía.</p>
<p><b>StackMFF-V4</b><br>Red neuronal entrenada con pilas de enfoque habituales. Resultados fuertes con poco ajuste; más rápida con GPU. Evítala si el dominio difiere mucho.</p>
<p><b>OpenFocus A / B / C</b><br><b>A</b> media ponderada por varianza con prior del último fotograma (rápida, color estable). <b>B</b> selección por mapa de profundidad (el fotograma más nítido por píxel; superficies lisas, microscopía). <b>C</b> mezcla por bandas de pirámide laplaciana (natural en texturas). A y B usan ventana de varianza; C no tiene radio, solo suavizado.</p>""",
    "help_registration_body": """<h3>Métodos de registro</h3>
<p>La alineación es una elección única: un método por render, o «Sin alinear» para fusionar tal cual.</p>
<p><b>Sin alinear</b><br>Omite el registro; el más rápido y correcto cuando la pila ya está alineada (cámara fija).</p>
<p><b>Homografía</b><br>Alineación proyectiva basada en características (SIFT + RANSAC); ideal para corrección geométrica global.</p>
<p><b>ECC</b><br>Coeficiente de correlación mejorado a nivel subpíxel; útil para ajustes finos o cuando las características no son fiables.</p>
<p><b>OpenFocus (rápida)</b><br>Similitud acotada (traslación + rotación + escala) con ecualización de brillo y recorte automático; menos de un segundo por fotograma. Los fotogramas con desplazamiento superior al 8% del tamaño se dejan sin alinear en lugar de corregirse a medias. Usa ECC u Homografía para geometría no rígida o extrema.</p>""",
    "help_tile_body": """<h3>Ajustes de bloques</h3>
<p><b>Activar procesamiento por bloques</b>: procesa imágenes grandes en bloques para reducir la memoria máxima.</p>
<p><b>Tamaño de bloque</b>: lado de cada bloque cuadrado (512–2048 es lo habitual).</p>
<p><b>Solape</b>: píxeles compartidos entre bloques adyacentes para que las uniones no se vean.</p>
<p><b>Umbral</b>: solo se dividen las imágenes cuyo lado mayor lo supere; las pequeñas se fusionan enteras.</p>""",
}
