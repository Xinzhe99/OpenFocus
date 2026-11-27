import datetime
import importlib
import os
from importlib.util import find_spec

from PyQt6.QtCore import QObject, pyqtSignal

from .en import EN


def __getattr__(name):
    """Keep the Chinese table available as a package attribute on demand."""
    if name == 'ZH':
        from .zh import ZH
        return ZH
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


_MANUAL_TITLES = {
    'en': 'User Manual', 'zh': '用户手册', 'ja': 'ユーザーマニュアル',
    'es': 'Manual de usuario', 'de': 'Benutzerhandbuch', 'fr': "Manuel d'utilisation",
    'ru': 'Руководство пользователя', 'ko': '사용자 설명서', 'pt': 'Manual do usuário',
    'it': 'Manuale utente', 'nl': 'Gebruikershandleiding', 'pl': 'Podręcznik użytkownika',
}

_ABOUT_LABELS = {
    'en': ('About OpenFocus', 'License', 'Release Terms (EULA)'),
    'zh': ('关于 OpenFocus', '许可证', '发行版使用条款（EULA）'),
    'ja': ('OpenFocus について', 'ライセンス', 'リリース利用規約（EULA）'),
    'es': ('Acerca de OpenFocus', 'Licencia', 'Condiciones de la versión (EULA)'),
    'de': ('Über OpenFocus', 'Lizenz', 'Nutzungsbedingungen der Ausgabe (EULA)'),
    'fr': ('À propos d’OpenFocus', 'Licence', 'Conditions de la version (EULA)'),
    'ru': ('Об OpenFocus', 'Лицензия', 'Условия выпуска (EULA)'),
    'ko': ('OpenFocus 정보', '라이선스', '배포판 이용 약관(EULA)'),
    'pt': ('Sobre o OpenFocus', 'Licença', 'Termos da versão (EULA)'),
    'it': ('Informazioni su OpenFocus', 'Licenza', 'Condizioni della versione (EULA)'),
    'nl': ('Over OpenFocus', 'Licentie', 'Gebruiksvoorwaarden van de versie (EULA)'),
    'pl': ('O OpenFocus', 'Licencja', 'Warunki wersji (EULA)'),
}

_SUPPORT_KEYS = (
    'support_title', 'support_heading', 'support_message', 'status_credit', 'status_sponsor',
    'support_acknowledgments', 'support_roster_loading', 'support_roster_empty',
    'support_roster_offline',
)

_SUPPORT_TEXT = {
    'en': ('Support Us', 'Support OpenFocus',
           'If you would like a public thank-you, include your display name in the payment note.',
           'Thanks for crediting OpenFocus when you share your work. It helps more people discover us.',
           'Lead sponsor opportunity', 'Supporters', 'Loading supporters…',
           'Supporters who choose to be listed will appear here.', 'The supporter list is temporarily unavailable.'),
    'zh': ('支持我们', '支持 OpenFocus',
           '感谢您支持 OpenFocus。如果愿意公开鸣谢，请在转账备注中写下希望显示的昵称。',
           '感谢您在发布作品时署名 OpenFocus，让更多人认识我们。',
           '首席赞助商正招募中', '赞助者鸣谢', '正在加载赞助者名单…',
           '同意公开鸣谢的赞助者会在这里显示。', '暂时无法加载赞助者名单。'),
    'ja': ('応援する', 'OpenFocus を応援',
           '謝辞への掲載を希望する場合は、送金メモに表示名を記入してください。',
           '作品を公開する際に OpenFocus をクレジットに入れていただけると、より多くの方に知ってもらえます。',
           'メインスポンサー募集中', '支援者への謝辞', '支援者を読み込み中…',
           '掲載に同意した支援者がここに表示されます。', '支援者一覧を一時的に表示できません。'),
    'es': ('Apóyanos', 'Apoya OpenFocus',
           'Si deseas aparecer en los agradecimientos, incluye tu nombre público en la nota del pago.',
           'Gracias por mencionar a OpenFocus al compartir tu trabajo. Así más personas nos conocerán.',
           'Buscamos patrocinador principal', 'Agradecimientos', 'Cargando la lista…',
           'Aquí aparecerán las personas que hayan aceptado figurar en la lista.', 'La lista no está disponible en este momento.'),
    'de': ('Unterstützen', 'OpenFocus unterstützen',
           'Wenn du öffentlich genannt werden möchtest, gib deinen Anzeigenamen im Verwendungszweck an.',
           'Danke, dass du OpenFocus beim Teilen deiner Werke nennst. So lernen uns mehr Menschen kennen.',
           'Hauptsponsor gesucht', 'Dank an Unterstützende', 'Liste wird geladen…',
           'Hier erscheinen Unterstützende, die einer Nennung zugestimmt haben.', 'Die Liste ist vorübergehend nicht verfügbar.'),
    'fr': ('Nous soutenir', 'Soutenir OpenFocus',
           'Pour figurer dans les remerciements publics, indiquez votre nom d’affichage dans la note du paiement.',
           'Merci de citer OpenFocus lors de la publication de vos œuvres. Vous nous aidez à nous faire connaître.',
           'Devenez notre sponsor principal', 'Remerciements', 'Chargement de la liste…',
           'Les personnes qui acceptent d’être mentionnées apparaîtront ici.', 'La liste est temporairement indisponible.'),
    'ru': ('Поддержать нас', 'Поддержать OpenFocus',
           'Чтобы получить публичную благодарность, укажите имя для отображения в комментарии к переводу.',
           'Спасибо, что упоминаете OpenFocus, когда публикуете работы. Так о нас узнают больше людей.',
           'Ищем главного спонсора', 'Благодарности', 'Загрузка списка…',
           'Здесь будут указаны те, кто согласился на публикацию имени.', 'Список временно недоступен.'),
    'ko': ('후원하기', 'OpenFocus 후원하기',
           '감사의 글에 이름을 공개하고 싶다면 송금 메모에 표시할 이름을 적어 주세요.',
           '작품을 공개할 때 OpenFocus를 표기해 주시면 더 많은 분이 저희를 알게 됩니다. 감사합니다.',
           '대표 후원사 모집 중', '후원자에게 감사드립니다', '후원자 목록을 불러오는 중…',
           '이름 공개에 동의한 후원자가 여기에 표시됩니다.', '현재 후원자 목록을 불러올 수 없습니다.'),
    'pt': ('Apoie-nos', 'Apoie o OpenFocus',
           'Se quiser receber um agradecimento público, inclua seu nome de exibição na nota do pagamento.',
           'Obrigado por citar o OpenFocus ao publicar seus trabalhos. Assim mais pessoas nos conhecem.',
           'Buscamos patrocinador principal', 'Agradecimentos', 'Carregando a lista…',
           'Quem aceitar aparecer na lista será mencionado aqui.', 'A lista está temporariamente indisponível.'),
    'it': ('Sostienici', 'Sostieni OpenFocus',
           'Per comparire nei ringraziamenti pubblici, indica il tuo nome da mostrare nella nota del pagamento.',
           'Grazie per citare OpenFocus quando pubblichi i tuoi lavori. Così più persone ci conosceranno.',
           'Cerchiamo uno sponsor principale', 'Ringraziamenti', 'Caricamento dell’elenco…',
           'Qui saranno indicate le persone che hanno acconsentito alla pubblicazione del nome.', 'L’elenco è temporaneamente non disponibile.'),
    'nl': ('Steun ons', 'Steun OpenFocus',
           'Wil je openbaar bedankt worden? Vermeld dan je weergavenaam bij de betaling.',
           'Bedankt dat je OpenFocus vermeldt wanneer je je werk deelt. Zo leren meer mensen ons kennen.',
           'Hoofdsponsor gezocht', 'Dank aan onze sponsors', 'Lijst wordt geladen…',
           'Sponsors die een vermelding willen, verschijnen hier.', 'De sponsorlijst is tijdelijk niet beschikbaar.'),
    'pl': ('Wesprzyj nas', 'Wesprzyj OpenFocus',
           'Aby otrzymać publiczne podziękowanie, podaj nazwę wyświetlaną w tytule przelewu.',
           'Dziękujemy za wzmiankę o OpenFocus przy publikowaniu prac. Dzięki temu więcej osób nas pozna.',
           'Szukamy głównego sponsora', 'Podziękowania', 'Wczytywanie listy…',
           'Tutaj znajdą się osoby, które zgodziły się na publikację swojej nazwy.', 'Lista jest chwilowo niedostępna.'),
}

_SLIDER_LABELS = {
    'en': ('Radius:', 'Smoothing:', 'Feather:', 'OpenFocus A does not currently use smoothing.'),
    'zh': ('半径：', '平滑度：', '羽化：', 'OpenFocus A 目前不使用平滑度。'),
    'ja': ('半径：', '平滑度：', 'ぼかし：', 'OpenFocus A では現在、平滑度は使用されません。'),
    'es': ('Radio:', 'Suavizado:', 'Desvanecimiento:', 'OpenFocus A aún no utiliza el suavizado.'),
    'de': ('Radius:', 'Glättung:', 'Weiche Kante:', 'OpenFocus A verwendet die Glättung derzeit nicht.'),
    'fr': ('Rayon :', 'Lissage :', 'Contour progressif :', 'OpenFocus A n’utilise pas encore le lissage.'),
    'ru': ('Радиус:', 'Сглаживание:', 'Растушёвка:', 'OpenFocus A пока не использует сглаживание.'),
    'ko': ('반경:', '평활화:', '페더:', 'OpenFocus A는 현재 평활화를 사용하지 않습니다.'),
    'pt': ('Raio:', 'Suavização:', 'Difusão:', 'O OpenFocus A ainda não usa a suavização.'),
    'it': ('Raggio:', 'Smussamento:', 'Sfumatura:', 'OpenFocus A non usa ancora lo smussamento.'),
    'nl': ('Straal:', 'Vloeiend maken:', 'Doezelaar:', 'OpenFocus A gebruikt vloeiend maken nog niet.'),
    'pl': ('Promień:', 'Wygładzanie:', 'Wtapianie:', 'OpenFocus A nie używa jeszcze wygładzania.'),
}

_WHOLE_LEVEL_HINTS = {
    'en': 'The field shows the nearest whole level. The saved value {value} is kept until you edit this field.',
    'zh': '此处显示最接近的整数级别。编辑此项前，保留已保存的值 {value}。',
    'ja': '最も近い整数のレベルを表示しています。この欄を編集するまで、保存値 {value} を保持します。',
    'es': 'Se muestra el nivel entero más cercano. Se conserva el valor guardado {value} hasta que edite este campo.',
    'de': 'Das Feld zeigt die nächstgelegene ganze Stufe. Der gespeicherte Wert {value} bleibt erhalten, bis Sie dieses Feld bearbeiten.',
    'fr': 'Ce champ affiche le niveau entier le plus proche. La valeur enregistrée {value} est conservée jusqu’à la modification de ce champ.',
    'ru': 'В поле показан ближайший целый уровень. Сохранённое значение {value} сохраняется до редактирования этого поля.',
    'ko': '가장 가까운 정수 단계로 표시합니다. 이 값을 편집할 때까지 저장된 값 {value}을(를) 유지합니다.',
    'pt': 'O campo mostra o nível inteiro mais próximo. O valor guardado {value} é mantido até editar este campo.',
    'it': 'Il campo mostra il livello intero più vicino. Il valore salvato {value} viene mantenuto finché non si modifica questo campo.',
    'nl': 'Dit veld toont het dichtstbijzijnde gehele niveau. De opgeslagen waarde {value} blijft behouden totdat u dit veld bewerkt.',
    'pl': 'Pole pokazuje najbliższy poziom całkowity. Zapisana wartość {value} zostanie zachowana do czasu edycji tego pola.',
}

_PARAMETER_HELP_KEYS = (
    'parameter_help_tiles', 'parameter_help_registration_width',
    'parameter_help_ai_mode', 'parameter_help_kernel', 'parameter_help_radius',
    'parameter_help_smoothing', 'parameter_help_feather', 'parameter_help_levels',
    'parameter_help_reference',
)

_PARAMETER_HELP_EXAMPLES = {
    'en': (
        'Two 1024 px tiles share the highlighted 256 px strip. OpenFocus blends that strip to soften the seam. Smaller tiles use less memory and can change the result.',
        'Homography and ECC can estimate alignment at 1024 px on a 4096 px frame. The transform still applies to the full working frame. Lower widths are faster, but small features can be harder to match.',
        'Suppose two source pixels have weights of 60% and 40%. ① Soft blend mixes both. ② Hard pick copies the 60% pixel. Try hard pick to avoid mixing at a boundary, or soft blend for smoother transitions.',
        'The center is evaluated using a small or large neighborhood. A wider window makes isolated changes less influential but can wash out tiny features.',
        'Radius 1 covers a 3 × 3 focus window. Radius 2 covers 5 × 5. A larger window pools more evidence but can miss fine detail near a focus boundary.',
        'Higher smoothing can reduce small variations in focus choices or weights. The transition can look less abrupt, but small distinct regions may blend into their surroundings.',
        'Feathering blends neighboring frames across a focus boundary in OpenFocus B. A wider blend softens the seam but can soften fine detail at the edge.',
        'DTCWT analyzes detail at multiple scales. More levels also consider coarser structures, which can change the blend and require more processing.',
        'With frame 2 as the reference, frames 1 and 3 move into frame 2’s coordinate system. Changing the reference can change the overlap and final crop.',
    ),
    'zh': (
        '两个 1024 像素的分块共享高亮的 256 像素区域。OpenFocus 在该区域渐变混合以减弱接缝。较小的分块更省内存，但可能改变融合结果。',
        '对于宽 4096 像素的帧，单应性和 ECC 可在 1024 像素宽度下估计变换，再将变换应用于完整工作帧。估计宽度越小越快，但细小特征可能更难匹配。',
        '例如，两个源像素的权重分别为 60% 和 40%。① 软合成混合两者。② 硬选择直接复制权重为 60% 的像素。硬选择可避免边界混色，软合成可让过渡更平滑。',
        '中心位置的判断参考周围邻域。较大的窗口能减弱孤立变化的影响，也可能抹平很小的细节。',
        '半径为 1 时，清晰度窗口为 3 × 3。半径为 2 时，窗口为 5 × 5。较大的窗口参考更多像素，但也可能漏掉焦点边界附近的细节。',
        '提高平滑度可减小焦点选择或权重的局部波动。过渡可能更平顺，但细小的独立区域也可能融入周围。',
        'OpenFocus B 在焦点边界处混合相邻帧。增大羽化可减弱接缝，但也可能使边缘细节变软。',
        'DTCWT 在多种尺度上分析细节。层数更多时会纳入更粗尺度的结构，可能改变融合结果，也需要更多计算。',
        '以第 2 帧为参考时，第 1 帧和第 3 帧会对齐到第 2 帧的坐标。更换参考帧可能改变重叠范围和最终裁剪。',
    ),
    'ja': (
        '1024 px のタイル 2 枚が、強調表示された 256 px の領域を共有します。OpenFocus はそこで合成して継ぎ目を抑えます。小さいタイルはメモリを節約しますが、結果が変わることがあります。',
        'ホモグラフィーと ECC は幅 4096 px のフレームを 1024 px に縮小して位置を推定します。変換は作業フレーム全体に適用します。幅を小さくすると速くなりますが、細部の照合は難しくなります。',
        '2 つの画素の重みが 60% と 40% の例です。① ソフト合成は両方を混ぜ、② ハード選択は 60% の画素を使います。境界の混色を避けるにはハード、滑らかな遷移にはソフトを試してください。',
        '中央画素を狭い範囲または広い範囲から評価します。範囲が広いと孤立した変化に強くなりますが、細部が失われることがあります。',
        '半径 1 の評価窓は 3 × 3、半径 2 は 5 × 5 です。大きな窓は多くの画素を使いますが、焦点の境界付近の細部を見落とすことがあります。',
        '平滑度を上げると焦点の選択や重みの細かなばらつきが減りますが、小さな領域が周囲に溶け込むことがあります。',
        'OpenFocus B は焦点境界で隣接フレームを混ぜます。範囲を広げると継ぎ目が目立ちにくくなりますが、境界付近の細部も柔らかくなることがあります。',
        'DTCWT は複数のスケールで細部を解析します。レベルを増やすとより粗い構造が加わり、合成結果や処理量が変わります。',
        'フレーム 2 を基準にするとフレーム 1 と 3 がその座標にそろいます。基準を変えると重なりや最終的な切り抜きが変わることがあります。',
    ),
    'es': (
        'Dos bloques de 1024 px comparten la franja de 256 px resaltada. OpenFocus la mezcla para suavizar la unión. Los bloques menores ahorran memoria y pueden cambiar el resultado.',
        'Homografía y ECC pueden estimar la alineación a 1024 px en un fotograma de 4096 px. La transformación se aplica al fotograma de trabajo completo. Un ancho menor acelera el cálculo, pero dificulta reconocer detalles pequeños.',
        'Ejemplo con píxeles ponderados al 60 % y al 40 %. ① La mezcla suave combina ambos. ② La selección dura copia el píxel del 60 %. Prueba la selección dura para evitar mezclas en los bordes o la suave para transiciones graduales.',
        'El centro se evalúa con una vecindad pequeña o grande. Una ventana más amplia reduce la influencia de cambios aislados, pero puede borrar detalles finos.',
        'Un radio de 1 usa una ventana de 3 × 3. Un radio de 2 usa 5 × 5. Una ventana mayor reúne más información, pero puede omitir detalles junto a un límite de enfoque.',
        'Un suavizado mayor reduce variaciones de enfoque o peso. Puede suavizar la transición y diluir regiones pequeñas en su entorno.',
        'OpenFocus B mezcla fotogramas vecinos en el límite de enfoque. Una transición más ancha disimula la unión y puede suavizar los detalles del borde.',
        'DTCWT analiza el detalle a varias escalas. Más niveles incorporan estructuras mayores, lo que puede cambiar la fusión y requerir más cálculo.',
        'Con el fotograma 2 como referencia, el 1 y el 3 se alinean con sus coordenadas. Otra referencia puede cambiar la superposición y el recorte final.',
    ),
    'de': (
        'Zwei 1024 px große Kacheln teilen den markierten 256 px breiten Streifen. OpenFocus überblendet ihn an der Naht. Kleinere Kacheln sparen Speicher und können das Ergebnis verändern.',
        'Homographie und ECC können die Ausrichtung eines 4096 px breiten Bildes bei 1024 px schätzen. Die Transformation gilt für das gesamte Arbeitsbild. Weniger Breite spart Zeit, erschwert aber den Abgleich feiner Merkmale.',
        'Beispiel mit zwei Pixeln und Gewichten von 60 % und 40 %. ① Weiche Mischung verbindet beide. ② Harte Auswahl übernimmt das 60-%-Pixel. Harte Auswahl vermeidet eine Mischung an Kanten, weiche Mischung glättet Übergänge.',
        'Der Mittelpunkt wird anhand seiner Umgebung beurteilt. Ein größeres Fenster dämpft einzelne Änderungen, kann aber feine Details verwischen.',
        'Radius 1 ergibt ein 3 × 3 Fenster, Radius 2 ein 5 × 5 Fenster. Ein größeres Fenster nutzt mehr Bildpunkte, kann aber Details an Fokusgrenzen übersehen.',
        'Stärkere Glättung mindert lokale Schwankungen bei Fokuswahl oder Gewichten. Übergänge werden sanfter, kleine Bereiche können mit ihrer Umgebung verschmelzen.',
        'OpenFocus B überblendet Nachbarbilder an Fokusgrenzen. Eine breitere Überblendung mildert die Naht, kann aber feine Kantendetails weicher zeichnen.',
        'DTCWT analysiert Details auf mehreren Skalen. Mehr Ebenen beziehen gröbere Strukturen ein und können Fusion und Rechenaufwand verändern.',
        'Mit Bild 2 als Referenz werden Bilder 1 und 3 in seinem Koordinatensystem ausgerichtet. Eine andere Referenz kann Überlappung und Beschnitt verändern.',
    ),
    'fr': (
        'Deux tuiles de 1024 px partagent la bande de 256 px en surbrillance. OpenFocus les mélange dans cette bande pour atténuer le raccord. Des tuiles plus petites économisent la mémoire et peuvent changer le résultat.',
        'L’homographie et ECC peuvent estimer l’alignement à 1024 px pour une image de 4096 px. La transformation s’applique à toute l’image de travail. Une largeur moindre accélère le calcul, mais complique la recherche de petits détails.',
        'Deux pixels sources ont des poids de 60 % et 40 %. ① Le mélange doux combine les deux. ② La sélection stricte copie le pixel à 60 %. Essayez la sélection stricte pour éviter un mélange au bord ou le mélange doux pour lisser les transitions.',
        'Le centre est évalué dans un voisinage plus ou moins large. Une grande fenêtre atténue les variations isolées, mais peut effacer les petits détails.',
        'Un rayon de 1 donne une fenêtre de 3 × 3, un rayon de 2 une fenêtre de 5 × 5. Une grande fenêtre utilise plus de pixels, mais peut manquer les détails près d’une limite de mise au point.',
        'Un lissage plus fort réduit les variations locales de choix de mise au point ou de poids. Les transitions sont plus douces, mais de petites zones peuvent se fondre dans leur voisinage.',
        'OpenFocus B mélange les images voisines à la limite de mise au point. Élargir le mélange adoucit le raccord et peut atténuer les détails du bord.',
        'DTCWT analyse les détails à plusieurs échelles. Des niveaux supplémentaires incluent des structures plus larges, ce qui peut modifier la fusion et alourdir le calcul.',
        'Avec l’image 2 comme référence, les images 1 et 3 s’alignent dans son repère. Changer de référence peut modifier le recouvrement et le recadrage final.',
    ),
    'ru': (
        'Два блока по 1024 пикселя имеют общую выделенную полосу в 256 пикселей. OpenFocus смешивает изображения в этой полосе, сглаживая шов. Меньшие блоки экономят память и могут изменить результат.',
        'Гомография и ECC могут оценить выравнивание кадра шириной 4096 пикселей при ширине 1024 пикселя. Преобразование применяется ко всему рабочему кадру. Меньшая ширина ускоряет расчёт, но затрудняет поиск мелких деталей.',
        'У двух пикселей веса 60% и 40%. ① Мягкое смешивание объединяет оба. ② Жёсткий выбор копирует пиксель с весом 60%. Жёсткий вариант избегает смешивания на границе, мягкий сглаживает переходы.',
        'Центр оценивается по его окружению. Широкое окно уменьшает влияние одиночных изменений, но может размыть мелкие детали.',
        'Радиус 1 даёт окно 3 × 3, радиус 2 — окно 5 × 5. Большее окно учитывает больше пикселей, но может пропустить детали на границе фокуса.',
        'Сильнее сглаживание уменьшает локальные колебания выбора фокуса или весов. Переходы становятся мягче, но небольшие участки могут слиться с окружением.',
        'OpenFocus B смешивает соседние кадры на границе фокуса. Более широкое смешивание скрывает шов, но может смягчить детали на краю.',
        'DTCWT анализирует детали на разных масштабах. Дополнительные уровни учитывают более крупные структуры, что может изменить смешивание и увеличить объём вычислений.',
        'Если опорным выбран кадр 2, кадры 1 и 3 выравниваются в его координатах. Другая опора может изменить перекрытие и итоговую обрезку.',
    ),
    'ko': (
        '1024 px 타일 두 개가 강조된 256 px 띠를 공유합니다. OpenFocus는 이 구간을 혼합해 이음새를 줄입니다. 타일이 작으면 메모리를 덜 쓰지만 결과가 달라질 수 있습니다.',
        '호모그래피와 ECC는 너비가 4096 px인 프레임을 1024 px로 줄여 정렬을 추정할 수 있습니다. 변환은 전체 작업 프레임에 적용됩니다. 너비가 작으면 빠르지만 작은 특징을 맞추기 어려울 수 있습니다.',
        '두 픽셀의 가중치가 60%와 40%인 예입니다. ① 소프트 혼합은 둘을 섞습니다. ② 하드 선택은 60% 픽셀을 복사합니다. 경계를 섞지 않으려면 하드 선택, 매끄러운 전환이 필요하면 소프트 혼합을 사용해 보세요.',
        '중심은 주변 영역을 참고해 판단합니다. 창이 넓으면 고립된 변화의 영향은 줄지만 작은 세부가 사라질 수 있습니다.',
        '반경 1은 3 × 3 초점 창, 반경 2는 5 × 5 창을 사용합니다. 넓은 창은 더 많은 픽셀을 고려하지만 초점 경계의 세부를 놓칠 수 있습니다.',
        '평활화가 강하면 초점 선택이나 가중치의 작은 변화가 줄어듭니다. 전환은 부드러워지지만 작은 영역이 주변과 섞일 수 있습니다.',
        'OpenFocus B는 초점 경계에서 인접 프레임을 혼합합니다. 혼합 영역이 넓으면 이음새를 줄이지만 가장자리의 세부도 흐려질 수 있습니다.',
        'DTCWT는 여러 스케일에서 세부를 분석합니다. 레벨을 늘리면 더 큰 구조를 고려하므로 합성 결과가 달라지고 연산량이 늘 수 있습니다.',
        '2번 프레임을 기준으로 설정하면 1번과 3번 프레임이 그 좌표에 맞춰집니다. 기준을 바꾸면 겹치는 영역과 최종 자르기가 달라질 수 있습니다.',
    ),
    'pt': (
        'Dois blocos de 1024 px compartilham a faixa destacada de 256 px. O OpenFocus mistura essa faixa para suavizar a junção. Blocos menores poupam memória e podem mudar o resultado.',
        'Homografia e ECC podem estimar o alinhamento a 1024 px em um quadro de 4096 px. A transformação vale para o quadro de trabalho inteiro. Uma largura menor acelera o cálculo, mas dificulta encontrar detalhes pequenos.',
        'Dois pixels têm pesos de 60% e 40%. ① A mistura suave combina os dois. ② A seleção rígida copia o pixel de 60%. Experimente a seleção rígida para evitar mistura na borda ou a suave para transições graduais.',
        'O centro é avaliado usando a vizinhança. Uma janela maior reduz a influência de mudanças isoladas, mas pode apagar detalhes finos.',
        'Um raio de 1 usa uma janela de foco de 3 × 3. Um raio de 2 usa 5 × 5. Uma janela maior usa mais pixels, mas pode perder detalhes perto do limite de foco.',
        'Mais suavização reduz pequenas variações nas escolhas de foco ou nos pesos. A transição pode suavizar, mas pequenas regiões podem se misturar ao entorno.',
        'O OpenFocus B mescla quadros vizinhos no limite de foco. Uma mistura mais larga suaviza a junção e pode reduzir detalhes na borda.',
        'O DTCWT analisa detalhes em várias escalas. Mais níveis incluem estruturas maiores, o que pode alterar a fusão e exigir mais processamento.',
        'Com o quadro 2 como referência, os quadros 1 e 3 são alinhados às suas coordenadas. Outra referência pode alterar a sobreposição e o recorte final.',
    ),
    'it': (
        'Due riquadri da 1024 px condividono la fascia evidenziata da 256 px. OpenFocus fonde quella fascia per attenuare la giunzione. Riquadri più piccoli risparmiano memoria e possono cambiare il risultato.',
        'Omografia ed ECC possono stimare l’allineamento a 1024 px per un fotogramma da 4096 px. La trasformazione si applica all’intero fotogramma di lavoro. Una larghezza minore accelera il calcolo ma complica la corrispondenza dei dettagli fini.',
        'Due pixel hanno pesi del 60% e del 40%. ① La fusione morbida li combina. ② La selezione netta copia il pixel al 60%. Prova la selezione netta per evitare miscele sui bordi o la fusione morbida per transizioni graduali.',
        'Il centro è valutato considerando il suo intorno. Una finestra più ampia attenua variazioni isolate, ma può cancellare dettagli minuti.',
        'Il raggio 1 usa una finestra di messa a fuoco da 3 × 3, il raggio 2 da 5 × 5. Una finestra più ampia usa più pixel ma può perdere dettagli al confine di messa a fuoco.',
        'Una levigatura più forte riduce le piccole variazioni di messa a fuoco o di peso. La transizione può ammorbidirsi e le regioni piccole possono fondersi con lo sfondo circostante.',
        'OpenFocus B fonde i fotogrammi vicini ai confini di messa a fuoco. Una sfumatura più ampia attenua la giunzione, ma può ammorbidire anche i dettagli del bordo.',
        'DTCWT analizza i dettagli su più scale. Livelli in più considerano strutture più grandi e possono cambiare la fusione e aumentare il calcolo.',
        'Con il fotogramma 2 come riferimento, i fotogrammi 1 e 3 si allineano alle sue coordinate. Un altro riferimento può cambiare la sovrapposizione e il ritaglio finale.',
    ),
    'nl': (
        'Twee tegels van 1024 px delen de gemarkeerde strook van 256 px. OpenFocus mengt de tegels daar voor een vloeiende naad. Kleinere tegels besparen geheugen en kunnen het resultaat wijzigen.',
        'Homografie en ECC kunnen uitlijning op 1024 px schatten voor een beeld van 4096 px. De transformatie geldt voor het volledige werkbeeld. Een kleinere breedte versnelt de berekening maar maakt fijne kenmerken lastiger te vinden.',
        'Twee pixels krijgen een gewicht van 60% en 40%. ① Zacht mengen combineert beide. ② Hard selecteren kopieert de 60%-pixel. Kies hard om menging aan de rand te voorkomen of zacht voor geleidelijke overgangen.',
        'Het midden wordt beoordeeld op basis van de omgeving. Een groter venster dempt afzonderlijke veranderingen maar kan fijne details uitwissen.',
        'Straal 1 gebruikt een scherptevenster van 3 × 3, straal 2 een van 5 × 5. Een groter venster gebruikt meer pixels maar kan details bij de scherptegrens missen.',
        'Meer afvlakking vermindert lokale verschillen in focuskeuze of gewicht. De overgang wordt zachter, maar kleine gebieden kunnen opgaan in hun omgeving.',
        'OpenFocus B mengt naburige beelden aan de scherptegrens. Bredere menging verzacht de naad maar kan details langs de rand vervagen.',
        'DTCWT analyseert details op meerdere schalen. Extra niveaus nemen grovere structuren mee en kunnen de fusie en rekentijd veranderen.',
        'Met beeld 2 als referentie worden beelden 1 en 3 uitgelijnd in zijn coördinaten. Een andere referentie kan de overlap en uiteindelijke uitsnede wijzigen.',
    ),
    'pl': (
        'Dwa kafelki 1024 px współdzielą wyróżniony pas 256 px. OpenFocus miesza je na tym pasie, aby złagodzić łączenie. Mniejsze kafelki oszczędzają pamięć i mogą zmienić wynik.',
        'Homografia i ECC mogą oszacować wyrównanie w szerokości 1024 px dla klatki 4096 px. Przekształcenie dotyczy pełnej klatki roboczej. Niższa szerokość przyspiesza obliczenia, ale utrudnia dopasowanie drobnych cech.',
        'Dwa piksele mają wagi 60% i 40%. ① Miękkie mieszanie łączy oba. ② Twardy wybór kopiuje piksel z wagą 60%. Twardy wybór pozwala uniknąć mieszania na krawędzi, a miękki wygładza przejścia.',
        'Środek ocenia się na podstawie jego otoczenia. Większe okno tłumi pojedyncze zmiany, ale może zatrzeć drobne szczegóły.',
        'Promień 1 daje okno ostrości 3 × 3, promień 2 daje 5 × 5. Większe okno uwzględnia więcej pikseli, ale może pominąć detale przy granicy ostrości.',
        'Mocniejsze wygładzanie ogranicza drobne wahania wyboru ostrości lub wag. Przejścia stają się łagodniejsze, ale małe obszary mogą zlać się z otoczeniem.',
        'OpenFocus B miesza sąsiednie klatki na granicy ostrości. Szersze mieszanie łagodzi łączenie, ale może zmiękczyć detale przy krawędzi.',
        'DTCWT analizuje detale w różnych skalach. Więcej poziomów uwzględnia większe struktury, co może zmienić wynik i zwiększyć ilość obliczeń.',
        'Po wybraniu klatki 2 jako odniesienia klatki 1 i 3 są wyrównywane do jej współrzędnych. Inne odniesienie może zmienić część wspólną i końcowe kadrowanie.',
    ),
}

_UPPER_BOUNDS = {
    'en': 'at most {size}', 'zh': '最多 {size}', 'ja': '最大 {size}',
    'es': 'como máximo {size}', 'de': 'höchstens {size}',
    'fr': 'au maximum {size}', 'ru': 'не более {size}',
    'ko': '최대 {size}', 'pt': 'no máximo {size}', 'it': 'al massimo {size}',
    'nl': 'maximaal {size}', 'pl': 'co najwyżej {size}',
}

_GALLERY_EXTRA_KEYS = (
    'menu_account',
    'menu_my_downloads',
    'gallery_hide_fusion_result',
    'gallery_settings_tab',
    'gallery_settings_storage_section',
    'gallery_settings_cache_limit',
    'gallery_settings_cache_usage',
    'gallery_download_reveal',
    'gallery_download_failed',
)

_GALLERY_EXTRA_TEXT = {
    'en': (
        'Account', 'My Downloads', 'Hide Fusion Result', 'Gallery & Danmaku',
        'Gallery Storage & Cache', 'Cache size limit (MB)',
        'Current usage: {used_mb} MB / {limit_mb} MB',
        'Show in Folder', 'Download failed: {reason}',
    ),
    'zh': (
        '账户', '我的下载', '隐藏融合结果', '画廊与弹幕',
        '画廊存储与缓存', '缓存上限（MB）',
        '当前占用：{used_mb} MB / {limit_mb} MB',
        '在文件夹中显示', '下载失败：{reason}',
    ),
    'ja': (
        'アカウント', 'ダウンロード一覧', '合成結果を隠す', 'ギャラリーと弾幕',
        'ギャラリー保存先とキャッシュ', 'キャッシュ上限（MB）',
        '現在の使用量：{used_mb} MB / {limit_mb} MB',
        'フォルダーで表示', 'ダウンロードに失敗しました：{reason}',
    ),
    'es': (
        'Cuenta', 'Mis descargas', 'Ocultar resultado de fusión', 'Galería y comentarios',
        'Almacenamiento y caché de la galería', 'Límite de caché (MB)',
        'Uso actual: {used_mb} MB / {limit_mb} MB',
        'Mostrar en carpeta', 'Error de descarga: {reason}',
    ),
    'de': (
        'Konto', 'Meine Downloads', 'Fusionsergebnis ausblenden', 'Galerie & Danmaku',
        'Galerie-Speicher & Cache', 'Cache-Limit (MB)',
        'Aktuelle Belegung: {used_mb} MB / {limit_mb} MB',
        'Im Ordner anzeigen', 'Download fehlgeschlagen: {reason}',
    ),
    'fr': (
        'Compte', 'Mes téléchargements', 'Masquer le résultat de fusion', 'Galerie et commentaires',
        'Stockage et cache de la galerie', 'Limite du cache (Mo)',
        'Utilisation actuelle : {used_mb} Mo / {limit_mb} Mo',
        'Afficher dans le dossier', 'Échec du téléchargement : {reason}',
    ),
    'ru': (
        'Аккаунт', 'Мои загрузки', 'Скрыть результат слияния', 'Галерея и комментарии',
        'Хранилище и кэш галереи', 'Лимит кэша (МБ)',
        'Использовано: {used_mb} МБ / {limit_mb} МБ',
        'Показать в папке', 'Ошибка загрузки: {reason}',
    ),
    'ko': (
        '계정', '내 다운로드', '융합 결과 숨기기', '갤러리 및 탄막',
        '갤러리 저장소 및 캐시', '캐시 용량 제한(MB)',
        '현재 사용량: {used_mb} MB / {limit_mb} MB',
        '폴더에서 보기', '다운로드 실패: {reason}',
    ),
    'pt': (
        'Conta', 'Meus downloads', 'Ocultar resultado da fusão', 'Galeria e comentários',
        'Armazenamento e cache da galeria', 'Limite de cache (MB)',
        'Uso atual: {used_mb} MB / {limit_mb} MB',
        'Mostrar na pasta', 'Falha no download: {reason}',
    ),
    'it': (
        'Account', 'I miei download', 'Nascondi risultato di fusione', 'Galleria e commenti',
        'Archiviazione e cache della galleria', 'Limite cache (MB)',
        'Utilizzo attuale: {used_mb} MB / {limit_mb} MB',
        'Mostra nella cartella', 'Download non riuscito: {reason}',
    ),
    'nl': (
        'Account', 'Mijn downloads', 'Fusieresultaat verbergen', 'Galerij & opmerkingen',
        'Galerijopslag en cache', 'Cachelimiet (MB)',
        'Huidig gebruik: {used_mb} MB / {limit_mb} MB',
        'Tonen in map', 'Downloaden mislukt: {reason}',
    ),
    'pl': (
        'Konto', 'Moje pobrania', 'Ukryj wynik łączenia', 'Galeria i komentarze',
        'Pamięć i pamięć podręczna galerii', 'Limit pamięci podręcznej (MB)',
        'Bieżące użycie: {used_mb} MB / {limit_mb} MB',
        'Pokaż w folderze', 'Pobieranie nie powiodło się: {reason}',
    ),
}

_GALLERY_PACKS_CACHE = None
_EXPORT_3D_PACKS_CACHE = None


def _gallery_pack(code: str):
    global _GALLERY_PACKS_CACHE
    if _GALLERY_PACKS_CACHE is None:
        try:
            from .gallery_i18n import GALLERY_TRANSLATIONS
            _GALLERY_PACKS_CACHE = GALLERY_TRANSLATIONS
        except Exception:
            _GALLERY_PACKS_CACHE = {}
    return _GALLERY_PACKS_CACHE.get(code)


def _export_3d_pack(code: str):
    global _EXPORT_3D_PACKS_CACHE
    if _EXPORT_3D_PACKS_CACHE is None:
        try:
            from .export_3d_i18n import EXPORT_3D_TRANSLATIONS
            _EXPORT_3D_PACKS_CACHE = EXPORT_3D_TRANSLATIONS
        except Exception:
            _EXPORT_3D_PACKS_CACHE = {}
    return _EXPORT_3D_PACKS_CACHE.get(code)


def _apply_pack_overlays(code: str, table: dict) -> dict:
    """Apply runtime overrides to a loaded pack without touching other packs."""
    title = _MANUAL_TITLES.get(code)
    if title is not None:
        table['menu_github_manual'] = f'{title} (GitHub)'
    about = _ABOUT_LABELS.get(code)
    if about is not None:
        table.update(zip(('menu_about', 'about_license', 'about_eula'), about))
    support = _SUPPORT_TEXT.get(code)
    if support is not None:
        table.update(zip(_SUPPORT_KEYS, support))
        table['support_alipay'] = '支付宝 Alipay' if code == 'zh' else 'Alipay'
    labels = _SLIDER_LABELS.get(code)
    if labels is not None:
        table.update(dict(zip(
            ('label_of_radius', 'label_of_smoothing', 'label_of_feather',
             'label_of_smoothing_a_hint'), labels)))
    hint = _WHOLE_LEVEL_HINTS.get(code)
    if hint is not None:
        table['whole_level_saved_value'] = hint
    examples = _PARAMETER_HELP_EXAMPLES.get(code)
    if examples is not None:
        table.update(zip(_PARAMETER_HELP_KEYS, examples))
    bound = _UPPER_BOUNDS.get(code)
    if bound is not None:
        table['storage_available_upper_bound'] = bound
    gallery = _gallery_pack(code)
    if gallery:
        table.update(gallery)
    gallery_extra = _GALLERY_EXTRA_TEXT.get(code)
    if gallery_extra is not None:
        table.update(zip(_GALLERY_EXTRA_KEYS, gallery_extra))
    export_3d = _export_3d_pack(code)
    if export_3d:
        table.update(export_3d)
    return table


class TranslationManager(QObject):
    languageChanged = pyqtSignal()

    def __init__(self):
        super().__init__()

        # Auto-detect from the shipped and source language packs.
        # The legacy UTC+8 fallback stays for hosts where QLocale reports an
        # unusable value (some stripped Linux images answer 'C').
        detected = None
        try:
            from PyQt6.QtCore import QLocale
            name = QLocale.system().name() or ""  # e.g. 'zh_CN', 'pt_BR'
            detected = self._match_locale(name)
        except Exception:
            detected = None
        if detected is None:
            try:
                offset = datetime.datetime.now().astimezone().utcoffset()
                if offset is not None and int(offset.total_seconds()) == 28800:
                    detected = 'zh'
            except Exception:
                pass

        # Defer non-English packs until requested so startup does not parse
        # and overlay all 12 translation tables up front.
        self._translations = {"en": _apply_pack_overlays("en", dict(EN))}
        self._unavailable_packs = set()
        self._extra_attempted = False
        self._all_loaded = False

        self._current_lang = detected if self.has_language(detected) else 'en'

    @property
    def current_lang(self):
        return self._current_lang

    @current_lang.setter
    def current_lang(self, lang_code):
        # Render workers set the language without emitting a UI signal.
        self.has_language(lang_code)
        self._current_lang = lang_code

    @property
    def translations(self):
        self._ensure_all_packs_loaded()
        return self._translations

    @translations.setter
    def translations(self, value):
        self._translations = value
        self._all_loaded = True

    # -- language packs ------------------------------------------------------

    # Ordered so the menu lists languages consistently no matter what the
    # filesystem returns. 'en'/'zh' live in en.py/zh.py; 'ja'/'es' ship in
    # _extra.py; everything else is one <code>.py module in this package.
    PACK_ORDER = ("en", "zh", "es", "de", "fr", "ru", "ko", "pt", "it", "nl",
                  "pl", "ja")
    LANG_NAMES = {
        "en": "English", "zh": "中文", "ja": "日本語", "es": "Español",
        "de": "Deutsch", "fr": "Français", "ru": "Русский",
        "ko": "한국어", "pt": "Português", "it": "Italiano",
        "nl": "Nederlands", "pl": "Polski",
    }

    @staticmethod
    def _match_locale(name: str):
        """Map a QLocale name ('pt_BR', 'zh_Hans_CN') onto a pack code."""
        if not name:
            return None
        low = name.lower().replace("-", "_")
        primary = low.split("_")[0]
        # QLocale answers 'C'/'POSIX' when no user locale is configured; there
        # is no such pack, and treating it as a code would blank the UI.
        if primary in ("c", "posix"):
            return None
        # QLocale marks simplified Chinese as 'zh_Hans_CN'; both scripts use
        # the same zh pack, which is Simplified.
        if primary in ("zh", "ja", "ko"):
            return primary
        # Portuguese: one pack covers both, labelled pt-BR in the source text.
        if primary == "pt":
            return "pt"
        return primary or None

    def _load_extra_packs(self) -> None:
        if self._extra_attempted:
            return
        self._extra_attempted = True
        try:
            from ._extra import EXTRA_TRANSLATIONS
            for lang_code, lang_dict in EXTRA_TRANSLATIONS.items():
                merged = self._translations.setdefault(lang_code, {})
                merged.update(lang_dict)
                _apply_pack_overlays(lang_code, merged)
        except Exception as exc:  # noqa: BLE001 - report, never hide
            print(f"Warning: extra language pack failed to load: {exc}")

    def _scan_extra_pack_codes(self, warn_on_error: bool = False):
        here = os.path.dirname(os.path.abspath(__file__))
        try:
            files = os.listdir(here)
        except FileNotFoundError:
            return set()
        except OSError as exc:
            if warn_on_error:
                print(f"Warning: cannot scan for language packs: {exc}")
            return set()
        extras = set()
        for fname in files:
            if not fname.endswith(".py"):
                continue
            code = fname[:-len(".py")]
            if not code.startswith("_") and len(code) == 2:
                extras.add(code)
        return extras

    def _load_pack(self, code: str) -> None:
        if code in self._translations or code in self._unavailable_packs:
            return
        if code in ('ja', 'es'):
            self._load_extra_packs()
            if code in self._translations:
                return
        try:
            if code == 'zh':
                # A static import keeps the Chinese pack in frozen builds.
                from .zh import ZH
                pack = ZH
            else:
                module = importlib.import_module(f'.{code}', __name__)
                pack = getattr(module, 'PACK', None)
            if not isinstance(pack, dict) or not pack:
                raise ValueError('module has no non-empty language dictionary')
            self._translations[code] = _apply_pack_overlays(code, dict(pack))
        except Exception as exc:  # noqa: BLE001 - report, never hide
            self._unavailable_packs.add(code)
            print(f"Warning: language pack '{code}' failed to load: {exc}")

    def _ensure_all_packs_loaded(self) -> None:
        if self._all_loaded:
            return
        self._load_pack('zh')
        self._load_extra_packs()
        candidates = set(self.PACK_ORDER) | self._scan_extra_pack_codes(warn_on_error=True)
        for code in sorted(candidates):
            self._load_pack(code)
        self._all_loaded = True

    def has_language(self, code):
        """Check that a language can be used, loading its pack on demand."""
        if code in self._translations:
            return True
        if not code or self._all_loaded or code in self._unavailable_packs:
            return False
        if code in self.PACK_ORDER or code in self._scan_extra_pack_codes():
            self._load_pack(code)
        return code in self._translations

    def _language_labels(self, codes):
        shipped = [code for code in self.PACK_ORDER if code == 'en' or code in codes]
        extras = sorted(codes - set(self.PACK_ORDER))
        return [(code, self.LANG_NAMES.get(code, code.upper()))
                for code in (*shipped, *extras)]

    def menu_languages(self):
        """List installed languages without importing unused shipped packs."""
        if self._all_loaded:
            return self.available_languages()
        codes = set(self._translations)
        for code in self.PACK_ORDER:
            if code in codes or code in self._unavailable_packs:
                continue
            if (code in ('ja', 'es') and not self._extra_attempted
                    and find_spec('._extra', __name__) is not None):
                codes.add(code)
            elif find_spec(f'.{code}', __name__) is not None:
                codes.add(code)
        for code in sorted(self._scan_extra_pack_codes() - set(self.PACK_ORDER)):
            self._load_pack(code)
            if code in self._translations:
                codes.add(code)
        return self._language_labels(codes)

    def available_languages(self):
        """Return the labels of the packs that loaded successfully, in menu order."""
        self._ensure_all_packs_loaded()
        return self._language_labels(self._translations.keys())

    def set_language(self, lang_code):
        if self.has_language(lang_code):
            self._current_lang = lang_code
            self.languageChanged.emit()

    def get(self, key, default=None):
        val = self._translations.get(self.current_lang, {}).get(key)
        if val is None:
            # Fallback to english
            val = self._translations.get('en', {}).get(key)
        return val if val is not None else (default if default is not None else key)

    def t(self, key):
        return self.get(key)


# Global instance
trans = TranslationManager()
