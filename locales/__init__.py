import datetime
import importlib
import os
from importlib.util import find_spec

from PyQt6.QtCore import QObject, pyqtSignal

from .en import EN
from ._retouch_macro import RETOUCH_MACRO_TRANSLATIONS
from .ui_polish_i18n import get_ui_polish_pack


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
        'Account', 'My Downloads', 'Hide Fusion Result', 'Community & Danmaku',
        'Community Storage & Cache', 'Cache size limit (MB)',
        'Current usage: {used_mb} MB / {limit_mb} MB',
        'Show in Folder', 'Download failed: {reason}',
    ),
    'zh': (
        '账户', '我的下载', '隐藏融合结果', '社区与弹幕',
        '社区存储与缓存', '缓存上限（MB）',
        '当前占用：{used_mb} MB / {limit_mb} MB',
        '在文件夹中显示', '下载失败：{reason}',
    ),
    'ja': (
        'アカウント', 'ダウンロード一覧', '合成結果を隠す', 'コミュニティと弾幕',
        'コミュニティ保存先とキャッシュ', 'キャッシュ上限（MB）',
        '現在の使用量：{used_mb} MB / {limit_mb} MB',
        'フォルダーで表示', 'ダウンロードに失敗しました：{reason}',
    ),
    'es': (
        'Cuenta', 'Mis descargas', 'Ocultar resultado de fusión', 'Comunidad y comentarios',
        'Almacenamiento y caché de la comunidad', 'Límite de caché (MB)',
        'Uso actual: {used_mb} MB / {limit_mb} MB',
        'Mostrar en carpeta', 'Error de descarga: {reason}',
    ),
    'de': (
        'Konto', 'Meine Downloads', 'Fusionsergebnis ausblenden', 'Community & Danmaku',
        'Community-Speicher & Cache', 'Cache-Limit (MB)',
        'Aktuelle Belegung: {used_mb} MB / {limit_mb} MB',
        'Im Ordner anzeigen', 'Download fehlgeschlagen: {reason}',
    ),
    'fr': (
        'Compte', 'Mes téléchargements', 'Masquer le résultat de fusion', 'Communauté et commentaires',
        'Stockage et cache de la communauté', 'Limite du cache (Mo)',
        'Utilisation actuelle : {used_mb} Mo / {limit_mb} Mo',
        'Afficher dans le dossier', 'Échec du téléchargement : {reason}',
    ),
    'ru': (
        'Аккаунт', 'Мои загрузки', 'Скрыть результат слияния', 'Сообщество и комментарии',
        'Хранилище и кэш сообщества', 'Лимит кэша (МБ)',
        'Использовано: {used_mb} МБ / {limit_mb} МБ',
        'Показать в папке', 'Ошибка загрузки: {reason}',
    ),
    'ko': (
        '계정', '내 다운로드', '융합 결과 숨기기', '커뮤니티 및 탄막',
        '커뮤니티 저장소 및 캐시', '캐시 용량 제한(MB)',
        '현재 사용량: {used_mb} MB / {limit_mb} MB',
        '폴더에서 보기', '다운로드 실패: {reason}',
    ),
    'pt': (
        'Conta', 'Meus downloads', 'Ocultar resultado da fusão', 'Comunidade e comentários',
        'Armazenamento e cache da comunidade', 'Limite de cache (MB)',
        'Uso atual: {used_mb} MB / {limit_mb} MB',
        'Mostrar na pasta', 'Falha no download: {reason}',
    ),
    'it': (
        'Account', 'I miei download', 'Nascondi risultato di fusione', 'Comunità e commenti',
        'Archiviazione e cache della comunità', 'Limite cache (MB)',
        'Utilizzo attuale: {used_mb} MB / {limit_mb} MB',
        'Mostra nella cartella', 'Download non riuscito: {reason}',
    ),
    'nl': (
        'Account', 'Mijn downloads', 'Fusieresultaat verbergen', 'Community & opmerkingen',
        'Communityopslag en cache', 'Cachelimiet (MB)',
        'Huidig gebruik: {used_mb} MB / {limit_mb} MB',
        'Tonen in map', 'Downloaden mislukt: {reason}',
    ),
    'pl': (
        'Konto', 'Moje pobrania', 'Ukryj wynik łączenia', 'Społeczność i komentarze',
        'Pamięć i pamięć podręczna społeczności', 'Limit pamięci podręcznej (MB)',
        'Bieżące użycie: {used_mb} MB / {limit_mb} MB',
        'Pokaż w folderze', 'Pobieranie nie powiodło się: {reason}',
    ),
}

_ALIGNMENT_CONTROLS_KEYS = (
    'label_align_max_shift_x', 'align_max_shift_x_hint',
    'label_align_max_shift_y', 'align_max_shift_y_hint',
    'label_align_max_rotation', 'align_max_rotation_hint',
    'label_align_max_scale_change', 'align_max_scale_change_hint',
    'label_align_auto_crop', 'align_auto_crop_hint',
    'label_align_interpolation', 'align_interp_hint',
    'align_interp_lanczos', 'align_interp_cubic', 'align_interp_linear',
    'label_align_area', 'btn_align_area', 'btn_align_area_hint',
    'btn_clear_align_area', 'btn_clear_align_area_hint',
    'align_area_badge', 'drag_align_area_hint',
    'btn_show_aligned', 'btn_show_aligned_computing', 'btn_show_aligned_hint',
    'source_aligned_badge', 'show_aligned_failed',
    'alignment_reason_shift_x_limit', 'alignment_reason_shift_y_limit',
    'alignment_reason_rotation_limit', 'alignment_reason_scale_limit',
)

_ALIGNMENT_CONTROLS_TEXT = {
    'en': (
        'Max X Shift:', 'Maximum allowed horizontal shift as a fraction of frame width.',
        'Max Y Shift:', 'Maximum allowed vertical shift as a fraction of frame height.',
        'Max Rotation (°):', 'Maximum allowed per-frame rotation in degrees.',
        'Max Scale Change:', 'Maximum allowed per-frame scale ratio change (e.g. 0.15 = ±15%).',
        'Auto-Crop Black Borders', 'Automatically crop invalid border margins after alignment. Disable to keep the full canvas.',
        'Interpolation:', 'Resampling filter used when warping aligned frames.',
        'Lanczos4 (Sharpest)', 'Bicubic (Balanced)', 'Bilinear (Fastest)',
        'Detection Area:', 'Align Area', 'Restrict alignment transform estimation to a box on the source canvas while warping the full image.',
        'Clear', 'Clear the Alignment Detection Area and estimate alignment on the full frame.',
        'Align Area', 'Drag on the source image to set the Alignment Detection Area',
        'Show Aligned', 'Aligning…', 'Preview aligned source frames before rendering.',
        'Aligned', 'Aligned preview failed: {reason}',
        'Horizontal shift exceeded the configured limit',
        'Vertical shift exceeded the configured limit',
        'Rotation exceeded the configured limit',
        'Scale change exceeded the configured limit',
    ),
    'zh': (
        '水平位移上限：', '允许的最大水平位移比例（占图像宽度）。',
        '垂直位移上限：', '允许的最大垂直位移比例（占图像高度）。',
        '最大旋转角（°）：', '允许的单步最大旋转角度（度）。',
        '最大缩放变化：', '允许的单步最大缩放比例变化（如 0.15 表示 ±15%）。',
        '自动裁剪黑边', '对齐后自动裁剪无效黑边。关闭时保留完整画布尺寸。',
        '重采样插值：', '对齐变换时使用的像素重采样滤波器。',
        'Lanczos4（最锐利）', '双三次 Bicubic（均衡）', '双线性 Bilinear（最快）',
        '配准检测区：', '配准检测区', '仅在源图像选定区域内估计对齐变换（可排除暗角黑边干扰），同时仍对整幅图像进行对齐与融合。',
        '清除', '清除配准检测区域，恢复全图估计对齐。',
        '配准检测区', '在源图像上拖拽框选配准检测区域',
        '预览对齐效果', '对齐预览中…', '渲染前在源图视图中实时预览对齐后的帧。',
        '已对齐预览', '对齐预览失败：{reason}',
        '水平位移超出设定上限',
        '垂直位移超出设定上限',
        '旋转角度超出设定上限',
        '缩放变化超出设定上限',
    ),
    'ja': (
        '水平シフト上限：', 'フレーム幅に対する水平移動の許容上限。',
        '垂直シフト上限：', 'フレーム高さに対する垂直移動の許容上限。',
        '最大回転角 (°)：', 'フレーム間で許容される最大回転角度（度）。',
        '最大倍率変化：', 'フレーム間で許容される最大スケール変化（例: 0.15 = ±15%）。',
        '黒縁を自動トリミング', '位置合わせ後の無効な黒縁を自動トリミングします。無効にすると元のキャンバスサイズを保持します。',
        '補間方式：', '位置合わせ変形時に使用するリサンプリング補間フィルター。',
        'Lanczos4（高精細）', 'バイキュービック（バランス）', 'バイリニア（高速）',
        '検出領域：', '位置合わせ領域', 'ソース画像上の指定枠内のみで変換を推定し、画像全体を変形・合成します。',
        'クリア', '位置合わせ検出領域を解除してフレーム全体で推定します。',
        '検出領域', 'ソース画像上をドラッグして位置合わせ検出領域を指定',
        '位置合わせプレビュー', '位置合わせ中…', 'レンダリング前に位置合わせ済みのソースフレームをプレビューします。',
        '位置合わせ済み', '位置合わせプレビューに失敗しました：{reason}',
        '水平シフトが設定上限を超えました',
        '垂直シフトが設定上限を超えました',
        '回転角が設定上限を超えました',
        '倍率変化が設定上限を超えました',
    ),
    'es': (
        'Desplaz. X máx.:', 'Desplazamiento horizontal máximo permitido como fracción del ancho.',
        'Desplaz. Y máx.:', 'Desplazamiento vertical máximo permitido como fracción del alto.',
        'Rotación máx. (°):', 'Rotación máxima permitida por fotograma en grados.',
        'Cambio escala máx.:', 'Cambio de escala máximo permitido por fotograma (p. ej., 0.15 = ±15 %).',
        'Recortar bordes negros', 'Recorta automáticamente los bordes inválidos tras alinear. Desactívelo para conservar el lienzo completo.',
        'Interpolación:', 'Filtro de remuestreo utilizado al deformar los fotogramas alineados.',
        'Lanczos4 (Más nítido)', 'Bicúbica (Equilibrada)', 'Bilineal (Más rápida)',
        'Área de detección:', 'Área de alineación', 'Limita la estimación del alineamiento a un recuadro en el lienzo fuente mientras transforma la imagen completa.',
        'Limpiar', 'Borra el área de detección de alineación y estima en todo el fotograma.',
        'Área alineación', 'Arrastre sobre la imagen fuente para definir el área de detección',
        'Ver alineado', 'Alineando…', 'Previsualiza los fotogramas fuente alineados antes de renderizar.',
        'Alineado', 'Error en la vista previa alineada: {reason}',
        'El desplazamiento horizontal superó el límite configurado',
        'El desplazamiento vertical superó el límite configurado',
        'La rotación superó el límite configurado',
        'El cambio de escala superó el límite configurado',
    ),
    'de': (
        'Max. X-Versatz:', 'Maximal zulässiger horizontaler Versatz als Anteil der Bildbreite.',
        'Max. Y-Versatz:', 'Maximal zulässiger vertikaler Versatz als Anteil der Bildhöhe.',
        'Max. Drehung (°):', 'Maximal zulässige Drehung pro Bild in Grad.',
        'Max. Skalierung:', 'Maximal zulässige Skalierungsänderung pro Bild (z. B. 0,15 = ±15 %).',
        'Schwarze Ränder zuschneiden', 'Schneidet ungültige Ränder nach der Ausrichtung automatisch zu. Deaktivieren für volle Bildgröße.',
        'Interpolation:', 'Resampling-Filter beim Transformieren ausgerichteter Bilder.',
        'Lanczos4 (Am schärfsten)', 'Bikubisch (Ausgewogen)', 'Bilinear (Am schnellsten)',
        'Erkennungsbereich:', 'Ausrichtbereich', 'Beschränkt die Ausrichtungsschätzung auf ein Rechteck im Quellbild, während das gesamte Bild transformiert wird.',
        'Löschen', 'Erkennungsbereich entfernen und auf dem gesamten Bild schätzen.',
        'Ausrichtbereich', 'Auf dem Quellbild ziehen, um den Ausrichtbereich festzulegen',
        'Ausgerichtet zeigen', 'Ausrichten…', 'Zeigt ausgerichtete Quellbilder bereits vor dem Rendern an.',
        'Ausgerichtet', 'Ausrichtungsvorschau fehlgeschlagen: {reason}',
        'Horizontaler Versatz überschreitet das Limit',
        'Vertikaler Versatz überschreitet das Limit',
        'Drehung überschreitet das Limit',
        'Skalierungsänderung überschreitet das Limit',
    ),
    'fr': (
        'Décalage X max. :', 'Décalage horizontal maximal autorisé en fraction de la largeur.',
        'Décalage Y max. :', 'Décalage vertical maximal autorisé en fraction de la hauteur.',
        'Rotation max. (°) :', 'Rotation maximale autorisée par image en degrés.',
        'Échelle max. :', 'Variation d’échelle maximale autorisée par image (ex. 0,15 = ±15 %).',
        'Recadrer les bords noirs', 'Recadre automatiquement les bords invalides après alignement. Désactivez pour garder le cadre complet.',
        'Interpolation :', 'Filtre de rééchantillonnage utilisé lors de la transformation des images.',
        'Lanczos4 (Plus net)', 'Bicubique (Équilibré)', 'Bilinéaire (Plus rapide)',
        'Zone de détection :', 'Zone d’alignement', 'Limite l’estimation d’alignement à un rectangle sur l’image source tout en transformant toute l’image.',
        'Effacer', 'Efface la zone de détection et estime sur toute l’image.',
        'Zone align.', 'Faites glisser sur l’image source pour définir la zone de détection',
        'Aperçu aligné', 'Alignement…', 'Prévisualise les images sources alignées avant le rendu.',
        'Aligné', 'Échec de l’aperçu aligné : {reason}',
        'Le décalage horizontal a dépassé la limite configurée',
        'Le décalage vertical a dépassé la limite configurée',
        'La rotation a dépassé la limite configurée',
        'Le changement d’échelle a dépassé la limite configurée',
    ),
    'ru': (
        'Макс. сдвиг X:', 'Максимальный горизонтальный сдвиг как доля ширины кадра.',
        'Макс. сдвиг Y:', 'Максимальный вертикальный сдвиг как доля высоты кадра.',
        'Макс. поворот (°):', 'Максимальный допустимый поворот на кадр в градусах.',
        'Макс. масштаб:', 'Максимальное допустимое изменение масштаба (напр. 0.15 = ±15%).',
        'Автообрезка чёрных полей', 'Автоматически обрезать пустые края после выравнивания. Отключите для сохранения полного кадра.',
        'Интерполяция:', 'Фильтр передискретизации при трансформации выровненных кадров.',
        'Lanczos4 (Чётче)', 'Бикубическая (Баланс)', 'Билинейная (Быстрее)',
        'Область оценки:', 'Область выравнивания', 'Ограничивает оценку выравнивания выделенной областью на исходном кадре, трансформируя всё изображение.',
        'Сбросить', 'Очистить область оценки выравнивания и использовать весь кадр.',
        'Область выравн.', 'Выделите область на исходном изображении для оценки выравнивания',
        'Показать выровненные', 'Выравнивание…', 'Предпросмотр выровненных исходных кадров до рендеринга.',
        'Выровнено', 'Ошибка предпросмотра выравнивания: {reason}',
        'Горизонтальный сдвиг превысил заданный предел',
        'Вертикальный сдвиг превысил заданный предел',
        'Поворот превысил заданный предел',
        'Изменение масштаба превысило заданный предел',
    ),
    'ko': (
        '최대 X 이동:', '프레임 너비 대비 허용되는 최대 수평 이동 비율.',
        '최대 Y 이동:', '프레임 높이 대비 허용되는 최대 수직 이동 비율.',
        '최대 회전(°):', '프레임당 허용되는 최대 회전 각도(도).',
        '최대 배율 변화:', '프레임당 허용되는 최대 배율 변화(예: 0.15 = ±15%).',
        '검은 테두리 자동 자르기', '정렬 후 유효하지 않은 테두리를 자동으로 자릅니다. 끄면 전체 캔버스 크기를 유지합니다.',
        '보간 방식:', '정렬된 프레임을 변환할 때 사용할 리샘플링 필터.',
        'Lanczos4 (가장 선명)', '바이큐빅 (균형)', '바이리니어 (가장 빠름)',
        '감지 영역:', '정렬 영역', '원본 캔버스의 지정 영역에서만 정렬 변환을 추정하고 전체 이미지를 변형합니다.',
        '지우기', '정렬 감지 영역을 지우고 전체 프레임에서 추정합니다.',
        '정렬 영역', '원본 이미지에서 드래그하여 정렬 감지 영역 지정',
        '정렬 미리보기', '정렬 중…', '렌더링 전에 정렬된 원본 프레임을 미리 봅니다.',
        '정렬됨', '정렬 미리보기 실패: {reason}',
        '수평 이동이 설정된 한도를 초과했습니다',
        '수직 이동이 설정된 한도를 초과했습니다',
        '회전이 설정된 한도를 초과했습니다',
        '배율 변화가 설정된 한도를 초과했습니다',
    ),
    'pt': (
        'Desloc. X máx.:', 'Deslocamento horizontal máximo permitido como fração da largura.',
        'Desloc. Y máx.:', 'Deslocamento vertical máximo permitido como fração da altura.',
        'Rotação máx. (°):', 'Rotação máxima permitida por quadro em graus.',
        'Escala máx.:', 'Variação máxima de escala permitida por quadro (ex.: 0,15 = ±15%).',
        'Recortar bordas pretas', 'Recorta automaticamente as bordas inválidas após o alinhamento. Desative para manter o quadro inteiro.',
        'Interpolação:', 'Filtro de reamostragem usado ao transformar quadros alinhados.',
        'Lanczos4 (Mais nítido)', 'Bicúbica (Equilibrada)', 'Bilinear (Mais rápida)',
        'Área de detecção:', 'Área de alinhamento', 'Restringe a estimativa de alinhamento a um retângulo na imagem fonte enquanto transforma a imagem inteira.',
        'Limpar', 'Limpa a área de detecção e estima o alinhamento no quadro completo.',
        'Área alinh.', 'Arraste na imagem fonte para definir a área de detecção',
        'Mostrar alinhado', 'Alinhando…', 'Pré-visualize os quadros fonte alinhados antes de renderizar.',
        'Alinhado', 'Falha na pré-visualização alinhada: {reason}',
        'O deslocamento horizontal excedeu o limite configurado',
        'O deslocamento vertical excedeu o limite configurado',
        'A rotação excedeu o limite configurado',
        'A variação de escala excedeu o limite configurado',
    ),
    'it': (
        'Spostam. X max:', 'Spostamento orizzontale massimo consentito come frazione della larghezza.',
        'Spostam. Y max:', 'Spostamento verticale massimo consentito come frazione dell’altezza.',
        'Rotazione max (°):', 'Rotazione massima consentita per fotogramma in gradi.',
        'Scala max:', 'Variazione massima di scala consentita per fotogramma (es. 0,15 = ±15%).',
        'Ritaglia bordi neri', 'Ritaglia automaticamente i bordi non validi dopo l’allineamento. Disattiva per mantenere l’intera tela.',
        'Interpolazione:', 'Filtro di ricampionamento usato per deformare i fotogrammi allineati.',
        'Lanczos4 (Più nitido)', 'Bicubica (Bilanciata)', 'Bilineare (Più veloce)',
        'Area rilevamento:', 'Area allineamento', 'Limita la stima dell’allineamento a un riquadro sull’immagine sorgente trasformando comunque l’intera immagine.',
        'Cancella', 'Rimuove l’area di rilevamento e stima l’allineamento sull’intero fotogramma.',
        'Area allin.', 'Trascina sull’immagine sorgente per definire l’area di rilevamento',
        'Mostra allineato', 'Allineamento…', 'Visualizza in anteprima i fotogrammi sorgente allineati prima del rendering.',
        'Allineato', 'Anteprima allineata non riuscita: {reason}',
        'Lo spostamento orizzontale ha superato il limite impostato',
        'Lo spostamento verticale ha superato il limite impostato',
        'La rotazione ha superato il limite impostato',
        'La variazione di scala ha superato il limite impostato',
    ),
    'nl': (
        'Max. X-verschuiving:', 'Maximaal toegestane horizontale verschuiving als fractie van de breedte.',
        'Max. Y-verschuiving:', 'Maximaal toegestane verticale verschuiving als fractie van de hoogte.',
        'Max. rotatie (°):', 'Maximaal toegestane rotatie per frame in graden.',
        'Max. schaalwijziging:', 'Maximaal toegestane schaalverandering per frame (bijv. 0,15 = ±15%).',
        'Zwarte randen bijsnijden', 'Snijdt ongeldige randen na uitlijning automatisch bij. Schakel uit om het volledige formaat te behouden.',
        'Interpolatie:', 'Resamplingfilter voor het transformeren van uitgelijnde frames.',
        'Lanczos4 (Scherpst)', 'Bikubisch (Gebalanceerd)', 'Bilineair (Snelst)',
        'Detectiegebied:', 'Uitlijngebied', 'Beperkt de uitlijningsschatting tot een kader op het bronbeeld terwijl het hele beeld wordt getransformeerd.',
        'Wissen', 'Wis het uitlijngebied en schat de uitlijning op het volledige beeld.',
        'Uitlijngebied', 'Sleep op het bronbeeld om het uitlijngebied in te stellen',
        'Uitgelijnd tonen', 'Uitlijnen…', 'Bekijk uitgelijnde bronframes alvast vóór het renderen.',
        'Uitgelijnd', 'Uitgelijnd voorbeeld mislukt: {reason}',
        'Horizontale verschuiving overschreed de ingestelde limiet',
        'Verticale verschuiving overschreed de ingestelde limiet',
        'Rotatie overschreed de ingestelde limiet',
        'Schaalwijziging overschreed de ingestelde limiet',
    ),
    'pl': (
        'Maks. przesun. X:', 'Maksymalne dopuszczalne przesunięcie poziome jako ułamek szerokości.',
        'Maks. przesun. Y:', 'Maksymalne dopuszczalne przesunięcie pionowe jako ułamek wysokości.',
        'Maks. obrót (°):', 'Maksymalny dopuszczalny obrót na klatkę w stopniach.',
        'Maks. zmiana skali:', 'Maksymalna dopuszczalna zmiana skali na klatkę (np. 0,15 = ±15%).',
        'Auto-przycinanie krawędzi', 'Automatycznie przycina czarne marginesy po wyrównaniu. Wyłącz, aby zachować pełny rozmiar.',
        'Interpolacja:', 'Filtr próbkowania używany przy przekształcaniu wyrównanych klatek.',
        'Lanczos4 (Najostrzejsza)', 'Bikubiczna (Zrównoważona)', 'Dwuliniowa (Najszybsza)',
        'Obszar detekcji:', 'Obszar wyrównania', 'Ogranicza estymację wyrównania do ramki na obrazie źródłowym przy pełnym przekształceniu obrazu.',
        'Wyczyść', 'Usuwa obszar detekcji wyrównania i estymuje na pełnej klatce.',
        'Obszar wyrówn.', 'Przeciągnij na obrazie źródłowym, aby ustawić obszar detekcji',
        'Pokaż wyrównane', 'Wyrównywanie…', 'Podgląd wyrównanych klatek źródłowych przed renderowaniem.',
        'Wyrównano', 'Podgląd wyrównania nie powiódł się: {reason}',
        'Przesunięcie poziome przekroczyło ustawiony limit',
        'Przesunięcie pionowe przekroczyło ustawiony limit',
        'Obrót przekroczył ustawiony limit',
        'Zmiana skali przekroczyła ustawiony limit',
    ),
}

_DEVELOPER_OPTIONS_KEYS = (
    'developer_options_tab',
    'dev_options_intro',
    'dev_key_section',
    'dev_key_label',
    'dev_key_placeholder',
    'dev_key_show',
    'dev_key_hide',
    'dev_key_verify_btn',
    'dev_key_verifying',
    'dev_key_lock_btn',
    'dev_status_locked',
    'dev_status_unlocked',
    'dev_status_locked_detail',
    'dev_status_unlocked_detail',
    'dev_status_unlocked_expires',
    'dev_key_err_empty',
    'dev_key_err_invalid',
    'dev_key_err_expired',
    'dev_key_err_network',
    'dev_key_err_rate_limited',
    'dev_key_verified_ok',
    'dev_key_locked_msg',
    'dev_features_section',
    'dev_feature_community',
    'dev_feature_community_hint',
    'dev_feature_requires_unlock',
)

_DEVELOPER_OPTIONS_TEXT = {
    'en': (
        'Developer Options',
        'Developer Options control pre-release features that are hidden by default. Unlocking requires a valid developer key issued by the OpenFocus admin platform.',
        'Developer Key',
        'Access key:',
        'Enter admin-issued developer key',
        'Show',
        'Hide',
        'Verify & Unlock',
        'Verifying developer key…',
        'Lock & Clear Key',
        'Locked',
        'Unlocked',
        'No active developer key. In-development features remain hidden.',
        'Key {prefix} ({label}) · Scopes: {features}',
        'Key {prefix} ({label}) · Scopes: {features} · Expires: {expires}',
        'Please enter a developer key.',
        'Developer key is invalid or has been revoked.',
        'Developer key has expired.',
        'Unable to reach the verification server. Check your network and try again.',
        'Too many verification attempts. Please wait and try again.',
        'Developer key verified and unlocked.',
        'Developer key cleared. In-development features have been locked.',
        'Feature Switches',
        'Enable Community features (in development)',
        'Shows Community menus, Gallery & Danmaku settings, fusion showcase, and community sync. Help → Account always remains available.',
        'Verify a developer key with Community access to toggle this feature.',
    ),
    'zh': (
        '开发者选项',
        '开发者选项用于控制默认隐藏的开发中功能。解锁需要 OpenFocus 管理平台颁发的有效开发者密钥。',
        '开发者密钥',
        '访问密钥：',
        '输入管理员颁发的开发者密钥',
        '显示',
        '隐藏',
        '验证并解锁',
        '正在验证开发者密钥…',
        '锁定并清除密钥',
        '已锁定',
        '已解锁',
        '未激活开发者密钥，开发中功能保持隐藏。',
        '密钥 {prefix}（{label}）· 权限：{features}',
        '密钥 {prefix}（{label}）· 权限：{features} · 到期：{expires}',
        '请输入开发者密钥。',
        '开发者密钥无效或已被吊销。',
        '开发者密钥已过期。',
        '无法连接验证服务器，请检查网络后重试。',
        '验证请求过于频繁，请稍后重试。',
        '开发者密钥验证通过，已解锁。',
        '已清除开发者密钥并重新锁定开发中功能。',
        '功能开关',
        '启用社区功能（开发中）',
        '开启后显示社区菜单、画廊与弹幕设置、融合展示窗口及社区同步。“帮助 → 账户”登录入口始终保持可用。',
        '需先验证包含社区权限的开发者密钥方可切换此功能。',
    ),
    'ja': (
        '開発者オプション',
        '開発者オプションでは、既定で非表示の開発中機能を管理します。ロック解除には OpenFocus 管理プラットフォーム発行の有効な開発者キーが必要です。',
        '開発者キー',
        'アクセスキー：',
        '管理者発行の開発者キーを入力',
        '表示',
        '非表示',
        '検証してロック解除',
        '開発者キーを検証中…',
        'ロックしてキーを削除',
        'ロック中',
        'ロック解除済み',
        '有効な開発者キーがありません。開発中機能は非表示のままです。',
        'キー {prefix}（{label}）・権限：{features}',
        'キー {prefix}（{label}）・権限：{features}・有効期限：{expires}',
        '開発者キーを入力してください。',
        '開発者キーが無効か、取り消されています。',
        '開発者キーの有効期限が切れています。',
        '検証サーバーに接続できません。ネットワークを確認して再試行してください。',
        '検証の試行回数が多すぎます。しばらくしてから再試行してください。',
        '開発者キーを検証し、ロックを解除しました。',
        '開発者キーを削除し、開発中機能を再ロックしました。',
        '機能スイッチ',
        'コミュニティ機能を有効にする（開発中）',
        'コミュニティメニュー、ギャラリーと弾幕の設定、合成ショーケース、同期を表示します。「ヘルプ → アカウント」は常に利用できます。',
        'この機能を切り替えるには、コミュニティ権限のある開発者キーを検証してください。',
    ),
    'es': (
        'Opciones de desarrollador',
        'Las opciones de desarrollador controlan funciones en desarrollo ocultas por defecto. Para desbloquearlas se requiere una clave válida emitida por la plataforma de administración de OpenFocus.',
        'Clave de desarrollador',
        'Clave de acceso:',
        'Introduce la clave de desarrollador',
        'Mostrar',
        'Ocultar',
        'Verificar y desbloquear',
        'Verificando clave de desarrollador…',
        'Bloquear y borrar clave',
        'Bloqueado',
        'Desbloqueado',
        'No hay ninguna clave de desarrollador activa. Las funciones en desarrollo siguen ocultas.',
        'Clave {prefix} ({label}) · Permisos: {features}',
        'Clave {prefix} ({label}) · Permisos: {features} · Caduca: {expires}',
        'Introduce una clave de desarrollador.',
        'La clave de desarrollador no es válida o fue revocada.',
        'La clave de desarrollador ha caducado.',
        'No se pudo conectar con el servidor de verificación. Comprueba tu red e inténtalo de nuevo.',
        'Demasiados intentos de verificación. Espera un momento e inténtalo de nuevo.',
        'Clave de desarrollador verificada y desbloqueada.',
        'Clave borrada. Las funciones en desarrollo se han bloqueado.',
        'Interruptores de funciones',
        'Activar funciones de Comunidad (en desarrollo)',
        'Muestra los menús de Comunidad, los ajustes de galería y comentarios, el escaparate de fusión y la sincronización. Ayuda → Cuenta permanece siempre disponible.',
        'Verifica una clave de desarrollador con acceso a Comunidad para cambiar esta opción.',
    ),
    'de': (
        'Entwickleroptionen',
        'Die Entwickleroptionen steuern standardmäßig ausgeblendete Vorabfunktionen. Zum Entsperren ist ein gültiger Entwicklerschlüssel der OpenFocus-Verwaltungsplattform erforderlich.',
        'Entwicklerschlüssel',
        'Zugangsschlüssel:',
        'Vom Administrator ausgestellten Schlüssel eingeben',
        'Anzeigen',
        'Verbergen',
        'Prüfen & Entsperren',
        'Entwicklerschlüssel wird geprüft…',
        'Sperren & Schlüssel löschen',
        'Gesperrt',
        'Entsperrt',
        'Kein aktiver Entwicklerschlüssel. Funktionen in Entwicklung bleiben ausgeblendet.',
        'Schlüssel {prefix} ({label}) · Bereiche: {features}',
        'Schlüssel {prefix} ({label}) · Bereiche: {features} · Läuft ab: {expires}',
        'Bitte gib einen Entwicklerschlüssel ein.',
        'Der Entwicklerschlüssel ist ungültig oder wurde widerrufen.',
        'Der Entwicklerschlüssel ist abgelaufen.',
        'Verifizierungsserver nicht erreichbar. Prüfe die Netzwerkverbindung und versuche es erneut.',
        'Zu viele Prüfversuche. Bitte warte kurz und versuche es erneut.',
        'Entwicklerschlüssel bestätigt und entsperrt.',
        'Entwicklerschlüssel gelöscht. Funktionen in Entwicklung wurden wieder gesperrt.',
        'Funktionsschalter',
        'Community-Funktionen aktivieren (in Entwicklung)',
        'Zeigt Community-Menüs, Galerie- & Danmaku-Einstellungen, Fusions-Showcase und Synchronisierung. Hilfe → Konto bleibt immer verfügbar.',
        'Prüfe einen Entwicklerschlüssel mit Community-Berechtigung, um diese Funktion umzuschalten.',
    ),
    'fr': (
        'Options développeur',
        'Les options développeur contrôlent les fonctionnalités en cours de développement masquées par défaut. Le déverrouillage nécessite une clé développeur valide délivrée par la plateforme d’administration OpenFocus.',
        'Clé développeur',
        'Clé d’accès :',
        'Saisir la clé développeur fournie par l’administrateur',
        'Afficher',
        'Masquer',
        'Vérifier et déverrouiller',
        'Vérification de la clé développeur…',
        'Verrouiller et effacer la clé',
        'Verrouillé',
        'Déverrouillé',
        'Aucune clé développeur active. Les fonctionnalités en développement restent masquées.',
        'Clé {prefix} ({label}) · Portées : {features}',
        'Clé {prefix} ({label}) · Portées : {features} · Expire le : {expires}',
        'Veuillez saisir une clé développeur.',
        'La clé développeur est invalide ou a été révoquée.',
        'La clé développeur a expiré.',
        'Impossible de joindre le serveur de vérification. Vérifiez votre réseau et réessayez.',
        'Trop de tentatives de vérification. Veuillez patienter et réessayer.',
        'Clé développeur vérifiée et déverrouillée.',
        'Clé effacée. Les fonctionnalités en développement ont été verrouillées.',
        'Interrupteurs de fonctionnalités',
        'Activer les fonctionnalités Communauté (en développement)',
        'Affiche les menus Communauté, les réglages Galerie et commentaires, la vitrine de fusion et la synchronisation. Aide → Compte reste toujours disponible.',
        'Vérifiez une clé développeur disposant de l’accès Communauté pour modifier cette option.',
    ),
    'ru': (
        'Параметры разработчика',
        'Параметры разработчика управляют функциями в разработке, скрытыми по умолчанию. Для разблокировки требуется действующий ключ разработчика с платформы администрирования OpenFocus.',
        'Ключ разработчика',
        'Ключ доступа:',
        'Введите выданный администратором ключ разработчика',
        'Показать',
        'Скрыть',
        'Проверить и разблокировать',
        'Проверка ключа разработчика…',
        'Заблокировать и очистить ключ',
        'Заблокировано',
        'Разблокировано',
        'Нет активного ключа разработчика. Функции в разработке остаются скрытыми.',
        'Ключ {prefix} ({label}) · Доступ: {features}',
        'Ключ {prefix} ({label}) · Доступ: {features} · Действует до: {expires}',
        'Введите ключ разработчика.',
        'Ключ разработчика недействителен или отозван.',
        'Срок действия ключа разработчика истёк.',
        'Не удалось связаться с сервером проверки. Проверьте сеть и повторите попытку.',
        'Слишком много попыток проверки. Подождите и повторите попытку.',
        'Ключ разработчика подтверждён и разблокирован.',
        'Ключ очищен. Функции в разработке снова заблокированы.',
        'Переключатели функций',
        'Включить функции сообщества (в разработке)',
        'Показывает меню сообщества, настройки галереи и комментариев, витрину слияния и синхронизацию. «Справка → Аккаунт» остаётся доступным всегда.',
        'Подтвердите ключ разработчика с доступом к сообществу, чтобы переключить эту функцию.',
    ),
    'ko': (
        '개발자 옵션',
        '개발자 옵션은 기본적으로 숨겨진 개발 중 기능을 제어합니다. 잠금을 해제하려면 OpenFocus 관리 플랫폼에서 발급한 유효한 개발자 키가 필요합니다.',
        '개발자 키',
        '액세스 키:',
        '관리자가 발급한 개발자 키 입력',
        '표시',
        '숨기기',
        '확인 및 잠금 해제',
        '개발자 키 확인 중…',
        '잠그고 키 지우기',
        '잠김',
        '잠금 해제됨',
        '활성화된 개발자 키가 없습니다. 개발 중인 기능은 숨겨진 상태로 유지됩니다.',
        '키 {prefix} ({label}) · 권한: {features}',
        '키 {prefix} ({label}) · 권한: {features} · 만료: {expires}',
        '개발자 키를 입력해 주세요.',
        '개발자 키가 유효하지 않거나 취소되었습니다.',
        '개발자 키가 만료되었습니다.',
        '인증 서버에 연결할 수 없습니다. 네트워크를 확인하고 다시 시도해 주세요.',
        '인증 시도가 너무 많습니다. 잠시 후 다시 시도해 주세요.',
        '개발자 키가 확인되어 잠금이 해제되었습니다.',
        '개발자 키를 지우고 개발 중인 기능을 다시 잠갔습니다.',
        '기능 스위치',
        '커뮤니티 기능 사용(개발 중)',
        '커뮤니티 메뉴, 갤러리 및 탄막 설정, 융합 쇼케이스, 커뮤니티 동기화를 표시합니다. 도움말 → 계정은 항상 사용할 수 있습니다.',
        '이 기능을 전환하려면 커뮤니티 권한이 있는 개발자 키를 먼저 확인하세요.',
    ),
    'pt': (
        'Opções de desenvolvedor',
        'As opções de desenvolvedor controlam recursos em desenvolvimento ocultos por padrão. O desbloqueio exige uma chave válida emitida pela plataforma administrativa do OpenFocus.',
        'Chave de desenvolvedor',
        'Chave de acesso:',
        'Digite a chave de desenvolvedor fornecida pelo administrador',
        'Mostrar',
        'Ocultar',
        'Verificar e desbloquear',
        'Verificando chave de desenvolvedor…',
        'Bloquear e limpar chave',
        'Bloqueado',
        'Desbloqueado',
        'Nenhuma chave de desenvolvedor ativa. Recursos em desenvolvimento permanecem ocultos.',
        'Chave {prefix} ({label}) · Escopos: {features}',
        'Chave {prefix} ({label}) · Escopos: {features} · Expira em: {expires}',
        'Digite uma chave de desenvolvedor.',
        'A chave de desenvolvedor é inválida ou foi revogada.',
        'A chave de desenvolvedor expirou.',
        'Não foi possível contatar o servidor de verificação. Verifique sua rede e tente novamente.',
        'Muitas tentativas de verificação. Aguarde e tente novamente.',
        'Chave de desenvolvedor verificada e desbloqueada.',
        'Chave removida. Os recursos em desenvolvimento foram bloqueados.',
        'Chaves de recursos',
        'Ativar recursos da Comunidade (em desenvolvimento)',
        'Exibe os menus da Comunidade, ajustes de galeria e comentários, vitrine de fusão e sincronização. Ajuda → Conta permanece sempre disponível.',
        'Verifique uma chave de desenvolvedor com acesso à Comunidade para alternar este recurso.',
    ),
    'it': (
        'Opzioni sviluppatore',
        'Le opzioni sviluppatore controllano le funzionalità in sviluppo nascoste per impostazione predefinita. Lo sblocco richiede una chiave valida emessa dalla piattaforma di amministrazione OpenFocus.',
        'Chiave sviluppatore',
        'Chiave di accesso:',
        'Inserisci la chiave sviluppatore fornita dall’amministratore',
        'Mostra',
        'Nascondi',
        'Verifica e sblocca',
        'Verifica chiave sviluppatore in corso…',
        'Blocca e cancella chiave',
        'Bloccato',
        'Sbloccato',
        'Nessuna chiave sviluppatore attiva. Le funzionalità in sviluppo restano nascoste.',
        'Chiave {prefix} ({label}) · Ambiti: {features}',
        'Chiave {prefix} ({label}) · Ambiti: {features} · Scadenza: {expires}',
        'Inserisci una chiave sviluppatore.',
        'La chiave sviluppatore non è valida o è stata revocata.',
        'La chiave sviluppatore è scaduta.',
        'Impossibile raggiungere il server di verifica. Controlla la rete e riprova.',
        'Troppi tentativi di verifica. Attendi e riprova.',
        'Chiave sviluppatore verificata e sbloccata.',
        'Chiave cancellata. Le funzionalità in sviluppo sono state bloccate.',
        'Interruttori funzionalità',
        'Abilita funzionalità Community (in sviluppo)',
        'Mostra i menu Community, le impostazioni Galleria e commenti, la vetrina di fusione e la sincronizzazione. Aiuto → Account rimane sempre disponibile.',
        'Verifica una chiave sviluppatore con accesso Community per attivare questa opzione.',
    ),
    'nl': (
        'Ontwikkelaarsopties',
        'Ontwikkelaarsopties beheren functies in ontwikkeling die standaard verborgen zijn. Voor ontgrendeling is een geldige ontwikkelaarssleutel van het OpenFocus-beheerplatform vereist.',
        'Ontwikkelaarssleutel',
        'Toegangssleutel:',
        'Voer de door de beheerder verstrekte sleutel in',
        'Tonen',
        'Verbergen',
        'Verifiëren en ontgrendelen',
        'Ontwikkelaarssleutel verifiëren…',
        'Vergrendelen en sleutel wissen',
        'Vergrendeld',
        'Ontgrendeld',
        'Geen actieve ontwikkelaarssleutel. Functies in ontwikkeling blijven verborgen.',
        'Sleutel {prefix} ({label}) · Bereiken: {features}',
        'Sleutel {prefix} ({label}) · Bereiken: {features} · Verloopt: {expires}',
        'Voer een ontwikkelaarssleutel in.',
        'De ontwikkelaarssleutel is ongeldig of ingetrokken.',
        'De ontwikkelaarssleutel is verlopen.',
        'Kan de verificatieserver niet bereiken. Controleer uw netwerk en probeer het opnieuw.',
        'Te veel verificatiepogingen. Wacht even en probeer het opnieuw.',
        'Ontwikkelaarssleutel geverifieerd en ontgrendeld.',
        'Sleutel gewist. Functies in ontwikkeling zijn weer vergrendeld.',
        'Functieschakelaars',
        'Communityfuncties inschakelen (in ontwikkeling)',
        'Toont Community-menu’s, galerij- en opmerkingsinstellingen, fusieshowcase en synchronisatie. Help → Account blijft altijd beschikbaar.',
        'Verifieer een ontwikkelaarssleutel met Community-toegang om deze functie te schakelen.',
    ),
    'pl': (
        'Opcje programisty',
        'Opcje programisty sterują funkcjami w fazie rozwoju, domyślnie ukrytymi. Odblokowanie wymaga prawidłowego klucza programisty wydanego przez platformę administracyjną OpenFocus.',
        'Klucz programisty',
        'Klucz dostępu:',
        'Wpisz klucz programisty wydany przez administratora',
        'Pokaż',
        'Ukryj',
        'Zweryfikuj i odblokuj',
        'Weryfikowanie klucza programisty…',
        'Zablokuj i usuń klucz',
        'Zablokowano',
        'Odblokowano',
        'Brak aktywnego klucza programisty. Funkcje w fazie rozwoju pozostają ukryte.',
        'Klucz {prefix} ({label}) · Zakresy: {features}',
        'Klucz {prefix} ({label}) · Zakresy: {features} · Wygasa: {expires}',
        'Wpisz klucz programisty.',
        'Klucz programisty jest nieprawidłowy lub został unieważniony.',
        'Klucz programisty wygasł.',
        'Nie można połączyć się z serwerem weryfikacji. Sprawdź sieć i spróbuj ponownie.',
        'Zbyt wiele prób weryfikacji. Poczekaj chwilę i spróbuj ponownie.',
        'Klucz programisty został zweryfikowany i odblokowany.',
        'Usunięto klucz programisty. Funkcje w fazie rozwoju zostały zablokowane.',
        'Przełączniki funkcji',
        'Włącz funkcje Społeczności (w fazie rozwoju)',
        'Wyświetla menu Społeczności, ustawienia galerii i komentarzy, prezentację podczas łączenia oraz synchronizację. Pomoc → Konto pozostaje zawsze dostępne.',
        'Zweryfikuj klucz programisty z uprawnieniem Społeczności, aby przełączyć tę funkcję.',
    ),
}

_GALLERY_PACKS_CACHE = None
_RAW_DEVELOP_PACKS_CACHE = None
_PRO_EXPORTS_PACKS_CACHE = None
_WATCH_TEMPLATES_PACKS_CACHE = None
_LIVE_PREVIEW_PACKS_CACHE = None
_SUBSTACK_PACKS_CACHE = None
_RENDER_QUEUE_PACKS_CACHE = None
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


def _raw_develop_pack(code: str):
    global _RAW_DEVELOP_PACKS_CACHE
    if _RAW_DEVELOP_PACKS_CACHE is None:
        try:
            from .raw_develop_i18n import RAW_DEVELOP_TRANSLATIONS
            _RAW_DEVELOP_PACKS_CACHE = RAW_DEVELOP_TRANSLATIONS
        except Exception:
            _RAW_DEVELOP_PACKS_CACHE = {}
    return _RAW_DEVELOP_PACKS_CACHE.get(code)


def _pro_exports_pack(code: str):
    global _PRO_EXPORTS_PACKS_CACHE
    if _PRO_EXPORTS_PACKS_CACHE is None:
        try:
            from .pro_exports_i18n import PRO_EXPORTS_TRANSLATIONS
            _PRO_EXPORTS_PACKS_CACHE = PRO_EXPORTS_TRANSLATIONS
        except Exception:
            _PRO_EXPORTS_PACKS_CACHE = {}
    return _PRO_EXPORTS_PACKS_CACHE.get(code)


def _watch_templates_pack(code: str):
    global _WATCH_TEMPLATES_PACKS_CACHE
    if _WATCH_TEMPLATES_PACKS_CACHE is None:
        try:
            from .watch_templates_i18n import WATCH_TEMPLATES_TRANSLATIONS
            _WATCH_TEMPLATES_PACKS_CACHE = WATCH_TEMPLATES_TRANSLATIONS
        except Exception:
            _WATCH_TEMPLATES_PACKS_CACHE = {}
    return _WATCH_TEMPLATES_PACKS_CACHE.get(code)


def _live_preview_pack(code: str):
    global _LIVE_PREVIEW_PACKS_CACHE
    if _LIVE_PREVIEW_PACKS_CACHE is None:
        try:
            from .live_preview_i18n import LIVE_PREVIEW_TRANSLATIONS
            _LIVE_PREVIEW_PACKS_CACHE = LIVE_PREVIEW_TRANSLATIONS
        except Exception:
            _LIVE_PREVIEW_PACKS_CACHE = {}
    return _LIVE_PREVIEW_PACKS_CACHE.get(code)


def _substack_pack(code: str):
    global _SUBSTACK_PACKS_CACHE
    if _SUBSTACK_PACKS_CACHE is None:
        try:
            from .substack_i18n import SUBSTACK_TRANSLATIONS
            _SUBSTACK_PACKS_CACHE = SUBSTACK_TRANSLATIONS
        except Exception:
            _SUBSTACK_PACKS_CACHE = {}
    return _SUBSTACK_PACKS_CACHE.get(code)


def _render_queue_pack(code: str):
    global _RENDER_QUEUE_PACKS_CACHE
    if _RENDER_QUEUE_PACKS_CACHE is None:
        try:
            from .render_queue_i18n import RENDER_QUEUE_TRANSLATIONS
            _RENDER_QUEUE_PACKS_CACHE = RENDER_QUEUE_TRANSLATIONS
        except Exception:
            _RENDER_QUEUE_PACKS_CACHE = {}
    return _RENDER_QUEUE_PACKS_CACHE.get(code)


def _export_3d_pack(code: str):
    global _EXPORT_3D_PACKS_CACHE
    if _EXPORT_3D_PACKS_CACHE is None:
        try:
            from .export_3d_i18n import EXPORT_3D_TRANSLATIONS
            _EXPORT_3D_PACKS_CACHE = EXPORT_3D_TRANSLATIONS
        except Exception:
            _EXPORT_3D_PACKS_CACHE = {}
    return _EXPORT_3D_PACKS_CACHE.get(code)


def _context_help_pack(code: str):
    try:
        from .context_help_i18n import _context_help_pack as _load_pack
        return _load_pack(code)
    except Exception:
        return None


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
    align_controls = _ALIGNMENT_CONTROLS_TEXT.get(code)
    if align_controls is not None:
        table.update(zip(_ALIGNMENT_CONTROLS_KEYS, align_controls))
    raw_develop = _raw_develop_pack(code)
    if raw_develop:
        table.update(raw_develop)
    dev_opts = _DEVELOPER_OPTIONS_TEXT.get(code)
    if dev_opts is not None:
        table.update(zip(_DEVELOPER_OPTIONS_KEYS, dev_opts))
    render_queue = _render_queue_pack(code)
    if render_queue:
        table.update(render_queue)
    substack = _substack_pack(code)
    if substack:
        table.update(substack)
    retouch_macro = RETOUCH_MACRO_TRANSLATIONS.get(code)
    if retouch_macro:
        table.update(retouch_macro)
    live_preview = _live_preview_pack(code)
    if live_preview:
        table.update(live_preview)
    watch_templates = _watch_templates_pack(code)
    if watch_templates:
        table.update(watch_templates)
    pro_exports = _pro_exports_pack(code)
    if pro_exports:
        table.update(pro_exports)
    export_3d = _export_3d_pack(code)
    if export_3d:
        table.update(export_3d)
    context_help = _context_help_pack(code)
    if context_help:
        table.update(context_help)
    table.update(get_ui_polish_pack(code))
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
