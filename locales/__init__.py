
import datetime
import os

from PyQt6.QtCore import QObject, pyqtSignal

from .en import EN
from .zh import ZH

class TranslationManager(QObject):
    languageChanged = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.current_lang = 'en'
        
        # Auto-detect from the packs that actually loaded. Register shipped
        # languages in PACK_ORDER; source installations may discover extras.
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
        self._pending_autodetect = detected

        self.translations = {"en": dict(EN), "zh": dict(ZH)}

        self._load_language_packs()
        manual_titles = {
            'en': 'User Manual', 'zh': '用户手册', 'ja': 'ユーザーマニュアル',
            'es': 'Manual de usuario', 'de': 'Benutzerhandbuch', 'fr': "Manuel d'utilisation",
            'ru': 'Руководство пользователя', 'ko': '사용자 설명서', 'pt': 'Manual do usuário',
            'it': 'Manuale utente', 'nl': 'Gebruikershandleiding', 'pl': 'Podręcznik użytkownika',
        }
        for code, title in manual_titles.items():
            if code in self.translations:
                self.translations[code]['menu_github_manual'] = f'{title} (GitHub)'

        support_text = {
            'en': ('Support Us', 'Thank you for supporting OpenFocus! Include your username in the transfer note to receive permanent acknowledgment in the project.', 'Sponsor OpenFocus and help it grow'),
            'zh': ('支持我们', '感谢您支持 OpenFocus！请在转账备注中填写您的用户名，项目将永久鸣谢您。', '欢迎成为 OpenFocus 的赞助商，帮助 OpenFocus 发展'),
            'ja': ('応援する', 'OpenFocus へのご支援ありがとうございます。送金時のメモにユーザー名を記入すると、プロジェクトに永続的に謝辞が掲載されます。', 'OpenFocus の開発をスポンサーとして支援する'),
            'es': ('Apóyanos', '¡Gracias por apoyar OpenFocus! Incluye tu nombre de usuario en la nota de la transferencia para recibir un reconocimiento permanente en el proyecto.', 'Patrocina OpenFocus y ayúdalo a crecer'),
            'de': ('Unterstützen', 'Vielen Dank für Ihre Unterstützung von OpenFocus! Geben Sie Ihren Benutzernamen im Verwendungszweck an, um dauerhaft im Projekt erwähnt zu werden.', 'OpenFocus sponsern und die Entwicklung fördern'),
            'fr': ('Nous soutenir', 'Merci de soutenir OpenFocus ! Indiquez votre nom d’utilisateur dans la note du transfert pour recevoir une mention permanente dans le projet.', 'Parrainez OpenFocus et aidez le projet à grandir'),
            'ru': ('Поддержать нас', 'Спасибо за поддержку OpenFocus! Укажите имя пользователя в комментарии к переводу, чтобы получить постоянную благодарность в проекте.', 'Станьте спонсором и помогите развитию OpenFocus'),
            'ko': ('후원하기', 'OpenFocus를 후원해 주셔서 감사합니다! 송금 메모에 사용자 이름을 적어 주시면 프로젝트에 영구적으로 감사 인사를 남깁니다.', 'OpenFocus의 발전을 후원해 주세요'),
            'pt': ('Apoie-nos', 'Obrigado por apoiar o OpenFocus! Inclua seu nome de usuário na nota da transferência para receber um agradecimento permanente no projeto.', 'Patrocine o OpenFocus e ajude o projeto a crescer'),
            'it': ('Sostienici', 'Grazie per sostenere OpenFocus! Inserisci il tuo nome utente nella nota del trasferimento per ricevere un ringraziamento permanente nel progetto.', 'Sponsorizza OpenFocus e aiutalo a crescere'),
            'nl': ('Steun ons', 'Bedankt voor uw steun aan OpenFocus! Vermeld uw gebruikersnaam in de omschrijving van de overschrijving voor een blijvende vermelding in het project.', 'Sponsor OpenFocus en help het project groeien'),
            'pl': ('Wesprzyj nas', 'Dziękujemy za wsparcie OpenFocus! Podaj nazwę użytkownika w tytule przelewu, aby otrzymać trwałe podziękowanie w projekcie.', 'Sponsoruj OpenFocus i wspieraj jego rozwój'),
        }
        for code, (title, message, sponsor) in support_text.items():
            if code in self.translations:
                self.translations[code].update(support_title=title, support_message=message,
                                               status_sponsor=sponsor)
        labels = {
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
        for code, values in labels.items():
            if code in self.translations:
                self.translations[code].update(dict(zip(
                    ('label_of_radius', 'label_of_smoothing', 'label_of_feather',
                     'label_of_smoothing_a_hint'), values)))

        whole_level_hints = {
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
        for code, text in whole_level_hints.items():
            if code in self.translations:
                self.translations[code]['whole_level_saved_value'] = text

        parameter_help_keys = (
            'parameter_help_tiles', 'parameter_help_registration_width',
            'parameter_help_ai_mode', 'parameter_help_kernel', 'parameter_help_radius',
            'parameter_help_smoothing', 'parameter_help_feather', 'parameter_help_levels',
            'parameter_help_reference',
        )
        parameter_help_examples = {
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
                'Promień 1 daje okno ostrości 3 × 3, promień 2 daje 5 × 5. Większe okno uwzględnia więcej pikseli, ale może pominąć szczegóły przy granicy ostrości.',
                'Mocniejsze wygładzanie ogranicza drobne wahania wyboru ostrości lub wag. Przejścia stają się łagodniejsze, ale małe obszary mogą zlać się z otoczeniem.',
                'OpenFocus B miesza sąsiednie klatki na granicy ostrości. Szersze mieszanie łagodzi łączenie, ale może zmiękczyć detale przy krawędzi.',
                'DTCWT analizuje detale w różnych skalach. Więcej poziomów uwzględnia większe struktury, co może zmienić wynik i zwiększyć ilość obliczeń.',
                'Po wybraniu klatki 2 jako odniesienia klatki 1 i 3 są wyrównywane do jej współrzędnych. Inne odniesienie może zmienić część wspólną i końcowe kadrowanie.',
            ),
        }
        for code, values in parameter_help_examples.items():
            if code in self.translations:
                self.translations[code].update(zip(parameter_help_keys, values))

        upper_bounds = {
            'en': 'at most {size}', 'zh': '最多 {size}', 'ja': '最大 {size}',
            'es': 'como máximo {size}', 'de': 'höchstens {size}',
            'fr': 'au maximum {size}', 'ru': 'не более {size}',
            'ko': '최대 {size}', 'pt': 'no máximo {size}', 'it': 'al massimo {size}',
            'nl': 'maximaal {size}', 'pl': 'co najwyżej {size}',
        }
        for code, text in upper_bounds.items():
            if code in self.translations:
                self.translations[code]['storage_available_upper_bound'] = text

        # Apply the detected language only now: the packs had to exist first,
        # otherwise a locale whose pack failed to load would leave
        # current_lang pointing at a table that is not in the dict.
        pending = getattr(self, "_pending_autodetect", None)
        if pending and pending in self.translations:
            self.current_lang = pending
        self._pending_autodetect = None

    # -- language packs ------------------------------------------------------

    # Ordered so the menu lists languages consistently no matter what the
    # filesystem returns. 'en'/'zh' live in en.py/zh.py; 'ja'/'es' ship in
    # _extra.py; everything else is one <code>.py module in this package.
    PACK_ORDER = ("en", "zh", "ja", "es", "de", "fr", "ru", "ko", "pt", "it",
                  "nl", "pl")
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

    def _load_language_packs(self) -> None:
        """Merge every community pack into self.translations.

        Shipped packs are registered in PACK_ORDER and can be imported from
        archives without source files on disk. Source installations may also
        discover extra two-character modules with a PACK dict. Report broken
        packs without preventing the remaining languages from loading.
        """
        import importlib

        loaded = {"en", "zh"}  # imported at the top of this module

        # _extra.py keeps ja/es in one dict for the original release.
        try:
            from ._extra import EXTRA_TRANSLATIONS
            for lang_code, lang_dict in EXTRA_TRANSLATIONS.items():
                self.translations.setdefault(lang_code, {}).update(lang_dict)
                loaded.add(lang_code)
        except Exception as exc:  # noqa: BLE001 - report, never hide
            print(f"Warning: extra language pack failed to load: {exc}")

        # Known packs work even when modules are stored in a frozen archive.
        candidates = set(self.PACK_ORDER) - loaded
        here = os.path.dirname(os.path.abspath(__file__))
        try:
            files = os.listdir(here)
        except OSError as exc:
            print(f"Warning: cannot scan for language packs: {exc}")
            files = ()
        for fname in files:
            if not fname.endswith(".py"):
                continue
            code = fname[:-len(".py")]
            if not code.startswith("_") and len(code) == 2 and code not in loaded:
                candidates.add(code)
        for code in sorted(candidates):
            try:
                module = importlib.import_module(f".{code}", __name__)
                pack = getattr(module, "PACK", None)
                if not isinstance(pack, dict) or not pack:
                    raise ValueError("module has no non-empty PACK dict")
                self.translations.setdefault(code, {}).update(pack)
                loaded.add(code)
            except Exception as exc:  # noqa: BLE001 - report, never hide
                print(f"Warning: language pack '{code}' failed to load: {exc}")

    def available_languages(self):
        """[(code, display_name)] for every fully loaded pack, menu order."""
        order = {c: i for i, c in enumerate(self.PACK_ORDER)}
        codes = [c for c in self.translations if c not in ("en",)]
        codes.sort(key=lambda c: (order.get(c, 999), c))
        return [("en", self.LANG_NAMES["en"])] + [
            (c, self.LANG_NAMES.get(c, c.upper())) for c in codes]

    def set_language(self, lang_code):
        if lang_code in self.translations:
            self.current_lang = lang_code
            self.languageChanged.emit()

    def get(self, key, default=None):
        val = self.translations.get(self.current_lang, {}).get(key)
        if val is None:
            # Fallback to english
            val = self.translations.get('en', {}).get(key)
        return val if val is not None else (default if default is not None else key)
    
    def t(self, key):
        return self.get(key)

# Global instance
trans = TranslationManager()
