# Manual do usuário do OpenFocus

## Sumário

1. [Introdução](#1-introduction)
2. [Primeiros passos](#2-getting-started)
3. [Visão geral da interface principal](#3-main-interface-overview)
4. [Importação de imagens e vídeo](#4-importing-images-and-video)
5. [Visualização e navegação pelas imagens](#5-viewing-and-navigating-images)
6. [Visualização de comparação Cortina](#6-wipe-compare-view)
7. [Registro de imagens](#7-image-registration)
8. [Métodos de fusão de imagens](#8-image-fusion-methods)
9. [Transformações de imagem](#9-image-transformations)
10. [Retoque, rótulos e anotações](#10-retouching-labels-and-annotations)
11. [Configurações](#11-settings-and-configuration)
12. [Processamento em lote](#12-batch-processing)
13. [Exportação de resultados](#13-exporting-results)
14. [Interface de linha de comando](#14-command-line-interface)
15. [Atalhos de teclado](#15-keyboard-shortcuts)
16. [Solução de problemas](#16-troubleshooting)

---

## 1. Introdução

O OpenFocus é um aplicativo desktop profissional de fusão de imagens multifoco, voltado a pesquisadores, engenheiros e profissionais de processamento de imagens. O software processa sequências de imagens multifoco com algoritmos avançados de registro e fusão para gerar imagens totalmente em foco, com profundidade de campo total.

### Principais recursos

- **Fusão de imagens multifoco**: combine várias imagens com pontos de foco diferentes em uma única imagem totalmente focada
- **Vários algoritmos de fusão**: escolha entre filtro guiado, DCT, DTCWT, GFG-FGF e fusão por IA (aprendizado profundo)
- **Registro de imagens**: alinhe sequências desalinhadas com os métodos ECC ou Homografia
- **Comparação Cortina**: inspecione o alinhamento e compare resultados com uma divisória A/B
- **Arquivos de projeto**: salve e restaure sessões de trabalho completas
- **TIFF multipágina**: pilhas Z em arquivo único carregam página a página e voltam a ser exportadas em um único arquivo
- **Mapa de profundidade**: mapas de posição de foco por pixel, com esquemas de pseudocor personalizados
- **Barra de escala**: barras de escala para publicação, com calibração automática em µm/px
- **Recuperação automática após travamento**: a última sessão de trabalho é oferecida para restauração depois de um travamento
- **Multilíngue**: inglês, 中文, 日本語 e espanhol, com detecção automática do idioma do sistema
- **CLI headless**: automatize o OpenFocus pela linha de comando, para automação e CI
- **Exportação flexível**: salve resultados como imagens individuais, pastas ou animações GIF
- **Transformações de imagem**: gire, inverta e redimensione pilhas de imagens
- **Adição de rótulos**: adicione rótulos de texto às pilhas registrada e de entrada
- **Preferências persistentes**: suas preferências, o idioma e as pilhas abertas recentemente sobrevivem a reinicializações

### Formatos de entrada compatíveis

- Arquivos de imagem: JPG/JPEG, PNG, BMP, TIFF, WebP e outros formatos comuns
- Arquivos de vídeo: formato MP4 (decodificado automaticamente em sequências de imagens)
- Pastas com sequências de imagens numeradas
- Arquivos RAW de câmera (DNG, CR2/CR3, NEF, ARW, RAF, ORF, RW2 e outros formatos
  RAW comuns). A decodificação usa `rawpy` (LibRaw), que o `pip install openfocus`
  instala e as compilações para Windows/macOS já incluem

### Formatos de saída compatíveis

- Imagens individuais: JPG, PNG, BMP, TIFF, WebP
- Sequências de imagens: salvas como pastas
- Animações: formato GIF

---

## 2. Primeiros passos

### Iniciar o aplicativo

Depois da instalação, abra o OpenFocus assim:

1. **Windows (executável)**: clique duas vezes no atalho ou no arquivo executável do OpenFocus
2. **Código-fonte Python**: execute `python main.py` no seu ambiente conda ativado

Ao iniciar, você verá a janela principal do aplicativo com uma interface de tema escuro.

### Fluxo de início rápido

1. Importe sua sequência de imagens (Arquivo → Abrir pasta ou arrastar e soltar)
2. Defina as opções de fusão no painel direito
3. Se quiser, ative o registro para alinhar as imagens
4. Ajuste o tamanho do kernel se necessário
5. *(opcional)* Marque **Prévia rápida** para testar os parâmetros primeiro em um rascunho reduzido
6. Clique em "Iniciar renderização" para começar o processamento
7. Exporte os resultados

---

## 3. Visão geral da interface principal

A interface do OpenFocus é composta por três áreas principais:

### Painel esquerdo: área de exibição de imagens

A parte esquerda da janela mostra suas imagens em uma visualização dividida:

- **Painel da imagem de origem (superior)**: mostra a sequência de imagens de entrada
- **Painel da imagem de resultado (inferior)**: mostra os resultados de fusão ou de registro
- **Controle deslizante de navegação**: use o controle abaixo de cada painel para percorrer os quadros
- **Rótulos de informação da imagem**: mostram a posição do quadro atual e as dimensões da imagem

Você pode redimensionar os painéis arrastando a alça do divisor entre eles.

### Painel direito: painel de controle

O painel direito reúne todas as opções de configuração, organizadas na vertical:

1. **Configurações de fusão**: selecione o algoritmo de fusão e o tamanho do kernel
2. **Configurações de registro**: ative/desative os métodos de alinhamento
3. **Botões de ação**: Restaurar padrões e Iniciar renderização
4. **Lista de imagens de origem**: mostra todas as imagens carregadas, com os nomes dos arquivos
5. **Lista de saída**: mostra os resultados gerados

### Barra de menus

A barra de menus superior dá acesso a todas as funções:

- **Arquivo**: operações de importação/exportação e a lista de arquivos recentes
- **Editar**: transformações de imagem e rótulos
- **Ferramentas**: processamento em lote
- **Configurações**: aceleração por GPU, idioma, blocos, registro e configurações de threads
- **Ajuda**: informações do ambiente, contato, verificação de atualizações e a pasta de logs

---

## 4. Importação de imagens e vídeo

### Abrir uma pasta de imagens

1. Vá em **Arquivo → Abrir pasta** ou pressione `Ctrl+O`
2. Selecione uma pasta que contenha sua sequência de imagens
3. Aparece um diálogo de escala de carregamento — escolha a escala (100% = resolução total)
4. Clique em OK para carregar as imagens

Marque **Não perguntar novamente** nesse diálogo para sempre carregar na escala
escolhida, sem aviso; **Configurações → Perguntar a escala de carregamento na importação**
reativa o aviso. Uma pilha carregada ainda pode ser reduzida depois com
**Editar → Redimensionar**.

O software irá automaticamente:
- Detectar e ordenar os arquivos de imagem por nome
- Exibir a primeira imagem no painel de origem
- Preencher a lista de imagens de origem com todos os quadros

### Abrir arquivos de vídeo

1. Vá em **Arquivo → Abrir vídeo** ou pressione `Ctrl+Shift+O`
2. Selecione um arquivo de vídeo (MP4, AVI, MOV, MKV, WMV, FLV, WEBM)
3. O vídeo é decodificado automaticamente em quadros individuais
4. Cada quadro passa a fazer parte da pilha de imagens

### Arquivos de projeto

Arquivo → Salvar projeto grava toda a sua sessão de trabalho em um arquivo
`.ofproj`: a lista exata de arquivos de origem (na ordem), a escala de
carregamento, o método de fusão, as opções de registro, o tamanho do kernel, as
configurações de blocos/threads/registro, as configurações de rótulo e o quadro
atual. Arquivo → Abrir projeto reconstrói essa sessão exatamente (as imagens de
origem são referenciadas, não copiadas — mantenha-as no lugar). O título da
janela mostra o projeto aberto.

### Pilha de demonstração

O diálogo de boas-vindas oferece **Carregar pilha de demonstração**: seis fotos
de exemplo incluídas, de uma pilha de foco real, para você experimentar o fluxo
completo imediatamente, sem ter imagens próprias. Também é útil para verificar
uma instalação.

### Curva de nitidez

Uma pequena curva acima do controle deslizante de navegação da origem mostra a
nitidez por quadro (normalizada). Quadros desfocados aparecem como vales
marcados com um ponto vermelho. Clique em qualquer ponto da curva para ir até
aquele quadro; passar o mouse mostra o número do quadro.

### Pilhas TIFF multipágina

Softwares de microscopia (ZEN, NIS-Elements, ImageJ/Fiji) costumam exportar uma
pilha Z inteira como UM único arquivo TIFF multipágina. O OpenFocus lê cada
página desse arquivo como um quadro individual da pilha de imagens — arraste ou
abra o arquivo e a pilha aparece completa (os quadros recebem nomes como
`<file>_page_001`). TIFFs de página única funcionam exatamente como antes. O
caminho inverso também funciona:
**Arquivo → Salvar pilha → (Pilha registrada / Pilha de entrada) → Salvar como TIFF multipágina**
grava a pilha inteira em um único `.tif`, preservando a profundidade de 16 bits.

### Barra de escala (figuras para publicação)

Configurações → Barra de escala… define uma barra de escala impressa nas imagens
exportadas. O tamanho do pixel (µm/px) é detectado automaticamente nos metadados
TIFF — a convenção `unit=micron` do ImageJ, o `PhysicalSizeX` do OME-TIFF ou uma
unidade explícita de resolução em centímetros — para que pilhas de microscopia
(inclusive TIFFs multipágina) se calibrem sozinhas. A resolução de impressora
(72/300 dpi e afins) deliberadamente *não* é tratada como calibração, e um valor
manual digitado sempre vence os metadados detectados: uma barra errada com
aparência plausível em uma figura é pior do que nenhuma. Escolha o canto e a cor
(Automática inverte em relação ao fundo); o diálogo mostra uma prévia ao vivo.
Com a barra ativada, toda exportação (imagem individual, pilhas em pasta, TIFF
multipágina, GIF, exportar todos, lote) leva uma barra no estilo de publicação:
um comprimento arredondado em 1-2-5 (~12% da largura da imagem) com texto legível
e contornado. A CLI oferece suporte por `--scale-bar` (calibração automática) e
`--scale-um <um/px>`.

### Mapa de profundidade (posição de foco)

**Ferramentas → Mapa de profundidade…** estima posições de foco com a medida de atividade
do método escolhido (DCT, DTCWT, GFG-FGF, StackMFF V3, OpenFocus B/C). Não é uma superfície
medida nem um mapa exato da origem dos pixels fusionados. Escolha os quadros marcados
atuais ou a origem registrada da saída selecionada; a falta de procedência não é substituída
pela pilha atual. Cálculo, textura e exportação ficam vinculados ao mesmo instantâneo da origem.

Os seis métodos exigem um clique explícito em **Calcular**. Abrir o diálogo ou mudar
a medida, a origem, o registro ou os parâmetros não inicia o cálculo. O editor fica logo
abaixo do seletor de medida: os kernels mediano DCT e inicial GFG aceitam ímpares de 1 a 51
(padrão 7); B usa uma janela de variância e C uma de energia passa-faixa, ímpares de 1 a 21 px
(padrão 11). Os níveis de decomposição DTCWT são editáveis aqui como rascunho local,
inicializado com os parâmetros da origem. StackMFF V3 mostra o próprio modo de saída de IA
(mistura suave por padrão, ou seleção rígida) e uma explicação, sem kernel numérico: a
mistura suave estima uma posição contínua ponderada por probabilidade, a seleção rígida usa
o quadro mais provável por pixel. Nenhuma é uma distância física. Os rascunhos começam com os parâmetros da saída correspondente
ou os valores lembrados por método na janela principal. Ficam guardados por origem e
método durante esta abertura, sem alterar a janela principal, o projeto ou a saída registrada.

Mudar os ajustes de cálculo marca o mapa 2D anterior como **não atualizado**, limpa a
superfície 3D e bloqueia todas as exportações, inclusive dados brutos, profundidade física
e OBJ, até um novo cálculo terminar com sucesso. O trabalho ativo é cancelado de forma
cooperativa. Depois, pressione **Calcular** novamente; nenhum cálculo é enfileirado ou
reiniciado automaticamente. Calcular permanece visível e fica desativado durante cálculo,
cancelamento e exportação do modelo.

- **Mapa de cores**: escala de cinza (padrão de publicação), 15 paletas
  científicas (Turbo, Viridis, Magma, …) e a sua própria — *Novo esquema
  personalizado…* monta um gradiente a partir de qualquer número de paradas de
  cor (clique duas vezes em uma parada para editar posição e cor); esquemas
  personalizados são salvos e oferecidos na lista suspensa.
- **Inverter / Gama / Suavização**: inverte o eixo de profundidade, reequilibra o
  tom (gama < 1 clareia índices baixos) e remove ruído pontual com uma passagem
  sensível a bordas.
- **Sobreposição**: mistura o mapa 2D com a saída associada em opacidade ajustável,
  somente quando os domínios de coordenadas correspondem.
- **Exportar**: a versão colorida aceita PNG/JPEG/TIFF/BMP. O TIFF bruto de 16 bits
  codifica índices de foco normalizados de 0 a 65535; não são dados de ponto flutuante
  sem perdas nem distâncias físicas. JPEG/BMP são recusados para os dados brutos.
- **Modelo 3D / Textura 3D**: o lado maior da malha tem 600 vértices por padrão.
  Cada nova abertura escolhe **Automática (cores reais)**: usa a saída associada somente
  quando sua identidade e domínio de coordenadas/ROI correspondem ao mapa concluído.
  Caso contrário, usa o primeiro quadro participante do cálculo concluído, após o registro
  e na ordem efetivamente usada. O nome da saída ou quadro utilizado é mostrado.
  Também é possível escolher explicitamente a saída correspondente, um quadro participante
  ou **Pseudocor de profundidade**; uma textura indisponível é informada. A prévia 3D e o PNG
  de textura do OBJ compartilham a mesma imagem. O mapa 2D mantém sua própria paleta e
  sobreposição; mudanças de paleta ou Gama não recolorem fotos. Só a textura em pseudocor
  responde à paleta, ao Gama e à inversão. A prévia parada e a exportação compartilham o
  mesmo campo de alturas amostrado, sem suavização extra oculta. O relevo relativo é uma
  visualização em unidades de imagem.

O cálculo roda em segundo plano, com progresso e cancelamento; o método de IA
reusa o mesmo processamento em blocos econômico em memória da fusão por IA, então
pilhas profundas funcionam. O equivalente na CLI é `--depth-map [path]`, com
`--depth-colormap`.

### Recuperação automática após travamento

O OpenFocus mantém continuamente um instantâneo da sua sessão de trabalho (pilha,
configurações, rótulos, quadro atual). Se o aplicativo fechar inesperadamente —
travamento, queda de energia ou encerramento forçado — a próxima inicialização
detecta isso e oferece **Restaurar sessão** para voltar exatamente aonde você
estava. Saídas normais nunca mostram o aviso. Os dados de recuperação ficam junto
às suas configurações (instalações portáteis os mantêm ao lado do executável,
então eles viajam com você).

### Arquivos recentes

Cada pilha que você abre é lembrada em **Arquivo → Abrir recentes** (até 8
entradas). Pressione `Alt+1`–`Alt+8` para reabrir uma entrada diretamente. Reabrir
a mesma pilha a move para o topo; entradas cujos arquivos não existem mais podem
ser removidas com confirmação, e **Limpar lista de recentes** esvazia o menu.

### Arrastar e soltar

Você também pode arrastar uma pasta diretamente sobre a janela do aplicativo para
importar imagens. Soltar mais arquivos sobre uma pilha existente os anexa (após
uma verificação de tamanho).

### Orientação EXIF

Arquivos JPEG/TIFF de câmera com a tag EXIF Orientation são girados/corrigidos
automaticamente durante a importação, então fotos em retrato carregam na vertical.

### Origens de 16 bits

Entradas PNG/TIFF de 16 bits são detectadas automaticamente e preservadas de ponta
a ponta: a fusão roda em profundidade total e os resultados podem ser salvos de
volta como PNG/TIFF de 16 bits (exportações JPG/BMP são convertidas para 8 bits,
já que esses formatos não armazenam 16 bits).

Arquivos RAW de câmera são revelados em 16 bits (balanço de branco da câmera, gama
sRGB) com o clareamento automático desativado, para que todos os quadros de uma
pilha recebam o mesmo mapeamento de exposição; exposição automática por quadro
faria o brilho piscar entre os quadros. Salve como TIFF ou PNG para manter os
dados de 16 bits.

### Limpar a pilha

Para remover todas as imagens carregadas, vá em **Arquivo → Limpar pilha** ou
pressione `Ctrl+W`.

---

## 5. Visualização e navegação pelas imagens

### Navegar pela pilha de imagens

- **Controle deslizante**: arraste o controle abaixo do painel de origem para percorrer os quadros
- **Lista de arquivos**: clique em qualquer nome na lista de imagens de origem do painel direito
- **Teclado**: use as setas (esquerda/direita) para navegar

### Zoom e deslocamento

- **Aumentar/diminuir zoom**: roda do mouse sobre a imagem
- **Zoom fino**: segure `Ctrl` + roda do mouse para incrementos menores
- **Ajustar à janela**: clique duas vezes na imagem para alternar entre ajustar e visualização em 100%
- **Deslocar**: clique e arraste para mover a imagem dentro do painel

### Navegação sincronizada

Ao navegar no painel de origem, o painel de resultados mostra o quadro de resultado
correspondente, se ele existir.

---

## 6. Visualização de comparação Cortina

A visualização Cortina sobrepõe duas imagens em um único quadro, com uma divisória
vertical arrastável, tornando desalinhamentos e diferenças de qualidade visíveis
diretamente na comparação.

### Abrir a visualização Cortina

1. Renderize ao menos um resultado (fusão ou registro)
2. Clique no botão **Cortina** no canto superior direito da barra de título do painel de resultados
3. A área de resultados passa para a visualização de comparação; clique em **Cortina** novamente para voltar

### Interpretando a visualização

- **Lado A (à esquerda da divisória)**: um quadro de origem — por padrão, ele acompanha o controle do painel de origem
- **Lado B (à direita da divisória)**: a renderização mais recente
- **Divisória branca + alça redonda**: arraste na horizontal para mover o divisor
- Arrastar bordas que cruzam a divisória revela erros de registro difíceis de ver em painéis lado a lado

### Escolher o que comparar

A barra na parte inferior da visualização de comparação tem um grupo para cada lado:

- **Lado A** — marque **Seguir origem** (padrão) para acompanhar o controle do painel de origem, ou desmarque para fixar o lado A em qualquer quadro com seu próprio controle
- **Lado B** — marque **Resultado mais recente** (padrão) para mostrar sempre a renderização mais nova, ou desmarque para percorrer seu histórico de saídas com o controle deslizante

Os dois lados se atualizam ao vivo: novas renderizações, movimentos do controle de
origem e exclusões de quadros aparecem imediatamente.

### Zoom e deslocamento

A visualização de comparação usa uma única transformação para as duas imagens:

- **Zoom**: roda do mouse (aproxima em torno do cursor)
- **Deslocar**: clique e arraste em qualquer ponto, exceto na divisória
- **Redefinir**: clique duas vezes

O modo Cortina e o modo ROI compartilham o painel de resultados, então entrar em um
sai do outro. Limpar a pilha também sai do modo Cortina.

---

## 7. Registro de imagens

O registro de imagens corrige o desalinhamento espacial entre os quadros da sua
pilha. Ele é essencial quando as imagens têm pequenos deslocamentos ou mudanças de
perspectiva.

### Métodos de registro

#### ECC (coeficiente de correlação aprimorado)

- Usa otimização para encontrar o melhor alinhamento com base na correlação
- Adequado a desalinhamentos sutis
- Oferece precisão subpixel
- Mais lento que a Homografia, porém mais preciso

#### Homografia

- Baseado em correspondência de pontos de feição (feições SIFT/ORB)
- Lida com transformações geométricas maiores
- Mais rápido que o ECC para o alinhamento inicial
- Bom para imagens com mudanças de perspectiva significativas

#### OpenFocus (rápido)

- Alinhamento por similaridade limitada (translação + rotação + escala),
  com equalização de brilho por quadro e corte automático
- Menos de um segundo por quadro — bem mais rápido que ECC ou Homografia
- Quadros que se deslocam mais de **8% do tamanho do quadro são deixados sem
  alinhamento** em vez de corrigidos pela metade. Ajuste com a opção de CLI
  `--align-shift-limit`
- Para geometrias não rígidas ou extremas, escolha ECC ou Homografia

### Ativar o registro

1. No painel direito, localize o grupo **Registro**
2. Escolha um método na lista suspensa: **Sem alinhamento**, **OpenFocus (rápido)**,
   **ECC** ou **Homografia**
3. O alinhamento rápido do OpenFocus também pode ser encadeado na frente de ECC ou
   Homografia (primeiro uma passada global grosseira, depois a passada local precisa)

### Quando usar registro

- Suas imagens têm desalinhamento visível
- As imagens foram capturadas com a câmera na mão
- A cena tem paralaxe (objetos mais próximos se deslocam entre os quadros)
- Os resultados mostram artefatos de fantasma ou duplicação

### Configurações de registro

Acesse ajustes adicionais em **Configurações → Configurações de registro**:
- **Largura de redução**: controla a resolução do pré-processamento (padrão: 1024 px)
- Valores menores = processamento mais rápido, possivelmente com menos precisão
- Valores maiores = processamento mais lento, possivelmente com mais precisão

---

## 8. Métodos de fusão de imagens

O OpenFocus oferece dez algoritmos de fusão (mais um modo apenas de alinhamento).
Cada um tem características próprias, adequadas a diferentes tipos de imagem.

### Filtro guiado (padrão)

- **Algoritmo**: usa filtragem guiada para uma fusão que preserva bordas
- **Melhor para**: uso geral, com equilíbrio entre velocidade e qualidade
- **Vantagens**: boa preservação de bordas, processamento rápido
- **Parâmetro**: tamanho do kernel (ajustável de 1 a 51, apenas valores ímpares)
  - Kernel menor: mais detalhe local, possivelmente mais ruído
  - Kernel maior: resultados mais suaves, melhor supressão de ruído

### DCT (Transformada Discreta de Cosseno)

- **Algoritmo**: fusão no domínio da frequência usando coeficientes DCT
- **Melhor para**: imagens com muita textura, imagens científicas
- **Vantagens**: preserva texturas finas, base matemática sólida
- **Parâmetro**: o tamanho do kernel controla a janela de processamento

### DTCWT (Transformada Wavelet Complexa de Árvore Dupla)

- **Algoritmo**: fusão no domínio wavelet em múltiplas escalas
- **Melhor para**: cenas complexas com vários níveis de detalhe
- **Vantagens**: excelente análise multiescala, boa direcionalidade
- **Regra de fusão**: uma decisão por pixel e por nível para todas as orientações e
  canais de cor (N-vias, com verificação de consistência por maioria); a faixa
  passa-baixa segue a decisão mais grosseira em vez de fazer a média de todos os quadros
- **Parâmetro**: níveis de wavelet (1–6, padrão 4) no painel direito. O método
  não usa kernel espacial, então o controle de kernel fica oculto.

### GFG-FGF (Gradiente de quatro vizinhanças do tipo gaussiano + filtro guiado rápido)

- **Algoritmo**: implementação de Fu et al. (*Laser & Optoelectronics
  Progress*, 2024): um mapa de foco por diferença de desfoque |I − mean(I)| é
  enriquecido com o operador de gradiente de quatro vizinhanças do tipo gaussiano
  (GFG), refinado por um filtro guiado rápido, decidido pixel a pixel (empates
  resolvidos por uma medida de pequena região com Sobel 3×3), e os mapas de decisão
  binária são suavizados novamente pelo filtro guiado rápido antes da mistura
  ponderada.
- **Melhor para**: microscopia e imagens com regiões de foco bem definidas
- **Vantagens**: rápido, com limites nítidos nas regiões de foco; a memória não
  cresce com o número de quadros
- **Parâmetro**: Kernel = tamanho do filtro de média do mapa de foco grosseiro

### StackMFF (V1, aprendizado profundo)

- **Algoritmo**: uma U-Net 3D que prevê a luminância fundida de todo o stack
  numa passagem; Cb/Cr vêm do fotograma de origem mais próximo
- **Melhor para**: stacks sem ordem de foco fixa ou sem série contínua
- **Vantagens**: independente da ordem, sem parâmetros, cores dentro do gamut
- **Nota**: 6 M de parâmetros; stacks grandes beneficiam-se da GPU

### StackMFF V2 (aprendizado profundo)

- **Algoritmo**: estimativa de foco por camada refinada por uma camada
  recorrente bidirecional; a rede regride o índice do plano focal e amostra o
  stack. **Os fotogramas devem manter a ordem de foco original** — inverter
  está desativado
- **Saída de IA**: Dura (padrão, regra do artigo) ou Suave (mistura ponderada
  por probabilidade)
- **Mapas de profundidade**: suportados; Suave = camada esperada, Dura = camada amostrada
- **Melhor para**: varreduras de foco contínuas com um modelo muito pequeno e rápido
- **Requisitos**: PyTorch; cerca de 0,05 M de parâmetros

### StackMFF-V3 (Aprendizado profundo)

- **Algoritmo**: classificação da origem pixel a pixel (requer PyTorch). A rede
  prevê, para cada pixel, de qual quadro de entrada ele vem e depois mistura os
  quadros por essas probabilidades ou escolhe um único quadro
- **Modo de saída da IA** (painel direito, apenas para este método):
  - **Mistura suave (ponderada)** — padrão; cada pixel é uma mistura dos quadros
    ponderada por probabilidade. Transições suaves, sem emendas duras; corresponde
    à configuração relatada no artigo
  - **Escolha rígida (quadro único)** — cada pixel é copiado do quadro mais
    provável. Seleção mais nítida, sem interpolação entre quadros; uma emenda em
    nível de pixel pode aparecer onde dois quadros empatam
- **Melhor para**: resultados da mais alta qualidade quando há GPU disponível
- **Vantagens**: invariante à ordem (a ordem dos quadros nunca muda o resultado),
  aceita qualquer tamanho de pilha, sem ajuste por método, otimização automática
- **Requisitos**: instalação do PyTorch, GPU CUDA opcional (sem torchvision)
- **Observação**: pode ficar indisponível se o PyTorch não estiver instalado

### OpenFocus A / B / C

- **Algoritmo**: média ponderada por variância, seleção por mapa de profundidade de
  variância e fusão por pirâmide laplaciana
- **OpenFocus A**: média ponderada por variância com prior do último quadro —
  rápida, com cores estáveis em pilhas curtas
- **OpenFocus B**: seleção por mapa de profundidade de variância — escolhe o quadro
  mais nítido por pixel; melhor em superfícies lisas e em microscopia
- **OpenFocus C**: mistura por faixa com pirâmide laplaciana — aparência natural em
  assuntos texturizados, realça o contraste local
- **Parâmetros**: A e B aceitam uma janela de variância (controle de kernel); C não
  tem raio. Os três aceitam um sigma de suavização.

**Suavização**: uma caixa de rotação abaixo do controle de kernel, com o
significado do método selecionado (faixa 0–8, passo 0,2):

| Método | Rótulo | Padrão | Efeito |
|---|---|---|---|
| OpenFocus A | Suavização da mistura (σ) | 2,0 | Suavização gaussiana do mapa de pesos por quadro — com que amplitude as regiões nítidas dos quadros vizinhos são misturadas |
| OpenFocus B | Suavização da profundidade (σ) | 1,2 | Suavização gaussiana do mapa de profundidade — largura da transição entre quadros |
| OpenFocus C | Suavização da mistura (σ) | 3,0 | Suavização gaussiana do mapa de pesos de cada faixa da pirâmide |

0 mantém a escolha por pixel mais nítida (detalhe máximo, mas com ruído pontual em
pilhas ruidosas); valores maiores suavizam transições e halos. Cada método lembra o
próprio valor entre sessões, e o processamento em lote usa a mesma regra.

C não tem Raio: o Método C do Helicon expõe apenas Suavização, então o controle de
kernel fica oculto em C em vez de não fazer nada silenciosamente.

CLI: `--depth-smoothing SIGMA` (B) e `--blend-smoothing SIGMA` (A, C).

**As opções de brilho são por método.** *Igualar exposição antes da fusão* e
*Remover deriva de brilho* são lembradas separadamente para cada método de fusão,
porque respondem a uma pergunta de cada pilha, não a uma preferência global. Ambas
vêm ativadas por padrão no OpenFocus A/B/C (que não têm calibração de exposição
embutida); os outros métodos mantêm os padrões históricos (igualação de exposição
ativada, remoção de deriva desativada). Ao trocar de método, aparecem os valores
próprios daquele método.

### Somente alinhamento (sem fusão)

Selecione **Somente alinhamento (sem fusão)** como método para registrar a pilha e
enviar os quadros alinhados à lista de saída sem fusionar — prático para inspecionar
a qualidade do registro ou exportar quadros alinhados.

### Selecionar um método de fusão

1. No painel direito, localize o grupo **Fusão**
2. Escolha um método na **lista suspensa** (exatamente um fica sempre selecionado;
   a primeira entrada registra sem fusionar)
3. Ajuste o tamanho do kernel quando aplicável — o rótulo mostra o que o kernel
   significa para o método selecionado
4. Clique em **Iniciar renderização** para processar

### Guia de comparação de algoritmos

| Algoritmo | Velocidade | Qualidade | Melhor para |
|-----------|-------|---------|----------|
| Filtro guiado | Rápida | Boa | Uso geral |
| DCT | Média | Boa | Texturas |
| DTCWT | Média | Muito boa | Cenas complexas |
| GFG-FGF | Rápida | Boa | Regiões de foco |
| Fusão por IA (StackMFF-V3) | Lenta (GPU) | Excelente | Melhor qualidade |
| OpenFocus A | Muito rápida | Muito boa | Pilhas curtas, estabilidade de cor |
| OpenFocus B | Muito rápida | Muito boa | Superfícies lisas, microscopia |
| OpenFocus C | Rápida | Muito boa | Assuntos texturizados, aparência natural |

### Prévia rápida (renderização de rascunho)

Ajustar parâmetros em pilhas grandes é lento em resolução total. Marque
**Prévia rápida** acima do botão de renderização e o OpenFocus primeiro renderiza
um rascunho reduzido (lado mais longo com 1200 px) — normalmente várias vezes mais
rápido, inclusive o registro. A prévia é exibida com uma nota "Prévia" no resumo da
barra de status após a renderização e **não** é adicionada ao histórico de saída.
Desmarque a caixa e renderize de novo para produzir o resultado em resolução total.
O modo de prévia não se aplica a renderizações de ROI e nunca contamina o cache de
alinhamento em resolução total.

---

## 9. Transformações de imagem

### Rotação

1. Vá em **Editar → Girar**
2. Escolha o tipo de rotação:
   - **90° no sentido horário**: gira todas as imagens 90 graus para a direita
   - **90° no sentido anti-horário**: gira todas as imagens 90 graus para a esquerda
   - **180°**: vira as imagens de cabeça para baixo

### Inversão

1. Vá em **Editar → Inverter**
2. Escolha o tipo de inversão:
   - **Inverter horizontalmente**: espelha da esquerda para a direita
   - **Inverter verticalmente**: espelha de cima para baixo

### Redimensionamento

1. Vá em **Editar → Redimensionar**
2. Aparece um diálogo com as opções de redimensionamento
3. Informe as novas dimensões ou a porcentagem de escala
4. Todas as imagens da pilha são redimensionadas proporcionalmente

### Ordem das transformações

As transformações são aplicadas à pilha inteira, mantendo o alinhamento entre os
quadros.

---

## 10. Retoque, rótulos e anotações

### Retocar o resultado (Editar → Retocar resultado…, `Ctrl+T`)

A fusão ainda pode deixar um defeito local — um halo em torno de uma borda de alto
contraste, uma cerda tirada do quadro errado, ruído em uma área plana. O pincel de
retoque copia pixels de um quadro de origem para o resultado:

1. Renderize um resultado (ou selecione um na lista de saída; clique com o botão
   direito → *Retocar resultado…* também funciona) e pressione `Ctrl+T`.
2. Pinte sobre o defeito com o botão esquerdo do mouse. Com **Escolher
   automaticamente o quadro mais nítido por traço** ativado (padrão), cada traço
   primeiro seleciona o quadro que tem mais detalhe sob o pincel — corrigir um halo
   costuma ser um único arrasto. Desative para escolher o quadro você mesmo com o
   controle deslizante ou `,` / `.`.
3. Ajuste o **raio** (`[` / `]`), a **dureza** (bordas suaves escondem a emenda) e a
   **opacidade**. Toques sobrepostos dentro de um mesmo traço nunca se acumulam.
4. **Mostrar quadro de origem** (`Tab`) mostra uma prévia do quadro a partir do qual
   você está pintando. A roda dá zoom, arrastar com o botão direito/meio desloca, `F`
   ajusta a imagem.
5. `Ctrl+Z` / `Ctrl+Y` desfazem e refazem traços inteiros. **Aplicar como nova saída**
   adiciona a imagem retocada à lista de saída (sufixo `_Retouched`); a saída
   original permanece intacta.

O pincel usa os quadros alinhados que produziram o resultado, então ele se alinha
com ele; resultados de 16 bits continuam em 16 bits. Renderizações de ROI e pilhas
redimensionadas não podem ser retocadas (os quadros não correspondem mais à geometria
do resultado).

### Adicionar rótulos

1. Certifique-se de ter resultados renderizados ou imagens carregadas
2. Vá em **Editar → Adicionar rótulo**
3. Digite o texto do rótulo no diálogo
4. Os rótulos são adicionados tanto à pilha registrada quanto à pilha de entrada

### Excluir rótulos

- **Excluir rótulos da pilha registrada**: remove os rótulos das imagens alinhadas
- **Excluir rótulos da pilha de entrada**: remove os rótulos das imagens originais

Acesse essas opções no submenu **Editar → Excluir rótulo**.

---

## 11. Configurações

### Configurações de blocos (otimização de memória)

Para imagens grandes, o OpenFocus usa processamento em blocos para evitar problemas
de memória.

Acesse em **Configurações → Configurações de blocos**:

- **Tamanho do bloco**: tamanho de cada bloco de processamento (padrão: 1024 px)
- **Sobreposição dos blocos**: sobreposição entre blocos para uma mistura sem emendas (padrão: 256 px)
- **Limite para blocos**: tamanho de imagem acima do qual os blocos são usados (padrão: 2048 px)

**Recomendações**:
- Bloco menor: menos uso de memória, processamento mais lento
- Bloco maior: mais uso de memória, processamento mais rápido
- Aumente a sobreposição se notar emendas visíveis nos resultados

### Configurações de registro

Acesse em **Configurações → Configurações de registro**:

- **Largura de redução**: resolução de pré-processamento para o registro
  - Padrão: 1024 px
  - Menor para velocidade, maior para precisão

### Referência de desempenho

Consulte [PERFORMANCE.md](./PERFORMANCE.md) para números medidos de vazão e de
memória em pilhas reais grandes, além de orientações práticas para pilhas profundas
e entradas de alta resolução.

### Configurações do número de threads

Controle o processamento paralelo na CPU:

Acesse em **Configurações → Configurações de threads**:

- `0` seleciona o agendamento automático e é o padrão para novas configurações.
- Valores positivos limitam as threads; os limites salvos são preservados.
- Capacidade da CPU, modo de recursos e memória disponível podem reduzir a concorrência.

### Aceleração por GPU

O OpenFocus usa a GPU (CUDA/MPS) apenas para o modelo neural de IA (StackMFF-V3);
algoritmos clássicos sempre rodam na CPU.

- Alterne em **Configurações → Aceleração por GPU**
- Desmarcar força tudo na CPU — útil quando os drivers da GPU dão problema
- O painel de status mostra `CPU*` enquanto a GPU está desativada
- A escolha vale a partir da próxima renderização e é lembrada

### Tamanho do lote de IA

Acesse em **Configurações → Tamanho do lote de IA**:

- O limite de lote salvo é 2 por padrão; atualmente é processado um bloco por chamada em todos os dispositivos.
- Aumentar esse limite não ativa lotes paralelos na GPU.

### Temas

Dois temas de interface estão disponíveis: **Escuro (padrão)** e **Claro**. Alterne
em **Configurações → Tema**; a escolha é aplicada imediatamente (a barra de título do
Windows acompanha) e é lembrada. Na primeira inicialização, o usuário escolhe um tema
no diálogo de boas-vindas.

### Modo portátil

Crie um arquivo vazio chamado `OpenFocus.portable` ao lado de `OpenFocus.exe` (o zip
portátil já vem com um) e todos os dados do usuário — configurações, logs, arquivos
recentes — passam a ser armazenados em uma pasta `OpenFocusData` ao lado do
aplicativo, em vez do seu perfil de usuário, para que a instalação possa viajar em um
pendrive USB. Exclua o marcador para voltar ao perfil normal.

### Metadados EXIF

Os metadados da origem (modelo da câmera, exposição, orientação e assim por diante)
são mantidos: a orientação é aplicada no carregamento, e as exportações PNG/TIFF/WebP
incorporam o EXIF original automaticamente. Exportações JPG/BMP não conseguem
armazenar o EXIF completo e são gravadas sem ele.

### Entrada HEIC/HEIF

Fotos de iPhone carregam diretamente. Requer o pacote opcional `pillow-heif`
(`pip install pillow-heif`; incluído nas compilações empacotadas).

### Atualizações

Arquivos de projeto (`.ofproj`) se comportam como um documento: a barra de título
mostra o nome do projeto com um `*` enquanto houver alterações não salvas, e fechar o
aplicativo ou abrir outro projeto pergunta antes de descartá-las. O projeto armazena
a lista de quadros, a escala de carregamento, todas as opções de renderização, os
rótulos e o quadro atual — as imagens em si são referenciadas, não copiadas.

Ajuda → Verificar atualizações compara sua versão com as GitHub Releases.
**Baixar e instalar** busca o novo instalador dentro do aplicativo (com progresso)
e o executa; Abrir página de downloads leva à lista de lançamentos. A verificação
roda silenciosamente no máximo uma vez por dia.

### Cache de registro

O OpenFocus pode armazenar os resultados de registro em uma pasta `.openfocus_cache`
ao lado da pilha de origem. Reabrir a mesma pilha (com os arquivos inalterados)
dispensa totalmente o alinhamento. Alterne em **Configurações → Cache de registro**;
entradas obsoletas são substituídas automaticamente.

### Configurações persistidas

Tudo o que segue sobrevive automaticamente a uma reinicialização do aplicativo:
número de threads, parâmetros de blocos, largura de redução do registro, tamanho do
lote da fusão por IA (StackMFF-V3), a opção de aceleração por GPU, a opção do cache de
registro, o idioma da interface, o tamanho da janela/leiaute dos painéis, o formato de
arrastar e exportar e as pilhas abertas recentemente. Os diálogos de configuração
salvam no momento em que você confirma; tudo também é salvo quando o aplicativo
fecha. As preferências são armazenadas por usuário em `OpenFocus/OpenFocus.ini`
(formato INI do QSettings).

### Idioma da interface

Alterne a qualquer momento em **Configurações → Idioma → English / 中文**. A alteração
entra em vigor imediatamente e é lembrada nas próximas sessões.

### Verificação automática de atualizações

Uma vez por dia, poucos segundos após a inicialização, o OpenFocus verifica
silenciosamente no GitHub se há uma versão mais nova. Quando existe, uma faixa
aparece acima dos painéis de imagem: a versão, **Atualizar agora** (baixa e reinicia
em um único passo), **Novidades** (notas da versão) e um botão para dispensar.
Dispensar oculta a faixa apenas para aquela versão — a próxima versão a traz de volta,
e **Ajuda → Verificar atualizações** a reativa imediatamente.

Enquanto o download roda em segundo plano, a barra de status mostra o progresso. Você
também pode verificar manualmente a qualquer momento em **Ajuda → Verificar
atualizações**.

---

## 12. Processamento em lote

Processe várias pastas automaticamente:

1. Vá em **Ferramentas → Processamento em lote** ou clique na opção de lote
2. No diálogo de lote:
   - **Adicionar pastas**: selecione pastas que contenham sequências de imagens
   - **Remover selecionadas**: remove as pastas selecionadas da lista
   - **Limpar tudo**: remove todas as pastas
3. Defina as opções de saída:
   - **Criar subpasta na pasta de origem**: cria os resultados dentro de cada pasta de entrada
   - **Igual à pasta de origem**: sobrescreve ou salva junto aos arquivos de entrada
   - **Especificar pasta de saída**: informe o local de saída
4. Revise as configurações atuais de fusão/registro
5. Clique em **Iniciar processamento em lote** para começar

O diálogo de lote mostra o progresso em tempo real. Você pode cancelar o
processamento a qualquer momento.

### Observações sobre o processamento em lote

- Usa as mesmas configurações de fusão e registro do processamento manual
- Cada pasta é processada de forma independente
- Os nomes dos arquivos incluem carimbo de data/hora e parâmetros para identificação

---

## 13. Exportação de resultados

### Salvar um resultado individual

1. Selecione na lista de saída o resultado que deseja salvar
2. Vá em **Arquivo → Salvar** ou pressione `Ctrl+S`
3. Escolha o formato e o local
4. Clique em Salvar

### Formato de arrastar para exportação

Arrastar resultados da lista de saída grava arquivos de imagem no formato escolhido
no menu de contexto da lista de saída (**Formato de arrastar**: JPG, PNG ou TIFF). A
escolha é lembrada.

### Salvar a pilha registrada

1. Vá em **Arquivo → Salvar pilha → Pilha registrada**
2. Escolha o formato de gravação:
   - **Salvar como pasta**: salva cada quadro alinhado como um arquivo de imagem separado
   - **Salvar como GIF**: cria um GIF animado da sequência alinhada

### Salvar a pilha de entrada

1. Vá em **Arquivo → Salvar pilha → Pilha de entrada**
2. Escolha o formato de gravação:
   - **Salvar como pasta**: salva os quadros originais como imagens separadas
   - **Salvar como GIF**: cria um GIF animado da sequência de entrada

### Formatos de exportação

- **JPG**: comprimido, boa relação qualidade/tamanho
- **PNG**: sem perdas, aceita transparência
- **BMP**: não comprimido, qualidade máxima
- **TIFF**: alta qualidade, aceita camadas
- **GIF**: formato de animação

---

## 14. Interface de linha de comando

O OpenFocus pode rodar sem a interface gráfica — útil para pipelines com scripts e
para CI. A partir de um ambiente Python (ou ao lado do executável empacotado):

```bash
# Fuse a folder with the default Guided Filter
python main.py --input ./stack_folder --output ./result/fused.png

# DTCWT with ECC registration, 8 threads
python main.py -i ./stack_folder -o ./result/fused.png -m dtcwt -a ecc -t 8

# StackMFF-V3 forced to CPU with a custom tile size
python main.py -i ./stack_folder -o ./result/fused.png -m stackmffv3 --cpu --tile-size 512

# Explicit file list; video files are also accepted as input
python main.py -i img1.jpg img2.jpg img3.jpg -o fused.png -m gfgfgf

# Single multi-page TIFF Z-stack with a metadata-calibrated scale bar
python main.py -i ./zstack.tif -o fused.png -m dtcwt --scale-bar
```

| Opção | Descrição |
|--------|-------------|
| `--input, -i` | Pasta de imagens, arquivo de vídeo ou um ou mais arquivos de imagem |
| `--output, -o` | Caminho da imagem de saída (a extensão define o formato) |
| `--method, -m` | `stackmffv3` (padrão), `none`, `openfocus_a`, `openfocus_b`, `openfocus_c`, `guided_filter`, `dct`, `dtcwt`, `gfgfgf` |
| `--align, -a` | `none` (padrão), `homography`, `ecc`, `both`, `openfocus` |
| `--align-shift-limit` | apenas no alinhamento `openfocus`: deslocamento máximo aceito como fração do tamanho do quadro (padrão 0,08; quadros acima disso ficam sem alinhamento) |
| `--exposure-match` | Iguala a média de cada quadro à do último quadro antes da fusão (padrão: ativado; `--no-exposure-match` desativa). Ganhos por pilha inteira, então a saída em blocos continua contínua; vale para todos os métodos e para a referência de brilho do alinhador OpenFocus |
| `--base-frame` | Etapa opcional após a fusão: iguala as baixas frequências da saída ao quadro `last`/`first` (`none` por padrão) |
| `--sort` | Ordem dos quadros para `openfocus_a/b/c`: `asc` (padrão, ordem dos arquivos), `desc` (inverte uma varredura de longe para perto), `off` |
| `--dust-map WHITE` | Correção de campo plano (mapa de poeira) a partir de uma foto desfocada de campo branco |
| `--kernel, -k` | Tamanho do kernel (forçado a ímpar; na família openfocus é a origem do raio da janela de variância) |
| `--depth-smoothing SIGMA` | apenas `openfocus_b`: sigma gaussiano da transição de profundidade entre quadros (padrão 1,2; 0 preserva o máximo de detalhe) |
| `--blend-smoothing SIGMA` | apenas `openfocus_a` / `openfocus_c`: sigma gaussiano do mapa de pesos por quadro (padrões 2,0 para A e 3,0 para C; 0 mantém a escolha mais nítida por pixel) |
| `--threads, -t` | Limite de threads; `0` = automático (padrão), valores positivos limitam a concorrência |
| `--resource-preference` | Modo de recursos: `balanced` (padrão), `responsive` ou `maximum` |
| `--batch-size` | Limite do lote de blocos de IA (padrão 2); atualmente um bloco por chamada |
| `--cpu` | Força o uso da CPU mesmo quando há GPU disponível |
| `--downscale` | Largura de redução para detecção de feições do registro |
| `--tile-size` | Tamanho do bloco para imagens grandes |
| `--output-dir, -d` | Modo lote: funde cada pasta de entrada, gravando `<folder>.png` por pilha |
| `--scale-bar` | Desenha uma barra de escala (calibração detectada automaticamente nos metadados TIFF) |
| `--depth-map [path]` | Também salva um mapa colorido de posição de foco (profundidade) (padrão: `<output>_depth.png`) |
| `--depth-colormap` | Mapa de cores para `--depth-map`: turbo, viridis, jet, gray, … (padrão turbo) |
| `--scale-um` | Tamanho do pixel em µm/px, sobrepondo os metadados (implica `--scale-bar`) |

Códigos de saída: `0` sucesso, `1` erro de processamento, `2` erro de uso. Executar
`python main.py --help` mostra todas as opções; iniciar com um caminho simples
(`python main.py C:\stack`) ainda abre a interface gráfica com essa pasta carregada.

---

## 15. Atalhos de teclado

### Operações de arquivo

| Atalho | Ação |
|----------|--------|
| `Ctrl+O` | Abrir pasta |
| `Ctrl+Shift+O` | Abrir vídeo |
| `Ctrl+S` | Salvar resultado |
| `Ctrl+Shift+S` | Salvar pilha registrada |
| `Ctrl+W` | Limpar pilha |
| `Ctrl+Q` | Sair |
| `Ctrl+R` | Iniciar renderização |
| `Ctrl+Shift+R` | Comparar todos os métodos |
| `Ctrl+T` | Retocar resultado |
| `Ctrl+M` | Mapa de profundidade |
| `Ctrl+E` | Alternar comparação Cortina |
| `Ctrl+Shift+B` | Processamento em lote |
| `Ctrl+Z` / `Ctrl+Y` | Desfazer / Refazer (edições da pilha) |

### Navegação

| Atalho | Ação |
|----------|--------|
| `Seta esquerda` | Quadro anterior |
| `Seta direita` | Próximo quadro |
| `Roda do mouse` | Aumentar/diminuir zoom |
| `Ctrl + Roda` | Zoom fino |
| `Clique duplo` | Alternar entre ajuste e visualização em 100% |

### Geral

| Atalho | Ação |
|----------|--------|
| `Espaço` | Alternar modo de deslocamento |
| `Delete` | Excluir item selecionado |

---

## 16. Solução de problemas

### Problemas comuns

#### "Nenhuma imagem carregada"

- Verifique se você importou uma pasta de imagens ou um vídeo antes
- Verifique se a pasta contém formatos de imagem compatíveis

#### Os resultados da fusão parecem borrados ou com fantasmas

- Ative o registro de imagens (escolha um na lista: Homografia / ECC / OpenFocus rápido)
- Garanta que as imagens estejam bem alinhadas antes da fusão
- Experimente outro algoritmo de fusão
- Aumente o tamanho do kernel para resultados mais suaves

#### IA indisponível

- Instale o PyTorch: `pip install torch torchvision`
- Para suporte a GPU, instale o PyTorch com CUDA
- Confira em **Ajuda → Informações do ambiente**

#### Erros de falta de memória

- Ative o processamento em blocos em **Configurações → Configurações de blocos**
- Reduza o tamanho do bloco
- Use tamanhos de kernel menores
- Redimensione as imagens para dimensões menores

#### Falha no registro

- Tente um único método de registro por vez
- Reduza a largura de redução em **Configurações → Configurações de registro**
- Garanta que as imagens tenham contraste e feições suficientes

#### Processamento lento

- Aumente o número de threads em **Configurações → Configurações de threads**
- Use uma CPU mais potente
- Reduza a resolução das imagens
- Use algoritmos de fusão mais simples (filtro guiado em vez de fusão por IA)

#### Aviso do Windows SmartScreen na primeira inicialização

Os binários distribuídos não são assinados. Clique em **Mais informações → Executar
assim mesmo**. A primeira inicialização após descompactar pode levar de 30 a 60
segundos.

#### O macOS diz que o aplicativo "não pode ser aberto" ou "está danificado"

O aplicativo não é assinado nem notarizado. Depois de mover o `OpenFocus.app` para
Aplicativos, execute:

```bash
xattr -cr /Applications/OpenFocus.app
```

ou clique com o botão direito no aplicativo e escolha **Abrir**. As compilações
distribuídas para macOS são para Apple Silicon (M1/M2/M3/M4); Macs Intel podem rodar
a partir do código-fonte.

### Obtendo ajuda

1. **Informações do ambiente**: vá em **Ajuda → Informações do ambiente** para verificar sua instalação
2. **Fale conosco**: vá em **Ajuda → Fale conosco** para informações de suporte

---

## Apêndice: detalhes dos algoritmos de fusão

### Filtro guiado

O filtro guiado faz uma suavização que preserva bordas usando uma imagem de guia. Ele
calcula transformações lineares locais entre a imagem de guia e a imagem de entrada e
depois as aplica para produzir a saída. Esse método se destaca por preservar bordas
enquanto reduz o ruído.

### DCT (Transformada Discreta de Cosseno)

A DCT transforma as imagens para o domínio da frequência, onde as decisões de fusão são
tomadas com base na magnitude dos coeficientes. Componentes de alta frequência
(detalhes) são preservados, enquanto os de baixa frequência (regiões suaves) são
combinados conforme a energia local.

### DTCWT (Transformada Wavelet Complexa de Árvore Dupla)

A DTCWT fornece uma decomposição multiescala e multidirecional das imagens. Ela captura
feições em diferentes escalas e orientações, permitindo regras de fusão sofisticadas que
preservam tanto a textura quanto a informação estrutural.

### GFG-FGF

Implementa Fu et al. (2024): o mapa de foco grosseiro |I − mean(I)| é enriquecido com o
gradiente de quatro vizinhanças do tipo gaussiano — o modelo 3×3 T = [1 2 1; 2 4 2; 1 2 1]/16
dá P = I⊗T, e G = (|I(x±1,y) − P|)/2 + (|I(x,y±1) − P|)/2, com valores abaixo de
T0 = 0,005 descartados — e depois refinado por um filtro guiado rápido. Cada pixel
escolhe o quadro com o maior valor de foco (empates: medida de pequena região com Sobel
3×3), e os mapas de decisão binária são suavizados pelo filtro guiado rápido antes da
mistura ponderada.

### Fusão por IA (StackMFF-V3)

Uma abordagem de aprendizado profundo que usa uma rede neural treinada em grandes
conjuntos de pares de imagens multifoco. A rede aprende estratégias de fusão ótimas
automaticamente, oferecendo resultados de ponta quando configurada corretamente.

---

*O OpenFocus é um projeto de código aberto. Para atualizações, relatos de bugs e
sugestões de recursos, visite o repositório do projeto.*
