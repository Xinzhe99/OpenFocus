# Manual do usuário do OpenFocus

## Sumário

1. [Introdução](#1-introdução)
2. [Primeiros passos](#2-primeiros-passos)
3. [Visão geral da interface principal](#3-visão-geral-da-interface-principal)
4. [Importação de imagens e vídeo](#4-importação-de-imagens-e-vídeo)
5. [Visualização e navegação pelas imagens](#5-visualização-e-navegação-pelas-imagens)
6. [Visualização de comparação cortina](#6-visualização-de-comparação-cortina)
7. [Registro de imagens](#7-registro-de-imagens)
8. [Métodos de fusão de imagens](#8-métodos-de-fusão-de-imagens)
9. [Transformações de imagem](#9-transformações-de-imagem)
10. [Retoque, rótulos e anotações](#10-retoque-rótulos-e-anotações)
11. [Configurações](#11-configurações)
12. [Processamento em lote](#12-processamento-em-lote)
13. [Exportação de resultados](#13-exportação-de-resultados)
14. [Interface de linha de comando](#14-interface-de-linha-de-comando)
15. [Atalhos de teclado](#15-atalhos-de-teclado)
16. [Solução de problemas](#16-solução-de-problemas)

---

## 1. Introdução

O OpenFocus é um aplicativo desktop profissional de fusão de imagens multifoco, voltado a pesquisadores, engenheiros e profissionais de processamento de imagens. Ele processa sequências de imagens com diferentes planos de foco usando algoritmos avançados de registro e fusão, e gera imagens completamente nítidas com profundidade de campo total.

### Principais recursos

- **Fusão de imagens multifoco**: combina várias imagens com pontos de foco diferentes em uma única imagem totalmente focada
- **Dez algoritmos de fusão**: filtro guiado, DCT, DTCWT, GFG-FGF, StackMFF V1/V2/V3, OpenFocus A/B/C, além do modo somente alinhamento
- **Registro de imagens**: alinha sequências desalinhadas com OpenFocus (rápido), ECC, homografia ou a cadeia homografia → ECC
- **Fundo automático**: substituir um fundo empilhado irregular por um fotograma real e ver onde nenhum fotograma captou a cena com nitidez
- **Visualização de comparação cortina**: verifica o alinhamento e compara resultados com um divisor A/B
- **Arquivos de projeto**: restauram referências de entrada e configurações reproduzíveis; os pixels do resultado são exportados separadamente
- **Modelos de processamento**: salvam e reaplicam configurações de fusão, registro e recursos a qualquer pilha carregada
- **TIFF multipágina**: pilhas Z de um único arquivo são carregadas página a página e exportadas de volta para um único arquivo
- **Mapa de profundidade**: mapas de posição de foco por pixel com esquemas de pseudocor personalizados
- **Barra de escala**: barras de escala com qualidade de publicação e calibração automática em µm/px
- **Recuperação automática após falhas**: a última sessão de trabalho é oferecida após um travamento
- **Multilíngue**: doze idiomas de interface — English, 中文, 日本語, Español, Deutsch, Français, Русский, 한국어, Português, Italiano, Nederlands, Polski — com detecção automática do idioma do sistema
- **CLI headless e para agentes**: automatiza pipelines headless para CI ou controla uma janela em execução pelo canal de controle local
- **Exportação flexível**: salva resultados como imagens individuais, pastas ou animações GIF
- **Transformações de imagem**: rotação, inversão e redimensionamento de pilhas
- **Rótulos**: adiciona texto à pilha de origem, à pilha alinhada ou a saídas específicas
- **Configurações persistentes**: preferências, idioma e pilhas abertas recentemente sobrevivem a reinicializações

### Formatos de entrada compatíveis

- Arquivos de imagem: JPG/JPEG, PNG, BMP, TIFF, WebP e outros formatos comuns
- Arquivos de vídeo: MP4, AVI, MOV, MKV, WMV, FLV e WEBM (decodificados em sequências de imagens)
- Pastas com sequências de imagens numeradas
- Arquivos RAW de câmera (DNG, CR2/CR3, NEF, ARW, RAF, ORF, RW2 e outros formatos RAW comuns). A decodificação usa `rawpy` (LibRaw), instalado por `pip install openfocus` e incluído nas compilações Windows/macOS

Pilhas PNG transparentes podem ser fundidas no aplicativo desktop, no processamento em lote desktop ou na interface de linha de comando, sem fusão agrupada. Cada quadro deve ser um PNG de 8 ou 16 bits, com as mesmas dimensões e a mesma profundidade de bits, e a mesma máscara alfa. A máscara deve ser zero fora de um único retângulo não vazio e ter opacidade constante diferente de zero em todo o retângulo. Opacidade constante parcialmente transparente é permitida. O retângulo visível é fundido de forma independente, então cores ocultas não podem influenciar a medição de foco nem os limites dos blocos. A saída PNG conserva o tamanho original da tela e a opacidade; as cores ocultas da saída são zero. As visualizações de origem e resultado do desktop mostram a transparência sobre um tabuleiro de xadrez.

Carregue em tamanho real, escolha nenhum alinhamento, desative a prévia rápida e a ROI, então renderize e salve como PNG. O modo em lote exige um método de fusão e saída PNG sem salvar uma pilha alinhada. Alinhamento, redimensionamento ou redução, correção de campo plano, mapas de profundidade, comparação de métodos, retoque e barras de escala não são suportados para dados transparentes, e rótulos não podem ser gravados em uma exportação. Um método de fusão que altera as dimensões do retângulo é rejeitado. Máscaras variáveis ou com bordas suavizadas, máscaras diferentes entre quadros, entrada TIFF transparente e saída não PNG, incluindo GIF e TIFF multipágina, não são suportadas. A CLI também rejeita a limpeza do sensor e os mapas de profundidade para entrada transparente.

Por exemplo, o equivalente na CLI é `python main.py --input ./png-stack --output ./result.png --method guided_filter --align none`.

### Formatos de saída compatíveis

Em **Salvar como**, TIFF comum salva o resultado como uma única imagem; TIFF em camadas inclui os quadros de origem como camadas. Linear DNG aparece sempre, mas salvá-lo exige um resultado linear da câmera sem ajustes incorporados, obtido com **Abrir RAW nativo** e com calibração da câmera.

- Imagens individuais: JPG, PNG, BMP, TIFF, WebP
- Sequências de imagens: salvas como pastas
- Animações: formato GIF
- Modelos 3D: OBJ com MTL, textura PNG e metadados JSON

---

## 2. Primeiros passos

### Iniciar o aplicativo

A forma preferida de instalar uma compilação empacotada é o site do projeto, **https://openfocus.top**: ele aponta para o instalador atual do Windows, a compilação do macOS e o pacote do Linux, com scripts de instalação para terminal. Prefira-o a recompilar: esses artefatos são a saída exata do pipeline de lançamento. `pip install openfocus` instala apenas a interface de linha de comando headless (a wheel não depende de PyQt6 de propósito); use uma compilação desktop ou uma cópia do código-fonte para a GUI.

Para instalar pelo terminal, visualize primeiro o comando e depois execute-o sem `--dry-run`: no macOS ou Linux, `curl -fsSL https://openfocus.top/install.sh | sh -s -- --dry-run` e depois `curl -fsSL https://openfocus.top/install.sh | sh` (no Linux instala a CLI); no Windows, `irm https://openfocus.top/install.ps1 | iex` no PowerShell.

Após a instalação, inicie o OpenFocus de uma destas formas:

1. **Windows (executável)**: dê dois cliques no atalho ou no executável do OpenFocus
2. **macOS**: arraste `OpenFocus.app` para Aplicativos e abra-o de lá
3. **Código-fonte Python**: execute `pip install -r requirements.txt` e depois `python main.py` no ambiente ativado

A primeira inicialização abre a janela de boas-vindas, onde você pode escolher o tema e carregar a pilha de demonstração incluída. Um perfil novo usa o tema claro por padrão; a escolha é lembrada.

### Fluxo de início rápido

1. Importe a sequência de imagens (Arquivo → Abrir pasta ou arrastar e soltar)
2. Configure as opções de fusão no painel direito
3. O registro começa em **ECC**; escolha **Sem alinhamento** se os quadros já estiverem alinhados
4. Ajuste o tamanho do kernel, se preciso
5. *(opcional)* Abra **Configurações de renderização** e marque **Prévia rápida** para testar os parâmetros primeiro em um rascunho reduzido
6. Clique em **Iniciar renderização** para começar o processamento
7. Exporte os resultados

---

## 3. Visão geral da interface principal

A interface do OpenFocus tem três áreas principais:

### Painel esquerdo: área de exibição de imagens

A parte esquerda da janela mostra as imagens em uma visualização dividida:

- **Painel de imagem de origem (superior)**: mostra a sequência de imagens de entrada
- **Painel de imagem de resultado (inferior)**: mostra os resultados de fusão ou registro
- **Controle deslizante de navegação**: use o controle abaixo de cada painel para percorrer os quadros
- **Rótulos informativos**: mostram a posição do quadro atual e as dimensões da imagem

Você pode redimensionar os painéis arrastando o divisor entre eles.

### Painel direito: painel de controle

O painel direito reúne todas as opções de configuração, de cima para baixo:

1. **Configurações de fusão**: seleção do algoritmo e de seus parâmetros
2. **Configurações de registro**: ativação ou desativação dos métodos de alinhamento
3. **Botões de ação**: Iniciar renderização e as ações secundárias Redefinir, Configurações de renderização e renderizar todos os métodos. A janela de configurações reúne prévia rápida, ordem dos quadros, igualação de exposição, remoção de deriva de brilho e nitidez da saída
4. **Lista de imagens de origem**: todas as imagens carregadas com seus nomes de arquivo
5. **Lista de saídas**: os resultados gerados

### Barra de menus

A barra de menus superior dá acesso a todas as funções:

- **Arquivo**: operações de importação e exportação, projetos, modelos de processamento e lista de arquivos recentes
- **Editar**: transformações de imagem, retoque e rótulos
- **Ferramentas**: processamento em lote, comparação de métodos, mapa de profundidade, correção de campo plano, barra de escala e rótulos
- **Configurações**: configurações avançadas (desempenho, registro e diagnósticos de alinhamento, limpeza do sensor, nomes de saída, IA e inicialização), aceleração por GPU, tema e idioma
- **Ajuda**: informações do ambiente, contato, verificação de atualizações e pasta de logs

---

## 4. Importação de imagens e vídeo

### Abrir uma pasta de imagens

1. Vá em **Arquivo → Abrir pasta** ou pressione `Ctrl+O`
2. Selecione uma pasta contendo a sequência de imagens
3. Uma janela de escala de carregamento aparece: escolha a escala (100% = resolução total)
4. Clique em OK para carregar as imagens

Marque **Não perguntar novamente** nessa janela para sempre carregar na escala escolhida sem aviso; **Configurações → Configurações avançadas → Perguntar escala de carregamento na importação** reativa o aviso. Uma pilha já carregada ainda pode ser reduzida depois com **Editar → Redimensionar**.

O software executa automaticamente:

- Detecção e ordenação dos arquivos de imagem por nome, em ordem natural (por exemplo, `frame2` antes de `frame10`)
- Exibição da primeira imagem no painel de origem
- Preenchimento da lista de imagens de origem com todos os quadros

### Abrir arquivos de vídeo

1. Vá em **Arquivo → Abrir vídeo** ou pressione `Ctrl+Shift+O`
2. Selecione um arquivo de vídeo (MP4, AVI, MOV, MKV, WMV, FLV, WEBM)
3. O vídeo é decodificado automaticamente em quadros individuais
4. Cada quadro passa a fazer parte da pilha de imagens

### Arquivos de projeto

Arquivo → Salvar projeto armazena referências de entrada e configurações reproduzíveis em um `.ofproj`: ordem exata dos quadros (incluindo páginas TIFF e quadros duplicados), exclusões, rotações/inversões, escala de carregamento, parâmetros de renderização por método, rótulos, estilo da barra de escala e quadro atual. Os pixels do resultado e do retoque devem ser exportados separadamente. Operações que não podem ser reconstruídas a partir dos arquivos de origem exigem salvar antes uma nova pilha de entrada.

Arquivo → Abrir projeto reconstrói os quadros registrados antes de substituir a pilha atual. Se as origens foram movidas, escolha a nova raiz de origem e confirme o mapeamento completo entre caminhos antigos e novos. Salve o projeto novamente para manter as referências realocadas.

Um projeto se comporta como um documento: a barra de título mostra seu nome com um `*` enquanto houver mudanças não salvas, e fechar o aplicativo ou abrir outro projeto pergunta antes de descartá-las. As imagens de origem são referenciadas, não copiadas.

### Modelos de processamento

Um modelo de processamento contém configurações, nunca caminhos de origem. **Arquivo → Salvar modelo de processamento…** grava um `.oftemplate` com o método e os parâmetros de fusão atuais, o método e os parâmetros de registro, as opções de brilho por método, o número de threads, as configurações de blocos, a largura de redução do registro, o tamanho do lote de IA do StackMFF, a nitidez, a ordem inversa e o indicador de redução de ruído do sensor. O caminho de uma máscara de defeitos deliberadamente não é armazenado.

**Arquivo → Aplicar modelo de processamento…** aplica essas configurações à pilha já carregada; não substitui as origens atuais. O seletor também aceita um projeto `.ofproj` (as configurações são extraídas sem exigir suas origens) e uma receita `.json` que contenha configurações de projeto. Aplicar um modelo marca o projeto como modificado e pode alterar parâmetros de algoritmos e configurações de recursos, então revise o painel direito antes de renderizar. As receitas continuam sendo o formato headless e incluem o caminho da máscara de defeitos e uma impressão digital.

### Pilha de demonstração

A janela de boas-vindas oferece **Carregar pilha de demonstração**: seis fotos de exemplo incluídas, de uma pilha de foco real, para testar imediatamente o fluxo completo sem imagens próprias. Também é útil para verificar uma instalação.

### Curva de nitidez

Uma pequena curva acima do controle de navegação de origem mostra a nitidez de cada quadro (normalizada). Quadros desfocados aparecem como vales marcados com um ponto vermelho. Clique em qualquer ponto da curva para ir a esse quadro; passar o cursor mostra o número do quadro.

### Pilhas TIFF multipágina

Softwares de microscopia (ZEN, NIS-Elements, ImageJ/Fiji) frequentemente exportam toda uma pilha Z como UM único arquivo TIFF multipágina. O OpenFocus lê cada página desse arquivo como um quadro individual da pilha: solte ou abra o arquivo e a pilha aparece completa (os quadros são nomeados `<file>_page_001` e assim por diante). TIFFs de página única se comportam exatamente como antes. O inverso também funciona: **Arquivo → Salvar pilha → (Pilha registrada / Pilha de entrada) → Salvar como TIFF multipágina** grava a pilha inteira em um único `.tif`, preservando a profundidade de 16 bits.

### Barra de escala (figuras para publicação)

Se o destino escolhido não tiver metadados utilizáveis, a calibração manual será selecionada e exigirá um tamanho de pixel válido.

Ferramentas → Barra de escala… configura uma barra de escala gravada nas imagens exportadas. O tamanho do pixel (µm/px) é detectado automaticamente nos metadados TIFF (convenção `unit=micron` do ImageJ, `PhysicalSizeX` do OME-TIFF ou unidade de resolução explícita em centímetros), então pilhas de microscopia (incluindo TIFF multipágina) se calibram sozinhas. A resolução de impressora (72/300 dpi e similares) deliberadamente *não* é tratada como calibração, e um valor digitado manualmente sempre prevalece sobre os metadados detectados: uma barra errada mas plausível em uma figura é pior que nenhuma. Escolha o canto e a cor (Auto inverte conforme o fundo); a janela mostra uma prévia ao vivo. Com a barra ativada, toda exportação (imagem única, pilhas em pastas, TIFF multipágina, GIF, exportar tudo, lote) carrega uma barra em estilo de publicação: um comprimento arredondado da série 1-2-5 (~12% da largura da imagem) com texto contornado e legível. A CLI oferece suporte com `--scale-bar` (calibração automática) e `--scale-um <um/px>`.

A prévia é desenhada na geometria nativa em pixels do destino antes de ser ajustada à área visível. Ajustar, 100%, deslocar e navegar entre quadros mudam apenas a visualização. Aplicar confirma uma única edição desfazível; Cancelar preserva a calibração anterior. Saídas históricas mantêm sua própria calibração e estilo quando outra pilha de origem é carregada.

Para removê-la, use **Ferramentas → Barra de escala → Remover barra de escala**: *Da pilha de origem* e *Do resultado* restringem o escopo "Aplicar a" da janela à tela restante, enquanto *De todos os lugares* desativa a barra. Ambas as telas e a comparação A/B são redesenhadas imediatamente; remover a barra de uma tela que nunca a mostrou deixa a configuração inalterada.

### Mapa de profundidade (posição de foco)

**Ferramentas → Mapa de profundidade…** estima posições de foco usando a medida de atividade do método selecionado. Não é uma superfície medida nem um mapa exato da origem dos pixels: métodos de pirâmide/wavelet estimam atividade e as variantes StackMFF informam uma posição de quadro estimada. Escolha os quadros marcados atuais ou a origem registrada da saída selecionada. Proveniência histórica ausente nunca é substituída pela pilha atual. A origem e a textura permanecem vinculadas a essa captura. Mapas de profundidade não estão disponíveis para pilhas PNG transparentes, nem na GUI nem na CLI.

Todos os sete métodos com capacidade de profundidade — DCT, DTCWT, GFG-FGF, OpenFocus B, OpenFocus C, StackMFF V2 e StackMFF V3 — exigem um clique explícito em **Calcular**. StackMFF V1 não tem mapa de profundidade. Abrir a janela ou mudar a medida, origem, registro ou parâmetros nunca inicia o cálculo. O editor de parâmetros aparece logo abaixo do seletor de medida: o kernel mediano DCT e o kernel inicial GFG usam valores ímpares de 1 a 51 (7 por padrão). OpenFocus B usa um raio de resposta de foco e suavização da profundidade; OpenFocus C usa **Suavização da mistura** para suavizar seu mapa de profundidade, sem raio. Os níveis de decomposição DTCWT são editáveis aqui como rascunho local, inicializado com os parâmetros da origem. StackMFF V3 mostra seu próprio modo de saída de IA (mistura suave por padrão, ou escolha rígida) e uma explicação em vez de um kernel numérico: o mapa informa uma posição de quadro contínua, não uma distância física. A mistura suave usa a posição ponderada por probabilidade; a escolha rígida usa o quadro mais provável por pixel. StackMFF V2 não tem parâmetros e informa a camada focal amostrada por sua rede. Os rascunhos partem dos parâmetros da saída correspondente ou das configurações lembradas do método na janela principal. São lembrados por origem e método enquanto a janela está aberta; editá-los não altera a janela principal, o projeto nem a saída registrada.

Mudar as configurações de cálculo marca o mapa 2D anterior como **desatualizado**, limpa a superfície 3D e bloqueia todas as exportações, incluindo coloridas, brutas, físicas e OBJ, até que um novo cálculo seja concluído com sucesso. Um cálculo ativo é cancelado cooperativamente; após o término do cancelamento, pressione **Calcular** novamente. Nenhum cálculo seguinte é enfileirado. Calcular permanece visível e fica desabilitado durante cálculo, cancelamento ou exportação.

- **Mapa de cores**: escala de cinza (padrão de publicação), 15 paletas científicas (Turbo, Viridis, Magma, …) e as suas próprias — *Novo esquema personalizado…* monta um gradiente a partir de qualquer número de paradas de cor (dê dois cliques em uma parada para editar posição e cor); esquemas personalizados são salvos e oferecidos na lista.
- **Inverter / Gama / Suavização**: inverte o eixo de profundidade, reequilibra o tom (gama < 1 clareia índices baixos) e remove o ruído salpicado com uma passada que respeita bordas.
- **Sobreposição**: disponível apenas para a saída associada com coordenadas correspondentes.
- **Exportação bruta**: o TIFF de 16 bits armazena um índice normalizado codificado de 0 a 65535; não são dados float sem perda nem uma distância física. JPEG/BMP são recusados.
- **TIFF de índices e TIFF de imagem alternativa**: exporta a estimativa de profundidade atual como índices de quadro sem sinal de 16 bits ou como uma imagem montada a partir dos pixels de origem. Os valores 1–N referem-se à ordem exata de quadros usada como entrada do cálculo de profundidade, após qualquer registro e reordenação; 0 marca pixels inválidos. No máximo 65.535 quadros são suportados. Posições fracionárias estimadas são arredondadas para o quadro mais próximo, com valores intermediários arredondados para cima. Esses índices estimam a posição de foco; não são a proveniência real dos pixels de um resultado fundido. A imagem alternativa seleciona o pixel do quadro de entrada de profundidade correspondente em cada local válido, com pixels inválidos zerados. Quando alguma entrada é de 16 bits, valores de 8 bits são promovidos para 16 bits multiplicando-os por 257. As exportações usam a ROI do mapa de profundidade e preservam áreas inválidas como zero.
- **Profundidade física**: informe início Z, passo positivo, unidade e direção de aquisição, depois confirme o mapeamento original de captura. Alternar entre a pilha atual e uma saída limpa a confirmação, mesmo que nomes e índices coincidam; reconfirme antes de exportar. Quadros excluídos mantêm seus intervalos. O TIFF Float32 armazena micrômetros, NaN para dados ausentes e metadados de origem. Um OBJ físico também precisa de calibração XY; usa escala Z física 1:1.
- **Modelo 3D / Textura 3D**: escolha o lado mais longo da grade (600 vértices por padrão). Cada janela recém-aberta usa **Automático (cor real)**: usa a saída associada somente quando identidade e coordenadas/ROI coincidem com o mapa concluído; caso contrário, usa o primeiro quadro participante daquele cálculo concluído, após registro e qualquer reordenação. O nome da saída ou do quadro resolvido é exibido. Você pode escolher explicitamente a saída correspondente, um quadro participante ou **Pseudocor de profundidade**; imagens indisponíveis são relatadas, nunca substituídas por saídas não relacionadas. Essa textura é compartilhada pela prévia 3D e pelo PNG de textura do OBJ. O mapa 2D mantém sua própria paleta e sobreposição: mudar paleta ou Gama não recolore uma textura em cores reais. O pseudocor explícito segue paleta, Gama e inversão. A prévia estabilizada e a exportação compartilham o mesmo campo de altura amostrado, sem suavização oculta extra. O relevo relativo é expresso em unidades de imagem.
- **Exportação do modelo**: OBJ, MTL, textura PNG e metadados JSON formam um grupo de arquivos; mantenha-os juntos. A exportação roda em segundo plano. Conflitos são verificados para cada membro, a preparação é organizada em etapas, o cancelamento mantém os arquivos existentes e falhas de confirmação revertem substituições. Salvar um modelo não exporta sua imagem de fusão pai nem remove a proteção de alterações não salvas dessa imagem.

O cálculo roda em segundo plano com progresso e cancelamento; os métodos de IA reutilizam o mesmo processamento em blocos seguro para memória da fusão por IA, então pilhas profundas funcionam. O equivalente na CLI é `--depth-map [path]` com `--depth-colormap`; ele grava apenas o mapa colorido (padrão `<output>_depth.png`), funciona somente para os métodos com capacidade de profundidade acima e é rejeitado para entrada transparente ou agrupada. A GUI também exporta profundidade bruta, índices de quadro, imagens alternativas, profundidade física e modelos 3D.

### Recuperação automática após travamento

O OpenFocus mantém continuamente uma captura da sessão de trabalho (pilha, configurações, rótulos, quadro atual). Se o aplicativo fechar inesperadamente — travamento, queda de energia ou encerramento forçado —, a próxima inicialização detecta e oferece **Restaurar sessão** para voltar exatamente aonde você estava. Saídas limpas nunca mostram o aviso. Os dados de recuperação ficam junto às configurações (instalações portáteis os mantêm ao lado do executável, para viajarem com você).

### Arquivos recentes

Cada pilha aberta é lembrada em **Arquivo → Abrir recentes** (até 8 entradas, numeradas no submenu). Clique em uma entrada para reabri-la; não há atalho de teclado global para arquivos recentes. Reabrir a mesma pilha a move para o topo; entradas cujos arquivos não existem mais podem ser removidas com confirmação, e **Limpar lista recente** esvazia o menu.

### Arrastar e soltar

Você também pode arrastar uma pasta diretamente para a janela do aplicativo para importar imagens. Soltar mais arquivos sobre uma pilha existente os adiciona ao final (após verificação de tamanho).

### Orientação EXIF

Arquivos JPEG/TIFF de câmera com tag EXIF Orientation são girados/invertidos automaticamente durante a importação, para que fotos em retrato sejam carregadas em pé.

### Origens de 16 bits

Entradas PNG/TIFF de 16 bits são detectadas automaticamente e preservadas de ponta a ponta: a fusão roda em profundidade total e os resultados podem ser salvos novamente como PNG/TIFF de 16 bits (exportações JPG/BMP são convertidas para 8 bits, pois esses formatos não armazenam 16 bits).

Arquivos RAW de câmera são revelados em 16 bits (balanço de branco da câmera, gama sRGB) com o clareamento automático desativado, para que cada quadro da pilha receba o mesmo mapeamento de exposição; exposição automática por quadro faria o brilho piscar entre quadros. Salve como TIFF ou PNG para manter os dados de 16 bits.

### Perfis de cor

Exportações individuais em PNG, TIFF, JPEG e WebP mantêm o perfil ICC RGB quando todos os quadros de origem têm o mesmo perfil incorporado. Pilhas mistas ou sem perfil são exportadas sem perfil ICC. O OpenFocus preserva o perfil; não converte os pixels de origem para um espaço de cor comum nem faz gerenciamento de cor na prévia. Converta imagens com perfis diferentes para um espaço comum antes de carregá-las. A entrada RAW é revelada em sRGB e não herda o perfil da câmera.

Em um TIFF multipágina, todas as páginas do arquivo devem ter o mesmo perfil RGB válido; caso contrário, seus quadros são tratados como sem perfil. Quadros de vídeo também são tratados como sem perfil. Quadros adicionados e os selecionados para uma renderização contribuem para seu perfil, que permanece com esse resultado para exportações posteriores.

### Limpar a pilha

Para remover todas as imagens carregadas, vá em **Arquivo → Limpar pilha** ou pressione `Ctrl+W`.

---

## 5. Visualização e navegação pelas imagens

### Navegar pela pilha de imagens

- **Controle deslizante**: arraste o controle abaixo do painel de origem para percorrer os quadros
- **Lista de arquivos**: clique em qualquer nome de arquivo na lista de imagens de origem do painel direito; com a lista em foco, Cima/Baixo movem a seleção e o painel de origem a acompanha
- **Teclado**: não há atalho global de quadro anterior/seguinte; coloque o foco na lista de origem primeiro

### Revisar quadros semelhantes

Clique em **Revisar quadros** abaixo do cabeçalho da lista de origem para encontrar quadros adjacentes cujos pixels são quase idênticos. O botão muda para **Cancelar revisão** enquanto a análise roda. Cada linha sugerida indica seu quadro de referência e a diferença; inspecione ambos antes de usar a caixa de origem para excluir um quadro da fusão. As sugestões nunca alteram caixas nem excluem arquivos. Essa comparação conservadora pode não detectar quadros semelhantes; mudanças restritas a uma pequena região focada não são suavizadas pela média. Os resultados são descartados se a pilha mudar. As sugestões são mostradas apenas quando tanto o quadro sugerido quanto sua referência estão incluídos; alterar uma caixa atualiza as sugestões exibidas.

### Zoom e deslocamento

- **Aumentar/Diminuir zoom**: roda do mouse sobre a imagem
- **Zoom fino**: mantenha `Ctrl` e use a roda para incrementos menores
- **Ajustar à janela**: clique duas vezes na imagem para alternar entre ajuste e visualização em 100%
- **Deslocar**: clique e arraste para mover a imagem dentro do painel

### Navegação sincronizada

Ao navegar no painel de origem, o painel de resultado mostra o quadro de resultado correspondente, se disponível.

---

## 6. Visualização de comparação cortina

A visualização Cortina sobrepõe duas imagens em um único quadro com um divisor vertical arrastável, tornando desalinhamentos e diferenças de qualidade diretamente visíveis dos dois lados.

### Abrir a visualização Cortina

1. Renderize pelo menos um resultado (fusão ou registro)
2. Pressione o botão **Cortina** no canto superior direito da barra de título do painel de resultado
3. A área de resultado muda para a visualização de comparação; pressione **Cortina** novamente para voltar

### Interpretar a visualização

- **Lado A (à esquerda do divisor)**: um quadro de origem — por padrão, segue o controle do painel de origem
- **Lado B (à direita do divisor)**: a renderização mais recente
- **Divisor branco + alça redonda**: arraste horizontalmente para mover a divisão
- Arrastar bordas que cruzam o divisor revela erros de registro difíceis de ver em painéis lado a lado

### Escolher o que comparar

A barra na parte inferior da visualização tem um grupo por lado:

- **Lado A**: marque **Seguir origem** (padrão) para acompanhar o controle do painel de origem, ou desmarque para fixar o lado A em qualquer quadro com seu próprio controle
- **Lado B**: marque **Último resultado** (padrão) para sempre mostrar a renderização mais recente, ou desmarque para percorrer o histórico de saídas com seu controle

Ambos os lados são atualizados ao vivo: novas renderizações, movimentos do controle de origem e exclusões de quadros são refletidos imediatamente.

### Zoom e deslocamento

A visualização de comparação compartilha uma única transformação para as duas imagens:

- **Zoom**: roda do mouse (zoom ao redor do cursor)
- **Deslocar**: clique e arraste em qualquer lugar, exceto no divisor
- **Redefinir**: clique duas vezes

O modo Cortina e o modo ROI compartilham o painel de resultado, então entrar em um sai do outro. Limpar a pilha também sai do modo Cortina.

---

## 7. Registro de imagens

O registro corrige o desalinhamento espacial entre os quadros da pilha. É essencial quando as imagens apresentam pequenos deslocamentos ou mudanças de perspectiva.

### Métodos de registro

#### ECC (Enhanced Correlation Coefficient)

- Usa otimização para encontrar o melhor alinhamento com base na correlação
- Adequado a desalinhamentos sutis
- Precisão subpixel
- Mais lento que a homografia, mas mais preciso

#### Homografia

- Baseada na correspondência de pontos característicos (SIFT/ORB)
- Lida com transformações geométricas maiores
- Mais rápida que o ECC para alinhamento inicial
- Boa para imagens com mudanças significativas de perspectiva

#### OpenFocus (rápido)

- Alinhamento por similaridade limitada (translação + rotação + escala), com equalização de brilho por quadro e corte automático
- Busca transformações de similaridade limitadas usando diferenças de imagem amostradas e aplica warp com interpolação Lanczos
- Quadros com deslocamento superior a **8% do tamanho do quadro permanecem sem alinhamento** em vez de serem corrigidos parcialmente. Altere o limite com `--align-shift-limit` na CLI
- Para geometria não rígida ou extrema, escolha ECC ou homografia
- **Tratamento de quadros rejeitados**: por padrão, continua com o quadro não alinhado. Escolha **Parar em quadro rejeitado** para falhar a renderização ou o item de lote antes de produzir um resultado quando a qualidade do detector for inferior a 0,06 ou o movimento estimado exceder o limite de deslocamento configurado, 8° de rotação ou 5% de mudança de escala. Os quadros são numerados a partir de 1 na ordem da pilha processada. A referência geométrica é o quadro 1 por padrão; uma correspondência direta fraca pode ser tentada novamente no quadro adjacente mais próximo da referência. Essa política detecta estimativas rejeitadas, não todas as correspondências incorretas: baixa textura, padrões repetidos, oclusão, movimento amplo ou mudanças de foco ainda podem enganar o registro. Inspecione o resultado.
- Na CLI, selecione a mesma política com `--align openfocus --align-param 'failure_policy="fail"'` (use as aspas adequadas ao shell). Itens de lote usam a política selecionada de forma independente. O padrão é `"continue"`.
- Defina **Quadro de referência** como um número baseado em 1 na ordem da pilha incluída e processada para ancorar a geometria nesse quadro. A ordem é preservada; a equalização de brilho usa sua própria referência. Cada pilha de lote deve conter esse número, ou o item falha. Exemplo de CLI: `--align openfocus --align-param reference_frame=3`. Escolher uma referência não corrige paralaxe nem uma correspondência ruim; inspecione o resultado.

Os relatórios de alinhamento identificam a referência e o resultado de cada quadro. Um fallback significa que o método não conseguiu estabelecer o alinhamento ou depende de uma estimativa anterior falha; o motivo informado distingue estimativas fracas e movimento fora dos limites quando conhecido. Os valores de qualidade são específicos de cada método, não uma garantia de precisão. As transformações mapeiam coordenadas de origem para a tela de referência da etapa de registro antes do corte comum; métodos compostos relatam suas etapas separadamente. Homografia e ECC também aceitam o parâmetro opcional `failure_policy="fail"` para parar quando seus algoritmos detectam falha. O padrão continua `"continue"`; inspecione os resultados mesmo quando o alinhamento for bem-sucedido.

Para inspecionar uma renderização, ative **Configurações → Configurações avançadas → Registro → Coletar diagnósticos de alinhamento** e renderize novamente. Isso recalcula o alinhamento em vez de reutilizar imagens em cache. O status de conclusão destaca fallbacks; abra **Configurações → Configurações avançadas → Registro → Diagnósticos de alinhamento** para o relatório por quadro. Ele mostra os motivos de falha disponíveis, pontuações específicas do método, contagens de correspondência, quadros referenciados e transformações. Números e nomes de referência remetem à lista de origem original, inclusive quando quadros são excluídos. Renderizações em lote e agrupadas não fornecem esse relatório.

### Ativar o registro

1. No painel direito, localize o grupo **Registro**
2. Escolha um método na lista: **Sem alinhamento**, **OpenFocus (rápido)**, **Alinhar (ECC)**, **Alinhar (homografia)** ou **Homografia → ECC** (encadeado)
3. Para OpenFocus (rápido), escolha um quadro de referência. Em **Configurações → Configurações avançadas → Registro**, escolha **Tratamento de quadros rejeitados**; ao alterá-lo, a escolha é aplicada aos métodos de registro
4. **Coletar diagnósticos de alinhamento** agora fica em Configurações avançadas (veja acima)

### Quando usar registro

- As imagens apresentam desalinhamento visível
- As imagens foram capturadas à mão
- A cena tem paralaxe (objetos próximos se deslocam entre quadros)
- Os resultados mostram artefatos fantasma ou duplicação

### Configurações de registro

As configurações adicionais ficam em **Configurações → Configurações avançadas → Registro**:

- **Largura de redução**: resolução de pré-processamento para registro (1024 px por padrão)
- Valores menores = processamento mais rápido, possivelmente menos preciso
- Valores maiores = processamento mais lento, possivelmente mais preciso
- **Coletar diagnósticos de alinhamento** e **Cache de registro** ficam na mesma seção

---

## 8. Métodos de fusão de imagens

O OpenFocus oferece dez algoritmos de fusão (além de um modo somente alinhamento). Um perfil novo começa em **StackMFF V3** com registro **ECC**; as listas abaixo descrevem todas as opções. Cada um tem características distintas adequadas a diferentes tipos de imagem.

### Filtro guiado

- **Algoritmo**: filtragem guiada para uma fusão que preserva bordas
- **Ideal para**: uso geral, equilíbrio entre velocidade e qualidade
- **Vantagens**: boa preservação de bordas, processamento rápido
- **Parâmetro**: tamanho do kernel (ajustável de 1 a 51, apenas valores ímpares)
  - Kernel menor: mais detalhe local, possivelmente mais ruído
  - Kernel maior: resultados mais suaves, melhor supressão de ruído

### DCT (Transformada Discreta de Cosseno)

- **Algoritmo**: fusão no domínio da frequência por coeficientes DCT
- **Ideal para**: imagens ricas em textura, imagem científica
- **Vantagens**: preserva texturas finas, base matemática sólida
- **Parâmetro**: o tamanho do kernel controla a janela de processamento

### DTCWT (Transformada Wavelet Complexa de Árvore Dupla)

- **Algoritmo**: fusão no domínio wavelet multiescala
- **Ideal para**: cenas complexas com vários níveis de detalhe
- **Vantagens**: excelente análise multiescala, boa direcionalidade
- **Regra de fusão**: uma decisão por pixel e nível para todas as orientações e canais de cor (N vias, com verificação de consistência majoritária); a banda passa-baixa segue a decisão mais grosseira em vez de calcular a média de todos os quadros
- **Parâmetro**: níveis de wavelet (1–6, 4 por padrão) no painel direito. O método não usa kernel espacial, então o controle de kernel fica oculto.

### GFG-FGF (gradiente de quatro vizinhanças do tipo gaussiano + filtro guiado rápido)

- **Algoritmo**: implementação de Fu et al. (*Laser & Optoelectronics Progress*, 2024): um mapa de foco por diferença de desfoque |I − mean(I)| é enriquecido com o operador de gradiente de quatro vizinhanças do tipo gaussiano (GFG), refinado por um filtro guiado rápido, decidido por pixel (empates resolvidos por uma medida de pequena região Sobel 3×3), e os mapas de decisão binários são suavizados novamente pelo filtro guiado rápido antes da mistura ponderada.
- **Ideal para**: microscopia e imagens com regiões de foco claras
- **Vantagens**: rápido, limites de região de foco nítidos; a memória não cresce com o número de quadros
- **Parâmetro**: kernel = tamanho do filtro de média do mapa de foco grosseiro

### StackMFF (V1, aprendizado profundo)

- **Algoritmo**: uma rede de atenção de escala única mistura a luminância das fontes com pesos softmax; Cb/Cr vêm do quadro cuja luminância é mais próxima da luminância fundida
- **Ideal para**: fusão neural de uma pilha de foco completa
- **Vantagens**: sem parâmetros e cor dentro da gama; a luminância fundida independe da ordem dos quadros, mas a crominância pode variar em caso de empate nas distâncias de luminância
- **Nota**: cerca de 156.172 parâmetros; pilhas maiores se beneficiam da GPU

### StackMFF V2 (aprendizado profundo)

- **Algoritmo**: uma rede de atenção de escala única normaliza a profundidade de foco esperada e amostra o quadro de origem mais próximo. Os índices de profundidade dependem da ordem da pilha. **Os quadros devem manter sua ordem original de foco** — a inversão é desativada para este método
- **Mapas de profundidade**: suportados; informa a camada focal amostrada
- **Ideal para**: varreduras de foco contínuas em que um modelo muito pequeno e rápido importa
- **Requisitos**: ONNX Runtime e o modelo ONNX incluído.

### StackMFF V3 (aprendizado profundo)

É o método pré-selecionado em uma nova instalação.

- **Algoritmo**: classificação de origem pixel a pixel (ONNX Runtime). A rede prediz, para cada pixel, de qual imagem de entrada ele vem, e então mistura as imagens por essas probabilidades ou escolhe uma imagem
- **Modo de saída de IA** (painel direito, apenas para este método):
  - **Mistura suave (ponderada)** — padrão; cada pixel é uma mistura das imagens ponderada por probabilidade. Transições suaves, sem emendas duras; corresponde à configuração relatada no artigo
  - **Escolha rígida (quadro único)** — cada pixel é copiado do quadro mais provável. Seleção mais nítida, sem interpolação entre quadros; uma emenda no nível do pixel pode aparecer onde dois quadros empatam
- **Ideal para**: qualidade máxima quando há GPU disponível
- **Vantagens**: invariante à ordem (a ordem dos quadros nunca muda o resultado), aceita pilhas dentro dos recursos disponíveis, sem ajuste por método, otimização automática
- **Requisitos**: ONNX Runtime e o modelo ONNX incluído.

Cada bloco neural usa todos os quadros. Quadros de origem e dados temporários de alinhamento podem usar o disco; espaço livre e memória disponível limitam o tamanho da pilha. Os blocos espaciais de V2 aproximam a inferência da imagem inteira.


### OpenFocus A / B / C

- **OpenFocus A**: suaviza seu mapa de resposta de foco multiescala e então seleciona os pixels em foco.
- **OpenFocus B**: mistura quadros vizinhos guiado por um mapa de profundidade suavizado.
- **OpenFocus C**: reconstrói bandas multiescala guiado por um mapa de profundidade suavizado.
- **Parâmetros**: A e B usam o controle de kernel para a escala de resposta de foco; C não tem raio. Os três têm um parâmetro de suavização que controla a força das transições.

**Suavização**: um único campo numérico abaixo do controle de kernel, cujo significado depende do método selecionado (intervalo 0–8, passo 0,2):

| Método | Rótulo | Padrão | Efeito |
|---|---|---|---|
| OpenFocus A | Suavização da mistura | 2,0 | Controla a suavização do mapa de resposta de foco antes da seleção dos pixels |
| OpenFocus B | Suavização da profundidade | 1,2 | Controla a suavização do mapa de profundidade que guia a mistura |
| OpenFocus C | Suavização da mistura | 3,0 | Controla a suavização do mapa de profundidade que guia a reconstrução multiescala |

Valores mais altos aumentam a suavização das transições. Cada método lembra seu próprio valor entre sessões, e o processamento em lote aplica a mesma regra.

C não tem parâmetro de raio, então seu controle de kernel fica oculto.

CLI: `--depth-smoothing VALUE` (B) e `--blend-smoothing VALUE` (A, C).

**As opções de brilho são por método.** *Igualar exposição antes da fusão* e *Remover deriva de brilho* são lembradas separadamente para cada método de fusão, porque respondem a uma questão de cada pilha e não a uma preferência global. Ambas ficam ativadas por padrão em OpenFocus A/B/C (que não têm calibração de exposição integrada); os outros métodos mantêm os padrões históricos (igualação de exposição ativada, remoção de deriva desativada). Trocar de método mostra os valores daquele método, e esses valores sobrevivem ao reinício. **Redefinir** limpa os valores lembrados de brilho, suavização e raio de todos os métodos.

**Nitidez da saída fundida** aplica uma máscara de nitidez suave após a fusão, antes de o resultado ser convertido para sua profundidade de saída. Fica desativada por padrão e pode ser ativada em **Configurações de renderização** ou nas opções do lote; o sinalizador equivalente da CLI é `--sharpen-output`. Afeta a imagem fundida, não os quadros registrados nem os mapas de profundidade. Pode tornar as bordas mais distintas, mas amplificar ruído e halos. É salva com o projeto e lembrada entre sessões.

### Somente alinhamento (sem fusão)

Selecione **Somente alinhamento (sem fusão)** para registrar a pilha e adicionar uma entrada de pilha à lista de saídas. Os quadros permanecem na memória até você exportá-los; o somente alinhamento não cria cache extra nem grava ao lado dos arquivos de origem. Salve a entrada para criar uma pasta na destinação escolhida. Os nomes de arquivo originais são mantidos, com a extensão do formato escolhido.

### Selecionar um método de fusão

1. No painel direito, localize o grupo **Fusão**
2. Escolha um método na **lista suspensa** (exatamente um está sempre selecionado; a primeira entrada registra sem fundir)
3. Ajuste o tamanho do kernel, se aplicável — o rótulo mostra o que o kernel significa para o método selecionado
4. Clique em **Iniciar renderização** para processar

### Guia de comparação de algoritmos

| Algoritmo | Velocidade | Qualidade | Ideal para |
|-----------|-------|---------|----------|
| Filtro guiado | Rápida | Boa | Uso geral |
| DCT | Média | Boa | Texturas |
| DTCWT | Média | Muito boa | Cenas complexas |
| GFG-FGF | Rápida | Boa | Regiões de foco |
| StackMFF (V1) | Lenta (GPU) | Não avaliada aqui | Luminância independente da ordem; crominância pode variar em caso de empate nas distâncias de luminância |
| StackMFF V2 | Rápida (GPU) | Não avaliada aqui | Varreduras contínuas com um modelo minúsculo |
| StackMFF V3 | Lenta (GPU) | Excelente | Qualidade máxima |
| OpenFocus A | Depende da pilha | Depende da pilha | Seleção multiescala de pixels em foco |
| OpenFocus B | Depende da pilha | Depende da pilha | Mistura de quadros vizinhos guiada por profundidade |
| OpenFocus C | Depende da pilha | Depende da pilha | Reconstrução multiescala guiada por profundidade |

Abra **Ferramentas → Comparação de métodos** para ajustar vários métodos independentemente. Nome, rótulos de parâmetros e controles de cada método permanecem juntos quando a janela estreita. As mudanças atualizam a prévia automaticamente. **Prévia ROI (rápida)** funde a área selecionada com o contexto ao redor; **Fusão de imagem inteira** verifica o resultado da imagem toda e permite inspecionar outra área sem renderizar novamente. Resultados ROI podem diferir dos de imagem inteira porque estatísticas de exposição e alguns algoritmos dependem da extensão da imagem.
A **fusão de imagem inteira** é selecionada por padrão.

Uma dica visível explica como selecionar ou desmarcar um resultado: clique nele ou use `Tab` para colocá-lo em foco e pressione `Enter` ou `Espaço`.

Use **Tamanho da prévia (px)** acima dos painéis direitos para escolher um tamanho quadrado de 160 a 960 pixels, arrastando o controle ou digitando um número. Todos os métodos usam o mesmo tamanho e se ajustam à largura disponível. O tamanho é lembrado para a próxima abertura. Alterá-lo mantém a ampliação atual e ajusta o quadrado de inspeção ao redor do centro, limitado às bordas da imagem.

Pressione **F11** ou o botão **Tela cheia (F11)** para usar a tela toda; o texto do botão muda para **Sair da tela cheia (F11)**. A barra de título também pode maximizar a janela. **Esc** sai primeiro da tela cheia e só fecha a janela em uma segunda pressão. Voltar da tela cheia restaura o tamanho anterior da janela, e a grade de resultados se reorganiza para a largura disponível nos dois casos.

Arraste a imagem de origem com o botão esquerdo para mover o quadrado branco de inspeção; ao soltar, ele permanece no lugar. Os painéis direitos mostram exatamente essa região da mesma pilha alinhada. Use a roda para mudar a ampliação da região, arraste com o botão direito para deslocar a origem, **Ctrl+roda** para ampliar a origem e **F** para ajustar. Trocar de quadro de origem ou redimensionar a janela preserva a seleção. Cancelar interrompe o trabalho pendente; fechar aguarda a operação nativa atual terminar com segurança.

### Prévia rápida (renderização de rascunho)

Ajustar parâmetros em pilhas grandes é lento em resolução total. Abra **Configurações de renderização** e marque **Prévia rápida**; o OpenFocus renderiza primeiro um rascunho com metade do tamanho linear (nunca abaixo de 320 px no lado maior), normalmente várias vezes mais rápido, incluindo registro. A mensagem de conclusão (e a dica do resultado mais recente) indica **Qualidade: prévia**, e o rascunho **não** é adicionado ao histórico de saídas. Desmarque a caixa e renderize novamente para obter o resultado em resolução total. O modo de prévia não se aplica a renderizações ROI e nunca contamina o cache de alinhamento em resolução total.

---

## 9. Transformações de imagem

### Rotação

1. Vá em **Editar → Girar**
2. Escolha o tipo de rotação:
   - **90° horário**: gira todas as imagens 90 graus para a direita
   - **90° anti-horário**: gira todas as imagens 90 graus para a esquerda
   - **180°**: vira as imagens de cabeça para baixo

### Inversão

1. Vá em **Editar → Inverter**
2. Escolha o tipo de inversão:
   - **Inversão horizontal**: espelha da esquerda para a direita
   - **Inversão vertical**: espelha de cima para baixo

### Redimensionamento

1. Vá em **Editar → Redimensionar**
2. Uma janela com opções de redimensionamento aparece
3. Informe as novas dimensões ou o percentual de escala
4. Todas as imagens da pilha são redimensionadas proporcionalmente

### Ordem das transformações

As transformações se aplicam a toda a pilha de imagens, mantendo o alinhamento entre os quadros.

---

## 10. Retoque, rótulos e anotações

### Retocar o resultado (Editar → Retocar resultado…, `Ctrl+T`)

A fusão ainda pode deixar um defeito local — um halo ao redor de uma borda de alto contraste, uma cerda vinda do quadro errado, ruído em uma área plana. O espaço de trabalho copia pixels dos quadros exatos que produziram o resultado selecionado:

1. Selecione uma saída na lista (clicar com o botão direito → *Retocar resultado…* também funciona) e pressione `Ctrl+T`. Sem seleção, um seletor lista todos os resultados e indica quais têm quadros de origem registrados.
2. O quadro de origem fica à esquerda e o resultado à direita. Ambos compartilham zoom e posição: a roda amplia, arrastar com o botão direito ou central (ou Espaço+arrastar) desloca, `F` ajusta e `1` mostra 100%.
3. Escolha uma miniatura, arraste o controle de quadros ou pressione `,` / `.`. Um traço fica bloqueado nesse quadro e nas configurações atuais do pincel; o quadro não pode mudar até o traço terminar.
4. Escolha **Pincel de origem** (`B`) ou **Pincel de restauração** (`E`). Diâmetro, dureza e opacidade têm controle e campo numérico; `[` / `]` ajusta o diâmetro. Com `Ctrl` e a roda para cima, o diâmetro aumenta; para baixo, diminui. A roda sem modificadores ajusta o zoom. O pincel de restauração devolve os pixels à imagem que você abriu, não a uma saída anterior.
5. Mantenha `Tab` para ver o resultado antes desta sessão e `Alt` para espiar o quadro de origem. A referência de nitidez é opcional, cobre apenas a tela de origem e usa uma escala compartilhada por todos os quadros.
6. `Ctrl+Z` / `Ctrl+Y` desfazem e refazem traços inteiros. **Restaurar tudo…** é uma única etapa desfazível. **Aplicar como nova saída** adiciona uma entrada chamada `<nome> · Retocado N` e a seleciona; o original permanece intacto. A nova entrada é marcada como *Não exportada* até você exportá-la pela janela principal, e o OpenFocus a protege da remoção automática do histórico e de fechar sem uma decisão.

Aplique o retoque como nova saída, selecione-a e abra **Salvar como**. Escolha TIFF em camadas na lista de formatos para salvar o resultado com seus quadros de origem. Rótulos e barras de escala ativos não são incluídos; mudanças de anotação não salvas permanecem protegidas. Geometria de origem e máscaras de validade são mantidas. O resultado e as origens devem ser imagens RGB ou em escala de cinza opacas com a mesma profundidade de 8 ou 16 bits, e cada retângulo de origem deve caber no resultado. A exportação suporta até 128 origens; nomes de origem são limitados a 255 caracteres e não podem conter NUL nem caracteres fora do Plano Multilíngue Básico, como muitos emojis. Ela recusa contextos inseguros e trabalhos estimados em mais de 512 MiB de memória de trabalho ou acima do orçamento de RAM disponível, relatando a falha em vez de escrever uma exportação incompleta. O GIMP 3.2.6 no Windows importa as camadas de 8 e 16 bits, incluindo seus deslocamentos e o preenchimento transparente de registro. O resultado fica visível e as camadas de origem ocultas; o GIMP também adiciona uma camada de fundo visível. Outros editores não foram testados.

Para uma correção tonal local, selecione **Pincel de exposição**, ajuste sua quantidade em centésimos de stop (100 dobra os valores dos pixels, -100 os divide pela metade) e pinte no resultado. A prévia é atualizada enquanto você pinta. Cada traço ajusta o resultado atual uma vez, mesmo onde as pinceladas se sobrepõem; outro traço se acumula. Use a opacidade e a dureza do pincel para controlar a mistura. Altas luzes podem estourar ao clarear. Mantenha **Antes** para comparar com o resultado como foi aberto, pressione Esc durante um traço para cancelá-lo, ou use desfazer/refazer depois do traço. São edições de pixels, não camadas ajustáveis. Aplique como nova saída e exporte a nova entrada para salvar os pixels editados; um projeto não armazena esses pixels.

**Ajustes do resultado…** altera todo o resultado de retoque atual. Sua prévia amostrada é atualizada ao mudar exposição e contraste; marque **Antes** para comparar com o resultado antes de abrir o ajuste. Ambos os controles usam centésimos de stop: a exposição multiplica os valores dos pixels e depois o contraste escala a distância deles em relação ao cinza médio. Valores extremos podem estourar altas luzes ou sombras. **OK** aplica o ajuste como uma única edição desfazível; **Cancelar** deixa o resultado inalterado. Durante o processamento, Cancelar solicita uma reversão. A imagem inteira deve caber no limite de memória do histórico de retoque para esta operação desfazível. Publique e exporte o resultado como descrito acima.

A proveniência viaja com a saída: os quadros alinhados, os deslocamentos de corte/colagem da ROI, o corte de bordas DCT e a profundidade de bits são registrados quando o resultado é renderizado. Arquivos de lote, resultados legados e entradas de pasta de somente alinhamento não têm origem registrada, então são listados como indisponíveis em vez de serem deduzidos da pilha atual. Traços não aplicados e histórico de desfazer são apenas da sessão; salvar um projeto não os armazena.

Para um arquivo de imagem existente, escolha **Editar → Retocar arquivos de imagem…**, depois selecione o resultado e uma ou mais imagens de origem. As origens já devem estar alinhadas ao resultado na mesma grade de pixels; dimensões iguais não provam alinhamento. São aceitas imagens PNG, TIFF, JPEG e BMP de página única com dimensões verticais, canais e profundidade de 8 ou 16 bits correspondentes; a orientação EXIF é aplicada ao carregar. Imagens com alfa e multipágina não são suportadas. Os quadros selecionados são fornecidos pelo usuário, não são proveniência registrada de renderização. Aplicar cria uma saída separada não exportada; salve-a na lista de saídas. Os arquivos escolhidos são protegidos contra sobrescrita durante esta sessão, incluindo exportações de mapas de profundidade e malhas 3D. Como em outros resultados, arquivos de projeto não armazenam esses pixels nem o histórico do pincel.

### Substituir o fundo (Fundo automático)

Ao empilhar, o fundo fica nítido juntamente com o motivo. O fundo de uma imagem
empilhada costuma ficar um mosaico de desfoques diferentes, com um halo ao redor
do motivo. **Fundo automático…** na área de retoque o substitui por um quadro
real da mesma pilha. O motivo permanece como foi empilhado. O fundo é derivado
de um quadro e do resultado empilhado; o processamento pode reconstruir, ajustar
e mesclar seus pixels.

1. Abra o resultado na área de retoque e clique em **Fundo automático…**.
2. Escolha o fundo:
   - **Automático** (recomendado) usa o fundo suave, salvo se a pilha registrou
     focada grande parte do fundo. Nesse caso usa o nítido.
   - **Fundo suave** usa o quadro com o fundo mais desfocado.
   - **Fundo nítido** usa o quadro focado mais longe.
3. Deixe **Limpar halos nas bordas** ligado para remover halos ao longo das
   bordas do motivo. Nunca vai além do que os quadros registraram. Desligue
   para manter o aspecto das bordas da pilha.
4. Marque **Manter as partes suaves ligadas ao motivo** quando partes do motivo
   ficam logo fora da profundidade de campo, como a ponta arredondada de um
   espinho. Também pode manter objetos de fundo junto ao motivo tal como foram
   empilhados, por isso verifique o resultado.
5. Clique em **Substituir fundo**. Uma pilha de 15 quadros de 20 megapixels
   demora cerca de um minuto e precisa de cerca de 3,5 GB de memória livre.
   **Cancelar** interrompe.

A alteração é um único passo que pode ser anulado, e os traços já pintados no
motivo são mantidos. Use o **Pincel de restauração** para recuperar os pixels
empilhados onde os preferir, por exemplo uma folha de erva desfocada à frente do
motivo. Quando o motivo ocupa quase todo o enquadramento, como numa cena do chão
da floresta, a imagem fica tal como foi empilhada e aparece um aviso.

Se abrir o retoque num resultado que já foi retocado, o fundo automático analisa
o resultado empilhado de onde ele veio e aplica o novo fundo sobre a sua imagem
retocada. Os retoques a pincel no motivo são mantidos e um fundo já substituído
nunca volta a ser analisado, por isso uma segunda execução não acumula halos.

Após uma execução, **Mostrar dados ausentes** colore os locais que nenhum
quadro mostrou com nitidez. São os locais que devem ser verificados ou retocados à mão:

- magenta: fundo que nenhum quadro registrou sem o motivo à frente
- ciano: motivo escondido da câmera
- amarelo: motivo fora do intervalo de foco da pilha
- ardósia: áreas lisas demais para avaliar

**Salvar mapa de dados ausentes…** grava a prévia colorida e uma
máscara em preto e branco por cor. O fundo automático precisa dos quadros que
produziram o resultado e não está disponível para resultados de região (ROI).
Os novos pontos de controle de fundo restauram exatamente a imagem, o histórico
acessível e os diagnósticos sem recalcular o fundo.

### Gerenciamento de rótulos

Abra **Ferramentas → Rótulos** após carregar imagens ou produzir um resultado. Cada rótulo tem seu próprio destino: a pilha de origem, a pilha alinhada ou os resultados de saída. Rótulos de saída podem se aplicar a todos os resultados ou ao resultado selecionado pelo nome; renomear ou reordenar esse resultado não altera a associação.

Use **Adicionar** para criar outro rótulo, selecione uma entrada para editá-la e use sua caixa para ativar ou desativar. Insira texto fixo ou `{value}` para um número de quadro. O número é o valor inicial mais o índice do quadro (base zero) multiplicado pelo intervalo; uma saída única sempre usa o valor inicial.

Escolha todos os quadros, o atual ou um intervalo personalizado como `3-5,8`. Números de quadro inválidos ou indisponíveis impedem a aplicação do rascunho. Intervalos atual e personalizado seguem as identidades de origem selecionadas quando a pilha é reordenada; rótulos de todos os quadros também se aplicam a quadros adicionados. Destinos ausentes permanecem associados às suas identidades originais até você escolher explicitamente um novo destino.

A prévia usa o destino real. Navegue com Anterior/Próximo ou digite o número do quadro, escolha **Ajustar** ou **100%**, e arraste a imagem para deslocá-la em 100%. Posição e tamanho da fonte usam pixels da imagem. Uma fonte substituta ou um desenho malsucedido é relatado. Rótulos preservam a profundidade de bits original quando exportados como PNG ou TIFF.

**Aplicar e fechar** confirma todo o rascunho como uma única mudança desfazível. Cancelar, Esc e fechar a janela o descartam. Selecionar outro rótulo mantém as edições do rascunho. **Excluir** remove apenas a entrada selecionada, incluindo a última. As ações de remoção do menu limpam rótulos de origem, da pilha alinhada, de saída ou todos de forma independente, e também podem ser desfeitas. Aplicar rótulos muda a exibição da sessão; exporte uma imagem para gravar as anotações em um arquivo.

---

## 11. Configurações

Abra **Configurações → Configurações avançadas** para preferências de desempenho, registro, limpeza do sensor, nomes de saída, IA StackMFF e inicialização. As mudanças permanecem rascunhos até **OK**; Cancelar, Esc e fechar as descartam. Reduzir o tamanho do bloco atualiza imediatamente o limite visível de sobreposição para metade de um bloco. O limite de lote salvo da seção de IA é mantido, mas a inferência atualmente envia um bloco por chamada; a largura de registro pode mudar a precisão do alinhamento. A coleta de diagnósticos de alinhamento é uma caixa da seção Registro.

Downloads automáticos de atualização rodam quando o aplicativo está ocioso e nunca instalam sozinhos. Um download concluído permanece pronto até você escolher explicitamente **Reiniciar e atualizar**. A reinicialização protege edições ativas e trabalho não salvo, depois grava os dados de recuperação antes de iniciar a substituição. Cancelar um download ou preparação aguarda o limite seguro atual. Uma compilação preparada completa e mais recente pode ser instalada offline.

A correção de campo plano altera os pixels de trabalho da sessão sem sobrescrever os arquivos de origem. Selecione a referência branca e um quadro de prévia, depois gere a prévia novamente após mudar origem ou escala. As imagens de referência e de trabalho/base devem compartilhar domínio de coordenadas e tamanho. Aplicar prepara as duas pilhas antes de confirmar; a remoção opcional do quadro de referência faz parte da mesma etapa de desfazer. Exporte as imagens de origem corrigidas quando precisar preservar esses pixels fora da sessão.

O processamento em lote aceita uma imagem de campo branco opcional em suas opções. Escolha uma referência desfocada na resolução original dos quadros; ela é dimensionada com a pilha de entrada e aplicada antes do registro, usando o tamanho de poeira padrão. Uma referência ilegível ou de tamanho incompatível faz a pilha afetada falhar visivelmente. A referência é excluída das entradas de pasta, incluindo processamento agrupado; pelo menos dois quadros de origem devem permanecer para a fusão. A correção de campo plano não pode ser combinada com entrada transparente. Deixe o campo vazio para processar sem correção. A CLI também aceita a referência com `--dust-map WHITE`.

A limpeza do sensor é opcional e roda nos quadros originais em coordenadas do sensor antes do registro. Selecione uma máscara de defeitos explícita para corrigir pixels quentes ou mortos conhecidos; o OpenFocus não infere uma máscara a partir das imagens do espécime. A máscara deve ser uma imagem de canal único de 8 ou 16 bits com as mesmas dimensões da origem: zero significa não afetado, e 1 ou o valor máximo do canal marca defeitos. Revise sua extensão antes de processar. Máscaras que cobrem mais de 10% do quadro, ou defeitos sem um vizinho não mascarado, são rejeitadas. Mantenha a máscara na resolução original e não a selecione como imagem de origem. Para processamento na interface, carregue os quadros de origem em escala total ao usar uma máscara de defeitos; a máscara nunca é redimensionada. A limpeza do sensor (incluindo redução de ruído) não pode ser aplicada a uma pilha da interface após rotação, inversão, correção de campo plano ou outra mudança de origem: recarregue os quadros originais ou desative a limpeza. Isso também vale para projetos restaurados com esses históricos de origem. O processamento ROI pré-alinhado não pode ser combinado com limpeza do sensor; saia do modo ROI e renderize a pilha completa. As configurações de limpeza pertencem à sessão e ao projeto salvo, não às preferências globais. Lote e CLI processam suas entradas originais separadamente, com limpeza do sensor antes da correção de campo plano e do registro. Máscaras de defeitos exigem arquivos de origem com orientação EXIF neutra. Arquivos orientados, RAW de câmera e origens cuja orientação não pode ser estabelecida são rejeitados porque seus pixels decodificados podem não corresponder mais às coordenadas do sensor. Use imagens de origem e uma máscara na mesma orientação de pixels conhecida, com metadados de orientação neutros. A redução de ruído sozinha não exige máscara de defeitos.

A redução de ruído opcional do sensor é independente da máscara de defeitos e fica desativada por padrão. Inspecione o resultado em tamanho real para decidir se sua suavização é adequada ao espécime. A limpeza do sensor preserva os arquivos de origem e a profundidade de imagem de 8 ou 16 bits; é separada da correção de campo plano. A CLI aceita `--defect-mask-path MASK` e `--sensor-denoise`; receitas e a API de receitas Python capturam essas opções. Receitas registram uma impressão digital da máscara e rejeitam uma máscara ausente ou alterada. A limpeza da CLI não pode ser combinada com entrada transparente. O processamento agrupado aplica a limpeza aos quadros originais, não aos resultados intermediários do grupo.

Receitas exigem que sua localização de saída esteja fora de todas as pastas de entrada e rejeitam uma entrada selecionada explicitamente como saída. Escolha um local de saída separado para proteger as imagens de origem e evitar incluir um resultado anterior em uma nova execução. As configurações de processamento de um projeto podem ser reutilizadas como modelo ou receita mesmo que suas origens originais tenham sido movidas: **Arquivo → Salvar modelo de processamento…** armazena as configurações sem caminhos de origem e **Arquivo → Aplicar modelo de processamento…** as aplica a uma pilha carregada, enquanto `openfocus recipe create --project` grava uma receita headless que ainda exige caminhos `--input` explícitos. Modelos carregam o indicador de redução de ruído do sensor, mas não o caminho da máscara de defeitos; receitas armazenam o caminho da máscara e sua impressão digital. Abrir o projeto em si ainda exige suas origens e qualquer referência de campo plano registrada.

Quando uma referência de arquivo é aplicada em Ferramentas na escala base da pilha, um projeto salvo registra seu caminho absoluto, hash de conteúdo e tamanho de poeira, e a reaplica uma vez às origens originais ao abrir. Mantenha o arquivo de referência inalterado e no lugar: uma referência ausente ou alterada impede a abertura do projeto. Uma referência selecionada da pilha, ou correção em outra escala de trabalho/base, ainda exige exportar a pilha de origem corrigida antes de salvar um projeto. Se a escala de trabalho mudar após a correção, restaure-a para a escala de correção antes de salvar ou exporte a pilha corrigida.

**Ajuda → Enviar comentários** abre os logs locais. Copiar copia o log visível da sessão; o ZIP de comentários contém os logs de sessão e do aplicativo completo, além de informações do ambiente. Revise o pacote antes de enviá-lo você mesmo. Abrir o GitHub apenas abre o rastreador. Cancelar uma exportação deixa os arquivos inalterados; um erro mantém o arquivo de destino anterior e mostra o caminho que falhou.

Cada inicialização grava seu próprio log rotativo, então trabalhos GUI/CLI concorrentes mantêm registros separados. IDs de tarefa ligam parâmetros, tempo decorrido e resultados concluídos/cancelados/falhos; erros inesperados incluem um traceback completo. O ZIP de comentários também inclui o relatório de falha atual e as versões das dependências. Use **Ajuda → Abrir pasta de logs** ou o caminho mostrado na página de comentários para encontrar os arquivos. Os logs rotacionam em 1 MB com três backups por processo; logs de sessões inativas com mais de 14 dias são removidos. Para logs de suporte detalhados, defina `OPENFOCUS_LOG_LEVEL=DEBUG` antes de iniciar; `OPENFOCUS_LOG_DIR` opcionalmente seleciona outra pasta de logs.

### Configurações de blocos (otimização de memória)

Para imagens grandes, o OpenFocus usa processamento em blocos para evitar problemas de memória.

Acesso: **Configurações → Configurações avançadas → Desempenho**:

- **Tamanho do bloco**: tamanho de cada bloco de processamento (1024 px por padrão)
- **Sobreposição de blocos**: sobreposição entre blocos para uma mistura sem emendas (256 px por padrão)
- **Limite de blocos**: tamanho de imagem acima do qual os blocos são usados (2048 px por padrão)

**Recomendações**:

- Blocos menores: menos memória, processamento mais lento
- Blocos maiores: mais memória, processamento mais rápido
- Aumente a sobreposição se notar emendas visíveis nos resultados

### Configurações de registro

Acesso: **Configurações → Configurações avançadas → Registro**:

- **Largura de redução**: resolução de pré-processamento para registro
  - Padrão: 1024 px
  - Menor para velocidade, maior para precisão
- **Coletar diagnósticos de alinhamento**: recalcula o alinhamento na próxima renderização e habilita **Configurações → Configurações avançadas → Registro → Diagnósticos de alinhamento**

### Referência de desempenho

Veja [PERFORMANCE.md](./PERFORMANCE.md) para taxa de transferência e memória medidas em pilhas reais grandes, além de orientações práticas para pilhas profundas e entradas de alta resolução.

O painel de status mostra memória física usada e total quando o sistema operacional fornece as duas leituras; caso contrário, mostra `N/D`.

### Configurações do número de threads

Controle o processamento paralelo da CPU:

Acesso: **Configurações → Configurações avançadas → Desempenho**:

- `0` seleciona o agendamento automático e é o padrão para novas configurações.
- Valores positivos são limites superiores; limites salvos sobrevivem a atualizações.
- Capacidade de CPU, modo de recursos e margem de memória podem reduzir a concorrência. Mais threads não tornam necessariamente uma renderização mais rápida.

### Modo de recursos

**Configurações → Configurações avançadas → Desempenho → Modo de recursos** decide quanto da máquina uma renderização pode usar. Ele nunca altera a imagem fundida.

| Modo | Comportamento |
|---|---|
| Equilibrado (padrão) | Mantém livres cerca de 20% dos núcleos físicos e uma reserva de memória para o desktop |
| Responsivo | Usa cerca de metade dos núcleos e uma janela de trabalho menor; renderizações mais lentas, máquina mais responsiva |
| Máximo | Usa todos os núcleos físicos permitidos pelo orçamento de memória |

O planejador considera núcleos físicos, afinidade e cotas de CPU, e executa uma única camada paralela por vez. Trabalhos pesados compartilham a admissão durante preparação e publicação; cancelar aguarda a chamada nativa ativa. Memória desconhecida ou insuficiente produz um erro de recursos. A grade de blocos selecionada permanece inalterada; escolher blocos menores pode mudar os pixels de saída.

### Aceleração por GPU

StackMFF V1/V2/V3 usam ONNX Runtime. DirectML no Windows e CoreML no macOS podem acelerar a inferência; caso contrário, usa-se a CPU. A disponibilidade do provedor não comprova execução na GPU. O suporte a outras GPUs requer validação.

- Alterne em **Configurações → Aceleração por GPU**
- Desmarcar força os métodos neurais na CPU — útil quando drivers de GPU dão problema
- O painel de status mostra `N/D` quando nenhum acelerador é usado ou está disponível, e `CPU*` quando existe um acelerador mas o método selecionado é somente CPU
- A escolha se aplica à próxima renderização e é lembrada

### Tamanho do lote de IA

Acesso: **Configurações → Configurações avançadas → IA**:

- O limite de lote salvo é 2 por padrão; a política de execução atual limita a inferência real a um bloco por chamada em todos os dispositivos.
- Aumentar esse limite não ativa lotes de GPU não validados nem reduz o tamanho espacial dos blocos.

### Temas

Dois temas de interface estão disponíveis: **Claro (padrão)** e **Escuro**. Troque em **Configurações → Tema**; a escolha é aplicada imediatamente (a barra de título do Windows acompanha) e lembrada. Na primeira execução, o usuário escolhe um tema na janela de boas-vindas; uma escolha salva sempre prevalece.

### Modo portátil

Crie um arquivo vazio chamado `OpenFocus.portable` ao lado de `OpenFocus.exe` (o ZIP portátil já inclui um) e todos os dados do usuário — configurações, logs, arquivos recentes — são armazenados em uma pasta `OpenFocusData` ao lado do aplicativo em vez do perfil do usuário, para que a instalação possa viajar em um pendrive. Exclua o marcador para voltar ao perfil normal.

### Metadados EXIF

Os metadados de origem (modelo da câmera, exposição, orientação e assim por diante) são mantidos: a orientação é aplicada ao carregar, e exportações PNG/TIFF/WebP incorporam o EXIF original automaticamente. Exportações JPG/BMP não armazenam o EXIF completo e são gravadas sem ele.

### Entrada HEIC/HEIF

Fotos de iPhone carregam diretamente. Requer `pillow-heif`, que `pip install openfocus` instala e as compilações empacotadas incluem.

### Atualizações

**Ajuda → Verificar atualizações** compara sua versão com as informações de `openfocus.top`. Uma versão mais recente é baixada do mesmo site em segundo plano enquanto o aplicativo está ocioso; quando a compilação está preparada, o banner e a área de status oferecem **Reiniciar e atualizar**, que a aplica offline. Compilações que não podem se atualizar sozinhas abrem a página de downloads do site. A verificação automática roda silenciosamente no máximo uma vez por dia.

### Cache de registro

O OpenFocus pode armazenar resultados de registro em uma pasta `.openfocus_cache` ao lado da pilha de origem. Reabrir a mesma pilha (arquivos inalterados) então pula o alinhamento por completo. Ative ou desative em **Configurações → Configurações avançadas → Registro → Cache de registro**; entradas obsoletas são substituídas automaticamente.

### Configurações persistidas

Tudo o que segue sobrevive automaticamente ao reinício: número de threads, modo de recursos, parâmetros de blocos, largura de redução do registro, tamanho de lote da fusão por IA (StackMFF V3), aceleração por GPU, cache de registro e opções de diagnósticos de alinhamento, idioma da interface, tamanho da janela e layout dos painéis, formato de exportação por arrastar, opções de brilho por método, valores de suavização e raio, e pilhas abertas recentemente. As janelas de configurações salvam no momento em que você confirma; tudo também é salvo ao fechar. As preferências são armazenadas por usuário em `OpenFocus/OpenFocus.ini` (formato INI do QSettings).

### Idioma da interface

Troque a qualquer momento em **Configurações → Idioma**, que lista os doze pacotes (English, 中文, 日本語, Español, Deutsch, Français, Русский, 한국어, Português, Italiano, Nederlands, Polski). A mudança se aplica imediatamente e é lembrada para sessões futuras.

### Verificação automática de atualizações

Uma vez por dia, alguns segundos após a inicialização, o OpenFocus verifica `openfocus.top` silenciosamente. As notas da versão também vêm desse site. Se houver uma versão mais recente, um banner aparece acima dos painéis de imagem com a versão, **Atualizar agora** (aplica uma compilação preparada ou inicia o download), **Novidades** (notas da versão) e um botão para dispensar. Dispensá-lo oculta o banner apenas para aquela versão; a próxima versão o traz de volta, e **Ajuda → Verificar atualizações** o reativa imediatamente.

Enquanto o download roda em segundo plano, a barra de status mostra o progresso. Você também pode verificar manualmente a qualquer momento em **Ajuda → Verificar atualizações**.

---

## 12. Processamento em lote

Processe várias pastas automaticamente:

1. Vá em **Ferramentas → Processamento em lote**
2. Na janela de lote:
   - **Adicionar pastas**: selecione pastas contendo sequências de imagens
   - **Arrastar pastas**: solte uma ou mais pastas de pilhas em qualquer lugar da janela (ou da lista) para enfileirá-las de uma vez; soltar muda para o modo multipasta e as miniaturas aparecem conforme são decodificadas
   - **Remover pastas**: remove as pastas selecionadas da lista
3. Configure a saída:
   - **Salvar em subpasta**: cria os resultados em cada pasta de entrada
   - **Salvar na mesma pasta**: salva ao lado da entrada preservando os arquivos existentes
   - **Pasta personalizada**: especifique o local de saída
4. Revise as configurações atuais de fusão e registro
5. Clique em **Iniciar processamento em lote**

A janela de lote mostra o progresso em tempo real. Você pode cancelar o processamento a qualquer momento.

### Observações sobre o processamento em lote

- Parte de uma captura das configurações da janela principal; os controles do lote as substituem.
- Cada pasta ou grupo dividido é processado independentemente pelo mesmo pipeline.
- **Processar pilhas grandes em grupos** é uma caixa opcional no modo multipasta. Quando marcada, **Tamanho do grupo** (mínimo 2, 4 por padrão) e **Sobreposição** (de 0 ao tamanho do grupo menos 2) aparecem; quando desmarcada, esses controles ficam ocultos e o agrupamento é desativado. A sobreposição do desktop começa em 1 e é limitada ao intervalo permitido quando o tamanho muda. Os grupos são fundidos em rodadas usando espaço temporário em disco; o progresso identifica cada rodada. Fusão, igualação de exposição, correção base e registro usam os quadros e referências de cada grupo, incluindo rodadas intermediárias, então o resultado se aproxima do processamento da pilha completa de uma vez. O agrupamento exige um método de fusão; profundidade por quadro original, proveniência e exportações de quadros alinhados ficam indisponíveis. O modo de pasta dividida não é suportado, então a caixa é rejeitada nele. Grupos registrados usam seu primeiro quadro como referência de coordenadas. Cada registro remove uma borda conservadora para interpolação, e rodadas posteriores cruzam os retângulos sobreviventes, incluindo qualquer cauda de um único quadro. A imagem final é, portanto, cortada. Cada grupo registrado deve ter mais de 8 pixels nas duas dimensões; rotações difíceis ou transformações projetivas podem não ter um retângulo seguro e produzir um erro. Use entradas pré-alinhadas sem registro ou desative o agrupamento se isso ocorrer.
- A entrada agrupada deve ter dimensões orientadas correspondentes e consistir em arquivos de quadro único com cabeçalhos legíveis pelo Pillow. RAW de câmera, arquivos multipágina e arquivos com canais alfa ou transparência não são suportados, incluindo pilhas PNG transparentes. Remova o alfa ou desative o agrupamento para processar essas pilhas. Execuções de teste da CLI verificam essas restrições. Na linha de comando, um exemplo é `python main.py --input stack --output fused.png --group-size 8 --group-overlap 1`. A entrada agrupada da CLI deve ser uma pasta e não pode ser combinada com `--depth-map`. Uma referência de alinhamento OpenFocus diferente para toda a pilha não pode ser combinada com agrupamento. Uma referência de campo plano é aplicada aos quadros originais antes do registro; a nitidez de saída é aplicada uma vez após o grupo final.
- **O que é gravado e onde, por modo.** Com **somente alinhamento**, o destino selecionado recebe uma pasta `<pilha>_aligned` por pilha e o painel de saída lista essa pasta. Com **fusão + salvar alinhados**, a imagem fundida é gravada como `<pilha>.<ext>` no destino e seus quadros alinhados vão para `<pilha>_aligned` ao lado; o painel de saída lista apenas o resultado fundido. Com **fusão sem exportar alinhados**, apenas a imagem fundida é gravada. O painel de saída nomeia as pastas exatamente como estão no disco.
- Quadros alinhados mantêm os nomes de arquivo originais. Nomes duplicados e pastas existentes recebem sufixos numéricos. Arquivos de origem nunca são substituídos.
- Cópias `*_registered*` remanescentes de versões anteriores não são carregadas como quadros quando o arquivo original está presente na mesma pasta; a mensagem de carregamento informa quantas foram ignoradas.
- PNG/TIFF preservam dados de 16 bits; escolher explicitamente JPG/BMP converte para 8 bits.
- O cancelamento aguarda o worker parar. Arquivos confirmados permanecem disponíveis; tarefas parciais, falhas, canceladas e não iniciadas são contadas separadamente.

---

## 13. Exportação de resultados

### Salvar um resultado individual

1. Selecione o resultado que deseja salvar na lista de saídas
2. Vá em **Arquivo → Salvar** ou pressione `Ctrl+S`
3. Escolha formato e local
4. Clique em Salvar

Em **Configurações → Configurações avançadas → Nomes de saída**, um prefixo de nome de arquivo opcional ajuda a identificar resultados recém-renderizados. O exemplo é atualizado enquanto você digita; use até 40 letras, números, sublinhados ou hífens. Deixe vazio para a nomenclatura original. Confirme com **OK** para lembrar, ou cancele para descartar a mudança. O prefixo passa a fazer parte dos novos nomes da lista de saídas e de seus nomes de exportação sugeridos; resultados existentes mantêm seus nomes. Você ainda pode editar um nome na janela de salvamento. Essa preferência não renomeia arquivos de origem nem muda nomes de saída de lotes.

### Formato de arrastar para exportação

Arrastar resultados da lista de saídas grava arquivos de imagem no formato escolhido no menu de contexto da lista (**Formato ao arrastar**: JPG, PNG, WebP ou TIFF). A escolha é lembrada.

### Salvar a pilha registrada

1. Vá em **Arquivo → Salvar pilha → Pilha registrada**
2. Escolha o formato de salvamento:
   - **Salvar como pasta**: escolha um destino e uma codificação; cada pilha recebe uma nova subpasta contendo seus nomes de quadro originais
   - **Salvar como GIF**: cria um GIF animado da sequência alinhada

### Salvar a pilha de entrada

1. Vá em **Arquivo → Salvar pilha → Pilha de entrada**
2. Escolha o formato de salvamento:
   - **Salvar como pasta**: salva os quadros originais como imagens separadas
   - **Salvar como GIF**: cria um GIF animado da sequência de entrada

### Formatos de exportação

- **JPG**: comprimido, boa relação qualidade/tamanho
- **PNG**: sem perda, preserva dados de imagem de 8 e 16 bits
- **BMP**: sem compressão, converte dados de 16 bits para 8 bits
- **TIFF**: preserva dados de 8 e 16 bits; a exportação multipágina mantém a pilha inteira em um arquivo
- **GIF**: animação com paleta de 8 bits; o cancelamento deixa intacto um destino existente

---

## 14. Interface de linha de comando

`pip install openfocus` inclui o modelo de IA, seus pesos e a decodificação HEIC e RAW de câmera. O `dtcwt` é ignorado no Python 3.13 ou superior, onde a entrada DTCWT permanece desabilitada com uma dica.

O OpenFocus pode rodar sem a interface gráfica — útil para pipelines com script e integração contínua. O script de console `openfocus` (instalado por `pip install openfocus`) e `python main.py` de uma cópia do código expõem os mesmos comandos. As opções de fusão funcionam com ambas as formas; os subcomandos são:

```bash
openfocus version                       # imprime a versão
openfocus describe                      # JSON: métodos, parâmetros, capacidades, comandos de controle
openfocus gui [--control] [--socket NAME] [paths…]   # inicia o aplicativo desktop
openfocus ctl <command> [flags]          # controla uma janela em execução (ver AGENT_CONTROL_SPEC.md)
openfocus recipe create FILE --input … --output … [processing flags]
openfocus recipe run FILE [--dry-run]    # reproduz uma receita headless salva
openfocus retouch replay CHECKPOINT --base BASE --sources … --output RESULT
```

`openfocus gui` inicia em segundo plano; `--control` aguarda o canal de controle e imprime seu JSON de conexão. `openfocus describe` e `version` não exigem entrada. O fluxo de trabalho para agentes está documentado em [AGENT_CONTROL_SPEC.md](./AGENT_CONTROL_SPEC.md) e `skills/openfocus-control/SKILL.md`.

De um ambiente Python (ou ao lado do executável empacotado):

```bash
# Funde uma pasta com o método de IA padrão
python main.py --input ./stack_folder --output ./result/fused.png

# DTCWT com registro ECC, 8 threads
python main.py -i ./stack_folder -o ./result/fused.png -m dtcwt -a ecc -t 8

# StackMFF V3 forçado na CPU com um tamanho de bloco personalizado
python main.py -i ./stack_folder -o ./result/fused.png -m stackmffv3 --cpu --tile-size 512

# Lista explícita de arquivos; arquivos de vídeo também são aceitos como entrada
python main.py -i img1.jpg img2.jpg img3.jpg -o fused.png -m gfgfgf

# Uma única pilha Z TIFF multipágina com barra de escala calibrada por metadados
python main.py -i ./zstack.tif -o fused.png -m dtcwt --scale-bar

# Inspeciona quadros adjacentes semelhantes sem fundir nem alterar a entrada
python main.py -i ./stack_folder --suggest-duplicates
```

`--suggest-duplicates` aceita uma pasta ou uma lista ordenada de arquivos de imagem de quadro único. Ele imprime um objeto JSON com os arrays `files` e `suggestions`; `index` e `reference_index` de cada sugestão são posições baseadas em zero em `files`, e `difference` é a diferença média normalizada das amostras. Adicione `--json report.json` para salvar o mesmo objeto. Este modo de revisão não se combina com `--output`, `--output-dir`, `--json-progress`, `--dry-run` nem com as opções de limpeza do sensor. O relatório compara arquivos adjacentes na ordem de entrada fornecida, sem saber quais quadros você pode excluir depois. Se excluir uma referência, reconsidere seu sucessor ou revise os arquivos restantes na ordem pretendida.

Por exemplo, adicione `--align openfocus --alignment-report alignment.json` a uma renderização única. O objeto JSON contém `method`, `input`, `frame_count` e `entries`. O `frame_index` de cada entrada é baseado em zero na ordem de entrada decodificada, antes da ordenação para fusão; imagens multipágina se expandem na ordem das páginas e quadros de vídeo na ordem de decodificação. Métodos compostos incluem entradas separadas para cada etapa. O relatório é gravado atomicamente após processamento bem-sucedido e não pode substituir uma entrada ou outra saída. Modos em lote, agrupado e de teste recusam esta opção.

| Opção | Descrição |
|--------|-------------|
| `--suggest-duplicates` | Relata quadros adjacentes quase idênticos para revisão manual; não os funde nem os exclui |
| `--input, -i` | Pasta(s) de imagens, arquivo de vídeo, um ou mais arquivos de imagem ou TIFF multipágina |
| `--output, -o` | Caminho da imagem de saída (a extensão define o formato) |
| `--output-dir, -d` | Modo lote: funde cada pasta de entrada gravando `<pasta>.png` por pilha |
| `--resume` | Modo lote: ignora pastas cuja saída já existe |
| `--config FILE` | Modo lote: substituições JSON por pasta, p. ex. `{"stackA": {"method": "dct"}}` |
| `--method, -m` | `stackmffv3` (padrão), `stackmffv2`, `stackmff`, `openfocus_a`, `openfocus_b`, `openfocus_c`, `guided_filter`, `dct`, `dtcwt`, `gfgfgf` |
| `--param KEY=JSON_VALUE` | Define qualquer parâmetro de fusão; repetível e prevalece sobre opções legadas |
| `--align, -a` | `none` (padrão), `homography`, `ecc`, `both`, `openfocus` |
| `--align-param KEY=JSON_VALUE` | Define qualquer parâmetro de registro; repetível |
| `--align-shift-limit` | Somente alinhamento `openfocus`: deslocamento máximo aceito como fração do tamanho do quadro (0,08 padrão; acima disso o quadro fica sem alinhamento) |
| `--alignment-report PATH` | Salva os resultados de alinhamento por quadro como JSON para uma renderização única não agrupada com alinhamento ativo; lote e grupos não são suportados |
| `--exposure-match` | Iguala a média de cada quadro à do último antes da fusão (ativado por padrão; `--no-exposure-match` desativa). Ganhos para toda a pilha, então a saída em blocos permanece sem emendas; aplica-se a todos os métodos e à referência de brilho do alinhador OpenFocus |
| `--base-frame` | Etapa opcional pós-fusão: iguala as baixas frequências da saída ao quadro `last`/`first` (`none` padrão) |
| `--sort` | Ordem de quadros para `openfocus_a/b/c`: `asc` (padrão, ordem de arquivo), `desc` (inverte uma varredura de longe para perto), `off` |
| `--frame-order` | Pasta de entrada: `auto` (padrão) ordena os quadros de perto para longe pelos passos de foco EXIF quando todos os têm (o bracketing de foco da OM SYSTEM começa no meio da profundidade); senão, por nome; `name` sempre pelo nome do arquivo. Arquivos listados explicitamente mantêm sua ordem |
| `--sharpen-output` | Aplica uma máscara de nitidez suave à saída fundida (desativada por padrão) |
| `--dust-map WHITE` | Correção de campo plano (mapa de poeira) a partir de uma tomada de campo branco desfocada |
| `--defect-mask-path MASK` | Máscara de defeitos em coordenadas do sensor reparada antes do registro |
| `--sensor-denoise` | Redução de ruído conservadora do sensor antes do registro |
| `--kernel, -k` | Tamanho do kernel (forçado ímpar); controla a escala de resposta de foco em `openfocus_a` e `openfocus_b`; C não tem raio |
| `--depth-smoothing VALUE` | Apenas `openfocus_b`: força da suavização de transição de profundidade (1,2 padrão) |
| `--blend-smoothing VALUE` | Apenas `openfocus_a` / `openfocus_c`: força da suavização de transição (2,0 para A, 3,0 para C por padrão) |
| `--ai-mode {soft,hard}` | Saída do StackMFF V3: mistura ponderada por probabilidade (padrão) ou quadro mais provável por pixel |
| `--threads, -t` | Limite superior de threads de trabalho; `0` = automático (padrão), valores positivos limitam a concorrência |
| `--resource-preference` | `balanced` (padrão), `responsive` ou `maximum`; controla o agendamento de recursos |
| `--batch-size` | Limite superior do lote de blocos de IA (2 padrão); a inferência atual usa um bloco por chamada |
| `--cpu` | Força a CPU mesmo com GPU disponível |
| `--downscale` | Largura de redução para detecção de características do registro |
| `--tile-enabled` / `--no-tile-enabled` | Ativa a fusão em blocos quando necessário (padrão: ativada) |
| `--tile-size` | Tamanho do bloco para imagens grandes (padrão: automático) |
| `--tile-overlap` | Sobreposição de blocos em pixels (padrão: automática) |
| `--tile-threshold` | Tamanho mínimo da imagem que ativa os blocos (padrão: automático) |
| `--group-size` | Funde uma pasta em grupos deste número de quadros; `0` desativa o agrupamento (padrão). Somente pastas; sem `--depth-map`, entrada RAW/alfa/multipágina nem quadro de referência OpenFocus diferente do padrão |
| `--group-overlap` | Quadros compartilhados por grupos consecutivos; `0` padrão, no máximo o tamanho do grupo menos 2 |
| `--scale-bar` | Desenha uma barra de escala (calibração detectada automaticamente nos metadados TIFF) |
| `--scale-um` | Tamanho do pixel em µm/px, substituindo metadados (implica `--scale-bar`) |
| `--depth-map [path]` | Também salva um mapa colorido de posição de foco (padrão `<output>_depth.png`); apenas métodos com capacidade de profundidade, não agrupados nem transparentes. No modo lote, apenas o caminho padrão é permitido |
| `--depth-colormap` | Mapa de cores para `--depth-map`: turbo, viridis, jet, gray, … (turbo padrão) |
| `--auto-background [preset]` | Substitui o fundo empilhado pelo de um quadro alinhado, como o Fundo automático no retoque: `auto` (padrão), `soft` (fundo mais desfocado) ou `sharp` (quadro focado mais longe). Precisa dos quadros alinhados, portanto não funciona com fusão em grupos |
| `--background-mode` | Fundo automático: `clean` (padrão) também remove os halos de borda do empilhamento, dentro do que os quadros registraram; `faithful` os mantém |
| `--keep-soft-parts` | Fundo automático: mantém as partes suaves presas ao objeto em que algum quadro focou |
| `--save-stacked [path]` | Com `--auto-background`, também salva o resultado antes da substituição (padrão `<output>_stacked<ext>`). No modo em lote só o caminho padrão é permitido |
| `--missing-data-map [path]` | Com `--auto-background`, também salva o mapa de dados ausentes e uma máscara de 8 bits por classe (padrão `<output>_missing.png`). No modo em lote só o caminho padrão é permitido |
| `--json FILE` | Grava um resumo de resultado legível por máquina em FILE ao terminar |
| `--json-progress` | Imprime um objeto JSON de progresso por etapa em stdout (NDJSON); o texto vai para stderr |
| `--dry-run` | Valida as entradas e imprime o plano sem fundir |

Não existe um valor `--method none`: a CLI sempre funde. Use o aplicativo desktop para saída de somente alinhamento.

### Receitas de processamento

Uma receita é um arquivo JSON portátil de opções de processamento da CLI com caminhos relativos à receita. Crie uma a partir das opções que você passaria a uma execução normal e depois reproduza-a:

```bash
openfocus recipe create line.json --input ./stack --output ./out/fused.tif \
  --method dtcwt --align ecc --threads 8
openfocus recipe run line.json
openfocus recipe run line.json --dry-run
```

`recipe create --project project.ofproj --input ./stack --output ./out/fused.tif` captura as configurações de processamento de um projeto com entradas escolhidas explicitamente; as origens do projeto podem ter sido movidas. Receitas rejeitam controles transitórios como `--dry-run`, `--resume`, `--json` e `--config`, e exigem que a saída fique fora de todas as pastas de entrada (e seja diferente de uma entrada selecionada explicitamente). Uma máscara de defeitos é armazenada por caminho relativo com uma impressão digital SHA-256, então uma máscara ausente ou alterada faz a execução falhar. Edições de origem do projeto (rotação, inversão, campo plano) não podem ser capturadas; exporte primeiro a pilha corrigida.

A captura do projeto transfere para a receita da CLI as configurações de fusão, registro, brilho, ordem, nitidez e blocos, e incorpora as configurações de processamento para a interface. Exclusões de quadros, escala de carregamento, rótulos e edições não são transferidos; prepare a nova pilha de entrada de acordo. As opções de agrupamento podem ser capturadas a partir de argumentos da CLI.

O formato JSON versão 1 contém `version`, um array `input`, `output` ou `output_dir`, e um objeto `options`. Os caminhos são relativos ao arquivo da receita; mova a receita junto com seus arquivos de entrada para preservar essas relações. A criação captura as configurações da CLI e os parâmetros resolvidos do algoritmo. Campos desconhecidos, valores inválidos e versões não suportadas são rejeitados. Uma execução de teste verifica a entrada sem gravar arquivos de resultado. Os arquivos de projeto existentes permanecem inalterados.

Na interface, Arquivo → Salvar modelo de processamento salva as configurações sem caminhos de origem. Arquivo → Aplicar modelo de processamento aplica um modelo, projeto ou receita baseada em projeto à pilha atual sem substituir suas entradas, transformações, exclusões ou rótulos. Um modelo não é, por si só, uma receita executável. As configurações de interface incorporadas em uma receita são independentes de suas opções de CLI; se editar o JSON, mantenha ambas coerentes.

A mesma API headless está disponível sem importar PyQt:

```python
from core.recipes import capture_project_recipe, capture_recipe, load_recipe, save_recipe, run_recipe

capture_recipe("workflow.json", ["-i", "stack_folder", "-o", "result/fused.png", "-m", "dct"])
capture_project_recipe("from-project.json", "session.ofproj",
                       ["-i", "new_stack", "-o", "result/fused.png"])
recipe = load_recipe("workflow.json")  # dicionário validado com caminhos relativos
save_recipe("workflow.json", recipe)
exit_code = run_recipe("workflow.json", dry_run=True)  # use False para processar
```

`capture_recipe` interpreta seus argumentos como caminhos de CLI comuns relativos ao diretório atual; `save_recipe` interpreta os caminhos armazenados relativos à receita. A receita registra configurações e referências, incluindo uma impressão digital de conteúdo para uma máscara de defeitos de sensor selecionada explicitamente. Ela não registra os pixels de entrada, as versões do software, o hardware nem os checksums da pilha de origem; guarde-os separadamente quando a proveniência importar. Configurações idênticas não garantem pixels idênticos entre versões de software, drivers ou dispositivos; operações neurais em GPU podem ser não determinísticas. Use `--cpu` quando uma execução em CPU for necessária e conserve o software instalado e as entradas.

### Reproduzir um checkpoint de retoque

Use os controles de salvar/carregar pontos de controle do espaço de retoque com a base original e as origens processadas. Sessões com fundo automático, incluindo análises bem-sucedidas sem mudanças nos pixels, salvam instantâneos ZIP v2 `.ofretouch` sem perdas com metadados JSON e dados NumPy: a imagem de trabalho exata, os blocos acessíveis de desfazer/refazer, os estados de diagnóstico e a procedência das origens. Carregar v2 não recalcula o fundo. Sessões apenas de pincel/tons podem usar JSON v1 compacto; pontos de controle de fundo antigos em JSON v1 continuam reproduzíveis.

A verificação das origens ocorre em segundo plano e pode ser cancelada. Antes de substituir o destino, o salvamento recarrega o arquivo preparado e confere a igualdade exata da imagem atual, dos blocos acessíveis de desfazer/refazer e dos estados de diagnóstico. Origem incompatível, falha de verificação, cancelamento ou memória insuficiente mantêm as edições atuais e qualquer ponto de controle existente inalterados.

Se o limite de registro do checkpoint impedir outra edição, aplique as edições atuais e inicie uma nova sessão de retoque para continuar.

Salve um checkpoint de retoque no espaço de retoque, depois forneça a imagem base original e cada quadro de origem processado na mesma ordem da sessão:

```bash
openfocus retouch replay edits.ofretouch --base fused.png --sources frame1.png frame2.png --output retouched.tif
```

Ambos os formatos ainda exigem a base original e todos os quadros realmente usados para o retoque, na ordem da sessão, após alinhamento e qualquer corte de ROI ou outro processamento, com seus valores de pixel e profundidade de bits originais. O carregamento reconstrói os deslocamentos de quadro e o comportamento de alinhamento registrados e verifica a base, os quadros ordenados e as máscaras de validade; pixels, geometrias ou máscaras incompatíveis são rejeitados. Cada caminho deve conter uma imagem. A saída deve ser PNG ou TIFF e preserva a profundidade de bits da base. O ponto de controle e as imagens de entrada não podem ser usados como saída.

Códigos de saída: `0` sucesso, `1` erro de processamento, `2` erro de uso. Executar `python main.py --help` mostra todas as opções; iniciar com um caminho simples (`python main.py C:\stack`) ainda abre a interface com essa pasta carregada.

---

## 15. Atalhos de teclado

### Operações de arquivo

| Atalho | Ação |
|----------|--------|
| `Ctrl+O` | Abrir pasta |
| `Ctrl+Shift+O` | Abrir vídeo |
| `Ctrl+Alt+O` | Abrir projeto |
| `Ctrl+Alt+S` | Salvar projeto |
| `Ctrl+S` | Salvar resultado |
| `Ctrl+Shift+S` | Salvar pilha registrada |
| `Ctrl+W` | Limpar pilha |
| `Ctrl+Q` | Sair |
| `Ctrl+R` | Iniciar renderização |
| `Ctrl+Shift+R` | Comparar todos os métodos |
| `Ctrl+T` | Retocar resultado |
| `Ctrl+M` | Mapa de profundidade |
| `Ctrl+E` | Alternar comparação cortina |
| `Ctrl+Shift+B` | Processamento em lote |
| `Ctrl+Z` / `Ctrl+Y` | Desfazer / Refazer (edições de pilha) |

### Navegação

| Atalho | Ação |
|----------|--------|
| `Cima` / `Baixo` (lista de origem em foco) | Quadro anterior / seguinte |
| `Roda do mouse` | Aumentar/diminuir zoom |
| `Ctrl + roda` | Zoom fino |
| `Clique duplo` | Alternar ajuste/100% |

### Geral

| Atalho | Ação |
|----------|--------|
| `Espaço` (mantido, ponteiro sobre uma imagem) | Modo de deslocamento temporário |
| `Delete` | Exclui o item de lista em foco; também limpa a ROI quando o ponteiro está sobre a imagem de origem |

---

## 16. Solução de problemas

### Problemas comuns

#### "Nenhuma imagem carregada"

- Verifique se você importou antes uma pasta de imagens ou um vídeo
- Confira se a pasta contém formatos de imagem suportados

#### Os resultados da fusão parecem borrados ou com fantasmas

- Ative o registro de imagens (escolha homografia / ECC / OpenFocus rápido)
- Garanta que as imagens estejam alinhadas antes da fusão
- Experimente outro algoritmo de fusão
- Aumente o tamanho do kernel para resultados mais suaves

#### IA indisponível

- Verifique o ONNX Runtime e o modelo incluído.
- Verifique em **Ajuda → Informações do ambiente**

#### Erros de falta de memória

- Ative o processamento em blocos em **Configurações → Configurações avançadas → Desempenho**
- Reduza o tamanho do bloco
- Use kernels menores
- Redimensione as imagens

#### Falha no registro

- Use **OpenFocus (rápido)** para movimento rígido limitado, ou ative **Parar em quadro rejeitado** para falhar cedo em vez de aceitar quadros não alinhados
- Reduza a largura de redução em **Configurações → Configurações avançadas → Registro**
- Garanta que as imagens tenham contraste e características suficientes
- Ative **Coletar diagnósticos de alinhamento**, renderize novamente e leia **Configurações → Configurações avançadas → Registro → Diagnósticos de alinhamento** para os quadros com falha

#### Processamento lento

- Deixe os threads em automático (`0`); capacidade de CPU, modo de recursos e margem de memória já limitam a concorrência
- Use o modo de recursos Máximo para um lote não assistido, ou filtro guiado / GFG-FGF em vez de um método de IA
- Reduza a resolução das imagens ou use **Prévia rápida** durante o ajuste

#### Aviso do Windows SmartScreen na primeira inicialização

A assinatura dos binários do Windows depende da versão específica (consulte as notas da versão); a assinatura não é garantida. Baixe-os de **https://openfocus.top** em vez de um espelho de terceiros. Se o SmartScreen avisar, clique em **Mais informações → Executar assim mesmo**. A primeira inicialização após descompactar pode levar 30–60 segundos.

#### O macOS diz que o aplicativo "não pode ser aberto" ou "está danificado"

A compilação do macOS não é notarizada. Baixe a compilação Apple Silicon em **https://openfocus.top**. Se instalar pelo DMG movendo `OpenFocus.app` para Aplicativos, execute:

```bash
xattr -cr /Applications/OpenFocus.app
```

Se instalar pelo terminal com `install.sh`, o app fica em `~/Applications`; execute `xattr -cr ~/Applications/OpenFocus.app`.

ou clique com o botão direito no aplicativo e escolha **Abrir**. As compilações macOS publicadas são para Apple Silicon (M1/M2/M3/M4); Macs Intel podem executar a partir do código-fonte.

### Obtendo ajuda

1. **Informações do ambiente**: vá em **Ajuda → Informações do ambiente** para verificar sua instalação
2. **Contatar suporte**: vá em **Ajuda → Fale conosco** para informações de suporte

---

## Apêndice: detalhes dos algoritmos de fusão

### Filtro guiado

O filtro guiado realiza suavização que preserva bordas usando uma imagem de orientação. Calcula transformações lineares locais entre a orientação e as imagens de entrada, depois as aplica para produzir a saída. Este método se destaca em preservar bordas enquanto reduz ruído.

### DCT (Transformada Discreta de Cosseno)

A DCT transforma as imagens para o domínio da frequência, onde as decisões de fusão se baseiam na magnitude dos coeficientes. Componentes de alta frequência (detalhes) são preservados enquanto os de baixa frequência (regiões suaves) são fundidos conforme a energia local.

### DTCWT (Transformada Wavelet Complexa de Árvore Dupla)

A DTCWT fornece decomposição multiescala e multidirecional das imagens. Captura características em diferentes escalas e orientações, permitindo regras de fusão sofisticadas que preservam textura e estrutura.

### GFG-FGF

Implementa Fu et al. (2024): o mapa de foco grosseiro |I − mean(I)| é enriquecido pelo gradiente de quatro vizinhanças do tipo gaussiano — o template 3×3 T = [1 2 1; 2 4 2; 1 2 1]/16 dá P = I⊗T, e G = (|I(x±1,y) − P|)/2 + (|I(x,y±1) − P|)/2, descartando valores abaixo de T0 = 0,005 — e depois refinado por um filtro guiado rápido. Cada pixel escolhe o quadro com o maior valor de foco (empates: medida de pequena região Sobel 3×3), e os mapas de decisão binários são suavizados pelo filtro guiado rápido antes da mistura ponderada.

### Fusão por IA (StackMFF V3)

Abordagem de aprendizado profundo usando uma rede neural treinada em grandes conjuntos de pares de imagens multifoco. A rede aprende estratégias de fusão ideais automaticamente e oferece resultados de ponta quando configurada corretamente.

---

*O OpenFocus é um projeto de código aberto. Para atualizações, relatos de bugs e solicitações de recursos, visite o repositório do projeto.*
