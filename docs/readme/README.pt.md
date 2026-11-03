<p align="center">
  <a href="https://github.com/Xinzhe99/OpenFocus">
    <img src="../../assets/OpenFocus.png" alt="OpenFocus" width="128">
  </a>
</p>

<h1 align="center">OpenFocus</h1>

<p align="center"><strong>Cada pixel, do seu quadro mais nítido.</strong></p>

<p align="center">
  Dez algoritmos de fusão, cinco opções de alinhamento (incluindo nenhum) e um espaço de retoque vinculado à fonte para macro, microscopia e pilhas industriais. 16 bits de ponta a ponta, uma CLI sem interface gráfica e uma interface em doze idiomas.
</p>

<p align="center">
  <a href="https://github.com/Xinzhe99/OpenFocus/releases"><img src="https://img.shields.io/github/v/release/Xinzhe99/OpenFocus?label=release&color=2f81f7" alt="Versão"></a>
  <a href="https://github.com/Xinzhe99/OpenFocus/stargazers"><img src="https://img.shields.io/github/stars/Xinzhe99/OpenFocus?color=e3b341" alt="Estrelas"></a>
  <a href="../../LICENSE"><img src="https://img.shields.io/badge/license-MIT-3fb950" alt="Licença: MIT"></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Windows-10%2F11-0078D7?logo=windows&logoColor=white" alt="Windows 10/11">
  <img src="https://img.shields.io/badge/macOS-Apple%20Silicon-000000?logo=apple&logoColor=white" alt="macOS Apple Silicon">
  <img src="https://img.shields.io/badge/Linux-CLI-FCC624?logo=linux&logoColor=black" alt="Linux CLI">
  <a href="https://pypi.org/project/openfocus/"><img src="https://img.shields.io/badge/PyPI-openfocus-3775A9?logo=pypi&logoColor=white" alt="PyPI"></a>
</p>

<p align="center">
  <a href="#install">Baixar</a> ·
  <a href="../USER_MANUAL_PT.md">Documentação</a> ·
  <a href="../../CHANGELOG.md">Alterações</a> ·
  <a href="https://doi.org/10.5281/zenodo.23050823">DOI</a>
</p>

<p align="center">
  <a href="../../README.md">English</a> ·
  <a href="README.zh-CN.md">简体中文</a> ·
  <a href="README.ja.md">日本語</a> ·
  <a href="README.es.md">Español</a> ·
  <a href="README.de.md">Deutsch</a> ·
  <a href="README.fr.md">Français</a> ·
  <a href="README.ru.md">Русский</a> ·
  <a href="README.ko.md">한국어</a> ·
  <strong>Português</strong> ·
  <a href="README.it.md">Italiano</a> ·
  <a href="README.nl.md">Nederlands</a> ·
  <a href="README.pl.md">Polski</a>
</p>

<p align="center">
  <img src="../../assets/ui.png" alt="Interface do OpenFocus" width="880">
</p>

## 📢 Novidades

> [!NOTE]
> **v1.75.0** — Fusão agrupada, fluxos de processamento reutilizáveis, pilhas PNG transparentes, diagnóstico de alinhamento, exportações de profundidade aprimoradas e fluxo de desktop renovado. Consulte o registro de alterações para entradas suportadas e limitações.

As versões anteriores estão no [registro de alterações](../../CHANGELOG.md).

📸 Se você publicar imagens criadas com o OpenFocus, considere adicionar uma nota como:

> Criado com OpenFocus — https://github.com/Xinzhe99/OpenFocus

Não é obrigatório, mas é muito apreciado.

<a id="features"></a>
## Recursos

<table>
<tr>
<td width="46%" valign="middle">

### Fusão por IA: StackMFF V3

O método padrão usa uma rede neural para estimar o quadro de origem mais provável de cada pixel e depois mistura as contribuições previstas ou escolhe o quadro mais provável. O benchmark sintético compara os métodos com uma referência conhecida; os resultados dependem da pilha e das configurações.

[StackMFF V3 →](https://github.com/Xinzhe99/StackMFF-V3) · [Benchmark →](../../tools/synthetic_benchmark.py)

</td>
<td width="54%">

<a href="../../assets/readme/ai.mp4"><picture><source srcset="../../assets/readme/ai.gif" type="image/gif"><img src="../../assets/readme/ai_poster.jpg" alt="StackMFF V3 contra o filtro guiado na mesma pilha" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Comparação cortina

Arraste o divisor entre um quadro e o resultado fundido para ver exatamente o que a mistura mudou: deriva de alinhamento, halos nas bordas e detalhe perdido, em qualquer zoom.

[Manual →](../USER_MANUAL_PT.md)

</td>
<td width="54%">

<a href="../../assets/readme/wipe.mp4"><picture><source srcset="../../assets/readme/wipe.gif" type="image/gif"><img src="../../assets/readme/wipe_poster.jpg" alt="Comparação A/B entre um quadro de origem e o resultado fundido" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Comparação de métodos

Por padrão, cada método funde a imagem inteira; o quadrado de inspeção da origem permanece interativo sem repetir os cálculos da imagem inteira. Depois, compare os resultados lado a lado no mesmo recorte e adote o vencedor como nova saída ou copie seus parâmetros para a janela principal.

</td>
<td width="54%">

<a href="../../assets/readme/method-comparison.mp4"><picture><source srcset="../../assets/readme/method-comparison.gif" type="image/gif"><img src="../../assets/readme/method-comparison_poster.jpg" alt="Comparação lado a lado de oito métodos de fusão" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Mapa de profundidade e 3D

Posição de foco por pixel com sete métodos compatíveis com profundidade, mapas de cor científicos, calibração Z física manual (início e passo; os metadados TIFF fornecem apenas a calibração XY) e exportação OBJ texturizada da superfície.

</td>
<td width="54%">

<a href="../../assets/readme/depth.mp4"><picture><source srcset="../../assets/readme/depth.gif" type="image/gif"><img src="../../assets/readme/depth_poster.jpg" alt="Mapa de profundidade e superfície 3D da mesma pilha" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Retoque vinculado à fonte

Pinte a partir dos quadros que realmente produziram o resultado: zoom e deslocamento compartilhados, controle do quadro de origem, pincel suave, pincel de restauração, desfazer por traço e proteção de pixels não exportados. Aplique o retoque como nova saída e use **Exportar TIFF em camadas** para salvá-la com seus quadros de origem. Rótulos e barras de escala ativos não são incluídos. O GIMP 3.2.6 no Windows abre o resultado e as camadas de origem; outros editores não foram testados.

</td>
<td width="54%">

<a href="../../assets/readme/retouch.mp4"><picture><source srcset="../../assets/readme/retouch.gif" type="image/gif"><img src="../../assets/readme/retouch_poster.jpg" alt="Espaço de retoque com quadro de origem e resultado" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Processamento em lote

Aponte para uma pasta de pilhas, escolha um método e um destino, e deixe trabalhar. O modo somente alinhamento grava uma pasta por pilha; a fusão grava a imagem fundida e lista exatamente isso. A fusão agrupada opcional troca a equivalência exata com a pilha completa por memória e espaço temporário limitados.

</td>
<td width="54%">

<a href="../../assets/readme/batch.mp4"><picture><source srcset="../../assets/readme/batch.gif" type="image/gif"><img src="../../assets/readme/batch_poster.jpg" alt="Diálogo de processamento em lote" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Barras de escala para publicação

Calibradas a partir de metadados TIFF ou digitadas manualmente, desenhadas com a fonte e o comprimento reais e incorporadas em todas as exportações: imagens avulsas, pilhas, TIFF multipágina, GIF e lotes. Uma barra que não cabe de forma legível avisa em vez de imprimir um comprimento que não desenha.

[Manual →](../USER_MANUAL_PT.md)

</td>
<td width="54%">

<a href="../../assets/readme/scalebar.mp4"><picture><source srcset="../../assets/readme/scalebar.gif" type="image/gif"><img src="../../assets/readme/scalebar_poster.jpg" alt="Diálogo da barra de escala com prévia" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Rótulos e legendas

Mantenha uma lista gerenciada de legendas, cada uma com seu escopo (pilha de origem, pilha alinhada ou uma saída específica), intervalo de quadros e estilo. A prévia e todas as exportações usam o mesmo destino e número de quadro, então `{value}` é lido da mesma forma em Salvar, Salvar como, arrastar e Exportar tudo.

</td>
<td width="54%">

<a href="../../assets/readme/labels.mp4"><picture><source srcset="../../assets/readme/labels.gif" type="image/gif"><img src="../../assets/readme/labels_poster.jpg" alt="Gerenciador de rótulos com prévia" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Controlável por agente

Cada janela, widget e menu é acessível por um canal local protegido por token: um agente carrega uma pilha, renderiza, retoca, compara, calcula profundidade, exporta e responde a diálogos por comando tipado, caminho de widget ou eventos reais de mouse e teclado. Não é preciso capturar tela nem usar OCR; o canal inicia com o aplicativo de desktop.

[Protocolo →](../AGENT_CONTROL_SPEC.md) · [Skill →](../../skills/openfocus-control/SKILL.md)

</td>
<td width="54%">

<a href="../../assets/readme/agent.mp4"><picture><source srcset="../../assets/readme/agent.gif" type="image/gif"><img src="../../assets/readme/agent_poster.jpg" alt="Um agente controla a interface ao vivo" width="100%" /></picture></a>

</td>
</tr>
</table>

**Também incluído:**

- **Fluxo de focus stacking** — navegador de pilhas, curva de nitidez, prévia rápida, render por ROI e ajustes por método lembrados entre reinicializações.
- **Pronto para microscopia** — 16 bits de ponta a ponta, µm/px dos metadados TIFF, correção de campo plano por mapa de poeira, pilhas Z TIFF multipágina e exportação OBJ texturizada.
- **Perfis de cor** — exportações avulsas em PNG, TIFF, JPEG e WebP mantêm um perfil ICC RGB compartilhado quando todos os quadros de origem têm o mesmo perfil. Pilhas mistas ou sem perfil são exportadas sem perfil; os pixels não são convertidos entre espaços de cor.
- **PNG transparente** — pilhas com máscaras retangulares de opacidade constante correspondentes podem ser fundidas em tamanho real e salvas como PNG. Consulte as restrições no manual.
- **CLI sem interface** — o mesmo pipeline como `openfocus`, com modo lote, `--dry-run`, `--resume` e saída JSON para scripts.
- **Controlável por agente** — um canal local protegido por token controla a interface ao vivo (comandos de pipeline tipados e controle genérico de widgets, menus, mouse e teclado), com uma skill, um executor de fumaça de 14 verificações e uma varredura completa de ações de menu.
- **Projetos e recuperação** — sessões `.ofproj`, recuperação automática após falhas e desfazer/refazer em edições de pilha e anotações.
- **Doze idiomas de interface** — English, 中文, 日本語, Español, Deutsch, Français, Русский, 한국어, Português, Italiano, Nederlands, Polski. Detectados pela localidade do sistema; um novo idioma é um arquivo `locales/<code>.py`.
- **Leve em memória** — a rede neural só é carregada quando o método de IA é usado, então uma sessão com métodos clássicos fica em torno de 110 MB em vez de 260 MB.

<a id="install"></a>
## ⬇️ Instalação

Baixe a compilação mais recente para Windows ou macOS no [site do OpenFocus](https://openfocus.top/#download). Os mesmos arquivos e versões anteriores também estão no [GitHub Releases](https://github.com/Xinzhe99/OpenFocus/releases).

| Plataforma | Arquivo | Tipo |
|------------|---------|------|
| Windows 10/11 (64 bits) | `OpenFocus-*-setup.exe` | **Instalador** (recomendado) |
| Windows 10/11 (64 bits) | `OpenFocus-*-windows-x64.zip` | Portátil (descompacte e execute) |
| macOS Apple Silicon (M1–M4) | `OpenFocus-*-macos.dmg` | Imagem de disco (arraste para Aplicativos) |
| macOS Apple Silicon (M1–M4) | `OpenFocus-*-macos-arm64.zip` | Portátil (descompacte) |

O estado de assinatura das compilações do Windows consta nas notas da versão. As compilações do macOS não são notarizadas, então a primeira inicialização pode exibir um aviso de segurança: **Windows** SmartScreen → "Mais informações" → "Executar assim mesmo" (a primeira inicialização pode levar 30–60 s); **macOS** execute `xattr -cr /Applications/OpenFocus.app` no Terminal, ou tente abrir o aplicativo uma vez e permita-o em **Ajustes do Sistema → Privacidade e Segurança → Abrir assim mesmo** (a partir do macOS 15, clicar com o botão direito → Abrir não contorna mais Gatekeeper).

Para instalar pelo terminal, visualize primeiro o comando para macOS ou Linux e depois execute-o sem `--dry-run`. No Linux, isso instala a CLI sem interface; as compilações para macOS exigem Apple Silicon; o app do DMG instala-se em `/Applications` e a instalação pelo terminal em `~/Applications`.

```bash
curl -fsSL https://openfocus.top/install.sh | sh -s -- --dry-run
curl -fsSL https://openfocus.top/install.sh | sh
```

No Windows, execute o instalador pelo PowerShell:

```powershell
irm https://openfocus.top/install.ps1 | iex
```

O download direto do instalador do Windows também está disponível com
`curl -fL https://openfocus.top/download/latest/windows-setup -o OpenFocus-setup.exe`.

### 🤖 Integração com agentes

O aplicativo de desktop inicia automaticamente um canal local de agente protegido por token. Use `openfocus ctl` para comandos tipados ou ações genéricas de interface na janela ao vivo. Agentes podem carregar pilhas, renderizar, exportar, abrir e preencher diálogos, responder a seletores de arquivos, enviar eventos de mouse e teclado e tirar capturas de tela.

- Skill: [`skills/openfocus-control/SKILL.md`](../../skills/openfocus-control/SKILL.md)
- Protocolo e referência de comandos: [`docs/AGENT_CONTROL_SPEC.md`](../AGENT_CONTROL_SPEC.md)
- Executor de ponta a ponta: `python tools/agent_gui_smoke.py --launch`
- Varredura de todas as ações: `python tools/agent_action_sweep.py --launch`

### 💻 Uso pela linha de comando

Você também pode instalar a CLI no Windows, macOS e Linux — o mesmo pipeline, sem a pilha gráfica:

```bash
pip install openfocus
openfocus -i ./stack_folder -o ./result/fused.png
```

Consulte o [manual da CLI](../USER_MANUAL_PT.md#14-interface-de-linha-de-comando).

<a id="community"></a>
## Comunidade e suporte

- **Documentação** — [Manual](../USER_MANUAL_PT.md) (doze idiomas) · [Alterações](../../CHANGELOG.md) · [Desempenho](../PERFORMANCE.md) · [Métodos de fusão](../FUSION_METHODS.md)
- **Citação** — [CITATION.cff](../../CITATION.cff); cada versão é arquivada no Zenodo ([DOI](https://doi.org/10.5281/zenodo.23050823))
- **Feedback** — abra uma [issue](https://github.com/Xinzhe99/OpenFocus/issues) para bugs, ideias e trabalho nos algoritmos
- **Créditos** — cada algoritmo de fusão e registro mantém sua licença de pesquisa original; o StackMFF V3 fica em [Xinzhe99/StackMFF-V3](https://github.com/Xinzhe99/StackMFF-V3)

<a id="license"></a>
## Licença

O aplicativo OpenFocus é distribuído sob a [licença MIT](../../LICENSE).

**Exceção da série StackMFF.** Os algoritmos de fusão StackMFF — atualmente StackMFF V3 e qualquer futura edição de ponta — permanecem propriedade do projeto OpenFocus, que detém todos os direitos sobre a série StackMFF. Você pode usá-los como parte do OpenFocus para suas próprias imagens e trabalho interno. Copiar, portar ou migrar esses algoritmos para outro projeto de código aberto exige permissão prévia por escrito.

<a id="developing"></a>
## Desenvolvimento

Executar a partir de um clone:

```bash
conda create -n openfocus python=3.10
conda activate openfocus
pip install -r requirements.txt
python main.py
```

Empacotamento com PyInstaller: [docs/BUILD_COMMANDS.md](../BUILD_COMMANDS.md).

Adicionar um algoritmo de fusão ou registro: [CONTRIBUTING.md](../../CONTRIBUTING.md).
A [especificação de extensibilidade](../ALGORITHM_EXTENSIBILITY_SPEC.md) define os contratos compartilhados de registro, parâmetros, execução e persistência.

<a id="star-history"></a>
## ⭐ Histórico de estrelas

[![Star History Chart](https://api.star-history.com/svg?repos=Xinzhe99/OpenFocus&type=Date)](https://star-history.com/#Xinzhe99/OpenFocus&Date)
