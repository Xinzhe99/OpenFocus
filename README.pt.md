<p align="center">
  <a href="https://github.com/Xinzhe99/OpenFocus">
    <img src="assets/OpenFocus.png" alt="OpenFocus" width="128">
  </a>
</p>

<h1 align="center">OpenFocus</h1>

<p align="center"><strong>Todos os planos de foco. Uma imagem totalmente nítida.</strong></p>

<p align="center">
  Oito algoritmos de fusão, três modos de registro e um espaço de retoque ligado à fonte para macro, microscopia e pilhas industriais. 16 bits de ponta a ponta, CLI sem interface e uma UI em doze idiomas.
</p>

<p align="center">
  <a href="https://github.com/Xinzhe99/OpenFocus/releases"><img src="https://img.shields.io/github/v/release/Xinzhe99/OpenFocus?label=release&color=2f81f7" alt="Release"></a>
  <a href="https://github.com/Xinzhe99/OpenFocus/stargazers"><img src="https://img.shields.io/github/stars/Xinzhe99/OpenFocus?color=e3b341" alt="Stars"></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/license-MIT-3fb950" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Windows-10%2F11-0078D7?logo=windows&logoColor=white" alt="Windows 10/11">
  <img src="https://img.shields.io/badge/macOS-Apple%20Silicon-000000?logo=apple&logoColor=white" alt="macOS Apple Silicon">
  <img src="https://img.shields.io/badge/Linux-CLI-FCC624?logo=linux&logoColor=black" alt="Linux CLI">
  <a href="https://pypi.org/project/openfocus/"><img src="https://img.shields.io/badge/PyPI-openfocus-3775A9?logo=pypi&logoColor=white" alt="PyPI"></a>
</p>

<p align="center">
  <a href="#download">Baixar</a> ·
  <a href="./docs/USER_MANUAL_PT.md">Documentação</a> ·
  <a href="#command-line-usage">CLI</a> ·
  <a href="./CHANGELOG.md">Alterações</a> ·
  <a href="https://doi.org/10.5281/zenodo.23050823">DOI</a>
</p>

<p align="center">
  <a href="README.md">English</a> ·
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
  <img src="assets/ui.png" alt="Interface do OpenFocus" width="880">
</p>

## 📢 News

> [!NOTE]
> **v1.68.1** — a exportação em lote segue o modo escolhido: apenas alinhamento cria uma pasta `<pilha>_aligned` e a lista, a fusão lista só a imagem fundida, e sem exportar o alinhamento grava apenas essa imagem. Cópias `*_registered*` deixadas na pasta de origem não são mais carregadas como quadros. Também foram refeitos rótulos, barras de escala, comparação de métodos, exportação de profundidade/modelo, atualizações e retoque conforme a auditoria em `docs/ux-specs`.

<a id="download"></a>
## Download e instalação

Baixe uma versão em [Releases](https://github.com/Xinzhe99/OpenFocus/releases):

| Plataforma | Arquivo | Tipo |
|------------|---------|------|
| Windows 10/11 (64 bits) | `OpenFocus-*-setup.exe` | **Instalador** (recomendado) |
| Windows 10/11 (64 bits) | `OpenFocus-*-windows-x64.zip` | Portátil (descompactar e executar) |
| macOS Apple Silicon (M1–M4) | `OpenFocus-*-macos.dmg` | Imagem de disco (arrastar para Aplicativos) |
| macOS Apple Silicon (M1–M4) | `OpenFocus-*-macos-arm64.zip` | Portátil (descompactar) |

O aplicativo não é assinado; no primeiro início o Windows pode pedir “Mais informações → Executar assim mesmo” e o macOS pode exigir clique direito → Abrir ou `xattr -cr /Applications/OpenFocus.app`.

## Recursos principais

<table>

<tr>
<td width="46%" valign="middle">

### Fusão por IA: StackMFF V3

O método padrão classifica cada pixel pelo quadro de origem mais provável com uma rede neural. Na mesma pilha registrada, mantém as bordas das letras mais nítidas e as superfícies mais limpas que o Guided Filter (nitidez laplaciana 27,1 contra 28,5) e lidera todos os métodos em PSNR e SSIM contra verdade sintética, com menos pixels fantasmas que qualquer algoritmo clássico.

[Algoritmos →](#algorithms) · [Benchmark →](tools/synthetic_benchmark.py)

</td>
<td width="54%">

<img src="assets/comparison_ai_vs_gff.png" alt="Fusão por IA: StackMFF V3" width="100%" />

</td>
</tr>

<tr>
<td width="46%" valign="middle">

### Comparação A/B

Arraste o divisor entre um quadro e o resultado para ver o que a fusão mudou: desvio de registro, halos nas bordas e detalhe perdido, em qualquer zoom.

[Manual →](./docs/USER_MANUAL_PT.md)

</td>
<td width="54%">

<img src="assets/readme/wipe.png" alt="Comparação A/B entre um quadro e o resultado" width="100%" />

</td>
</tr>

<tr>
<td width="46%" valign="middle">

### Comparação de métodos

Renderize todos os métodos uma vez, compare lado a lado no mesmo recorte e adote o vencedor como nova saída ou copie seus parâmetros para a janela principal.

</td>
<td width="54%">

<img src="assets/readme/compare.png" alt="Oito métodos de fusão em paralelo" width="100%" />

</td>
</tr>

<tr>
<td width="46%" valign="middle">

### Mapa de profundidade e 3D

Posição de foco por pixel com cinco medidas, mapas de cor científicos, calibração Z física dos metadados TIFF e exportação OBJ texturizada da superfície.

</td>
<td width="54%">

<img src="assets/readme/depth.png" alt="Mapa de profundidade e superfície 3D da mesma pilha" width="100%" />

</td>
</tr>

<tr>
<td width="46%" valign="middle">

### Retoque ligado à fonte

Pinte a partir dos quadros que realmente produziram o resultado: zoom e deslocamento compartilhados, controle de quadro, pincel suave, pincel de restauração, desfazer por traço e proteção de pixels não exportados.

</td>
<td width="54%">

<img src="assets/readme/retouch.png" alt="Espaço de retoque com quadro e resultado" width="100%" />

</td>
</tr>

<tr>
<td width="46%" valign="middle">

### Processamento em lote

Aponte para uma pasta de pilhas, escolha método e destino e deixe rodar. Somente alinhamento grava uma pasta por pilha; a fusão grava e lista apenas a imagem fundida.

</td>
<td width="54%">

<img src="assets/readme/batch.png" alt="Diálogo de processamento em lote" width="100%" />

</td>
</tr>

<tr>
<td width="46%" valign="middle">

### Barras de escala de publicação

Calibradas por metadados TIFF ou manualmente, desenhadas com a fonte e o comprimento reais e gravadas em toda exportação: imagem, pilha, TIFF, GIF e lote. Se não couber de forma legível, o diálogo avisa em vez de desenhar uma barra menor que o rótulo.

[Manual →](./docs/USER_MANUAL_PT.md)

</td>
<td width="54%">

<img src="assets/readme/scalebar.png" alt="Diálogo de barra de escala com pré-visualização" width="100%" />

</td>
</tr>

<tr>
<td width="46%" valign="middle">

### Rótulos e legendas

As legendas ficam numa lista gerenciada: cada uma com seu escopo (fonte, pilha alinhada ou uma saída específica), faixa de quadros e estilo. A pré-visualização e todas as exportações usam o mesmo destino e número de quadro, então `{value}` lê igual em Salvar, Salvar como, arrastar e Exportar tudo.

</td>
<td width="54%">

<img src="assets/readme/labels.png" alt="Rótulos e legendas" width="100%" />

</td>
</tr>
</table>

**Também incluído:**

- **Fluxo de focus stacking** — navegador de pilhas, curva de nitidez, pré-visualização rápida, render ROI e ajustes por método lembrados entre sessões.
- **Pronto para microscopia** — 16 bits de ponta a ponta, µm/px do TIFF, correção de campo plano, Z-stacks TIFF multipágina.
- **CLI sem interface** — a mesma linha de processamento como `openfocus`, com lotes, `--dry-run`, `--resume` e saída JSON.
- **Projetos e recuperação** — sessões `.ofproj`, recuperação após falha, desfazer/refazer em pilhas e anotações.
- **Doze idiomas de interface** — English, 简体中文, 日本語, Español, Deutsch, Français, Русский, 한국어, Português, Italiano, Nederlands, Polski; detectados do sistema, um idioma novo é um arquivo `locales_<code>.py`.
- **Leve na memória** — os pesos neurais só carregam ao usar o método AI, então uma sessão clássica fica em torno de 110 MB em vez de 260 MB.

## Início rápido

```bash
# CLI completa: todos os métodos, modelo de IA, HEIC e RAW de câmera
pip install openfocus

# Fundir uma pasta de imagens
openfocus -i ./stack_folder -o ./result/fused.png

# DTCWT com ECC e 8 threads
openfocus -i ./stack_folder -o ./result/fused.png -m dtcwt -a ecc -t 8
```

Executar a GUI a partir do código:

```bash
conda create -n openfocus python=3.10
conda activate openfocus
pip install -r requirements.txt
python main.py
```

## Documentação

- [Manual do usuário (português)](./docs/USER_MANUAL_PT.md)
- [Histórico de alterações](./CHANGELOG.md)
- [Desempenho e pilhas grandes](./docs/PERFORMANCE.md)
- [Comandos de compilação](./docs/BUILD_COMMANDS.md)
- [Paridade de algoritmos](./docs/ALGORITHM_PARITY.md)

A referência completa em inglês está em [README.md](./README.md).

## Citação e licença

Se o OpenFocus ajudar seu trabalho, cite o projeto usando [CITATION.cff](./CITATION.cff).

O projeto é distribuído sob a [licença MIT](./LICENSE). Cada algoritmo de fusão/registro mantém também sua licença original.
