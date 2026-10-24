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

# <img src="assets/OpenFocus.png" alt="OpenFocus Logo" width="120"> OpenFocus

OpenFocus é uma estação de trabalho de código aberto para registro e fusão de imagens de foco múltiplo, com qualidade comparável à de softwares comerciais de focus stacking e código totalmente aberto e extensível.

<p align="left">
  <a href="https://www.python.org/downloads/release/python-3100/"><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+" /></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?logo=open-source-initiative&logoColor=white" alt="License: MIT" /></a>
  <a href="https://github.com/Xinzhe99/OpenFocus"><img src="https://img.shields.io/badge/GitHub-Repository-181717?logo=github&logoColor=white" alt="GitHub Repository" /></a>
  <a href="https://github.com/Xinzhe99/OpenFocus/releases"><img src="https://img.shields.io/badge/Windows%20%7C%20macOS-Download-0078D7?logo=github&logoColor=white" alt="Download" /></a>
  <a href="https://doi.org/10.5281/zenodo.23050823"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.23050823-0078D7?logo=doi&logoColor=white" alt="DOI" /></a>
</p>

<p align="center">
  <img src="assets/ui.png" alt="Interface do OpenFocus" width="820">
</p>

> **v1.68.0** — a exportação em lote segue o modo escolhido: apenas alinhamento cria uma pasta `<pilha>_aligned` e a lista, a fusão lista só a imagem fundida, e sem exportar o alinhamento grava apenas essa imagem. Cópias `*_registered*` deixadas na pasta de origem não são mais carregadas como quadros. Também foram refeitos rótulos, barras de escala, comparação de métodos, exportação de profundidade/modelo, atualizações e retoque conforme a auditoria em `docs/ux-specs`.

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

- **Oito métodos de fusão** — Guided Filter, DCT, DTCWT, GFG-FGF, StackMFF-V3 (rede neural) e OpenFocus A/B/C.
- **Três modos de registro** — similaridade rápida, ECC e homografia; também é possível fundir sem registro.
- **Fluxo de focus stacking** — navegador de pilha, comparação A/B, mapas de profundidade, comparação de métodos, processamento em lote e pré-visualização rápida.
- **Retoque com proveniência** — pixels dos quadros que realmente produziram o resultado, zoom/posição compartilhados, controle de quadro de origem, pincel suave, pincel de restauração, desfazer/refazer por traço e proteção de saídas não exportadas.
- **Pronto para microscopia e publicação** — 16 bits ponta a ponta, µm/px de TIFF, correção de campo plano, barras de escala, Z-stack TIFF multipágina e exportação OBJ texturizada.
- **CLI sem interface** — o mesmo pipeline via `openfocus`, com lote, `--dry-run`, `--resume` e saída JSON.
- **Doze idiomas de interface** — English, 简体中文, 日本語, Español, Deutsch, Français, Русский, 한국어, Português, Italiano, Nederlands, Polski.

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
