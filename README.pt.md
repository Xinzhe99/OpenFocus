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

- **Oito métodos de fusão** — Guided Filter, DCT, DTCWT, GFG-FGF, StackMFF-V3 (rede neural) e OpenFocus A/B/C.
- **Três modos de registro** — similaridade rápida, ECC e homografia; também é possível fundir sem registro.
- **Fluxo de focus stacking** — navegador de pilha, comparação A/B, mapas de profundidade, comparação de métodos, processamento em lote e pré-visualização rápida.
- **Retoque com proveniência** — pixels dos quadros que realmente produziram o resultado, zoom/posição compartilhados, controle de quadro de origem, pincel suave, pincel de restauração, desfazer/refazer por traço e proteção de saídas não exportadas.
- **Pronto para microscopia e publicação** — 16 bits ponta a ponta, µm/px de TIFF, correção de campo plano, barras de escala, Z-stack TIFF multipágina e exportação OBJ texturizada.
- **CLI sem interface** — o mesmo pipeline via `openfocus`, com lote, `--dry-run`, `--resume` e saída JSON.
- **Doze idiomas de interface** — English, 简体中文, 日本語, Español, Deutsch, Français, Русский, 한국어, Português, Italiano, Nederlands, Polski.

<a id="command-line-usage"></a>
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
