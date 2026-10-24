<p align="center">
  <a href="https://github.com/Xinzhe99/OpenFocus">
    <img src="assets/OpenFocus.png" alt="OpenFocus" width="128">
  </a>
</p>

<h1 align="center">OpenFocus</h1>

<p align="center"><strong>Tutti i piani di fuoco. Una sola immagine completamente nitida.</strong></p>

<p align="center">
  Otto algoritmi di fusione, tre modalità di registrazione e un'area di ritocco legata alla sorgente per macro, microscopia e stack industriali. 16 bit end-to-end, CLI headless e un'interfaccia in dodici lingue.
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
  <a href="#download">Download</a> ·
  <a href="./docs/USER_MANUAL_IT.md">Documentazione</a> ·
  <a href="#command-line-usage">CLI</a> ·
  <a href="./CHANGELOG.md">Novità</a> ·
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
  <a href="README.pt.md">Português</a> ·
  <strong>Italiano</strong> ·
  <a href="README.nl.md">Nederlands</a> ·
  <a href="README.pl.md">Polski</a>
</p>

<p align="center">
  <img src="assets/ui.png" alt="Interfaccia di OpenFocus" width="880">
</p>

## 📢 News

> [!NOTE]
> **v1.68.1** — l'esportazione batch segue la modalità scelta: solo allineamento crea una cartella `<stack>_aligned` e la elenca, la fusione elenca solo l'immagine fusa, e senza esportare l'allineamento scrive solo quell'immagine. Le copie `*_registered*` rimaste nella cartella sorgente non vengono più caricate come fotogrammi. Rivisti anche etichette, barre di scala, confronto metodi, export di profondità/modello, aggiornamenti e ritocco secondo l'audit in `docs/ux-specs`.

<a id="download"></a>
## Download e installazione

Scarica una build dalla pagina [Releases](https://github.com/Xinzhe99/OpenFocus/releases):

| Piattaforma | File | Tipo |
|-------------|------|------|
| Windows 10/11 (64 bit) | `OpenFocus-*-setup.exe` | **Installer** (consigliato) |
| Windows 10/11 (64 bit) | `OpenFocus-*-windows-x64.zip` | Portabile (estrarre ed eseguire) |
| macOS Apple Silicon (M1–M4) | `OpenFocus-*-macos.dmg` | Immagine disco (trascinare in Applicazioni) |
| macOS Apple Silicon (M1–M4) | `OpenFocus-*-macos-arm64.zip` | Portabile (estrarre) |

L’app non è firmata: al primo avvio Windows può chiedere “Ulteriori informazioni → Esegui comunque”; su macOS usa clic destro → Apri oppure `xattr -cr /Applications/OpenFocus.app`.

## Funzionalità

- **Otto metodi di fusione** — Guided Filter, DCT, DTCWT, GFG-FGF, StackMFF-V3 (rete neurale) e OpenFocus A/B/C.
- **Tre modalità di registrazione** — similarità rapida, ECC e omografia; è possibile anche fondere senza registrazione.
- **Workflow focus stacking** — browser dello stack, confronto A/B, mappe di profondità, confronto metodi, elaborazione batch e anteprima rapida.
- **Ritocco con provenienza** — pixel dai frame che hanno realmente prodotto il risultato, zoom/posizione condivisi, cursore del frame sorgente, pennello morbido, pennello di ripristino, annulla/ripeti per tratto e protezione degli output non esportati.
- **Pronto per microscopia e pubblicazione** — 16 bit end-to-end, µm/px da TIFF, correzione flat-field, barre di scala, Z-stack TIFF multipagina ed export OBJ texturizzato.
- **CLI headless** — la stessa pipeline con `openfocus`, inclusi batch, `--dry-run`, `--resume` e output JSON.
- **Dodici lingue UI** — English, 简体中文, 日本語, Español, Deutsch, Français, Русский, 한국어, Português, Italiano, Nederlands, Polski.

<a id="command-line-usage"></a>
## Avvio rapido

```bash
# CLI completa: tutti i metodi, modello IA, HEIC e RAW della fotocamera
pip install openfocus

# Fondere una cartella di immagini
openfocus -i ./stack_folder -o ./result/fused.png

# DTCWT con ECC e 8 thread
openfocus -i ./stack_folder -o ./result/fused.png -m dtcwt -a ecc -t 8
```

Avviare la GUI dai sorgenti:

```bash
conda create -n openfocus python=3.10
conda activate openfocus
pip install -r requirements.txt
python main.py
```

## Documentazione

- [Manuale utente (italiano)](./docs/USER_MANUAL_IT.md)
- [Registro delle modifiche](./CHANGELOG.md)
- [Prestazioni e stack di grandi dimensioni](./docs/PERFORMANCE.md)
- [Comandi di build](./docs/BUILD_COMMANDS.md)
- [Parità degli algoritmi](./docs/ALGORITHM_PARITY.md)

Il riferimento completo in inglese è in [README.md](./README.md).

## Citazione e licenza

Se OpenFocus è utile al tuo lavoro, cita il progetto tramite [CITATION.cff](./CITATION.cff).

Il progetto è distribuito con [licenza MIT](./LICENSE). Ogni algoritmo di fusione/registrazione mantiene anche la propria licenza originale.
