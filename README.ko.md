<p align="center">
  <a href="https://github.com/Xinzhe99/OpenFocus">
    <img src="assets/OpenFocus.png" alt="OpenFocus" width="128">
  </a>
</p>

<h1 align="center">OpenFocus</h1>

<p align="center"><strong>모든 초점면을, 한 장의 완전 선명한 이미지로.</strong></p>

<p align="center">
  8가지 융합 알고리즘, 3가지 정합 모드, 그리고 매크로·현미경·산업용 스택을 위한 소스 고정 리터치 워크스페이스. 16비트 전 구간 유지, 헤드리스 CLI, 12개 언어 UI.
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
  <a href="#download">다운로드</a> ·
  <a href="./docs/USER_MANUAL_KO.md">문서</a> ·
  <a href="#command-line-usage">CLI</a> ·
  <a href="./CHANGELOG.md">변경 기록</a> ·
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
  <strong>한국어</strong> ·
  <a href="README.pt.md">Português</a> ·
  <a href="README.it.md">Italiano</a> ·
  <a href="README.nl.md">Nederlands</a> ·
  <a href="README.pl.md">Polski</a>
</p>

<p align="center">
  <img src="assets/ui.png" alt="OpenFocus 인터페이스" width="880">
</p>

## 📢 News

> [!NOTE]
> **v1.68.1** — 배치 출력이 선택한 모드을 따릅니다. 정렬만 하면 `<스택>_aligned` 폴더를 만들어 목록에 표시하고, 융합 시에는 융합 이미지만 표시하며, 정렬을 저장하지 않으면 융합 이미지만 기록합니다. 원본 폴더에 남은 `*_registered*` 사본은 프레임으로 로드되지 않습니다. `docs/ux-specs` 감사에 따라 라벨, 스케일 바, 방법 비교, 깊이/모델 내보내기, 업데이트, 리터치도 개편했습니다.

<a id="download"></a>
## 다운로드 및 설치

[Releases](https://github.com/Xinzhe99/OpenFocus/releases)에서 빌드를 받을 수 있습니다.

| 플랫폼 | 파일 | 유형 |
|--------|------|------|
| Windows 10/11(64비트) | `OpenFocus-*-setup.exe` | **설치 프로그램**(권장) |
| Windows 10/11(64비트) | `OpenFocus-*-windows-x64.zip` | 포터블(압축 해제 후 실행) |
| macOS Apple Silicon(M1–M4) | `OpenFocus-*-macos.dmg` | 디스크 이미지(응용 프로그램으로 드래그) |
| macOS Apple Silicon(M1–M4) | `OpenFocus-*-macos-arm64.zip` | 포터블(압축 해제) |

앱은 서명되지 않았으므로 첫 실행 시 경고가 나타날 수 있습니다. Windows는 “추가 정보 → 실행”, macOS는 오른쪽 클릭 → 열기 또는 `xattr -cr /Applications/OpenFocus.app`을 사용하세요.

## 주요 기능

<table>

<tr>
<td width="46%" valign="middle">

### 8가지 융합 알고리즘

4가지 고전 알고리즘에 OpenFocus A/B/C 제품군과 StackMFF-V3 신경망 모델까지, 모두 같은 정합 결과 위에서 실행됩니다. 피사체별로 비교해 고르세요.

[알고리즘 →](#algorithms)

</td>
<td width="54%">

<img src="assets/readme/result.png" alt="OpenFocus 창의 융합 결과" width="100%" />

</td>
</tr>

<tr>
<td width="46%" valign="middle">

### A/B 와이프 비교

분할선을 소스 프레임과 융합 결과 위로 끌어 융합이 무엇을 바꿨는지 확인하세요. 정합 어긋남, 가장자리 번짐, 사라진 디테일이 어떤 배율에서도 한눈에 보입니다.

[사용 설명서 →](./docs/USER_MANUAL_KO.md)

</td>
<td width="54%">

<img src="assets/readme/wipe.png" alt="소스 프레임과 결과의 A/B 비교" width="100%" />

</td>
</tr>

<tr>
<td width="46%" valign="middle">

### 방법 비교

모든 방법을 한 번에 렌더링하고 같은 크롭에서 나란히 확인한 뒤, 가장 좋은 결과를 새 출력으로 채택하거나 파라미터를 메인 창으로 복사하세요.

</td>
<td width="54%">

<img src="assets/readme/compare.png" alt="8가지 융합 방법 나란히 비교" width="100%" />

</td>
</tr>

<tr>
<td width="46%" valign="middle">

### 깊이 맵과 3D

다섯 가지 초점 측정으로 픽셀별 초점 위치를 추정하고, 연구용 컬러맵, TIFF 메타데이터의 물리 Z 보정, 텍스처 OBJ 표면 내보내기를 지원합니다.

</td>
<td width="54%">

<img src="assets/readme/depth.png" alt="같은 스택의 깊이 맵과 3D 표면" width="100%" />

</td>
</tr>

<tr>
<td width="46%" valign="middle">

### 소스 고정 리터치

결과를 실제로 만든 프레임에서 그립니다. 공유 줌/패닝, 소스 프레임 슬라이더, 부드러운 브러시, 복원 브러시, 획 단위 실행 취소, 미내보내기 픽셀 보호.

</td>
<td width="54%">

<img src="assets/readme/retouch.png" alt="소스 프레임과 결과를 나란히 둔 리터치 작업 공간" width="100%" />

</td>
</tr>

<tr>
<td width="46%" valign="middle">

### 배치 처리

스택이 담긴 폴더를 지정하고 방법과 대상을 고른 뒤 그냥 두세요. 정렬만 하면 스택마다 폴더 하나, 융합은 융합 이미지만 기록하고 표시합니다.

</td>
<td width="54%">

<img src="assets/readme/batch.png" alt="배치 처리 대화상자" width="100%" />

</td>
</tr>

<tr>
<td width="46%" valign="middle">

### 출판용 스케일 바

TIFF 메타데이터 또는 직접 입력으로 보정하고 실제 글꼴과 실제 길이로 그려 모든 내보내기에 새깁니다. 단일 이미지, 스택, TIFF, GIF, 배치 출력 모두 포함됩니다.

</td>
<td width="54%">

<img src="assets/readme/scalebar.png" alt="실시간 미리보기가 있는 스케일 바 대화상자" width="100%" />

</td>
</tr>

</table>

**그 밖에도:**

- **포커스 스태킹 워크플로** — 스택 탐색, 선명도 곡선, 빠른 미리보기, ROI 렌더, 재시작 후에도 유지되는 방법별 설정.
- **현미경 대응** — 16비트 전 구간 유지, TIFF µm/px, 플랫 필드 보정, 다중 페이지 TIFF Z 스택.
- **헤드리스 CLI** — GUI와 같은 파이프라인을 `openfocus`로 제공. 배치, `--dry-run`, `--resume`, JSON 출력 지원.
- **프로젝트와 복구** — `.ofproj` 세션, 충돌 자동 복구, 스택 편집과 주석의 실행 취소/다시 실행.
- **12개 UI 언어** — English, 简体中文, 日本語, Español, Deutsch, Français, Русский, 한국어, Português, Italiano, Nederlands, Polski. 시스템 로캘에서 자동 감지하며 새 언어는 `locales_<code>.py` 파일 하나로 추가됩니다.
- **메모리 절약** — 신경망 가중치는 AI 방법을 쓸 때만 로드되어, 고전 방법 세션은 260MB가 아닌 약 110MB로 대기합니다.

## 빠른 시작

```bash
# 모든 알고리즘, AI 모델, HEIC/카메라 RAW를 포함한 전체 CLI
pip install openfocus

# 이미지 폴더 합성
openfocus -i ./stack_folder -o ./result/fused.png

# DTCWT + ECC, 8스레드
openfocus -i ./stack_folder -o ./result/fused.png -m dtcwt -a ecc -t 8
```

소스에서 GUI 실행:

```bash
conda create -n openfocus python=3.10
conda activate openfocus
pip install -r requirements.txt
python main.py
```

## 문서

- [사용자 설명서(한국어)](./docs/USER_MANUAL_KO.md)
- [변경 내역](./CHANGELOG.md)
- [성능 및 대형 스택](./docs/PERFORMANCE.md)
- [빌드 명령](./docs/BUILD_COMMANDS.md)
- [알고리즘 호환성](./docs/ALGORITHM_PARITY.md)

전체 영어 문서는 [README.md](./README.md)를 참고하세요.

## 인용 및 라이선스

OpenFocus가 연구나 작업에 도움이 되었다면 [CITATION.cff](./CITATION.cff)의 정보로 인용해 주세요.

이 프로젝트는 [MIT 라이선스](./LICENSE)로 배포됩니다. 각 합성/정렬 알고리즘에는 해당 원본 라이선스도 함께 적용됩니다.
