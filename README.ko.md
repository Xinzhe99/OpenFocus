<p align="center">
  <a href="https://github.com/Xinzhe99/OpenFocus">
    <img src="assets/OpenFocus.png" alt="OpenFocus" width="128">
  </a>
</p>

<h1 align="center">OpenFocus</h1>

<p align="center"><strong>모든 초점면을, 한 장의 완전 선명한 이미지로.</strong></p>

<p align="center">
  10가지 융합 알고리즘, 3가지 정합 모드, 그리고 매크로·현미경·산업용 스택을 위한 소스 고정 리터치 워크스페이스. 16비트 전 구간 유지, 헤드리스 CLI, 12개 언어 UI.
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
  <a href="#install">다운로드</a> ·
  <a href="./docs/USER_MANUAL_KO.md">문서</a> ·
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
> **v1.71.0** — 알고리즘이 하나의 불변 레지스트리로 선언됩니다. 새 융합·정합 방법은 형식이 지정된 매개변수를 가진 설명자 하나로 렌더링, 배치, 비교, 깊이, CLI에 나타납니다. 실행할 때마다 별도의 롤링 로그를 기록하고 작업 ID로 매개변수, 소요 시간, 전체 스택 추적을 연결합니다. 피드백 ZIP에는 크래시 보고서, RAM/GPU 정보, 의존성 버전이 포함됩니다. 진단은 Qt와 수치 라이브러리 임포트 전에 시작하므로 시작 실패도 기록됩니다. 8개 알고리즘은 v1.70.0과 바이트 단위로 동일합니다.

<a id="features"></a>
## 주요 기능

<table>

<tr>
<td width="46%" valign="middle">

### AI 융합: StackMFF V3

기본 방법은 신경망으로 각 픽셀을 가장 가능성 높은 소스 프레임에 분류합니다. 같은 정합 결과에서 글자 가장자리는 Guided Filter보다 선명하고 표면은 더 깨끗합니다(라플라시안 선명도 27.1 대 28.5). 합성 정답 데이터에서 PSNR과 SSIM 모두 모든 방법을 앞서며, 고스트 픽셀도 어떤 고전 알고리즘보다 적습니다.

[StackMFF V3 →](https://github.com/Xinzhe99/StackMFF V3) · [벤치마크 →](tools/synthetic_benchmark.py)

</td>
<td width="54%">

<a href="assets/readme/ai.mp4"><picture><source srcset="assets/readme/ai.gif" type="image/gif"><img src="assets/readme/ai_poster.jpg" alt="AI 융합: StackMFF V3" width="100%" /></picture></a>

</td>
</tr>

<tr>
<td width="46%" valign="middle">

### A/B 와이프 비교

분할선을 소스 프레임과 융합 결과 위로 끌어 융합이 무엇을 바꿨는지 확인하세요. 정합 어긋남, 가장자리 번짐, 사라진 디테일이 어떤 배율에서도 한눈에 보입니다.

[사용 설명서 →](./docs/USER_MANUAL_KO.md)

</td>
<td width="54%">

<a href="assets/readme/wipe.mp4"><picture><source srcset="assets/readme/wipe.gif" type="image/gif"><img src="assets/readme/wipe_poster.jpg" alt="소스 프레임과 결과의 A/B 비교" width="100%" /></picture></a>

</td>
</tr>

<tr>
<td width="46%" valign="middle">

### 방법 비교

모든 방법을 한 번에 렌더링하고 같은 크롭에서 나란히 확인한 뒤, 가장 좋은 결과를 새 출력으로 채택하거나 파라미터를 메인 창으로 복사하세요.

</td>
<td width="54%">

<a href="assets/readme/method-comparison.mp4"><picture><source srcset="assets/readme/method-comparison.gif" type="image/gif"><img src="assets/readme/method-comparison_poster.jpg" alt="8가지 융합 방법 나란히 비교" width="100%" /></picture></a>

</td>
</tr>

<tr>
<td width="46%" valign="middle">

### 깊이 맵과 3D

다섯 가지 초점 측정으로 픽셀별 초점 위치를 추정하고, 연구용 컬러맵, TIFF 메타데이터의 물리 Z 보정, 텍스처 OBJ 표면 내보내기를 지원합니다.

</td>
<td width="54%">

<a href="assets/readme/depth.mp4"><picture><source srcset="assets/readme/depth.gif" type="image/gif"><img src="assets/readme/depth_poster.jpg" alt="같은 스택의 깊이 맵과 3D 표면" width="100%" /></picture></a>

</td>
</tr>

<tr>
<td width="46%" valign="middle">

### 소스 고정 리터치

결과를 실제로 만든 프레임에서 그립니다. 공유 줌/패닝, 소스 프레임 슬라이더, 부드러운 브러시, 복원 브러시, 획 단위 실행 취소, 미내보내기 픽셀 보호.

</td>
<td width="54%">

<a href="assets/readme/retouch.mp4"><picture><source srcset="assets/readme/retouch.gif" type="image/gif"><img src="assets/readme/retouch_poster.jpg" alt="소스 프레임과 결과를 나란히 둔 리터치 작업 공간" width="100%" /></picture></a>

</td>
</tr>

<tr>
<td width="46%" valign="middle">

### 배치 처리

스택이 담긴 폴더를 지정하고 방법과 대상을 고른 뒤 그냥 두세요. 정렬만 하면 스택마다 폴더 하나, 융합은 융합 이미지만 기록하고 표시합니다.

</td>
<td width="54%">

<a href="assets/readme/batch.mp4"><picture><source srcset="assets/readme/batch.gif" type="image/gif"><img src="assets/readme/batch_poster.jpg" alt="배치 처리 대화상자" width="100%" /></picture></a>

</td>
</tr>

<tr>
<td width="46%" valign="middle">

### 출판용 스케일 바

TIFF 메타데이터 또는 직접 입력으로 보정하고 실제 글꼴과 실제 길이로 그려 모든 내보내기에 새깁니다. 단일 이미지, 스택, TIFF, GIF, 배치 출력 모두 포함됩니다. 읽을 수 있게 들어가지 않으면, 라벨과 다른 짧은 막대를 그리는 대신 그 사실을 알려 줍니다.

[사용 설명서 →](./docs/USER_MANUAL_KO.md)

</td>
<td width="54%">

<a href="assets/readme/scalebar.mp4"><picture><source srcset="assets/readme/scalebar.gif" type="image/gif"><img src="assets/readme/scalebar_poster.jpg" alt="실시간 미리보기가 있는 스케일 바 대화상자" width="100%" /></picture></a>

</td>
</tr>

<tr>
<td width="46%" valign="middle">

### 라벨과 캡션

캡션을 관리되는 목록으로 유지합니다. 각 항목은 대상(소스, 정합된 스택 또는 특정 출력), 프레임 범위, 스타일을 따로 가집니다. 미리보기와 모든 내보내기가 같은 대상과 프레임 번호를 쓰므로 `{value}`는 저장, 다른 이름으로 저장, 끌어내기, 모두 내보내기에서 동일하게 읽힙니다.

</td>
<td width="54%">

<a href="assets/readme/labels.mp4"><picture><source srcset="assets/readme/labels.gif" type="image/gif"><img src="assets/readme/labels_poster.jpg" alt="라벨과 캡션" width="100%" /></picture></a>

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

<a id="install"></a>
## ⬇️ 설치

설정 없이 [Releases](https://github.com/Xinzhe99/OpenFocus/releases)에서 빌드를 받으세요:

| 플랫폼 | 파일 | 유형 |
|--------|------|------|
| Windows 10/11(64비트) | `OpenFocus-*-setup.exe` | **설치 프로그램**(권장) |
| Windows 10/11(64비트) | `OpenFocus-*-windows-x64.zip` | 포터블(압축 해제 후 실행) |
| macOS Apple Silicon(M1–M4) | `OpenFocus-*-macos.dmg` | 디스크 이미지(응용 프로그램으로 드래그) |
| macOS Apple Silicon(M1–M4) | `OpenFocus-*-macos-arm64.zip` | 포터블(압축 해제) |

앱이 서명되지 않아 첫 실행 시 경고가 나올 수 있습니다. **Windows**는 SmartScreen → '추가 정보' → '그래도 실행'(첫 실행은 30–60초 걸릴 수 있음), **macOS**는 우클릭 → '열기' 또는 터미널에서 `xattr -cr /Applications/OpenFocus.app`.

### 💻 명령줄

Windows, macOS, Linux에서는 CLI로도 설치할 수 있습니다. 같은 파이프라인을 GUI 없이 실행합니다:

```bash
pip install openfocus
openfocus -i ./stack_folder -o ./result/fused.png
```

AI 모델과 가중치, HEIC 및 카메라 RAW 디코딩이 모두 이 한 번의 설치에 포함됩니다. Python 3.13+에서는 `dtcwt`가 빠지고 DTCWT 항목이 설명과 함께 비활성화됩니다.

<a id="community"></a>
## 커뮤니티 및 지원

- **문서** — [사용 설명서](./docs/USER_MANUAL_KO.md)(12개 언어) · [변경 기록](./CHANGELOG.md) · [성능](./docs/PERFORMANCE.md) · [알고리즘 비교](./docs/ALGORITHM_PARITY.md)
- **인용** — [CITATION.cff](./CITATION.cff); 모든 릴리스는 Zenodo에 보관됩니다([DOI](https://doi.org/10.5281/zenodo.23050823))
- **피드백** — 버그, 아이디어, 알고리즘 개선은 [issue](https://github.com/Xinzhe99/OpenFocus/issues)로
- **크레딧** — 각 융합·정합 알고리즘은 원래 연구 라이선스를 따릅니다. StackMFF V3는 [Xinzhe99/StackMFF V3](https://github.com/Xinzhe99/StackMFF V3)에 있습니다

<a id="developing"></a>
## 개발

저장소에서 실행:

```bash
conda create -n openfocus python=3.10
conda activate openfocus
pip install -r requirements.txt
python main.py
```

PyInstaller 패키징은 [docs/BUILD_COMMANDS.md](./docs/BUILD_COMMANDS.md)를 참고하세요.

<a id="star-history"></a>
## ⭐ 스타 기록

[![Star History Chart](https://api.star-history.com/svg?repos=Xinzhe99/OpenFocus&type=Date)](https://star-history.com/#Xinzhe99/OpenFocus&Date)
