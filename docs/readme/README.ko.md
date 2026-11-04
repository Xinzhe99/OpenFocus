<p align="center">
  <a href="https://github.com/Xinzhe99/OpenFocus">
    <img src="../../assets/OpenFocus.png" alt="OpenFocus" width="128">
  </a>
</p>

<h1 align="center">OpenFocus</h1>

<p align="center"><strong>모든 초점면을 하나의 올인포커스 이미지로.</strong></p>

<p align="center">
  열 가지 합성 알고리즘, 다섯 가지 정합 옵션(정합 안 함 포함), 그리고 매크로·현미경·산업용 스택을 위한 소스 고정 리터치 작업 공간. 16비트를 끝까지 유지하고, 헤드리스 CLI와 12개 언어 UI를 제공합니다.
</p>

<p align="center">
  <a href="https://github.com/Xinzhe99/OpenFocus/releases"><img src="https://img.shields.io/github/v/release/Xinzhe99/OpenFocus?label=release&color=2f81f7" alt="Release"></a>
  <a href="https://github.com/Xinzhe99/OpenFocus/stargazers"><img src="https://img.shields.io/github/stars/Xinzhe99/OpenFocus?color=e3b341" alt="Stars"></a>
  <a href="../../LICENSE"><img src="https://img.shields.io/badge/license-MIT-3fb950" alt="License: MIT"></a>
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
  <a href="../USER_MANUAL_KO.md">문서</a> ·
  <a href="../../CHANGELOG.md">변경 기록</a> ·
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
  <strong>한국어</strong> ·
  <a href="README.pt.md">Português</a> ·
  <a href="README.it.md">Italiano</a> ·
  <a href="README.nl.md">Nederlands</a> ·
  <a href="README.pl.md">Polski</a>
</p>

<p align="center">
  <img src="../../assets/ui.png" alt="OpenFocus 인터페이스" width="880">
</p>

## 📢 소식

> [!NOTE]
> **v1.75.0** — 그룹 합성, 재사용 가능한 처리 워크플로, 투명 PNG 스택, 정합 진단, 강화된 깊이 내보내기, 새로워진 데스크톱 워크플로를 제공합니다. 지원 입력과 제한은 변경 기록을 참고하세요.

이전 릴리스는 [변경 기록](../../CHANGELOG.md)에서 확인할 수 있습니다.

📸 OpenFocus로 만든 이미지를 공개한다면 다음 문구를 덧붙여 주세요.

> Created with OpenFocus — https://github.com/Xinzhe99/OpenFocus

필수는 아니지만 큰 힘이 됩니다.

<a id="features"></a>
## 기능

<table>
<tr>
<td width="46%" valign="middle">

### AI 합성: StackMFF V3

StackMFF V3는 AI로 각 프레임의 선명한 부분을 합쳐 초점이 맞는 이미지를 만듭니다.

[StackMFF V3 →](https://github.com/Xinzhe99/StackMFF-V3) · [벤치마크 →](../../tools/synthetic_benchmark.py)

</td>
<td width="54%">

<a href="../../assets/readme/ai.mp4"><picture><source srcset="../../assets/readme/ai.gif" type="image/gif"><img src="../../assets/readme/ai_poster.jpg" alt="같은 스택에서 StackMFF V3와 가이드 필터" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### A/B 와이프 비교

분할선을 움직여 소스 프레임과 합성 결과를 직접 비교합니다.

[사용자 설명서 →](../USER_MANUAL_KO.md)

</td>
<td width="54%">

<a href="../../assets/readme/wipe.mp4"><picture><source srcset="../../assets/readme/wipe.gif" type="image/gif"><img src="../../assets/readme/wipe_poster.jpg" alt="소스 프레임과 합성 결과의 A/B 비교" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### 방식 비교

합성 방식을 나란히 비교하고 선택한 결과를 출력으로 채택하거나 설정을 메인 창에 적용합니다.

</td>
<td width="54%">

<a href="../../assets/readme/method-comparison.mp4"><picture><source srcset="../../assets/readme/method-comparison.gif" type="image/gif"><img src="../../assets/readme/method-comparison_poster.jpg" alt="여덟 가지 합성 방식의 나란한 비교" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### 깊이 맵과 3D

깊이 맵을 생성하고 수동 Z 보정과 텍스처가 있는 3D OBJ 내보내기를 지원합니다.

</td>
<td width="54%">

<a href="../../assets/readme/depth.mp4"><picture><source srcset="../../assets/readme/depth.gif" type="image/gif"><img src="../../assets/readme/depth_poster.jpg" alt="같은 스택의 깊이 맵과 3D 표면" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### 소스 고정 리터치

결과를 만든 소스 프레임에서 브러시로 리터치하고 소스가 포함된 레이어 TIFF를 내보낼 수 있습니다.

</td>
<td width="54%">

<a href="../../assets/readme/retouch.mp4"><picture><source srcset="../../assets/readme/retouch.gif" type="image/gif"><img src="../../assets/readme/retouch_poster.jpg" alt="소스 프레임과 결과를 나란히 둔 리터치 작업 공간" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### 배치 처리

여러 스택을 일괄 처리하고 선택적으로 그룹 합성을 사용해 메모리 사용량을 줄입니다.

</td>
<td width="54%">

<a href="../../assets/readme/batch.mp4"><picture><source srcset="../../assets/readme/batch.gif" type="image/gif"><img src="../../assets/readme/batch_poster.jpg" alt="배치 처리 대화상자" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### 출판용 스케일 바

TIFF 메타데이터 또는 직접 입력으로 보정한 스케일 바를 이미지에 내보냅니다.

[사용자 설명서 →](../USER_MANUAL_KO.md)

</td>
<td width="54%">

<a href="../../assets/readme/scalebar.mp4"><picture><source srcset="../../assets/readme/scalebar.gif" type="image/gif"><img src="../../assets/readme/scalebar_poster.jpg" alt="실시간 미리보기가 있는 스케일 바 대화상자" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### 레이블과 캡션

소스, 정합된 스택 또는 출력에 프레임 범위와 스타일을 지정한 레이블과 캡션을 추가합니다.

</td>
<td width="54%">

<a href="../../assets/readme/labels.mp4"><picture><source srcset="../../assets/readme/labels.gif" type="image/gif"><img src="../../assets/readme/labels_poster.jpg" alt="레이블과 캡션" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Agent로 제어 가능

토큰으로 보호된 로컬 채널을 통해 Agent가 창을 제어하고 이미지를 처리하며 결과를 내보냅니다.

[프로토콜 →](../../docs/AGENT_CONTROL_SPEC.md) · [스킬 →](../../skills/openfocus-control/SKILL.md)

</td>
<td width="54%">

<a href="../../assets/readme/agent.mp4"><picture><source srcset="../../assets/readme/agent.gif" type="image/gif"><img src="../../assets/readme/agent_poster.jpg" alt="Agent가 실제 GUI를 조작: 불러오기, AI 렌더링, 비교, 리터치, 와이프, 깊이, 레이블" width="100%" /></picture></a>

</td>
</tr>
</table>

**그 밖에:**

- **초점 스태킹 워크플로**——스택 탐색, 선명도 곡선, 빠른 미리보기, ROI 렌더링, 방식별 설정을 재시작 후에도 유지.
- **현미경 대응**——16비트를 끝까지 유지, TIFF µm/px, 플랫 필드 보정, 센서 결함 보정, 다중 페이지 TIFF Z 스택, 텍스처가 있는 OBJ 내보내기.
- **색상 프로파일**——단일 PNG, TIFF, JPEG, WebP 내보내기에 모든 소스가 공유하는 RGB ICC 프로파일을 유지하며, 혼합되거나 태그가 없으면 프로파일을 붙이거나 색을 변환하지 않습니다.
- **투명 PNG**——같은 사각형의 일정 불투명도 마스크를 가진 스택을 원래 크기로 합성해 저장합니다(제한은 사용자 설명서 참고).
- **헤드리스 CLI**——GUI와 같은 파이프라인을 `openfocus` 명령으로 제공하며 배치, `--dry-run`, `--resume`, JSON 출력을 지원합니다.
- **프로젝트와 복구**——`.ofproj` 세션, 충돌 자동 복구, 스택 편집과 주석의 실행 취소/다시 실행.
- **12개 UI 언어**——시스템 로캘에서 자동 감지하며 `locales/<code>.py`로 언어를 추가할 수 있습니다.
- **메모리 절약**——AI 방식을 사용할 때만 신경망 가중치를 불러옵니다.

<a id="install"></a>
## ⬇️ 설치

최신 Windows 또는 macOS 빌드는 [OpenFocus 공식 사이트](https://openfocus.top/#download)에서 내려받으세요. 같은 파일과 이전 버전은 [GitHub Releases](https://github.com/Xinzhe99/OpenFocus/releases)에도 있습니다.

설치와 문제 해결은 [사용자 설명서](../USER_MANUAL_KO.md)를 참고하세요.

터미널에서도 설치할 수 있습니다. Linux에는 헤드리스 CLI가 설치되며 macOS에는 Apple Silicon이 필요합니다. DMG는 `/Applications`에, 터미널 설치는 `~/Applications`에 설치됩니다.

```bash
curl -fsSL https://openfocus.top/install.sh | sh -s -- --dry-run
curl -fsSL https://openfocus.top/install.sh | sh
```

Windows에서는 PowerShell에서 설치 프로그램을 실행합니다.

```powershell
irm https://openfocus.top/install.ps1 | iex
```

Windows 설치 파일을 직접 받을 수도 있습니다.

```powershell
curl.exe -fL https://openfocus.top/download/latest/windows-setup -o OpenFocus-setup.exe
```

### 🤖 Agent 통합

데스크톱 앱은 토큰으로 보호된 로컬 Agent 채널을 자동으로 시작합니다. `openfocus ctl`로 실행 중인 창에 형식화된 명령이나 일반 UI 작업을 보낼 수 있습니다. Agent는 스택 불러오기, 렌더링, 내보내기, 대화상자 열기와 입력, 파일 선택 응답, 마우스·키보드 이벤트 전송, 스크린샷 촬영을 수행할 수 있습니다.

- 스킬: [`skills/openfocus-control/SKILL.md`](../../skills/openfocus-control/SKILL.md)
- 프로토콜과 명령 참조: [`docs/AGENT_CONTROL_SPEC.md`](../../docs/AGENT_CONTROL_SPEC.md)
- 종단 간 실행기: `python tools/agent_gui_smoke.py --launch`
- 전체 메뉴 동작 스윕: `python tools/agent_action_sweep.py --launch`

### 💻 명령줄

Windows, macOS, Linux에서 CLI를 설치할 수도 있습니다. 같은 파이프라인을 GUI 없이 실행합니다.

```bash
pip install openfocus
openfocus -i ./stack_folder -o ./result/fused.png
```

자세한 내용은 [CLI 설명서](../USER_MANUAL_KO.md#14-명령줄-인터페이스)를 참고하세요.

<a id="community"></a>
## 커뮤니티와 지원

질문이나 아이디어, 의견이 있으신가요? [Discord 커뮤니티](https://discord.gg/afzmwgPt)에 참여해 주세요. 가능한 한 빨리 답변드리지만, 베이징 시간(UTC+8)을 사용하므로 시차에 따라 답변이 늦어질 수 있습니다.

- **문서** — [사용자 설명서](../USER_MANUAL_KO.md)(12개 언어) · [변경 기록](../../CHANGELOG.md) · [성능](../../docs/PERFORMANCE.md) · [합성 방식](../../docs/FUSION_METHODS.md)
- **공식 사이트** — [openfocus.top](https://openfocus.top/)(다운로드, 설치 스크립트, 미러)
- **인용** — [CITATION.cff](../../CITATION.cff). 모든 릴리스는 Zenodo에 보관됩니다([DOI](https://doi.org/10.5281/zenodo.23050823))
- **피드백** — 버그, 아이디어, 알고리즘 개선은 [issue](https://github.com/Xinzhe99/OpenFocus/issues)로 알려 주세요
- **크레딧** — 각 합성·정합 알고리즘은 원래 연구 라이선스를 따릅니다. StackMFF V3는 [Xinzhe99/StackMFF-V3](https://github.com/Xinzhe99/StackMFF-V3)에 있습니다

<a id="license"></a>
## 라이선스

OpenFocus 애플리케이션은 [MIT 라이선스](../../LICENSE)로 배포됩니다.

**StackMFF 시리즈 예외.** StackMFF 합성 알고리즘(현재 StackMFF V3와 향후 같은 시리즈 버전 포함)은 OpenFocus 프로젝트에 귀속되며, 프로젝트가 StackMFF 시리즈의 모든 권리를 보유합니다. OpenFocus의 일부로 자신의 이미지와 내부 작업에 사용할 수 있지만, 다른 오픈 소스 프로젝트로 복사, 이식, 이동하려면 사전 서면 허가가 필요합니다.

<a id="developing"></a>
## 개발

체크아웃에서 실행:

```bash
conda create -n openfocus python=3.10
conda activate openfocus
pip install -r requirements.txt
python main.py
```

PyInstaller 패키징은 [docs/BUILD_COMMANDS.md](../../docs/BUILD_COMMANDS.md)를 참고하세요.

<a id="star-history"></a>
## ⭐ Star 기록

[![Star History Chart](https://api.star-history.com/svg?repos=Xinzhe99/OpenFocus&type=Date)](https://star-history.com/#Xinzhe99/OpenFocus&Date)
