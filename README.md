# <img src="assets/OpenFocus.png" alt="OpenFocus Logo" width="120"> OpenFocus

OpenFocus delivers focus stacking quality that rivals commercial-grade software, while staying fully open source and easy to extend.

<p align="left">
  <a href="https://www.python.org/downloads/release/python-3100/"><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+" /></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?logo=open-source-initiative&logoColor=white" alt="License: MIT" /></a>
  <a href="https://github.com/your-org/OpenFocus"><img src="https://img.shields.io/badge/GitHub-Repository-181717?logo=github&logoColor=white" alt="GitHub Repository" /></a>
</p>

## 📢 News

> [!NOTE]
> 🎉 **2026.01.13**: Optimized ROI mode processing and fixed bugs to improve performance and stability.
 
> 🎉 **2026.01.12**: Added drag-and-drop image import on Mac and refactored core modules for improved code maintainability and readability.

> 🎉 **2026.01.10**: Added bilingual support, a status dashboard, and new ROI fusion options to improve efficiency and flexibility.

> 🎉 **2026.01.09**: Improved UI and navigation, faster parallel processing, multi-folder batch support, and bug fixes.

> 🎉 **2025.12.11**: Added functionality to read image stacks in video format.
 
> 🎉 **2025.12.11**: Thanks to Rangj for providing the C++ implementation of the GFG-FGF fusion algorithm, which is now available in the software.

> 🎉 **2025.12.11**: We have fixed some bugs and added configuration options such as block-wise fusion to avoid OOM (Out of Memory) issues.

> 🎉 **2025.12.05**: OpenFocus officially released — welcome to try it.

<a id="environment-setup"></a>
## ⚙️ Environment Setup
```bash
conda create -n openfocus python=3.10
conda activate openfocus
pip install opencv-python pyqt6 numpy imageio dtcwt scipy torch torchvision 
python main.py
```

> **Pre-built package (Windows only):** Grab the compact Windows build from the [Releases](https://github.com/Xinzhe99/OpenFocus/releases) page; other platforms can run from source.
## Table of Contents
- [⚙️ Environment Setup](#environment-setup)
- [🔭 Overview](#overview)
- [✨ Highlights](#highlights)
- [🧪 Fusion & Registration Methods](#fusion--registration-methods)
- [📚 References](#references)
- [🤝 Contribution](#contribution)
- [📄 License](#license)

<a id="overview"></a>
## 🔭 Overview
OpenFocus is a PyQt6-based multi-focus registration and fusion workstation that delivers commercial-grade alignment and blending results. The project is fully open source (MIT License) and runs on CPU by default with optional GPU acceleration for the StackMFF V4 neural model.

<p align="center">
	<img src="assets/ui.jpg" alt="OpenFocus UI" width="720">
</p>

<a id="highlights"></a>
## ✨ Highlights
- **Beginner-Friendly**: Plug-and-play workflows with unapologetically simple, guided operations.
- **Flexible Processing Flows**: Run fusion-only, registration-only, or combined registration + fusion pipelines depending on your workload.
- **Batch Automation**: Kick off batch jobs across multiple folders with live progress, cancellation, and automatic output organization.
- **Annotation & Export Toolkit**: Overlay labels, export GIF animations, and save processed stacks in JPG/PNG/BMP/TIFF with consistent metadata handling.
- **AI-Assisted Fusion**: Ship with StackMFF V4 to unlock deep-learning-quality fusion alongside classic signal-processing methods.

<a id="fusion--registration-methods"></a>
## 🧪 Algorithms
### Fusion Algorithms

- **Guided Filter**: Fast edge-preserving fusion that enhances contrast while suppressing noise.
- **DCT Multi-Focus Fusion**: Frequency-domain technique optimized for crisp detail recovery.
- **Dual-Tree Complex Wavelet Transform (DTCWT)**: Multi-scale representation that preserves fine texture structures.
- **GFG-FGF**: GFG-FGF is based on a generalized four-neighborhood Gaussian gradient (GFG) operator combined with a fast guided filter (FGF). 
- **StackMFF V4**: Pretrained deep model delivering state-of-the-art focus stacking quality.

### Registration Algorithms
- **Homography**: Performs feature-based projective alignment using keypoint matching and RANSAC to handle global perspective transformations.
- **ECC**: Performs intensity-based alignment by maximizing the enhanced correlation coefficient for precise, sub-pixel registration.
  
> **License Notice:** Every fusion/registration algorithm included comes from open-source research implementations. When using or redistributing them, please follow each algorithm’s original license terms in addition to the OpenFocus MIT license.

<a id="references"></a>
## 📚 References
- M. B. A. Haghighat, A. Aghagolzadeh, and H. Seyedarabi, "Multi-focus image fusion for visual sensor networks in DCT domain," *Computers & Electrical Engineering*, vol. 37, no. 5, pp. 789-797, 2011.
- J. J. Lewis, R. J. O'Callaghan, S. G. Nikolov, D. R. Bull, and N. Canagarajah, "Pixel- and region-based image fusion with complex wavelets," *Information Fusion*, vol. 8, no. 2, pp. 119-130, 2007.
- S. Li, X. Kang, and J. Hu, "Image fusion with guided filtering," *IEEE Transactions on Image Processing*, vol. 22, no. 7, pp. 2864-2875, 2013.
- 付宏语, 巩岩, 汪路涵, 等. 多聚焦显微图像融合算法[J]. Laser & Optoelectronics Progress, 2024, 61(6): 0618022-0618022-9.

<a id="contribution"></a>
## 🤝 Contribution
We welcome community contributions of all kinds:
1. **Issues**: Report bugs, request features, or propose UX enhancements.
2. **Algorithm & Performance Work**: Share new fusion/registration ideas, optimizations.

> Bug reports or suggestions? Please open an issue so we can follow up quickly.

<a id="license"></a>
## 📄 License
This project is released under the [MIT License](./LICENSE). Feel free to use, modify, and distribute within the terms of the license.

<p align="center" style="font-size:1.25rem; font-weight:600;">
  If OpenFocus helps you, please consider leaving a ⭐ on the repository!
</p>

















