# Third-party notices

OpenFocus ships code and model weights from other projects. This file records
the origin, the license, and the changes made here.

## StackMFF V3 — neural fusion model

- **Upstream**: https://github.com/Xinzhe99/StackMFF-V3
- **License**: MIT, Copyright (c) 2026 Xinzhe Xie
- **Vendored code**: `core/models/stackmff_v3/network.py` (the network
  architecture: `StackMFFLoop`, `StackMFFV3` and their building blocks) and
  `core/models/stackmff_v3/model_io.py` (checkpoint loading and the released
  architecture arguments). The upstream `Dataloader.py` is not included —
  OpenFocus fuses at inference only.
- **Changes**: the module imports were rewritten for this package layout
  (`stackmff_v3.network` → `core.models.stackmff_v3.network`), the default
  weights path points at OpenFocus's `weights/` directory, and the docstring of
  the package was rewritten. The network math is unchanged.
- **Weights**: `weights/stackmff_v3_final.pth` is the released checkpoint from
  the upstream v1.0 release (658,587 bytes, SHA-256
  `39b2ea3738e8fea1676c379238d6be81a87988fc9952dbd6ac3825d5d7249f46`).
- **Integration**: `fusion_methods/stackmffv3.py` is OpenFocus code (not
  upstream); it adapts the upstream `fuse.py` to this project's tiling, bit
  depth and output-mode conventions.
- **Citation**: see the upstream README for the StackMFF V3 paper reference.

## StackMFF (V1) — neural stack fusion

- **Upstream**: https://github.com/Xinzhe99/StackMFF
- **License**: MIT, as declared in the upstream README
- **Vendored code**: `core/models/stackmff_v1/network.py` — the 3D U-Net
  (`StackMFF`, `Conv3D_Block`, `Deconv3D_Block`, `CoordAtt_3d`). The upstream
  training and prediction scripts are not included; OpenFocus fuses at
  inference only.
- **Changes**: the `__main__` smoke block and its prints were removed, and the
  module docstring was rewritten. The network math is unchanged.
- **Weights**: `weights/stackmff_final.pth` is the upstream released
  `weights/checkpoint.pth` (24,405,662 bytes, SHA-256
  `5445a79657c87d05a6b4cafd79d6a893b3f0c2713222a9ef8e1daf6e2e35ff80`).
- **Integration**: `fusion_methods/stackmff_v1.py` is OpenFocus code; it pads
  to a multiple of 16, reconstructs colour from the closest source frame and
  restores the source bit depth.
- **Citation**: Xie et al., *Applied Intelligence* 55:503 (2025),
  https://doi.org/10.1007/s10489-025-06383-8

## StackMFF V2 — neural stack fusion with focal-depth regression

- **Upstream**: https://github.com/Xinzhe99/StackMFF-V2
- **License**: MIT, as declared in the upstream README
- **Vendored code**: `core/models/stackmff_v2/network.py` — `StackMFF_V2`,
  `FeatureExtraction`, `LayerInteraction`, `DepthMapCreation` and their
  building blocks. Training, evaluation and CLI scripts are not included.
- **Changes**: the `__main__` block and the debug print in the bridge
  constructor were removed, and the module docstring was rewritten.
- **Weights**: `weights/stackmff_v2_final.pth` is the upstream released
  `model.pth` (254,009 bytes, SHA-256
  `cb923ed0265765e8471c7fcb15a08d1cee296809254e03e97418c664d655d179`).
- **Integration**: `fusion_methods/stackmff_v2.py` is OpenFocus code; it pads
  to a multiple of 32 and exposes the reference hard sampling plus an optional
  probability-weighted (soft) blend. The model is order-sensitive by design:
  the recurrent layer reads the stack in focus order.
- **Citation**: Xie et al., *Engineering Applications of Artificial
  Intelligence* 162 (2025) 112667,
  https://doi.org/10.1016/j.engappai.2025.112667

The MIT text below covers both the vendored code and the released weights.

```
MIT License

Copyright (c) 2026 Xinzhe Xie

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
