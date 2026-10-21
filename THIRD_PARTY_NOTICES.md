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
