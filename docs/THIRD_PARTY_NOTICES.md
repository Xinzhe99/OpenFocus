# Third-party notices

OpenFocus ships code and model weights from other projects. This file records
the origin, the license, and the changes made here.

## PyOpenCL — OpenCL GPU access

- **Upstream**: https://github.com/inducer/pyopencl
- **Version**: 2025.2.7 or later
- **License**: MIT. Used without modification for OpenCL device access. Frozen
  builds retain the distribution metadata, including the original license and
  any bundled OpenCL loader notices.

## Numba and llvmlite — CPU depth-filter compilation

- **Upstream**: https://github.com/numba/numba and
  https://github.com/numba/llvmlite
- **Versions**: Numba 0.61.2 or later with its compatible llvmlite runtime.
- **Copyright**: Numba (c) 2012, Anaconda, Inc.; llvmlite (c) 2014, Anaconda Inc.
- **License**: BSD 2-Clause; used without modification to compile the Q12
  integer recurrence on the CPU. The frozen build retains the complete original
  `numba-*.dist-info` and `llvmlite-*.dist-info` license files, including their
  third-party and LLVM notices.

## threadpoolctl — native thread limits

- **Upstream**: https://github.com/joblib/threadpoolctl
- **Version**: 3.5 or later
- **License**: BSD 3-Clause. Used without modification to bound codec, OpenMP
  and BLAS threads during batch conversion and thumbnails.

```text
Copyright (c) 2019, threadpoolctl contributors

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of copyright holder nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## imagecodecs — TIFF LZW decoding

- **Upstream**: https://github.com/cgohlke/imagecodecs
- **Version**: 2024.9.22 or later
- **License**: BSD 3-Clause, Copyright (c) Christoph Gohlke. Used without
  modification for lossless TIFF LZW decoding. Frozen builds retain the
  distribution metadata and its original license and third-party notices.

## psdtags and tifffile — layered TIFF export

- **Upstream**: https://github.com/cgohlke/psdtags and
  https://github.com/cgohlke/tifffile
- **Versions**: psdtags 2024.5.24 and tifffile 2024.8.30
- **License**: BSD 3-Clause. Copyright (c) 2022-2024 Christoph Gohlke
  (psdtags); Copyright (c) 2008-2024 Christoph Gohlke (tifffile).

```
BSD 3-Clause License

Copyright (c) 2022-2024, Christoph Gohlke (psdtags)
Copyright (c) 2008-2024, Christoph Gohlke (tifffile)
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
```

## faststack — EXIF focus ordering

- **Upstream**: https://github.com/AlanRockefeller/faststack
- **License**: MIT, Copyright © 2025 Alan Rockefeller.
- **Adapted code**: `core/focus_order.py`, which reads OM SYSTEM focus-step
  maker notes and orders folder inputs near to far. The maker-note IFD
  parsing, direction detection and ExifTool fallback follow
  `faststack/io/focus_order.py`.
- **Changes**: rewritten for this package (plain path strings, a 1 MiB
  bounded header read, OpenFocus logging, optional ExifTool lookup with a
  60 s timeout); the public entry point returns paths plus an evidence
  label.

```
MIT License

Copyright © 2025 Alan Rockefeller

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the “Software”), to deal in the Software without
restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.
```

## StackMFF V3 — neural fusion model

- **Upstream**: https://github.com/Xinzhe99/StackMFF-V3
- **License of the bundled snapshot**: MIT, Copyright (c) 2026 Xinzhe Xie.
  Upstream releases through [v2.0](https://github.com/Xinzhe99/StackMFF-V3/blob/v2.0/LICENSE)
  used MIT; the [current upstream license](https://github.com/Xinzhe99/StackMFF-V3/blob/main/LICENSE)
  is a separate source-available license.
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

## Retrained StackMFF V1 and V2 — neural stack fusion

- **Original methods**: [StackMFF](https://github.com/Xinzhe99/StackMFF) and
  [StackMFF V2](https://github.com/Xinzhe99/StackMFF-V2), originally MIT licensed.
- **Retrained variants**: `core/models/stackmff_v1/network.py` and
  `core/models/stackmff_v2/network.py` adapt the V3-backbone retraining releases.
  They reuse the MIT-licensed V3 snapshot described above, but the retrained
  wrappers and checkpoints are derivatives distributed under the StackMFF V3
  License 1.0 below (Copyright (c) 2026 Xinzhe Xie), unless the licensor grants
  separate written terms. This license restricts distribution, incorporation
  into other projects and commercial use.
- **V1 weights**: `weights/stackmff_final.pth` (658,906 bytes, SHA-256
  `8a530228077a4dc0476258f55dd6e00d0b00d04861b1177bdbe4767eb449fb7d`).
- **V2 weights**: `weights/stackmff_v2_final.pth` (659,194 bytes, SHA-256
  `f7f2ee36453d6073d62b89289133012b95116c07b208e76f5ba6d00a1f3f2cda`).
- **Integration**: both backbones pad to a multiple of 64 and OpenFocus preserves
  source bit depth. V1 runs its published two-scale, smoothed inference at the
  original input size, blends luma, and reconstructs chroma from the closest
  source. V2 pads by reflection (or edge replication for very small inputs)
  before normalizing expected depth and sampling the source; its depth indices
  depend on frame order.
- **ONNX identity**: the V1 graph SHA-256 is
  `f7ba388afcefb988f38707b349d5609b4a56770d81c8405d881e50f605b18a19`;
  the V3 graph SHA-256 is
  `7b92bf31c984eac2b93d6000158b4802b7d62c23140d52e28a47676f58d5c4f7`.
  They contain distinct learned initializers. V1's checkpoint records a 0.7
  weight soup of the released V3 model and its retraining checkpoint; sharing
  the backbone can produce visually similar results on some source stacks.
  No image-independent difference between final outputs is guaranteed.
- **Development diagnosis**: `python tools/diagnose_stackmff.py --torch`
  compares the shipped ONNX graphs with their original PyTorch checkpoints
  using a two-layer 64×64 tensor, and exercises whole/tiled public fusion on
  two 64×80 colour frames. `--gpu` requests the runtime's verified accelerator
  for those consumers, repeats V1/V3 in one process, and reports actual device,
  output hashes and differences. `--weights-dir` can inspect a packaged build.
  PyTorch remains optional and is not used by the product runtime.
- **Citations**: Xie et al., *Applied Intelligence* 55:503 (2025),
  https://doi.org/10.1007/s10489-025-06383-8; Xie et al., *Engineering
  Applications of Artificial Intelligence* 162 (2025) 112667,
  https://doi.org/10.1016/j.engappai.2025.112667.

The MIT text below covers the vendored V3 snapshot and its released V3 weights;
the retrained V1/V2 derivatives have the separate terms that follow it.

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

## StackMFF V3 License 1.0 (retrained V1/V2 derivatives)

In the original text below, “this repository” refers to the StackMFF V3
research/retraining release, not to all files in the OpenFocus repository.
The terms apply here to the retrained V1/V2 derivatives identified above.

```text
# StackMFF V3 License 1.0

SPDX-License-Identifier: LicenseRef-StackMFF-V3-1.0

Copyright (c) 2026 Xinzhe Xie. All rights reserved.

This is a source-available license for research and personal use. It is **not** an open-source license: it does not allow the software to be included in other software projects, or redistributed, without written permission. By using, copying or modifying the Software you accept these terms.

## 1. Definitions

**Software** means the source code, scripts, documentation and model checkpoint distributed in this repository, and any part of them.

**Derivative** means a modified version of the Software, or any work that includes the Software or a modified version of it.

**Distribute** means to make the Software or a Derivative available to anyone other than yourself, by any means, including a public repository, package index, container image, installer, plugin, application bundle, or hosted service.

**Project** means any software program, library, application, tool, pipeline, product or service.

**You** means the individual or legal entity agreeing to these terms.

## 2. Permission

Subject to sections 3 to 6, you may use, copy and modify the Software, and run the released checkpoint, for research, teaching, evaluation, and personal use. Modifications are permitted only for your own use and must not be Distributed.

## 3. Conditions

Without the licensor's prior written permission, you must not:

1. use the Software, a Derivative, or the checkpoint as part of, as a dependency of, or in combination with any Project that is Distributed or otherwise made available to anyone outside your organisation — whether that Project is open source, noncommercial, commercial or otherwise, including a public repository, a released application, a plugin or a service;
2. Distribute the Software, the checkpoint, or a Derivative;
3. use the Software, the checkpoint, or a Derivative for any commercial purpose, including in a product or service offered for a fee, or in the business operations of a for-profit organisation;
4. remove or alter any copyright, license or attribution notice in the Software or a Derivative.

Nothing in this license limits rights you have under applicable law, such as fair use.

## 4. Written permission

Send requests in writing to the licensor at guobuyuwork@163.com. Permission is effective only if given in writing by the licensor and only on the terms it states. The licensor may grant permission for use in an open-source project, integration into another application, commercial use, or redistribution, under separate terms.

## 5. Attribution and citation

Any publication that reports results obtained with the Software must cite the accompanying paper. You may not use the names "StackMFF", "StackMFF V3", or the authors' names to endorse or promote a Project without written permission.

## 6. Termination

Your rights under this license end automatically if you breach these terms. Section 7 survives termination.

## 7. Disclaimer

As far as the law allows, the Software comes as is, without any warranty or condition, and the licensor will not be liable to you for any damages arising out of these terms or the use or nature of the Software, under any kind of legal claim.

## 8. Third-party material

This license covers the Software only. Datasets, third-party implementations and image-derived outputs retain the terms of their sources; see `THIRD_PARTY_NOTICES.md`.
```
