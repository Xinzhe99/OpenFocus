# GPU support

OpenFocus B can use OpenCL for depth resampling, filtering, activity and selection, guided smoothing and ordered source merging. Full GPU depth processing requires sufficient host and verified device memory. Otherwise bounded GPU activity and merging can still run. Standalone depth export can also use full GPU depth when admitted, or bounded GPU activity. B's other processing remains on the CPU. OpenFocus A, OpenFocus C, Guided Filter, GFG-FGF, DCT and DTCWT can use an OpenCL GPU for full-image fusion when GPU acceleration is enabled. OpenFocus C, GFG-FGF, DCT and DTCWT export depth maps on the CPU. OpenCL acceleration requires a GPU with a working driver and compiler. Unavailable or insufficient resources and device failures fall back to the CPU.

OpenFocus A, Guided Filter, GFG-FGF, DCT and DTCWT stream one source frame at a time to the GPU. Their host memory checks cover decoding, deferred alignment, upload and output readback under the render process's RAM limit. During upload they can reuse existing cached frames while keeping newly decoded frames only for the current transfer.

For OpenFocus B full GPU depth and OpenFocus A/C, Guided Filter, GFG-FGF, DCT and DTCWT GPU fusion, unified-memory devices use the host budget; discrete GPUs require a readable free-memory report. AMD devices use the OpenCL free-memory extension and NVIDIA devices use a uniquely matched `nvidia-smi` report. If discrete free memory is unknown, A, C, Guided Filter, GFG-FGF, DCT and DTCWT fall back to CPU and B skips full GPU depth; B's bounded GPU stages can still be used when admitted.

DTCWT retains the upstream wavelet transform on the CPU. Its OpenCL path uses
single-precision arithmetic, so GPU results can vary slightly by driver.
Unsupported floating-point ranges and non-finite samples use the CPU path.
Cancelling a GPU render waits for queued device work and ends the render.

Retouch’s **Auto background** and the CLI `--auto-background` can use OpenCL to select the peak, first sharpest frame, and median focus energy for every pixel in a large stack. The GPU processes bands that fit the device memory budget. The log records the device after the reduction completes. Small stacks, unavailable devices, and failed GPU operations use the CPU path. Turning off GPU acceleration in the desktop or passing `--cpu` on the CLI selects CPU processing for the entire replacement.

Windows, macOS and Linux desktop builds include PyOpenCL. The separate Linux CLI wheel declares it as a dependency. The PyOpenCL wheels include an ICD loader on Linux, but a compatible GPU driver and its OpenCL ICD must be installed separately (normally registered in `/etc/OpenCL/vendors`). Windows needs the GPU vendor's OpenCL driver. macOS supplies OpenCL through the system, but Apple has deprecated OpenCL and device and compiler availability depend on the macOS version and hardware. The presence of PyOpenCL or an OpenCL loader alone does not establish GPU support; platforms without a compatible PyOpenCL wheel may need a source build and OpenCL development libraries. See [PyOpenCL installation](https://documen.tician.de/pyopencl/misc.html).

ECC uses compatible OpenCL GPUs by default. On the supported Windows x64
OpenCV 5.0.0 build, large 8-bit and 16-bit images can use GPU `INTER_AREA`
for whole-block and fractional reductions up to 8× on each axis. OpenCV
handles the special 2×2 reduction. ECC estimates the transforms on the CPU.
With a compatible GPU and sufficient memory, large frames use the exact
Lanczos4 OpenCL warp: 8-bit and 16-bit images during ECC and homography
registration, and in any render with GPU acceleration when it resolves
deferred frames of any fusion method, as whole images or tiles, including
direct-mapped RGB TIFF sources. The warp reproduces OpenCV Lanczos4 pixel
for pixel with a device workspace under 64 MB, and the device proves this
on a small sample before it is used. Unsupported frames, unavailable
devices and failed GPU operations use the CPU. Turn off **GPU
Acceleration** in the desktop or pass `--cpu` to use the CPU for both
stages. The log events `registration_ecc_resize`, `registration_warp` and
`deferred_ecc_warp` record the work completed on the accelerator.

StackMFF V1, V2 and V3 ship with dynamic ONNX models. **Settings → AI acceleration
components** checks the hardware and offers a compatible optional pack once
per pack version. Windows Setup can also open this screen.

| Desktop platform | Optional native engines | Requirements | Download | Free disk |
| --- | --- | --- | ---: | ---: |
| Windows x64 | TensorRT 10.4, PyTorch 2.5.1 CUDA | NVIDIA CUDA 12.4 driver support and compute capability 7.5, 8.0, 8.6, 8.9 or 9.0 | 3.65 GB | 8 GiB |
| Linux x86-64 | PyTorch 2.6 CUDA | glibc 2.17+, NVIDIA CUDA 12.4 driver support and a supported Maxwell through Hopper GPU | 3.13 GB | 8 GiB |
| macOS arm64 | PyTorch 2.6 MPS | Apple Silicon, macOS 13 or later | 116 MB | 1 GiB |

OpenFocus tries an installed native engine, a compatible ONNX Runtime
accelerator and CPU in that order. Windows includes DirectML, and macOS uses
CoreML when available. The Linux desktop includes ONNX Runtime CPU. Source
and CLI environments can also use compatible CUDA or ROCm ONNX Runtime
distributions. The NVIDIA packs use CUDA device 0 and supply their own CUDA
libraries. For Linux, NVIDIA lists driver 550.54.15 for CUDA 12.4 Update 1.
[NVIDIA CUDA release notes](https://docs.nvidia.com/cuda/archive/12.4.1/cuda-toolkit-release-notes/index.html)

Compatible packs are reused across app updates. Windows Setup keeps the pack
in `%LOCALAPPDATA%\OpenFocus\inference-packs`. Portable installations keep it
in `OpenFocusData`, which should move with the application. Linux and macOS
store downloaded components under the user's OpenFocus data directory.

**Automatic** prefers mirrors for mainland China network connections and
official sources elsewhere. When the network region is unknown, Windows can
use its system **Country or region**. Use **Mirrors first** or **Official
first** to override the order. OpenFocus downloads from a catalog of sources
and can resume a partial download on the next source.

Choose **Install from folder…** to use saved archives listed in the matching
`core/accelerator_*.json` manifest. Each download and local file is checked
against its pinned size and SHA-256. The installer activates the isolated
worker after the GPU probe passes. The Linux and macOS probes execute all
three shipped StackMFF models.

TensorRT compiles an FP32 engine for each ONNX model and input shape and reuses compatible plans. PyTorch CUDA and MPS use the verified FP32 checkpoint for each shipped StackMFF model. When an engine declines the model, shape or memory budget, OpenFocus tries the next compatible engine. Decoding, registration, colour reconstruction and export have their own execution costs.

An integrated GPU is a GPU and can accelerate AI fusion when its driver and a supported ONNX Runtime execution provider can execute the bundled model. OpenFocus does not require a discrete GPU, dedicated VRAM or a particular adapter name. DirectML tries the enumerated hardware adapters, including integrated Intel/AMD adapters, preferring adapters with more dedicated memory without excluding shared-memory devices. Apple silicon also has an integrated GPU; CoreML may choose CPU, GPU or Neural Engine compute units. A CoreML profile proves use of the CoreML provider, not which compute unit it used.

Hardware inventory, engine availability and actual execution are reported separately. The managed pack check identifies eligible engines without running a model on the GUI thread. ONNX Runtime verifies its accelerator by running a small bundled model and inspecting the resulting placement profile. The environment report records successful inference on the requested model and gives the reason when an engine is skipped or fails.

An explicit CPU preference runs neural inference on CPU. With acceleration enabled, the status shows an eligible engine before a render and the actual engine after successful execution. Each ONNX accelerator session checks its first real inference profile, since placement can differ by model or input shape. Entirely CPU-placed inference is reported as CPU. Mixed placement is reported by provider name. Profiles are deleted after inspection and placement is retained in the bounded session cache.

If an engine fails or its memory budget is unknown or insufficient, OpenFocus can fall back for that job. Windows DirectML admission uses the selected adapter's DXGI LOCAL budget and host RAM. CUDA and ROCm require a readable free-VRAM estimate. An OpenVINO GPU provider needs a reliable free-VRAM estimate for admission. MPS uses unified memory, so its model, worker and staging buffers share the host-RAM budget. Image decoding, alignment and output also use system memory and temporary disk space.

The GPU Acceleration menu option is a preference for future supported jobs. For neural inference it remains available for a verified provider when its free memory is temporarily unreadable, so it can be enabled or disabled and a later job can retry the memory check. The execution status continues to show CPU while the provider is unadmitted.

For tiled StackMFF V1 and V3 inference, OpenFocus retains the two compiled input scales when GPU memory permits. Cached sessions are released and memory is checked again before a new shape falls back to CPU.

The spatial planner prefers the largest tile that fits a verified accelerator's
memory budget. Engine priority breaks ties, and CPU provides the final fallback.
The PyTorch CUDA and MPS worker bounds FP32 attention scores by processing part
of the query at a time. Every cross-layer query still uses all focus layers.
On CUDA, the worker can reuse admission for a checkpoint and layer count on
tiles up to the size it has already completed.

Large tiled renders read plain native-endian RGB TIFF directly when its samples
are contiguous and no resize or orientation transform is needed. Other sources
can be prepared once in their native bit depth when temporary space allows.
Exposure adjustment and float32 normalization then operate on each crop.
Applied registration and export can still require substantial temporary space.

During a render, a rejected engine for a model and input shape is skipped when the shape recurs. A later independent render can try the accelerator again. When inference fails, OpenFocus releases the failed session before admitting a retry. The provider or driver controls when its native memory is returned.

Multiscale inference releases temporary host arrays after their last use: the reduced input after its inference returns, and the reduced probabilities after they have been combined. Memory admission remains conservative; actual host memory usage and release depend on the allocator and runtime.

After V1 inference, OpenFocus weights the luma samples in their existing host buffer before summing them. This saves one float32 image per source frame during luma reconstruction.

Cancellation is cooperative. An ONNX Runtime, OpenCV or NumPy call already in progress finishes before cancellation is observed; surrounding resize and per-layer processing checks for cancellation between work units. A cancelled job does not start a CPU retry, and its heavy-job lease remains held until ongoing synchronous work has returned.
Local inventory observed on 2026-11-06 lists Intel UHD Graphics 770 and three virtual-display drivers. The system Python environment has ONNX Runtime 1.31.0 advertising AzureExecutionProvider and CPUExecutionProvider, which provides no supported AI accelerator here. A separate Windows validation environment uses ONNX Runtime DirectML 1.23.0: a complete GUI StackMFF V3 render of two 256×192 uint16 RGB frames ran on Intel UHD Graphics 770, with the real model profile confirming DirectML placement and the final output retaining uint16 dimensions. Turning acceleration off then completed another render on CPU and displayed CPU mode. Provider availability is therefore environment-specific; the system-Python inventory does not describe the Windows package's inference capability. CUDA/RTX 3090, ROCm, OpenVINO and CoreML remain unverified on physical devices here.
