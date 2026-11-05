# GPU support

StackMFF V1, V2 and V3 use the bundled dynamic ONNX models through ONNX Runtime. The current runtime can select DirectML on Windows or CoreML on macOS and otherwise uses the CPU. Classical fusion and application registration use the CPU; explicitly requested low-level CuPy registration is a separate backend. PyTorch is not bundled for production inference.

An available provider does not by itself prove that the GPU executed a model. OpenFocus probes an accelerator by running a small model and inspecting its ONNX Runtime profile before selecting it. ONNX Runtime may place some operations on the CPU, and CoreML may use different compute units. If an accelerator fails or its memory budget is insufficient, inference can fall back to CPU. The provider probe is not a substitute for validating a complete job on the target machine.

The current implementation has not established support across all GPU vendors and configurations. Previous CUDA, ROCm and Intel XPU preview information does not establish support in this ONNX Runtime path. Real hardware validation is needed before making claims about a particular device, driver or performance. GPU memory does not replace the system memory and temporary disk space used for images, alignment and output.
