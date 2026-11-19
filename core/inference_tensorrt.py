"""Build shape-specific FP32 TensorRT engines from ONNX inside the managed worker."""
from collections import deque
import ctypes
import hashlib
import hmac
import json
import os
from pathlib import Path
import sys
import tempfile


_MAX_PLAN_BYTES = 256 * 1024 ** 2


def workspace_bytes(layers, height, width):
    """Bound the builder scratch space reserved by desktop memory admission."""
    # TensorRT 10.4 needs 1,003,487,232 bytes for its 29x512x512 attention tactic.
    return min(1536 << 20, max(256 << 20, layers * height * width * 160))


def _driver_version():
    if os.name == 'nt':
        # This DLL is shipped and updated with the display driver. cuDriverGetVersion
        # alone would miss driver updates within one CUDA API version.
        path = Path(os.environ['SystemRoot']) / 'System32' / 'nvcuda.dll'
        info = path.stat()
        return f'{info.st_size}:{info.st_mtime_ns}'
    return Path('/proc/driver/nvidia/version').read_text().strip()


def _signing_key():
    # The key belongs to the local user, outside the cache path supplied on stdin.
    # A downloaded/copied engine and even an accompanying manifest cannot authorize it.
    if os.name == 'nt':
        parent = Path(os.environ['LOCALAPPDATA']) / 'OpenFocus' / 'runtime-keys'
    else:
        parent = Path.home() / '.local' / 'share' / 'OpenFocus' / 'runtime-keys'
    parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    path = parent / 'tensorrt-hmac-v1'
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, 'O_BINARY', 0), 0o600)
    except FileExistsError:
        key = path.read_bytes()
    else:
        key = os.urandom(32)
        with os.fdopen(fd, 'wb') as handle:
            handle.write(key)
    if len(key) != 32:
        raise OSError('Incomplete local TensorRT signing key')
    return key


class PlanCache:
    def __init__(self, directory, signature):
        directory.mkdir(parents=True, mode=0o700, exist_ok=True)
        self.key = _signing_key()
        self.identity = hashlib.sha256(json.dumps(signature).encode('utf-8')).digest()
        self.path = directory / f'of-{self.identity.hex()}.plan'

    def _mac(self, plan):
        auth = hmac.new(self.key, self.identity, hashlib.sha256)
        auth.update(plan)
        return auth.digest()

    def read(self):
        try:
            with self.path.open('rb') as handle:
                size = os.fstat(handle.fileno()).st_size - 32
                if not 0 < size <= _MAX_PLAN_BYTES:
                    return None
                signature = handle.read(32)
                plan = handle.read(size)
            if len(plan) == size and hmac.compare_digest(signature, self._mac(plan)):
                return plan
        except OSError:
            pass
        return None

    def store(self, plan):
        if len(plan) > _MAX_PLAN_BYTES:
            return
        fd, name = tempfile.mkstemp(dir=self.path.parent, suffix='.building')
        try:
            with os.fdopen(fd, 'wb') as handle:
                handle.write(self._mac(plan))
                handle.write(plan)
            os.replace(name, self.path)
        finally:
            Path(name).unlink(missing_ok=True)
        # Every image geometry creates a different specialization. Retain only
        # a bounded number on disk, including across app restarts.
        plans = sorted(self.path.parent.glob('of-*.plan'), key=lambda p: p.stat().st_mtime, reverse=True)
        for old in plans[16:]:
            old.unlink(missing_ok=True)


def resolve_weights(graph, digest, directory):
    """Include the actual external weights in an imported ONNX graph's identity."""
    import onnx
    from onnx import external_data_helper as data
    model = onnx.load_model_from_string(graph)
    if not any(data.uses_external_data(t) for t in data._get_all_tensors(model)):
        return graph, digest
    data.load_external_data_for_model(model, str(directory))
    graph = model.SerializeToString()
    return graph, hashlib.sha256(graph).hexdigest()


def _stable_attention(model):
    """TensorRT 10.4's fused attention loses precision on the shipped high logits.

    Spell out the equivalent max-subtracted softmax on these known opset-17
    graphs. Their attention scores can exceed 2,000 before normalization.
    """
    from onnx import TensorProto, helper
    axis = 'openfocus_softmax_axis'
    model.graph.initializer.append(helper.make_tensor(axis, TensorProto.INT64, [1], [-1]))
    nodes = []
    for node in model.graph.node:
        if node.op_type != 'Softmax':
            nodes.append(node)
            continue
        x, y = node.input[0], node.output[0]
        peak, shifted, exp, total = (y + suffix for suffix in ('/peak', '/shifted', '/exp', '/total'))
        nodes.extend((
            helper.make_node('ReduceMax', [x], [peak], axes=[-1], keepdims=1),
            helper.make_node('Sub', [x, peak], [shifted]),
            helper.make_node('Exp', [shifted], [exp]),
            helper.make_node('ReduceSum', [exp, axis], [total], keepdims=1),
            helper.make_node('Div', [exp, total], [y]),
        ))
    del model.graph.node[:]
    model.graph.node.extend(nodes)


def _specialize(graph, shape, stable_attention):
    import onnx
    model = onnx.load_model_from_string(graph)
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise ValueError('Expected one ONNX input and one logits output')
    for value in (model.graph.input[0], model.graph.output[0]):
        tensor = value.type.tensor_type
        if tensor.elem_type != onnx.TensorProto.FLOAT or len(tensor.shape.dim) != len(shape):
            raise ValueError('Expected rank-four float32 ONNX input and output')
        for dim, size in zip(tensor.shape.dim, shape):
            if dim.HasField('dim_value') and dim.dim_value != size:
                raise ValueError(f'ONNX tensor {value.name} has an incompatible static dimension')
            dim.dim_value = size
    if stable_attention:
        _stable_attention(model)
    return model.SerializeToString()


class TensorRT:
    def __init__(self, torch):
        try:
            import tensorrt as trt
        except ModuleNotFoundError:
            import tensorrt_bindings as trt
        import onnx  # Required for shape specialization, including in a frozen release.

        self.trt = trt
        self.torch = torch
        class BuildLogger(trt.ILogger):
            def __init__(self):
                super().__init__()
                self.recent = deque(maxlen=3)

            def log(self, severity, message):
                if severity <= trt.ILogger.Severity.WARNING:
                    self.recent.append(str(message)[:400])

        self.logger = BuildLogger()
        self.runtime = trt.Runtime(self.logger)
        # Importing a binding is insufficient on a driver/GPU unsupported by its builder.
        builder = trt.Builder(self.logger)
        if not builder or not self.runtime:
            raise RuntimeError('TensorRT builder/runtime unavailable')
        self.stream = torch.cuda.Stream(device=0)
        self.signature = None
        try:
            props = torch.cuda.get_device_properties(0)
            # Older Torch wheels omit GPU UUID from their Python properties.
            driver = ctypes.WinDLL('nvcuda.dll') if os.name == 'nt' else ctypes.CDLL('libcuda.so.1')
            device = ctypes.c_int()
            uuid = (ctypes.c_ubyte * 16)()
            if (driver.cuCtxGetDevice(ctypes.byref(device)) or
                    driver.cuDeviceGetUuid(ctypes.byref(uuid), device)):
                raise OSError('CUDA device identity unavailable')
            self.signature = (3, trt.__version__, onnx.__version__, torch.version.cuda,
                              _driver_version(), bytes(uuid).hex(), props.name,
                              props.major, props.minor, 'fp32-no-tf32')
        except (OSError, AttributeError, KeyError) as exc:
            print(f'TensorRT disk cache disabled: {exc}', file=sys.stderr)

    def _build(self, graph, shape, stable_attention, workspace):
        trt = self.trt
        self.logger.recent.clear()
        free, _ = self.torch.cuda.mem_get_info(0)
        if workspace > free // 2:
            raise MemoryError(f'TensorRT workspace requires {workspace >> 20} MiB with '
                              f'only {free >> 20} MiB free')
        builder = trt.Builder(self.logger)
        network = builder.create_network(0)
        parser = trt.OnnxParser(network, self.logger)
        if not parser.parse(_specialize(graph, shape, stable_attention)):
            error = str(parser.get_error(parser.num_errors - 1)) if parser.num_errors else 'unknown operator'
            raise ValueError(f'TensorRT could not import this graph: {error}')
        config = builder.create_builder_config()
        config.clear_flag(trt.BuilderFlag.TF32)
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace)
        config.builder_optimization_level = 1
        plan = builder.build_serialized_network(network, config)
        if plan is None:
            detail = ' | '.join(self.logger.recent) or 'builder returned no plan'
            raise RuntimeError(f'TensorRT could not build {shape} with '
                               f'{workspace >> 20} MiB workspace: {detail}')
        return bytes(plan)

    def load(self, graph, digest, shape, directory, *, reference):
        cache = None
        engine = None
        workspace = workspace_bytes(*shape[1:])
        if self.signature is not None:
            try:
                cache = PlanCache(directory, (*self.signature, digest, shape, workspace,
                                              reference is not None))
                plan = cache.read()
                if plan is not None:
                    engine = self.runtime.deserialize_cuda_engine(plan)
            except OSError as exc:
                print(f'TensorRT disk cache unavailable: {exc}', file=sys.stderr)
        if engine is not None:
            return self._bind(engine, shape)
        plan = self._build(graph, shape, reference is not None, workspace)
        # Run the reference before allocating TensorRT's execution context. Keep
        # only the reproducible input and expected logits across the two engines.
        if reference is not None:
            with self.torch.inference_mode():
                generator = self.torch.Generator(device='cuda:0').manual_seed(217)
                x = self.torch.rand(shape, generator=generator, device='cuda:0')
                expected = reference()(x)
            self.torch.cuda.empty_cache()
        engine = self.runtime.deserialize_cuda_engine(plan)
        if engine is None:
            raise RuntimeError('Failed to load locally built TensorRT engine')
        session = self._bind(engine, shape)
        if reference is not None:
            with self.torch.inference_mode():
                actual = session(x)
                error = actual.sub_(expected).abs_().max().item()
            if not error <= 1e-3:  # NaN and infinity fail the same parity check.
                raise ValueError(f'TensorRT logits differ from the verified PyTorch model '
                                 f'at {shape}: max absolute error {error:.6g}, limit 0.001')
        if cache is not None:
            try:
                cache.store(plan)
            except OSError as exc:
                print(f'TensorRT cache write failed: {exc}', file=sys.stderr)
        return session

    def _bind(self, engine, shape):
        trt = self.trt
        if engine.num_io_tensors != 2:
            raise ValueError('TensorRT engine must have one input and one output')
        names = {}
        for index in range(engine.num_io_tensors):
            name = engine.get_tensor_name(index)
            if (engine.get_tensor_dtype(name) != trt.float32 or
                    engine.get_tensor_location(name) != trt.TensorLocation.DEVICE or
                    engine.get_tensor_format(name) != trt.TensorFormat.LINEAR):
                raise ValueError('TensorRT I/O must be linear float32 CUDA storage')
            names[engine.get_tensor_mode(name)] = name
        src, dst = names[trt.TensorIOMode.INPUT], names[trt.TensorIOMode.OUTPUT]
        context = engine.create_execution_context()
        if (context is None or tuple(context.get_tensor_shape(src)) != shape or
                tuple(context.get_tensor_shape(dst)) != shape):
            raise ValueError('TensorRT cannot resolve the requested logits shape')

        return _Session(self, engine, context, src, dst)


class _Session:
    def __init__(self, backend, engine, context, src, dst):
        self.backend = backend
        self.engine = engine  # Must outlive the native execution context.
        self.context = context
        self.src = src
        self.dst = dst

    def __call__(self, x):
        torch = self.backend.torch
        stream = self.backend.stream
        y = torch.empty_like(x)
        current = torch.cuda.current_stream()
        stream.wait_stream(current)
        if (not self.context.set_tensor_address(self.src, x.data_ptr()) or
                not self.context.set_tensor_address(self.dst, y.data_ptr()) or
                not self.context.execute_async_v3(stream.cuda_stream)):
            raise RuntimeError('TensorRT CUDA execution failed')
        current.wait_stream(stream)
        return y
