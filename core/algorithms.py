"""Explicit, dependency-free algorithm metadata and parameter validation.

Runners and capability hooks are imported only when execution needs them.
"""
from dataclasses import dataclass
from importlib import import_module
import math
from numbers import Real
from collections.abc import Mapping


@dataclass(frozen=True)
class ParameterSpec:
    key: str
    label_key: str
    default: object
    minimum: object = None
    maximum: object = None
    step: object = 1
    decimals: int = 0
    choices: tuple = ()
    odd: bool = False
    hint_key: str | None = None
    choice_labels: tuple = ()


@dataclass(frozen=True)
class DepthSpec:
    runner: str
    parameters: tuple = ()


@dataclass(frozen=True)
class FusionAlgorithm:
    id: str
    label_key: str
    display_name: str
    file_tag: str
    runner: str
    parameters: tuple = ()
    depth: DepthSpec | None = None
    hint_key: str | None = None
    environment: str | None = None
    resource_estimator: str | None = None
    parallel_frames: bool = False
    ordered_plan: bool = False
    neural: bool = False
    sort_frames: bool = False
    roi: str | None = None
    promotes_color: bool = False
    tile_decider: str | None = None
    default_exposure_match: bool = True
    default_base_correct: bool = False
    tile_runner: str | None = None
    depth_planner: str | None = None
    # Repo-relative checkpoint a neural back-end loads when the caller does not
    # pass model_path. Keeps V3 and the older StackMFF models independent.
    weights: str | None = None
    # The stack's own focus order carries meaning (a recurrent model reads it);
    # reverse/dir-order controls must not reorder the frames.
    order_sensitive: bool = False
    # UI grouping: 'ai', 'spatial' or 'transform'. Contributing methods declare
    # their family so every selector can present them together.
    category: str = 'spatial'


@dataclass(frozen=True)
class RegistrationAlgorithm:
    id: str
    label_key: str
    file_tag: str
    runner: str | None = None
    parameters: tuple = ()
    steps: tuple = ()
    hint_key: str | None = None


_FUSION = {}
_REGISTRATION = {}


def _path(path):
    if not isinstance(path, str) or path.count(':') != 1:
        raise ValueError(f'Invalid lazy callable path: {path!r}')
    module, attrs = path.split(':')
    if not module or not attrs or any(not part.isidentifier() for part in (module + '.' + attrs).split('.')):
        raise ValueError(f'Invalid lazy callable path: {path!r}')


def resolve_runner(path):
    """Resolve ``module:Class.nested_callable`` without eager backend imports."""
    _path(path)
    module, attrs = path.split(':')
    result = import_module(module)
    for attr in attrs.split('.'):
        result = getattr(result, attr)
    if not callable(result):
        raise ValueError(f'Algorithm entry point is not callable: {path}')
    return result


def validate_parameter(spec, value):
    if spec.choices:
        if not any(type(value) is type(choice) and value == choice for choice in spec.choices):
            raise ValueError(f'Invalid choice for {spec.key}: {value!r}')
        return value
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f'{spec.key} must be a finite number')
    if spec.decimals == 0:
        if int(value) != value:
            raise ValueError(f'{spec.key} must be an integer')
        value = int(value)
    else:
        value = float(value)
    if spec.minimum is not None and value < spec.minimum:
        raise ValueError(f'{spec.key} must be at least {spec.minimum}')
    if spec.maximum is not None and value > spec.maximum:
        raise ValueError(f'{spec.key} must be at most {spec.maximum}')
    if spec.odd and value % 2 != 1:
        raise ValueError(f'{spec.key} must be odd')
    return value


def _parameters(specs):
    if not isinstance(specs, tuple):
        raise ValueError('Parameter descriptors must be a tuple')
    keys = set()
    for spec in specs:
        if (not isinstance(spec, ParameterSpec) or not isinstance(spec.key, str)
                or not spec.key.isidentifier() or not isinstance(spec.label_key, str) or not spec.label_key):
            raise ValueError('Invalid parameter descriptor')
        if spec.key in keys:
            raise ValueError(f'Duplicate parameter: {spec.key}')
        keys.add(spec.key)
        if not isinstance(spec.choices, tuple) or not isinstance(spec.decimals, int) or isinstance(spec.decimals, bool) or spec.decimals < 0:
            raise ValueError(f'Invalid parameter schema: {spec.key}')
        if not isinstance(spec.choice_labels, tuple) or (spec.choice_labels and len(spec.choice_labels) != len(spec.choices)):
            raise ValueError(f'Choice labels must match choices: {spec.key}')
        if any(not isinstance(label, str) or not label for label in spec.choice_labels):
            raise ValueError(f'Invalid choice label: {spec.key}')
        if not isinstance(spec.odd, bool) or (spec.hint_key is not None and not isinstance(spec.hint_key, str)):
            raise ValueError(f'Invalid parameter metadata: {spec.key}')
        if spec.choices:
            if any(not isinstance(v, (str, int, float, bool)) or isinstance(v, float) and not math.isfinite(v) for v in spec.choices):
                raise ValueError(f'Invalid choices: {spec.key}')
            if len({(type(v), v) for v in spec.choices}) != len(spec.choices):
                raise ValueError(f'Duplicate choices: {spec.key}')
        else:
            for bound in (spec.minimum, spec.maximum, spec.step):
                if bound is not None and (isinstance(bound, bool) or not isinstance(bound, Real) or not math.isfinite(bound)):
                    raise ValueError(f'Invalid numeric schema: {spec.key}')
            if spec.step is None or spec.step <= 0 or (spec.minimum is not None and spec.maximum is not None and spec.minimum > spec.maximum):
                raise ValueError(f'Invalid numeric bounds: {spec.key}')
            if spec.odd and spec.decimals:
                raise ValueError(f'Odd parameters must be integers: {spec.key}')
        validate_parameter(spec, spec.default)


def _identity(spec):
    if (not isinstance(spec.id, str) or not spec.id.isidentifier()
            or not isinstance(spec.label_key, str) or not spec.label_key or not isinstance(spec.file_tag, str)
            or (spec.hint_key is not None and not isinstance(spec.hint_key, str))):
        raise ValueError('Invalid algorithm identity')
    _parameters(spec.parameters)


def register_fusion(spec):
    if not isinstance(spec, FusionAlgorithm):
        raise ValueError('Expected FusionAlgorithm')
    _identity(spec)
    if spec.id == 'none':
        raise ValueError("Fusion algorithm id 'none' is reserved for alignment-only output")
    if spec.id in _FUSION:
        raise ValueError(f'Duplicate fusion algorithm: {spec.id}')
    if not isinstance(spec.display_name, str) or not spec.display_name or not spec.file_tag:
        raise ValueError('Fusion display name and file tag are required')
    for field in ('parallel_frames', 'ordered_plan', 'neural', 'sort_frames', 'promotes_color',
                  'default_exposure_match', 'default_base_correct', 'order_sensitive'):
        if not isinstance(getattr(spec, field), bool):
            raise ValueError(f'{field} must be boolean')
    if spec.sort_frames and spec.order_sensitive:
        raise ValueError('A sortable method cannot also require the input order')
    if spec.category not in ('ai', 'spatial', 'transform'):
        raise ValueError(f'Invalid fusion category: {spec.category!r}')
    if spec.weights is not None:
        if (not isinstance(spec.weights, str)
                or not spec.weights.startswith('weights/')
                or spec.weights.count('/') != 1
                or not spec.weights.split('/', 1)[1].endswith('.pth')):
            raise ValueError('weights must be weights/<file>.pth')
    _path(spec.runner)
    for hook in (spec.environment, spec.resource_estimator, spec.roi, spec.tile_decider,
                 spec.tile_runner, spec.depth_planner):
        if hook is not None:
            _path(hook)
    if spec.depth is not None:
        if not isinstance(spec.depth, DepthSpec):
            raise ValueError('Invalid depth descriptor')
        _path(spec.depth.runner)
        _parameters(spec.depth.parameters)
    _FUSION[spec.id] = spec


def register_registration(spec):
    if not isinstance(spec, RegistrationAlgorithm):
        raise ValueError('Expected RegistrationAlgorithm')
    _identity(spec)
    if spec.id in _REGISTRATION:
        raise ValueError(f'Duplicate registration algorithm: {spec.id}')
    if not isinstance(spec.steps, tuple):
        raise ValueError('Registration steps must be a tuple')
    if spec.runner is not None:
        _path(spec.runner)
        if spec.steps:
            raise ValueError('Atomic registration cannot declare composite steps')
    elif spec.id != 'none' and not spec.steps:
        raise ValueError('Registration requires a runner or steps')
    if spec.id == 'none' and (spec.runner is not None or spec.steps or spec.parameters):
        raise ValueError('none must be an empty alignment')
    for step in spec.steps:
        get_registration(step)
    inherited = {parameter.key: parameter for parameter in registration_parameter_specs(spec.steps)}
    for parameter in spec.parameters:
        if parameter.key in inherited and inherited[parameter.key] != parameter:
            raise ValueError(f'Conflicting composite registration parameter: {parameter.key}')
    _REGISTRATION[spec.id] = spec


def fusion_algorithms():
    return tuple(_FUSION.values())


def registration_algorithms():
    return tuple(_REGISTRATION.values())


def depth_algorithms():
    return tuple(spec for spec in _FUSION.values() if spec.depth is not None)


def get_fusion(id):
    try:
        return _FUSION[id]
    except (KeyError, TypeError):
        raise ValueError(f'Unknown fusion algorithm: {id!r}') from None


def get_registration(id):
    try:
        return _REGISTRATION[id]
    except (KeyError, TypeError):
        raise ValueError(f'Unknown registration algorithm: {id!r}') from None


def _resolve(specs, values):
    if values is not None and not isinstance(values, Mapping):
        raise ValueError('Algorithm parameters must be a mapping')
    values = {} if values is None else values
    known = {spec.key for spec in specs}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f'Unknown algorithm parameters: {", ".join(sorted(map(str, unknown)))}')
    return {spec.key: validate_parameter(spec, values.get(spec.key, spec.default)) for spec in specs}


def resolve_fusion_parameters(id, values=None, depth=False):
    spec = get_fusion(id)
    if depth:
        if spec.depth is None:
            raise ValueError(f'Fusion algorithm has no depth measure: {id}')
        return _resolve(spec.depth.parameters, values)
    return _resolve(spec.parameters, values)


def fusion_defaults(id, depth=False):
    return resolve_fusion_parameters(id, depth=depth)


def alignment_steps(id_or_steps):
    if isinstance(id_or_steps, str):
        spec = get_registration(id_or_steps)
        if spec.runner is not None:
            return (spec.id,)
        return tuple(atom for step in spec.steps for atom in alignment_steps(step))
    if not isinstance(id_or_steps, (tuple, list)):
        raise ValueError('Alignment must be an algorithm id or ordered steps')
    return tuple(atom for step in id_or_steps for atom in alignment_steps(step))


def registration_parameter_specs(id):
    steps = alignment_steps(id)
    specs = []
    if isinstance(id, str):
        specs.extend(get_registration(id).parameters)
    for step in steps:
        for parameter in get_registration(step).parameters:
            existing = next((item for item in specs if item.key == parameter.key), None)
            if existing is not None and existing != parameter:
                raise ValueError(f'Conflicting composite registration parameter: {parameter.key}')
            if existing is None:
                specs.append(parameter)
    return tuple(specs)


def resolve_registration_parameters(id, values=None):
    return _resolve(registration_parameter_specs(id), values)


def registration_defaults(id):
    return resolve_registration_parameters(id)


_K_GUIDED = ParameterSpec('kernel_size', 'label_kernel_guided', 31, 1, 51, 2, odd=True)
_K_DCT = ParameterSpec('kernel_size', 'label_kernel_dct', 7, 1, 51, 2, odd=True)
_K_GFG = ParameterSpec('kernel_size', 'label_kernel_gfg', 7, 1, 51, 2, odd=True)
_K_OF = ParameterSpec('kernel_size', 'label_kernel_of', 11, 1, 21, 2, odd=True)
_LEVELS = ParameterSpec('N', 'compare_levels', 4, 1, 6)
_MODE = ParameterSpec('mode', 'label_ai_mode', 'soft', choices=('soft', 'hard'), choice_labels=('radio_ai_soft', 'radio_ai_hard'))
_ADAPTER = 'core.algorithm_adapters:'
_FUSION_CLASS = 'core.multi_focus_fusion:MultiFocusFusion.'

register_fusion(FusionAlgorithm('stackmffv3', 'radio_stackmff', 'StackMFF V3', 'StackMFFV3', _FUSION_CLASS + '_fuse_stackmffv3', (_MODE,), DepthSpec(_ADAPTER + 'depth_neural', (_MODE,)), hint_key='radio_stackmff_hint', environment=_FUSION_CLASS + '_validate_ai_environment', neural=True, roi=_ADAPTER + 'roi_neural', promotes_color=True, tile_decider=_ADAPTER + 'neural_needs_tiles', tile_runner=_FUSION_CLASS + '_fuse_tiled_ai_batched', depth_planner='core.depth_map:_plan_neural_depth', weights='weights/stackmff_v3_final.pth', category='ai'))
register_fusion(FusionAlgorithm('stackmffv2', 'radio_stackmff_v2', 'StackMFF V2', 'StackMFFV2', _FUSION_CLASS + '_fuse_stackmffv2', (), DepthSpec(_ADAPTER + 'depth_stackmffv2', ()), hint_key='radio_stackmff_v2_hint', environment=_FUSION_CLASS + '_validate_ai_environment', neural=True, order_sensitive=True, promotes_color=True, tile_decider=_ADAPTER + 'stackmffv2_needs_tiles', depth_planner='core.depth_map:_plan_stackmffv2_depth', weights='weights/stackmff_v2_final.pth', category='ai'))
register_fusion(FusionAlgorithm('stackmff', 'radio_stackmff_v1', 'StackMFF', 'StackMFF', _FUSION_CLASS + '_fuse_stackmff', (), hint_key='radio_stackmff_v1_hint', environment=_FUSION_CLASS + '_validate_ai_environment', neural=True, promotes_color=True, tile_decider=_ADAPTER + 'neural_needs_tiles', tile_runner=_FUSION_CLASS + '_fuse_tiled_stackmff', weights='weights/stackmff_final.pth', category='ai'))
register_fusion(FusionAlgorithm('openfocus_a', 'radio_openfocus_a', 'OpenFocus A', 'OpenFocusA', _FUSION_CLASS + '_fuse_openfocus_a', (_K_OF, ParameterSpec('blend_smoothing', 'label_blend_smoothing', 2.0, 0, 8, .2, 1)), hint_key='radio_openfocus_a_hint', sort_frames=True, roi=_ADAPTER + 'roi_openfocus_a', default_base_correct=True, category='spatial'))
register_fusion(FusionAlgorithm('openfocus_b', 'radio_openfocus_b', 'OpenFocus B', 'OpenFocusB', _FUSION_CLASS + '_fuse_openfocus_b', (_K_OF, ParameterSpec('depth_smoothing', 'label_depth_smoothing', 1.2, 0, 8, .2, 1)), DepthSpec(_ADAPTER + 'depth_openfocus_b', (_K_OF,)), hint_key='radio_openfocus_b_hint', sort_frames=True, roi=_ADAPTER + 'roi_openfocus_b', default_base_correct=True, category='spatial'))
register_fusion(FusionAlgorithm('guided_filter', 'radio_guided_filter', 'Guided Filter', 'GuidedFilter', _FUSION_CLASS + '_fuse_guided_filter', (_K_GUIDED,), hint_key='radio_guided_filter_hint', environment=_FUSION_CLASS + '_validate_spatial_environment', parallel_frames=True, roi=_ADAPTER + 'roi_guided', promotes_color=True, category='spatial'))
register_fusion(FusionAlgorithm('gfgfgf', 'radio_gfg', 'GFG-FGF', 'GFGFGF', _FUSION_CLASS + '_fuse_gfgfgf', (_K_GFG,), DepthSpec(_ADAPTER + 'depth_gfg', (_K_GFG,)), hint_key='radio_gfg_hint', parallel_frames=True, roi=_ADAPTER + 'roi_gfg', promotes_color=True, category='spatial'))
register_fusion(FusionAlgorithm('openfocus_c', 'radio_openfocus_c', 'OpenFocus C', 'OpenFocusC', _FUSION_CLASS + '_fuse_openfocus_c', (ParameterSpec('blend_smoothing', 'label_blend_smoothing', 3.0, 0, 8, .2, 1),), DepthSpec(_ADAPTER + 'depth_openfocus_c', (ParameterSpec('kernel_size', 'depth_kernel_energy', 11, 1, 21, 2, odd=True),)), hint_key='radio_openfocus_c_hint', sort_frames=True, roi=_ADAPTER + 'roi_openfocus_c', default_base_correct=True, category='transform'))
register_fusion(FusionAlgorithm('dct', 'radio_dct', 'DCT', 'DCT', _FUSION_CLASS + '_fuse_dct', (_K_DCT,), DepthSpec(_ADAPTER + 'depth_dct', (_K_DCT,)), environment=_FUSION_CLASS + '_validate_dct_environment', hint_key='radio_dct_hint', roi=_ADAPTER + 'roi_dct', promotes_color=True, category='transform'))
register_fusion(FusionAlgorithm('dtcwt', 'radio_dtcwt', 'DTCWT', 'DTCWT', _FUSION_CLASS + '_fuse_dtcwt', (_LEVELS,), DepthSpec(_ADAPTER + 'depth_dtcwt', (_LEVELS,)), environment=_FUSION_CLASS + '_validate_transform_environment', hint_key='radio_dtcwt_hint', resource_estimator=_ADAPTER + 'dtcwt_resources', parallel_frames=True, ordered_plan=True, roi=_ADAPTER + 'roi_dtcwt', promotes_color=True, tile_decider=_ADAPTER + 'dtcwt_needs_tiles', category='transform'))
register_registration(RegistrationAlgorithm('none', 'check_align_none', '', hint_key='check_align_none_hint'))
register_registration(RegistrationAlgorithm('homography', 'check_align_homography', 'Homography', _ADAPTER + 'registration_homography', hint_key='check_align_homography_hint'))
register_registration(RegistrationAlgorithm('ecc', 'check_align_ecc', 'ECC', _ADAPTER + 'registration_ecc', hint_key='check_align_ecc_hint'))
register_registration(RegistrationAlgorithm('openfocus', 'check_align_of', 'OpenFocusAlign', _ADAPTER + 'registration_openfocus', (ParameterSpec('max_shift_ratio', 'label_align_shift_limit', .08, 0, 1, .01, 2),), hint_key='check_align_of_hint'))
register_registration(RegistrationAlgorithm('both', 'check_align_both', 'Homography_ECC', steps=('homography', 'ecc'), hint_key='check_align_both_hint'))
