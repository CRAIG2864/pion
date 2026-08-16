# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import gc
import logging
import math
import warnings
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.dist_checkpointing.mapping import (
    ReplicaId,
    ShardedStateDict,
    ShardedTensorFactory,
)
from megatron.core.fusions.fused_bias_geglu import (
    bias_geglu_impl,
    quick_gelu,
    weighted_bias_quick_geglu_impl,
)
from megatron.core.fusions.fused_bias_gelu import bias_gelu_impl
from megatron.core.fusions.fused_bias_swiglu import bias_swiglu_impl, weighted_bias_swiglu_impl
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import (
    get_tensor_model_parallel_group_if_none,
    nvtx_range_pop,
    nvtx_range_push,
)

try:
    import transformer_engine  # pylint: disable=unused-import

    HAVE_TE = True
except ImportError:
    HAVE_TE = False


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _RandomizedDCTFactors:
    """Factors for Q = P1 D1 C P2 D2 C P3 with orthonormal DCT-II matrix C."""

    permutation_1: torch.Tensor
    signs_1: torch.Tensor
    permutation_2: torch.Tensor
    signs_2: torch.Tensor
    permutation_3: torch.Tensor


def _orthonormal_dct_ii(vectors: torch.Tensor) -> torch.Tensor:
    """Apply an orthonormal DCT-II to batches of row-stored column vectors."""
    size = vectors.shape[-1]
    if size < 1:
        raise ValueError('The DCT dimension must be positive.')

    reordered = torch.cat(
        (vectors[..., ::2], vectors[..., 1::2].flip(dims=(-1,))), dim=-1
    )
    spectrum = torch.fft.fft(reordered, dim=-1)
    phase = -math.pi * torch.arange(size, dtype=vectors.dtype, device=vectors.device)
    phase = phase / (2.0 * size)
    transformed = spectrum.real * torch.cos(phase) - spectrum.imag * torch.sin(phase)
    transformed[..., 0] /= math.sqrt(size)
    if size > 1:
        transformed[..., 1:] /= math.sqrt(size / 2.0)
    return transformed


def _rademacher_signs(size: int, generator: torch.Generator) -> torch.Tensor:
    signs = torch.randint(0, 2, (size,), generator=generator, dtype=torch.int64)
    return signs.mul(2).sub(1).to(dtype=torch.float64)


def _randomized_dct_factors(
    size: int, generator: torch.Generator
) -> _RandomizedDCTFactors:
    return _RandomizedDCTFactors(
        permutation_1=torch.randperm(size, generator=generator),
        signs_1=_rademacher_signs(size, generator),
        permutation_2=torch.randperm(size, generator=generator),
        signs_2=_rademacher_signs(size, generator),
        permutation_3=torch.randperm(size, generator=generator),
    )


def _apply_randomized_dct(
    vectors: torch.Tensor, factors: _RandomizedDCTFactors
) -> torch.Tensor:
    """Apply the randomized orthogonal transform represented by ``factors``."""
    result = vectors.index_select(-1, factors.permutation_3)
    result = _orthonormal_dct_ii(result)
    result = result * factors.signs_2
    result = result.index_select(-1, factors.permutation_2)
    result = _orthonormal_dct_ii(result)
    result = result * factors.signs_1
    return result.index_select(-1, factors.permutation_1)


def _antipodal_parseval_frame(
    row_count: int, column_count: int, generator: torch.Generator
) -> torch.Tensor:
    """Construct the full or maximal-partial antipodal Parseval frame from PAIR."""
    if row_count >= 2 * column_count:
        half_rows = row_count // 2
        factors = _randomized_dct_factors(half_rows, generator)
        selected_columns = torch.randperm(half_rows, generator=generator)[:column_count]
        selected_basis = torch.zeros(column_count, half_rows, dtype=torch.float64)
        selected_basis[torch.arange(column_count), selected_columns] = 1.0
        frame_half = _apply_randomized_dct(selected_basis, factors).transpose(0, 1)
        frame = torch.cat((frame_half, -frame_half), dim=0) / math.sqrt(2.0)
        if row_count % 2 == 1:
            frame = torch.cat(
                (frame, torch.zeros(1, column_count, dtype=torch.float64)), dim=0
            )
    else:
        paired_rows = row_count - column_count
        factors = _randomized_dct_factors(column_count, generator)
        orthogonal = _apply_randomized_dct(
            torch.eye(column_count, dtype=torch.float64), factors
        ).transpose(0, 1)
        frame = torch.cat(
            (
                orthogonal[:paired_rows] / math.sqrt(2.0),
                -orthogonal[:paired_rows] / math.sqrt(2.0),
                orthogonal[paired_rows:],
            ),
            dim=0,
        )

    row_permutation = torch.randperm(row_count, generator=generator)
    return frame.index_select(0, row_permutation)


def _parseval_frame(
    row_count: int, column_count: int, generator: torch.Generator
) -> torch.Tensor:
    """Construct a randomized Parseval frame with orthonormal columns."""
    factors = _randomized_dct_factors(row_count, generator)
    selected_columns = torch.randperm(row_count, generator=generator)[:column_count]
    selected_basis = torch.zeros(column_count, row_count, dtype=torch.float64)
    selected_basis[torch.arange(column_count), selected_columns] = 1.0
    return _apply_randomized_dct(selected_basis, factors).transpose(0, 1)


def _swiglu_pair_frames(
    inner_width: int, hidden_size: int, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Construct the aligned antipodal gate frame and same-sign up frame."""
    if inner_width >= 2 * hidden_size:
        half_inner_width = inner_width // 2
        gate_half = _parseval_frame(half_inner_width, hidden_size, generator)
        up_half = _parseval_frame(half_inner_width, hidden_size, generator)
        gate_frame = torch.cat((gate_half, -gate_half), dim=0) / math.sqrt(2.0)
        up_frame = torch.cat((up_half, up_half), dim=0) / math.sqrt(2.0)
        paired_dimension = hidden_size
        defect_dimension = 0
    else:
        paired_dimension = inner_width - hidden_size
        defect_dimension = 2 * hidden_size - inner_width
        factors = _randomized_dct_factors(hidden_size, generator)
        orthogonal = _apply_randomized_dct(
            torch.eye(hidden_size, dtype=torch.float64), factors
        ).transpose(0, 1)
        paired = orthogonal[:paired_dimension]
        unpaired = orthogonal[paired_dimension:]
        gate_frame = torch.cat(
            (paired / math.sqrt(2.0), -paired / math.sqrt(2.0), unpaired), dim=0
        )
        up_frame = torch.cat(
            (paired / math.sqrt(2.0), paired / math.sqrt(2.0), unpaired), dim=0
        )

    return gate_frame, up_frame, paired_dimension, defect_dimension


def _right_multiply_reverse_pair_j0_transpose(vectors: torch.Tensor) -> torch.Tensor:
    half = vectors.shape[-1] // 2
    return torch.cat(
        (-vectors[..., half:].flip(dims=(-1,)), vectors[..., :half].flip(dims=(-1,))),
        dim=-1,
    )


def _pair_activation_gamma(activation_func) -> float:
    if activation_func is F.gelu:
        return 1.823403463
    if activation_func is F.relu:
        return 2.0
    if activation_func is F.silu:
        return 1.517929407
    raise ValueError(
        'PAIR initialization supports exact GELU, ReLU, and SiLU activations only.'
    )


def _pair_activation_derivative(values: torch.Tensor, activation_func) -> torch.Tensor:
    """Evaluate the derivative used by the one-time PAIR Jacobian diagnostic."""
    if activation_func is F.gelu:
        normal_cdf = 0.5 * (1.0 + torch.erf(values / math.sqrt(2.0)))
        normal_pdf = torch.exp(-0.5 * values.square()) / math.sqrt(2.0 * math.pi)
        return normal_cdf + values * normal_pdf
    if activation_func is F.relu:
        return (values > 0.0).to(dtype=values.dtype)
    if activation_func is F.silu:
        sigmoid = torch.sigmoid(values)
        return sigmoid * (1.0 + values * (1.0 - sigmoid))
    raise ValueError(
        'PAIR diagnostics support exact GELU, ReLU, and SiLU activations only.'
    )


@torch.no_grad()
def _pair_initialization_diagnostics(
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    activation_func,
    input_second_moment: float,
    seed: int,
    calibration_size: int,
) -> dict[str, float]:
    """Compute the bounded, one-time structural diagnostics from the PAIR construction."""
    if calibration_size < 1:
        raise ValueError('PAIR diagnostic calibration size must be positive.')

    ffn_hidden_size, hidden_size = up_weight.shape
    half_ffn_size = ffn_hidden_size // 2
    if tuple(down_weight.shape) != (hidden_size, ffn_hidden_size):
        raise ValueError('PAIR diagnostics received incompatible fc1 and fc2 shapes.')

    pair_scale_inverse = math.sqrt(2.0)
    ga = up_weight[:half_ffn_size] * pair_scale_inverse
    gb = down_weight[:, :half_ffn_size] * pair_scale_inverse

    gamma = _pair_activation_gamma(activation_func)
    beta = 0.25 * math.log(gamma)
    positions = torch.linspace(-1.0, 1.0, hidden_size, dtype=torch.float64)
    mean_squared_scale = 2.0 * half_ffn_size / (hidden_size * input_second_moment)
    sigma_center = (
        math.sqrt(mean_squared_scale) / torch.exp(2.0 * beta * positions).mean().sqrt()
    )

    generator = torch.Generator(device='cpu')
    generator.manual_seed(int(seed) + 2)
    inputs = torch.randn(
        hidden_size, calibration_size, generator=generator, dtype=torch.float64
    ) * math.sqrt(input_second_moment)
    directions = torch.randn(
        hidden_size, calibration_size, generator=generator, dtype=torch.float64
    )

    input_norm = torch.linalg.vector_norm(inputs).clamp_min(torch.finfo(torch.float64).tiny)
    direction_norm = torch.linalg.vector_norm(directions).clamp_min(
        torch.finfo(torch.float64).tiny
    )

    preactivations = up_weight @ inputs
    zero_output = down_weight @ activation_func(preactivations)
    zero_output_ratio = torch.linalg.vector_norm(zero_output) / input_norm

    directional_branch_jacobian = down_weight @ (
        _pair_activation_derivative(preactivations, activation_func)
        * (up_weight @ directions)
    )
    residual_jacobian_directional_error = (
        torch.linalg.vector_norm(directional_branch_jacobian) / direction_norm
    )

    canonical_map = 0.5 * (gb @ ga)
    theta_preactivations = torch.cat(
        (ga @ inputs, torch.zeros(half_ffn_size, calibration_size, dtype=torch.float64)),
        dim=0,
    )
    theta_output = down_weight @ activation_func(theta_preactivations)
    theta_target = (canonical_map @ inputs) / math.sqrt(2.0)
    theta_escape_error = torch.linalg.vector_norm(theta_output - theta_target) / input_norm

    ga_gram = ga.transpose(0, 1) @ ga
    gb_gram = gb @ gb.transpose(0, 1)
    ga_eigenvalues = torch.linalg.eigvalsh(ga_gram).clamp_min(0.0)
    gb_eigenvalues = torch.linalg.eigvalsh(gb_gram).clamp_min(0.0)
    ga_rank_tolerance = (
        max(ga.shape)
        * torch.finfo(torch.float64).eps
        * ga_eigenvalues.max()
    )
    gb_rank_tolerance = (
        max(gb.shape)
        * torch.finfo(torch.float64).eps
        * gb_eigenvalues.max()
    )
    reverse_spectral_target = (4.0 / float(sigma_center) ** 4) * ga_gram
    reverse_spectral_error = torch.linalg.vector_norm(
        gb_gram - reverse_spectral_target
    ) / torch.linalg.vector_norm(gb_gram).clamp_min(torch.finfo(torch.float64).tiny)

    identity = torch.eye(hidden_size, dtype=torch.float64)
    canonical_skew_error = torch.linalg.vector_norm(
        canonical_map + canonical_map.transpose(0, 1)
    ) / torch.linalg.vector_norm(canonical_map).clamp_min(torch.finfo(torch.float64).tiny)
    canonical_orthogonality_error = torch.linalg.vector_norm(
        canonical_map.transpose(0, 1) @ canonical_map - identity
    ) / math.sqrt(hidden_size)

    return {
        'up_rank': float((ga_eigenvalues > ga_rank_tolerance).sum()),
        'down_rank': float((gb_eigenvalues > gb_rank_tolerance).sum()),
        'up_sigma_min': float(torch.sqrt(ga_eigenvalues.min())),
        'up_sigma_max': float(torch.sqrt(ga_eigenvalues.max())),
        'down_sigma_min': float(torch.sqrt(gb_eigenvalues.min())),
        'down_sigma_max': float(torch.sqrt(gb_eigenvalues.max())),
        'reverse_spectral_error': float(reverse_spectral_error),
        'zero_output_ratio': float(zero_output_ratio),
        'residual_jacobian_directional_error': float(
            residual_jacobian_directional_error
        ),
        'theta_pi_over_4_escape_error': float(theta_escape_error),
        'canonical_map_skew_error': float(canonical_skew_error),
        'canonical_map_orthogonality_error': float(canonical_orthogonality_error),
        'calibration_input_second_moment': float(inputs.square().mean()),
    }


@torch.no_grad()
def _pair_swiglu_initialization_diagnostics(
    fused_fc1_weight: torch.Tensor,
    down_weight: torch.Tensor,
    input_second_moment: float,
    seed: int,
    calibration_size: int,
    paired_dimension: int,
    defect_dimension: int,
) -> dict[str, float]:
    """Compute one-time structural diagnostics for the three SwiGLU matrices."""
    hidden_size = fused_fc1_weight.shape[1]
    ffn_hidden_size = fused_fc1_weight.shape[0] // 2
    inner_width = ffn_hidden_size // 2
    gate_weight, up_weight = torch.chunk(fused_fc1_weight, 2, dim=0)

    pair_scale_inverse = math.sqrt(2.0)
    gate_base = gate_weight[:inner_width] * pair_scale_inverse
    up_base = up_weight[:inner_width] * pair_scale_inverse
    down_base = down_weight[:, :inner_width] * pair_scale_inverse

    generator = torch.Generator(device='cpu')
    generator.manual_seed(int(seed) + 3)
    inputs = torch.randn(
        hidden_size, calibration_size, generator=generator, dtype=torch.float64
    ) * math.sqrt(input_second_moment)

    gated_hidden = F.silu(gate_weight @ inputs) * (up_weight @ inputs)
    zero_output = down_weight @ gated_hidden
    zero_output_ratio = torch.linalg.vector_norm(zero_output) / torch.linalg.vector_norm(
        inputs
    )

    gate_gram = gate_base.transpose(0, 1) @ gate_base
    up_gram = up_base.transpose(0, 1) @ up_base
    down_gram = down_base @ down_base.transpose(0, 1)
    gate_up_metric_error = torch.linalg.vector_norm(
        gate_gram - up_gram
    ) / torch.linalg.vector_norm(gate_gram)
    down_scale = torch.sum(down_gram * gate_gram) / torch.sum(gate_gram.square())
    down_metric_error = torch.linalg.vector_norm(
        down_gram - down_scale * gate_gram
    ) / torch.linalg.vector_norm(down_gram)

    gate_eigenvalues = torch.linalg.eigvalsh(gate_gram).clamp_min(0.0)
    up_eigenvalues = torch.linalg.eigvalsh(up_gram).clamp_min(0.0)
    down_eigenvalues = torch.linalg.eigvalsh(down_gram).clamp_min(0.0)
    machine_epsilon = torch.finfo(torch.float64).eps
    gate_rank_tolerance = max(gate_base.shape) * machine_epsilon * gate_eigenvalues.max()
    up_rank_tolerance = max(up_base.shape) * machine_epsilon * up_eigenvalues.max()
    down_rank_tolerance = max(down_base.shape) * machine_epsilon * down_eigenvalues.max()

    return {
        'gate_rank': float((gate_eigenvalues > gate_rank_tolerance).sum()),
        'up_rank': float((up_eigenvalues > up_rank_tolerance).sum()),
        'down_rank': float((down_eigenvalues > down_rank_tolerance).sum()),
        'gate_sigma_min': float(torch.sqrt(gate_eigenvalues.min())),
        'gate_sigma_max': float(torch.sqrt(gate_eigenvalues.max())),
        'up_sigma_min': float(torch.sqrt(up_eigenvalues.min())),
        'up_sigma_max': float(torch.sqrt(up_eigenvalues.max())),
        'down_sigma_min': float(torch.sqrt(down_eigenvalues.min())),
        'down_sigma_max': float(torch.sqrt(down_eigenvalues.max())),
        'zero_output_ratio': float(zero_output_ratio),
        'gate_up_metric_error': float(gate_up_metric_error),
        'down_metric_error': float(down_metric_error),
        'paired_dimension': float(paired_dimension),
        'defect_dimension': float(defect_dimension),
        'calibration_input_second_moment': float(inputs.square().mean()),
    }


def _om_pair_spectrum_metrics(
    prefix: str, singular_values: torch.Tensor, matrix_shape: tuple[int, int]
) -> dict[str, float]:
    """Summarize the nonzero singular spectrum used by one OM-PAIR projection."""
    machine_epsilon = torch.finfo(singular_values.dtype).eps
    sigma_max = singular_values.max()
    sigma_min = singular_values.min()
    rank_tolerance = max(matrix_shape) * machine_epsilon * sigma_max
    squared = singular_values.square()
    probabilities = squared / squared.sum()
    effective_rank = torch.exp(
        -(probabilities * probabilities.clamp_min(machine_epsilon).log()).sum()
    )
    quantiles = torch.quantile(
        singular_values,
        torch.tensor((0.25, 0.5, 0.75), dtype=singular_values.dtype),
    )
    condition_number = sigma_max / sigma_min
    return {
        f'{prefix}_rank': float((singular_values > rank_tolerance).sum()),
        f'{prefix}_sigma_min': float(sigma_min),
        f'{prefix}_sigma_q25': float(quantiles[0]),
        f'{prefix}_sigma_median': float(quantiles[1]),
        f'{prefix}_sigma_q75': float(quantiles[2]),
        f'{prefix}_sigma_max': float(sigma_max),
        f'{prefix}_condition_number': float(condition_number),
        f'{prefix}_gram_condition_number': float(condition_number.square()),
        f'{prefix}_effective_rank': float(effective_rank),
    }


@torch.no_grad()
def _om_pair_initialization_diagnostics(
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    target_up_singular_values: torch.Tensor,
    target_down_singular_values: torch.Tensor,
    activation_func,
    input_second_moment: float,
    seed: int,
    calibration_size: int,
) -> dict[str, float]:
    """Measure OM-PAIR zero output and per-matrix Standard-spectrum matching."""
    ffn_hidden_size, hidden_size = up_weight.shape
    half_ffn_size = ffn_hidden_size // 2
    pair_scale_inverse = math.sqrt(2.0)
    ga = up_weight[:half_ffn_size] * pair_scale_inverse
    gb = down_weight[:, :half_ffn_size] * pair_scale_inverse

    up_singular_values = torch.linalg.svdvals(ga)
    down_singular_values = torch.linalg.svdvals(gb)
    up_spectrum_match_error = torch.linalg.vector_norm(
        up_singular_values - target_up_singular_values
    ) / torch.linalg.vector_norm(target_up_singular_values)
    down_spectrum_match_error = torch.linalg.vector_norm(
        down_singular_values - target_down_singular_values
    ) / torch.linalg.vector_norm(target_down_singular_values)

    generator = torch.Generator(device='cpu')
    generator.manual_seed(int(seed) + 4)
    inputs = torch.randn(
        hidden_size, calibration_size, generator=generator, dtype=torch.float64
    ) * math.sqrt(input_second_moment)
    directions = torch.randn(
        hidden_size, calibration_size, generator=generator, dtype=torch.float64
    )
    input_norm = torch.linalg.vector_norm(inputs)
    direction_norm = torch.linalg.vector_norm(directions)
    preactivations = up_weight @ inputs
    zero_output = down_weight @ activation_func(preactivations)
    directional_branch_jacobian = down_weight @ (
        _pair_activation_derivative(preactivations, activation_func)
        * (up_weight @ directions)
    )
    bridge_singular_values = torch.linalg.svdvals(0.5 * (gb @ ga))

    metrics = {
        'up_spectrum_match_error': float(up_spectrum_match_error),
        'down_spectrum_match_error': float(down_spectrum_match_error),
        'zero_output_ratio': float(torch.linalg.vector_norm(zero_output) / input_norm),
        'residual_jacobian_directional_error': float(
            torch.linalg.vector_norm(directional_branch_jacobian) / direction_norm
        ),
        'calibration_input_second_moment': float(inputs.square().mean()),
    }
    metrics.update(
        _om_pair_spectrum_metrics('up', up_singular_values, tuple(up_weight.shape))
    )
    metrics.update(
        _om_pair_spectrum_metrics('down', down_singular_values, tuple(down_weight.shape))
    )
    metrics.update(
        _om_pair_spectrum_metrics(
            'bridge', bridge_singular_values, (hidden_size, hidden_size)
        )
    )
    return metrics


@torch.no_grad()
def build_orbit_matched_pair_mlp_weights(
    standard_up_weight: torch.Tensor,
    standard_down_weight: torch.Tensor,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build GELU OM-PAIR weights with each Standard shadow spectrum preserved."""
    standard_up = standard_up_weight.detach().to(device='cpu', dtype=torch.float64)
    standard_down = standard_down_weight.detach().to(device='cpu', dtype=torch.float64)
    ffn_hidden_size, hidden_size = standard_up.shape
    half_ffn_size = ffn_hidden_size // 2

    target_up_singular_values = torch.linalg.svdvals(standard_up)
    target_down_singular_values = torch.linalg.svdvals(standard_down)

    layer_seed = int(seed)
    generator_up_left = torch.Generator(device='cpu')
    generator_up_left.manual_seed(layer_seed)
    generator_up_right = torch.Generator(device='cpu')
    generator_up_right.manual_seed(layer_seed + 1)
    generator_down_left = torch.Generator(device='cpu')
    generator_down_left.manual_seed(layer_seed + 2)
    generator_down_right = torch.Generator(device='cpu')
    generator_down_right.manual_seed(layer_seed + 3)

    up_left_frame = _parseval_frame(half_ffn_size, hidden_size, generator_up_left)
    up_right_factors = _randomized_dct_factors(hidden_size, generator_up_right)
    ga = _apply_randomized_dct(
        up_left_frame * target_up_singular_values, up_right_factors
    )

    down_right_frame = _parseval_frame(
        half_ffn_size, hidden_size, generator_down_right
    )
    down_left_factors = _randomized_dct_factors(hidden_size, generator_down_left)
    gb = _apply_randomized_dct(
        down_right_frame * target_down_singular_values, down_left_factors
    ).transpose(0, 1)

    pair_scale = 1.0 / math.sqrt(2.0)
    up_weight = torch.cat((ga, ga), dim=0) * pair_scale
    down_weight = torch.cat((gb, -gb), dim=1) * pair_scale
    return (
        up_weight.contiguous(),
        down_weight.contiguous(),
        target_up_singular_values,
        target_down_singular_values,
    )


@torch.no_grad()
def _om_skew_pair_initialization_diagnostics(
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    target_up_singular_values: torch.Tensor,
    target_down_singular_values: torch.Tensor,
    down_spectrum_scale: float,
    activation_func,
    input_second_moment: float,
    seed: int,
    calibration_size: int,
) -> dict[str, float]:
    """Measure OM-Skew spectrum matching, metric alignment, and skew bridge."""
    metrics = _om_pair_initialization_diagnostics(
        up_weight=up_weight,
        down_weight=down_weight,
        target_up_singular_values=target_up_singular_values,
        target_down_singular_values=target_down_singular_values,
        activation_func=activation_func,
        input_second_moment=input_second_moment,
        seed=seed,
        calibration_size=calibration_size,
    )

    half_ffn_size = up_weight.shape[0] // 2
    pair_scale_inverse = math.sqrt(2.0)
    ga = up_weight[:half_ffn_size] * pair_scale_inverse
    gb = down_weight[:, :half_ffn_size] * pair_scale_inverse
    up_metric = ga.transpose(0, 1) @ ga
    down_metric = gb @ gb.transpose(0, 1)
    aligned_down_metric = float(down_spectrum_scale) ** 2 * up_metric
    bridge = 0.5 * (gb @ ga)
    down_singular_values = torch.linalg.svdvals(gb)
    shared_down_singular_values = (
        float(down_spectrum_scale) * torch.linalg.svdvals(ga)
    )

    metrics.update(
        {
            'down_spectrum_scale': float(down_spectrum_scale),
            'down_shared_spectrum_match_error': float(
                torch.linalg.vector_norm(
                    down_singular_values - shared_down_singular_values
                )
                / torch.linalg.vector_norm(shared_down_singular_values)
            ),
            'metric_alignment_error': float(
                torch.linalg.vector_norm(down_metric - aligned_down_metric)
                / torch.linalg.vector_norm(aligned_down_metric)
            ),
            'bridge_skew_error': float(
                torch.linalg.vector_norm(bridge + bridge.transpose(0, 1))
                / torch.linalg.vector_norm(bridge)
            ),
        }
    )
    return metrics


@torch.no_grad()
def build_orbit_matched_skew_pair_mlp_weights(
    standard_up_weight: torch.Tensor,
    standard_down_weight: torch.Tensor,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Build GELU OM-Skew PAIR with a shared Standard spectrum and skew bridge."""
    standard_up = standard_up_weight.detach().to(device='cpu', dtype=torch.float64)
    standard_down = standard_down_weight.detach().to(device='cpu', dtype=torch.float64)
    ffn_hidden_size, hidden_size = standard_up.shape
    half_ffn_size = ffn_hidden_size // 2

    if hidden_size % 2 != 0:
        raise ValueError('OM-Skew PAIR requires an even hidden size.')

    target_up_singular_values = torch.linalg.svdvals(standard_up)
    target_down_singular_values = torch.linalg.svdvals(standard_down)
    down_spectrum_scale = float(
        torch.linalg.vector_norm(target_down_singular_values)
        / torch.linalg.vector_norm(target_up_singular_values)
    )

    generator_u = torch.Generator(device='cpu')
    generator_u.manual_seed(int(seed))
    generator_v = torch.Generator(device='cpu')
    generator_v.manual_seed(int(seed) + 1)

    u = _antipodal_parseval_frame(half_ffn_size, hidden_size, generator_u)
    v_factors = _randomized_dct_factors(hidden_size, generator_v)
    ga = _apply_randomized_dct(u * target_up_singular_values, v_factors)

    gb_transpose_input = _right_multiply_reverse_pair_j0_transpose(u)
    gb_transpose_input = gb_transpose_input * target_up_singular_values
    gb = down_spectrum_scale * _apply_randomized_dct(
        gb_transpose_input, v_factors
    ).transpose(0, 1)

    pair_scale = 1.0 / math.sqrt(2.0)
    up_weight = torch.cat((ga, ga), dim=0) * pair_scale
    down_weight = torch.cat((gb, -gb), dim=1) * pair_scale
    return (
        up_weight.contiguous(),
        down_weight.contiguous(),
        target_up_singular_values,
        target_down_singular_values,
        down_spectrum_scale,
    )


def build_pair_mlp_weights(
    hidden_size: int,
    ffn_hidden_size: int,
    activation_func,
    input_second_moment: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the coupled PAIR up and down projection weights in float64 on CPU."""
    if hidden_size < 2 or hidden_size % 2 != 0:
        raise ValueError('PAIR initialization requires an even hidden size of at least two.')
    if ffn_hidden_size % 2 != 0:
        raise ValueError('PAIR initialization requires an even FFN hidden size.')
    if input_second_moment <= 0.0:
        raise ValueError('PAIR input second moment must be positive.')

    d = hidden_size
    m = ffn_hidden_size // 2
    if m < d:
        raise ValueError('PAIR initialization requires half the FFN hidden size to be at least d.')

    gamma = _pair_activation_gamma(activation_func)
    beta = 0.25 * math.log(gamma)
    positions = torch.linspace(-1.0, 1.0, d, dtype=torch.float64)
    mean_squared_scale = 2.0 * m / (d * input_second_moment)
    sigma_center = math.sqrt(mean_squared_scale) / torch.exp(2.0 * beta * positions).mean().sqrt()
    singular_values = sigma_center * torch.exp(beta * positions)

    layer_seed = int(seed)
    generator_v = torch.Generator(device='cpu')
    generator_v.manual_seed(layer_seed)
    generator_u = torch.Generator(device='cpu')
    generator_u.manual_seed(layer_seed + 1)

    v_factors = _randomized_dct_factors(d, generator_v)
    u = _antipodal_parseval_frame(m, d, generator_u)

    ga = _apply_randomized_dct(u * singular_values, v_factors)
    gb_transpose_input = u / singular_values
    gb_transpose_input = _right_multiply_reverse_pair_j0_transpose(gb_transpose_input)
    gb = (2.0 * _apply_randomized_dct(gb_transpose_input, v_factors)).transpose(0, 1)

    pair_scale = 1.0 / math.sqrt(2.0)
    up_weight = torch.cat((ga, ga), dim=0) * pair_scale
    down_weight = torch.cat((gb, -gb), dim=1) * pair_scale
    return up_weight.contiguous(), down_weight.contiguous()


def build_pair_swiglu_mlp_weights(
    hidden_size: int,
    ffn_hidden_size: int,
    input_second_moment: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Build fused gate/up and down PAIR weights for a bias-free SwiGLU MLP."""
    d = hidden_size
    h = ffn_hidden_size
    m = h // 2

    gamma = _pair_activation_gamma(F.silu)
    beta = 0.25 * math.log(gamma)
    positions = torch.linspace(-1.0, 1.0, d, dtype=torch.float64)
    normalized_singular_values = torch.exp(beta * positions)
    normalized_singular_values /= normalized_singular_values.square().mean().sqrt()
    scale_squared = h / (d * input_second_moment)
    scale = math.sqrt(scale_squared)

    layer_seed = int(seed)
    generator_v = torch.Generator(device='cpu')
    generator_v.manual_seed(layer_seed)
    generator_frames = torch.Generator(device='cpu')
    generator_frames.manual_seed(layer_seed + 1)
    generator_signs = torch.Generator(device='cpu')
    generator_signs.manual_seed(layer_seed + 2)

    v_factors = _randomized_dct_factors(d, generator_v)
    gate_frame, up_frame, paired_dimension, defect_dimension = _swiglu_pair_frames(
        m, d, generator_frames
    )
    up_signs = _rademacher_signs(d, generator_signs)

    scaled_spectrum = scale * normalized_singular_values
    gate_base = _apply_randomized_dct(gate_frame * scaled_spectrum, v_factors)
    up_base = _apply_randomized_dct(
        up_frame * scaled_spectrum * up_signs, v_factors
    )
    down_transpose_input = gate_frame / normalized_singular_values
    down_transpose_input = _right_multiply_reverse_pair_j0_transpose(
        down_transpose_input
    )
    down_base = (
        (4.0 / scale_squared)
        * _apply_randomized_dct(down_transpose_input, v_factors)
    ).transpose(0, 1)

    pair_scale = 1.0 / math.sqrt(2.0)
    gate_weight = torch.cat((gate_base, gate_base), dim=0) * pair_scale
    up_weight = torch.cat((up_base, up_base), dim=0) * pair_scale
    fused_fc1_weight = torch.cat((gate_weight, up_weight), dim=0)
    down_weight = torch.cat((down_base, -down_base), dim=1) * pair_scale
    return (
        fused_fc1_weight.contiguous(),
        down_weight.contiguous(),
        paired_dimension,
        defect_dimension,
    )


# pylint: disable=missing-class-docstring
@dataclass
class MLPSubmodules:
    """
    The dataclass for ModuleSpecs of MLP submodules
    including  linear fc1, activation function, linear fc2.
    """

    linear_fc1: Union[ModuleSpec, type] = None
    activation_func: Union[ModuleSpec, type] = None
    linear_fc2: Union[ModuleSpec, type] = None


class MLP(MegatronModule):
    """
    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.


    Returns an output and a bias to be added to the output.
    If config.add_bias_linear is False, the bias returned is None.

    We use the following notation:
     h: hidden size
     p: number of tensor model parallel partitions
     b: batch size
     s: sequence length
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: MLPSubmodules,
        is_expert: bool = False,
        input_size: Optional[int] = None,
        ffn_hidden_size: Optional[int] = None,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        super().__init__(config=config)

        self.config: TransformerConfig = config
        self._pair_diagnostics_collect = False
        self._pair_diagnostics_accumulator = None
        self._pair_initial_metrics: dict[str, float] = {}

        self.input_size = input_size if input_size != None else self.config.hidden_size

        self.tp_group = get_tensor_model_parallel_group_if_none(tp_group, is_expert=is_expert)
        if ffn_hidden_size is None:
            if is_expert:
                raise ValueError("MoE MLP requires `ffn_hidden_size`, but it was not provided.")
            warnings.warn(
                "MLP requires ffn_hidden_size, but it was not provided. Using \
                    config.ffn_hidden_size by default.",
                DeprecationWarning,
                stacklevel=2,
            )
            ffn_hidden_size = self.config.ffn_hidden_size

        # If this is a gated linear unit we double the output width
        # see https://arxiv.org/pdf/2002.05202.pdf
        # For GLU/SwiGLU, use stride=2 because each TP rank stores interleaved [gate, up] portions.
        # This is critical for correct weight resharding across different TP sizes.
        if self.config.gated_linear_unit:
            ffn_hidden_size *= 2
            fc1_stride = 2
        else:
            fc1_stride = 1

        # Use moe_latent_size only for routed experts. 'is_expert' is false for
        # shared_experts.
        use_latent_size = (self.config.moe_latent_size is not None) and is_expert

        self.linear_fc1 = build_module(
            submodules.linear_fc1,
            self.input_size if not use_latent_size else self.config.moe_latent_size,
            ffn_hidden_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_bias_linear,
            skip_bias_add=True,
            is_expert=is_expert,
            tp_comm_buffer_name="fc1",
            tp_group=tp_group,
            stride=fc1_stride,
        )

        if self.config.use_te_activation_func and not (submodules.activation_func is None):
            self.activation_func = build_module(submodules.activation_func, config=self.config)
        else:
            self.activation_func = self.config.activation_func

        self.linear_fc2 = build_module(
            submodules.linear_fc2,
            self.config.ffn_hidden_size,
            self.config.hidden_size if not use_latent_size else self.config.moe_latent_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=self.config.add_bias_linear,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=is_expert,
            tp_comm_buffer_name="fc2",
            tp_group=tp_group,
        )

    def set_layer_number(self, layer_number: int):
        """Record the global layer number and apply the configured coupled initialization."""
        self.layer_number = layer_number
        if not (
            self.config.pair_init
            or self.config.om_pair_init
            or self.config.om_skew_pair_init
        ):
            return

        fc1_weight = getattr(self.linear_fc1, 'weight', None)
        down_weight = getattr(self.linear_fc2, 'weight', None)
        if fc1_weight is None or down_weight is None:
            raise ValueError('PAIR initialization requires explicit fc1 and fc2 weight tensors.')
        if fc1_weight.is_meta or down_weight.is_meta:
            raise ValueError('PAIR initialization requires materialized fc1 and fc2 weights.')
        if not fc1_weight.is_floating_point() or not down_weight.is_floating_point():
            raise ValueError('PAIR initialization requires floating-point fc1 and fc2 weights.')

        fc1_output_size, hidden_size = fc1_weight.shape
        ffn_hidden_size = (
            fc1_output_size // 2 if self.config.gated_linear_unit else fc1_output_size
        )
        if tuple(down_weight.shape) != (hidden_size, ffn_hidden_size):
            raise ValueError(
                'PAIR initialization requires fc1 and fc2 weights with matching FFN dimensions.'
            )

        pair_seed = self.config.pair_init_seed + 1_000_003 * layer_number
        if self.config.om_skew_pair_init:
            (
                pair_fc1,
                pair_down,
                target_up_singular_values,
                target_down_singular_values,
                down_spectrum_scale,
            ) = build_orbit_matched_skew_pair_mlp_weights(
                standard_up_weight=fc1_weight,
                standard_down_weight=down_weight,
                seed=pair_seed,
            )
            if self.config.pair_diagnostics:
                self._pair_initial_metrics = _om_skew_pair_initialization_diagnostics(
                    up_weight=pair_fc1,
                    down_weight=pair_down,
                    target_up_singular_values=target_up_singular_values,
                    target_down_singular_values=target_down_singular_values,
                    down_spectrum_scale=down_spectrum_scale,
                    activation_func=self.config.activation_func,
                    input_second_moment=self.config.pair_init_input_second_moment,
                    seed=pair_seed,
                    calibration_size=self.config.pair_diagnostics_calibration_size,
                )
        elif self.config.om_pair_init:
            (
                pair_fc1,
                pair_down,
                target_up_singular_values,
                target_down_singular_values,
            ) = build_orbit_matched_pair_mlp_weights(
                standard_up_weight=fc1_weight,
                standard_down_weight=down_weight,
                seed=pair_seed,
            )
            if self.config.pair_diagnostics:
                self._pair_initial_metrics = _om_pair_initialization_diagnostics(
                    up_weight=pair_fc1,
                    down_weight=pair_down,
                    target_up_singular_values=target_up_singular_values,
                    target_down_singular_values=target_down_singular_values,
                    activation_func=self.config.activation_func,
                    input_second_moment=self.config.pair_init_input_second_moment,
                    seed=pair_seed,
                    calibration_size=self.config.pair_diagnostics_calibration_size,
                )
        elif self.config.gated_linear_unit:
            pair_fc1, pair_down, paired_dimension, defect_dimension = (
                build_pair_swiglu_mlp_weights(
                    hidden_size=hidden_size,
                    ffn_hidden_size=ffn_hidden_size,
                    input_second_moment=self.config.pair_init_input_second_moment,
                    seed=pair_seed,
                )
            )
            if self.config.pair_diagnostics:
                self._pair_initial_metrics = _pair_swiglu_initialization_diagnostics(
                    fused_fc1_weight=pair_fc1,
                    down_weight=pair_down,
                    input_second_moment=self.config.pair_init_input_second_moment,
                    seed=pair_seed,
                    calibration_size=self.config.pair_diagnostics_calibration_size,
                    paired_dimension=paired_dimension,
                    defect_dimension=defect_dimension,
                )
        else:
            pair_fc1, pair_down = build_pair_mlp_weights(
                hidden_size=hidden_size,
                ffn_hidden_size=ffn_hidden_size,
                activation_func=self.config.activation_func,
                input_second_moment=self.config.pair_init_input_second_moment,
                seed=pair_seed,
            )
            if self.config.pair_diagnostics:
                self._pair_initial_metrics = _pair_initialization_diagnostics(
                    up_weight=pair_fc1,
                    down_weight=pair_down,
                    activation_func=self.config.activation_func,
                    input_second_moment=self.config.pair_init_input_second_moment,
                    seed=pair_seed,
                    calibration_size=self.config.pair_diagnostics_calibration_size,
                )

        with torch.no_grad():
            fc1_weight.copy_(pair_fc1.to(device=fc1_weight.device, dtype=fc1_weight.dtype))
            down_weight.copy_(pair_down.to(device=down_weight.device, dtype=down_weight.dtype))
            fc1_bias = getattr(self.linear_fc1, 'bias', None)
            down_bias = getattr(self.linear_fc2, 'bias', None)
            if fc1_bias is not None:
                fc1_bias.zero_()
            if down_bias is not None:
                down_bias.zero_()

    def set_pair_diagnostics_collection(self, enabled: bool) -> None:
        """Enable one bounded collection window for PAIR activation statistics."""
        if enabled and not (
            self.config.pair_init
            or self.config.om_pair_init
            or self.config.om_skew_pair_init
        ):
            raise ValueError('PAIR diagnostics require PAIR-initialized MLP layers.')
        self._pair_diagnostics_collect = enabled
        self._pair_diagnostics_accumulator = None

    @torch.no_grad()
    def _accumulate_pair_diagnostics(
        self,
        hidden_states: torch.Tensor,
        activated_hidden_states: torch.Tensor,
        output: torch.Tensor,
        output_bias: Optional[torch.Tensor],
    ) -> None:
        if not self._pair_diagnostics_collect:
            return
        if activated_hidden_states.shape[-1] % 2 != 0:
            raise ValueError('PAIR parity diagnostics require an even activation width.')

        inputs = hidden_states.detach().float()
        activated = activated_hidden_states.detach().float()
        first_half, second_half = torch.chunk(activated, 2, dim=-1)
        even_hidden = (first_half + second_half) / math.sqrt(2.0)
        odd_hidden = (first_half - second_half) / math.sqrt(2.0)

        diagnostic_output = output.detach().float()
        if output_bias is not None:
            diagnostic_output = diagnostic_output + output_bias.detach().float()

        stats = torch.stack(
            (
                inputs.square().sum(),
                inputs.new_tensor(float(inputs.numel())),
                even_hidden.square().sum(),
                odd_hidden.square().sum(),
                inputs.new_tensor(float(first_half.numel())),
                diagnostic_output.square().sum(),
                inputs.new_tensor(float(diagnostic_output.numel())),
            )
        )
        if self._pair_diagnostics_accumulator is None:
            self._pair_diagnostics_accumulator = stats
        else:
            self._pair_diagnostics_accumulator.add_(stats)

    def consume_pair_diagnostics(self) -> torch.Tensor:
        """Return and clear the activation accumulator for the current collection window."""
        if self._pair_diagnostics_accumulator is None:
            raise RuntimeError('PAIR diagnostics were enabled but the MLP did not execute.')
        accumulator = self._pair_diagnostics_accumulator
        self._pair_diagnostics_accumulator = None
        self._pair_diagnostics_collect = False
        return accumulator

    def consume_pair_initial_metrics(self) -> dict[str, float]:
        """Return the construction diagnostics once without adding checkpoint state."""
        metrics = self._pair_initial_metrics
        self._pair_initial_metrics = {}
        return metrics

    def forward(self, hidden_states, per_token_scale=None):
        """Perform the forward pass through the MLP block."""
        # [s, b, 4 * h/p]
        nvtx_range_push(suffix="linear_fc1")
        intermediate_parallel, bias_parallel = self.linear_fc1(hidden_states)
        nvtx_range_pop(suffix="linear_fc1")

        nvtx_range_push(suffix="activation")
        if self.config.use_te_activation_func:
            if bias_parallel is not None:
                intermediate_parallel = intermediate_parallel + bias_parallel
            intermediate_parallel = self.activation_func(intermediate_parallel)
            if per_token_scale is not None:
                original_dtype = intermediate_parallel.dtype
                intermediate_parallel = intermediate_parallel * per_token_scale.unsqueeze(-1)
                intermediate_parallel = intermediate_parallel.to(original_dtype)
        elif self.config.bias_activation_fusion:
            if per_token_scale is not None:
                if self.activation_func == F.silu and self.config.gated_linear_unit:
                    # dtype is handled inside the fused kernel
                    intermediate_parallel = weighted_bias_swiglu_impl(
                        intermediate_parallel,
                        bias_parallel,
                        per_token_scale.unsqueeze(-1),
                        self.config.activation_func_fp8_input_store,
                    )
                elif self.activation_func == quick_gelu and self.config.gated_linear_unit:
                    intermediate_parallel = weighted_bias_quick_geglu_impl(
                        intermediate_parallel,
                        bias_parallel,
                        per_token_scale.unsqueeze(-1),
                        self.config.activation_func_fp8_input_store,
                        self.config.glu_linear_offset,
                        self.config.activation_func_clamp_value,
                    )
                else:
                    raise ValueError(
                        "Only support fusion of swiglu and quick_gelu with per_token_scale in MLP."
                    )
            else:
                if self.activation_func == F.gelu:
                    if self.config.gated_linear_unit:
                        intermediate_parallel = bias_geglu_impl(
                            intermediate_parallel, bias_parallel
                        )
                    else:
                        assert self.config.add_bias_linear is True
                        intermediate_parallel = bias_gelu_impl(intermediate_parallel, bias_parallel)
                elif self.activation_func == F.silu and self.config.gated_linear_unit:
                    intermediate_parallel = bias_swiglu_impl(
                        intermediate_parallel,
                        bias_parallel,
                        self.config.activation_func_fp8_input_store,
                        self.config.cpu_offloading
                        and self.config.cpu_offloading_activations
                        and HAVE_TE,
                    )
                else:
                    raise ValueError("Only support fusion of gelu and swiglu")
        else:
            if bias_parallel is not None:
                intermediate_parallel = intermediate_parallel + bias_parallel
            if self.config.gated_linear_unit:

                def glu(x):
                    x_glu, x_linear = torch.chunk(x, 2, dim=-1)
                    if (val := self.config.activation_func_clamp_value) is not None:
                        x_glu = x_glu.clamp(min=None, max=val)
                        x_linear = x_linear.clamp(min=-val, max=val)
                    return self.config.activation_func(x_glu) * (
                        x_linear + self.config.glu_linear_offset
                    )

                intermediate_parallel = glu(intermediate_parallel)
            else:
                intermediate_parallel = self.activation_func(intermediate_parallel)

            if per_token_scale is not None:
                original_dtype = intermediate_parallel.dtype
                intermediate_parallel = intermediate_parallel * per_token_scale.unsqueeze(-1)
                intermediate_parallel = intermediate_parallel.to(original_dtype)
        nvtx_range_pop(suffix="activation")

        # [s, b, h]
        nvtx_range_push(suffix="linear_fc2")

        output, output_bias = self.linear_fc2(intermediate_parallel)
        nvtx_range_pop(suffix="linear_fc2")

        self._accumulate_pair_diagnostics(
            hidden_states, intermediate_parallel, output, output_bias
        )

        if per_token_scale is not None and output_bias is not None:
            # if this MLP is an expert, and bias is required, we add the bias to output directly
            # without doing bda later.
            output += output_bias.unsqueeze(0) * per_token_scale.unsqueeze(-1)
            output_bias = None

        return output, output_bias

    # pylint: disable=missing-function-docstring
    def sharded_state_dict(
        self, prefix: str = "", sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ) -> ShardedStateDict:
        """Return the sharded state dictionary of the module."""
        sharded_state_dict = {}
        singleton_local_shards = (metadata or {}).get('singleton_local_shards', False)
        for name, module in self._modules.items():
            sub_sd = module.sharded_state_dict(f"{prefix}{name}.", sharded_offsets, metadata)
            if self.config.gated_linear_unit and name == "linear_fc1":
                for k, v in sub_sd.items():
                    if k in (f"{prefix}{name}.weight", f"{prefix}{name}.bias"):
                        sub_sd[k] = apply_swiglu_sharded_factory(
                            v, sharded_offsets, singleton_local_shards
                        )
            sharded_state_dict.update(sub_sd)
        return sharded_state_dict

    def backward_dw(self):
        self.linear_fc2.backward_dw()
        self.linear_fc1.backward_dw()


# pylint: disable=missing-function-docstring
def apply_swiglu_sharded_factory(
    original_sh_ten, sharded_offsets, singleton_local_shards: bool = False
):
    # We must split the tensor into 2 parts, each sharded separately.
    # This requires a ShardedTensorFactory which `chunk`s during saving
    # and `cat`s during loading

    swiglu_shard_axis = 0
    prepend_axis_num = len(sharded_offsets)
    original_shape = original_sh_ten.local_shape
    original_numel = int(np.prod(original_shape))
    local_axis_size = original_shape[swiglu_shard_axis]
    assert (
        original_sh_ten.global_offset[swiglu_shard_axis + prepend_axis_num] % local_axis_size == 0
    )
    rank_offset = (
        original_sh_ten.global_offset[swiglu_shard_axis + prepend_axis_num] // local_axis_size
    )
    axis_frag = original_sh_ten.axis_fragmentations[swiglu_shard_axis + prepend_axis_num]

    @torch.no_grad()
    def sh_ten_build_fn(
        key: str, t: torch.Tensor, replica_id: ReplicaId, flattened_range: Optional[slice]
    ):
        if singleton_local_shards:
            offset_w = (swiglu_shard_axis + prepend_axis_num, rank_offset, axis_frag)
            offset_v = (swiglu_shard_axis + prepend_axis_num, rank_offset, axis_frag)
            w_key = f'{key}_w'
            v_key = f'{key}_v'
        else:
            offset_w = (swiglu_shard_axis + prepend_axis_num, rank_offset, axis_frag * 2)
            offset_v = (
                swiglu_shard_axis + prepend_axis_num,
                rank_offset + axis_frag,
                axis_frag * 2,
            )
            w_key = key
            v_key = key

        tensor_w, tensor_v = torch.chunk(t, 2, dim=swiglu_shard_axis)
        return [
            ShardedTensor.from_rank_offsets(
                w_key,
                tensor_w,
                *sharded_offsets,
                offset_w,
                replica_id=replica_id,
                prepend_axis_num=prepend_axis_num,
            ),
            ShardedTensor.from_rank_offsets(
                v_key,
                tensor_v,
                *sharded_offsets,
                offset_v,
                replica_id=replica_id,
                prepend_axis_num=prepend_axis_num,
            ),
        ]

    def sh_ten_merge_fn(sub_state_dict):
        with torch.no_grad():
            try:
                return torch.cat(sub_state_dict)
            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                logger.warning(
                    f"CUDA OutOfMemoryError encountered during tensors merging."
                    f" Switching to CPU merge. (Error: {e})"
                )
                merged_sub_state_dict = torch.cat([t.cpu() for t in sub_state_dict])
                gc.collect()
                torch.cuda.empty_cache()
                return merged_sub_state_dict

    return ShardedTensorFactory(
        original_sh_ten.key,
        original_sh_ten.data,
        sh_ten_build_fn,
        sh_ten_merge_fn,
        original_sh_ten.replica_id,
        flattened_range=original_sh_ten.flattened_range,
    )
