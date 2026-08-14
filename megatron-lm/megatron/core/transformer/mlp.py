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
        if not self.config.pair_init:
            return
        if self.config.gated_linear_unit:
            raise ValueError('PAIR initialization requires a non-gated dense MLP.')

        up_weight = getattr(self.linear_fc1, 'weight', None)
        down_weight = getattr(self.linear_fc2, 'weight', None)
        if up_weight is None or down_weight is None:
            raise ValueError('PAIR initialization requires explicit fc1 and fc2 weight tensors.')
        if up_weight.is_meta or down_weight.is_meta:
            raise ValueError('PAIR initialization requires materialized fc1 and fc2 weights.')
        if not up_weight.is_floating_point() or not down_weight.is_floating_point():
            raise ValueError('PAIR initialization requires floating-point fc1 and fc2 weights.')

        ffn_hidden_size, hidden_size = up_weight.shape
        if tuple(down_weight.shape) != (hidden_size, ffn_hidden_size):
            raise ValueError(
                'PAIR initialization requires fc1 [2m, d] and fc2 [d, 2m] with matching dimensions.'
            )

        pair_seed = self.config.pair_init_seed + 1_000_003 * layer_number
        pair_up, pair_down = build_pair_mlp_weights(
            hidden_size=hidden_size,
            ffn_hidden_size=ffn_hidden_size,
            activation_func=self.config.activation_func,
            input_second_moment=self.config.pair_init_input_second_moment,
            seed=pair_seed,
        )

        with torch.no_grad():
            up_weight.copy_(pair_up.to(device=up_weight.device, dtype=up_weight.dtype))
            down_weight.copy_(pair_down.to(device=down_weight.device, dtype=down_weight.dtype))
            up_bias = getattr(self.linear_fc1, 'bias', None)
            down_bias = getattr(self.linear_fc2, 'bias', None)
            if up_bias is not None:
                up_bias.zero_()
            if down_bias is not None:
                down_bias.zero_()

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
