# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Bounded runtime diagnostics for PAIR-initialized dense MLP layers."""

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch

from megatron.core import mpu
from megatron.core.transformer.mlp import MLP
from megatron.core.utils import unwrap_model


@dataclass
class _PairParameterSnapshot:
    metric_prefix: str
    parameter: torch.nn.Parameter
    row_start: int
    row_end: int
    before: torch.Tensor


def pair_diagnostics_due(args) -> bool:
    """Return whether the upcoming optimizer step is a configured diagnostic step."""
    if not args.pair_diagnostics:
        return False
    step = args.curr_iteration + 1
    return step in args.pair_diagnostics_steps or step % args.pair_diagnostics_interval == 0


def _pair_mlps(model) -> List[MLP]:
    modules: List[MLP] = []
    for model_chunk in unwrap_model(model):
        for module in model_chunk.modules():
            if isinstance(module, MLP) and module.config.pair_init:
                modules.append(module)
    modules.sort(key=lambda module: module.layer_number)
    layer_numbers = [module.layer_number for module in modules]
    if not modules:
        raise RuntimeError('PAIR diagnostics found no PAIR-initialized MLP layers.')
    if len(layer_numbers) != len(set(layer_numbers)):
        raise RuntimeError('PAIR diagnostics require one dense MLP for each layer number.')
    return modules


def begin_pair_diagnostics(model) -> List[MLP]:
    """Start activation accumulation for one selected training step."""
    modules = _pair_mlps(model)
    for module in modules:
        module.set_pair_diagnostics_collection(True)
    return modules


def _main_parameter(parameter: torch.nn.Parameter) -> torch.Tensor:
    main_parameter = getattr(parameter, 'main_param', None)
    if main_parameter is not None:
        return main_parameter
    if parameter.dtype == torch.float32:
        return parameter
    raise RuntimeError(
        'PAIR diagnostics require the Megatron main parameter for reduced-precision weights.'
    )


def _main_gradient(parameter: torch.nn.Parameter) -> torch.Tensor:
    gradient = getattr(parameter, 'main_grad', None)
    if gradient is None:
        raise RuntimeError('PAIR diagnostics require the finalized Megatron main gradient.')
    return gradient


def _activation_metrics(module: MLP, stats: torch.Tensor) -> Dict[str, float]:
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(
            stats,
            op=torch.distributed.ReduceOp.SUM,
            group=mpu.get_data_parallel_group(with_context_parallel=True),
        )

    (
        input_sum_squared,
        input_count,
        even_sum_squared,
        odd_sum_squared,
        hidden_count,
        output_sum_squared,
        output_count,
    ) = stats
    tiny = torch.finfo(stats.dtype).tiny
    hidden_energy = (even_sum_squared + odd_sum_squared).clamp_min(tiny)
    prefix = f'pair/layer_{module.layer_number:02d}/runtime'
    return {
        f'{prefix}/input_second_moment': float(input_sum_squared / input_count),
        f'{prefix}/hidden_even_second_moment': float(even_sum_squared / hidden_count),
        f'{prefix}/hidden_odd_second_moment': float(odd_sum_squared / hidden_count),
        f'{prefix}/hidden_even_energy_fraction': float(even_sum_squared / hidden_energy),
        f'{prefix}/hidden_odd_energy_fraction': float(odd_sum_squared / hidden_energy),
        f'{prefix}/ffn_output_second_moment': float(output_sum_squared / output_count),
        f'{prefix}/ffn_output_input_rms_ratio': float(
            torch.sqrt(output_sum_squared / input_sum_squared.clamp_min(tiny))
        ),
    }


@torch.no_grad()
def capture_pair_diagnostics(
    modules: Sequence[MLP], include_initial_metrics: bool
) -> Tuple[Dict[str, float], List[_PairParameterSnapshot]]:
    """Capture activation and pre-clip gradient metrics immediately before the update."""
    metrics: Dict[str, float] = {}
    snapshots: List[_PairParameterSnapshot] = []

    for module in modules:
        layer_prefix = f'pair/layer_{module.layer_number:02d}'
        activation_stats = module.consume_pair_diagnostics()
        metrics.update(_activation_metrics(module, activation_stats))

        initial_metrics = module.consume_pair_initial_metrics()
        if include_initial_metrics:
            metrics.update(
                {
                    f'{layer_prefix}/init/{name}': value
                    for name, value in initial_metrics.items()
                }
            )

        fc1_parameter = module.linear_fc1.weight
        down_parameter = module.linear_fc2.weight
        if module.config.gated_linear_unit:
            gate_row_end = fc1_parameter.shape[0] // 2
            projection_slices = (
                ('gate', fc1_parameter, 0, gate_row_end),
                ('up', fc1_parameter, gate_row_end, fc1_parameter.shape[0]),
                ('down', down_parameter, 0, down_parameter.shape[0]),
            )
        else:
            projection_slices = (
                ('fc1', fc1_parameter, 0, fc1_parameter.shape[0]),
                ('fc2', down_parameter, 0, down_parameter.shape[0]),
            )

        for projection_name, parameter, row_start, row_end in projection_slices:
            main_parameter = _main_parameter(parameter)[row_start:row_end]
            main_gradient = _main_gradient(parameter)[row_start:row_end]
            metric_prefix = f'{layer_prefix}/{projection_name}'
            gradient_float = main_gradient.detach().float()
            parameter_float = main_parameter.detach().float()
            metrics[f'{metric_prefix}/grad_norm_preclip'] = float(
                torch.linalg.vector_norm(gradient_float)
            )
            metrics[f'{metric_prefix}/grad_rms_preclip'] = float(
                torch.sqrt(gradient_float.square().mean())
            )
            metrics[f'{metric_prefix}/weight_norm_preupdate'] = float(
                torch.linalg.vector_norm(parameter_float)
            )
            snapshots.append(
                _PairParameterSnapshot(
                    metric_prefix=metric_prefix,
                    parameter=parameter,
                    row_start=row_start,
                    row_end=row_end,
                    before=main_parameter.detach().clone(),
                )
            )

    return metrics, snapshots


@torch.no_grad()
def finish_pair_diagnostics(
    snapshots: Sequence[_PairParameterSnapshot], update_successful: bool
) -> Dict[str, float]:
    """Measure the exact main-parameter update after the optimizer step."""
    metrics: Dict[str, float] = {
        'pair/optimizer_update_applied': float(bool(update_successful))
    }
    for snapshot in snapshots:
        current = _main_parameter(snapshot.parameter)[
            snapshot.row_start:snapshot.row_end
        ].detach().float()
        before = snapshot.before.float()
        update = current - before
        weight_norm = torch.linalg.vector_norm(before)
        update_norm = torch.linalg.vector_norm(update)
        denominator = weight_norm.clamp_min(torch.finfo(weight_norm.dtype).tiny)
        metrics[f'{snapshot.metric_prefix}/update_norm'] = float(update_norm)
        metrics[f'{snapshot.metric_prefix}/update_rms'] = float(
            torch.sqrt(update.square().mean())
        )
        metrics[f'{snapshot.metric_prefix}/relative_update'] = float(
            update_norm / denominator
        )
    return metrics
