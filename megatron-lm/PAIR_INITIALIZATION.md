# PAIR 初始化使用说明

PAIR 初始化用于联合构造每个稠密 MLP 的输入投影 `linear_fc1` 和输出投影 `linear_fc2`。该功能只改变模型构造时的初始权重，不改变 `GPTModel` 的前向结构，也不改变 Pion 优化器的更新规则。

## 代码入口

| 文件 | 职责 |
| --- | --- |
| `megatron/core/transformer/mlp.py` | 构造 PAIR 权重，并写入每层 MLP 的 `linear_fc1` 与 `linear_fc2` |
| `megatron/core/transformer/transformer_config.py` | 保存 PAIR 开关、输入二阶矩和基础随机种子 |
| `megatron/training/arguments.py` | 定义命令行参数、检查支持范围，并把全局 `--seed` 传入 Transformer 配置 |

调用链如下：

```text
pretrain_gpt.py
  -> megatron.training.arguments
  -> TransformerConfig
  -> MLP.set_layer_number
  -> build_pair_mlp_weights
  -> linear_fc1.weight 与 linear_fc2.weight
```

## 启用方式

在现有 Megatron 启动参数中加入：

```bash
PAIR_ARGS=(
    --pair-init
    --pair-init-input-second-moment 1.0
    --no-bias-gelu-fusion
    --seed 1234
)
```

将 `${PAIR_ARGS[@]}` 传给 `pretrain_gpt.py`：

```bash
torchrun --nproc_per_node=1 pretrain_gpt.py \
    "${MODEL_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" \
    "${PARALLEL_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${PAIR_ARGS[@]}"
```

`--pair-init-input-second-moment` 表示进入 MLP 时单个坐标的二阶矩。当前实验使用 `1.0`。该值必须为正数。

`--seed` 同时控制 Megatron 的全局随机种子和 PAIR 的基础随机种子。第 `L` 层使用：

```text
pair_seed = seed + 1_000_003 * L
```

相同配置和相同随机种子会生成相同的逐层 PAIR 权重；不同层使用不同的随机正交方向。

## 60M 模型配置

原始 60M Pion 启动脚本使用 SwiGLU 和 `ffn_hidden_size=1376`，与当前 PAIR 入口的形状和激活函数要求不同。启用 PAIR 时采用以下模型参数：

```bash
MODEL_ARGS=(
    --normalization RMSNorm
    --pair-init
    --pair-init-input-second-moment 1.0
    --num-layers 8
    --hidden-size 512
    --ffn-hidden-size 2048
    --num-attention-heads 8
    --kv-channels 64
    --no-bias-gelu-fusion
)
```

GELU 是 Megatron 在未指定 `--swiglu`、`--squared-relu` 或 `--quick-geglu` 时使用的默认激活函数。PAIR 命令行入口当前要求标准 GELU，因此参数中不得包含这三个激活函数选项。

单卡配置需要满足：

```bash
PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
)
```

`--nproc_per_node`、可见 GPU 数量和数据并行规模应保持一致。单卡启动时三者均为 `1`。

## 支持范围

| 项目 | 要求 |
| --- | --- |
| 模型实现 | Megatron Core 模型 |
| MLP 类型 | 稠密、非门控 MLP |
| 激活函数 | 命令行训练入口使用标准 GELU |
| Tensor Parallel | `--tensor-model-parallel-size 1` |
| `hidden_size` | 大于等于 2 的偶数 |
| `ffn_hidden_size` | 偶数，并且大于等于 `2 * hidden_size` |
| 参数设备 | 模型构造时必须是实际张量，不能使用 meta device |
| 谱初始化 | 不能同时设置 `--init-spectral-norm-scale` |
| MoE | 当前入口只支持稠密 MLP |

PAIR 在 CPU 上以 `float64` 构造权重，然后转换到目标参数的设备和 dtype。存在 MLP bias 时，`linear_fc1` 和 `linear_fc2` 的 bias 会被置零。

初始化覆盖范围只包括 Transformer 层内的两个 MLP 投影。Attention、Embedding、Normalization 和输出层仍使用各自原有的初始化方法。

## 与 Pion 优化器配合

PAIR 和 Pion 承担不同阶段的工作：PAIR 决定训练开始前的 MLP 权重，Pion 决定训练过程中的二维权重更新。启用 Pion 时继续保留原有优化器参数：

```bash
--optimizer pion
```

Pion 的参数路由保持不变。二维非嵌入权重进入 Pion，其余参数进入 AdamW。

## 检查点语义

PAIR 在模型构造阶段写入初始权重。新训练会从这些权重开始；续训时，检查点加载会用检查点中的参数覆盖构造阶段的初始权重。因此，`--pair-init` 不会重新初始化已经从检查点恢复的模型。

启动新实验时，应使用新的检查点目录，并在实验名称中记录 `pair`、激活函数、输入二阶矩和随机种子。
