# PAIR 初始化使用说明

PAIR 初始化用于联合构造稠密 MLP 的权重。标准 GELU MLP 联合构造 `linear_fc1` 与 `linear_fc2`；无偏置 SwiGLU MLP 联合构造 gate、up、down 三个逻辑矩阵，其中 gate 与 up 融合存储在 `linear_fc1.weight`，down 存储在 `linear_fc2.weight`。

该功能在模型构造阶段覆盖 MLP 的初始权重，不改变 `GPTModel` 的前向结构。训练期仍由选定的优化器更新参数；启用 Pion 时，融合 FC1 中的 gate 与 up 分别维护 Pion 状态。

## 代码入口

| 文件 | 职责 |
| --- | --- |
| `megatron/core/transformer/mlp.py` | 构造 GELU 或 SwiGLU 的 PAIR 权重，写入每层 MLP，并计算初始化与激活统计 |
| `megatron/core/transformer/transformer_config.py` | 保存 PAIR 开关、输入二阶矩、基础随机种子和诊断配置 |
| `megatron/core/optimizer/pion.py` | 将融合 FC1 按 `[gate; up]` 分割并分别执行 Pion 更新 |
| `megatron/core/optimizer/pion_msign.py` | 在 Pion msign 变体中使用相同的 gate/up 布局 |
| `megatron/core/tensor_parallel/layers.py` | 传播 `is_fc1_gate_up` 参数属性 |
| `megatron/training/arguments.py` | 定义命令行参数并检查支持范围 |
| `megatron/training/pair_diagnostics.py` | 汇总逐层激活、梯度和参数更新指标 |
| `megatron/training/training.py` | 在训练步骤中触发诊断，并记录 NLL、PPL、泛化差距和 token 进度 |
| `opt_llama_60M_pion.sh` | 60M SwiGLU-PAIR 预训练入口 |

初始化调用链如下：

```text
pretrain_gpt.py
  -> megatron.training.arguments
  -> TransformerConfig
  -> MLP.set_layer_number
  -> build_pair_mlp_weights
     或 build_pair_swiglu_mlp_weights
  -> linear_fc1.weight 与 linear_fc2.weight
```

诊断调用链如下：

```text
MLP.forward
  -> 激活奇偶子空间统计
  -> capture_pair_diagnostics
  -> optimizer.step
  -> finish_pair_diagnostics
  -> TensorBoard、WandB 与 CSV
```

## 通用启用参数

```bash
PAIR_ARGS=(
    --pair-init
    --pair-init-input-second-moment 1.0
    --pair-diagnostics
    --pair-diagnostics-interval 1000
    --pair-diagnostics-steps 1 10 100
    --pair-diagnostics-calibration-size 16
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

`--pair-init-input-second-moment` 表示进入 MLP 时单个坐标的二阶矩。当前实验使用 `1.0`。

`--seed` 同时控制 Megatron 的全局随机种子和 PAIR 的基础随机种子。第 `L` 层使用：

```text
pair_seed = seed + 1_000_003 * L
```

相同配置和随机种子生成相同的逐层 PAIR 权重，不同层使用不同的随机正交方向。

## GELU 配置

标准 GELU MLP 使用两个物理矩阵：

```bash
MODEL_ARGS=(
    --pair-init
    --pair-init-input-second-moment 1.0
    --hidden-size 512
    --ffn-hidden-size 2048
    --disable-bias-linear
    --no-bias-gelu-fusion
)
```

当 `ffn_hidden_size=2m` 时，GELU-PAIR 构造输入投影的相同通道对和输出投影的相反符号通道对，使初始 MLP 分支输出为零。

## SwiGLU 配置

Megatron 的 SwiGLU 前向过程为：

```text
down(SiLU(gate(x)) * up(x))
```

融合 `linear_fc1.weight` 的前一半是 gate，后一半是 up。PAIR 使用以下外层布局：

```text
W_gate = [G_gate;  G_gate] / sqrt(2)
W_up   = [G_up;    G_up]   / sqrt(2)
W_down = [G_down, -G_down] / sqrt(2)
```

底层矩阵为：

```text
G_gate = a U_- Sigma V^T
G_up   = a U_+ Sigma D V^T
G_down = (4 / a^2) V J_0 Sigma^-1 U_-^T
```

其中 `D` 是 Rademacher 对角阵，当前整体增益 `rho=1`，尺度满足：

```text
a^2 = h / (d * input_second_moment)
```

记残差维度为 `d`，SwiGLU intermediate width 为 `h`，内部 frame 行数为 `m=h/2`。构造器根据 `m` 与 `2d` 的关系选择内部公式：

| 条件 | 构造 | 配对维数 | 缺陷维数 |
| --- | --- | ---: | ---: |
| `m >= 2d` | 完整 frame | `d` | `0` |
| `d <= m < 2d` | 最大配对 frame | `m-d` | `2d-m` |

两个构造都精确保持初始零输出、gate/up/down 满秩和三矩阵 metric matching。最大配对构造中的有限角度三路函数路径只在配对子空间完全一致，其余差异位于显式的缺陷子空间。

## 60M SwiGLU-PAIR 配置

当前 60M 入口保留原始 Pion 模型的参数匹配宽度：

```bash
MODEL_ARGS=(
    --normalization RMSNorm
    --pair-init
    --pair-init-input-second-moment 1.0
    --pair-diagnostics
    --pair-diagnostics-interval 1000
    --pair-diagnostics-steps 1 10 100
    --pair-diagnostics-calibration-size 16
    --num-layers 8
    --hidden-size 512
    --ffn-hidden-size 1376
    --num-attention-heads 8
    --kv-channels 64
    --swiglu
    --disable-bias-linear
)
```

该配置对应：

```text
d = 512
h = 1376
m = 688
paired_dimension = m - d = 176
defect_dimension = 2d - m = 336
```

因此 60M 模型使用最大配对 frame。融合 FC1 的物理形状为 `[2752, 512]`，down 权重形状为 `[512, 1376]`。

单卡并行参数为：

```bash
PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
)
```

当前实现要求 tensor parallel 为 `1`，从而保证外层正负通道对位于同一个 rank。单卡启动时 `--nproc_per_node` 也设为 `1`。

## 支持范围

| 项目 | 要求 |
| --- | --- |
| 模型实现 | Megatron Core 模型 |
| MLP 类型 | 稠密 GELU 或无偏置 SwiGLU |
| 激活函数 | 标准 GELU 或 `--swiglu` 对应的 SiLU |
| Tensor Parallel | `--tensor-model-parallel-size 1` |
| `hidden_size` | 大于等于 2 的偶数 |
| GELU `ffn_hidden_size` | 偶数，并且大于等于 `2 * hidden_size` |
| SwiGLU `ffn_hidden_size` | 偶数，并且大于等于 `2 * hidden_size` |
| 参数设备 | 模型构造时使用实际张量 |
| 谱初始化 | 不与 `--init-spectral-norm-scale` 同时启用 |
| MoE | 使用稠密 MLP |

PAIR 在 CPU 上以 `float64` 构造权重，再转换到目标参数的设备和 dtype。精确的无偏置构造使用 `--disable-bias-linear`。

初始化覆盖范围只包括 Transformer 层内的 MLP 投影。Attention、Embedding、Normalization 和输出层继续使用各自的初始化方法。

## 诊断指标

启用 `--pair-diagnostics` 后，初始化阶段记录：

| 指标组 | 内容 |
| --- | --- |
| 秩与谱 | gate、up、down 的秩和最小、最大奇异值 |
| 零输出 | 初始 MLP 输出与输入范数之比 |
| Metric matching | gate/up Gram 误差和 down 对应 Gram 误差 |
| 内部 frame | `paired_dimension` 与 `defect_dimension` |
| GELU 结构 | 反向谱误差、残差 Jacobian 方向误差和规范映射误差 |

在第 `1`、`10`、`100` 步以及之后每隔 `1000` 步记录：

| 指标组 | 内容 |
| --- | --- |
| 激活 | 输入二阶矩、隐藏状态偶子空间与奇子空间能量、MLP 输出二阶矩 |
| 梯度 | gate、up、down 的裁剪前梯度范数和 RMS |
| 更新 | gate、up、down 的更新范数、更新 RMS 和相对更新量 |
| 训练 | NLL、PPL、梯度范数、学习率、已消费 token 和每卡 token 吞吐量 |
| 泛化 | 验证 NLL、验证 PPL、训练窗口 NLL 和验证减训练 NLL 差值 |

记录位置如下：

```text
TensorBoard: --tensorboard-dir
PAIR CSV:    <save>/pair_diagnostics.csv
泛化 CSV:    <save>/generalization_metrics.csv
梯度 CSV:    <save>/grad_norm.csv
```

## 与 Pion 优化器配合

PAIR 决定训练开始前的 MLP 权重，Pion 决定训练过程中的二维权重更新。启用方式为：

```bash
--optimizer pion
```

二维非嵌入权重进入 Pion，其余参数进入 AdamW。SwiGLU 的融合 FC1 通过 `is_fc1_gate_up` 标记为 `[gate; up]` 布局，Pion 分别建立 `fc1_gate` 与 `fc1_up` 状态，更新后再按原行序重组。

## 检查点语义

PAIR 在模型构造阶段写入初始权重。新训练从这些权重开始；续训时，检查点中的模型参数和优化器状态覆盖构造阶段产生的初始状态。

启动新实验时使用新的检查点目录，并在实验名称中记录 `pair`、激活函数、输入二阶矩和随机种子。

## 启动入口

```bash
cd megatron-lm
bash opt_llama_60M_pion.sh
```

启动脚本依次执行 `SEEDS` 数组中的实验，每个随机种子使用独立的日志、TensorBoard 和检查点目录。
