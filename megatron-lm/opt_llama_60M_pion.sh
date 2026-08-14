#!/bin/bash
set -o pipefail
export CUDA_DEVICE_MAX_CONNECTIONS=1
export WANDB_SILENT=true
export TORCH_CPP_LOG_LEVEL=ERROR
export NCCL_ML_DISABLE=1
export NCCL_NVLS_ENABLE=1
export WANDB_MODE=offline
export CUDNN_FRONTEND_CUDART_LIB_NAME=libcudart.so.13

PORT=$((29500 + $$ % 100))
PION_ENV_PATH=/root/.venvs/pion
export CUDA_HOME=/usr/local/cuda
export CUDA_PATH=/usr/local/cuda
export CUDNN_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cudnn
export PATH=/usr/local/cuda/bin:${PION_ENV_PATH}/bin:${PATH}
unset LD_LIBRARY_PATH
export TOKENIZERS_PARALLELISM=true

# Total tokens, global batch size, and training iterations
# Strategy for parallelization
export CUDA_VISIBLE_DEVICES=0
# 1: 1.2; 2: 2.4; 4: 4.8; 8: 9.6;
TOKEN=9.6
# bash arithmetic only supports integer
# Convert TOKEN like 1.2 -> 12 (x10), then scale back.
if [[ "$TOKEN" == *.* ]]; then
    TOKEN_X10=${TOKEN/./}
    TOTAL_TOKENS=$((TOKEN_X10 * 10**8))
else
    TOTAL_TOKENS=$((TOKEN * 10**9))
fi
GLOBAL_BATCH=512
TRAIN_ITER=$((TOTAL_TOKENS / GLOBAL_BATCH / 256))

NNODES=1
NUM_GPUS=1
MICRO_BATCH_SIZE=8
if (( GLOBAL_BATCH % (NUM_GPUS * MICRO_BATCH_SIZE) != 0 )); then
    echo "GLOBAL_BATCH=${GLOBAL_BATCH} must be divisible by NUM_GPUS=${NUM_GPUS} * MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE}."
    exit 1
fi
ACCUMULATION_STEPS=$((GLOBAL_BATCH / NUM_GPUS / MICRO_BATCH_SIZE))
WORLD_SIZE=$((NUM_GPUS * $NNODES))


TP_SIZE=${TP_SIZE:-1}
PP_SIZE=${PP_SIZE:-1}

PARALLEL_ARGS=(
    --tensor-model-parallel-size ${TP_SIZE}
    --pipeline-model-parallel-size ${PP_SIZE}
    --micro-batch-size ${MICRO_BATCH_SIZE}
    --global-batch-size ${GLOBAL_BATCH}
)

DISTRIBUTED_ARGS=(
    --nproc_per_node $NUM_GPUS
    --nnodes $NNODES
)       
LR=${LR:-1e-3}
MIN_LR=${MIN_LR:-1e-5}
PION_MOMENTUM=${PION_MOMENTUM:-transported_ambient_ambient}
ADAM_BETA1=${ADAM_BETA1:-0.9}
ADAM_BETA2=${ADAM_BETA2:-0.95}
PION_BETA1=${PION_BETA1:-0.9}
PION_BETA2=${PION_BETA2:-0.95}
PION_UPDATE_SIDE=${PION_UPDATE_SIDE:-alternate}
PION_SECOND_MOMENTUM_ARGS=()
case "${USE_SECOND_MOMENTUM:-}" in
  true|True|TRUE|1|yes|Yes|YES|on|On|ON)
    PION_SECOND_MOMENTUM_ARGS=(--pion-use-second-momentum)
    ;;
esac
# Pretrain Script
PRETRAIN_SCRIPT="pretrain_gpt.py"
BASE_JOB_NAME=llama-60m-pion-pair-swiglu-9.6B-lr-${LR}-final-lr-${MIN_LR}-cosine-decay-momentum-${PION_MOMENTUM}
SEEDS=(1234 2345 3456)


TRAINING_ARGS=(
    --pion-scaling rms
    --pion-rms 0.2
    --pion-update-side ${PION_UPDATE_SIDE}
    --pion-momentum ${PION_MOMENTUM}
    "${PION_SECOND_MOMENTUM_ARGS[@]}"
    --use-same-init-for-output-layers
    --lr ${LR}
    --min-lr ${MIN_LR}
    --lr-warmup-iters 0
    --lr-decay-style cosine
    --lr-decay-iters $TRAIN_ITER
    --adam-beta1 ${ADAM_BETA1}
    --adam-beta2 ${ADAM_BETA2}
    --pion-beta1 ${PION_BETA1}
    --pion-beta2 ${PION_BETA2}
    --adam-eps 1e-8
    --optimizer pion
    --pion-degree 2
    --weight-decay 0.1
    --clip-grad 1.0
    --no-gradient-accumulation-fusion
)

C4_DATA_ROOT=/data/datasets/c4
TRAIN_BASE_PATH="${TRAIN_BASE_PATH:-${C4_DATA_ROOT}/megatron/train}"
VALID_BASE_PATH="${VALID_BASE_PATH:-${C4_DATA_ROOT}/megatron/val}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-${C4_DATA_ROOT}/tokenizer}"

DATA_PATH=""
while IFS= read -r file; do
    common_prefix=${file%".bin"}
    DATA_PATH+="1 ${common_prefix} "
done < <(find "$TRAIN_BASE_PATH" -type f -path "**.bin" 2>/dev/null)

VALID_DATA_PATH=""
while IFS= read -r file; do
    common_prefix=${file%".bin"}
    VALID_DATA_PATH+="${common_prefix} "
done < <(find "$VALID_BASE_PATH" -type f -path "**.bin" 2>/dev/null)

# The path to cache the data
DATA_PATH_CACHE="${DATA_PATH_CACHE:-${C4_DATA_ROOT}/megatron/cache}"

DATA_ARGS=(
    --tokenizer-model "${TOKENIZER_MODEL}"
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-hf-use-fast
    --seq-length 256
    --train-data-path $DATA_PATH
    --valid-data-path $VALID_DATA_PATH
    --full-validation
    --data-cache-path "${DATA_PATH_CACHE}"
    --train-iters $TRAIN_ITER 
    --num-dataset-builder-threads 8
    --num-workers 4
    # --no-mmap-bin-files
    --distributed-timeout-minutes 240
    --eval-interval 10000
)

MODEL_ARGS=(
    --normalization RMSNorm
    --pair-init
    --pair-init-input-second-moment 1.0
    --pair-diagnostics
    --pair-diagnostics-interval 1000
    --pair-diagnostics-steps 1 10 100
    --pair-diagnostics-calibration-size 16
    --use-same-init-for-output-layers
    --num-layers 8
    --hidden-size 512
    --ffn-hidden-size 1376
    --num-attention-heads 8
    --norm-epsilon 1e-6
    --kv-channels 64
    --max-position-embeddings 1024
    --attention-dropout 0
    --hidden-dropout 0
    --bf16
    --use-rotary-position-embeddings
    --rotary-base 10000
    --swiglu
    --untie-embeddings-and-output-weights
    --cross-entropy-loss-fusion
    --disable-bias-linear
    --transformer-impl transformer_engine
    --attention-backend fused
    --init-method-std 0.02
    --no-persist-layer-norm
    --use-cpu-initialization
)

for SEED in "${SEEDS[@]}"; do
    JOB_NAME="${BASE_JOB_NAME}-seed-${SEED}"
    REPO_PATH="/data/outputs/pion-pair-60m/${JOB_NAME}"
    TENSORBOARD_PATH="${REPO_PATH}/tensorboard/${JOB_NAME}"
    CHECKPOINT_PATH="/data/checkpoints/pion-pair-60m/${JOB_NAME}"
    WANDB_PATH="${REPO_PATH}/wandb/${JOB_NAME}"
    LOG_DIR="${REPO_PATH}/logs"
    LOG_FILE="${LOG_DIR}/${JOB_NAME}.log"

    mkdir -p "$LOG_DIR"
    mkdir -p "$TENSORBOARD_PATH"
    mkdir -p "$CHECKPOINT_PATH"
    mkdir -p "$WANDB_PATH"

    SEED_ARGS=(
        --seed ${SEED}
    )

    CKPT_ARGS=(
        --load ${CHECKPOINT_PATH}
        --ckpt-format "torch"
        --save-interval 10000
        --save ${CHECKPOINT_PATH}
        --save-initial-checkpoint
    )

    LOGGER_ARGS=(
        --log-params-norm
        --log-throughput
        --log-interval 100
        --tensorboard-log-interval 1
        --log-num-zeros-in-grad
        --log-validation-ppl-to-tensorboard
        --log-timers-to-tensorboard
        --log-memory-to-tensorboard
        --log-world-size-to-tensorboard
        --tensorboard-dir ${TENSORBOARD_PATH}
    )

    WANDB_ARGS=(
        --wandb-project test
        --wandb-exp-name ${JOB_NAME}
        --wandb-save-dir ${WANDB_PATH}
    )

    echo "[$(date '+%F %T')] starting seed=${SEED}, micro_batch_size=${MICRO_BATCH_SIZE}, accumulation_steps=${ACCUMULATION_STEPS}" | tee -a "$LOG_FILE"

    {
        PYTHONWARNINGS=ignore ${PION_ENV_PATH}/bin/python -m torch.distributed.run --master_port $PORT \
            ${DISTRIBUTED_ARGS[@]} \
            $PRETRAIN_SCRIPT \
            ${DATA_ARGS[@]} \
            ${MODEL_ARGS[@]} \
            ${TRAINING_ARGS[@]} \
            ${PARALLEL_ARGS[@]} \
            ${SEED_ARGS[@]} \
            ${CKPT_ARGS[@]} \
            ${LOGGER_ARGS[@]} \
            ${WANDB_ARGS[@]}
    } 2>&1 | grep --line-buffered -v -E "(Warning|DeprecationWarning|UserWarning|FutureWarning|WARNING|Deprecated)" | tee -a "$LOG_FILE"

    run_status=${PIPESTATUS[0]}
    if [[ ${run_status} -ne 0 ]]; then
        echo "[$(date '+%F %T')] seed=${SEED} failed with exit code=${run_status}." | tee -a "$LOG_FILE"
        exit ${run_status}
    fi

    echo "[$(date '+%F %T')] seed=${SEED} finished successfully." | tee -a "$LOG_FILE"
done
