#!/usr/bin/env bash
set -e


##################################
# 默认参数
NUM_GPUS=4              # 自动选择的 GPU 数量
GPU_IDS=""              # 手动指定的 GPU ID 列表，如 "0,1,2,3"
TRAIN_SCRIPT="train.py" # 训练脚本
##################################




##################################
while [[ $# -gt 0 ]]; do
  case "$1" in
    -g|--gpus)
      GPU_IDS="$2"
      shift 2
      ;;
    -n|--num-gpus)
      NUM_GPUS="$2"
      shift 2
      ;;
    -s|--script)
      TRAIN_SCRIPT="$2"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    -h|--help)
      echo "用法: $0 [-g gpu_ids] [-n num_gpus] [-s script] [-- extra args...]"
      exit 0
      ;;
    *)
      echo "未知选项: $1"
      exit 1
      ;;
  esac
done

# 剩余参数都透传给 python 脚本
TRAIN_ARGS=("$@")

# 如果没有手动指定 GPU，就自动选择利用率最低的 NUM_GPUS 个
if [[ -z "$GPU_IDS" ]]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "错误: 找不到 nvidia-smi，请确认已安装 NVIDIA 驱动。" >&2
    exit 1
  fi

  GPU_IDS=$(nvidia-smi \
    --query-gpu=index,utilization.gpu \
    --format=csv,noheader,nounits \
    | sort -t, -k2 -n \
    | head -n "$NUM_GPUS" \
    | cut -d',' -f1 \
    | paste -sd, -)

  if [[ -z "$GPU_IDS" ]]; then
    echo "错误: 无法从 nvidia-smi 获取 GPU 信息。" >&2
    exit 1
  fi
fi

export CUDA_VISIBLE_DEVICES="$GPU_IDS"

GPU_COUNT=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
echo "GPUS: $GPU_COUNT ($CUDA_VISIBLE_DEVICES)"

python "$TRAIN_SCRIPT" "${TRAIN_ARGS[@]}"
