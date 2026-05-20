#!/usr/bin/env bash
set -euo pipefail

device="${CUDA_DEVICE:-0}"
data_root="${VISA_ROOT:-./dataset/Visa}"
mode="${1:-test}"
shot="${SHOT:-1}"
seed="${SEED:-10}"

n_ctx=12
vl_reduction=4
pq_mid_dim=128
base_dir="${n_ctx}_${vl_reduction}_${pq_mid_dim}_train_on_visa_3adapters_batch8"
save_dir="./adaptclip_checkpoints/${base_dir}"
result_dir="./results/${base_dir}"

case "${mode}" in
  meta)
    python dataset/visa.py --root "${data_root}"
    ;;
  train)
    CUDA_VISIBLE_DEVICES="${device}" python train.py \
      --dataset visa \
      --train_data_path "${data_root}" \
      --save_path "${save_dir}" \
      --features_list 6 12 18 24 \
      --image_size 518 \
      --batch_size 8 \
      --print_freq 1 \
      --epoch 15 \
      --save_freq 1 \
      --n_ctx "${n_ctx}" \
      --vl_reduction "${vl_reduction}" \
      --pq_mid_dim "${pq_mid_dim}" \
      --visual_learner \
      --textual_learner \
      --pq_learner \
      --pq_context
    ;;
  test)
    CUDA_VISIBLE_DEVICES="${device}" python test.py \
      --dataset visa \
      --test_data_path "${data_root}" \
      --seed "${seed}" \
      --k_shots "${shot}" \
      --checkpoint_path "${save_dir}/epoch_15.pth" \
      --save_path "${result_dir}" \
      --features_list 6 12 18 24 \
      --image_size 518 \
      --batch_size 8 \
      --n_ctx "${n_ctx}" \
      --vl_reduction "${vl_reduction}" \
      --pq_mid_dim "${pq_mid_dim}" \
      --visual_learner \
      --textual_learner \
      --pq_learner \
      --pq_context
    ;;
  *)
    echo "Usage: $0 [meta|train|test]" >&2
    exit 2
    ;;
esac
