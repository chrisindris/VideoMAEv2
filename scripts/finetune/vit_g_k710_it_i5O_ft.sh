#!/usr/bin/env bash

# Script to finetune i5O.
# Based on ./vit_g_k710_it_k400_ft.sh
# usage: ./vit_g_k710_it_i5O_ft.sh

set -x

MASTER_ADDR=127.0.0.1
MASTER_PORT=6006
N_NODES=1
NODE_RANK=0

OMP_NUM_THREADS=1

OUTPUT_DIR='/data/i5O/finetuned/i5O/vit_g_hybrid_pt_1200e_k710_it_i5O_ft'
DATA_PATH='/data/i5O/anno_out/'
#MODEL_PATH='/data/i5O/pretrained/VideoMAEv2/vit_g_hybrid_pt_1200e_k710_ft.pth'
MODEL_PATH='/data/i5O/finetuned/i5O/vit_g_hybrid_pt_1200e_k710_it_i5O_ft/checkpoint-6/mp_rank_00_model_states.pt'

JOB_NAME=$1
PARTITION=${PARTITION:-"video"}
# 8 for 1 node, 16 for 2 node, etc.
GPUS=$(nvidia-smi -L | wc -l)
GPUS_PER_NODE=$GPUS
PY_ARGS=${@:1}

# batch_size can be adjusted according to the graphics card
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} \
	--master_port ${MASTER_PORT} --nnodes=${N_NODES} --node_rank=${NODE_RANK} --master_addr=${MASTER_ADDR} \
	run_class_finetuning.py \
	--model vit_giant_patch14_224 \
	--data_set i5O-trimmed \
	--nb_classes 20 \
	--data_path ${DATA_PATH} \
	--finetune ${MODEL_PATH} \
	--log_dir ${OUTPUT_DIR} \
	--output_dir ${OUTPUT_DIR} \
	--batch_size 1 \
	--input_size 224 \
	--short_side_size 224 \
	--save_ckpt_freq 10 \
	--num_frames 16 \
	--sampling_rate 4 \
	--num_sample 2 \
	--num_workers 8 \
	--opt adamw \
	--lr 1e-5 \
	--drop_path 0.25 \
	--layer_decay 0.9 \
	--opt_betas 0.9 0.999 \
	--weight_decay 0.1 \
	--warmup_epochs 1 \
	--epochs 10 \
	--test_num_segment 5 \
	--test_num_crop 3 \
	--dist_eval --enable_deepspeed \
	--start_epoch 6
${PY_ARGS}
