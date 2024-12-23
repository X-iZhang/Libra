#!/bin/bash
###################################################################################
# This script trains the Libra model in the first-stage feature alignment.
# You can switch between different base models (Meditron, Vicuna, LLaMA-2/3, etc.)
# by uncommenting the corresponding lines below.
#
# Usage:
#   1. Make sure you have DeepSpeed and necessary dependencies installed.
#   2. Adjust the paths to your training data, validation data, and image folder.
#   3. Uncomment the correct MODEL_VERSION and PROMPT_VERSION for your target model.
#   4. Run this script: `bash ./scripts/pretrain.sh`
###################################################################################

##########################
#      BASE MODELS       #
##########################

###################### LIBRA ######################
MODEL_VERSION="epfl-llm/meditron-7b"
PROMPT_VERSION="libra_v1"
###################### LIBRA ######################

##################### VICUNA ######################
#MODEL_VERSION="lmsys/vicuna-7b-v1.5"
#PROMPT_VERSION="vicuna_v1"
##################### VICUNA ######################

#################### LLaMA-2 #####################
#MODEL_VERSION="meta-llama/Llama-2-7b-chat-hf"
#PROMPT_VERSION="llama_2"
#################### LLaMA-2 ####################

#################### LLaMA-3.x ###################
#MODEL_VERSION="meta-llama/Llama-3.x-8B-Instruct"
#PROMPT_VERSION="llama_3"
#################### LLaMA-3.x ###################

###############################################################################
# Hyperparameters & Training Configurations
# (Adjust as needed for your setup.)
###############################################################################

# Data paths
TRAIN_DATA="./path/to/alignment_train_data.json"   # Path to training data (JSON)
VAL_DATA="./path/to/alignment_valid_data.json"     # Path to validation data
IMG_FOLDER="./path/to/image/folder"               # Folder containing the images

# Vision tower
VISION_TOWER="microsoft/rad-dino"                  # e.g., "openai/clip-vit-large-patch14"

# Output checkpoints directory
OUTPUT_DIR="./checkpoints/libra-v1.0-7b-pretrain"

# Deepspeed config file (you can adjust zero2.json as needed)
DEEPSPEED_CONFIG="./scripts/zero2.json"

# General training parameters
NUM_EPOCHS=1
TRAIN_BSZ=16
EVAL_BSZ=4
GRAD_ACC_STEPS=1
LR=2e-5

###############################################################################
# Run the training with DeepSpeed
###############################################################################

deepspeed libra/train/train_mem.py \
    --deepspeed ${DEEPSPEED_CONFIG} \
    --model_name_or_path ${MODEL_VERSION} \
    --version ${PROMPT_VERSION} \
    --freeze_backbone True \
    --data_path ${TRAIN_DATA} \
    --validation_data_path ${VAL_DATA} \
    --image_folder ${IMG_FOLDER} \
    --vision_tower ${VISION_TOWER} \
    --mm_projector_type TAC \
    --tune_mm_mlp_adapter True \
    --freeze_mm_mlp_adapter False \
    --mm_vision_select_layer all \
    --image_aspect_ratio pad \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --mm_vision_select_feature patch \
    --bf16 True \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs ${NUM_EPOCHS} \
    --per_device_train_batch_size ${TRAIN_BSZ} \
    --per_device_eval_batch_size ${EVAL_BSZ} \
    --gradient_accumulation_steps ${GRAD_ACC_STEPS} \
    --do_eval True \
    --compute_metrics True \
    --evaluation_strategy "steps" \
    --eval_steps 0.01 \
    --save_strategy "steps" \
    --save_steps 20000 \
    --save_total_limit 1 \
    --learning_rate ${LR} \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
        