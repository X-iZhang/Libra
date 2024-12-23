#!/bin/bash
###############################################################################
# Example usage:
#   bash ./scripts/libra_eval.sh sample 
#
# Possible modes: 'beam' or 'sample'
#   1) 'beam'   -> Uses Beam Search
#   2) 'sample' -> Uses Temperature/Top-p Sampling
#
# This script calls the libra.eval.eval_vqa_libra module for evaluation.
###############################################################################

# First argument determines the decoding mode; default is "sample"
MODE=${1:-sample}

# The conversation mode (conv-mode); default is "libra_v1"
# This should be the same PROMPT_VERSION used when training the model.
CONV_MODE="libra_v1"

# Paths to model and data
MODEL_VERSION="./path/to/your/model"   # or "X-iZhang/libra-v1.0-7b", etc.
QUESTION_FILE="./path/to/questions.jsonl"
IMAGE_FOLDER="./path/to/image/folder"
ANSWERS_FILE="./path/to/answers.jsonl"

# Print the chosen mode and conversation settings
echo "Decoding method: ${MODE}"
echo "Conversation mode: ${CONV_MODE}"

###############################################################################
# Depending on the mode, run the appropriate Python command
###############################################################################
if [ "${MODE}" == "beam" ]; then
    ###############################################################################
    # Beam Search
    ###############################################################################
    echo "Running Beam Search evaluation..."
    python -m libra.eval.eval_vqa_libra \
        --model-path "${MODEL_VERSION}" \
        --question-file "${QUESTION_FILE}" \
        --image-folder "${IMAGE_FOLDER}" \
        --answers-file "${ANSWERS_FILE}" \
        --num_beams 5 \
        --length_penalty 2 \
        --num_return_sequences 3 \
        --max_new_tokens 128 \
        --conv-mode "${CONV_MODE}"

elif [ "${MODE}" == "sample" ]; then
    ###############################################################################
    # Sampling (temperature, top-p)
    ###############################################################################
    echo "Running Sampling evaluation..."
    python -m libra.eval.eval_vqa_libra \
        --model-path "${MODEL_VERSION}" \
        --question-file "${QUESTION_FILE}" \
        --image-folder "${IMAGE_FOLDER}" \
        --answers-file "${ANSWERS_FILE}" \
        --temperature 0.9 \
        --top_p 0.6 \
        --max_new_tokens 128 \
        --conv-mode "${CONV_MODE}"

else
    ###############################################################################
    # Invalid mode
    ###############################################################################
    echo "Error: Unknown mode '${MODE}'. Please specify 'beam' or 'sample'."
    exit 1
fi
