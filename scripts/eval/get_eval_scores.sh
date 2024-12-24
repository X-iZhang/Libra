#!/bin/bash
############################################################################
# Example Usage:
#   bash ./scripts/eval/get_eval_scores.sh
#
# This script calls the `libra.eval.radiology_report` module to evaluate
# multiple model predictions (or different checkpoint outputs) against the
# reference data. The results are appended to a single CSV file, allowing
# you to compare performance side-by-side across multiple runs or versions.
############################################################################

# Path to the CSV file where metrics will be stored
SAVE_CSV_PATH="./scripts/eval/result_scores"

############################
# 1) Model Evaluation #1
############################
python -m libra.eval.radiology_report \
    --references ./path/to/your/references.jsonl \
    --predictions ./path/to/your/predictions_1.jsonl \
    --model-name model-lable-1 \
    --save-to-csv "${SAVE_CSV_PATH}"

############################
# 2) Model Evaluation #2
############################
python -m libra.eval.radiology_report \
    --references ./path/to/your/references.jsonl \
    --predictions ./path/to/your/predictions_2.jsonl \
    --model-name model-lable-2 \
    --save-to-csv "${SAVE_CSV_PATH}"

############################
# 3) Additional Models
############################
# If you have more models or checkpoints to evaluate, just copy/paste
# the block below and adjust paths, model-name, etc.

# python -m libra.eval.radiology_report \
#     --references ./path/to/your/references.jsonl \
#     --predictions ./path/to/your/predictions_3.jsonl \
#     --model-name model-lable-3 \
#     --save-to-csv "${SAVE_CSV_PATH}"