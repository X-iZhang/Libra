import argparse
import json
import os
import re
import sys

import evaluate
import numpy as np
import pandas as pd
from tqdm import tqdm

from libra.eval import temporal_f1_score

# Pre-load metrics
bertscore_metric = evaluate.load("bertscore")
rouge_metric = evaluate.load('rouge')
bleu_metric = evaluate.load("bleu")
meteor_metric = evaluate.load('meteor')


def clean_text(text: str) -> str:
    """
    Perform basic cleanup of text by removing newlines, dashes, and some special patterns.
    """
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'[_-]+', ' ', text)
    text = re.sub(r'\(___, __, __\)', '', text)
    text = re.sub(r'---, ---, ---', '', text)
    text = re.sub(r'\(__, __, ___\)', '', text)
    text = re.sub(r'[_-]+', ' ', text)
    text = re.sub(r'[^\w\s.,:;()\-]', '', text)
    text = re.sub(r'\s{2,}', ' ', text).strip()
    return text


def load_json(path: str) -> list:
    """
    Load a JSONL file and return a list of parsed objects.
    Each line should be a valid JSON object.
    """
    content = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            content.append(json.loads(line))
    return content


def extract_sections(data: list) -> list:
    """
    Extract relevant text sections (e.g., findings, impression, text)
    from a list of JSON objects and clean each item.
    """
    sections_list = []
    for item in data:
        if 'reference' in item:
            cleaned_text = clean_text(item['reference'].lower())
            sections_list.append(cleaned_text)
        elif 'findings' in item:
            cleaned_text = clean_text(item['findings'].lower())
            sections_list.append(cleaned_text)
        elif 'impression' in item:
            cleaned_text = clean_text(item['impression'].lower())
            sections_list.append(cleaned_text)
        elif 'text' in item:
            cleaned_text = clean_text(item['text'].lower())
            sections_list.append(cleaned_text)
    return sections_list


def append_results_to_csv(results: dict, model_name: str, csv_path: str) -> None:
    """
    Convert the results dictionary into a DataFrame and append it to a CSV file.
    Inserts 'Model Name' at the first column if it doesn't exist.
    Creates a new CSV if it doesn't exist, otherwise appends.
    """
    df = pd.DataFrame([results])
    df.insert(0, "Model Name", model_name)

    header = not os.path.isfile(csv_path)  # If file doesn't exist, write the header
    df.to_csv(csv_path, mode='a', header=header, index=False)


def evaluate_report(
    references: str,
    predictions: str,
) -> dict:
    """
    Evaluate the model outputs against reference texts using multiple metrics:
    - BLEU (1–4)
    - METEOR
    - ROUGE-L
    - BERTScore (F1)
    - Temporal F1

    Returns a dictionary of computed metrics.
    """
    # Load data
    references_data = load_json(references)
    predictions_data = load_json(predictions)

    # Basic validation: question_id alignment
    gt_ids = [item['question_id'] for item in references_data]
    pred_ids = [item['question_id'] for item in predictions_data]
    assert gt_ids == pred_ids, "Please make sure predictions and references are perfectly matched by question_id."

    # Extract text sections
    references_list = extract_sections(references_data)
    predictions_list = extract_sections(predictions_data)

    # Calculate metrics
    with tqdm(total=8, desc="Calculating metrics") as pbar:
        # BLEU-1
        bleu1 = bleu_metric.compute(
            predictions=predictions_list, 
            references=references_list, 
            max_order=1
        )['bleu']
        print(f"BLEU-1 Score: {round(bleu1 * 100, 2)}")
        pbar.update(1)

        # BLEU-2
        bleu2 = bleu_metric.compute(
            predictions=predictions_list, 
            references=references_list, 
            max_order=2
        )['bleu']
        print(f"BLEU-2 Score: {round(bleu2 * 100, 2)}")
        pbar.update(1)

        # BLEU-3
        bleu3 = bleu_metric.compute(
            predictions=predictions_list, 
            references=references_list, 
            max_order=3
        )['bleu']
        print(f"BLEU-3 Score: {round(bleu3 * 100, 2)}")
        pbar.update(1)

        # BLEU-4
        bleu4 = bleu_metric.compute(
            predictions=predictions_list, 
            references=references_list, 
            max_order=4
        )['bleu']
        print(f"BLEU-4 Score: {round(bleu4 * 100, 2)}")
        pbar.update(1)

        # ROUGE-L
        rougel = rouge_metric.compute(
            predictions=predictions_list, 
            references=references_list
        )['rougeL']
        print(f"ROUGE-L Score: {round(rougel * 100, 2)}")
        pbar.update(1)

        # METEOR
        meteor = meteor_metric.compute(
            predictions=predictions_list, 
            references=references_list
        )['meteor']
        print(f"METEOR Score: {round(meteor * 100, 2)}")
        pbar.update(1)

        # BERTScore (mean F1)
        bert_f1 = bertscore_metric.compute(
            predictions=predictions_list, 
            references=references_list, 
            model_type='distilbert-base-uncased'
        )['f1']
        bert_score = float(np.mean(bert_f1))
        print(f"Bert Score: {round(bert_score * 100, 2)}")
        pbar.update(1)

        # Temporal F1
        tem_f1 = temporal_f1_score(
            predictions=predictions_list, 
            references=references_list
        )["f1"]
        print(f"Temporal F1 Score: {round(tem_f1 * 100, 2)}")
        pbar.update(1)

    return {
        'BLEU1': round(bleu1 * 100, 2),
        'BLEU2': round(bleu2 * 100, 2),
        'BLEU3': round(bleu3 * 100, 2),
        'BLEU4': round(bleu4 * 100, 2),
        'METEOR': round(meteor * 100, 2),
        'ROUGE-L': round(rougel * 100, 2),
        'Bert_score': round(bert_score * 100, 2),
        'Temporal_entity_score': round(tem_f1 * 100, 2)
    }


def main():
    """
    Parse arguments, compute evaluation metrics, and append the results to a CSV file.
    """
    parser = argparse.ArgumentParser(
        description='Evaluation for Libra Generated Outputs'
    )
    parser.add_argument('--references', type=str, required=True,
                        help='Path to the ground truth file (JSONL).')
    parser.add_argument('--predictions', type=str, required=True,
                        help='Path to the prediction file (JSONL).')
    parser.add_argument('--model-name', type=str, required=True,
                        help='Unique model identifier for tracking in the results CSV.')
    parser.add_argument('--save-to-csv', type=str, required=True,
                        help='Path of the CSV file where results will be saved/appended.')
    args = parser.parse_args()

    # Calculate metrics
    scores_results = evaluate_report(
        references=args.references, 
        predictions=args.predictions
    )

    # Append results to CSV
    append_results_to_csv(scores_results, args.model_name, args.save_to_csv)


if __name__ == "__main__":
    main()