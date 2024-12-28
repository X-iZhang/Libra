# Finetune and Evaluation Libra on Custom Datasets

## Training/Validation Dataset Format

Convert your train/valid data to a JSON file of a List of all samples. Sample metadata should contain `id` (a unique identifier), `image` (the path to the image or images), and `conversations` (the conversation data between human and AI).

Here's a sample JSON for finetuning Libra for Radiology Report Generation:

```json
[
    {
        "id": 12345678,
        "image": [
            "files/p19/p19586697/s50637770/efb2c222-0fe78b2f-2bd67556-d10e01d8-72e87669.jpg",
            "files/p19/p19586697/s50637770/efb2c222-0fe78b2f-2bd67556-d10e01d8-72e87669.jpg"
        ],
        "conversations": [
            {
                "from": "human",
                "value": "<image>\nProvide a detailed description of the findings in the radiology image. Following clinical context: History: Chest tightness."
            },
            {
                "from": "gpt",
                "value": "The heart and mediastinum are normal. The lung fields are clear. The costophrenic angles are sharp. No infiltrates are present. There is no evidence of a pneumothorax."
            }
        ]
    },
    ...
]
```

<details>
<summary> If there is only one image, you only need to pass a single path to the "image". </summary>

```json
[   
    {
        ...
        "image": [
            "files/p19/p19586697/s50637770/efb2c222-0fe78b2f-2bd67556-d10e01d8-72e87669.jpg"
        ]
        ...
    } 
]
```
</details>

## Evaluation Dataset Format

Convert your eval data to a JSON Line file of a List of all samples. Sample metadata should contain `question_id` (a unique identifier), `image` (the path to the image or images), and `text` (the prompt or question to be answered by the model). The `reference` (the ground truth answer) is optional and can be used for scoring evaluations.

Here's a sample JSONL for evaluating Libra for Radiology Report Generation:

```json
{"question_id": 12345678, "image": ["files/p19/p19586697/s50637770/efb2c222-0fe78b2f-2bd67556-d10e01d8-72e87669.jpg", "files/p19/p19586697/s50637770/efb2c222-0fe78b2f-2bd67556-d10e01d8-72e87669.jpg"], "text": "Provide a detailed description of the findings in the radiology image. Following clinical context: History: Chest tightness.", "reference": "The heart and mediastinum are normal. The lung fields are clear. The costophrenic angles are sharp. No infiltrates are present. There is no evidence of a pneumothorax."}
...
```

## Tips

### Finetuning with Limited Data

If you have limited task-specific data, we recommend finetuning from [Libra checkpoints](https://huggingface.co/X-iZhang/libra-v1.0-7b) using LoRA. Follow this [finetune_lora.sh script](https://github.com/X-iZhang/Libra/blob/main/scripts/finetune_lora.sh).

### Finetuning with Sufficient Data

If you have sufficient task-specific data, you can perform full-model finetuning from [Libra checkpoints](https://huggingface.co/X-iZhang/libra-v1.0-7b). Follow this [finetune.sh script](https://github.com/X-iZhang/Libra/blob/main/scripts/finetune.sh).

### Hyperparameter Adjustment

Adjust the hyperparameters to fit your specific dataset and hardware constraints.

