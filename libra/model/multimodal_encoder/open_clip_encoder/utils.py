from typing import Optional, Dict
import os

import torch
import numpy as np
from open_clip import create_model_from_pretrained, get_tokenizer # works on open-clip-torch>=2.23.0, timm>=0.9.8
from open_clip.factory import HF_HUB_PREFIX, _MODEL_CONFIGS, load_state_dict



def get_clip_metrics(image_features, text_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics


def from_pretrained(
        model_name: str,
        config: Optional[Dict] = None,
        checkpoint_path: str = None
    ):
    if (not model_name.startswith(HF_HUB_PREFIX)
        and model_name not in _MODEL_CONFIGS
        and config is not None):
        _MODEL_CONFIGS[model_name] = config

    model, preprocess = create_model_from_pretrained(
        model_name=model_name,
        pretrained=checkpoint_path
    )

    tokenizer = get_tokenizer(model_name)

    return model, preprocess, tokenizer


def remove_transformer_pooler_weights(
        checkpoint_path, new_path="/tmp/biomed_clip/ckpt.pt"
    ):
    need_new = False
    state_dict = load_state_dict(checkpoint_path)
    for key in list(state_dict.keys()):
        if key.startswith("text.transformer.pooler"):
            need_new = True
            state_dict.pop(key)
    if need_new:
        os.makedirs(os.path.dirname(new_path), exist_ok=True)
        torch.save(state_dict, new_path)
        return new_path
    return checkpoint_path


if __name__ == "__main__":
    import sys
    remove_transformer_pooler_weights(*sys.argv[1:])