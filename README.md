<h1 align="center">
    Libra: Leveraging Temporal Images for Biomedical Radiology Analysis
</h1>

[![arXiv](https://img.shields.io/badge/Arxiv-2411.19378-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2411.19378) 
[![hf_space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/X-iZhang/libra-v1.0-7b)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow.svg?)](https://github.com/X-iZhang/Libra/blob/main/LICENSE)
[![Views](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FX-iZhang%2FLibra&count_bg=%2300C0FF&title_bg=%23004080&icon=&icon_color=%23FFFFFF&title=Views)](https://hits.seeyoufarm.com)

<details open><summary>📢 More Than Radiology: Codespace Features for MLLMs Workflow You’ll Love! 🎉 </summary><p>

>  * **LLaVA-Type & LLaMA_3 Support**: Deploy and train advanced models effortlessly.
>  * **Resume Training**: Resume training from checkpoints at any stage, whether for pre-training or fine-tuning.  
>  * **Validation Dataset**: Track model performance in real-time on `validation datasets` during training. 
>  * **Custom Metrics**: Go beyond `eval_loss` with metrics like `BLEU`, `ROUGE-L`, `RadGraph-F1` or define your own criteria on valid dataset.   
>  * **Smart Saving**: Automatically save the best model based on validation loss or custom evaluation scores.

</p></details>

<!-- ![architecture](./assets/libra_architecture.png) -->

## Contents
- [Install](#install)
- [Libra Weights](#libra-weights)
- [Quick Start](#quick-start)
- [Dataset](#dataset)
- [Train](#train)
- [Evaluation](#evaluation)

## Install
We strongly recommend that you create an environment from scratch as follows:
1. Clone this repository and navigate to Libra folder
```bash
git clone https://github.com/X-iZhang/Libra.git
cd Libra
```

2. Install Package
```Shell
conda create -n libra python=3.10 -y
conda activate libra
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

3. Install additional packages for Training and Evaluation cases
```Shell
pip install -e ".[train,eval]"
pip install flash-attn --no-build-isolation
```

<details>
<summary> Upgrade to latest code base </summary>

```Shell
git pull
pip install -e .
```

</details>

## Libra Weights

| Version | Base LLM | Vision Encoder| Checkpoint |
| ------- | ------- | ------- | ------- |
| Libra v1.0 | Meditron-7B | RAD-DINO | [X-iZhang/libra-v1.0-7b](https://huggingface.co/X-iZhang/libra-v1.0-7b) |

## Quick Start

### CLI Inference
We support running inference using the CLI. To use our model, run:
```Shell
python -m libra.serve.cli \
    --model-path X-iZhang/libra-v1.0-7b \
    --image-file "./path/to/current_image.jpg" "./path/to/previous_image.jpg"
    # If there is no previous image, only one path is needed.
```

### Script Inference
You can use the `libra_eval` function in `libra/eval/run_libra.py` to easily launch a model trained by yourself or us on local machine or in Google Colab, after installing this repository.

```Python
from libra.eval import libra_eval

# Define the model path, which can be a pre-trained model or your own fine-tuned model.
model_path = "X-iZhang/libra-v1.0-7b"  # Or your own model

# Define the paths to the images. The second image is optional for temporal comparisons.
image_files = [
    "./path/to/current/image.jpg", 
    "./path/to/previous/image.jpg"  # Optional: Only include if a reference image is available
]

# Define the prompt to guide the model's response. Add clinical instructions if needed.
prompt = (
    "Provide a detailed description of the findings in the radiology image. "
    "Following clinical context: ..."
)

# Specify the conversational mode, matching the PROMPT_VERSION used during training.
conv_mode = "libra_v1"

# Call the libra_eval function.
libra_eval(
    model_path=model_path,
    image_file=image_files,
    query=prompt,
    temperature=0.9,
    top_p=0.8,
    conv_mode=conv_mode,
    max_new_tokens=512
)
```
<details>
<summary>Meanwhile, you can use the Beam Search method to obtain output.</summary>

```Python
libra_eval(
    model_path=model_path,
    image_file=image_files,
    query=prompt,
    num_beams=5, 
    length_penalty=2,
    num_return_sequences=2,
    conv_mode=conv_mode,
    max_new_tokens=512
)
```

</details>

<details>
<summary>Additionally, you can directly use LoRA weights for inference.</summary>

```Python
libra_eval(
    model_path="./path/to/lora_weights",  # path to LoRA weights
    model_base="./path/to/base_model",  # path to base Libra model
    image_file=image_files,
    query=prompt,
    num_beams=5, 
    length_penalty=2,
    num_return_sequences=2,
    conv_mode=conv_mode,
    max_new_tokens=512
)
```

</details>

## Dataset

### Prepare Data

All the data we use comes from [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/) and its two variants, and we strictly follow the official split for `train/valid/test` division.

- Image Data

All images used for **Libra** come from the [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.0.0/) dataset in `.jpg` format. `DICOM` format is also supported and can be found in the [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/).

After downloading the images, they will be automatically organized into the following structure in `./path/to/playground/data`:

```
./data/physionet.org/files/mimic-cxr-jpg/2.0.0
└──files
    ├── p10
    │   └── p10000032
    │       └── s50414267
    │           ├── image1.jpg
    │           └── image2.jpg
    ├── p11
    ├── p12
    ├── ...
    └── p19
```

- Annotation Data

All annotations used for **Libra** come from the [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/) and its two variants. This includes Radiology Reports and other relevant Visual Question Answering. 

Please download the following datasets from the official website: `mimic-cxr-reports.zip` from [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/), [MIMIC-Diff-VQA](https://physionet.org/content/medical-diff-vqa/1.0.0/), and [MIMIC-Ext-*MIMIC-CXR-VQA*](https://physionet.org/content/mimic-ext-mimic-cxr-vqa/1.0.0/).

### Preprocess Data

- Radiology Report Sections

For free-text radiology report, we extract the `Findings`, `Impression`, `Indication`, `History`, `Comparison`, and `Technique` sections using the official [mimic-cxr](https://github.com/MIT-LCP/mimic-cxr/tree/master/txt) repository.

- Visual Question Answering for Chest X-ray

In [Medical-Diff-VQA](https://physionet.org/content/medical-diff-vqa/1.0.0/), the main image is used as the current image, and the reference image is used as the prior image. In [MIMIC-Ext-MIMIC-CXR-VQA](https://physionet.org/content/mimic-ext-mimic-cxr-vqa/1.0.0/), all cases use a dummy prior image.

### Data Download

| Alignment data files | Split | Size |
| ----- | ----- | -----: |
| [libra_alignment_train.json](https://drive.google.com/file/d/1AIT1b3eRXgJFp3FJmHci3haTunK1NTMA/view?usp=drive_link)| train | 780 MiB |
| [libra_alignment_valid.json](https://drive.google.com/file/d/1nvbUoDmw7j4HgXwZWiiACIhvZ6BvR2LX/view?usp=sharing)| valid | 79 MiB |

| Fine-Tuning data files | Split | Size |
| ----- | ----- | ----- |
| [libra_findings_section_train.json](https://drive.google.com/file/d/1rJ3G4uiHlzK_P6ZBUbAi-cDaWV-o6fcz/view?usp=sharing)| train | 159 MiB |
| [libra_findings_section_valid.json](https://drive.google.com/file/d/1IYwQS23veOU5SXWGYiTyq9VHUwkVESfD/view?usp=sharing)| valid | 79 MiB |

| Evaluation data files | Split | Size |
| --- | --- | ---: |
| [libra_findings_section_eval.jsonl](https://drive.google.com/file/d/1fy_WX616L8SgyAonadJ2fUIEaX0yrGrQ/view?usp=sharing)| eval | 2 MiB |


<details>

<summary>Meanwhile, here are some bonus evaluation data files.</summary>

| Evaluation data files | Split | Size |
| --- | --- | ---: |
| [libra_impressions_section_eval.jsonl](https://drive.google.com/file/d/16msRfk7XxCmq7ZPG82lKvsnnjqsRPv__/view?usp=sharing)| eval | 1 MiB |
| [libra_MIMIC-Ext-MIMIC-CXR-VQA_eval.jsonl](https://drive.google.com/file/d/1krPMwGGY6HP4sonNKlnkhLOoZrdjfVMW/view?usp=sharing)| eval | 4 MiB |
| [libra_MIMIC-Diff-VQA _eval.jsonl](https://drive.google.com/file/d/1tP_CxPMM9PiKTq1mLYRHICcyJ36Q13mC/view?usp=sharing)| eval | 20 MiB |

</details>



If you want to train or evaluate your own tasks or datasets, please refer to [`Custom_Data.md`](https://github.com/X-iZhang/Libra/blob/main/CUSTOM_DATA.md).


## Train
Libra adopt a two-stage training strategy: (1) visual feature alignment: the visual encoder and LLM weights are frozen, and the Temporal Alignment Connector is trained; (2) RRG downstream task fine-tuning: apply LoRA to fine-tune the pre-trained LLM on the Findings section generation task.

Libra is trained on 1 A6000 GPU with 48GB memory. To train on multiple GPUs, you can set the `per_device_train_batch_size` and the `gradient_accumulation_steps` accordingly. Always keep the global batch size the same: `per_device_train_batch_size` x `gradient_accumulation_steps` x `num_gpus`.

### Hyperparameters
We set reasonable hyperparameters based on our device. The hyperparameters used in both pretraining and LoRA finetuning are provided below.

1. Pretraining

| Hyperparameter | Global Batch Size | Learning rate | Epochs | Max length | Weight decay |
| --- | :---: | :---: | :---: | :---: | :---: |
| Libra-v1.0-7b | 16 | 2e-5 | 1 | 2048 | 0 |

2. LoRA finetuning

| Hyperparameter | Global Batch Size | Learning rate | Epochs | Max length | Weight decay | LoRA rank | LoRA alpha |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Libra-v1.0-7b | 16 | 2e-5 | 1 | 2048 | 0 | 128 | 256 |

### Download Meditron checkpoints (automatically)

Our base LLM model, [Meditron-7B](https://huggingface.co/epfl-llm/meditron-7b), adapted to the medical domain from the Llama-2-7B model, will be downloaded automatically when you run our provided training scripts. No action is needed on your part.

### Stage 1: visual feature alignment

Pretraining takes approximately 385 hours for Libra-v1.0-7b-pretrain on a single A6000 GPU (48GB) due to device limitations.

For detailed training scripts and guidelines, please refer to the following: [`pretrain.sh`](https://github.com/X-iZhang/Libra/blob/main/scripts/pretrain.sh) and [`pretrain_xformers.sh`](https://github.com/X-iZhang/Libra/blob/main/scripts/pretrain_xformers.sh) for [memory-efficient attention](https://arxiv.org/abs/2112.05682) implemented in [xFormers](https://github.com/facebookresearch/xformers).

- `--mm_projector_type TAC`: the Temporal Alignment Connector.
- `--vision_tower microsoft/rad-dino`: RAD-DINO is a vision transformer for encoding chest X-rays using DINOv2.
- `--mm_vision_select_layer all`: Use all image features from the encoder for the Layerwise Feature Extractor.
- `--tune_mm_mlp_adapter True`
- `--freeze_mm_mlp_adapter False` 

### Stage 2: RRG downstream task fine-tuning
You may download our pretrained projectors from the [`mm_tac_projector.bin`](https://huggingface.co/X-iZhang/libra-v1.0-7b) file. It takes around 213 hours for Libra-v1.0-7b on a single A6000 GPU (48GB) due to device limitations.

For detailed training scripts and guidelines, please refer to: [`finetune_lora.sh`](https://github.com/X-iZhang/Libra/blob/main/scripts/finetune_lora.sh).

- `--tune_mm_mlp_adapter False`
- `--freeze_mm_mlp_adapter True` 

If you have enough GPU memory: Use [`finetune.sh`](https://github.com/X-iZhang/Libra/blob/main/scripts/finetune.sh) to fine-tune the entire model. Alternatively, you can replace `zero3.json` with `zero3_offload.json` to offload some parameters to CPU RAM, though this will slow down the training speed.

If you are interested in continue finetuning Libra model to your own task/data, please check out [`Custom_Data.md`](https://github.com/X-iZhang/Libra/blob/main/CUSTOM_DATA.md).

### New Options to Note

- `--mm_projector_type TAC`: Specifies the Temporal Alignment Connector for Libra.
- `--vision_tower microsoft/rad-dino`: Uses RAD-DINO as the chest X-rays encoder.
- `--mm_vision_select_layer all`: Selects specific vision layers (e.g., -1, -2) or "all" for all layers.
- `--validation_data_path ./path/`: Path to the validation data.
- `--compute_metrics True`: Optionally computes metrics during validation. Note that this can consume significant memory. If GPU memory is insufficient, it is recommended to either disable this option or use a smaller validation dataset.


## Evaluation
In Libra-v1.0, we evaluate models on the MIMIC-CXR test split for the findings section generation task. You can download the evaluation data [here](https://drive.google.com/file/d/1fy_WX616L8SgyAonadJ2fUIEaX0yrGrQ/view?usp=sharing). To ensure reproducibility and output quality, we evaluate our model using the beam search strategy.

### 1. Generate Libra responses.

```Shell
python -m libra.eval.eval_vqa_libra \
    --model-path X-iZhang/libra-v1.0-7b \
    --question-file libra_findings_section_eval.jsonl \
    --image-folder ./physionet.org/files/mimic-cxr-jpg/2.0.0 \
    --answers-file /path/to/answer-file.jsonl \
    --num_beams 10 \
    --length_penalty 2 \
    --num_return_sequences 3 \
    --max_new_tokens 1024 \
    --conv-mode libra_v1
```

You can evaluate Libra on your custom datasets by converting your dataset to the [JSONL format](https://github.com/X-iZhang/Libra/blob/main/CUSTOM_DATA.md#evaluation-dataset-format) and evaluating using [`eval_vqa_libra.py`](https://github.com/X-iZhang/Libra/blob/main/libra/eval/eval_vqa_libra.py).

Additionally, you can execute the evaluation using the command line. For detailed instructions, see [`libra_eval.sh`](https://github.com/X-iZhang/Libra/blob/main/scripts/eval/libra_eval.sh).

```bash
bash ./scripts/eval/libra_eval.sh beam
```

### 2. Evaluate the generated report.

In our case, you can directly use `libra_findings_section_eval.jsonl` and `answer-file.jsonl` for basic evaluation, using [`radiology_report.py`](https://github.com/X-iZhang/Libra/blob/main/libra/eval/radiology_report.py).

```Python
from libra.eval import evaluate_report

references = "libra_findings_section_eval.jsonl"
predictions = "answer-file.jsonl"

resul = evaluate_report(references=references, predictions=predictions)

# Evaluation scores
resul
{'BLEU1': 51.25,
 'BLEU2': 37.48,
 'BLEU3': 29.56,
 'BLEU4': 24.54,
 'METEOR': 48.90,
 'ROUGE-L': 36.66,
 'Bert_score': 62.50,
 'Temporal_entity_score': 35.34}
```
Or use the command line to evaluate multiple references and store the results in a `.csv` file. For detailed instructions, see [`get_eval_scores.sh`](https://github.com/X-iZhang/Libra/blob/main/scripts/eval/get_eval_scores.sh).

```bash
bash ./scripts/eval/get_eval_scores.sh
```
### Metrics
- Temporal Entity F1

The $F1_{temp}$ score includes common radiology-related keywords associated with temporal changes. You can use [`temporal_f1.py`](https://github.com/X-iZhang/Libra/blob/main/libra/eval/temporal_f1.py) as follows:

```Python
from libra.eval import temporal_f1_score

predictions = [
    "The pleural effusion has progressively worsened since previous scan.",
    "The pleural effusion is noted again on the current scan."
]
references = [
    "Compare with prior scan, pleural effusion has worsened.",
    "Pleural effusion has worsened."
]

tem_f1_score = temporal_f1_score(
    predictions=predictions,
    references=references
)

# Temporal Entity F1 score
tem_f1_score
{'f1': 0.500000000075,
 'prediction_entities': [{'worsened'}, set()],
 'reference_entities': [{'worsened'}, {'worsened'}]}
```

- Radiology-specific Metrics

Some specific metrics may require configurations that could conflict with Libra. It is recommended to follow the official guidelines and use separate environments for evaluation: [$RG_{ER}$](https://pypi.org/project/radgraph/0.1.13/), [$CheXpert-F1$](https://pypi.org/project/f1chexbert/), [$RadGraph-F1$, $RadCliQ_{0}$, $CheXbert-vector$](https://github.com/rajpurkarlab/CXR-Report-Metric).


<!-- ## Overview 🔬
We propose **Libra** (**L**everaging Temporal **I**mages for **B**iomedical **R**adiology **A**nalysis), a novel framework tailored for radiology report generation (RRG) that incorporates temporal change information to address the challenges of interpreting medical images effectively.

Libra leverages RAD-DINO, a pre-trained visual transformer, as its image encoder to generate robust and scalable image features. These features are further refined by a **Temporal Alignment Connector (TAC)**, a key innovation in Libra's architecture. The TAC comprises:
* **Layerwise Feature Extractor (LFE)**: Captures high-granularity image feature embeddings from the encoder.
* **Temporal Fusion Module (TFM)**: Integrates temporal references from prior studies to enhance temporal awareness and reasoning.

These refined features are fed into Meditron, a specialised medical large language model (LLM), to generate comprehensive, temporally-aware radiology reports. Libra’s modular design seamlessly integrates state-of-the-art open-source pre-trained models for both image and text, aligning them through a temporal-aware adapter to ensure robust cross-modal reasoning and understanding.

Through a two-stage training strategy, Libra demonstrates the powerful potential of multimodal large language models (MLLMs) in specialised radiology applications. Extensive experiments on the **MIMIC-CXR dataset** highlight Libra's performance, setting a new state-of-the-art benchmark among models of the same parameter scale.

## Contributions 🛠

* **Temporal Awareness**: Libra captures and synthesises temporal changes in medical images, addressing the challenge of handling prior study citations in RRG tasks.
* **Innovative Architecture**: The Temporal Alignment Connector (TAC) ensures high-granularity feature extraction and temporal integration, significantly enhancing cross-modal reasoning capabilities.
* **State-of-the-Art Performance**: Libra achieves outstanding results on the MIMIC-CXR dataset, outperforming existing MLLMs in both accuracy and temporal reasoning. -->
 
<!-- ## Project Status 🚀

The code is currently being organised and will be available soon. **Please check back later for updates!**

We are actively preparing the repository to ensure a seamless experience for contributors and users. Stay tuned for the initial release and future enhancements. -->


<!-- ![architecture](./assets/libra_architecture.png) -->

## Acknowledgements 🙏

We sincerely thank the following projects for their contributions to **Libra**:

* [LLaVA](https://github.com/haotian-liu/LLaVA): A Large Language and Vision Assistant, laying the groundwork for multimodal understanding.
* [FastChat](https://github.com/lm-sys/FastChat): An Open Platform for Training, Serving, and Evaluating Large Language Model based Chatbots.
* [LLaMA](https://github.com/facebookresearch/llama): Open and efficient foundation language models that inspired our core language processing capabilities.
* [MEDITRON](https://github.com/epfLLM/meditron): Open and efficient medical Large language models.
* [RAD-DINO](https://huggingface.co/microsoft/rad-dino): An open and efficient biomedical image encoder, enabling robust radiological analysis.

## Citation ✒️

If you find our paper and code useful in your research and applications, please cite using this BibTeX:
```BibTeX
@misc{zhang2024libraleveragingtemporalimages,
      title={Libra: Leveraging Temporal Images for Biomedical Radiology Analysis}, 
      author={Xi Zhang and Zaiqiao Meng and Jake Lever and Edmond S. L. Ho},
      year={2024},
      eprint={2411.19378},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2411.19378}, 
}
```
