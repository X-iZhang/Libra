# Libra 완전 설정 가이드 & LoRA 구현

## 목차
- [Vision Encoder별 Input Size](#vision-encoder별-input-size)
- [사용 가능한 Vision Encoders](#사용-가능한-vision-encoders)
- [사용 가능한 LLM 모델](#사용-가능한-llm-모델)
- [하이퍼파라미터 설정](#하이퍼파라미터-설정)
- [LoRA 구현 상세](#lora-구현-상세)

---

## Vision Encoder별 Input Size

### 완전히 다릅니다!

| Vision Encoder | Input Size | Patch Size | Num Patches | Hidden Dim | Num Layers |
|---------------|-----------|------------|-------------|------------|------------|
| **RAD-DINO** | **518×518** | 14×14 | 1369 (37×37) | **768** | **12** |
| **CLIP-ViT-Large-336** | **336×336** | 14×14 | 576 (24×24) | **1024** | 24 |
| **CLIP-ViT-Large-224** | 224×224 | 14×14 | 256 (16×16) | 1024 | 24 |
| **BiomedCLIP** | 224×224 | 16×16 | 196 (14×14) | 768 | 12 |
| **SigLIP** | 384×384 | 16×16 | 576 (24×24) | 1152 | 27 |

**코드에서 자동 처리** (dino_encoder.py:135):
```python
@property
def num_patches(self):
    return (self.config.image_size // self.config.patch_size) ** 2
```

**주의사항**:
- TAC는 RAD-DINO (12 layers) 기준으로 하드코딩됨 (builder.py:29)
- 다른 encoder 사용 시 `layers_number` 수정 필요

---

## 사용 가능한 Vision Encoders

| Encoder | HuggingFace ID | Image Size | Hidden Size | 특징 |
|---------|---------------|------------|-------------|------|
| **RAD-DINO** ⭐ | `microsoft/rad-dino` | 518×518 | 768 | Radiology 전문, SOTA |
| **CLIP-Large** | `openai/clip-vit-large-patch14-336` | 336×336 | 1024 | 범용, 대규모 |
| **CLIP-Base** | `openai/clip-vit-base-patch16` | 224×224 | 768 | 경량 |
| **BiomedCLIP** | `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` | 224×224 | 768 | 의료 전문 |
| **SigLIP** | `google/siglip-so400m-patch14-384` | 384×384 | 1152 | 최신, 효율적 |

**설정 방법**:
```bash
# pretrain.sh or finetune_lora.sh
VISION_TOWER="microsoft/rad-dino"  # 변경 가능
```

---

## 사용 가능한 LLM 모델

| LLM Family | 파일 | HuggingFace Example | Hidden Size |
|-----------|------|-------------------|-------------|
| **LLaMA** | `libra_llama.py` | `meta-llama/Llama-2-7b-chat-hf`<br>`meta-llama/Llama-3-8B-Instruct`<br>`meta-llama/Llama-3.2-3B-Instruct` | 4096 (7B)<br>3072 (3B) |
| **Meditron** ⭐ | `libra_llama.py` | `epfl-llm/meditron-7b` | 4096 |
| **Vicuna** | `libra_llama.py` | `lmsys/vicuna-7b-v1.5` | 4096 |
| **Mistral** | `libra_mistral.py` | `mistralai/Mistral-7B-Instruct-v0.2` | 4096 |
| **Qwen2** | `libra_qwen2.py` | `Qwen/Qwen2.5-3B-Instruct` | 2048 |
| **Qwen3** | `libra_qwen3.py` | `Qwen/Qwen3-4B-Instruct-2507` | 3072 |
| **Phi-3** | `libra_phi3.py` | `microsoft/Phi-3-mini-4k-instruct` | 3072 |
| **Gemma** | `libra_gemma.py` | `google/gemma-2-2b-it` | 2304 |

**설정 방법** (pretrain.sh:18-42):
```bash
# Meditron (권장)
MODEL_VERSION="epfl-llm/meditron-7b"
PROMPT_VERSION="libra_v1"

# LLaMA-3
MODEL_VERSION="meta-llama/Llama-3-8B-Instruct"
PROMPT_VERSION="llama_3"

# Qwen3
MODEL_VERSION="Qwen/Qwen3-4B-Instruct-2507"
PROMPT_VERSION="qwen"
```

---

## 하이퍼파라미터 설정

### Stage 1 (Pretrain) - `scripts/pretrain.sh`

```bash
###############################################################################
# Stage 1: Visual Feature Alignment (TAC 학습)
###############################################################################

# ═══ 데이터 ═══
TRAIN_DATA="./path/to/libra_alignment_train.json"
VAL_DATA="./path/to/libra_alignment_valid.json"
IMG_FOLDER="./path/to/mimic-cxr-jpg/2.0.0"

# ═══ 모델 구성 ═══
MODEL_VERSION="epfl-llm/meditron-7b"
VISION_TOWER="microsoft/rad-dino"
PROMPT_VERSION="libra_v1"

# ═══ 학습 하이퍼파라미터 ═══
NUM_EPOCHS=1                    # Epoch
TRAIN_BSZ=16                    # Per-device batch size
EVAL_BSZ=4                      # Eval batch size
GRAD_ACC_STEPS=1                # Gradient accumulation
LR=2e-5                         # Learning rate (TAC)
WEIGHT_DECAY=0.                 # Weight decay
WARMUP_RATIO=0.03               # Warmup (3%)
LR_SCHEDULER="cosine"           # Scheduler type
MAX_LENGTH=2048                 # Token length

# ═══ 최적화 설정 ═══
DEEPSPEED_CONFIG="./scripts/zero2.json"
BF16=True                       # bfloat16 precision
TF32=True                       # TF32 for A100
GRADIENT_CHECKPOINTING=True     # Memory 절약

# ═══ TAC 학습 플래그 ═══
--freeze_backbone True          # ❄️ LLM frozen
--tune_mm_mlp_adapter True      # 🔥 TAC trainable
--freeze_mm_mlp_adapter False   # 🔥 TAC trainable
--mm_projector_type TAC         # Projector type
--mm_vision_select_layer all    # All 12 layers

# ═══ 저장/평가 ═══
--save_steps 20000              # Save every 20K steps
--save_total_limit 1            # Keep only 1 checkpoint
--eval_strategy "steps"         # Eval during training
--eval_steps 0.01               # Eval every 1%
--compute_metrics True          # Calculate BLEU, etc.
```

---

### Stage 2 (LoRA Finetune) - `scripts/finetune_lora.sh`

```bash
###############################################################################
# Stage 2: Downstream Task Fine-tuning (LLM 학습)
###############################################################################

# ═══ 데이터 ═══
TRAIN_DATA="./path/to/libra_findings_section_train.json"
VAL_DATA="./path/to/libra_findings_section_valid.json"
IMG_FOLDER="./path/to/mimic-cxr-jpg/2.0.0"

# ═══ 모델 구성 ═══
MODEL_VERSION="./checkpoints/libra-v1.0-7b-pretrain"
VISION_TOWER="microsoft/rad-dino"
PROMPT_VERSION="libra_v1"

# ═══ LoRA 하이퍼파라미터 ⭐ ═══
LORA_R=128                      # LoRA rank
LORA_ALPHA=256                  # LoRA alpha (scaling=2.0)
LORA_DROPOUT=0.05               # LoRA dropout
MM_PROJECTOR_LR=2e-5            # TAC learning rate (optional)

# ═══ 학습 하이퍼파라미터 ═══
NUM_EPOCHS=3                    # Epoch (더 많음)
TRAIN_BSZ=16                    # Per-device batch size
EVAL_BSZ=4
GRAD_ACC_STEPS=1
LR=2e-5                         # Learning rate (LoRA)
WEIGHT_DECAY=0.
WARMUP_RATIO=0.03
LR_SCHEDULER="cosine"
MAX_LENGTH=2048

# ═══ 최적화 설정 ═══
DEEPSPEED_CONFIG="./scripts/zero3.json"  # Zero-3
BF16=True
TF32=True
GRADIENT_CHECKPOINTING=True

# ═══ LoRA 학습 플래그 ═══
--lora_enable True              # 🔥 LoRA 활성화
--lora_r ${LORA_R}
--lora_alpha ${LORA_ALPHA}
--mm_projector_lr ${MM_PROJECTOR_LR}
--freeze_backbone True          # ❄️ LLM backbone frozen
--tune_mm_mlp_adapter False     # ❄️ TAC frozen
--freeze_mm_mlp_adapter True    # ❄️ TAC frozen

# ═══ 저장/평가 ═══
--save_steps 2000
--save_total_limit 1
--eval_strategy "steps"
--eval_steps 0.01
--compute_metrics True
```

---

## LoRA 구현 상세

### LoRA 초기화 과정

**위치**: `train.py:1626-1642`

```python
if training_args.lora_enable:
    from peft import LoraConfig, get_peft_model

    # Step 1: LoRA Config 생성
    lora_config = LoraConfig(
        r=training_args.lora_r,              # 128 (rank)
        lora_alpha=training_args.lora_alpha, # 256 (scaling factor)
        target_modules=find_all_linear_names(model),
        lora_dropout=training_args.lora_dropout,      # 0.05
        bias=training_args.lora_bias,                 # "none"
        task_type="CAUSAL_LM",
    )

    # Step 2: PEFT 모델로 변환
    model = get_peft_model(model, lora_config)
```

---

### Target Modules 자동 탐지

**위치**: `train.py:222-235`

```python
def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()

    # ❌ 제외할 모듈
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']

    for name, module in model.named_modules():
        # Vision/Projector 모듈은 건너뛰기
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue

        # ✅ Linear 레이어만 선택
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    # lm_head도 제외
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')

    return list(lora_module_names)
```

**결과** (Meditron-7B):
```python
target_modules = [
    'q_proj',      # Query projection
    'k_proj',      # Key projection
    'v_proj',      # Value projection
    'o_proj',      # Output projection
    'gate_proj',   # Gate projection (MLP)
    'up_proj',     # Up projection (MLP)
    'down_proj'    # Down projection (MLP)
]
```

---

### LoRA 적용 구조

#### 원래 Attention Layer
```
Input (4096)
    ↓
[Q Projection] (4096 → 4096)  ← 32M params (frozen)
```

#### LoRA 적용 후
```
Input (4096)
    ↓
[Q Projection (frozen)] ─┬─→ Output
                          │
                          └─→ [LoRA_A: 4096→128] → [LoRA_B: 128→4096]
                               (524K params, trainable)
```

**수식**:
```
h = W₀x + (α/r) · ΔWx
  = W₀x + (256/128) · B·A·x

where:
  W₀: frozen weights (4096×4096)
  A: trainable (4096×128)
  B: trainable (128×4096)
  α: scaling factor (256)
  r: rank (128)
```

---

### 학습되는 파라미터 계산

**Meditron-7B 기준**:

```python
# 32개 Transformer layers
# 각 layer당:
#   - q_proj, k_proj, v_proj, o_proj (Attention): 4개
#   - gate_proj, up_proj, down_proj (MLP): 3개
# 총: 7개 modules per layer

# 각 module당 LoRA 파라미터:
#   A: 4096 × 128 = 524,288
#   B: 128 × 4096 = 524,288
#   Total per module: 1,048,576 (~1M)

# 전체 학습 파라미터:
32 layers × 7 modules × 1M = 224M params
```

**메모리 효율**:
```
Full finetuning: 7B × 4 bytes = 28GB
LoRA: 224M × 4 bytes = 896MB

절약: 96.8%!
```

---

### LoRA Scaling Factor

```python
# finetune_lora.sh
LORA_R=128
LORA_ALPHA=256

# Effective scaling
scaling = α / r = 256 / 128 = 2.0
```

**의미**:
- LoRA 출력이 frozen weight 출력의 **2배** 스케일
- α가 클수록 LoRA 기여도 증가

---

### 저장 메커니즘

**위치**: `train.py:1816-1826`

```python
if training_args.lora_enable:
    # LoRA 파라미터만 저장
    state_dict = get_peft_state_maybe_zero_3(
        model.named_parameters(),
        training_args.lora_bias
    )
    # 파일명: adapter_model.bin (~900MB)

    # Non-LoRA trainables 저장
    non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
        model.named_parameters()
    )
    torch.save(non_lora_state_dict,
               os.path.join(output_dir, 'non_lora_trainables.bin'))
```

**저장 구조**:
```
./checkpoints/libra-v1.0-7b-lora/
├── adapter_model.bin          # LoRA weights (224M params)
├── adapter_config.json        # LoRA config
├── non_lora_trainables.bin    # mm_projector
└── config.json                # Model config
```

---

### Inference 시 로딩

**위치**: `builder.py:51-83`

```python
if 'lora' in model_name.lower() and model_base is not None:
    # Step 1: Base model 로드
    model = LibraLlamaForCausalLM.from_pretrained(
        model_base,  # "epfl-llm/meditron-7b"
        config=lora_cfg_pretrained
    )

    # Step 2: Non-LoRA trainables 로드
    non_lora_trainables = torch.load(
        os.path.join(model_path, 'non_lora_trainables.bin')
    )
    model.load_state_dict(non_lora_trainables, strict=False)

    # Step 3: LoRA weights 로드
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, model_path)

    # Step 4: LoRA merge (inference 속도 향상)
    model = model.merge_and_unload()
```

---

## 각 블록별 학습 설정

| 블록 | Stage 1 (Pretrain) | Stage 2 (LoRA Finetune) |
|------|-------------------|------------------------|
| **Vision Encoder** | ❄️ Frozen | ❄️ Frozen |
| **TAC (mm_projector)** | 🔥 **Trainable**<br>LR: 2e-5 | ❄️ Frozen |
| **LLM Backbone** | ❄️ Frozen | ❄️ Frozen |
| **LoRA Adapters** | ❌ 없음 | 🔥 **Trainable**<br>LR: 2e-5<br>Rank: 128<br>Alpha: 256 |

---

## Libra vs Standard LoRA 비교

| 항목 | Standard LoRA | Libra LoRA |
|------|--------------|-----------|
| **Target** | q_proj, v_proj만 | 7개 modules |
| **Rank** | 8-64 | **128** |
| **Alpha** | r과 동일 | **256** (2×r) |
| **적용 레이어** | Attention만 | Attention + MLP |
| **파라미터 수** | ~50M | **224M** |
| **성능** | 적당 | **높음** |

---

## 핵심 설정 요약

| 설정 | Stage 1 | Stage 2 (LoRA) | 비고 |
|------|---------|---------------|------|
| **Learning Rate** | 2e-5 | 2e-5 | 동일 |
| **Batch Size** | 16 | 16 | Global |
| **Epochs** | 1 | 3 | Stage 2가 3배 |
| **Max Length** | 2048 | 2048 | Token limit |
| **LoRA Rank** | - | 128 | High rank |
| **LoRA Alpha** | - | 256 | Scaling=2.0 |
| **DeepSpeed** | Zero-2 | Zero-3 | Stage 2 더 효율 |
| **Precision** | BF16 | BF16 | A100/A6000 |

---

## 참고 자료

- **논문**: [Libra (ACL 2025)](https://arxiv.org/abs/2411.19378)
- **코드**: [GitHub](https://github.com/X-iZhang/Libra)
- **LoRA 논문**: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- **PEFT**: [HuggingFace PEFT Library](https://github.com/huggingface/peft)
