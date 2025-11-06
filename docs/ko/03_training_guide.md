# Libra 학습 가이드 (Stage 1 & Stage 2)

## 목차
- [2-Stage 학습 전략 개요](#2-stage-학습-전략-개요)
- [Stage 1: Visual Feature Alignment](#stage-1-visual-feature-alignment)
- [Stage 2: Downstream Task Fine-tuning](#stage-2-downstream-task-fine-tuning)
- [학습 메커니즘 상세](#학습-메커니즘-상세)
- [완전한 학습 체크리스트](#완전한-학습-체크리스트)

---

## 2-Stage 학습 전략 개요

### 전체 비교표

| 항목 | **Stage 1: Pretrain** | **Stage 2: Finetune** |
|------|----------------------|----------------------|
| **한글명** | Visual Feature Alignment | Downstream Task Fine-tuning |
| **목적** | TAC가 Vision → LLM 변환 학습 | LLM이 Report Generation 학습 |
| **입력 데이터** | `libra_alignment_train.json` (780 MB) | `libra_findings_section_train.json` (159 MB) |
| **데이터 내용** | RRG + VQA 혼합 (다양한 task) | Findings section만 (특정 task) |
| **학습 대상** | 🔥 **TAC만** | 🔥 **LLM만 (LoRA)** |
| **Frozen** | Vision Encoder + LLM | Vision Encoder + TAC |
| **Epochs** | 1 | 3 |
| **학습 시간** | ~385시간 (16일) | ~213시간 (9일) |
| **Learning Rate** | 2e-5 | 2e-5 |
| **출력물** | `mm_tac_projector.bin` | `adapter_model.bin` (LoRA) |
| **스크립트** | `pretrain.sh` | `finetune_lora.sh` |

---

## Stage 1: Visual Feature Alignment

### 목적

```
[RAD-DINO features] → [TAC 학습 중] → [Meditron 입력 형식]
                          ↑
                    이 변환을 배운다!
```

Vision encoder의 출력(이미지 특징)을 LLM이 이해할 수 있는 형태로 변환하는 방법을 학습합니다.

---

### 데이터셋

**파일**: `libra_alignment_train.json` (780 MB)

**구성**:
- Radiology Report Generation (RRG)
- Visual Question Answering (VQA)
  - MIMIC-Diff-VQA
  - MIMIC-Ext-MIMIC-CXR-VQA
- 모든 섹션 포함 (Findings, Impression, Indication, History 등)

**목적**: TAC가 다양한 시각적 이해 능력 학습

**다운로드**:
```bash
# Google Drive
wget https://drive.google.com/file/d/1AIT1b3eRXgJFp3FJmHci3haTunK1NTMA/
```

---

### 학습 설정

**스크립트**: `scripts/pretrain.sh`

```bash
###############################################################################
# Stage 1 Hyperparameters
###############################################################################

# ═══ 데이터 ═══
TRAIN_DATA="./data/libra_alignment_train.json"
VAL_DATA="./data/libra_alignment_valid.json"
IMG_FOLDER="./data/mimic-cxr-jpg/2.0.0"

# ═══ 모델 구성 ═══
MODEL_VERSION="epfl-llm/meditron-7b"
VISION_TOWER="microsoft/rad-dino"
PROMPT_VERSION="libra_v1"

# ═══ 학습 하이퍼파라미터 ═══
NUM_EPOCHS=1                    # 1 epoch만
TRAIN_BSZ=16                    # Per-device batch size
EVAL_BSZ=4
GRAD_ACC_STEPS=1
LR=2e-5                         # Learning rate
WEIGHT_DECAY=0.
WARMUP_RATIO=0.03               # 3% warmup
LR_SCHEDULER="cosine"
MAX_LENGTH=2048

# ═══ TAC 학습 플래그 ═══
--freeze_backbone True          # ❄️ LLM frozen
--tune_mm_mlp_adapter True      # 🔥 TAC trainable
--freeze_mm_mlp_adapter False   # 🔥 TAC trainable
--mm_projector_type TAC
--mm_vision_select_layer all    # All 12 layers

# ═══ 최적화 설정 ═══
--bf16 True
--gradient_checkpointing True
--deepspeed ./scripts/zero2.json
```

---

### 학습되는 파라미터

| 블록 | 상태 | 파라미터 수 |
|------|------|-----------|
| Vision Encoder (RAD-DINO) | ❄️ Frozen | 87M |
| **TAC (mm_projector)** | 🔥 **Trainable** | **~50M** |
| LLM (Meditron-7B) | ❄️ Frozen | 7B |

**코드**: `train.py:1702-1705`

```python
if model_args.tune_mm_mlp_adapter:
    model.requires_grad_(False)  # 모든 파라미터 freeze
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = True   # TAC만 학습
```

---

### 출력물

```
./checkpoints/libra-v1.0-7b-pretrain/
├── mm_tac_projector.bin        # ⭐ TAC weights (핵심!)
├── config.json                 # 모델 설정
└── training_state.json         # 학습 상태
```

**사용 방법**:
```bash
# Stage 2에서 로드
--model_name_or_path ./checkpoints/libra-v1.0-7b-pretrain

# 또는 pretrained projector 사용
--model_name_or_path epfl-llm/meditron-7b
--pretrain_mm_mlp_adapter ./mm_tac_projector.bin
```

---

### 학습 시간

**A6000 1개 기준**: ~385 hours (16일)

**이유**:
- 대규모 데이터 (780 MB)
- TAC의 복잡한 attention 구조
- 1 epoch만 (overfitting 방지)

---

## Stage 2: Downstream Task Fine-tuning

### 목적

```
[이미지] → [TAC 고정] → [LLM LoRA 학습] → [Radiology Report]
                              ↑
                    Report 생성을 배운다!
```

LLM이 X-ray 이미지를 보고 정확한 Radiology Report를 생성하는 방법을 학습합니다.

---

### 데이터셋

**파일**: `libra_findings_section_train.json` (159 MB)

**구성**:
- Findings section generation만 집중
- MIMIC-CXR의 특정 섹션
- Temporal comparison 포함

**목적**: 특정 downstream task에 특화

**다운로드**:
```bash
wget https://drive.google.com/file/d/1rJ3G4uiHlzK_P6ZBUbAi-cDaWV-o6fcz/
```

---

### 학습 설정

**스크립트**: `scripts/finetune_lora.sh`

```bash
###############################################################################
# Stage 2 Hyperparameters
###############################################################################

# ═══ 데이터 ═══
TRAIN_DATA="./data/libra_findings_section_train.json"
VAL_DATA="./data/libra_findings_section_valid.json"
IMG_FOLDER="./data/mimic-cxr-jpg/2.0.0"

# ═══ 모델 구성 ═══
MODEL_VERSION="./checkpoints/libra-v1.0-7b-pretrain"  # Stage 1 출력
VISION_TOWER="microsoft/rad-dino"
PROMPT_VERSION="libra_v1"

# ═══ LoRA 하이퍼파라미터 ⭐ ═══
LORA_R=128                      # LoRA rank
LORA_ALPHA=256                  # LoRA alpha (scaling=2.0)
LORA_DROPOUT=0.05
MM_PROJECTOR_LR=2e-5            # TAC learning rate (optional)

# ═══ 학습 하이퍼파라미터 ═══
NUM_EPOCHS=3                    # 3 epochs
TRAIN_BSZ=16
EVAL_BSZ=4
GRAD_ACC_STEPS=1
LR=2e-5                         # LoRA learning rate
WEIGHT_DECAY=0.
WARMUP_RATIO=0.03
LR_SCHEDULER="cosine"
MAX_LENGTH=2048

# ═══ LoRA 학습 플래그 ═══
--lora_enable True              # 🔥 LoRA 활성화
--lora_r ${LORA_R}
--lora_alpha ${LORA_ALPHA}
--freeze_backbone True          # ❄️ LLM backbone frozen
--tune_mm_mlp_adapter False     # ❄️ TAC frozen
--freeze_mm_mlp_adapter True    # ❄️ TAC frozen

# ═══ 최적화 설정 ═══
--bf16 True
--gradient_checkpointing True
--deepspeed ./scripts/zero3.json  # Zero-3 (더 메모리 효율)
```

---

### 학습되는 파라미터

| 블록 | 상태 | 파라미터 수 |
|------|------|-----------|
| Vision Encoder | ❄️ Frozen | 87M |
| TAC | ❄️ Frozen | 50M |
| LLM Backbone | ❄️ Frozen | 7B |
| **LoRA Adapters** | 🔥 **Trainable** | **~224M** |

**LoRA 적용 위치**:
- LLM의 모든 Linear layers
- q_proj, k_proj, v_proj, o_proj (Attention)
- gate_proj, up_proj, down_proj (MLP)

---

### 출력물

```
./checkpoints/libra-v1.0-7b-lora/
├── adapter_model.bin           # ⭐ LoRA weights (224M)
├── adapter_config.json         # LoRA config
├── non_lora_trainables.bin     # TAC (복사본)
├── config.json
└── training_state.json
```

**사용 방법**:
```python
from libra.eval import libra_eval

result = libra_eval(
    model_path="./checkpoints/libra-v1.0-7b-lora",
    model_base="epfl-llm/meditron-7b",
    image_file=["current.jpg", "prior.jpg"],
    query="Describe the findings..."
)
```

---

### 학습 시간

**A6000 1개 기준**: ~213 hours (9일)

**이유**:
- 작은 데이터셋 (159 MB)
- 하지만 3 epochs
- LoRA로 메모리 효율적

---

## 학습 메커니즘 상세

### 왜 2-Stage가 필요한가?

#### 한 번에 학습하면 안되나요?

**문제점**:

1. **Catastrophic Forgetting**
   - TAC + LLM 동시 학습 시 Vision feature 변화에 LLM이 적응 못함

2. **학습 불안정**
   - 두 모듈의 학습 속도가 달라 수렴 어려움

3. **메모리 부족**
   - 모든 gradient 동시 계산 시 메모리 부족

#### 2-Stage의 장점

```
Stage 1: TAC만 집중 학습
         ↓
    안정적인 feature 생성
         ↓
Stage 2: LLM이 고정된 feature로 학습
         ↓
    안정적 수렴 + 높은 성능
```

---

### Learning Rate 설정

**Multi-LR 시스템** (libra_trainer.py:165-194)

```python
# Stage 1
optimizer_grouped_parameters = [
    {
        "params": [...],  # TAC weights
        "weight_decay": 0.0,
        "lr": 2e-5  # TAC LR
    }
]

# Stage 2
optimizer_grouped_parameters = [
    {
        "params": [...],  # LoRA weights
        "weight_decay": 0.0,
        "lr": 2e-5  # LoRA LR
    },
    {
        "params": [...],  # mm_projector (optional)
        "weight_decay": 0.0,
        "lr": 2e-6  # TAC LR (더 낮게)
    }
]
```

---

### Stage별 학습 비교표

| 측면 | Stage 1 | Stage 2 |
|------|---------|---------|
| **Batch Size** | 16 | 16 |
| **Global Batch** | 16 | 16 |
| **Epochs** | 1 | 3 |
| **LR** | 2e-5 | 2e-5 |
| **Scheduler** | Cosine | Cosine |
| **Warmup** | 3% | 3% |
| **Weight Decay** | 0 | 0 |
| **Max Length** | 2048 | 2048 |
| **DeepSpeed** | Zero-2 | Zero-3 |
| **Precision** | BF16 | BF16 |

---

## 완전한 학습 체크리스트

### Step 0: 환경 준비

```bash
# 1. 리포지토리 클론
git clone https://github.com/X-iZhang/Libra.git
cd Libra

# 2. 환경 설치
conda create -n libra python=3.10 -y
conda activate libra
pip install -e ".[train,eval]"
pip install flash-attn --no-build-isolation

# 3. CUDA 확인
python -c "import torch; print(torch.cuda.is_available())"
```

---

### Step 1: 데이터 준비

```bash
# 1. MIMIC-CXR 다운로드
# https://physionet.org/content/mimic-cxr-jpg/2.0.0/

# 2. Libra annotation 다운로드
mkdir -p ./data
cd ./data

# Stage 1 데이터
wget https://drive.google.com/file/d/1AIT1b3eRXgJFp3FJmHci3haTunK1NTMA/
wget https://drive.google.com/file/d/1nvbUoDmw7j4HgXwZWiiACIhvZ6BvR2LX/

# Stage 2 데이터
wget https://drive.google.com/file/d/1rJ3G4uiHlzK_P6ZBUbAi-cDaWV-o6fcz/
wget https://drive.google.com/file/d/1IYwQS23veOU5SXWGYiTyq9VHUwkVESfD/

# 구조 확인
tree -L 2
# ./data/
# ├── physionet.org/files/mimic-cxr-jpg/2.0.0/
# ├── libra_alignment_train.json
# ├── libra_alignment_valid.json
# ├── libra_findings_section_train.json
# └── libra_findings_section_valid.json
```

---

### Step 2: Stage 1 학습 (TAC)

```bash
# 1. pretrain.sh 수정
vim scripts/pretrain.sh

# 수정할 부분:
# - TRAIN_DATA 경로
# - VAL_DATA 경로
# - IMG_FOLDER 경로
# - OUTPUT_DIR 경로

# 2. 학습 시작
bash scripts/pretrain.sh

# 3. 모니터링 (별도 터미널)
watch -n 1 nvidia-smi
tail -f checkpoints/libra-v1.0-7b-pretrain/training.log

# 4. 결과 확인 (385시간 후)
ls ./checkpoints/libra-v1.0-7b-pretrain/
# mm_tac_projector.bin  ← 이것이 핵심!
```

**예상 소요 시간**:
- A6000 1개: 16일
- H100 1개: 2.7일

---

### Step 3: Stage 2 학습 (LoRA)

```bash
# 1. finetune_lora.sh 수정
vim scripts/finetune_lora.sh

# 수정할 부분:
# - MODEL_VERSION="./checkpoints/libra-v1.0-7b-pretrain"
# - TRAIN_DATA 경로
# - VAL_DATA 경로
# - OUTPUT_DIR 경로

# 2. 학습 시작
bash scripts/finetune_lora.sh

# 3. 결과 확인 (213시간 후)
ls ./checkpoints/libra-v1.0-7b-lora/
# adapter_model.bin        ← LoRA weights
# adapter_config.json
# non_lora_trainables.bin  ← TAC (복사본)
```

**예상 소요 시간**:
- A6000 1개: 9일
- H100 1개: 1.5일

---

### Step 4: Inference 테스트

```python
from libra.eval import libra_eval

# 테스트
result = libra_eval(
    model_path="./checkpoints/libra-v1.0-7b-lora",
    model_base="epfl-llm/meditron-7b",
    image_file=["./examples/current.jpg", "./examples/prior.jpg"],
    query="Describe the findings in comparison to the prior image.",
    conv_mode="libra_v1"
)

print(result)
```

---

### Step 5: 평가

```bash
# 1. Generate predictions
python -m libra.eval.run_libra \
    --model-path ./checkpoints/libra-v1.0-7b-lora \
    --model-base epfl-llm/meditron-7b \
    --question-file ./data/libra_findings_section_eval.jsonl \
    --image-folder ./data/mimic-cxr-jpg/2.0.0 \
    --answers-file ./results/answer-file.jsonl

# 2. Evaluate
python libra/eval/radiology_report.py \
    --references ./data/libra_findings_section_eval.jsonl \
    --predictions ./results/answer-file.jsonl
```

---

## Stage 건너뛰기 옵션

### Stage 1 건너뛰기 (공개 Projector 사용)

```bash
# 1. Pretrained TAC 다운로드
wget https://huggingface.co/X-iZhang/libra-v1.0-7b/resolve/main/mm_tac_projector.bin

# 2. Stage 2 바로 실행
# finetune_lora.sh에 추가:
--model_name_or_path epfl-llm/meditron-7b \
--pretrain_mm_mlp_adapter ./mm_tac_projector.bin \
```

**절약 시간**: 16일! (Stage 1 생략)

---

### Stage 2 건너뛰기 (공개 LoRA 사용)

```bash
# 1. Pretrained LoRA 다운로드
git lfs install
git clone https://huggingface.co/X-iZhang/libra-v1.0-7b

# 2. 바로 Inference
python -m libra.eval.run_libra \
    --model-path ./libra-v1.0-7b \
    --model-base epfl-llm/meditron-7b \
    ...
```

**절약 시간**: 9일! (Stage 2 생략)

---

## 요약

| 단계 | 시간 | GPU | 누적 | 건너뛰기 가능 |
|------|------|-----|------|-------------|
| Stage 1 | 385h (16일) | A6000 x1 | 16일 | ✅ (공개 TAC 사용) |
| Stage 2 | 213h (9일) | A6000 x1 | **25일** | ✅ (공개 LoRA 사용) |

**핵심**:
- Stage 1: TAC가 vision-language alignment 학습
- Stage 2: LLM이 report generation 학습
- 독립적이지만 순차적
- 공개 weights로 각 stage 건너뛰기 가능

---

## 참고 자료

- **논문**: [Libra (ACL 2025)](https://arxiv.org/abs/2411.19378)
- **코드**: [GitHub](https://github.com/X-iZhang/Libra)
- **모델**: [HuggingFace](https://huggingface.co/X-iZhang/libra-v1.0-7b)
- **데이터**: Google Drive (README 참조)
