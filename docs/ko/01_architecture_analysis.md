# Libra 아키텍처 & 코드 구조 완전 분석

## 목차
- [논문 핵심 (ACL 2025)](#논문-핵심-acl-2025)
- [전체 아키텍처](#전체-아키텍처)
- [TAC 상세 분석](#tac-상세-분석)
- [코드 구조](#코드-구조)
- [차원 변환 추적](#차원-변환-추적)

---

## 논문 핵심 (ACL 2025)

### 문제 정의
기존 Radiology Report Generation 방법들의 한계:
1. **Single-image analysis**: 과거 이미지 무시
2. **Rule-based heuristics**: 단순 규칙으로 다중 이미지 처리
3. **Temporal reasoning 부족**: 시간에 따른 변화 포착 실패

### Libra의 핵심 기여
1. **TAC (Temporal Alignment Connector)**: 시간적 변화를 정확히 포착하는 novel architecture
2. **Radiology-specific encoder**: RAD-DINO 활용
3. **SOTA 성능**: MIMIC-CXR에서 기존 방법 압도

**논문**: [arxiv:2411.19378](https://arxiv.org/abs/2411.19378)

---

## 전체 아키텍처

### 파이프라인 플로우

```
Input: Current CXR + Prior CXR
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 1. Vision Encoder (RAD-DINO)                                │
│    - 각 이미지를 12개 layer features로 인코딩                │
│    - Output: [12 layers, 1369 patches, 768 dim]            │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. TAC (Temporal Alignment Connector) ⭐ 핵심               │
│    ┌─────────────────────────────────────────┐             │
│    │ 2a. LFE (Layerwise Feature Extractor)  │             │
│    │     - 12 layers → 1 optimal layer       │             │
│    └─────────────────────────────────────────┘             │
│    ┌─────────────────────────────────────────┐             │
│    │ 2b. TFM (Temporal Feature Matching)    │             │
│    │     - Self-Attention (Current)          │             │
│    │     - Self-Attention (Prior)            │             │
│    │     - Cross-Attention (Cur ↔ Prior)     │             │
│    │     - Cosine Similarity Weighting       │             │
│    └─────────────────────────────────────────┘             │
│    - Output: [1369 patches, 4096 dim]                      │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. LLM (Meditron-7B)                                        │
│    - Input: Image tokens + Text prompt                      │
│    - Output: Radiology Report                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 코드 구조

### 디렉토리 구조
```
libra/model/
├── libra_arch.py                          # 메인 아키텍처 (200줄)
├── builder.py                             # 모델 빌더
├── multimodal_encoder/                    # Vision Encoders
│   ├── builder.py                         # Encoder 선택기
│   ├── dino_encoder.py                    # RAD-DINO wrapper (139줄)
│   ├── clip_encoder.py                    # CLIP wrapper (125줄)
│   ├── siglip_encoder.py                  # SigLIP wrapper
│   └── open_clip_encoder/                 # OpenCLIP variants
├── multimodal_projector/                  # TAC 구현
│   └── builder.py                         # ⭐ TAC 핵심 (166줄)
└── language_model/                        # LLM wrappers
    ├── libra_llama.py                     # LLaMA 계열
    ├── libra_mistral.py                   # Mistral
    ├── libra_phi3.py                      # Phi-3
    ├── libra_qwen2.py                     # Qwen2
    └── libra_gemma.py                     # Gemma
```

**총 코드 라인**: 2,220 lines

---

## TAC 상세 분석

### 1. LFE (Layerwise Feature Extractor)

**위치**: `libra/model/multimodal_projector/builder.py:31-42`

**목적**: 12개 layer 중 가장 유용한 정보 추출

**구조**:
```python
# 12개 layer를 1개로 압축
LFE = Sequential(
    # 12 → 6 channels
    SqueezeExcitation(12, 6) + Conv2d(12, 6),
    # 6 → 3 channels
    SqueezeExcitation(6, 3) + Conv2d(6, 3),
    # 3 → 1 channel
    SqueezeExcitation(3, 1) + Conv2d(3, 1)
)
```

**입력**: `[batch, 12, 1369, 768]` (12 layers)
**출력**: `[batch, 1369, 768]` (1 optimal layer)

**핵심 기법**:
- **Squeeze-Excitation**: Channel-wise attention으로 중요한 layer 강조
- **1x1 Convolution**: 차원 축소

---

### 2. TFM (Temporal Feature Matching)

**위치**: `libra/model/multimodal_projector/builder.py:106-122`

**목적**: Current와 Prior 이미지의 temporal relationship 학습

**구조** (PyTorch 코드):
```python
def TFM(self, cur_features, prev_features):
    # Step 1: Cosine Similarity 계산
    cos = calculate_cosine_similarity(cur_features, prev_features)
    # 유사도를 [0,1] → 8승으로 날카롭게
    # cos = ((cos + 1) / 2) ** 8

    # Step 2: Prior에 similarity-based weight 추가
    prev_weight = cos * self.LFE_prior_bias  # learnable bias
    prev_features = prev_features + prev_weight

    # Step 3: Self-Attention (각자 독립적으로)
    cur_features = norm1(cur_features + cur_self_attention(cur_features))
    prev_features = norm2(prev_features + prior_self_attention(prev_features))

    # Step 4: Cross-Attention (Current가 Prior를 참조)
    combined = norm3(cur_features + cross_attention(
        query=cur_features,
        key=prev_features,
        value=prev_features
    ))

    # Step 5: Residual + MLP projection
    output = norm4(cur_features + mlp_attn(combined))
    output = mlp_final(output)  # 768 → 4096 (LLM dimension)

    return output
```

**핵심 메커니즘**:

1. **Cosine Similarity Weighting** (line 111-113)
   - 두 이미지가 유사하면 prior의 영향력 증가
   - `pow(8)` 적용으로 차이를 극대화

2. **Dual Self-Attention** (line 115-116)
   - Current: 현재 이미지의 내부 관계 학습
   - Prior: 과거 이미지의 내부 관계 학습

3. **Cross-Attention** (line 117)
   - Current가 Query, Prior가 Key/Value
   - "현재 병변이 과거와 어떻게 다른가?"를 학습

4. **Residual Connection** (line 119)
   - Original current features 보존
   - Temporal 정보만 추가

---

## 차원 변환 추적

```
Input Images
├─ Current: [batch, 3, 518, 518]
└─ Prior:   [batch, 3, 518, 518]
    ↓
RAD-DINO Encoder (dino_encoder.py:51-108)
├─ 12 hidden layers 추출
└─ Output: [2, batch, 12, 1369, 768]
           ↑  ↑      ↑    ↑     ↑
           │  │      │    │     └─ Hidden dimension
           │  │      │    └─ Patches (37×37)
           │  │      └─ Layers
           │  └─ Batch size
           └─ (Current, Prior)
    ↓
TAC.LFE (builder.py:32-39)
└─ Output: [2, batch, 1, 1369, 768]
           (12 layers → 1 layer)
    ↓
TAC.TFM (builder.py:106-122)
├─ Self-Attention: [batch, 1369, 768]
├─ Cross-Attention: [batch, 1369, 768]
└─ MLP Projection: [batch, 1369, 4096]
                                  ↑
                                  └─ LLM input size
    ↓
Meditron-7B
└─ Output: Radiology Report Text
```

---

## 핵심 설계 결정

### 1. 왜 12 layers를 모두 사용?

**코드**: `dino_encoder.py:56-64`

```python
if self.select_layer == "all":
    all_layers_features = [hidden_state[:, 1:] for hidden_state in hidden_states[1:]]
    return torch.stack(all_layers_features)  # [12, batch, 1369, 768]
```

**이유**:
- 각 layer가 다른 수준의 visual feature 포착
- Low layer: 텍스처, 경계
- High layer: 의미론적 개념
- LFE가 optimal combination 학습

### 2. 왜 Cosine Similarity를 8승?

**코드**: `builder.py:87`

```python
cosine_similarities_normalized = ((cosine_similarities + 1) / 2).pow(8)
```

**이유**:
- 유사도 차이를 극대화
- 예: 0.8 → 0.1678, 0.9 → 0.4305, 1.0 → 1.0
- 명확한 변화만 강조 (노이즈 제거)

### 3. 왜 Cross-Attention만 단방향?

**코드**: `builder.py:101-103`

```python
def cros_att_block(self, x, y):
    # x=current (Query), y=prior (Key/Value)
    return self.cros_attention(x, y, y)[0]
```

**이유**:
- Current가 Prior를 참조 (과거를 보고 변화 판단)
- Prior는 Current를 볼 필요 없음 (시간 순서 준수)

---

## 핵심 코드 흐름

### Forward Pass

**위치**: `libra_arch.py:124-128`

```python
def encode_images(self, images):
    # images: [2, batch, 3, H, W] - 2 = (current, prior)

    # Step 1: Vision Tower
    image_features_temp = self.vision_tower(images)
    # Output: [[batch, 12, 1369, 768], [batch, 12, 1369, 768]]

    # Step 2: TAC (LFE + TFM)
    image_features = self.mm_projector(image_features_temp)
    # Output: [batch, 1369, 4096]

    return image_features
```

### TAC Forward

**위치**: `builder.py:124-132`

```python
def forward(self, image_features):
    cur_features, prev_features = image_features

    # LFE: 12 layers → 1 layer
    cur_features = self.LFE(cur_features).squeeze(1)
    prev_features = self.LFE(prev_features).squeeze(1)

    # TFM: Temporal reasoning
    output = self.TFM(cur_features, prev_features)

    return output
```

---

## 다른 Projector와 비교

| Projector | 구조 | Temporal | 파라미터 |
|-----------|------|----------|---------|
| **Linear** | 1-layer MLP | ❌ | ~3M |
| **MLP-2x** | 2-layer MLP | ❌ | ~8M |
| **TAC** | LFE + TFM | ✅ | ~50M |

**TAC의 장점**:
- Explicit temporal modeling
- Layer-wise feature selection
- Attention-based feature fusion

---

## 실전 코드 사용 예시

### TAC 활성화
```bash
# pretrain.sh or finetune_lora.sh
--mm_projector_type TAC \
--mm_vision_select_layer all \       # 12 layers 사용
--mm_vision_select_feature patch \   # CLS token 제외
```

### MLP로 변경 (비교 실험)
```bash
--mm_projector_type mlp2x_gelu \
--mm_vision_select_layer -2 \        # 마지막 layer만
--mm_vision_select_feature patch \
```

---

## 핵심 Takeaways

1. **TAC = LFE + TFM**
   - LFE: 12 layers → best features
   - TFM: Temporal reasoning via attention

2. **코드 구조가 논문과 1:1 매칭**
   - `builder.py:32-39` = LFE
   - `builder.py:106-122` = TFM

3. **모듈성이 우수**
   - TAC 교체 가능 (1줄 수정)
   - Encoder 교체 가능 (1줄 수정)

4. **2-Stage 학습이 핵심**
   - Stage 1: TAC가 temporal reasoning 학습
   - Stage 2: LLM이 report generation 학습

5. **실용성**
   - Public pretrained weights 제공
   - Single image도 지원 (dummy prior)
   - 다양한 LLM 백엔드 지원

---

## 참고 자료

- **논문**: [Libra: Leveraging Temporal Images for Biomedical Radiology Analysis](https://arxiv.org/abs/2411.19378)
- **코드**: [GitHub Repository](https://github.com/X-iZhang/Libra)
- **모델**: [HuggingFace](https://huggingface.co/X-iZhang/libra-v1.0-7b)
- **학회**: ACL 2025
