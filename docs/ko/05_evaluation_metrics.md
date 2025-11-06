# Radiology Report Generation 평가 지표 완전 가이드

## 목차
- [BLEU](#bleu-bilingual-evaluation-understudy)
- [METEOR](#meteor-metric-for-evaluation-of-translation-with-explicit-ordering)
- [ROUGE-L](#rouge-l-recall-oriented-understudy-for-gisting-evaluation)
- [RaTEScore](#ratescore-radiology-text-evaluation-score)
- [RG_ER (RadGraph Entity Recall)](#rg_er-radgraph-entity-recall)
- [지표 간 관계 & 종합 분석](#지표-간-관계--종합-분석)

---

## BLEU (Bilingual Evaluation Understudy)

### 정의
n-gram 중복을 기반으로 생성된 텍스트와 참조 텍스트의 유사도를 측정

### 계산 방법

**코드**: `radiology_report.py:113-145`

```python
# BLEU-1: Unigram (단어 단위) 매칭
bleu1 = bleu_metric.compute(
    predictions=predictions_list,
    references=references_list,
    max_order=1  # 1-gram
)['bleu']

# BLEU-4: 1~4-gram 평균
bleu4 = bleu_metric.compute(
    max_order=4
)['bleu']
```

**수식**:
```
BLEU-N = BP × exp(∑(wₙ × log(pₙ)))

where:
  pₙ = n-gram precision
  BP = Brevity Penalty (길이 페널티)
  wₙ = weights (균등 분배)
```

### 예시
```
Reference: "No acute cardiopulmonary abnormality"
Prediction: "No cardiopulmonary abnormality detected"

1-gram matches: "no", "cardiopulmonary", "abnormality" (3/4)
→ BLEU-1 = 75.0

4-gram matches: 0개 (완전히 다른 순서)
→ BLEU-4 = 낮음
```

### 범위 & 해석
- **범위**: 0~100 (높을수록 좋음)
- **BLEU-1 (50.5)**: 단어 50.5%가 참조와 일치
- **BLEU-4 (23.3)**: 4-gram 패턴 23.3% 일치

### Radiology에서의 의미
- ✅ **장점**: 빠르고 간단, 대규모 평가 가능
- ⚠️ **단점**:
  - 동의어 무시 ("enlarged" vs "increased")
  - 임상적 정확성 무시
  - 어순에 민감

---

## METEOR (Metric for Evaluation of Translation with Explicit ORdering)

### 정의
동의어, 어근, 그리고 단어 정렬을 고려한 고급 텍스트 유사도 지표

### 핵심 특징
```
1. Exact Match: "cardiomegaly" = "cardiomegaly" ✅
2. Stem Match: "enlarged" ≈ "enlargement" ✅
3. Synonym Match: "increased" ≈ "enlarged" ✅ (via WordNet)
4. Paraphrase Match: 의미적 유사성
```

### 수식
```
METEOR = Fₘₑₐₙ × (1 - Penalty)

where:
  Fₘₑₐₙ = Harmonic mean of Precision & Recall
  Penalty = 청크 단편화 페널티 (순서 고려)
```

### 예시
```
Reference: "Heart is enlarged"
Prediction: "Cardiac enlargement noted"

BLEU: 낮음 (단어 안맞음)
METEOR: 높음 (동의어 인식)
  - "heart" ≈ "cardiac"
  - "enlarged" ≈ "enlargement"
```

### 범위 & 해석
- **범위**: 0~100
- **48.5**: 동의어 고려 시 48.5% 일치
- BLEU보다 높은 경우 = 의미는 맞지만 표현이 다름

### Radiology에서의 의미
- ✅ **장점**: 의학 용어 동의어 인식
- ⚠️ **단점**: WordNet 기반 (의학 특화 부족)

---

## ROUGE-L (Recall-Oriented Understudy for Gisting Evaluation)

### 정의
최장 공통 부분 수열(LCS)을 기반으로 문장 구조 유사도 측정

### 수식
```
ROUGE-L = (1 + β²) × (Rₗcs × Pₗcs) / (Rₗcs + β²×Pₗcs)

where:
  Rₗcs = LCS_length / Reference_length (Recall)
  Pₗcs = LCS_length / Prediction_length (Precision)
  β = 1.2 (Recall 중시)
```

### 예시
```
Reference: "No acute cardiopulmonary process"
Prediction: "No cardiopulmonary abnormality"

LCS: "No cardiopulmonary" (길이=2)
```

### 범위 & 해석
- **범위**: 0~100
- **35.2**: 문장 구조 35.2% 일치

### Radiology에서의 의미
- ✅ **장점**: 문장 흐름 평가
- ⚠️ **단점**: 부정 표현 민감하지 않음

---

## RaTEScore (Radiology Text Evaluation Score)

### 정의
의료 Entity 중심으로 설계된 Radiology 전용 평가 지표 (EMNLP 2024)

### 아키텍처
```
Input Reports (Pred vs Ref)
    ↓
┌──────────────────────────────────────┐
│ 1. Medical Entity Recognition (NER) │
│    - Anatomical regions             │
│    - Findings/Observations          │
│    - Diseases                       │
│    - Severity                       │
└──────────────────────────────────────┘
    ↓
┌──────────────────────────────────────┐
│ 2. Synonym Disambiguation Encoding  │
│    - "cardiomegaly" = "enlarged heart"│
│    - Medical synonym database       │
└──────────────────────────────────────┘
    ↓
┌──────────────────────────────────────┐
│ 3. Negation-Aware Scoring           │
│    - "no evidence" vs "evidence" ✅  │
│    - "stable" vs "improved" ✅       │
└──────────────────────────────────────┘
    ↓
RaTEScore (0~100)
```

### 예시
```
Reference: "No acute cardiopulmonary process"
Prediction: "Acute pneumonia in right lower lobe"

RaTEScore: 매우 낮음 ⚠️
  - Entity mismatch: "no process" vs "pneumonia"
  - Negation error: "no" vs "present"
  - Clinical significance: 오진!
```

### 범위 & 해석
- **범위**: 0~100
- **61.1**: 임상적 Entity 61.1% 일치
- **가장 중요한 지표** (임상의 평가와 highest correlation)

### Radiology에서의 의미
- ✅ **장점**:
  - 임상적으로 의미있는 평가
  - 부정 표현 정확히 처리
  - 의학 동의어 인식
- ⚠️ **단점**: 계산 복잡, 느림

**논문**: [EMNLP 2024](https://arxiv.org/abs/2406.16845)

---

## RG_ER (RadGraph Entity Recall)

### 정의
RadGraph 기반으로 임상 Entity와 Relation의 재현율(Recall) 측정

### RadGraph 구조
```
Entities (4 types):
  - Anatomical location (ANAT)
  - Observation (OBS)
  - Observation modifier (OBS-MOD)
  - Observation uncertainty (OBS-U)

Relations (3 types):
  - located_at (OBS → ANAT)
  - modify (OBS-MOD → OBS)
  - suggestive_of (OBS → OBS)
```

### 예시
```
Reference: "Small left pleural effusion"

RadGraph:
  Entities:
    - "effusion" (OBS) ✅
    - "small" (OBS-MOD) ✅
    - "left" (ANAT) ✅
    - "pleural" (ANAT) ✅

Prediction: "Left pleural effusion"

RadGraph:
  Entities:
    - "effusion" (OBS) ✅
    - "left" (ANAT) ✅
    - "pleural" (ANAT) ✅

Missing:
  - "small" (OBS-MOD) ❌

Entity Recall: 3/4 = 75.0%
```

### 범위 & 해석
- **범위**: 0~100
- **37.5**: 참조 Entity의 37.5%가 생성에 포함
- **낮은 이유**: 상세한 세부사항 누락

### Radiology에서의 의미
- ✅ **장점**:
  - 완전성(Completeness) 평가
  - 중요한 findings 누락 탐지
- ⚠️ **단점**: Precision 측정 안함

**관련**: F1-RadGraph = 2 × (Precision × Recall) / (Precision + Recall)

---

## 지표 간 관계 & 종합 분석

### Libra-v1.0-3b 점수 해석

```
BLEU1: 50.5  ✅ 단어 절반 일치 (준수)
BLEU4: 23.3  ⚠️ 구문 구조는 다름
METEOR: 48.5 ✅ 동의어 고려 시 절반 일치
ROUGE-L: 35.2 ⚠️ 문장 흐름 차이
RaTEScore: 61.1 ✅✅ 임상적으로 61% 정확 (매우 중요!)
RG_ER: 37.5  ⚠️ 세부사항 누락
```

### 종합 평가

| 측면 | 평가 |
|------|------|
| **어휘 다양성** | 좋음 (BLEU1 50.5) |
| **구문 구조** | 보통 (BLEU4 23.3) |
| **의미 유사성** | 좋음 (METEOR 48.5) |
| **임상 정확성** | 좋음 (RaTEScore 61.1) ⭐ |
| **완전성** | 개선 필요 (RG_ER 37.5) |

### 가장 중요한 지표 우선순위

```
1. RaTEScore (61.1) ⭐⭐⭐
   → 임상의 평가와 가장 높은 상관관계
   → 임상적 의사결정에 직접 영향

2. RG_ER (37.5) ⭐⭐
   → 완전성 평가
   → 의료 과실 방지

3. METEOR (48.5) ⭐
   → 의학 용어 동의어 고려
   → BLEU보다 신뢰성 높음

4. BLEU4 (23.3)
   → 빠른 벤치마킹용
   → 학계 비교 표준

5. ROUGE-L (35.2)
   → 보조 지표
```

### 지표 조합 해석

```
Case 1: RaTEScore 높음 + RG_ER 낮음
→ 언급한 내용은 정확하지만 불완전 (누락 있음)

Case 2: RaTEScore 낮음 + BLEU 높음
→ 표현은 비슷하지만 임상적으로 오류 (심각!)

Case 3: 모든 지표 균형
→ 이상적인 모델
```

---

## 핵심 메시지

**Libra-v1.0-3b 평가**:
- ✅ **임상적 정확성 우수** (RaTEScore 61.1)
- ✅ **어휘 선택 적절** (METEOR 48.5)
- ⚠️ **세부사항 보완 필요** (RG_ER 37.5)
- ⚠️ **문장 구조 개선 가능** (BLEU4 23.3)

전통적 NLP 지표(BLEU, ROUGE)는 **언어적 유사성**만 측정하지만,
**RaTEScore와 RG_ER은 임상적 정확성과 완전성**을 측정하므로
의료 AI 평가에서 훨씬 중요합니다!

---

## 참고 자료

- **BLEU**: [Paper](https://aclanthology.org/P02-1040/)
- **METEOR**: [Paper](https://aclanthology.org/W05-0909/)
- **ROUGE**: [Paper](https://aclanthology.org/W04-1013/)
- **RaTEScore**: [EMNLP 2024](https://arxiv.org/abs/2406.16845)
- **RadGraph**: [Paper](https://arxiv.org/abs/2106.14463)
