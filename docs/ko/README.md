# Libra 완전 분석 & 실전 가이드 (한국어)

> **Libra**: Leveraging Temporal Images for Biomedical Radiology Analysis (ACL 2025)
>
> 본 문서는 Libra 프로젝트의 심층 분석, 학습 가이드, 그리고 최적화 방법을 다룹니다.

---

## 📚 문서 목차

### 1. [아키텍처 & 코드 구조 분석](./01_architecture_analysis.md)
**내용**:
- 논문 핵심 개념 (ACL 2025)
- 전체 아키텍처 파이프라인
- TAC (Temporal Alignment Connector) 상세 분석
  - LFE (Layerwise Feature Extractor)
  - TFM (Temporal Feature Matching)
- 코드 구조 (2,220 lines)
- 차원 변환 추적
- 핵심 설계 결정

**주요 내용**:
- TAC가 12개 vision layer를 1개로 압축
- Cosine Similarity를 8승하는 이유
- Cross-Attention이 단방향인 이유
- 코드와 논문의 1:1 매칭

---

### 2. [MIMIC-CXR Temporal Pairing 방법론](./02_temporal_pairing.md)
**내용**:
- Libra의 Prior Image Retrieval 알고리즘
- Same-day Studies 특수 처리
- 다른 연구들과 비교 (MLRG, TiBiX, CoCa-CXR)
- MIMIC-CXR 통계
- 실전 데이터 처리 코드

**주요 내용**:
- `StudyDate` + `StudyTime` 기반 pairing
- 같은 날 여러 촬영 시 동일한 prior 사용
- 67% 환자가 2회 이상 촬영
- 100% 재현 가능한 Python 코드 제공

---

### 3. [학습 가이드 (Stage 1 & Stage 2)](./03_training_guide.md)
**내용**:
- 2-Stage 학습 전략 완전 분석
- Stage 1: Visual Feature Alignment (TAC 학습)
- Stage 2: Downstream Task Fine-tuning (LoRA)
- 학습 메커니즘 상세
- 완전한 학습 체크리스트

**주요 내용**:
- Stage 1: 385시간 (16일), TAC만 학습
- Stage 2: 213시간 (9일), LLM LoRA 학습
- 왜 2-Stage가 필요한가?
- 공개 weights로 각 stage 건너뛰기 가능

---

### 4. [완전 설정 가이드 & LoRA 구현](./04_complete_settings_lora.md)
**내용**:
- Vision Encoder별 Input Size 비교
- 사용 가능한 Vision Encoders (RAD-DINO, CLIP, BiomedCLIP 등)
- 사용 가능한 LLM 모델 (Meditron, LLaMA, Qwen, Mistral 등)
- 전체 하이퍼파라미터 설정
- LoRA 구현 상세
  - Target Modules 자동 탐지
  - 학습 파라미터 계산 (224M)
  - Scaling Factor (α/r = 2.0)

**주요 내용**:
- RAD-DINO: 518×518, 768 dim, 12 layers
- CLIP-Large: 336×336, 1024 dim, 24 layers
- LoRA rank=128, alpha=256 (aggressive 설정)
- 7개 modules per layer (Attention + MLP)

---

### 5. [평가 지표 완전 가이드](./05_evaluation_metrics.md)
**내용**:
- BLEU (1, 2, 3, 4): n-gram precision
- METEOR: 동의어, 어근 고려
- ROUGE-L: 최장 공통 부분 수열
- RaTEScore: Radiology 전용 (EMNLP 2024) ⭐
- RG_ER: RadGraph Entity Recall
- 지표 간 관계 & 종합 분석

**주요 내용**:
- RaTEScore가 가장 중요 (임상 평가와 highest correlation)
- BLEU는 언어적 유사성만 측정
- RG_ER로 완전성 평가
- Libra-v1.0-3b 점수 상세 해석

---

### 6. [H100 GPU 최적화 가이드](./06_h100_optimization.md)
**내용**:
- GPU 스펙 비교 (A6000 vs H100)
- 학습 시간 계산
- 최적 GPU 수 추천
- Multi-GPU 설정 가이드
- 비용 비교

**주요 내용**:
- **H100 1개 추천**: 4.2일, $300, 가장 효율적
- H100 vs A6000: ~6-7× 빠름
- H100 2개: 2일 완료 (급할 때)
- H100 4개 이상: 비추천 (비용 낭비)

---

## 🎯 빠른 시작

### 환경 설정
```bash
git clone https://github.com/X-iZhang/Libra.git
cd Libra
conda create -n libra python=3.10 -y
conda activate libra
pip install -e ".[train,eval]"
pip install flash-attn --no-build-isolation
```

### 데이터 다운로드
```bash
# MIMIC-CXR (PhysioNet 계정 필요)
# https://physionet.org/content/mimic-cxr-jpg/2.0.0/

# Libra annotations (Google Drive)
mkdir -p ./data
cd ./data
wget <Google Drive links from README>
```

### 학습
```bash
# Stage 1: TAC 학습 (16일 @ A6000)
bash scripts/pretrain.sh

# Stage 2: LoRA 학습 (9일 @ A6000)
bash scripts/finetune_lora.sh
```

---

## 📊 주요 성능 지표

### Libra-v1.0-7b (MIMIC-CXR Findings)

| 지표 | 점수 | 의미 |
|------|------|------|
| **BLEU-1** | 51.3 | 단어 51.3% 일치 |
| **BLEU-4** | 24.5 | 4-gram 24.5% 일치 |
| **METEOR** | 48.9 | 동의어 고려 48.9% |
| **ROUGE-L** | 36.7 | 문장 구조 36.7% |
| **RaTEScore** | 61.5 | 임상 정확도 61.5% ⭐ |
| **RG_ER** | 37.6 | Entity 재현율 37.6% |

**비교**: Med-CXRGen-F 대비 BLEU4 +138%, RG_ER +58%

---

## 🏆 핵심 특징

### 1. TAC (Temporal Alignment Connector)
- **LFE**: 12 layers → 1 optimal layer
- **TFM**: Temporal reasoning via attention
- **Cosine Similarity Weighting**: 8승 적용

### 2. 2-Stage 학습
- **Stage 1**: TAC가 vision-language alignment 학습
- **Stage 2**: LLM이 report generation 학습
- **독립적이지만 순차적**

### 3. LoRA Aggressive 설정
- **Rank**: 128 (high)
- **Alpha**: 256 (scaling=2.0)
- **Target**: 7 modules per layer
- **파라미터**: 224M (전체의 3.2%)

---

## 💡 주요 발견

### Temporal Pairing
- **Same-day 특수 처리**: 임상적으로 타당
- **67% 환자**가 2회 이상 촬영
- **100% 재현 가능**한 알고리즘

### 평가 지표
- **RaTEScore 최우선**: 임상의 평가와 highest correlation
- **전통적 NLP 지표 한계**: 동의어, 부정 표현 무시
- **RG_ER로 완전성 평가**: 중요한 findings 누락 탐지

### GPU 최적화
- **H100 1개 충분**: 4.2일, $300
- **6-7× speedup**: A6000 대비
- **Multi-GPU**: 급할 때만 2개

---

## 🔗 원본 리소스

- **논문**: [arXiv:2411.19378](https://arxiv.org/abs/2411.19378) (ACL 2025)
- **코드**: [GitHub](https://github.com/X-iZhang/Libra)
- **모델**: [HuggingFace](https://huggingface.co/X-iZhang/libra-v1.0-7b)
- **MIMIC-CXR**: [PhysioNet](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)
- **ReXrank**: [Leaderboard](https://rexrank.ai)

---

## 📝 문서 작성 정보

- **작성일**: 2025년
- **대상**: Libra 사용자, 연구자, 개발자
- **언어**: 한국어
- **기반**: Libra v1.0 (ACL 2025)

---

## 🤝 기여

본 문서는 Libra 프로젝트 이해를 돕기 위한 비공식 가이드입니다.

**문의사항**:
- GitHub Issues: [Libra Repository](https://github.com/X-iZhang/Libra/issues)
- 논문 저자: Xi Zhang 외

---

## 📜 라이선스

본 문서는 Libra 프로젝트의 라이선스를 따릅니다.
- **코드**: Apache License 2.0
- **모델**: HuggingFace Model License

---

**Happy Training! 🚀**
