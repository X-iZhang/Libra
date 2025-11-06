# H100 GPU 최적화 학습 가이드

## 목차
- [GPU 스펙 비교](#gpu-스펙-비교)
- [학습 시간 계산](#학습-시간-계산)
- [최적 GPU 수 추천](#최적-gpu-수-추천)
- [Multi-GPU 설정 가이드](#multi-gpu-설정-가이드)
- [비용 비교](#비용-비교)

---

## GPU 스펙 비교

| 항목 | A6000 48GB | H100 80GB | 배율 |
|------|-----------|-----------|------|
| **Memory** | 48 GB GDDR6 | 80 GB HBM3 | 1.67× |
| **Memory BW** | 768 GB/s | 2,000 GB/s | 2.6× |
| **FP16 Tensor** | 309.7 TFLOPS | 1,513 TFLOPS | **4.9×** |
| **BF16 Tensor** | 309.7 TFLOPS | 1,979 TFLOPS | **6.4×** |
| **TF32 Tensor** | 155 TFLOPS | 989 TFLOPS | **6.4×** |
| **실제 학습 속도** | 1× (기준) | **~6-7×** | - |

**주요 벤치마크**:
- H100 vs A100: 2.2-3.3× 빠름
- A100 vs A6000: ~2× 빠름
- **H100 vs A6000**: **~6-7×** (실제 학습 기준)

---

## 학습 시간 계산

### 현재 상황 (A6000 1개)
```
Stage 1 (Pretrain):  385 hours (16일)
Stage 2 (LoRA):      213 hours (9일)
─────────────────────────────────
Total:               598 hours (25일)
```

### H100 1개로 학습
```
Stage 1: 385 / 6 = 64 hours (2.7일) ⚡
Stage 2: 213 / 6 = 36 hours (1.5일) ⚡
─────────────────────────────────
Total:   100 hours (4.2일)
```

**결론**: **H100 1개만으로도 충분합니다!** 🎉

---

## 최적 GPU 수 추천

### 시나리오 1: H100 1개 ⭐ 추천

```
학습 시간: 4.2일
비용: ~$3/hour × 100h = $300
메모리: 80GB (충분)
구성: 간단
```

**장점**:
- ✅ 설정 간단 (DeepSpeed Zero-2면 충분)
- ✅ 비용 효율적
- ✅ 디버깅 쉬움

**단점**:
- ⚠️ 4일 대기

---

### 시나리오 2: H100 2개

```
학습 시간: 2.1일
비용: ~$3/hour × 2 GPUs × 50h = $300
Scaling efficiency: 95%
```

**설정**:
```bash
# pretrain.sh
TRAIN_BSZ=8           # Per-device 절반
GRAD_ACC_STEPS=2      # Accumulation 2배
# Global Batch = 8 × 2 × 2 = 32 (2배)
```

**장점**:
- ✅ 2배 빠름
- ✅ 동일한 총 비용

**단점**:
- ⚠️ 설정 복잡도 증가
- ⚠️ Communication overhead

---

### 시나리오 3: H100 4개

```
학습 시간: 1.1일
비용: ~$3/hour × 4 GPUs × 26h = $312
Scaling efficiency: 90%
```

**설정**:
```bash
# pretrain.sh
TRAIN_BSZ=4
GRAD_ACC_STEPS=4
# Global Batch = 4 × 4 × 4 = 64 (4배)
```

**장점**:
- ✅ 하루만에 완료!
- ✅ 큰 batch size (더 안정적)

**단점**:
- ⚠️ Communication overhead 증가 (10% 손실)
- ⚠️ NVLink 필요

---

### 시나리오 4: H100 8개 (비추천)

```
학습 시간: 0.6일 (15시간)
비용: ~$3/hour × 8 GPUs × 15h = $360
Scaling efficiency: 80%
```

**비추천 이유**:
- ❌ 비용 대비 효율 낮음
- ❌ Communication overhead 심각
- ❌ Batch size 너무 큼

---

## 최종 추천

### 💡 Best Choice: H100 1개

**이유**:
1. ✅ 4.2일이면 충분히 빠름 (A6000 대비 6배)
2. ✅ 가장 비용 효율적 ($300)
3. ✅ 설정 간단 (스크립트 그대로 사용)
4. ✅ 80GB 메모리로 여유있음
5. ✅ 디버깅 쉬움

**스크립트 수정 불필요**:
```bash
# 그대로 실행
bash scripts/pretrain.sh      # 64 hours
bash scripts/finetune_lora.sh # 36 hours
```

---

### ⚡ 빠른 완료 필요 시: H100 2개

**상황**:
- 논문 데드라인
- 빠른 iteration 필요
- 비용 여유 있음

**설정**:
```bash
# pretrain.sh, finetune_lora.sh
TRAIN_BSZ=8           # 16 → 8
GRAD_ACC_STEPS=2      # 1 → 2

# DeepSpeed 설정
# zero2.json은 그대로 사용 가능
```

**예상 시간**: 2.1일 (50시간)

---

## Multi-GPU 설정 가이드

### H100 2개 설정

```bash
# 1. pretrain.sh 수정
TRAIN_BSZ=8           # 16 → 8
GRAD_ACC_STEPS=2      # 1 → 2

# 2. 실행
CUDA_VISIBLE_DEVICES=0,1 bash scripts/pretrain.sh

# 3. 자동으로 2 GPU 감지됨
# Global Batch Size = 8 × 2 × 2 = 32
```

### H100 4개 설정

```bash
# 1. pretrain.sh 수정
TRAIN_BSZ=4           # 16 → 4
GRAD_ACC_STEPS=4      # 1 → 4

# 2. DeepSpeed config 확인
# scripts/zero3.json 사용 (메모리 효율)

# 3. 실행
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/pretrain.sh

# Global Batch Size = 4 × 4 × 4 = 64
```

---

## 비용 비교

| 설정 | 시간 | GPU-hours | 비용 (@$3/h) | 효율 |
|------|------|-----------|-------------|------|
| **A6000 1개** | 598h | 598 | $1,794 | 기준 |
| **H100 1개** | 100h | 100 | $300 ✅ | 17% |
| **H100 2개** | 50h | 100 | $300 ✅ | 17% |
| **H100 4개** | 26h | 104 | $312 | 17% |
| **H100 8개** | 15h | 120 | $360 ⚠️ | 20% |

**결론**: H100 1-2개가 최적의 비용 효율!

---

## 상세 시간표

### H100 1개 스케줄

| Day | Task | Hours | Progress |
|-----|------|-------|----------|
| **Day 1** | Stage 1 시작 | 0-24h | TAC 학습 중 |
| **Day 2** | Stage 1 계속 | 24-48h | TAC 학습 중 |
| **Day 3** | Stage 1 완료 → Stage 2 시작 | 48-64h → 64-72h | LoRA 시작 |
| **Day 4** | Stage 2 계속 | 72-96h | LoRA 학습 중 |
| **Day 5** | Stage 2 완료 ✅ | 96-100h | 완료! |

---

### H100 2개 스케줄

| Day | Task | Hours | Progress |
|-----|------|-------|----------|
| **Day 1** | Stage 1 | 0-24h | TAC 학습 중 |
| **Day 2** | Stage 1 완료 → Stage 2 | 24-32h → 32-48h | LoRA 시작 |
| **Day 3** | Stage 2 완료 ✅ | 48-50h | 완료! |

---

## 실전 체크리스트

### H100 1개 (추천) ✅
- [ ] H100 80GB 인스턴스 준비
- [ ] CUDA 12.0+ 설치
- [ ] 스크립트 경로만 수정
- [ ] `bash scripts/pretrain.sh` 실행
- [ ] 64시간 후 Stage 1 완료 확인
- [ ] `bash scripts/finetune_lora.sh` 실행
- [ ] 36시간 후 완료 🎉

### H100 2개 (빠른 완료) ⚡
- [ ] H100 80GB × 2 인스턴스 준비
- [ ] NVLink 연결 확인 (권장)
- [ ] TRAIN_BSZ=8, GRAD_ACC_STEPS=2 수정
- [ ] `CUDA_VISIBLE_DEVICES=0,1` 설정
- [ ] 실행 및 2일 대기

---

## 최종 답변

### Q: H100 몇 대로 학습하면 맞을까?

**A: H100 1개면 충분합니다!** 🎯

**이유**:
1. ✅ **시간**: 4.2일 (A6000 대비 6배 빠름)
2. ✅ **비용**: $300 (가장 효율적)
3. ✅ **간편**: 설정 변경 불필요
4. ✅ **메모리**: 80GB로 여유있음

**추가 GPU는 언제?**
- 🔥 **급할 때만**: 2개 사용 (2일 완료)
- 🚫 **4개 이상**: 비추천 (비용 낭비)

---

## 참고 자료

- **NVIDIA H100**: [Datasheet](https://www.nvidia.com/en-us/data-center/h100/)
- **NVIDIA A6000**: [Specs](https://www.nvidia.com/en-us/design-visualization/rtx-a6000/)
- **DeepSpeed**: [Documentation](https://www.deepspeed.ai/)
- **Performance Benchmarks**: Multiple sources (2024)
