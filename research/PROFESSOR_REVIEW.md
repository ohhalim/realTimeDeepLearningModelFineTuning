# 논문 심사 평가서 (Peer Review)

**Conference**: ICML 2026 / NeurIPS 2025
**Paper ID**: Anonymous Submission
**Title**: "Hierarchical StyleLoRA-Transformer: Efficient and Interpretable Style Transfer for Expressive Jazz Piano Generation"
**Reviewer**: Senior Professor in Music AI & Deep Learning (20+ years experience)

---

## 평점 요약

| Category | Score | Weight |
|----------|-------|--------|
| Novelty & Originality | 6/10 | 30% |
| Technical Quality | 4/10 | 25% |
| Experimental Validation | 2/10 | 25% |
| Clarity & Presentation | 7/10 | 10% |
| Reproducibility | 5/10 | 10% |
| **Overall** | **4.5/10** | **REJECT** |

**Decision**: **Strong Reject** → **Conditional Accept (Major Revision Required)**

---

## 주요 문제점 (Critical Issues)

### 🔴 CRITICAL: 실험 결과 없음

**문제점:**
```
Table 1에 제시된 모든 수치가 실제 실험 결과가 아님:
- Perplexity: 23.0 (claimed)
- Harmonic Accuracy: 0.795 (claimed)
- Style Classification: 0.924 (claimed)

코드 확인 결과: train_stage3_lora_finetuning.py는
DUMMY DATA만 사용 (DummyDataset class)
```

**심각도**: **논문 rejection 사유**

**요구사항**:
- 최소 50개 Brad Mehldau MIDI 파일로 실제 학습 필요
- 3회 이상 반복 실험 + 표준편차 제시
- Held-out test set에서 평가

---

### 🔴 CRITICAL: Layer Assignment에 대한 근거 부족

**문제점:**
```python
# 현재 구현 (arbitrary)
Harmony LoRA → Layers 0-3  (왜?)
Voicing LoRA → Layers 4-7  (근거는?)
Rhythm LoRA  → Layers 8-11 (검증됨?)
```

**논문 주장 (Section 3.2):**
> "Our layer assignment is motivated by analysis of layer-wise
> representations in transformer language models [Tenney et al., 2019]"

**문제**:
1. Tenney et al.은 **NLP** 연구 (BERT), 음악이 아님
2. 음악에서 layer-wise specialization 검증 안 됨
3. Ablation study로 다른 assignment 비교 안 함

**요구사항**:
- 각 layer의 representation을 분석 (attention map visualization)
- 다른 layer assignment와 비교 (예: 0-5, 6-11 / 랜덤 assignment)
- 음악 도메인에서의 layer specialization 증명

---

### 🔴 CRITICAL: Multi-Task Loss 가중치가 Ad-Hoc

**현재 코드:**
```python
# train_stage3_lora_finetuning.py
lambda_harmony: float = 0.1  # 왜 0.1?
lambda_voicing: float = 0.1  # 왜 동일?
lambda_rhythm: float = 0.1   # 근거는?
```

**문제점:**
1. 가중치 선택에 대한 ablation study 없음
2. 각 loss의 scale이 다른데 동일 가중치 사용
3. 논문에 hyperparameter tuning 과정 없음

**요구사항**:
- Grid search: λ ∈ {0.01, 0.05, 0.1, 0.5, 1.0}
- Loss balancing 기법 적용 (예: uncertainty weighting)
- 각 loss 항의 magnitude 분석

---

### 🟡 MAJOR: Harmony/Voicing/Rhythm Loss 구현이 Naive

**Harmony Loss 문제 (objective_metrics.py:77-90):**
```python
def extract_chord(self, tokens: torch.Tensor) -> str:
    # Jaccard similarity로 chord 인식
    pitch_classes = set((notes % 12).tolist())

    # 문제: 매우 단순한 chord recognition
    # Brad Mehldau의 복잡한 voicing (예: E♭7♯9♭13) 인식 불가
```

**개선 방향:**
- Pretrained chord recognition model 사용 (BTC dataset)
- Root, chord type, extensions 분리 인식
- Voice leading analysis 추가

**Rhythm Loss 문제 (train_stage3_lora_finetuning.py:140-156):**
```python
def rhythm_loss(self, logits, targets):
    # 단순히 time-shift token에 cross-entropy
    # 문제: Syncopation, swing feel 고려 안 함
```

**개선 방향:**
- Beat-level vs off-beat 구분
- Swing ratio 측정
- Microtiming deviation 분석

---

### 🟡 MAJOR: Style Encoder와 Main Model 연결 불명확

**문제점:**
```python
# style_encoder.py에 StyleContrastiveModel 구현됨
# 하지만 train_stage3_lora_finetuning.py에서 사용 안 함!

# train_stage3_lora_finetuning.py:32
from ..models.style_encoder import StyleContrastiveModel
# ← import만 하고 실제로 사용 안 함
```

**논문 주장 (Section 3.5):**
> "Stage 2: Learn style embeddings via contrastive learning"

**실제 코드**: Stage 2 training script 없음!

**요구사항**:
- `train_stage2_contrastive.py` 구현 필요
- Style embedding이 Stage 3에서 어떻게 사용되는지 명확히
- Style conditioning mechanism 추가

---

### 🟡 MAJOR: Baseline 구현 없음

**논문 Table 1:**
```
Method                    | PPL  | Harm. Acc | Style Acc
--------------------------|------|-----------|----------
Music Transformer         | 42.3 | 0.614     | 0.482
Full Fine-tuning          | 28.1 | 0.728     | 0.864
Single LoRA               | 25.7 | 0.751     | 0.891
AdaLoRA                   | 24.9 | 0.763     | 0.902
Ours (Hierarchical)       | 23.0 | 0.795     | 0.924
```

**코드 확인**: Baseline 구현 없음!

**요구사항**:
- `baselines/music_transformer.py`
- `baselines/full_finetuning.py`
- `baselines/single_lora.py`
- `baselines/adalora.py`
- 동일한 데이터/하이퍼파라미터로 공정한 비교

---

### 🟡 MAJOR: Human Evaluation 없음

**논문 주장 (Section 4.5):**
> "We conduct pairwise preference tests with 30 professional jazz musicians"
> "Ours vs. Music Transformer: 87% prefer ours"

**문제**: 실제 human evaluation 수행 안 됨!

**요구사항**:
- IRB approval 받기
- 최소 20명 이상의 musicians recruit
- Randomized, blind listening test
- Statistical significance test (p < 0.05)

---

## 수학적 오류 (Mathematical Errors)

### Error 1: LoRA Formulation 부정확

**논문 (Section 3.2):**
```latex
W = W_0 + α · ΔW, where ΔW = BA
```

**문제**: Scaling이 두 번 적용됨

**올바른 formulation:**
```latex
h = W_0 x + (α/r) · B A x
```

여기서 α = lora_alpha, r = rank

**코드도 수정 필요** (hierarchical_lora.py:63):
```python
# 현재 (잘못됨)
return result * self.scaling

# 올바른 구현
# scaling = alpha / r은 이미 올바름
```

실제로 코드는 맞는데 논문 표기가 틀림.

---

### Error 2: Relative Position Embedding 설명 부정확

**논문 (Section 3.2):**
```latex
Attn(X) = Softmax((QK^T)/√d_k + R)V
```

**문제**: R의 차원이 불명확

**올바른 formulation:**
```latex
scores_{ij} = (q_i · k_j)/√d_k + (q_i · r_{i-j})/√d_k
```

여기서 r_{i-j}는 relative position embedding for distance (i-j)

---

### Error 3: Triplet Loss 표기 오류

**논문에 triplet loss 언급만 있고 수식 없음**

**추가 필요:**
```latex
L_triplet = max(0, margin + d(a,p) - d(a,n))
```

여기서:
- a = anchor (Mehldau piece)
- p = positive (another Mehldau piece)
- n = negative (Evans/Jarrett/Peterson)
- d = cosine distance = 1 - cosine_similarity

---

## 구현 품질 문제 (Implementation Issues)

### Issue 1: 메모리 비효율

**문제점** (music_transformer.py:345-360):
```python
def forward(self, input_ids, mask=None):
    # 매 forward마다 causal mask 생성
    if mask is None:
        mask = self.create_causal_mask(seq_len, device)
    # ← 캐싱 안 함, 매번 재생성
```

**개선**:
```python
def __init__(self, ...):
    self.register_buffer('causal_mask_cache', None)

def forward(self, input_ids, mask=None):
    if mask is None:
        if (self.causal_mask_cache is None or
            self.causal_mask_cache.size(0) < seq_len):
            self.causal_mask_cache = self.create_causal_mask(...)
        mask = self.causal_mask_cache[:seq_len, :seq_len]
```

---

### Issue 2: Gradient Checkpointing 없음

대형 모델 (86M params)인데 gradient checkpointing 없음
→ VRAM 사용량 높음

**추가 필요**:
```python
from torch.utils.checkpoint import checkpoint

def forward(self, x):
    for layer in self.layers:
        x = checkpoint(layer, x, mask)  # Save memory
```

---

### Issue 3: Mixed Precision Training 없음

**현재**: FP32만 사용
**문제**: 느리고 메모리 많이 씀

**추가 필요**:
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    logits = model(input_ids)
    loss = criterion(logits, labels)
```

---

## Related Work 문제

### 누락된 중요 논문들:

1. **MuseNet (OpenAI, 2019)**: Multi-instrument music generation
2. **Jukebox (OpenAI, 2020)**: Raw audio generation
3. **MusicLM (Google, 2023)**: Text-to-music with contrastive learning
4. **MusicGen (Meta, 2023)**: Current SOTA
5. **LoRA for Music**: 음악 도메인에 LoRA 적용한 선행 연구 찾아야
6. **Style Transfer in Music**: Mor et al. (2018) 외에 더 많은 연구 있음

### 추가 필요:

- **Controllable Music Generation**: Controllability 관련 연구
- **Hierarchical Music Models**: Hierarchical VAE, Hierarchical RNN
- **Jazz Piano Modeling**: 재즈 피아노 특화 연구

---

## 재현성 문제 (Reproducibility Issues)

### 문제 1: Random Seed 고정 안 됨

```python
# train_stage3_lora_finetuning.py에 seed 설정 없음!

# 추가 필요:
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
```

---

### 문제 2: 데이터 전처리 스크립트 없음

**필요**:
- MIDI → tokens 변환 스크립트
- Data augmentation (transpose, tempo)
- Train/val/test split 재현 가능하게

---

### 문제 3: Checkpoint 저장/로딩 불완전

**현재** (train_stage3_lora_finetuning.py:253-260):
```python
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, checkpoint_path)
```

**문제**: Hyperparameters, config 저장 안 함

**개선**:
```python
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'config': vars(args),  # ← 추가
    'lora_weights': {
        'harmony': model.lora_controller.harmony_weight,
        'voicing': model.lora_controller.voicing_weight,
        'rhythm': model.lora_controller.rhythm_weight,
        'dynamics': model.lora_controller.dynamics_weight,
    },
}, checkpoint_path)
```

---

## 논문 작성 품질 (Writing Quality)

### 장점:
✅ Abstract 명확함
✅ Motivation 잘 설명됨
✅ Figure 계획 좋음

### 단점:
❌ 수식 표기 일부 부정확
❌ Related work 너무 짧음 (20+ 논문 추가 필요)
❌ Experimental setup 세부사항 부족
❌ Failure cases 분석 너무 짧음

---

## 교수로서의 최종 평가

### Strengths (강점):

1. **Novel Idea**: Hierarchical LoRA decomposition은 참신함
2. **Clear Motivation**: 음악 스타일의 compositional nature 잘 포착
3. **Interpretable Control**: 각 LoRA 가중치 조절 기능은 실용적
4. **Code Quality**: 구현이 깔끔하고 모듈화 잘 됨

### Weaknesses (약점):

1. **🔴 CRITICAL: 실험 결과 없음** - 모든 수치가 claimed
2. **🔴 CRITICAL: Layer assignment 근거 부족**
3. **🔴 CRITICAL: Baseline 구현 없음**
4. **🟡 MAJOR: Multi-task loss 설계가 ad-hoc**
5. **🟡 MAJOR: Style encoder 미사용**
6. **🟡 MAJOR: Human evaluation 없음**

---

## 요구사항 (Requirements for Acceptance)

### Must Fix (Rejection 회피):

1. ✅ 실제 Brad Mehldau 데이터로 학습 (최소 50 files)
2. ✅ Baseline 4개 모두 구현 및 비교
3. ✅ Ablation study 실제 수행 (different layer assignments)
4. ✅ Multi-task loss 가중치 tuning + 분석
5. ✅ Human evaluation (최소 20명)

### Should Fix (Accept 가능하지만 권장):

6. ⭐ Style encoder Stage 2 training 구현
7. ⭐ Harmony/Rhythm loss 개선 (더 sophisticated)
8. ⭐ Related work 확장 (20+ 논문)
9. ⭐ Mixed precision training 추가
10. ⭐ Reproducibility: seed, config 저장

### Nice to Have:

11. 💡 Real-time generation demo
12. 💡 Interactive web interface
13. 💡 Audio samples in supplementary material

---

## 수정 작업 계획

다음 순서로 수정하겠습니다:

### Phase 1: Critical Fixes (1주)
- [ ] Baseline 모델 4개 구현
- [ ] Layer assignment ablation study 설계
- [ ] Multi-task loss 개선
- [ ] Reproducibility 개선 (seed, config)

### Phase 2: Experimental Validation (2주)
- [ ] 실제 MIDI 데이터 수집 및 전처리
- [ ] 모든 baseline 학습
- [ ] Ablation studies 수행
- [ ] 통계적 유의성 검증

### Phase 3: Evaluation (1주)
- [ ] Objective metrics 개선
- [ ] Human evaluation 설계 및 수행
- [ ] Qualitative analysis

### Phase 4: Paper Revision (1주)
- [ ] 수식 수정
- [ ] Related work 확장
- [ ] Experimental results 추가
- [ ] Figures 생성

---

## 결론

**현재 상태**: 아이디어는 좋지만 실험 검증 부족
**평가**: **4.5/10 (Reject → Major Revision)**
**예상 작업량**: 4-6주

**교수 추천**:
1. Resubmit을 목표로 철저히 수정
2. 실험 먼저, 논문은 나중에
3. Baseline과의 공정한 비교 필수
4. Human evaluation 꼭 수행

이 논문이 **ICML/NeurIPS에 accept되려면** 위의 critical issues를 모두 해결해야 합니다.

---

**Reviewer Confidence**: 5/5 (Expert)
**Recommendation**: **Major Revision Required**
