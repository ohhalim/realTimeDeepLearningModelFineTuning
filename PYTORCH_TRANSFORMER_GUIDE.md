## PyTorch Music Transformer 완전 가이드

Brad Mehldau 스타일 재즈 피아노 생성을 위한 **순수 PyTorch Music Transformer** 구현 가이드

---

## 목차

1. [개요](#1-개요)
2. [아키텍처](#2-아키텍처)
3. [환경 설정](#3-환경-설정)
4. [데이터 준비](#4-데이터-준비)
5. [모델 학습](#5-모델-학습)
6. [음악 생성](#6-음악-생성)
7. [하이퍼파라미터 튜닝](#7-하이퍼파라미터-튜닝)
8. [문제 해결](#8-문제-해결)

---

## 1. 개요

### 1.1 무엇인가?

**PyTorch Music Transformer**는 Transformer 아키텍처를 사용하여 MIDI 음악을 생성하는 딥러닝 모델입니다.

### 1.2 왜 PyTorch인가?

| 장점 | 설명 |
|------|------|
| **직관적** | Python 코드처럼 작성 가능 |
| **디버깅 쉬움** | 동적 그래프, 익숙한 Python 에러 |
| **커뮤니티** | 최신 논문 대부분 PyTorch 구현 |
| **유연성** | 커스텀 레이어/loss 쉽게 추가 |

### 1.3 QLoRA vs PyTorch Transformer

| 항목 | QLoRA | PyTorch Transformer |
|------|-------|---------------------|
| 기반 | GPT-2 (언어 모델) | Music Transformer (음악 전용) |
| 파라미터 | 전체의 0.1% 학습 | 전체 학습 |
| 메모리 | ~4GB | ~8-12GB |
| 목적 | 빠른 실험 | 최고 품질 |
| 커스터마이징 | 제한적 | 완전 자유 |

### 1.4 Magenta vs PyTorch

| 항목 | Magenta | PyTorch Transformer |
|------|---------|---------------------|
| 프레임워크 | TensorFlow | PyTorch |
| 학습 곡선 | 가파름 | 완만함 |
| 문서화 | 풍부 | 직접 작성 필요 |
| 제어 | 높음 (established) | 최고 (full control) |

---

## 2. 아키텍처

### 2.1 Music Transformer 개요

Music Transformer는 2018년 Google에서 발표한 모델로, 긴 시퀀스의 음악을 생성하는데 특화되어 있습니다.

**핵심 특징:**
- **Relative Position Embeddings**: 긴 시퀀스 처리
- **Self-Attention**: 음악의 장기 의존성 학습
- **Autoregressive**: 이전 토큰 기반 다음 토큰 예측

### 2.2 모델 구조

```
Input MIDI Tokens
    ↓
Token Embedding
    ↓
[Transformer Layer × N]
    ├─ Relative Multi-Head Attention
    ├─ Layer Normalization
    ├─ Feed-Forward Network
    └─ Layer Normalization
    ↓
Output Projection
    ↓
Softmax (next token probabilities)
```

### 2.3 MIDI 토크나이제이션

MIDI 이벤트를 정수 토큰으로 변환:

```python
NOTE_ON:    0-127    # 노트 시작 (pitch)
NOTE_OFF:   128-255  # 노트 종료 (pitch)
TIME_SHIFT: 256-383  # 시간 이동 (10ms 단위)
VELOCITY:   384-447  # 음량 (0-63)
SPECIAL:    448-511  # PAD, BOS, EOS, MASK
```

예시:
```
[BOS, NOTE_ON(60), VELOCITY(40), TIME_SHIFT(12), NOTE_OFF(60), ...]
 = [449, 60, 424, 268, 188, ...]
```

---

## 3. 환경 설정

### 3.1 필수 요구사항

- **Python**: 3.8 이상
- **GPU**: NVIDIA GPU 8GB+ (RTX 3060 Ti 이상 권장)
- **CUDA**: 11.8 이상
- **디스크**: 10GB+

### 3.2 설치

```bash
# 1. 가상환경 생성
python3 -m venv pytorch-env
source pytorch-env/bin/activate  # Linux/Mac
# 또는
pytorch-env\Scripts\activate  # Windows

# 2. PyTorch 설치 (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. 의존성 설치
pip install -r requirements-pytorch.txt

# 4. GPU 확인
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## 4. 데이터 준비

### 4.1 MIDI 파일 수집

Brad Mehldau MIDI 파일을 `data/raw_midi/` 폴더에 저장:

```bash
mkdir -p data/raw_midi
# 30-50개 Brad Mehldau MIDI 파일을 복사
```

**권장 곡:**
- Anything Goes
- Blackbird
- Exit Music (For a Film)
- Ron's Place
- Song-Song

### 4.2 샘플 데이터 생성 (테스트용)

실제 MIDI가 없다면 샘플 생성:

```bash
python scripts/generate_sample_midi.py \
  --output data/raw_midi \
  --count 30 \
  --duration 120
```

### 4.3 데이터 검증

```bash
# MIDI 파일 개수 확인
ls data/raw_midi/*.mid | wc -l

# 샘플 MIDI 파일 확인
ls -lh data/raw_midi/ | head
```

---

## 5. 모델 학습

### 5.1 설정 확인

`configs/pytorch_transformer_config.yaml`:

```yaml
model:
  d_model: 512       # 모델 차원
  num_heads: 8       # Attention heads
  num_layers: 6      # Transformer layers

training:
  batch_size: 4      # GPU 메모리에 따라 조정
  epochs: 100
  learning_rate: 1.0e-4
```

### 5.2 학습 시작

```bash
# 기본 설정으로 학습
python scripts/train_pytorch_transformer.py \
  --config configs/pytorch_transformer_config.yaml

# TensorBoard로 모니터링 (별도 터미널)
tensorboard --logdir=models/finetuned/pytorch_transformer
```

### 5.3 체크포인트에서 재개

```bash
python scripts/train_pytorch_transformer.py \
  --config configs/pytorch_transformer_config.yaml \
  --resume models/finetuned/pytorch_transformer/checkpoint_epoch_50.pt
```

### 5.4 학습 모니터링

**TensorBoard 지표:**
- `train_loss`: 학습 손실 (감소해야 함)
- `val_loss`: 검증 손실 (감소해야 함)
- `val_perplexity`: 혼란도 (낮을수록 좋음)
- `learning_rate`: 학습률 (cosine schedule)

**주의사항:**
- Val loss가 증가하면 **overfitting** → early stopping
- Train/Val loss 차이가 크면 **regularization** 강화

### 5.5 예상 학습 시간

| GPU | 배치 크기 | 에폭당 시간 | 총 시간 (100 에폭) |
|-----|----------|------------|-------------------|
| RTX 3060 Ti (8GB) | 2 | ~15분 | ~25시간 |
| RTX 3080 (10GB) | 4 | ~10분 | ~17시간 |
| RTX 3090 (24GB) | 8 | ~7분 | ~12시간 |
| A100 (40GB) | 16 | ~5분 | ~8시간 |

---

## 6. 음악 생성

### 6.1 기본 생성

```bash
python scripts/generate_pytorch_transformer.py \
  --checkpoint models/finetuned/pytorch_transformer/best_model.pt \
  --output output/generated.mid
```

### 6.2 여러 샘플 생성

```bash
python scripts/generate_pytorch_transformer.py \
  --checkpoint models/finetuned/pytorch_transformer/best_model.pt \
  --output output/brad_style.mid \
  --num_samples 10 \
  --temperature 1.2
```

### 6.3 Primer 타입

```bash
# Random primer (기본)
--primer_type random

# 스케일 primer (C major ascending)
--primer_type scale

# 코드 primer (C major chord)
--primer_type chord
```

### 6.4 샘플링 파라미터

| 파라미터 | 값 | 효과 |
|---------|-----|------|
| `--temperature` | 0.5 | 보수적, 반복적 |
| `--temperature` | 1.0 | 균형잡힌 (기본값) |
| `--temperature` | 1.5 | 창의적, 예측 불가능 |
| `--top_k` | 20 | 제한적 선택 |
| `--top_k` | 40 | 균형 (기본값) |
| `--top_k` | 80 | 다양성 증가 |
| `--top_p` | 0.8 | 보수적 |
| `--top_p` | 0.9 | 균형 (기본값) |
| `--top_p` | 0.95 | 창의적 |

### 6.5 MIDI 재생

생성된 MIDI 파일 재생:

- **MuseScore** (무료): https://musescore.org/
- **GarageBand** (Mac)
- **FL Studio** / **Ableton Live**
- **Online MIDI Player**: https://onlinesequencer.net/

---

## 7. 하이퍼파라미터 튜닝

### 7.1 모델 크기 조정

**더 작은 모델 (메모리 부족 시):**

```yaml
model:
  d_model: 256      # 512 → 256
  num_layers: 4     # 6 → 4
  d_ff: 1024        # 2048 → 1024

training:
  batch_size: 2     # 4 → 2
```

**더 큰 모델 (고품질):**

```yaml
model:
  d_model: 768      # 512 → 768
  num_layers: 12    # 6 → 12
  d_ff: 3072        # 2048 → 3072

training:
  batch_size: 2     # GPU 메모리 고려
```

### 7.2 학습률 조정

```yaml
training:
  learning_rate: 5.0e-5  # 더 안정적
  learning_rate: 2.0e-4  # 더 빠른 수렴 (기본값)
  learning_rate: 5.0e-4  # 더 빠름 (불안정 가능)
```

### 7.3 Regularization

**Overfitting 방지:**

```yaml
model:
  dropout: 0.2      # 0.1 → 0.2

training:
  weight_decay: 0.05  # 0.01 → 0.05

data:
  augment: true     # Transpose 증강
```

### 7.4 데이터 증강

더 많은 증강 방법 (코드 수정 필요):

- **Transpose**: ±12 semitones
- **Tempo scaling**: 0.8x - 1.2x
- **Velocity variation**: ±20
- **Time stretching**: Slight variations

---

## 8. 문제 해결

### 8.1 일반적인 오류

#### GPU out of memory

**증상**:
```
RuntimeError: CUDA out of memory
```

**해결책**:
```yaml
# 배치 크기 감소
training:
  batch_size: 2  # 또는 1

# 또는 모델 크기 감소
model:
  d_model: 256
  num_layers: 4
```

#### Loss가 감소하지 않음

**원인**:
- Learning rate가 너무 낮음
- 데이터 품질 문제
- 모델 너무 작음

**해결책**:
```yaml
training:
  learning_rate: 2.0e-4  # 증가

model:
  num_layers: 8  # 증가
```

#### Overfitting (Val loss 증가)

**원인**:
- 데이터 부족
- 모델 너무 큼
- Regularization 부족

**해결책**:
```yaml
model:
  dropout: 0.2  # 증가

training:
  weight_decay: 0.05  # 증가

# Early stopping 사용
# 또는 더 많은 MIDI 파일 추가
```

### 8.2 생성 품질 개선

**단조로운 생성:**
```bash
# Temperature 증가
--temperature 1.3

# Top-k 증가
--top_k 60
```

**너무 무작위:**
```bash
# Temperature 감소
--temperature 0.8

# Top-k 감소
--top_k 30
```

**반복적인 패턴:**
- 더 많은 데이터 추가
- 모델 크기 증가
- Dropout 증가

---

## 9. 고급 기능

### 9.1 커스텀 Loss 함수

`scripts/train_pytorch_transformer.py` 수정:

```python
# Focal Loss for harder examples
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

# 사용
criterion = FocalLoss()
```

### 9.2 Warm-up Scheduler

```python
from torch.optim.lr_scheduler import LinearLR, SequentialLR

# Warm-up + Cosine
warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=1000)
cosine_scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

scheduler = SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, cosine_scheduler],
    milestones=[1000]
)
```

### 9.3 Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    with autocast():
        output = model(input)
        loss = criterion(output, target)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

---

## 10. 평가 지표

### 10.1 자동 지표

```python
# Perplexity (낮을수록 좋음)
perplexity = torch.exp(val_loss)

# Accuracy (참고용)
acc = (predictions == targets).float().mean()
```

### 10.2 음악적 평가

- **화성 진행**: Brad Mehldau 스타일 코드 진행 재현
- **리듬 패턴**: 재즈 특유의 스윙, 싱코페이션
- **멜로디**: 자연스러운 프레이징
- **다이나믹스**: 벨로시티 변화

### 10.3 인간 평가

A/B 테스트:
1. 원본 Brad Mehldau MIDI
2. 생성된 MIDI
3. 블라인드 테스트 (5-10명)
4. 설문: "어느 것이 Brad Mehldau 스타일에 가까운가?"

---

## 11. 다음 단계

### 11.1 모델 개선

- [ ] 더 큰 모델 (d_model=768, layers=12)
- [ ] 더 많은 데이터 (100+ MIDI 파일)
- [ ] Conditional generation (스타일 제어)
- [ ] Multi-track 생성 (베이스, 드럼 추가)

### 11.2 배포

- [ ] ONNX 변환 (추론 최적화)
- [ ] TorchScript (모바일/프로덕션)
- [ ] FastAPI 서버
- [ ] 웹 인터페이스 (React)

### 11.3 실시간 생성

- [ ] Streaming inference
- [ ] Chunk-based generation
- [ ] Low-latency optimization

---

## 12. 참고 자료

### 논문
- [Music Transformer](https://arxiv.org/abs/1809.04281) - Original paper
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer
- [Relative Position Embeddings](https://arxiv.org/abs/1803.02155)

### 코드
- [Official Magenta Implementation](https://github.com/magenta/magenta/tree/main/magenta/models/music_transformer)
- [Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)

### 블로그
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [PyTorch Transformer Tutorial](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)

---

## 라이센스

학습 및 연구 목적으로만 사용

---

**작성일**: 2025-11-16
**버전**: 1.0
**프로젝트**: realTimeDeepLearningModelFineTuning - PyTorch Edition
