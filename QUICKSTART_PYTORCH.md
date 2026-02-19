# PyTorch Music Transformer 빠른 시작

Brad Mehldau 스타일 재즈 피아노 생성을 위한 순수 PyTorch 구현

## 5분 빠른 시작

### 1. 환경 설정

```bash
# 가상환경 생성 및 활성화
python3 -m venv pytorch-env
source pytorch-env/bin/activate

# 의존성 설치
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements-pytorch.txt

# GPU 확인
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### 2. 데이터 준비

```bash
# 샘플 MIDI 생성 (테스트용)
python scripts/generate_sample_midi.py --count 30 --duration 120

# 또는 Brad Mehldau MIDI 파일을 data/raw_midi/에 복사
mkdir -p data/raw_midi
# MIDI 파일들을 복사...
```

### 3. 모델 학습

```bash
# 기본 설정으로 학습 시작
python scripts/train_pytorch_transformer.py \
  --config configs/pytorch_transformer_config.yaml

# 별도 터미널에서 TensorBoard 실행
tensorboard --logdir=models/finetuned/pytorch_transformer
# http://localhost:6006 접속
```

### 4. 음악 생성

```bash
# 학습된 모델로 MIDI 생성
python scripts/generate_pytorch_transformer.py \
  --checkpoint models/finetuned/pytorch_transformer/best_model.pt \
  --output output/generated.mid \
  --num_samples 5

# 생성된 MIDI 파일 확인
ls -lh output/
```

---

## 주요 파라미터

### 학습

| 파라미터 | 기본값 | 설명 |
|---------|-------|------|
| `--config` | configs/pytorch_transformer_config.yaml | 설정 파일 |
| `--resume` | None | 체크포인트에서 재개 |

### 생성

| 파라미터 | 기본값 | 설명 |
|---------|-------|------|
| `--checkpoint` | (필수) | 모델 체크포인트 경로 |
| `--num_samples` | 1 | 생성할 샘플 수 |
| `--temperature` | 1.0 | 샘플링 temperature |
| `--top_k` | 40 | Top-k 샘플링 |
| `--primer_type` | random | Primer 타입 (random/scale/chord) |

---

## 설정 커스터마이징

`configs/pytorch_transformer_config.yaml` 수정:

```yaml
# GPU 메모리 부족 시
training:
  batch_size: 2  # 4 → 2

# 더 빠른 학습 (품질 trade-off)
training:
  epochs: 50  # 100 → 50

model:
  num_layers: 4  # 6 → 4
```

---

## 문제 해결

### GPU out of memory
```yaml
# configs/pytorch_transformer_config.yaml
training:
  batch_size: 1  # 최소화
```

### 학습이 너무 느림
```yaml
model:
  d_model: 256  # 512 → 256
  num_layers: 4  # 6 → 4
```

### 생성 품질 낮음
- 더 많은 MIDI 파일 추가 (50개 이상)
- 더 많은 에폭 학습 (200+)
- Temperature 조정 (0.8 - 1.5)

---

## 다음 단계

상세한 내용은 [PYTORCH_TRANSFORMER_GUIDE.md](./PYTORCH_TRANSFORMER_GUIDE.md) 참조

**주요 내용**:
- 아키텍처 상세 설명
- 하이퍼파라미터 튜닝
- 고급 기능 (커스텀 loss, mixed precision)
- 평가 지표 및 A/B 테스트
