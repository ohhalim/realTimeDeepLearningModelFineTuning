# Phase 1: 첫 번째 파인튜닝 (Charlie Parker) 🎺

**목표**: 실제로 동작하는 첫 번째 AI 모델 학습 및 테스트

**예상 시간**: 12-16시간 (대부분 학습 대기)
**실제 작업**: 2-3시간
**비용**: $10-20 (약 13,000-26,000원)

---

## 사전 준비

✅ Phase 0 완료 확인:
```bash
python scripts/phase0/check_setup.py
# 모든 체크 통과해야 함
```

---

## Step 1: 학습 설정 파일 작성 (15분)

### 1-1. 설정 파일 생성

`configs/charlie_parker_v1.yaml` 파일을 만듭니다:

```yaml
# Model configuration
model:
  vocab_size: 2048  # Aria tokenizer vocab
  d_model: 256
  n_heads: 4
  n_layers: 4
  d_ff: 1024
  max_len: 512
  dropout: 0.1
  n_tasks: 5
  n_genres: 2

# Training configuration
training:
  batch_size: 16  # RTX 3090: 16, RTX 4090: 32
  learning_rate: 1e-4
  weight_decay: 0.01
  n_epochs: 50
  warmup_steps: 500
  grad_clip_norm: 1.0

  # Multi-task learning
  corruption_prob: 0.5  # 50% of samples get corrupted

  # Checkpointing
  checkpoint_dir: "checkpoints/charlie_parker_v1"
  save_every_n_steps: 500
  keep_n_checkpoints: 3

  # Logging
  log_every_n_steps: 50
  eval_every_n_steps: 500

  # Optimization
  use_amp: true  # Mixed precision (GPU only)

  # Reproducibility
  seed: 42
  deterministic: true

# Data configuration
data:
  train_data: "data/tokenized/train"
  val_data: "data/tokenized/val"
  num_workers: 4
```

### 1-2. 작은 모델로 테스트 (선택)

먼저 작은 모델로 빠르게 테스트하고 싶다면:

`configs/charlie_parker_tiny.yaml`:
```yaml
model:
  d_model: 128  # 작게
  n_heads: 2
  n_layers: 2

training:
  batch_size: 8
  n_epochs: 10  # 빠르게
```

---

## Step 2: 데이터 전처리 (30분-1시간)

### 2-1. MIDI → 토큰 변환

```bash
# Aria tokenizer로 MIDI 변환
python scripts/preprocess_midi.py \
  --input data/jazz_midi/ \
  --output data/tokenized/ \
  --chunk_size 512 \
  --stride 256

# 출력 예시:
# Processing 124 MIDI files...
# Total chunks created: 3,847
# Average tokens per chunk: 456
# Saved to: data/tokenized/
```

### 2-2. 학습/검증 분할

```bash
# 80/20 split
python scripts/split_dataset.py \
  --input data/tokenized/ \
  --train_ratio 0.8 \
  --seed 42

# 출력:
# Train: 3,078 chunks
# Val: 769 chunks
```

---

## Step 3: RunPod에서 학습 시작 (8-12시간)

### 3-1. RunPod Pod 시작

1. RunPod 로그인: https://runpod.io
2. **Pods** → **+ Deploy**
3. GPU 선택: **RTX 3090** ($0.34/hour)
4. Template: **RunPod PyTorch 2.0**
5. Volume: **50GB Network Volume** (데이터 저장용, 선택)
6. **Deploy** 클릭

Pod이 시작될 때까지 1-2분 대기

### 3-2. 코드 업로드

```bash
# Pod SSH 정보 확인
# RunPod Dashboard에서 SSH 주소 복사
# 예: ssh root@x.x.x.x -p 12345 -i ~/.ssh/id_ed25519

# 간단히 pod 주소를 alias로 설정
export RUNPOD_SSH="root@x.x.x.x -p 12345 -i ~/.ssh/id_ed25519"

# 코드 업로드 (rsync)
rsync -avz --exclude='venv' --exclude='data' --exclude='checkpoints' \
  ./ $RUNPOD_SSH:/workspace/jazz-ai/

# 데이터 업로드
rsync -avz data/tokenized/ $RUNPOD_SSH:/workspace/jazz-ai/data/tokenized/

# 예상 시간: 5-10분 (데이터 크기에 따라)
```

### 3-3. SSH 접속 및 학습 시작

```bash
# SSH 접속
ssh $RUNPOD_SSH

# Pod 내부에서:
cd /workspace/jazz-ai

# 의존성 설치 (처음만)
pip install -r requirements.txt

# 학습 시작!
python scripts/train.py \
  --config configs/charlie_parker_v1.yaml \
  2>&1 | tee logs/training_v1.log

# 출력:
# ==============================================================
# Starting Multi-Task Training
# ==============================================================
#   Device: cuda
#   Model parameters: 8,234,560
#   Batch size: 16
#   Learning rate: 0.0001
#   Epochs: 50
#   Total steps: 9,619
#   Corruption prob: 0.5
#   Seed: 42
# ==============================================================
#
# Epoch 1/50: 100%|████████| 192/192 [02:15<00:00]
#   loss: 6.234, lr: 1.00e-04
# [Step 500] Val loss: 5.987
#   ✓ Saved checkpoint
# ...
```

**예상 학습 시간**:
- RTX 3090: 8-10시간 (50 epochs)
- RTX 4090: 5-6시간 (50 epochs)

### 3-4. 학습 모니터링 (백그라운드)

학습을 백그라운드로 돌리려면:

```bash
# tmux 사용 (권장)
tmux new -s training
python scripts/train.py --config configs/charlie_parker_v1.yaml
# Ctrl+B, D로 detach

# 나중에 다시 보기
tmux attach -t training

# 또는 nohup
nohup python scripts/train.py --config configs/charlie_parker_v1.yaml > training.log 2>&1 &
```

로컬에서 로그 확인:

```bash
# 실시간 로그 확인 (로컬에서)
ssh $RUNPOD_SSH "tail -f /workspace/jazz-ai/logs/training_v1.log"
```

---

## Step 4: 중간 체크포인트 테스트 (학습 중)

학습이 진행되는 동안, 중간 체크포인트를 다운로드해서 테스트할 수 있습니다:

### 4-1. 체크포인트 다운로드

```bash
# 로컬로 체크포인트 다운로드
rsync -avz $RUNPOD_SSH:/workspace/jazz-ai/checkpoints/charlie_parker_v1/checkpoint_step_1000.pt \
  ./checkpoints/charlie_parker_v1/

# 파일 크기 확인
ls -lh checkpoints/charlie_parker_v1/checkpoint_step_1000.pt
# ~35MB 정도
```

### 4-2. 샘플 생성

```bash
# 로컬에서 샘플 생성
python scripts/generate.py \
  --checkpoint checkpoints/charlie_parker_v1/checkpoint_step_1000.pt \
  --prompt data/test_prompts/blues_in_f.mid \
  --output samples/test_1000_blues.mid \
  --max_tokens 256 \
  --temperature 0.9

# MIDI → MP3 변환 (청취)
python scripts/midi_to_mp3.py \
  --input samples/test_1000_blues.mid \
  --output samples/test_1000_blues.mp3

# 들어보기
open samples/test_1000_blues.mp3  # macOS
# xdg-open samples/test_1000_blues.mp3  # Linux
```

**평가 기준**:
- Step 1000: 아직 많이 이상함
- Step 3000: 리듬감 생김
- Step 5000: 멜로디 라인 생김
- Step 8000+: 재즈스러워짐

---

## Step 5: 학습 완료 및 다운로드

### 5-1. 학습 완료 확인

```bash
# SSH로 접속해서 확인
ssh $RUNPOD_SSH

cd /workspace/jazz-ai

# 마지막 로그 확인
tail -50 logs/training_v1.log

# 출력 예시:
# Epoch 50/50 completed in 145.3s
#
# ======================================================================
# Training completed!
#   Best val loss: 2.347
# ======================================================================
```

### 5-2. Best model 다운로드

```bash
# 로컬로 전체 체크포인트 폴더 다운로드
rsync -avz $RUNPOD_SSH:/workspace/jazz-ai/checkpoints/charlie_parker_v1/ \
  ./checkpoints/charlie_parker_v1/

# Best model 확인
ls -lh checkpoints/charlie_parker_v1/best_model.pt
```

### 5-3. Pod 종료 (비용 절약!)

**중요**: 학습이 끝나면 바로 Pod을 종료해야 합니다!

RunPod Dashboard → Pods → **Terminate**

또는 CLI:
```bash
# Pod ID 확인
runpod pod list

# Pod 종료
runpod pod stop <pod_id>
```

**비용 계산**:
- 10시간 × $0.34/hour = $3.40 (약 4,400원)

---

## Step 6: 로컬에서 최종 테스트

### 6-1. 다양한 프롬프트로 생성

```bash
# Blues
python scripts/generate.py \
  --checkpoint checkpoints/charlie_parker_v1/best_model.pt \
  --prompt data/test_prompts/blues_in_f.mid \
  --output samples/v1_blues.mid

# Bebop
python scripts/generate.py \
  --checkpoint checkpoints/charlie_parker_v1/best_model.pt \
  --prompt data/test_prompts/bebop_lick.mid \
  --output samples/v1_bebop.mid

# Ballad
python scripts/generate.py \
  --checkpoint checkpoints/charlie_parker_v1/best_model.pt \
  --prompt data/test_prompts/ballad.mid \
  --output samples/v1_ballad.mid \
  --temperature 0.7  # Ballad는 더 부드럽게
```

### 6-2. 다양한 Temperature 실험

```bash
# Temperature 0.5 (보수적)
python scripts/generate.py \
  --checkpoint checkpoints/charlie_parker_v1/best_model.pt \
  --prompt data/test_prompts/blues_in_f.mid \
  --temperature 0.5 \
  --output samples/v1_blues_temp0.5.mid

# Temperature 0.9 (균형)
python scripts/generate.py \
  --checkpoint checkpoints/charlie_parker_v1/best_model.pt \
  --prompt data/test_prompts/blues_in_f.mid \
  --temperature 0.9 \
  --output samples/v1_blues_temp0.9.mid

# Temperature 1.2 (모험적)
python scripts/generate.py \
  --checkpoint checkpoints/charlie_parker_v1/best_model.pt \
  --prompt data/test_prompts/blues_in_f.mid \
  --temperature 1.2 \
  --output samples/v1_blues_temp1.2.mid

# 3개 들어보고 가장 좋은 temperature 찾기
```

### 6-3. 레이턴시 벤치마크

```bash
# 실제 레이턴시 측정
python -m realjazz.benchmarking \
  --checkpoint checkpoints/charlie_parker_v1/best_model.pt \
  --n_runs 100 \
  --device cpu

# 출력:
# 📊 Benchmark Results:
#   Device: cpu
#   Mean latency: 67.3ms ± 8.2ms
#   p50: 65.1ms
#   p95: 82.4ms
#   p99: 93.7ms
#
# ✅ Meets real-time target (<100ms)!
```

---

## Step 7: 결과 정리 및 문서화

### 7-1. 학습 곡선 시각화

```bash
python scripts/plot_training.py \
  --log checkpoints/charlie_parker_v1/training_log.json \
  --output figures/charlie_parker_v1_training.png
```

### 7-2. 샘플 정리

```bash
# 모든 샘플을 MP3로 변환
for mid in samples/v1_*.mid; do
  python scripts/midi_to_mp3.py --input "$mid"
done

# samples/ 폴더 정리
samples/
├── v1_blues.mid
├── v1_blues.mp3
├── v1_bebop.mid
├── v1_bebop.mp3
├── v1_ballad.mid
├── v1_ballad.mp3
├── v1_blues_temp0.5.mp3
├── v1_blues_temp0.9.mp3
└── v1_blues_temp1.2.mp3
```

### 7-3. README 작성

`checkpoints/charlie_parker_v1/README.md`:

```markdown
# Charlie Parker AI - v1

첫 번째 학습 결과

## 학습 정보

- **데이터**: 124 MIDI files (3.7 hours)
- **Epochs**: 50
- **Training time**: 9.2 hours
- **GPU**: RTX 3090
- **Cost**: $3.13

## 결과

- **Best val loss**: 2.347
- **Latency**: 67.3ms (p50)
- **Parameters**: 8.2M

## 샘플

- `samples/v1_blues.mp3`: Blues improvisation
- `samples/v1_bebop.mp3`: Bebop licks
- `samples/v1_ballad.mp3`: Ballad melody

## 평가

✅ 장점:
- 리듬감 좋음
- 재즈 스타일 학습됨
- 레이턴시 목표 달성

⚠️  개선점:
- 가끔 반복적인 패턴
- 화성감 약함
- 더 많은 데이터 필요

## 다음 단계

→ Phase 2에서 개선
```

---

## 체크리스트

Phase 1 완료 확인:

- [ ] 데이터 전처리 완료
- [ ] RunPod에서 50 epochs 학습 완료
- [ ] Best model 다운로드
- [ ] 샘플 생성 및 청취
- [ ] 레이턴시 <100ms 확인
- [ ] 학습 곡선 시각화
- [ ] README 작성
- [ ] 비용 $20 이하

---

## 실제 비용

| 항목 | 비용 |
|------|------|
| RunPod GPU (10시간) | $3.40 |
| Network Volume (선택) | $0.50 |
| **총계** | **~$4** |

**약 5,200원** (커피 1-2잔 값)

---

## 문제 해결

### Q1: "CUDA out of memory" 에러

```yaml
# configs/charlie_parker_v1.yaml 수정
training:
  batch_size: 8  # 16 → 8로 줄이기
```

### Q2: 학습이 너무 느림

- RTX 4090로 업그레이드 ($0.69/hour)
- 또는 Epoch 수 줄이기 (50 → 30)

### Q3: 샘플이 이상함

- 더 많은 epoch 필요 (50 → 80)
- Temperature 조정 (0.9 → 0.7)
- 더 많은 데이터 수집

### Q4: RunPod 접속 안 됨

```bash
# SSH 키 확인
ssh-add -l

# 없으면 추가
ssh-add ~/.ssh/id_ed25519
```

---

## 다음 단계

✅ Phase 1 완료!

**다음**: `guides/PHASE2_EVALUATION.md`

→ 모델을 객관적으로 평가하고 v2로 개선하기

---

**축하합니다! 첫 번째 AI 모델 완성! 🎉**
