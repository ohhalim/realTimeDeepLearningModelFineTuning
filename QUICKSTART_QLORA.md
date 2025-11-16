# QLoRA 빠른 시작 가이드

Brad Mehldau 스타일 재즈 피아노 생성을 위한 QLoRA 파인튜닝 빠른 시작 가이드입니다.

## 1단계: 환경 설정 (5분)

```bash
# 가상환경 생성
python3 -m venv qlora-env
source qlora-env/bin/activate

# 패키지 설치
pip install -r requirements-qlora.txt

# GPU 확인
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

## 2단계: MIDI 데이터 준비 (10분)

```bash
# MIDI 파일을 data/raw_midi/ 폴더에 저장
mkdir -p data/raw_midi
# Brad Mehldau MIDI 파일 30개를 이 폴더에 복사

# 파일 확인
ls -lh data/raw_midi/
```

## 3단계: 파인튜닝 실행 (2-8시간)

```bash
# QLoRA 파인튜닝 시작
python scripts/train_qlora.py --config configs/qlora_config.yaml

# 별도 터미널에서 TensorBoard 실행
tensorboard --logdir=models/finetuned/qlora_brad_mehldau
```

## 4단계: 음악 생성 (1분)

```bash
# 파인튜닝된 모델로 음악 생성
python scripts/generate_qlora.py \
  --checkpoint models/finetuned/qlora_brad_mehldau \
  --output output/brad_mehldau_style_1.mid \
  --temperature 1.0

# 여러 샘플 생성
python scripts/generate_qlora.py \
  --checkpoint models/finetuned/qlora_brad_mehldau \
  --output output/brad_style.mid \
  --num_samples 5 \
  --temperature 1.2
```

## 5단계: MIDI 재생

생성된 MIDI 파일을 DAW나 MIDI 플레이어로 재생:
- MuseScore (무료)
- GarageBand (Mac)
- FL Studio
- Ableton Live

---

## 설정 조정

### GPU 메모리 부족 시

`configs/qlora_config.yaml` 파일 수정:

```yaml
batch_size: 2  # 4 → 2로 감소
gradient_accumulation_steps: 16  # 증가
```

### 학습 속도 향상

```yaml
epochs: 50  # 100 → 50으로 감소
qlora:
  lora_r: 8  # 16 → 8로 감소
```

---

## 문제 해결

### CUDA out of memory
```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
python scripts/train_qlora.py --config configs/qlora_config.yaml
```

### No MIDI files found
```bash
# 경로 확인
ls data/raw_midi/*.mid
# 파일이 없으면 MIDI 파일을 해당 폴더에 복사
```

---

## 다음 단계

자세한 내용은 [QLORA_GUIDE.md](./QLORA_GUIDE.md)를 참조하세요.

**주요 내용**:
- 하이퍼파라미터 조정
- 고급 생성 기법
- 품질 평가 방법
- 문제 해결 상세 가이드
