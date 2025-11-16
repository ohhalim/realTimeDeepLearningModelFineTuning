# Real-time Deep Learning Model Fine-tuning

Brad Mehldau 스타일의 재즈 피아노를 생성하는 딥러닝 모델 파인튜닝 프로젝트

## 🎹 두 가지 접근 방식

### 1. Magenta Music Transformer (TensorFlow)
전통적인 음악 생성 모델 - 높은 품질, 높은 리소스 요구

### 2. QLoRA Fine-tuning (PyTorch) ⭐ 추천
메모리 효율적인 최신 기법 - 낮은 GPU 요구사항, 빠른 학습

## 빠른 시작

### 옵션 A: QLoRA (추천 ⭐)

메모리 효율적이고 빠른 학습을 원한다면 QLoRA를 사용하세요.

```bash
# 1. 환경 설정
python3 -m venv qlora-env
source qlora-env/bin/activate
pip install -r requirements-qlora.txt

# 2. MIDI 파일 준비
mkdir -p data/raw_midi
# Brad Mehldau MIDI 파일 30개를 data/raw_midi/에 복사

# 3. QLoRA 파인튜닝 (2-8시간)
python scripts/train_qlora.py --config configs/qlora_config.yaml

# 4. 음악 생성
python scripts/generate_qlora.py \
  --checkpoint models/finetuned/qlora_brad_mehldau \
  --output output/brad_style.mid \
  --num_samples 5
```

**상세 가이드**: [QUICKSTART_QLORA.md](./QUICKSTART_QLORA.md) | [QLORA_GUIDE.md](./QLORA_GUIDE.md)

---

### 옵션 B: Magenta Music Transformer

전통적인 방식으로 높은 품질을 원한다면 Magenta를 사용하세요.

```bash
# 1. 환경 설정
python3 -m venv magenta-env
source magenta-env/bin/activate
pip install -r requirements.txt

# 2. MIDI 파일 준비
# data/raw_midi/에 Brad Mehldau MIDI 파일 30개 추가

# 3. 데이터 전처리
python scripts/preprocess.py

# 4. 모델 학습
python scripts/train.py

# 5. 음악 생성
python scripts/generate.py
```

**상세 가이드**: [MAGENTA_FINETUNING_GUIDE.md](./MAGENTA_FINETUNING_GUIDE.md)

## 프로젝트 구조
```
.
├── configs/
│   └── qlora_config.yaml      # QLoRA 설정 파일
├── data/
│   ├── raw_midi/              # Brad Mehldau 원본 MIDI 파일 (30개+)
│   ├── processed/             # 전처리된 데이터
│   └── tfrecord/              # TensorFlow Record 형식
├── models/
│   ├── pretrained/            # 사전 학습된 모델
│   └── finetuned/
│       ├── brad_mehldau/      # Magenta 파인튜닝
│       └── qlora_brad_mehldau/  # QLoRA 파인튜닝
├── scripts/
│   ├── preprocess.py          # Magenta 데이터 전처리
│   ├── train.py               # Magenta 학습
│   ├── generate.py            # Magenta 생성
│   ├── train_qlora.py         # QLoRA 학습 ⭐
│   └── generate_qlora.py      # QLoRA 생성 ⭐
├── notebooks/                 # Jupyter 노트북
└── output/                    # 생성된 MIDI 파일
```

## 요구사항

### QLoRA (추천)
- Python 3.8+
- PyTorch 2.1.0+
- CUDA 11.8+ (NVIDIA GPU 8GB+ 권장)
- HuggingFace Transformers, PEFT, bitsandbytes

### Magenta
- Python 3.8+
- TensorFlow 2.11.0
- Magenta 2.1.4
- CUDA (GPU 16GB+ 권장)

## 라이센스
학습 및 연구 목적으로만 사용
