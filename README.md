# Real-time Deep Learning Model Fine-tuning

Brad Mehldau 스타일의 재즈 피아노를 생성하는 딥러닝 모델 파인튜닝 프로젝트

## 🎹 세 가지 접근 방식

### 1. QLoRA Fine-tuning (PyTorch) ⭐ Phase 1 (현재)
**빠른 실험 및 학습**
- 메모리 효율적 (8GB GPU)
- PyTorch 생태계
- 빠른 프로토타이핑

### 2. Magenta RealTime (JAX) 🚀 Phase 2 (미래)
**진짜 실시간 음악 생성**
- 실시간 스트리밍 (RTF ≥ 1×)
- Text/Audio prompts
- Live audio injection
- Colab TPU (무료) 또는 40GB GPU

### 3. Magenta Music Transformer (TensorFlow)
**전통적 접근**
- 높은 품질
- 오프라인 생성

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

---

### 옵션 C: Magenta RealTime (Phase 2 🚀)

**진짜 실시간 음악 생성** - 추후 통합 예정

```bash
# Colab TPU에서 실험
# 1. Colab 노트북 열기
https://colab.research.google.com/github/magenta/magenta-realtime/blob/main/notebooks/Magenta_RT_Demo.ipynb

# 2. Brad Mehldau 오디오로 파인튜닝
# (공식 파인튜닝 노트북 참조)

# 3. 실시간 생성
# Text/Audio prompts로 스타일 제어
# Live audio injection으로 상호작용
```

**통합 로드맵**: [MAGENTA_RT_ROADMAP.md](./MAGENTA_RT_ROADMAP.md)

**주요 리소스**:
- [GitHub](https://github.com/magenta/magenta-realtime)
- [HuggingFace Model](https://huggingface.co/google/magenta-realtime)
- [Research Paper](https://arxiv.org/abs/2508.04651)

## 프로젝트 구조
```
.
├── configs/
│   └── qlora_config.yaml          # QLoRA 설정 파일
├── data/
│   └── raw_midi/                  # Brad Mehldau 원본 MIDI 파일 (30개+)
├── models/
│   └── finetuned/
│       ├── qlora_brad_mehldau/    # QLoRA 파인튜닝 (Phase 1)
│       └── magenta_rt/            # Magenta RT 파인튜닝 (Phase 2)
├── scripts/
│   ├── train_qlora.py             # QLoRA 학습 ⭐
│   ├── generate_qlora.py          # QLoRA 생성 ⭐
│   └── generate_sample_midi.py    # 샘플 MIDI 생성
├── notebooks/
│   └── Magenta_RT_Demo.ipynb      # Colab 데모 (복사본)
├── output/                        # 생성된 MIDI/오디오 파일
├── QLORA_GUIDE.md                 # QLoRA 상세 가이드
├── QUICKSTART_QLORA.md            # QLoRA 빠른 시작
└── MAGENTA_RT_ROADMAP.md          # Magenta RT 통합 로드맵
```

## 요구사항

### Phase 1: QLoRA (현재)
- Python 3.8+
- PyTorch 2.1.0+
- CUDA 11.8+ (NVIDIA GPU 8GB+ 권장)
- HuggingFace Transformers, PEFT, bitsandbytes

### Phase 2: Magenta RealTime (미래)
- Python 3.12
- JAX
- Colab TPU (무료) 또는 40GB GPU (A100, A6000)
- Linux (로컬 배포 시)

### 옵션: Magenta Music Transformer
- Python 3.8+
- TensorFlow 2.11.0
- Magenta 2.1.4
- CUDA (GPU 16GB+ 권장)

## 개발 로드맵

### ✅ Phase 1: QLoRA 기반 프로토타이핑 (완료)
- [x] QLoRA 학습 파이프라인
- [x] MIDI 토크나이저
- [x] 생성 스크립트
- [x] 문서화

### 🚧 Phase 1.5: 데이터 수집 및 실험 (진행중)
- [ ] Brad Mehldau MIDI 데이터 수집 (30-50개)
- [ ] QLoRA 모델 학습
- [ ] 생성 품질 평가

### 🔮 Phase 2: Magenta RealTime 통합 (예정)
- [ ] Colab TPU 파인튜닝 실험
- [ ] Brad Mehldau 스타일 임베딩
- [ ] 실시간 생성 시스템
- [ ] 웹 인터페이스

**상세 로드맵**: [MAGENTA_RT_ROADMAP.md](./MAGENTA_RT_ROADMAP.md)

## 라이센스
학습 및 연구 목적으로만 사용
