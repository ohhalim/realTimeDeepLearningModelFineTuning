# Real-time Deep Learning Model Fine-tuning

Brad Mehldau 스타일의 재즈 피아노를 생성하는 Magenta Music Transformer 파인튜닝 프로젝트

## 빠른 시작

### 1. 환경 설정
```bash
python3 -m venv magenta-env
source magenta-env/bin/activate
pip install -r requirements.txt
```

### 2. MIDI 파일 준비
- `data/raw_midi/` 폴더에 Brad Mehldau MIDI 파일 30개 추가

### 3. 데이터 전처리
```bash
python scripts/preprocess.py
```

### 4. 모델 학습
```bash
python scripts/train.py
```

### 5. 음악 생성
```bash
python scripts/generate.py
```

## 상세 가이드

전체 프로세스는 [MAGENTA_FINETUNING_GUIDE.md](./MAGENTA_FINETUNING_GUIDE.md)를 참조하세요.

## 프로젝트 구조
```
.
├── data/
│   ├── raw_midi/          # Brad Mehldau 원본 MIDI 파일 (30개)
│   ├── processed/         # 전처리된 데이터
│   └── tfrecord/         # TensorFlow Record 형식
│       ├── train/        # 학습 데이터 (24개)
│       ├── val/          # 검증 데이터 (3개)
│       └── test/         # 테스트 데이터 (3개)
├── models/
│   ├── pretrained/       # 사전 학습된 모델
│   └── finetuned/        # 파인튜닝된 모델
├── scripts/
│   ├── preprocess.py     # 데이터 전처리
│   ├── train.py          # 모델 학습
│   ├── generate.py       # 음악 생성
│   └── realtime_generate.py  # 실시간 생성
├── notebooks/            # Jupyter 노트북
└── output/              # 생성된 MIDI 파일
```

## 요구사항
- Python 3.8+
- TensorFlow 2.11.0
- Magenta
- CUDA (GPU 사용 시)

## 라이센스
학습 및 연구 목적으로만 사용
