# QLoRA 파인튜닝 가이드 - Brad Mehldau 스타일

## 개요

이 가이드는 **QLoRA (Quantized Low-Rank Adaptation)**를 사용하여 Brad Mehldau 스타일의 재즈 피아노 음악을 생성하는 모델을 파인튜닝하는 방법을 설명합니다.

### QLoRA란?

QLoRA는 대규모 언어 모델을 효율적으로 파인튜닝하는 기술로, 다음과 같은 장점이 있습니다:

- **메모리 효율성**: 4-bit 양자화로 GPU 메모리 사용량 75% 감소
- **빠른 학습**: LoRA를 통해 전체 파라미터의 0.1%만 학습
- **높은 성능**: 전체 파인튜닝과 유사한 성능
- **저사양 GPU 지원**: 8GB GPU에서도 대규모 모델 파인튜닝 가능

## 목차

1. [환경 설정](#1-환경-설정)
2. [데이터 준비](#2-데이터-준비)
3. [QLoRA 파인튜닝 실행](#3-qlora-파인튜닝-실행)
4. [음악 생성](#4-음악-생성)
5. [하이퍼파라미터 조정](#5-하이퍼파라미터-조정)
6. [문제 해결](#6-문제-해결)

---

## 1. 환경 설정

### 1.1 필수 요구사항

- **Python**: 3.8 이상
- **GPU**: NVIDIA GPU (CUDA 지원, 8GB VRAM 이상 권장)
- **CUDA**: 11.8 이상
- **디스크 공간**: 10GB 이상

### 1.2 가상환경 생성

```bash
# Python 가상환경 생성
python3 -m venv qlora-env
source qlora-env/bin/activate  # Linux/Mac
# 또는
qlora-env\Scripts\activate  # Windows
```

### 1.3 패키지 설치

```bash
# QLoRA 전용 requirements 설치
pip install -r requirements-qlora.txt

# GPU 확인
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 1.4 디렉토리 구조

```
realTimeDeepLearningModelFineTuning/
├── configs/
│   └── qlora_config.yaml        # QLoRA 설정 파일
├── data/
│   └── raw_midi/                # Brad Mehldau MIDI 파일 (30개)
├── models/
│   └── finetuned/
│       └── qlora_brad_mehldau/  # 파인튜닝 결과
├── scripts/
│   ├── train_qlora.py           # QLoRA 학습 스크립트
│   └── generate_qlora.py        # 음악 생성 스크립트
└── output/                      # 생성된 MIDI 파일
```

---

## 2. 데이터 준비

### 2.1 MIDI 파일 수집

Brad Mehldau의 MIDI 파일을 `data/raw_midi/` 폴더에 저장합니다.

**권장 곡 목록**:
- Anything Goes
- Blackbird
- Exit Music (For a Film)
- Paranoid Android
- River Man
- Ron's Place
- Song-Song
- Unrequited
- Young and Foolish
- ...기타 Brad Mehldau 솔로/트리오 곡들

**최소 요구사항**:
- 파일 수: 30개 이상 (더 많을수록 좋음)
- 파일 길이: 각 1분 이상
- 포맷: .mid 또는 .midi

### 2.2 데이터 검증

```bash
# MIDI 파일 확인
ls -lh data/raw_midi/

# 파일 개수 확인
ls data/raw_midi/*.mid | wc -l
```

---

## 3. QLoRA 파인튜닝 실행

### 3.1 설정 파일 확인

`configs/qlora_config.yaml` 파일을 열어 설정을 확인합니다:

```yaml
# 주요 설정
model_name: "gpt2"              # Base model
midi_dir: "data/raw_midi"       # MIDI 데이터 경로
epochs: 100                     # 학습 에폭
batch_size: 4                   # 배치 크기
learning_rate: 2.0e-4           # 학습률

# QLoRA 설정
qlora:
  lora_r: 16                    # LoRA rank
  lora_alpha: 32                # LoRA alpha
  lora_dropout: 0.05            # Dropout
```

### 3.2 학습 시작

```bash
# 기본 설정으로 학습
python scripts/train_qlora.py --config configs/qlora_config.yaml
```

**예상 학습 시간**:
- RTX 3090 (24GB): 약 2-3시간
- RTX 3080 (10GB): 약 4-5시간
- RTX 3060 (8GB): 약 6-8시간

### 3.3 학습 모니터링

```bash
# TensorBoard 실행
tensorboard --logdir=models/finetuned/qlora_brad_mehldau

# 브라우저에서 http://localhost:6006 접속
```

**확인 사항**:
- Loss가 점진적으로 감소하는지
- Validation loss가 증가하지 않는지 (overfitting 방지)
- GPU 메모리 사용량

---

## 4. 음악 생성

### 4.1 기본 생성

```bash
# 1개 샘플 생성
python scripts/generate_qlora.py \
  --checkpoint models/finetuned/qlora_brad_mehldau \
  --output output/brad_style_1.mid
```

### 4.2 여러 샘플 생성

```bash
# 10개 샘플 생성 (다양한 temperature)
python scripts/generate_qlora.py \
  --checkpoint models/finetuned/qlora_brad_mehldau \
  --output output/brad_style.mid \
  --num_samples 10 \
  --temperature 1.2
```

### 4.3 생성 파라미터

| 파라미터 | 값 | 효과 |
|---------|-----|------|
| `--temperature` | 0.5 | 보수적, 원본과 유사 |
| `--temperature` | 1.0 | 균형잡힌 창의성 (기본값) |
| `--temperature` | 1.5 | 매우 창의적, 예측 불가능 |
| `--max_length` | 512 | 짧은 곡 (~30초) |
| `--max_length` | 1024 | 중간 길이 (~1-2분) |
| `--max_length` | 2048 | 긴 곡 (~3-4분) |
| `--tempo` | 60-180 | BPM 설정 |

---

## 5. 하이퍼파라미터 조정

### 5.1 메모리 부족 시

```yaml
# configs/qlora_config.yaml
batch_size: 2                    # 4 → 2로 감소
gradient_accumulation_steps: 16  # 8 → 16으로 증가
```

또는 더 작은 모델 사용:

```yaml
model_name: "gpt2"  # 이미 가장 작은 모델
# 또는
model_name: "distilgpt2"  # 더 작은 distilled 버전
```

### 5.2 학습이 너무 느릴 때

```yaml
# LoRA rank 감소
qlora:
  lora_r: 8  # 16 → 8
  lora_alpha: 16  # 32 → 16

# 에폭 수 감소
epochs: 50  # 100 → 50
```

### 5.3 품질 개선

```yaml
# 더 큰 모델 사용 (16GB+ GPU 필요)
model_name: "gpt2-medium"

# LoRA rank 증가
qlora:
  lora_r: 32
  lora_alpha: 64

# 더 많은 에폭
epochs: 200
```

### 5.4 Overfitting 방지

```yaml
# Dropout 증가
qlora:
  lora_dropout: 0.1  # 0.05 → 0.1

# Weight decay 증가
weight_decay: 0.05  # 0.01 → 0.05
```

---

## 6. 문제 해결

### 6.1 일반적인 오류

#### 오류: CUDA out of memory

**해결책**:
```yaml
# 배치 크기 감소
batch_size: 1
gradient_accumulation_steps: 32
```

또는:
```bash
# 환경 변수 설정
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

#### 오류: No MIDI files found

**해결책**:
```bash
# MIDI 파일 경로 확인
ls data/raw_midi/*.mid

# 경로가 다르면 config 수정
# configs/qlora_config.yaml
midi_dir: "your/actual/path/to/midi/files"
```

#### 오류: Model quality is poor

**해결책**:
1. 더 많은 MIDI 파일 추가 (최소 30개, 권장 50개 이상)
2. 더 많은 에폭 학습 (100 → 200)
3. Learning rate 조정
4. Temperature 파라미터 조정 (생성 시)

### 6.2 성능 비교

**원본 Magenta vs QLoRA**:

| 항목 | Magenta | QLoRA |
|------|---------|-------|
| GPU 메모리 | ~16GB | ~4GB |
| 배치 크기 | 4 | 4-16 |
| 학습 시간 | 6-8시간 | 2-3시간 |
| 파라미터 | 전체 | ~0.1% |
| 품질 | 높음 | 유사 |
| 유연성 | 낮음 | 높음 |

---

## 7. 고급 사용법

### 7.1 데이터 증강

MIDI 파일이 부족한 경우 데이터 증강을 사용할 수 있습니다:

```python
# scripts/augment_midi.py (별도 작성 필요)
# - Transpose (±5 semitones)
# - Tempo change (0.9x - 1.1x)
# - Velocity scaling
```

### 7.2 앙상블

여러 체크포인트를 사용해 앙상블 생성:

```bash
# 다른 체크포인트로 여러 번 생성 후 선택
python scripts/generate_qlora.py \
  --checkpoint models/finetuned/qlora_brad_mehldau/checkpoint-1000 \
  --num_samples 5

python scripts/generate_qlora.py \
  --checkpoint models/finetuned/qlora_brad_mehldau/checkpoint-2000 \
  --num_samples 5
```

### 7.3 조건부 생성 (Conditional Generation)

특정 패턴으로 시작하는 음악 생성 (추후 구현):

```python
# Primer MIDI를 사용한 생성
primer_midi = "path/to/primer.mid"
# 구현 필요
```

---

## 8. 평가 및 분석

### 8.1 정량적 평가

- **Perplexity**: 낮을수록 좋음
- **Loss**: Validation loss 확인
- **생성 품질**: 음표 수, 길이, 다양성

### 8.2 정성적 평가

- Brad Mehldau 스타일 특징 재현 여부
- 화성 진행의 자연스러움
- 리듬 패턴의 재즈적 특징
- 음악적 일관성

### 8.3 A/B 테스트

```bash
# 여러 temperature로 생성
for temp in 0.8 1.0 1.2; do
  python scripts/generate_qlora.py \
    --temperature $temp \
    --output output/test_temp_${temp}.mid
done

# 블라인드 테스트로 최적 temperature 선택
```

---

## 9. 다음 단계

### 9.1 개선 방향

- [ ] 더 큰 base model 사용 (gpt2-medium, gpt2-large)
- [ ] 더 많은 Brad Mehldau MIDI 데이터 수집 (50-100개)
- [ ] 데이터 증강 기법 적용
- [ ] 조건부 생성 (특정 코드 진행 입력)
- [ ] 실시간 생성 인터페이스 개발

### 9.2 다른 스타일 적용

이 QLoRA 프레임워크는 다른 피아니스트 스타일에도 적용 가능:
- Bill Evans
- Keith Jarrett
- Herbie Hancock
- Chick Corea

---

## 10. 참고 자료

### 논문
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)

### 라이브러리
- [HuggingFace PEFT](https://github.com/huggingface/peft)
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
- [Transformers](https://github.com/huggingface/transformers)

### 음악 생성
- [Magenta](https://magenta.tensorflow.org/)
- [Music Transformer](https://arxiv.org/abs/1809.04281)

---

## 라이센스 및 윤리

⚠️ **중요**:
- Brad Mehldau MIDI 파일 사용 시 저작권 확인
- 학습 및 연구 목적으로만 사용
- 생성된 음악 배포 시 적절한 크레딧 표기
- 상업적 사용 전 법적 검토 필요

---

**작성일**: 2025-11-16
**버전**: 1.0
**프로젝트**: realTimeDeepLearningModelFineTuning - QLoRA Edition
