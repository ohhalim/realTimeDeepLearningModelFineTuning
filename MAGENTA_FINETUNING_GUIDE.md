# Magenta RT 모델 파인튜닝 가이드 - Brad Mehldau 스타일

## 개요
이 가이드는 Brad Mehldau의 MIDI 파일을 사용하여 Magenta의 real-time 음악 생성 모델을 파인튜닝하는 전체 프로세스를 설명합니다.

## 목차
1. [환경 설정](#1-환경-설정)
2. [데이터 수집 및 준비](#2-데이터-수집-및-준비)
3. [데이터 전처리](#3-데이터-전처리)
4. [모델 선택](#4-모델-선택)
5. [파인튜닝 실행](#5-파인튜닝-실행)
6. [모델 평가](#6-모델-평가)
7. [실시간 생성](#7-실시간-생성)

---

## 1. 환경 설정

### 1.1 필수 소프트웨어 설치

```bash
# Python 가상환경 생성
python3 -m venv magenta-env
source magenta-env/bin/activate

# Magenta 설치
pip install --upgrade pip
pip install magenta

# 추가 의존성
pip install tensorflow==2.11.0
pip install pretty-midi
pip install pyfluidsynth
```

### 1.2 디렉토리 구조

```
realTimeDeepLearningModelFineTuning/
├── data/
│   ├── raw_midi/           # Brad Mehldau 원본 MIDI 파일
│   ├── processed/          # 전처리된 데이터
│   └── tfrecord/           # TensorFlow Record 형식
├── models/
│   ├── pretrained/         # 사전 학습된 모델
│   └── finetuned/          # 파인튜닝된 모델
├── scripts/
│   ├── preprocess.py       # 전처리 스크립트
│   ├── train.py            # 학습 스크립트
│   └── generate.py         # 생성 스크립트
├── notebooks/              # Jupyter 노트북
└── output/                 # 생성된 MIDI 파일
```

---

## 2. 데이터 수집 및 준비

### 2.1 Brad Mehldau MIDI 파일 수집

**필요한 파일 수**: 약 30개

**추천 곡 목록**:
- Anything Goes
- Blackbird
- Exit Music (For a Film)
- Paranoid Android
- River Man
- Ron's Place
- Song-Song
- Unrequited
- Young and Foolish
- 기타 Brad Mehldau 트리오/솔로 곡들

### 2.2 MIDI 파일 저장

```bash
# MIDI 파일을 data/raw_midi/ 폴더에 저장
mkdir -p data/raw_midi
# 30개의 .mid 또는 .midi 파일을 이 폴더에 복사
```

### 2.3 데이터 품질 확인 사항

- ✅ MIDI 파일이 손상되지 않았는지 확인
- ✅ 피아노 트랙이 포함되어 있는지 확인
- ✅ 적절한 템포 정보가 있는지 확인
- ✅ 최소 1분 이상의 길이

---

## 3. 데이터 전처리

### 3.1 MIDI 검증 및 정제

**스크립트**: `scripts/preprocess.py`

```python
import os
import pretty_midi
from magenta.music import midi_io
from magenta.music import sequences_lib

def validate_midi_files(midi_dir):
    """MIDI 파일의 유효성 검사"""
    valid_files = []

    for filename in os.listdir(midi_dir):
        if filename.endswith(('.mid', '.midi')):
            filepath = os.path.join(midi_dir, filename)
            try:
                midi_data = pretty_midi.PrettyMIDI(filepath)
                # 최소 길이 확인 (60초)
                if midi_data.get_end_time() >= 60:
                    valid_files.append(filepath)
                    print(f"✓ {filename}: {midi_data.get_end_time():.1f}초")
                else:
                    print(f"✗ {filename}: 너무 짧음 ({midi_data.get_end_time():.1f}초)")
            except Exception as e:
                print(f"✗ {filename}: 오류 - {e}")

    print(f"\n총 {len(valid_files)}개의 유효한 파일")
    return valid_files
```

### 3.2 NoteSequence 변환

```python
def convert_to_notesequence(midi_files, output_dir):
    """MIDI를 Magenta NoteSequence로 변환"""
    os.makedirs(output_dir, exist_ok=True)

    for midi_file in midi_files:
        try:
            # MIDI를 NoteSequence로 변환
            ns = midi_io.midi_file_to_note_sequence(midi_file)

            # 양자화 (16분음표 기준)
            qns = sequences_lib.quantize_note_sequence(ns, steps_per_quarter=4)

            # 저장
            basename = os.path.basename(midi_file).replace('.mid', '')
            output_file = os.path.join(output_dir, f"{basename}.tfrecord")

            # TFRecord로 저장
            # (실제 구현은 모델에 따라 다름)

            print(f"✓ 변환 완료: {basename}")
        except Exception as e:
            print(f"✗ 변환 실패 {midi_file}: {e}")
```

### 3.3 데이터 분할

```python
# 학습/검증/테스트 분할 (80/10/10)
# 30개 파일의 경우: 24개 학습, 3개 검증, 3개 테스트
```

---

## 4. 모델 선택

### 4.1 추천 모델: **Music Transformer**

**이유**:
- 긴 시퀀스의 음악적 구조 학습에 뛰어남
- Brad Mehldau의 복잡한 화성과 즉흥 연주 패턴 학습 가능
- Real-time 생성 지원
- Pre-trained checkpoint 제공

**대안**:
- **Performance RNN**: 더 간단하지만 표현력이 낮음
- **Piano Genie**: Real-time 인터랙티브에 특화

### 4.2 사전 학습 모델 다운로드

```bash
# Music Transformer 체크포인트 다운로드
mkdir -p models/pretrained
cd models/pretrained

# Magenta의 공식 체크포인트 다운로드
# (URL은 Magenta GitHub에서 확인)
wget https://storage.googleapis.com/magentadata/models/music_transformer/checkpoints/unconditional_model_16.ckpt
```

---

## 5. 파인튜닝 실행

### 5.1 학습 설정

**스크립트**: `scripts/train.py`

```python
# 주요 하이퍼파라미터
BATCH_SIZE = 4              # GPU 메모리에 따라 조정
LEARNING_RATE = 0.0001      # 파인튜닝이므로 낮게 설정
NUM_EPOCHS = 50             # 데이터가 적으므로 여러 에폭
SEQUENCE_LENGTH = 2048      # 긴 시퀀스 (약 2분)
CHECKPOINT_INTERVAL = 5     # 5 에폭마다 저장
```

### 5.2 학습 명령어

```bash
# Music Transformer 파인튜닝
magenta_music_transformer_train \
  --config='unconditional' \
  --data_dir=data/tfrecord/train \
  --output_dir=models/finetuned/brad_mehldau \
  --restore_checkpoint=models/pretrained/unconditional_model_16.ckpt \
  --learning_rate=0.0001 \
  --batch_size=4 \
  --num_train_steps=10000 \
  --steps_per_checkpoint=500
```

### 5.3 학습 모니터링

```bash
# TensorBoard로 학습 과정 모니터링
tensorboard --logdir=models/finetuned/brad_mehldau
```

**확인 사항**:
- Loss가 점진적으로 감소하는지
- Overfitting 징후 (validation loss 증가)
- 학습 시간 (30개 파일: 약 2-6시간)

---

## 6. 모델 평가

### 6.1 정량적 평가

```python
# 검증 데이터에 대한 Loss 계산
# Perplexity 측정
# Negative Log-Likelihood (NLL)
```

### 6.2 정성적 평가

```bash
# 테스트 생성
python scripts/generate.py \
  --checkpoint=models/finetuned/brad_mehldau/model.ckpt-10000 \
  --output_dir=output/test_generation \
  --num_outputs=10 \
  --temperature=1.0
```

**평가 기준**:
- ✅ Brad Mehldau의 화성 진행 특징이 나타나는가?
- ✅ 재즈 특유의 리듬 패턴이 있는가?
- ✅ 음악적으로 일관성이 있는가?
- ✅ 창의적인 변주가 포함되는가?

### 6.3 A/B 테스트

```bash
# 원본 사전학습 모델과 파인튜닝 모델 비교
# 각각 5개씩 생성하여 블라인드 테스트
```

---

## 7. 실시간 생성

### 7.1 Real-time 생성 스크립트

**스크립트**: `scripts/realtime_generate.py`

```python
import magenta
from magenta.models.music_transformer import music_transformer

def generate_realtime(checkpoint_path, primer_midi=None):
    """실시간으로 음악 생성"""

    # 모델 로드
    config = music_transformer.MusicTransformerConfig()
    model = music_transformer.MusicTransformer(config)
    model.restore(checkpoint_path)

    # Primer 설정 (선택적)
    if primer_midi:
        primer_ns = midi_io.midi_file_to_note_sequence(primer_midi)
    else:
        primer_ns = None

    # 생성 (실시간)
    generated_ns = model.generate(
        primer=primer_ns,
        temperature=1.0,
        max_length=2048
    )

    return generated_ns
```

### 7.2 인터랙티브 세션

```bash
# MIDI 키보드 입력을 받아 즉시 생성
python scripts/realtime_generate.py \
  --checkpoint=models/finetuned/brad_mehldau/model.ckpt-10000 \
  --use_midi_input \
  --midi_device=0 \
  --temperature=1.0
```

### 7.3 생성 파라미터 조정

| 파라미터 | 값 | 효과 |
|---------|-----|------|
| Temperature | 0.5 | 안전하고 예측 가능한 생성 |
| Temperature | 1.0 | 균형잡힌 창의성 (기본값) |
| Temperature | 1.5 | 매우 창의적이지만 불안정할 수 있음 |
| Max Length | 512 | 짧은 프레이즈 (~30초) |
| Max Length | 2048 | 긴 즉흥연주 (~2분) |

---

## 8. 문제 해결

### 8.1 일반적인 문제

**문제**: GPU 메모리 부족
```bash
# 해결: 배치 크기 줄이기
BATCH_SIZE = 2  # 또는 1
```

**문제**: Overfitting (과적합)
```python
# 해결:
# 1. Dropout 증가
# 2. 데이터 증강 (transpose, time stretch)
# 3. Early stopping
```

**문제**: 생성 품질이 낮음
```python
# 해결:
# 1. 더 많은 에폭 학습
# 2. Learning rate 조정
# 3. Temperature 조정
# 4. 더 많은 MIDI 데이터 추가
```

### 8.2 데이터 증강

```python
# MIDI 트랜스포즈 (-5 ~ +5 반음)
# 템포 변화 (0.9x ~ 1.1x)
# 이를 통해 30개 파일을 150개 이상으로 증강 가능
```

---

## 9. 다음 단계

### 9.1 단기 목표
- [ ] 30개 Brad Mehldau MIDI 파일 수집
- [ ] 환경 설정 및 데이터 전처리
- [ ] 첫 번째 파인튜닝 실행
- [ ] 생성 결과 평가

### 9.2 중기 목표
- [ ] 데이터 증강으로 학습 데이터 확장
- [ ] 하이퍼파라미터 최적화
- [ ] 여러 체크포인트 비교
- [ ] 최고 성능 모델 선택

### 9.3 장기 목표
- [ ] Real-time MIDI 입력 통합
- [ ] 웹 인터페이스 개발
- [ ] 다른 재즈 피아니스트 스타일 추가
- [ ] 라이브 공연 시스템 구축

---

## 10. 참고 자료

### 공식 문서
- [Magenta GitHub](https://github.com/magenta/magenta)
- [Music Transformer Paper](https://arxiv.org/abs/1809.04281)
- [Magenta Blog](https://magenta.tensorflow.org/blog)

### Brad Mehldau 리소스
- [Brad Mehldau 공식 웹사이트](https://bradmehldau.com/)
- MIDI 데이터베이스 (MuseScore, Piano MIDI 등)

### 커뮤니티
- Magenta Google Group
- r/MachineLearning
- r/jazz

---

## 라이센스 및 윤리적 고려사항

⚠️ **중요**:
- Brad Mehldau의 MIDI 파일 사용 시 저작권 확인 필요
- 학습용 목적으로만 사용
- 생성된 음악 배포 시 적절한 크레딧 표기
- 상업적 사용 전 법적 검토 필요

---

**작성일**: 2025-11-16
**버전**: 1.0
**프로젝트**: realTimeDeepLearningModelFineTuning
