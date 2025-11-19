# Phase 0: 환경 설정 및 데이터 준비 🛠️

**목표**: 첫 번째 모델 학습을 위한 모든 준비 완료

**예상 시간**: 3-4시간
**비용**: 무료

---

## 체크리스트

- [ ] Python 환경 구축
- [ ] 의존성 설치
- [ ] MIDI 데이터 50개 이상 수집
- [ ] 데이터 검증 완료
- [ ] RunPod 계정 및 크레딧 준비

---

## Step 1: Python 환경 구축 (15분)

### 1-1. Python 버전 확인

```bash
python3 --version
# 필요: Python 3.9 이상 (권장: 3.10 or 3.11)
```

Python이 없거나 버전이 낮으면:
- **macOS**: `brew install python@3.11`
- **Windows**: https://python.org에서 다운로드
- **Linux**: `sudo apt install python3.11`

### 1-2. 가상환경 생성

```bash
# 프로젝트 디렉토리로 이동
cd /path/to/realTimeDeepLearningModelFineTuning

# 가상환경 생성
python3 -m venv venv

# 활성화
source venv/bin/activate  # macOS/Linux
# Windows: venv\Scripts\activate

# 확인 (venv가 앞에 표시되어야 함)
which python
# /path/to/realTimeDeepLearningModelFineTuning/venv/bin/python
```

### 1-3. pip 업그레이드

```bash
pip install --upgrade pip setuptools wheel
```

---

## Step 2: 의존성 설치 (10분)

### 2-1. PyTorch 설치

**CPU 버전** (로컬 개발용):
```bash
pip install torch torchvision torchaudio
```

**GPU 버전** (CUDA 있으면):
```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 2-2. 나머지 의존성 설치

```bash
pip install -r requirements.txt
```

### 2-3. 설치 확인

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import numpy; print('NumPy OK')"
python -c "import mido; print('Mido OK')"
```

모두 OK면 성공!

---

## Step 3: MIDI 데이터 수집 (1-2시간)

### Option 1: Lakh MIDI Dataset (추천) ⭐

**장점**: 방대한 양, 다양한 장르, 무료
**단점**: 용량 큼 (178 GB), jazz 필터링 필요

```bash
# 매칭된 서브셋만 다운로드 (15 GB)
cd data/
wget http://hog.ee.columbia.edu/craffel/lmd/lmd_matched.tar.gz

# 압축 해제 (10-20분 소요)
tar -xzf lmd_matched.tar.gz

# Jazz만 필터링
cd ..
python scripts/phase0/filter_jazz.py \
  --input data/lmd_matched/ \
  --output data/jazz_midi/ \
  --min_files 50

# 결과 확인
ls data/jazz_midi/*.mid | wc -l
# 50개 이상이면 OK
```

### Option 2: 직접 MIDI 파일 수집

**Charlie Parker 관련 MIDI 검색**:

1. **MuseScore** (https://musescore.com)
   - 검색: "Charlie Parker"
   - 필터: MIDI 다운로드 가능
   - 50-100개 다운로드

2. **FreeMIDI** (https://freemidi.org)
   - Jazz → Bebop 카테고리
   - Charlie Parker 곡 검색

3. **BitMidi** (https://bitmidi.com)
   - "Charlie Parker" 검색
   - 합법적 무료 MIDI

**수집 팁**:
- 최소 50개 필요 (권장 100-200개)
- 너무 짧은 곡은 제외 (<30초)
- 다양한 템포, 키 포함

### Option 3: PiJAMA Dataset

```bash
# Jazz Performance Analysis MIDI Archive
cd data/
git clone https://github.com/spotify/PiJAMA-dataset.git

# 또는 직접 다운로드
wget https://github.com/spotify/PiJAMA-dataset/archive/refs/heads/main.zip
unzip main.zip
```

---

## Step 4: 데이터 검증 (30분)

### 4-1. MIDI 파일 검증

```bash
# Corrupt 파일 체크 및 제거
python scripts/phase0/validate_midi.py \
  --input data/jazz_midi/ \
  --remove_corrupt

# 출력 예시:
# Checking 127 MIDI files...
# ✓ 124 files OK
# ✗ 3 files corrupt (removed)
# Final count: 124 files
```

### 4-2. 데이터셋 통계 분석

```bash
python scripts/phase0/analyze_dataset.py \
  --input data/jazz_midi/ \
  --output data/dataset_stats.json

# 출력 예시:
# Dataset Statistics
# ==================
# Total files: 124
# Total duration: 3h 42m
# Average length: 1m 48s
# Tempo range: 120-240 BPM
# Note range: C2-C7
# Polyphony: 1-8 voices
```

### 4-3. 샘플 청취

```bash
# 랜덤 5개 파일을 MP3로 변환해서 들어보기
python scripts/phase0/sample_to_mp3.py \
  --input data/jazz_midi/ \
  --n_samples 5 \
  --output data/samples/

# macOS에서 재생
open data/samples/sample_*.mp3

# 들어보고 quality 체크:
# - 너무 이상한 소리는 제외
# - 재즈 스타일이 맞는지 확인
```

---

## Step 5: RunPod 설정 (20분)

### 5-1. 계정 생성

1. https://runpod.io 접속
2. "Sign Up" 클릭
3. 이메일 인증

### 5-2. 크레딧 충전

1. Dashboard → Billing
2. "Add Credit" 클릭
3. $10 충전 (약 13,000원)
   - PayPal 또는 카드 결제
   - 첫 실험용으로 충분

### 5-3. SSH 키 설정 (파일 전송용)

```bash
# SSH 키가 없으면 생성
ssh-keygen -t ed25519 -C "your_email@example.com"
# Enter 3번 (기본값 사용)

# Public key 복사
cat ~/.ssh/id_ed25519.pub
# ssh-ed25519 AAAA...  (이 내용 복사)

# RunPod에 등록:
# RunPod Dashboard → Settings → SSH Keys
# "Add SSH Key" → 붙여넣기
```

### 5-4. 첫 Pod 생성 연습 (지금은 바로 삭제)

1. Dashboard → Pods → "Deploy"
2. GPU 선택: **RTX 3090** ($0.34/hour)
3. Template: **RunPod PyTorch 2.0**
4. "Deploy" 클릭
5. Pod 시작되면 (1-2분)
6. **바로 "Terminate"** (연습용이므로)

→ 실제 학습은 Phase 1에서 시작

---

## Step 6: 최종 체크 (5분)

### 환경 체크리스트

```bash
# 전체 시스템 체크
python scripts/phase0/check_setup.py

# 출력 예시:
# ✅ Python 3.11.5
# ✅ PyTorch 2.1.0
# ✅ CUDA available: False (CPU mode OK for development)
# ✅ MIDI library (mido) OK
# ✅ Data directory: 124 MIDI files (3h 42m)
# ✅ RunPod account: Ready (check manually)
#
# 🎉 All checks passed! Ready for Phase 1.
```

### 데이터 체크리스트

- [ ] 최소 50개 MIDI 파일 (권장 100+)
- [ ] 총 길이 1시간 이상 (권장 3시간+)
- [ ] Corrupt 파일 제거됨
- [ ] 샘플 들어보고 quality OK
- [ ] `data/jazz_midi/` 에 정리됨

### RunPod 체크리스트

- [ ] 계정 생성 완료
- [ ] $10 크레딧 충전
- [ ] SSH 키 등록
- [ ] Pod 생성/삭제 연습 완료

---

## 문제 해결 (Troubleshooting)

### Q1: "torch not found" 에러

```bash
# PyTorch 재설치
pip uninstall torch
pip install torch
```

### Q2: MIDI 파일 다운로드가 너무 느림

- Lakh dataset 대신 직접 수집 (Option 2)
- 또는 PiJAMA dataset 사용 (Option 3)

### Q3: RunPod 크레딧 충전 실패

- PayPal 계정 없으면 만들기
- 또는 카드 직접 결제

### Q4: SSH 키 등록이 안 됨

```bash
# 키 형식 확인
cat ~/.ssh/id_ed25519.pub
# "ssh-ed25519" 로 시작해야 함

# 안 되면 RSA 키로 재생성
ssh-keygen -t rsa -b 4096
```

---

## 다음 단계

✅ Phase 0 완료!

**다음**: `guides/PHASE1_TRAINING.md`

→ 첫 번째 Charlie Parker AI 모델 학습 시작!

---

## 예상 시간 요약

| Task | 시간 |
|------|------|
| Python 환경 | 15분 |
| 의존성 설치 | 10분 |
| MIDI 수집 | 1-2시간 |
| 데이터 검증 | 30분 |
| RunPod 설정 | 20분 |
| 최종 체크 | 5분 |
| **총계** | **2.5-3.5시간** |

**실제로는** 기다리는 시간(다운로드 등) 제외하면 **1-2시간 작업**

---

**준비 완료되면 Phase 1으로! 🚀**
