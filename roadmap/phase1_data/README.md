# Phase 1: 데이터 수집 (1주, 무료-$20)

**목표**: Brad Mehldau MIDI 20-50개 수집 및 전처리
**예상 시간**: 5-10시간 (1주일 분산)
**비용**: $0-20

---

## ✅ 체크리스트

- [ ] MIDI 파일 20개 수집
- [ ] 품질 검증 (재생해서 확인)
- [ ] 전처리 스크립트 작성
- [ ] Tokenization 구현
- [ ] Train/Val/Test 분할 (70/15/15)
- [ ] GitHub에 업로드 (코드만, MIDI는 .gitignore)

---

## 1. 데이터 소스 (우선순위 순)

### Option 1: 무료 - 직접 채보 (시간 많이 걸림)
```
장점: 완전 무료, 저작권 문제 없음
단점: 1곡당 2-4시간 소요, 품질 보장 어려움
추천: 시간 많고 예산 없으면
```

### Option 2: 유료 - 악보 구매 후 MIDI 변환 (**추천**)
```
사이트: Musicnotes.com, Sheet Music Plus
가격: $3-7/곡
장점: 전문가 채보, 높은 품질
단점: 비용 발생

추천 곡 (Brad Mehldau):
1. "Blackbird" (Beatles cover)
2. "Exit Music" (Radiohead cover)
3. "River Man" (Nick Drake cover)
4. "Paranoid Android" (Radiohead cover)
5. "All The Things You Are" (standards)
```

### Option 3: 무료 - 커뮤니티 요청
```
Reddit: r/piano, r/jazz, r/WeAreTheMusicMakers
Discord: Jazz piano servers
Forum: PianoGroove.com

템플릿 메시지:
"Hi! I'm working on a jazz AI project and looking for Brad Mehldau MIDI
transcriptions for educational purposes. Anyone willing to share?"
```

### Option 4: 무료 - YouTube → MIDI 변환 (품질 낮음)
```
도구: AnthemScore, Basic Pitch (Spotify)
장점: 무료, 많은 영상
단점: 품질 낮음, 에러 많음, 수정 필요

추천: 테스트용으로만 사용
```

---

## 2. 데이터 수집 전략

### 최소 구성 (무료)
```
Classical piano: 10개 (MAESTRO dataset 사용)
Brad Mehldau: 5-10개 (직접 채보 or 커뮤니티)
─────────────────
총: 15-20개
```

### 권장 구성 ($20)
```
Classical piano: 20개 (MAESTRO)
Brad Mehldau: 10개 (악보 구매)
Other jazz pianists: 5개 (Bill Evans, etc.)
─────────────────
총: 35개
```

### 이상적 구성 ($50)
```
Classical piano: 50개 (MAESTRO + ASAP)
Brad Mehldau: 25개 (악보 구매 + 전문가 채보)
Other jazz pianists: 10개
─────────────────
총: 85개
```

---

## 3. 데이터 수집 실행

### Step 1: Classical Piano 데이터 (무료)

```python
# download_classical.py
import urllib.request
import zipfile
import os

# MAESTRO dataset (Google Magenta)
MAESTRO_URL = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip"

def download_maestro(output_dir="data/raw/classical"):
    os.makedirs(output_dir, exist_ok=True)

    print("Downloading MAESTRO dataset...")
    zip_path = f"{output_dir}/maestro.zip"

    urllib.request.urlretrieve(MAESTRO_URL, zip_path)
    print(f"Downloaded to {zip_path}")

    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)

    print(f"✅ MAESTRO dataset ready in {output_dir}")

    # 선별 (우리는 20개만 필요)
    import glob
    midi_files = glob.glob(f"{output_dir}/**/*.mid*", recursive=True)
    print(f"Total MIDI files: {len(midi_files)}")
    print(f"Select 20 for training")

if __name__ == "__main__":
    download_maestro()
```

실행:
```bash
python download_classical.py
```

### Step 2: Brad Mehldau MIDI 수집

#### 방법 A: 악보 구매 ($20)

```
1. Musicnotes.com 접속
2. "Brad Mehldau" 검색
3. 10곡 선택 (각 $2-3)
4. PDF 다운로드
5. MuseScore로 열기
6. File → Export → MIDI
```

#### 방법 B: 직접 채보 (무료, 시간 많이 걸림)

```
1. YouTube에서 Brad Mehldau 연주 찾기
2. MuseScore 열기
3. 영상 보며 직접 입력
4. 체크: 화성, 리듬, 다이나믹스
5. Export → MIDI

추천 도구:
- MuseScore (무료)
- Transcribe! ($39, 속도 조절 기능)
```

#### 방법 C: 커뮤니티 요청 (무료)

Reddit 템플릿:
```
Title: [Request] Brad Mehldau MIDI transcriptions for AI project

Body:
Hi r/piano,

I'm building an AI model that learns to improvise jazz piano in Brad Mehldau's
style as a personal learning project. I'm looking for MIDI transcriptions
(not audio) for educational/research purposes.

If anyone has transcriptions they're willing to share, I'd really appreciate it!

Songs I'm looking for:
- Blackbird
- Exit Music
- River Man
- Anything from "The Art of the Trio" series

Thanks!
```

### Step 3: 데이터 검증

```python
# validate_midi.py
import pretty_midi
import glob
import os

def validate_midi_file(midi_path):
    """MIDI 파일 품질 검증"""
    try:
        midi = pretty_midi.PrettyMIDI(midi_path)

        # 기본 체크
        duration = midi.get_end_time()
        num_notes = sum(len(inst.notes) for inst in midi.instruments)

        # 품질 기준
        if duration < 10:
            return False, "Too short (< 10 seconds)"
        if duration > 600:
            return False, "Too long (> 10 minutes)"
        if num_notes < 50:
            return False, "Too few notes (< 50)"

        # Piano 악기 확인
        has_piano = any(inst.program in range(0, 8) for inst in midi.instruments)
        if not has_piano:
            return False, "No piano instrument found"

        return True, f"✅ OK (duration={duration:.1f}s, notes={num_notes})"

    except Exception as e:
        return False, f"Error: {str(e)}"

def validate_all(data_dir="data/raw"):
    """모든 MIDI 파일 검증"""
    midi_files = glob.glob(f"{data_dir}/**/*.mid*", recursive=True)

    print(f"Found {len(midi_files)} MIDI files")
    print("="*60)

    valid_files = []
    invalid_files = []

    for midi_path in midi_files:
        is_valid, message = validate_midi_file(midi_path)

        filename = os.path.basename(midi_path)
        print(f"{filename[:40]:<40} {message}")

        if is_valid:
            valid_files.append(midi_path)
        else:
            invalid_files.append((midi_path, message))

    print("="*60)
    print(f"✅ Valid: {len(valid_files)}")
    print(f"❌ Invalid: {len(invalid_files)}")

    return valid_files, invalid_files

if __name__ == "__main__":
    valid, invalid = validate_all()

    # 유효한 파일 목록 저장
    with open("data/valid_files.txt", "w") as f:
        for path in valid:
            f.write(f"{path}\n")

    print(f"\n✅ Valid file list saved to data/valid_files.txt")
```

실행:
```bash
python validate_midi.py
```

---

## 4. 데이터 전처리

### Tokenization 구현

```python
# preprocessing/tokenizer.py
import pretty_midi
import numpy as np
from typing import List, Tuple
import torch

class SimpleMIDITokenizer:
    """Simplified MIDI tokenizer for piano

    Token structure:
    - 0-87: NOTE_ON (88 pitches)
    - 88-187: TIME_SHIFT (100 bins, 0-1000ms in 10ms increments)
    - 188-219: VELOCITY (32 bins)
    - 220: PAD
    - 221: START
    - 222: END
    """
    def __init__(self):
        self.NOTE_ON_OFFSET = 0
        self.TIME_SHIFT_OFFSET = 88
        self.VELOCITY_OFFSET = 188
        self.PAD_TOKEN = 220
        self.START_TOKEN = 221
        self.END_TOKEN = 222
        self.vocab_size = 223

    def encode(self, midi_path: str, max_length: int = 512) -> List[int]:
        """MIDI 파일을 토큰 시퀀스로 변환"""
        midi = pretty_midi.PrettyMIDI(midi_path)

        # Piano 트랙만 추출
        piano_notes = []
        for inst in midi.instruments:
            if not inst.is_drum and inst.program in range(0, 8):
                piano_notes.extend(inst.notes)

        # 시간순 정렬
        piano_notes.sort(key=lambda x: x.start)

        tokens = [self.START_TOKEN]
        prev_time = 0.0

        for note in piano_notes:
            # Time shift
            time_delta = note.start - prev_time
            time_bins = min(int(time_delta * 100), 99)  # 10ms 단위
            tokens.append(self.TIME_SHIFT_OFFSET + time_bins)

            # Note pitch
            pitch = note.pitch - 21  # MIDI 21 = A0 (lowest piano key)
            if 0 <= pitch < 88:
                tokens.append(self.NOTE_ON_OFFSET + pitch)

                # Velocity
                vel_bin = note.velocity // 4  # 128 → 32 bins
                tokens.append(self.VELOCITY_OFFSET + vel_bin)

                prev_time = note.start

        tokens.append(self.END_TOKEN)

        # Pad or truncate
        if len(tokens) < max_length:
            tokens += [self.PAD_TOKEN] * (max_length - len(tokens))
        else:
            tokens = tokens[:max_length-1] + [self.END_TOKEN]

        return tokens

    def decode(self, tokens: List[int]) -> List[Tuple[int, float, float, int]]:
        """토큰을 (pitch, onset, duration, velocity) 리스트로 변환"""
        notes = []
        current_time = 0.0
        i = 0

        while i < len(tokens):
            token = tokens[i]

            if token == self.PAD_TOKEN or token == self.END_TOKEN:
                break

            if token == self.START_TOKEN:
                i += 1
                continue

            # Time shift
            if self.TIME_SHIFT_OFFSET <= token < self.VELOCITY_OFFSET:
                time_delta = (token - self.TIME_SHIFT_OFFSET) / 100.0
                current_time += time_delta
                i += 1

                if i >= len(tokens):
                    break

                # Note pitch
                pitch_token = tokens[i]
                if self.NOTE_ON_OFFSET <= pitch_token < self.TIME_SHIFT_OFFSET:
                    pitch = pitch_token - self.NOTE_ON_OFFSET + 21
                    i += 1

                    if i >= len(tokens):
                        break

                    # Velocity
                    vel_token = tokens[i]
                    if self.VELOCITY_OFFSET <= vel_token < self.PAD_TOKEN:
                        velocity = (vel_token - self.VELOCITY_OFFSET) * 4
                        i += 1

                        # Default duration
                        duration = 0.5

                        notes.append((pitch, current_time, duration, velocity))
                    else:
                        i += 1
                else:
                    i += 1
            else:
                i += 1

        return notes

if __name__ == "__main__":
    # 테스트
    tokenizer = SimpleMIDITokenizer()

    # 예시 MIDI 파일로 테스트
    tokens = tokenizer.encode("data/raw/test.mid")
    print(f"Encoded: {len(tokens)} tokens")
    print(f"First 10 tokens: {tokens[:10]}")

    # Decode
    notes = tokenizer.decode(tokens)
    print(f"Decoded: {len(notes)} notes")
    print(f"First note: {notes[0]}")
```

### 데이터셋 분할

```python
# preprocessing/split_data.py
import os
import glob
import random
import shutil

def split_dataset(data_dir="data/raw", output_dir="data/splits", split_ratio=(0.7, 0.15, 0.15)):
    """데이터를 train/val/test로 분할"""
    # 유효한 파일 목록 읽기
    with open("data/valid_files.txt", "r") as f:
        valid_files = [line.strip() for line in f.readlines()]

    # 셔플
    random.shuffle(valid_files)

    # 분할
    train_ratio, val_ratio, test_ratio = split_ratio
    n_total = len(valid_files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_files = valid_files[:n_train]
    val_files = valid_files[n_train:n_train+n_val]
    test_files = valid_files[n_train+n_val:]

    # 디렉토리 생성
    os.makedirs(f"{output_dir}/train", exist_ok=True)
    os.makedirs(f"{output_dir}/val", exist_ok=True)
    os.makedirs(f"{output_dir}/test", exist_ok=True)

    # 파일 복사
    for i, file_path in enumerate(train_files):
        filename = os.path.basename(file_path)
        shutil.copy(file_path, f"{output_dir}/train/{i:04d}_{filename}")

    for i, file_path in enumerate(val_files):
        filename = os.path.basename(file_path)
        shutil.copy(file_path, f"{output_dir}/val/{i:04d}_{filename}")

    for i, file_path in enumerate(test_files):
        filename = os.path.basename(file_path)
        shutil.copy(file_path, f"{output_dir}/test/{i:04d}_{filename}")

    print(f"✅ Dataset split complete:")
    print(f"   Train: {len(train_files)} files")
    print(f"   Val: {len(val_files)} files")
    print(f"   Test: {len(test_files)} files")

if __name__ == "__main__":
    split_dataset()
```

실행:
```bash
python preprocessing/split_data.py
```

---

## 5. 최종 체크리스트

### 데이터 수집
- [ ] Classical piano: 10-50개
- [ ] Brad Mehldau: 5-25개
- [ ] 총 20개 이상 확보

### 품질 검증
- [ ] 모든 MIDI 재생 확인
- [ ] validate_midi.py 실행
- [ ] valid_files.txt 생성

### 전처리
- [ ] Tokenizer 구현 완료
- [ ] Tokenization 테스트 성공
- [ ] Train/Val/Test 분할 완료

### GitHub
- [ ] 전처리 코드 커밋
- [ ] .gitignore에 MIDI 추가
- [ ] README 업데이트

---

## 🎯 Phase 1 완료!

데이터 준비가 완료되었습니다!

**다음 단계**: `../phase2_quickstart/README.md`

첫 AI 재즈 샘플을 생성하세요! 🚀
