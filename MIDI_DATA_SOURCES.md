# Brad Mehldau 스타일 MIDI 데이터 수집 가이드

딥러닝 모델 학습을 위한 Brad Mehldau 재즈 피아노 MIDI 데이터 수집 방법 완벽 가이드

---

## 목차

1. [개요](#1-개요)
2. [Brad Mehldau 전용 MIDI](#2-brad-mehldau-전용-midi)
3. [일반 재즈 피아노 MIDI](#3-일반-재즈-피아노-midi)
4. [오디오 → MIDI 변환](#4-오디오--midi-변환)
5. [데이터 품질 검증](#5-데이터-품질-검증)
6. [라이센스 및 저작권](#6-라이센스-및-저작권)
7. [권장 워크플로우](#7-권장-워크플로우)

---

## 1. 개요

### 1.1 필요한 데이터 양

| 용도 | 최소 | 권장 | 이상적 |
|------|------|------|--------|
| 테스트 | 5-10개 | 15-20개 | - |
| 프로토타입 | 20-30개 | 40-50개 | - |
| 프로덕션 | 50개 | 100개 | 200개+ |

**파일 요구사항:**
- 형식: `.mid` 또는 `.midi`
- 최소 길이: 1분 이상
- 권장 길이: 2-5분
- 품질: 실제 연주 녹음 또는 고품질 전사

### 1.2 데이터 품질의 중요성

❌ **나쁜 데이터:**
- 자동 생성된 MIDI (질 낮음)
- 불완전한 전사
- 양손 분리 안됨
- 템포 정보 없음

✅ **좋은 데이터:**
- 수동 전사 (전문가)
- 표현력 있는 벨로시티
- 페달링 정보
- 정확한 타이밍

---

## 2. Brad Mehldau 전용 MIDI

### 2.1 MuseScore (무료 ⭐)

**웹사이트**: https://musescore.com

**장점:**
- 완전 무료
- MIDI 파일 직접 다운로드
- 커뮤니티 전사 (품질 다양)

**Brad Mehldau 검색 결과:**
- "Paris" - https://musescore.com/morusque/paris-brad-mehldau
- "Born to Trouble" - https://musescore.com/user/20720796/scores/6611439
- "Blackbird" - https://musescore.com/user/17971866/scores/14314474

**다운로드 방법:**
```bash
# 1. MuseScore 웹사이트 방문
# 2. "Brad Mehldau" 검색
# 3. 악보 페이지에서 "Download" → "MIDI" 선택
```

**예상 파일 수:** 10-20개

**품질:** ⭐⭐⭐⭐ (커뮤니티에 따라 다름)

### 2.2 Piano-Play (유료)

**웹사이트**: https://piano-play.com/bradmehldau.html

**장점:**
- 전문 재즈 피아니스트 전사
- 높은 품질
- MIDI, PDF, XML 등 다양한 형식

**가격:** 악보당 $5-15

**곡 목록:**
- Anything Goes
- Exit Music (For a Film)
- Paranoid Android
- 기타 솔로/트리오 곡들

**품질:** ⭐⭐⭐⭐⭐

### 2.3 MySheetMusicTranscriptions (유료)

**웹사이트**: https://www.mysheetmusictranscriptions.com/popular/brad-mehldau

**장점:**
- 프로페셔널 전사
- MIDI, MSCZ, XML 등 제공
- 높은 정확도

**가격:** 곡당 $10-20

**품질:** ⭐⭐⭐⭐⭐

### 2.4 Pianotify (구독)

**웹사이트**: https://pianotify.com

**Brad Mehldau 곡:**
- Radiohead - Paranoid Android (Brad Mehldau arrangement)

**가격:** 월 구독 ($9.99/월)

**품질:** ⭐⭐⭐⭐

### 2.5 Jazzpiano.co.nz (무료/유료)

**웹사이트**: https://www.jazzpiano.co.nz

**특징:**
- 20개 파일 (leadsheets + 전사)
- Brad Mehldau 퍼포먼스 전사
- 일부 무료, 일부 유료

**품질:** ⭐⭐⭐⭐

---

## 3. 일반 재즈 피아노 MIDI

### 3.1 Weimar Jazz Database (연구용 ⭐⭐⭐)

**웹사이트**: https://jazzomat.hfm-weimar.de

**특징:**
- 300개 재즈 솔로 전사
- MIDI, PDF, SQL 테이블
- 코드 진행 메타데이터
- 학술 연구용 고품질

**다운로드:**
```bash
# 웹사이트에서 데이터셋 다운로드
# ZIP 파일 압축 해제
# MIDI 파일 추출
```

**라이센스:** 연구/교육 목적

**품질:** ⭐⭐⭐⭐⭐

### 3.2 Kaggle - Jazz ML Ready MIDI

**웹사이트**: https://www.kaggle.com/datasets/saikayala/jazz-ml-ready-midi

**특징:**
- 800+ MIDI 파일
- 머신러닝용 전처리
- 노트 추출 완료
- 무료 다운로드

**다운로드:**
```bash
# Kaggle 계정 필요
kaggle datasets download -d saikayala/jazz-ml-ready-midi
unzip jazz-ml-ready-midi.zip
```

**품질:** ⭐⭐⭐⭐

### 3.3 MIDIWORLD.COM (무료)

**웹사이트**: https://www.midiworld.com/jazz.htm

**특징:**
- 개별 재즈 MIDI 파일
- 무료 다운로드
- 다양한 재즈 스타일

**품질:** ⭐⭐⭐ (다양함)

### 3.4 Doug McKenzie Jazz Piano (무료)

**웹사이트**: https://bushgrafts.com/midi/

**특징:**
- 실제 라이브 연주 MIDI
- 무료 다운로드
- 재즈 피아노 전문

**품질:** ⭐⭐⭐⭐

### 3.5 WAVS.com (무료/상업 이용 가능)

**웹사이트**: https://wavs.com/midi/genres/jazz

**특징:**
- 재즈 MIDI (피아노, 드럼, 베이스)
- 상업적 이용 가능
- 앙상블 편곡

**품질:** ⭐⭐⭐

### 3.6 FreeMIDI.org (무료)

**웹사이트**: https://freemidi.org/genre-jazz

**특징:**
- 재즈 장르 MIDI 컬렉션
- 무료 다운로드
- 기본적인 품질

**품질:** ⭐⭐⭐

### 3.7 Kunst der Fuge (무료/제한)

**웹사이트**: https://www.kunstderfuge.com/new/jazz.htm

**특징:**
- 재즈 MIDI 파일
- 무료: 5 파일/일
- 프리미엄: 무제한

**품질:** ⭐⭐⭐

---

## 4. 오디오 → MIDI 변환

실제 Brad Mehldau 녹음을 MIDI로 변환하는 방법

### 4.1 Music Demixer (브라우저, 유료) ⭐

**웹사이트**: https://freemusicdemixer.com

**특징:**
- AI 기반 피아노 전사
- 96% 노트 정확도
- 91% 페달 감지
- 브라우저에서 실행 (프라이버시)

**가격:** $9.99/월 (무제한)

**사용 방법:**
```bash
# 1. Brad Mehldau MP3/WAV 파일 준비
# 2. freemusicdemixer.com 접속
# 3. 파일 업로드
# 4. AI 전사 (2-5분)
# 5. MIDI 다운로드
```

**품질:** ⭐⭐⭐⭐⭐

### 4.2 La Touche Musicale (PianoConvert)

**웹사이트**: https://latouchemusicale.com/en/tools/audio-to-midi-converter/

**특징:**
- 피아노 전용 전사
- 좌우손 분리
- 템포/박자 감지

**가격:**
- 무료: 처음 30초
- 전체 곡: $2.99/곡

**품질:** ⭐⭐⭐⭐

### 4.3 Songscription AI (무료)

**웹사이트**: https://www.songscription.ai

**특징:**
- 악보 + MIDI 출력
- PDF, MIDI, MusicXML 내보내기
- 무료 (제한적)

**품질:** ⭐⭐⭐⭐

### 4.4 Samplab (무료)

**웹사이트**: https://samplab.com/audio-to-midi

**특징:**
- 브라우저 기반
- 계정 불필요
- 폴리포닉 지원

**가격:** 무료

**품질:** ⭐⭐⭐

### 4.5 AnthemScore (소프트웨어, 유료)

**웹사이트**: https://www.lunaverus.com

**특징:**
- 로컬 실행 (오프라인)
- 높은 정확도
- MIDI + MusicXML

**가격:** $29-$99 (일회 구매)

**품질:** ⭐⭐⭐⭐⭐

### 4.6 RipX (프로페셔널, 유료)

**특징:**
- AI DAW
- 강력한 오디오-MIDI 변환
- 로컬 실행

**가격:** $99-$299

**품질:** ⭐⭐⭐⭐⭐

---

## 5. 데이터 품질 검증

### 5.1 자동 검증 스크립트

```python
import pretty_midi
import os

def validate_midi(midi_path):
    """MIDI 파일 품질 검증"""
    try:
        midi = pretty_midi.PrettyMIDI(midi_path)

        # 1. 최소 길이 확인 (60초)
        if midi.get_end_time() < 60:
            print(f"❌ {midi_path}: 너무 짧음 ({midi.get_end_time():.1f}초)")
            return False

        # 2. 노트 개수 확인
        total_notes = sum(len(inst.notes) for inst in midi.instruments)
        if total_notes < 100:
            print(f"⚠️  {midi_path}: 노트 수 적음 ({total_notes}개)")

        # 3. 템포 확인
        tempo_changes = midi.get_tempo_changes()
        if len(tempo_changes[1]) == 0:
            print(f"⚠️  {midi_path}: 템포 정보 없음")

        # 4. 피아노 트랙 확인
        has_piano = False
        for inst in midi.instruments:
            if not inst.is_drum:
                has_piano = True
                break

        if not has_piano:
            print(f"❌ {midi_path}: 피아노 트랙 없음")
            return False

        print(f"✓ {midi_path}: OK ({midi.get_end_time():.1f}초, {total_notes}개 노트)")
        return True

    except Exception as e:
        print(f"❌ {midi_path}: 오류 - {e}")
        return False

# 사용 예시
midi_dir = "data/raw_midi"
for file in os.listdir(midi_dir):
    if file.endswith(('.mid', '.midi')):
        validate_midi(os.path.join(midi_dir, file))
```

### 5.2 수동 검증 체크리스트

**필수 확인 사항:**
- [ ] 파일이 깨지지 않고 열림
- [ ] 최소 1분 이상 재생
- [ ] 피아노 소리 포함
- [ ] 템포 정보 있음
- [ ] 노트 벨로시티 변화 있음

**선택 확인 사항:**
- [ ] 좌우손 분리
- [ ] 페달링 정보
- [ ] 표현적 타이밍 (스윙, 루바토)
- [ ] 코드 진행 명확

### 5.3 MuseScore로 시각 확인

```bash
# MuseScore 설치 (무료)
# https://musescore.org/

# MIDI 파일 열기
musescore file.mid

# 확인 사항:
# - 악보가 읽기 쉬운가?
# - 박자표/조표가 정확한가?
# - 연주가 자연스러운가?
```

---

## 6. 라이센스 및 저작권

### 6.1 저작권 주의사항

⚠️ **중요:**
- Brad Mehldau의 음악은 **저작권 보호** 대상
- 상업적 배포 전 **라이센스 필수**
- 연구/학습 목적은 **Fair Use** 가능 (국가별 다름)

### 6.2 안전한 사용 방법

**✅ 허용 가능:**
- 개인 학습 및 연구
- 비상업적 프로토타입
- 학술 논문 (인용 명시)
- 오픈소스 연구 프로젝트

**❌ 주의 필요:**
- 상업적 음악 생성 서비스
- 생성 음악 판매
- Brad Mehldau 이름 무단 사용
- 원곡 복제/배포

### 6.3 라이센스별 정리

| 소스 | 라이센스 | 상업 이용 | 크레딧 |
|------|---------|----------|--------|
| MuseScore | CC BY-NC-SA | ❌ | 필수 |
| Weimar Jazz DB | 연구/교육 | ❌ | 필수 |
| WAVS.com | 상업 가능 | ✅ | 권장 |
| 구매한 MIDI | 개인 사용 | ❌ | - |
| 직접 전사 | 본인 소유 | ⚠️ | - |

### 6.4 권장 표기

생성된 음악 사용 시:

```
Generated using a model trained on Brad Mehldau's style.
Not affiliated with or endorsed by Brad Mehldau.
For research/educational purposes only.
```

---

## 7. 권장 워크플로우

### 7.1 초보자 (무료로 시작)

**목표:** 30개 MIDI 파일

```bash
# Step 1: MuseScore (10-15개)
# - Brad Mehldau 검색
# - 무료 MIDI 다운로드

# Step 2: Weimar Jazz Database (10개)
# - 재즈 피아노 솔로
# - 고품질 전사

# Step 3: Doug McKenzie (5-10개)
# - 라이브 재즈 피아노
# - bushgrafts.com/midi

# Step 4: 샘플 생성기 (테스트용)
python scripts/generate_sample_midi.py --count 5
```

**예상 비용:** $0

**예상 시간:** 2-3시간

### 7.2 중급자 (소액 투자)

**목표:** 50개 MIDI 파일

```bash
# Step 1: MuseScore (15개)
# Step 2: Weimar Jazz Database (15개)
# Step 3: Piano-Play 구매 (10개)
#   - 인기곡 선택 ($50-100)
# Step 4: La Touche Musicale (10개)
#   - 오디오 → MIDI 변환
#   - $2.99/곡 × 10 = $30
```

**예상 비용:** $80-130

**예상 시간:** 5-8시간

### 7.3 고급자 (프로페셔널)

**목표:** 100+ MIDI 파일

```bash
# Step 1: MuseScore (20개)
# Step 2: Weimar Jazz Database (20개)
# Step 3: Piano-Play 대량 구매 (30개)
#   - 주요 곡 컬렉션 ($150-300)
# Step 4: Music Demixer 구독 (30개)
#   - $9.99/월
#   - Brad Mehldau 앨범 전사
#   - Spotify → MP3 → MIDI
```

**예상 비용:** $200-400

**예상 시간:** 15-25시간

### 7.4 연구자 (최고 품질)

**목표:** 200+ MIDI 파일

```bash
# Step 1: 모든 무료 소스 (50개)
# Step 2: 모든 유료 전사 (50개)
# Step 3: RipX/AnthemScore 구매 (100개)
#   - 로컬 오디오 전사
#   - Brad Mehldau 전 앨범
#   - $99-299 (일회 구매)
```

**예상 비용:** $500-1000

**예상 시간:** 40-60시간

---

## 8. 데이터 조직화

### 8.1 디렉토리 구조

```
data/
├── raw_midi/
│   ├── brad_mehldau/
│   │   ├── musescore/
│   │   ├── purchased/
│   │   └── converted/
│   ├── jazz_piano/
│   │   ├── weimar/
│   │   └── kaggle/
│   └── samples/
├── processed/
│   ├── train/
│   ├── val/
│   └── test/
└── metadata/
    ├── sources.csv
    └── quality.csv
```

### 8.2 메타데이터 관리

**sources.csv:**
```csv
filename,source,license,quality,duration,notes
paris.mid,MuseScore,CC BY-NC-SA,4,180,Good transcription
anything_goes.mid,Piano-Play,Personal,5,210,Professional
weimar_001.mid,Weimar,Research,5,195,High quality
```

### 8.3 버전 관리

```bash
# Git LFS for large files
git lfs track "*.mid"
git lfs track "*.midi"

# .gitignore
data/raw_midi/*.mp3
data/raw_midi/*.wav
```

---

## 9. 법적 면책 조항

이 가이드는 **교육 및 연구 목적**으로만 제공됩니다.

**주의사항:**
- 모든 저작권법 준수 필요
- 상업적 사용 전 법률 자문 권장
- 라이센스 조건 반드시 확인
- 생성된 음악의 법적 책임은 사용자에게 있음

**안전한 접근:**
1. 무료 라이센스부터 시작
2. 상업 계획 시 변호사 상담
3. Fair Use 원칙 이해
4. 출처 명시 습관화

---

## 10. 추가 리소스

### 10.1 참고 자료

- **Copyright Guide**: https://www.copyright.gov/fair-use/
- **Creative Commons**: https://creativecommons.org/
- **Music AI Ethics**: https://www.musicaiethics.org/

### 10.2 커뮤니티

- **r/WeAreTheMusicMakers**: Reddit 음악 제작 커뮤니티
- **Magenta Discuss**: Google Group 토론
- **GitHub Issues**: 질문 및 토론

### 10.3 도구

- **MuseScore**: 무료 악보 편집기
- **pretty_midi**: Python MIDI 라이브러리
- **music21**: 음악 분석 라이브러리

---

## 요약

### 빠른 시작 (무료)

1. MuseScore에서 Brad Mehldau 검색 → 10개 다운로드
2. Weimar Jazz Database → 10개 다운로드
3. 샘플 생성기 → 10개 생성
4. **총 30개 파일로 학습 시작!**

### 최적 조합 (유료)

1. MuseScore (무료) → 15개
2. Weimar (무료) → 15개
3. Piano-Play ($100) → 10개
4. Music Demixer ($10) → 10개
5. **총 50개 고품질 파일**

**성공을 빕니다! 🎹✨**

---

**작성일**: 2025-11-16
**버전**: 1.0
**프로젝트**: realTimeDeepLearningModelFineTuning
