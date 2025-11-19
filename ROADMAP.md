# 찰리 파커 AI 프로젝트 로드맵 🎷

**목표**: 재즈 레전드의 즉흥 연주 스타일을 학습한 실시간 AI 만들기

**타겟**: 포트폴리오용 데모 + 실제 사용 가능한 잼 봇

---

## 전체 타임라인 및 비용

| Phase | 기간 | GPU 비용 | 난이도 | 결과물 |
|-------|------|----------|--------|--------|
| **Phase 0** | 1일 | 무료 | ⭐ | 환경 구축 완료 |
| **Phase 1** | 주말 (2일) | $10-20 | ⭐⭐ | 첫 번째 모델 |
| **Phase 2** | 1주 | $20-30 | ⭐⭐⭐ | 개선된 모델 |
| **Phase 3** | 2주 | $50-100 | ⭐⭐⭐ | 3-5개 뮤지션 AI |
| **Phase 4** | 1주 | $10-20 | ⭐⭐ | 실시간 잼 봇 |
| **Phase 5** | 1주 | 무료 | ⭐⭐ | 포트폴리오 완성 |
| **총계** | **6-8주** | **$90-180** | | **완성된 프로젝트** |

**총 비용**: 약 **12만원-24만원** (커피 30-60잔 값)

---

## Phase 0: 환경 설정 및 데이터 준비 🛠️

**목표**: 로컬 + 클라우드 개발 환경 구축, 데이터 수집

**기간**: 1일 (저녁 시간 활용 가능)

**비용**: 무료

### 0-1. 로컬 환경 설정

```bash
# 1. 이 레포지토리 클론 (이미 되어있음)
cd /path/to/realTimeDeepLearningModelFineTuning

# 2. Python 가상환경 생성
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 의존성 설치
pip install -r requirements.txt

# 4. 설치 확인
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import numpy; print('NumPy: OK')"
```

### 0-2. 데이터 수집

**Option 1: Lakh MIDI Dataset (추천)**
```bash
# 전체 데이터셋 다운로드 (178 GB - 시간 오래 걸림)
# wget http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz

# 또는 매칭된 서브셋만 (15 GB)
wget http://hog.ee.columbia.edu/craf/lmd/lmd_matched.tar.gz
tar -xzf lmd_matched.tar.gz

# Jazz만 필터링 (scripts/filter_jazz.py 사용)
python scripts/filter_jazz.py --input lmd_matched/ --output data/jazz_midi/
```

**Option 2: 특정 아티스트 MIDI 직접 수집**
- Charlie Parker MIDI 검색
- 합법적 출처: MuseScore, BitMidi, FreeMidi
- 최소 50-100개 파일 필요

**Option 3: PiJAMA Dataset (연구용)**
```bash
# PiJAMA jazz dataset
# https://github.com/spotify/jazz-performance-dataset
git clone https://github.com/spotify/PiJAMA-dataset.git
```

### 0-3. RunPod 계정 설정

1. **RunPod 가입**: https://runpod.io
2. **크레딧 충전**: $10 (13,000원) - 첫 실험용
3. **Pod 템플릿 선택**:
   - PyTorch 2.0+
   - CUDA 11.8+
   - RTX 3090 또는 4090

4. **SSH 키 등록** (로컬 ↔ RunPod 파일 전송용)

### 0-4. 데이터 검증

```bash
# 수집한 MIDI 파일 개수 확인
find data/jazz_midi/ -name "*.mid" | wc -l
# 목표: 최소 50개 이상

# 파일 검증 (corrupt 파일 제거)
python scripts/validate_midi.py --input data/jazz_midi/

# 통계 확인
python scripts/analyze_dataset.py --input data/jazz_midi/
# 출력: 총 시간, 평균 길이, 음역대, 템포 등
```

**체크리스트**:
- [ ] Python 환경 구축 완료
- [ ] MIDI 데이터 50개 이상 수집
- [ ] RunPod 계정 및 크레딧 준비
- [ ] 데이터 검증 완료

**예상 시간**: 3-4시간
**비용**: 무료

---

## Phase 1: 첫 번째 파인튜닝 (Charlie Parker) 🎺

**목표**: 실제로 동작하는 첫 번째 AI 모델 만들기

**기간**: 주말 1-2일

**비용**: $10-20 (13,000-26,000원)

### 1-1. 데이터 전처리

```bash
# Aria tokenizer로 MIDI 파일 변환
python scripts/preprocess_midi.py \
  --input data/jazz_midi/ \
  --output data/tokenized/ \
  --tokenizer aria \
  --chunk_size 512

# 학습/검증 데이터 분할 (80/20)
python scripts/split_dataset.py \
  --input data/tokenized/ \
  --train_ratio 0.8 \
  --seed 42
```

### 1-2. RunPod에서 학습 시작

```bash
# 1. RunPod Pod 시작 (RTX 3090)
# 2. 이 레포지토리 업로드
rsync -avz . runpod:/workspace/jazz-ai/

# 3. SSH로 접속
ssh runpod

# 4. 학습 시작
cd /workspace/jazz-ai
python scripts/train.py \
  --config configs/charlie_parker_small.yaml \
  --epochs 50 \
  --batch_size 16 \
  --checkpoint_dir checkpoints/charlie_parker_v1/

# 예상 시간: 6-8시간 (RTX 3090 기준)
# 비용: $0.34/hour × 8시간 = $2.72
```

### 1-3. 학습 모니터링

```bash
# 로그 확인
tail -f logs/training.log

# TensorBoard (선택)
tensorboard --logdir checkpoints/charlie_parker_v1/

# 중간 체크포인트 테스트
python scripts/generate.py \
  --checkpoint checkpoints/charlie_parker_v1/checkpoint_step_1000.pt \
  --prompt data/test_prompts/charlie_parker_intro.mid \
  --output samples/test_generation_1000.mid
```

### 1-4. 모델 다운로드 및 테스트

```bash
# RunPod에서 로컬로 다운로드
rsync -avz runpod:/workspace/jazz-ai/checkpoints/ ./checkpoints/

# 로컬에서 테스트
python scripts/generate.py \
  --checkpoint checkpoints/charlie_parker_v1/best_model.pt \
  --prompt data/test_prompts/blues_in_f.mid \
  --output samples/charlie_parker_v1_blues.mid \
  --max_tokens 512 \
  --temperature 0.9

# MIDI를 MP3로 변환 (청취용)
python scripts/midi_to_mp3.py \
  --input samples/charlie_parker_v1_blues.mid \
  --soundfont soundfonts/jazz_piano.sf2
```

**체크리스트**:
- [ ] 데이터 전처리 완료
- [ ] RunPod에서 학습 완료 (50 epochs)
- [ ] Best model 다운로드
- [ ] 샘플 생성 테스트 성공
- [ ] 결과를 귀로 들어보고 평가

**예상 시간**: 12-16시간 (대부분 학습 대기)
**실제 작업 시간**: 2-3시간
**비용**: $10-20

---

## Phase 2: 모델 평가 및 개선 📊

**목표**: 객관적 지표로 모델 평가, 성능 개선

**기간**: 1주 (퇴근 후 + 주말)

**비용**: $20-30

### 2-1. 정량 평가

```bash
# Perplexity 측정
python scripts/evaluate.py \
  --checkpoint checkpoints/charlie_parker_v1/best_model.pt \
  --test_data data/tokenized/test/ \
  --metrics perplexity

# 음악적 지표 측정
python scripts/evaluate.py \
  --checkpoint checkpoints/charlie_parker_v1/best_model.pt \
  --test_data data/tokenized/test/ \
  --metrics pctm pitch_class_kl note_density

# 레이턴시 벤치마크
python -m realjazz.benchmarking \
  --checkpoint checkpoints/charlie_parker_v1/best_model.pt \
  --n_runs 100
```

### 2-2. 정성 평가 (블라인드 테스트)

```bash
# 5개 샘플 생성 (다른 seed)
for i in {1..5}; do
  python scripts/generate.py \
    --checkpoint checkpoints/charlie_parker_v1/best_model.pt \
    --prompt data/test_prompts/blues_in_f.mid \
    --seed $i \
    --output samples/blind_test_${i}.mid
done

# 실제 Charlie Parker 곡 5개 추가
# → 총 10개를 섞어서 친구/지인에게 들려주기
# → "어떤 게 AI인지 맞춰보세요"
```

### 2-3. 모델 개선 (Iteration 2)

```bash
# 개선 포인트:
# 1. 더 많은 데이터 (100 → 200 MIDI files)
# 2. Longer training (50 → 100 epochs)
# 3. Corruption rate 조정 (0.5 → 0.7)

python scripts/train.py \
  --config configs/charlie_parker_v2.yaml \
  --epochs 100 \
  --batch_size 16 \
  --corruption_prob 0.7 \
  --checkpoint_dir checkpoints/charlie_parker_v2/

# RunPod에서 학습: 12-16시간
# 비용: $0.34 × 16 = $5.44
```

### 2-4. A/B 테스트

```bash
# v1 vs v2 비교
python scripts/compare_models.py \
  --model_a checkpoints/charlie_parker_v1/best_model.pt \
  --model_b checkpoints/charlie_parker_v2/best_model.pt \
  --test_prompts data/test_prompts/ \
  --output comparison_report.md

# 시각화
python scripts/visualize_comparison.py \
  --report comparison_report.md \
  --output figures/
```

**체크리스트**:
- [ ] 정량 평가 완료 (perplexity, PCTM 등)
- [ ] 블라인드 테스트 수행
- [ ] v2 모델 학습 완료
- [ ] v1 vs v2 비교 리포트 작성
- [ ] 개선 방향 도출

**예상 시간**: 20-25시간 (학습 대기 포함)
**실제 작업 시간**: 6-8시간
**비용**: $20-30

---

## Phase 3: 다른 뮤지션으로 확장 🎹

**목표**: 3-5명의 재즈 레전드 AI 만들기

**기간**: 2주

**비용**: $50-100

### 3-1. 타겟 뮤지션 선정

추천 리스트:
1. **Charlie Parker** (이미 완성) - Bebop
2. **Bill Evans** - Modal Jazz, 피아노
3. **John Coltrane** - Hard Bop, 색소폰
4. **Herbie Hancock** - Fusion, 키보드
5. **Miles Davis** - Cool Jazz, 트럼펫

각 뮤지션마다 **50-100 MIDI** 수집

### 3-2. 병렬 학습

```bash
# 각 뮤지션별로 학습 (RunPod에서 순차적으로)
for musician in bill_evans john_coltrane herbie_hancock; do
  python scripts/train.py \
    --config configs/${musician}.yaml \
    --data data/${musician}/ \
    --epochs 80 \
    --checkpoint_dir checkpoints/${musician}/
done

# 각 모델: 8-12시간
# 총 4개 × 10시간 = 40시간
# 비용: $0.34 × 40 = $13.6
```

### 3-3. 앙상블 모델 (선택)

```bash
# 여러 뮤지션 스타일을 섞을 수 있는 모델
python scripts/train_ensemble.py \
  --musicians charlie_parker bill_evans john_coltrane \
  --epochs 50 \
  --style_mixing_prob 0.3
```

### 3-4. 스타일 비교 데모

```bash
# 같은 프롬프트로 5명 스타일 생성
python scripts/generate_all_styles.py \
  --prompt data/test_prompts/autumn_leaves.mid \
  --models checkpoints/*/best_model.pt \
  --output samples/style_comparison/

# 시각화
python scripts/visualize_styles.py \
  --samples samples/style_comparison/ \
  --output figures/style_comparison.png
```

**체크리스트**:
- [ ] 5명 뮤지션 데이터 수집 완료
- [ ] 5개 모델 모두 학습 완료
- [ ] 스타일 비교 데모 생성
- [ ] 각 모델 블라인드 테스트

**예상 시간**: 60-80시간 (대부분 학습 대기)
**실제 작업 시간**: 15-20시간
**비용**: $50-100

---

## Phase 4: 실시간 잼 봇 완성 🎸

**목표**: MIDI 키보드 연결해서 실제로 같이 연주할 수 있는 봇

**기간**: 1주

**비용**: $10-20 (테스트용)

### 4-1. MIDI 입출력 설정

```bash
# MIDI 장비 확인
python scripts/list_midi_devices.py

# 가상 MIDI 포트 생성 (키보드 없을 때)
# macOS: IAC Driver
# Windows: loopMIDI
# Linux: virmidi
```

### 4-2. 실시간 추론 최적화

```bash
# 레이턴시 최적화
python scripts/optimize_for_realtime.py \
  --checkpoint checkpoints/charlie_parker_v2/best_model.pt \
  --output checkpoints/charlie_parker_v2_optimized.pt \
  --use_kv_cache \
  --quantize_int8

# 레이턴시 재측정
python -m realjazz.benchmarking \
  --checkpoint checkpoints/charlie_parker_v2_optimized.pt \
  --target_latency 100
```

### 4-3. 잼 봇 실행

```bash
# CLI 버전
python scripts/jam_demo.py \
  --checkpoint checkpoints/charlie_parker_v2_optimized.pt \
  --midi_input "Your MIDI Keyboard" \
  --midi_output "Virtual MIDI Out" \
  --style charlie_parker \
  --call_response  # 사용자 → AI → 사용자 → AI

# GUI 버전 (선택)
python scripts/jam_gui.py \
  --checkpoint checkpoints/charlie_parker_v2_optimized.pt
```

### 4-4. 녹음 및 데모 제작

```bash
# 잼 세션 녹음
python scripts/jam_demo.py \
  --checkpoint checkpoints/charlie_parker_v2_optimized.pt \
  --record \
  --output recordings/demo_session_1.mid

# MIDI → MP3 변환
python scripts/midi_to_mp3.py \
  --input recordings/demo_session_1.mid \
  --soundfont soundfonts/jazz_piano.sf2 \
  --output recordings/demo_session_1.mp3

# 영상 녹화 (OBS Studio 사용)
# → 키보드 치는 모습 + AI 응답 + 파형
```

**체크리스트**:
- [ ] MIDI 입출력 설정 완료
- [ ] 레이턴시 <100ms 달성
- [ ] 실시간 잼 봇 동작 확인
- [ ] 데모 녹음 3-5개 제작
- [ ] 영상 제작 (YouTube 업로드용)

**예상 시간**: 15-20시간
**실제 작업 시간**: 10-15시간
**비용**: $10-20

---

## Phase 5: 포트폴리오 및 배포 🚀

**목표**: GitHub, HuggingFace, 블로그에 프로젝트 공개

**기간**: 1주

**비용**: 무료

### 5-1. GitHub 정리

```bash
# README 작성
# 다음 섹션 포함:
# - 프로젝트 소개
# - 데모 영상/오디오
# - 기술 스택
# - 설치 및 사용법
# - 학습 결과 (그래프)
# - 라이선스

# 불필요한 파일 제거
echo "*.pyc" >> .gitignore
echo "checkpoints/" >> .gitignore
echo "data/" >> .gitignore

# 최종 커밋
git add .
git commit -m "Release v1.0: Charlie Parker AI + 4 jazz legends"
git tag v1.0
git push origin main --tags
```

### 5-2. HuggingFace에 모델 업로드

```bash
# HuggingFace Hub 설치
pip install huggingface-hub

# 로그인
huggingface-cli login

# 모델 업로드
python scripts/upload_to_huggingface.py \
  --checkpoint checkpoints/charlie_parker_v2/best_model.pt \
  --model_name "your-username/charlie-parker-jazz-ai" \
  --description "Real-time jazz improvisation AI trained on Charlie Parker MIDI"

# URL: https://huggingface.co/your-username/charlie-parker-jazz-ai
```

### 5-3. 블로그 포스팅

**Medium/브런치 포스트 작성**:

제목: "고졸 독학 개발자가 만든 찰리 파커 AI - 재즈 즉흥 연주 딥러닝 프로젝트"

목차:
1. 왜 이 프로젝트를 시작했나
2. 기술 스택 및 아키텍처
3. 데이터 수집 및 전처리
4. 학습 과정 및 어려움
5. 결과 및 데모
6. 배운 점 및 향후 계획
7. 오픈소스 공개

**포함할 것**:
- 데모 영상 (YouTube 임베드)
- 학습 그래프
- Before/After 비교
- GitHub 링크
- HuggingFace 링크

### 5-4. 데모 웹사이트 (선택)

```bash
# Streamlit으로 간단한 웹 데모
# scripts/streamlit_demo.py

streamlit run scripts/streamlit_demo.py

# Streamlit Cloud에 무료 배포
# → URL: https://your-jazz-ai.streamlit.app
```

### 5-5. 포트폴리오 최종 체크리스트

- [ ] GitHub README 완성도 높음
- [ ] HuggingFace 모델 5개 업로드
- [ ] YouTube 데모 영상 3개 이상
- [ ] 블로그 포스트 발행
- [ ] LinkedIn/Twitter 공유
- [ ] Streamlit 데모 (선택)
- [ ] 라이선스 명확히 (MIT or Apache 2.0)

**예상 시간**: 20-25시간
**비용**: 무료

---

## 최종 결과물 🎁

### 1. GitHub Repository
```
realTimeDeepLearningModelFineTuning/
├── README.md (킬러 README)
├── LICENSE
├── requirements.txt
├── ROADMAP.md (이 문서)
├── configs/ (5개 뮤지션 설정)
├── scripts/ (전체 파이프라인)
├── realjazz/ (코어 라이브러리)
├── docs/ (상세 문서)
├── samples/ (생성 샘플)
└── figures/ (그래프, 비교)
```

### 2. HuggingFace Models
- `your-username/charlie-parker-jazz-ai`
- `your-username/bill-evans-jazz-ai`
- `your-username/john-coltrane-jazz-ai`
- `your-username/herbie-hancock-jazz-ai`
- `your-username/miles-davis-jazz-ai`

### 3. YouTube 데모
- "Charlie Parker AI - 블루스 즉흥 연주"
- "5명의 재즈 레전드 AI 스타일 비교"
- "실시간 AI 잼 봇과 함께 연주하기"

### 4. 블로그 포스트
- Medium: 상세 기술 설명
- 브런치: 스토리텔링 중심
- LinkedIn: 프로젝트 요약

---

## 예상 총 비용 💰

| 항목 | 비용 |
|------|------|
| RunPod GPU (Phase 1-4) | $90-180 |
| MIDI 소프트웨어 (선택) | $0-50 |
| 도메인 (선택) | $0-10/year |
| **총계** | **$90-240** |

**약 12만원-31만원** (커피 30-80잔 값)

→ **포트폴리오 프로젝트로는 매우 저렴!**

---

## 예상 타임라인 📅

### 집중 모드 (풀타임 가능시)
- **총 기간**: 4주
- **하루 6-8시간 작업**

### 병행 모드 (백엔드 공부 + AI)
- **총 기간**: 6-8주
- **주중 저녁 2-3시간 + 주말 8-10시간**

### 느긋하게 모드
- **총 기간**: 3개월
- **주말만 활용**

---

## 성공 기준 ✅

### 최소 성공 (포트폴리오용)
- [ ] Charlie Parker AI 1개 완성
- [ ] 레이턴시 <100ms 달성
- [ ] 블라인드 테스트 50% 이상 통과
- [ ] GitHub + HuggingFace 공개
- [ ] 블로그 포스트 1개

### 중간 성공 (취업용)
- [ ] 3개 뮤지션 AI 완성
- [ ] 실시간 잼 봇 동작
- [ ] YouTube 데모 3개
- [ ] 기술 블로그 2-3개
- [ ] 커뮤니티 반응 (star, fork, 댓글)

### 완전 성공 (레퍼런스급)
- [ ] 5개 뮤지션 AI 완성
- [ ] Streamlit 웹 데모
- [ ] 논문 수준 평가 (perplexity, PCTM 등)
- [ ] 커뮤니티 기여 (Issue, PR)
- [ ] 컨퍼런스/밋업 발표

---

## 다음 단계

1. **Phase 0 시작**: `guides/PHASE0_SETUP.md` 읽기
2. **데이터 수집**: 주말에 MIDI 파일 모으기
3. **RunPod 가입**: $10 충전하고 대기
4. **커뮤니티 참여**:
   - r/MachineLearning
   - r/deeplearning
   - AI Korea Slack
   - Discord 서버들

---

**Let's build your Charlie Parker AI! 🎷**

**다음 문서**: `guides/PHASE0_SETUP.md`
