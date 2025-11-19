# 빠른 시작 가이드 🚀

**목표**: 2일 안에 첫 번째 찰리 파커 AI 만들기

---

## 이 프로젝트는...

재즈 레전드 **찰리 파커** 스타일을 학습한 실시간 즉흥 연주 AI입니다.

- ✅ **실시간** 반응 (<100ms 레이턴시)
- ✅ **재즈 스타일** 학습
- ✅ **MIDI 키보드**와 상호작용
- ✅ **저렴한 비용** (총 $90-180)
- ✅ **고졸도 가능** (포트폴리오용)

---

## 최소 요구사항

- **Python 3.9+**
- **인터넷 연결** (클라우드 GPU 사용)
- **$10-20 예산** (첫 모델용)
- **주말 시간** (2-3일)

---

## 30초 요약

```bash
# 1. 환경 설정
git clone <this-repo>
cd realTimeDeepLearningModelFineTuning
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. 데이터 수집
python scripts/phase0/filter_jazz.py \
  --input data/lmd_matched/ \
  --output data/jazz_midi/ \
  --min_files 50

# 3. 데이터 전처리
python scripts/preprocess_midi.py \
  --input data/jazz_midi/ \
  --output data/tokenized/

# 4. RunPod에서 학습 (8-10시간, $3-5)
python scripts/train.py \
  --config configs/charlie_parker_v1.yaml

# 5. 샘플 생성
python scripts/generate.py \
  --checkpoint checkpoints/charlie_parker_v1/best_model.pt \
  --prompt data/test_prompts/blues.mid \
  --output samples/my_first_ai_jazz.mid

# 완성! 🎉
```

---

## 단계별 가이드

### Phase 0: 환경 설정 (1일)
→ `guides/PHASE0_SETUP.md`

- Python 설치
- MIDI 데이터 50개 수집
- RunPod 계정 만들기
- **비용**: 무료
- **시간**: 3-4시간

### Phase 1: 첫 학습 (1-2일)
→ `guides/PHASE1_TRAINING.md`

- 데이터 전처리
- RunPod에서 학습
- 샘플 생성 및 테스트
- **비용**: $10-20
- **시간**: 12-16시간 (대부분 대기)

### Phase 2-5: 고급 기능
→ `ROADMAP.md`

- 모델 평가 및 개선
- 다른 뮤지션 (Bill Evans, Coltrane 등)
- 실시간 잼 봇
- 포트폴리오 완성

---

## 전체 로드맵

```
Week 1-2: Phase 0-1 (첫 모델 완성)
  ├─ 환경 설정
  ├─ 데이터 수집
  ├─ 첫 학습
  └─ ✅ Charlie Parker AI v1

Week 3-4: Phase 2-3 (개선 및 확장)
  ├─ 모델 평가
  ├─ v2 개선
  └─ ✅ 3-5개 뮤지션 AI

Week 5-6: Phase 4-5 (완성)
  ├─ 실시간 잼 봇
  ├─ 포트폴리오 정리
  └─ ✅ GitHub + HuggingFace 공개
```

---

## 예상 총 비용

| Phase | 비용 (USD) | 비용 (KRW) |
|-------|-----------|-----------|
| Phase 0 | $0 | 무료 |
| Phase 1 | $10-20 | 13,000-26,000원 |
| Phase 2 | $20-30 | 26,000-39,000원 |
| Phase 3 | $50-100 | 65,000-130,000원 |
| Phase 4 | $10-20 | 13,000-26,000원 |
| Phase 5 | $0 | 무료 |
| **총계** | **$90-180** | **117,000-234,000원** |

→ **커피 30-60잔 값으로 포트폴리오 프로젝트 완성!**

---

## 체크리스트

### 시작 전
- [ ] Python 3.9+ 설치됨
- [ ] Git 설치됨
- [ ] RunPod 계정 만들기 (https://runpod.io)
- [ ] $10 예산 준비

### Phase 0 완료
- [ ] 가상환경 생성
- [ ] 의존성 설치
- [ ] MIDI 데이터 50개 이상
- [ ] `python scripts/phase0/check_setup.py` 통과

### Phase 1 완료
- [ ] 데이터 전처리 완료
- [ ] RunPod에서 학습 완료
- [ ] Best model 다운로드
- [ ] 샘플 생성 테스트
- [ ] 레이턴시 <100ms 확인

---

## 문제 발생시

1. **Python 관련**:
   ```bash
   # 가상환경 재생성
   rm -rf venv
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **데이터 부족**:
   - MuseScore에서 직접 다운로드
   - 최소 50개만 있어도 시작 가능

3. **RunPod 오류**:
   - Pod 재시작
   - 다른 GPU 타입 시도 (RTX 4090)
   - SSH 키 재등록

4. **학습 실패**:
   - Batch size 줄이기 (16 → 8)
   - Epoch 줄이기 (50 → 30)
   - 로그 확인 (`logs/training.log`)

---

## 다음 단계

### 1. 지금 바로 시작
```bash
# 레포 클론
git clone <this-repo>
cd realTimeDeepLearningModelFineTuning

# Phase 0 시작
cat guides/PHASE0_SETUP.md
```

### 2. 커뮤니티 참여
- **GitHub Issues**: 질문/버그 리포트
- **Discussions**: 아이디어 공유
- **Discord**: 실시간 대화 (링크 추후 추가)

### 3. 포트폴리오 준비
- Phase 1 완료 후 → GitHub README 작성
- Phase 3 완료 후 → HuggingFace 업로드
- Phase 5 완료 후 → 블로그 포스팅

---

## 성공 사례 (예상)

**최소 성공 (2주)**:
- ✅ Charlie Parker AI 1개
- ✅ GitHub 레포 공개
- ✅ 샘플 MP3 5개
- ✅ README 작성

**중간 성공 (4주)**:
- ✅ 3개 뮤지션 AI
- ✅ HuggingFace 모델 공개
- ✅ 블로그 포스트 1개
- ✅ YouTube 데모

**완전 성공 (6-8주)**:
- ✅ 5개 뮤지션 AI
- ✅ 실시간 잼 봇 완성
- ✅ Streamlit 웹 데모
- ✅ 기술 블로그 3개
- ✅ 커뮤니티 반응 (star 100+)

---

## 자주 묻는 질문

**Q: 고졸인데 괜찮나요?**
A: 완전히 괜찮습니다! 포트폴리오가 전부입니다. 실제 동작하는 데모만 있으면 됩니다.

**Q: GPU가 없어도 되나요?**
A: 네! RunPod 클라우드 GPU를 시간당 결제로 사용합니다. 총 비용 $90-180.

**Q: 음악 이론을 몰라도 되나요?**
A: 네! 모델이 알아서 학습합니다. 재즈를 좋아하면 충분합니다.

**Q: 취업에 도움이 되나요?**
A: 네! 백엔드 + AI 포트폴리오 조합은 강력합니다.

**Q: 실패하면 어떡하죠?**
A: 커뮤니티에 질문하세요! GitHub Issues 또는 Discord.

---

## 연락처

- **GitHub**: [Issues](링크)
- **Email**: (추가 예정)
- **Discord**: (추가 예정)

---

**Let's build your Charlie Parker AI! 🎷**

**시작하기**: `guides/PHASE0_SETUP.md` 읽기
