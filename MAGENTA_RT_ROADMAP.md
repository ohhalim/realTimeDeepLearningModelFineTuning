# Magenta RealTime 통합 로드맵

Brad Mehldau 스타일 실시간 음악 생성을 위한 Magenta RealTime 통합 계획

## 개요

**Phase 1 (현재)**: QLoRA로 빠른 실험 및 학습
**Phase 2 (미래)**: Magenta RT로 진짜 실시간 생성 구현

---

## Phase 1: QLoRA 파인튜닝 (완료/진행중)

### 목적
- PyTorch 생태계 학습
- 빠른 프로토타이핑
- 메모리 효율적인 실험
- Brad Mehldau MIDI 데이터 준비 및 검증

### 완료 항목
- [x] QLoRA 학습 스크립트 (`train_qlora.py`)
- [x] 음악 생성 스크립트 (`generate_qlora.py`)
- [x] 설정 파일 (`qlora_config.yaml`)
- [x] 문서화 (QLORA_GUIDE.md, QUICKSTART_QLORA.md)

### 진행중
- [ ] 실제 Brad Mehldau MIDI 데이터 수집
- [ ] QLoRA 모델 학습 및 평가
- [ ] 생성 품질 테스트

---

## Phase 2: Magenta RealTime 통합

### 2.1 준비 단계 (1-2주)

#### 환경 설정
- [ ] Colab TPU 환경 세팅
  - Magenta RT 설치 및 테스트
  - 데모 노트북 실행 확인

- [ ] 데이터 준비
  - Brad Mehldau MIDI → Audio 변환
  - 데이터 품질 검증 (30-50개 샘플)
  - 오디오 포맷: 48kHz stereo

#### 기술 스택 학습
- [ ] JAX 프레임워크 기초
- [ ] SpectroStream 코덱 이해
- [ ] MusicCoCa 임베딩 모델 이해

### 2.2 파인튜닝 실험 (2-4주)

#### Colab 기반 파인튜닝
```python
# Magenta RT 파인튜닝 예시 (Colab Notebook)
from magenta_rt import system

# 1. 모델 로드
model = system.MagentaRT(tag="large", device="tpu:v2-8")

# 2. Brad Mehldau 오디오로 파인튜닝
# (공식 파인튜닝 노트북 참조)
```

#### 실험 항목
- [ ] 기본 파인튜닝 (vanilla)
  - Learning rate: 1e-4
  - Epochs: 10-20
  - Batch size: Colab TPU 최대치

- [ ] 스타일 임베딩 실험
  - Brad Mehldau 오디오 → MusicCoCa 임베딩
  - 임베딩 클러스터링 분석
  - 스타일 일관성 검증

- [ ] 하이퍼파라미터 튜닝
  - Temperature: 0.8 - 1.5
  - Top-K: 20 - 80
  - Guidance weight: 3.0 - 7.0

### 2.3 평가 및 검증 (1주)

#### 정량적 평가
- [ ] FDopenl3 (Fr´echet Distance)
- [ ] KLpasst (Kullback–Leibler divergence)
- [ ] CLAPscore (텍스트 adherence)

#### 정성적 평가
- [ ] Brad Mehldau 스타일 재현도
  - 화성 진행 특징
  - 리듬 패턴
  - 즉흥 연주 스타일

- [ ] A/B 테스트
  - 원본 모델 vs 파인튜닝 모델
  - 블라인드 테스트 (5-10명)

### 2.4 배포 및 통합 (2-3주)

#### 로컬 배포 옵션

**옵션 A: 40GB GPU (A100, A6000)**
```bash
# Docker 기반 배포
docker pull us-docker.pkg.dev/brain-magenta/magenta-rt/magenta-rt:gpu
docker run -it --gpus device=0 \
  -v ~/.cache/magenta_rt:/cache \
  magenta-rt python -m magenta_rt.generate
```

**옵션 B: 경량화 버전 개발**
- 모델 크기 축소 (760M → 300M)
- Quantization (INT8)
- 8GB GPU 타겟

#### API 서버 구축
```python
# FastAPI 기반 실시간 생성 서버
from fastapi import FastAPI
from magenta_rt import system

app = FastAPI()
model = system.MagentaRT()

@app.post("/generate")
async def generate_music(prompt: str):
    style = model.embed_style(prompt)
    chunk, state = model.generate_chunk(style=style)
    return {"audio": chunk.samples.tolist()}
```

#### 웹 인터페이스
- [ ] React 기반 UI
- [ ] WebSocket으로 실시간 스트리밍
- [ ] 프롬프트 믹싱 컨트롤
- [ ] Audio injection 기능

---

## 기술적 도전 과제

### 1. 메모리 최적화
**문제**: 40GB GPU 요구사항
**해결책**:
- Model quantization (FP16 → INT8)
- Gradient checkpointing
- LoRA/QLoRA 어댑터 적용 가능성 탐구

### 2. 실시간 성능
**문제**: RTF ≥ 1× 보장
**해결책**:
- Batch size 최적화
- JIT compilation (JAX)
- GPU 최적화

### 3. 스타일 일관성
**문제**: Brad Mehldau 스타일 유지
**해결책**:
- 고품질 데이터 큐레이션
- Style embedding 고정/가중치
- Fine-grained control 추가

### 4. JAX ↔ PyTorch 통합
**문제**: QLoRA (PyTorch) + Magenta RT (JAX)
**해결책**:
- 모델 가중치 변환 유틸리티
- 공통 데이터 파이프라인
- 하이브리드 워크플로우

---

## 타임라인

### Short-term (1-2개월)
- [ ] QLoRA 학습 완료
- [ ] Brad Mehldau MIDI 데이터 수집 완료
- [ ] Magenta RT Colab 파인튜닝 실험

### Mid-term (3-4개월)
- [ ] Magenta RT 파인튜닝 모델 완성
- [ ] 정량/정성 평가 완료
- [ ] 프로토타입 데모 (Colab)

### Long-term (6개월+)
- [ ] 로컬 배포 (GPU 또는 경량화)
- [ ] 웹 인터페이스 개발
- [ ] 실시간 공연 시스템

---

## 리소스 요구사항

### 컴퓨팅
- **학습**: Colab TPU (무료) 또는 A100 GPU
- **배포**: 40GB GPU 또는 경량화 버전 (8GB)

### 데이터
- **최소**: 30개 Brad Mehldau 오디오/MIDI
- **권장**: 50-100개
- **형식**: 48kHz stereo WAV (또는 MIDI → 변환)

### 인력
- 개발자 1명 (풀타임 기준)
- 음악 전문가 1명 (평가용, 파트타임)

---

## 참고 자료

### 공식 문서
- [Magenta RT GitHub](https://github.com/magenta/magenta-realtime)
- [HuggingFace Model](https://huggingface.co/google/magenta-realtime)
- [Research Paper](https://arxiv.org/abs/2508.04651)

### Colab 노트북
- [기본 데모](https://colab.research.google.com/github/magenta/magenta-realtime/blob/main/notebooks/Magenta_RT_Demo.ipynb)
- [파인튜닝 노트북](https://github.com/magenta/magenta-realtime/tree/main/notebooks) (업데이트 예정)
- [Audio Injection 데모](https://github.com/magenta/magenta-realtime/tree/main/notebooks) (업데이트 예정)

### 관련 프로젝트
- [MusicFX DJ](https://labs.google/fx/tools/music-fx-dj) - Lyria RT 기반
- [Lyria RealTime API](https://ai.google.dev/gemini-api/docs/music-generation)

---

## 성공 지표 (KPIs)

### 기술적 지표
- [ ] RTF ≥ 1× 달성
- [ ] 컨트롤 latency < 2초
- [ ] FDopenl3 < 100 (QLoRA 대비 개선)

### 음악적 지표
- [ ] Brad Mehldau 스타일 재현도 > 70% (블라인드 테스트)
- [ ] 화성 진행 정확도 > 80%
- [ ] 리듬 패턴 일관성 > 75%

### 사용자 경험
- [ ] 프롬프트 응답 시간 < 3초
- [ ] 연속 생성 시간 > 5분 (품질 유지)
- [ ] 웹 UI 반응성 < 100ms

---

**작성일**: 2025-11-16
**버전**: 1.0
**다음 업데이트**: QLoRA Phase 1 완료 후
