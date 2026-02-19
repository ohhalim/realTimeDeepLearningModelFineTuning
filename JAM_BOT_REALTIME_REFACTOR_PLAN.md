# Real-time Jazz Solo MIDI Model Build Playbook

작성일: 2026-02-19  
프로젝트: `realTimeDeepLearningModelFineTuning`

이 문서는 **FL Studio에서 실시간 MIDI 입력(코드/제스처)을 받아, 딥러닝 모델이 피아노 솔로 MIDI를 생성해 다시 DAW로 보내는 시스템**을 만들기 위한 실행 플레이북이다.  
핵심은 `실시간성`과 `스타일 학습`을 분리하고, `MVP(Stage A) -> 확장(Stage B/C)`로 단계화하는 것이다.

## 0) 목표 스펙 (KPI)

- 최종 목표:
  - E2E latency (입력 -> 첫 출력 노트): `< 80ms`
  - Jitter: `< 5~10ms`
  - 항상 `1~2 bars ahead` 선생성 후 스케줄링
  - 생성 실패 시 fallback 즉시 전환
  - 끊김(dead-air) 임계치: `gap >= 180ms`를 이벤트로 집계
- Stage A 현실 목표 (게이트용):
  - E2E latency median `<= 120ms`
  - 3분 연속 끊김 없는 동작
  - chord/gesture 변화 반응 정상
  - dead-air ratio `< 2%`, dead-air event rate `< 0.5 events/sec`

## 1) 실행 원칙

1. 먼저 동작하는 시스템을 만든다 (복잡도 최소화).
2. 지표 통과 후에만 확장한다 (Gate 기반).
3. 학습은 원격(RunPod), 공연 추론은 로컬 GPU.
4. 실험 코드와 공연 코드를 분리한다.

## 2) 전체 파이프라인 (Training/Inference 분리)

### Training (RunPod)
- role-conditioned 데이터 준비
- 토큰화
- Transformer 학습 (필요 시 adapter/LoRA)

### Inference (Local)
- MIDI input capture
- conditioning 업데이트
- 모델 추론 (context window, cache, quantization)
- 생성 MIDI 미래 시점 스케줄링

## 3) Stage 설계

### Stage A (성공확률 최우선 MVP)
- 역할: `lead` 1개
- 입력: chord/gesture MIDI
- 출력: solo MIDI
- 프롬프트 전략: gesture/chord-change 기반 + `100ms` 버퍼
- 최소 컨트롤 토큰: `ROLE`, `TEMPO`, `CHORD_UNKNOWN`, `COND_SEP`
- 출력은 항상 `1~2 bars ahead`
- 스케줄러 축소 하한:
  - `min_output_horizon_ms = 250`
  - `min_output_notes = 8`

### Stage B (컨트롤 강화)
- 토큰 확장: `CHORD/KEY/BAR/BEAT`
- role 확장: `accompaniment`, `call_response`
- 데이터를 conditioning/target 2트랙 구조로 고정

### Stage C (품질/스타일 강화)
- pretrained symbolic model + adapter/LoRA
- retrieval phrase fallback 결합
- 필요 시 RL post-training

## 4) 데이터 스키마

### 디렉토리 구조

```text
data/
  raw_midi/
  roles/
    lead/
      000001_xxx/
        conditioning.mid
        target.mid
        meta.json
```

### meta.json 필수 필드

```json
{
  "role": "lead",
  "tempo": 128,
  "chord_progression": "Am7 | D7 | Gmaj7 | Cmaj7",
  "style": "brad_mehldau",
  "strategy": "gesture-conditioned",
  "split": "train",
  "bar_index": 0
}
```

### 데이터 제작 원칙

- `conditioning.mid`: 코드/루트/트리거 신호
- `target.mid`: 생성하려는 솔로/반주 라인
- 같은 조건에서 여러 변주를 확보한다.
- 자동 전사 MIDI(PiJAMA 등) 사용 시 정렬/중복/깨진 이벤트 정리 필수

## 5) 토큰화 설계

- NOTE_ON: `0-127`
- NOTE_OFF: `128-255`
- TIME_SHIFT: `256-383`
- VELOCITY: `384-447`
- SPECIAL/CONTROL: `448+`

### Stage A 최소 컨트롤 토큰

- `ROLE_LEAD`, `ROLE_ACCOMPANIMENT`, `ROLE_CALL_RESPONSE`
- `TEMPO_SLOW`, `TEMPO_MEDIUM`, `TEMPO_FAST`
- `CHORD_UNKNOWN`
- `COND_SEP`

중요: 실시간 성능은 `tokens/bar` 영향이 크다. Stage A는 단순 유지, Stage B에서 압축 토큰화(REMI 계열 등) 검토.

### 토큰 할당 테이블 (Stage B 확장 대비)

| Range | Purpose | Stage |
|---|---|---|
| `0-127` | NOTE_ON pitch | A/B/C |
| `128-255` | NOTE_OFF pitch | A/B/C |
| `256-383` | TIME_SHIFT | A/B/C |
| `384-447` | VELOCITY | A/B/C |
| `448-451` | PAD/BOS/EOS/MASK | A/B/C |
| `452-459` | ROLE/TEMPO/CHORD_UNKNOWN/COND_SEP | A |
| `460-479` | CHORD vocab reserve | B |
| `480-495` | KEY/BAR/BEAT/TRIGGER reserve | B |
| `496-503` | ENERGY token reserve | B/C |
| `504-511` | future extension reserve | C |

현재 구현은 `452-459`까지만 실제 사용한다. Stage B에서 reserve 구간을 활성화한다.

## 6) 학습 레시피

### Stage A 기본 실행

```bash
# 1) role 데이터 생성
python scripts/prepare_role_dataset.py \
  --input_dir data/raw_midi \
  --output_dir data/roles \
  --role lead \
  --conditioning_mode lower_register \
  --transpose_all_keys

# 2) lead 학습
python scripts/train_pytorch_transformer.py \
  --config configs/roles/lead.yaml \
  --use_role_dataset \
  --role lead

# 3) conditioning 기반 생성
python scripts/generate_pytorch_transformer.py \
  --checkpoint models/finetuned/pytorch_transformer_roles/lead/best_model.pt \
  --conditioning_midi data/roles/lead/<sample>/conditioning.mid \
  --role lead \
  --max_length 512 \
  --context_window 256 \
  --output output/lead_conditioned.mid
```

Runpod 자동 실행:

```bash
bash scripts/runpod_train_stage_a.sh --mode all --role lead --overwrite
```

### 학습/추론 config 분리

- 학습 config:
  - `configs/pytorch_transformer_config.yaml`
  - `configs/roles/*.yaml`
- 추론 config:
  - `configs/inference/realtime_stage_a.yaml`
- 규칙:
  - 학습 `data.max_length = 512`
  - 추론 `generation.max_length = 512`, `context_window = 256`
  - 추론 파라미터는 학습 config에서 읽지 않는다.

### Stage A 권장 하이퍼파라미터

- layers: `6-8`
- d_model: `256-384`
- context/max_length: `256-512` (우선 짧게)
- batch_size: `8-32`
- augment: transpose `+-6` 또는 all keys

스타일이 약하면 우선순위:
1. conditioning/target 분리 데이터 점검
2. 전처리 품질 개선
3. adapter/LoRA 전환 고려
4. 샘플링(temperature/top-p) 조정

## 7) 오프라인 평가 (라이브 전 필수)

### 기본 로그

- train/val loss
- generation time (1~2 bars)
- TTFN (time-to-first-note)
- chord-change response delay
- dead-air threshold(ms) 기준 이벤트 수

### 자동 지표

- chord-tone ratio
- note density
- pitch range
- repetition (n-gram)
- dead-air rate (무음/출력 공백, `gap >= 180ms`)

Stage A 우선순위: 스타일 완벽도보다 `반응성 + 끊김 없음 + 지연`.

### 스크립트

- `scripts/eval_offline_metrics.py`
  - 출력: `TTFN`, `note_density`, `pitch stats`, `repetition_ratio`, `dead_air_ratio`
  - 기본 dead-air threshold: `180ms`

## 8) 실시간 추론 아키텍처 (로컬)

### 스레드 권장 구조

- Clock thread
- Input thread
- Processing thread
- Generation thread

### Stage A 프롬프트 전략

- 입력 후 `100ms` 버퍼로 chord 완성 대기
- conditioning window: 최근 `40~60 notes` 또는 최근 `1~2 bars`
- 매 이벤트 재생성 금지, chord-change/트리거에서만 재생성

### 스케줄링 규칙

- 항상 현재 시점보다 미래에 배치
- 생성 지연 시 길이 축소해도 미래 배치 보장
- 축소 하한 미만이면 즉시 fallback 전환:
  - `output_horizon_ms < 250`
  - `output_notes < 8`
- 실패 조건:
  - inference step > 임계치(ms)
  - queue underrun
- 실패 시 즉시 fallback phrase로 전환

## 9) 경량화/가속 우선순위

1. context window 축소
2. KV cache (Commit 3)
3. ONNX Runtime(CUDA)
4. int8 quantization

ONNX 체크:
- step-by-step generate 경로
- dynamic axis 설정
- PyTorch vs ONNX 샘플 일관성 검증

## 10) 드롭 구간 전용 설계

- ENERGY token(`LOW/MID/HIGH`) 또는 density 목표값 추가
- velocity/음역 상승 램프
- 강박 chord tone 비중 증가, 약박 passing tone 허용

이 확장은 Stage B 이후 진행.

## 11) 현재 리포지토리 반영 상태 (2026-02-19)

완료:
- `scripts/prepare_role_dataset.py`
- `models/midi_dataset.py` role-conditioned 학습 지원
- `scripts/train_pytorch_transformer.py` role-aware 옵션
- `scripts/generate_pytorch_transformer.py` conditioning primer 지원
- `scripts/train_role_models.py`
- `scripts/eval_offline_metrics.py`
- `configs/roles/*.yaml`
- `configs/inference/realtime_stage_a.yaml`
- `configs/tokenizer_realtime.yaml`
- `realtime/` 런타임 골격 (`clock`, `midi_input`, `prompt_builder`, `generation_worker`, `scheduler`, `main`)
- generation worker `KV cache 1차`(prompt reuse cache) 적용

미완:
- FL Studio 실제 MIDI 왕복 현장 검증
- latency/jitter 벤치마크 자동화
- fallback 자동 전환 로직
- ONNX/int8 추론 경로

## 12) 다음 커밋 순서 (실행 계획)

### Commit 1 (완료)
- role dataset 생성 파이프라인
- Stage A 최소 토큰/데이터로더

### Commit 2 (완료)
- role-conditioned 학습/생성 경로

### Commit 2.5 (완료)
- `scripts/eval_offline_metrics.py` 추가
- 다음 실행: dead-air threshold `180ms` 기준 baseline 리포트 생성

### Commit 3 (구현 완료, 현장 검증 대기)
- `KV cache` 1차 적용 (generation worker, prompt reuse)
- `realtime/clock.py`, `realtime/midi_input.py`, `realtime/prompt_builder.py`, `realtime/generation_worker.py`, `realtime/scheduler.py`, `realtime/main.py`
- 남은 작업: FL Studio MIDI round-trip (1 bar ahead) 실측

### Commit 4
- `scripts/benchmark_latency.py`
- fallback trigger 규칙 + phrase fallback

### Commit 5
- ONNX export/quantize + 로컬 벤치 (KV cache와 조합)

### Commit 6
- Stage B 토큰(`CHORD/KEY/BAR/BEAT`) 확장

## 13) 라이브 전 최소 체크리스트

- [ ] 로컬 추론 TTFN Stage A 목표(`<=120ms`) 달성
- [ ] 2 bars 생성 시간이 BPM 대비 충분히 빠름
- [ ] chord-change 반응 안정
- [ ] dead-air (`gap >= 180ms`) ratio < 2%
- [ ] fallback 즉시 동작
- [ ] Stage A 통과 후 최종 목표(`E2E < 80ms`)로 최적화 진행
