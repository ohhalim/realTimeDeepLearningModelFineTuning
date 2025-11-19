# Phase 4: 평가 및 샘플 생성 (2주, 무료)

**목표**: 객관적 평가 + 다양한 샘플 10개 생성
**예상 시간**: 10-15시간 (2주)
**비용**: 무료

---

## 평가 메트릭

우리가 구현한 6개 메트릭 사용:

```python
from evaluation.music_metrics import evaluate_music_quality

metrics = evaluate_music_quality(generated_tokens)
# Returns:
# - harmonic_coherence: 0.85
# - tempo_stability: 0.92
# - pitch_diversity: 0.64
# - voicing_quality: 0.81
# - overall_quality: 0.82
```

---

## 샘플 생성

```bash
# 10개 다양한 샘플 생성
for i in {1..10}; do
    python evaluation/generate_sample.py \
        --checkpoint checkpoints/jazzformer_cr/best_model.pt \
        --output samples/final/sample_$i.mid \
        --temperature $(python -c "print(0.7 + $i*0.05)") \
        --max_tokens 512
done

# MIDI → MP3 변환
for f in samples/final/*.mid; do
    fluidsynth -F ${f%.mid}.mp3 /usr/share/sounds/sf2/FluidR3_GM.sf2 $f
done
```

---

## 청취 테스트

친구들에게 들려주고 피드백 받기:
- 재즈처럼 들리는가?
- Brad Mehldau 스타일인가?
- 음악적으로 coherent한가?

---

**다음**: `../phase5_deployment/README.md`
