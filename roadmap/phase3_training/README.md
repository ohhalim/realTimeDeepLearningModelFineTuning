# Phase 3: 본격 학습 (4주, $50-100)

**목표**: JazzFormer-CR 전체 학습 및 고품질 모델 완성
**예상 시간**: 30-40시간 (4주)
**비용**: $50-100

---

## 핵심 내용

Phase 2가 성공했다면, 이제 전체 아키텍처로 학습합니다:

1. **Hierarchical LoRA-CR** 적용
2. **Multi-Scale Attention** 사용
3. **Contrastive Style Learning** 추가
4. **Adaptive Curriculum** 적용

상세 구현은 `../novel_architecture/` 참조

---

## 학습 계획

```
Week 1: Classical piano 사전학습 (MAESTRO dataset)
Week 2-3: Brad Mehldau 파인튜닝 (LoRA)
Week 4: 하이퍼파라미터 튜닝 + 최종 학습
```

---

## 실행

```bash
# Phase 2가 잘 동작하면
python training/train_jazzformer_cr.py \
    --config training/configs/phase3.yaml \
    --data_dir data/splits \
    --output_dir checkpoints/jazzformer_cr

# Lambda Labs A100 GPU 사용 ($0.50/시간)
# 예상: 20-30 시간 학습 = $10-15
```

---

**다음**: `../phase4_evaluation/README.md`
