# Hierarchical StyleLoRA-Transformer Architecture Design

## Paper Title
**"Hierarchical StyleLoRA-Transformer: Efficient and Interpretable Style Transfer for Expressive Jazz Piano Generation"**

Target: ICML 2026, NeurIPS 2025, or ICLR 2026

---

## 1. Novel Contributions

### 1.1 Hierarchical LoRA Decomposition
Traditional fine-tuning updates all parameters or uses single LoRA. We propose **hierarchical LoRA modules** that separately model:

- **Harmony LoRA (H-LoRA)**: Chord progressions and harmonic structure
- **Voicing LoRA (V-LoRA)**: Chord voicings and note distribution
- **Rhythm LoRA (R-LoRA)**: Timing, syncopation, and rhythmic patterns
- **Dynamics LoRA (D-LoRA)**: Velocity and articulation

**Key Insight**: Jazz piano style is compositional. Brad Mehldau's style = unique harmony choices + specific voicings + characteristic rhythm.

### 1.2 Style Contrastive Learning
Learn style embeddings by contrasting:
- Brad Mehldau performances vs. Bill Evans
- Brad Mehldau vs. Keith Jarrett
- Brad Mehldau vs. Oscar Peterson

**Loss Function**:
```
L_style = triplet_loss(anchor_mehldau, positive_mehldau, negative_other)
```

### 1.3 Multi-Scale Temporal Modeling
Three hierarchical levels:
1. **Bar-level Encoder** (4-8 bars): Macro structure, form
2. **Beat-level Encoder** (16-32 beats): Phrase structure
3. **Event-level Decoder** (512-2048 events): Note generation

### 1.4 Interpretable Style Control
Each LoRA module can be weighted:
```python
output = base_model(x) + α_H * H_LoRA(x) + α_V * V_LoRA(x) + α_R * R_LoRA(x)
```
Users can control: "80% Mehldau harmony + 50% Mehldau rhythm + 100% Mehldau voicing"

---

## 2. Architecture Details

### 2.1 Base Model
Music Transformer (Huang et al., 2018) with modifications:
- d_model: 768
- num_layers: 12
- num_heads: 12
- d_ff: 3072
- Parameters: ~86M (base)

### 2.2 Hierarchical LoRA Modules

```python
class HierarchicalLoRA(nn.Module):
    def __init__(self, d_model=768, r=16):
        # Harmony LoRA - applied to layers 0-3 (early layers capture harmony)
        self.harmony_lora = LoRAAdapter(
            target_layers=[0, 1, 2, 3],
            r=r, alpha=32
        )

        # Voicing LoRA - applied to layers 4-7 (middle layers)
        self.voicing_lora = LoRAAdapter(
            target_layers=[4, 5, 6, 7],
            r=r, alpha=32
        )

        # Rhythm LoRA - applied to layers 8-11 (late layers)
        self.rhythm_lora = LoRAAdapter(
            target_layers=[8, 9, 10, 11],
            r=r, alpha=32
        )

        # Dynamics LoRA - applied to output projection
        self.dynamics_lora = LoRAAdapter(
            target_layers=["output_proj"],
            r=r//2, alpha=16
        )
```

**Total LoRA Parameters**:
- H-LoRA: 4 layers × 2 × 768 × 16 = 98,304
- V-LoRA: 4 layers × 2 × 768 × 16 = 98,304
- R-LoRA: 4 layers × 2 × 768 × 16 = 98,304
- D-LoRA: 1 layer × 2 × 768 × 8 = 12,288
- **Total**: ~307K (0.36% of base model)

### 2.3 Style Encoder

```python
class StyleEncoder(nn.Module):
    """Encode a sequence into style embedding"""
    def __init__(self, d_model=768, d_style=256):
        self.bar_encoder = BarLevelEncoder(d_model)  # Process 4-8 bars
        self.style_projection = nn.Linear(d_model, d_style)

    def forward(self, sequence):
        # sequence: [batch, seq_len, d_model]
        bar_features = self.bar_encoder(sequence)  # [batch, n_bars, d_model]
        style_emb = self.style_projection(bar_features.mean(dim=1))  # [batch, d_style]
        return F.normalize(style_emb, dim=-1)
```

### 2.4 Multi-Scale Architecture

```
Input MIDI Tokens
    ↓
┌─────────────────────────────────────┐
│ Event-level Embedding (512 vocab)   │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Beat-level Aggregation (16 beats)   │  ← Learns phrase structure
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Bar-level Aggregation (4-8 bars)    │  ← Learns form, style
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Style Encoder → Style Embedding     │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Music Transformer (12 layers)       │
│  + Hierarchical LoRA (H/V/R/D)      │
└─────────────────────────────────────┘
    ↓
Output MIDI Tokens
```

---

## 3. Training Strategy

### 3.1 Three-Stage Training

**Stage 1: Pre-training (Optional)**
- Dataset: 10,000+ general MIDI files (Lakh, MAESTRO)
- Objective: Standard autoregressive language modeling
- Duration: 50 epochs
- GPU: 1× A100 (40GB)

**Stage 2: Style Contrastive Learning**
- Dataset: 50 Brad Mehldau + 50 Bill Evans + 50 Keith Jarrett
- Objective: Triplet loss + reconstruction loss
```python
L_total = L_reconstruction + λ_style * L_triplet
```
- Duration: 30 epochs
- GPU: 1× A100 (40GB)

**Stage 3: Hierarchical LoRA Fine-tuning**
- Dataset: 50-100 Brad Mehldau transcriptions only
- Freeze base model, train only LoRA parameters
- Objective: Multi-task loss
```python
L_total = L_next_token + λ_H * L_harmony + λ_V * L_voicing + λ_R * L_rhythm
```
- Duration: 100 epochs
- GPU: 1× RTX 4090 (24GB) or 1× A100

### 3.2 Multi-Task Objectives

**Harmony Loss**: Predict chord labels from note events
```python
def harmony_loss(logits, target_chords):
    # Extract chord from generated notes
    predicted_chords = extract_chords(logits)
    return cross_entropy(predicted_chords, target_chords)
```

**Voicing Loss**: Match note distribution within chords
```python
def voicing_loss(generated_notes, target_notes):
    # Compare pitch class distribution
    gen_dist = pitch_class_histogram(generated_notes)
    tgt_dist = pitch_class_histogram(target_notes)
    return kl_divergence(gen_dist, tgt_dist)
```

**Rhythm Loss**: Match timing distribution
```python
def rhythm_loss(generated_timing, target_timing):
    # Inter-onset interval distribution
    gen_ioi = compute_ioi(generated_timing)
    tgt_ioi = compute_ioi(target_timing)
    return wasserstein_distance(gen_ioi, tgt_ioi)
```

---

## 4. Evaluation Metrics

### 4.1 Objective Metrics

1. **Perplexity**: Standard language modeling metric
2. **Harmonic Accuracy**: Chord recognition accuracy (using pretrained chord detector)
3. **Rhythm Consistency**:
   - Tempo stability (variance)
   - Syncopation index
4. **Voicing Similarity**:
   - Pitch range distribution
   - Interval distribution
5. **Style Classification Accuracy**: Train classifier to distinguish Mehldau vs others

### 4.2 Subjective Metrics (Human Evaluation)

**Listening Test** (n=30 musicians, 5-point Likert scale):
1. "Does this sound like Brad Mehldau?" (Style accuracy)
2. "Is the harmony sophisticated?" (Musical quality)
3. "Is the rhythm interesting?" (Musical quality)
4. "Would you listen to this again?" (Overall quality)

**Turing Test**:
- 20 real Mehldau excerpts (10 seconds each)
- 20 generated excerpts (10 seconds each)
- Ask musicians: "Is this real or AI-generated?"
- Target: >40% fooling rate (random is 50%)

---

## 5. Baselines for Comparison

1. **Standard Fine-tuning**: Fine-tune all parameters (no LoRA)
2. **Single LoRA**: Standard LoRA without hierarchy
3. **Music Transformer (2018)**: Original paper implementation
4. **MusicLM (2023)**: Google's music generation model (if reproducible)
5. **Magenta Transformer**: Official Magenta implementation

---

## 6. Ablation Studies

Test each component's contribution:

| Model Variant | Perplexity ↓ | Harmonic Acc ↑ | Style Acc ↑ | Human Rating ↑ |
|---------------|--------------|----------------|-------------|----------------|
| Full Model | ? | ? | ? | ? |
| w/o H-LoRA | ? | ? | ? | ? |
| w/o V-LoRA | ? | ? | ? | ? |
| w/o R-LoRA | ? | ? | ? | ? |
| w/o Style Contrastive | ? | ? | ? | ? |
| w/o Multi-Scale | ? | ? | ? | ? |
| Single LoRA | ? | ? | ? | ? |

---

## 7. Expected Results

### 7.1 Quantitative Improvements
- **Perplexity**: 15-20% improvement over baseline
- **Style Classification**: >90% accuracy in distinguishing Mehldau
- **Parameter Efficiency**: 99.6% fewer trainable parameters vs full fine-tuning
- **Training Speed**: 3-4× faster than full fine-tuning

### 7.2 Qualitative Improvements
- More coherent harmonic progressions
- Characteristic Mehldau voicings (rootless voicings, upper structures)
- Syncopated rhythm patterns
- Interpretable style control

---

## 8. Paper Structure Outline

### Abstract (200 words)
- Problem: Style transfer in music generation
- Challenge: Computational cost, lack of interpretability
- Solution: Hierarchical LoRA decomposition
- Results: 99.6% parameter reduction, SOTA performance

### 1. Introduction
- Motivation: Expressive music generation
- Challenges: Data scarcity, style complexity
- Contributions: Hierarchical LoRA, style contrastive learning

### 2. Related Work
- Music generation models
- Parameter-efficient fine-tuning (LoRA, Adapters)
- Style transfer in music

### 3. Methodology
- 3.1 Problem Formulation
- 3.2 Hierarchical LoRA Architecture
- 3.3 Style Contrastive Learning
- 3.4 Multi-Scale Temporal Modeling
- 3.5 Training Strategy

### 4. Experiments
- 4.1 Experimental Setup
- 4.2 Datasets
- 4.3 Baseline Comparisons
- 4.4 Ablation Studies
- 4.5 Human Evaluation

### 5. Results and Analysis
- 5.1 Quantitative Results (tables)
- 5.2 Qualitative Analysis (audio examples)
- 5.3 Interpretability Analysis (LoRA weight visualization)
- 5.4 Computational Efficiency

### 6. Discussion
- Limitations
- Future work
- Broader impact

### 7. Conclusion

### References (50-60 papers)

---

## 9. Implementation Timeline

**Week 1-2**: Core architecture implementation
**Week 3-4**: Training pipeline and data preprocessing
**Week 5-8**: Experiments and baseline comparisons
**Week 9-10**: Ablation studies and analysis
**Week 11-12**: Human evaluation
**Week 13-14**: Paper writing
**Week 15-16**: Revisions and submission

---

## 10. Computational Requirements

- **Training**: 1× A100 (40GB) × 200 hours = ~$400 (cloud cost)
- **Evaluation**: 1× RTX 4090 × 50 hours
- **Storage**: 50GB for datasets, 20GB for checkpoints
- **Total estimated cost**: $500-800

---

## 11. Datasets

### Training Data
- **Brad Mehldau**: 50-100 transcriptions (from MIDI_DATA_SOURCES.md)
- **Contrastive**: 50 Bill Evans, 50 Keith Jarrett, 50 Oscar Peterson
- **Pre-training** (optional): Lakh MIDI Dataset (10K files)

### Evaluation Data
- Hold-out: 10 Brad Mehldau pieces (not in training)
- Real-world test: 5 new Mehldau performances (post-2024)

---

## 12. Code Repository Structure

```
research/
├── models/
│   ├── hierarchical_lora.py          # Core LoRA modules
│   ├── style_encoder.py              # Style contrastive learning
│   ├── music_transformer.py          # Base model
│   └── multiscale_aggregator.py      # Bar/Beat level encoders
├── training/
│   ├── train_stage1_pretraining.py
│   ├── train_stage2_contrastive.py
│   ├── train_stage3_lora_finetuning.py
│   └── losses.py                     # Multi-task losses
├── evaluation/
│   ├── objective_metrics.py
│   ├── human_evaluation_interface.py
│   └── generate_samples.py
├── data/
│   ├── midi_preprocessor.py
│   ├── style_dataset.py              # Triplet sampling
│   └── multiscale_tokenizer.py
├── experiments/
│   ├── run_baselines.py
│   ├── run_ablations.py
│   └── analyze_results.py
├── paper/
│   ├── paper.tex                     # LaTeX source
│   ├── figures/
│   └── tables/
└── configs/
    ├── stage1_pretraining.yaml
    ├── stage2_contrastive.yaml
    └── stage3_lora.yaml
```

---

## 13. Key References

1. Huang et al. (2018) - Music Transformer
2. Hu et al. (2021) - LoRA: Low-Rank Adaptation
3. Oore et al. (2018) - Learning to Create Piano Performances
4. Donahue et al. (2023) - MusicLM
5. Agostinelli et al. (2023) - MusicGen
6. Hawthorne et al. (2019) - Onsets and Frames
7. Payne (2019) - MuseNet
8. Dhariwal et al. (2020) - Jukebox

---

## 14. Novelty Checklist

✅ **New Architecture**: Hierarchical LoRA decomposition (H/V/R/D)
✅ **New Training Method**: Style contrastive learning for music
✅ **New Application**: Interpretable style control in music generation
✅ **Strong Baselines**: Compare against 5+ existing methods
✅ **Comprehensive Evaluation**: Objective + subjective metrics
✅ **Ablation Studies**: Test each component
✅ **Reproducibility**: Open-source code, clear hyperparameters
✅ **Broader Impact**: Discusses implications for music creation

---

## 15. Risk Mitigation

**Risk 1**: Insufficient Brad Mehldau data (50-100 MIDIs)
- **Mitigation**: Data augmentation (transpose, tempo shift), audio-to-MIDI conversion

**Risk 2**: Human evaluation is expensive/time-consuming
- **Mitigation**: Use Prolific/MTurk, compensate fairly ($15/hour), limit to 30 participants

**Risk 3**: Baselines are too strong
- **Mitigation**: Focus on efficiency (parameters, speed) and interpretability as main contributions

**Risk 4**: Results are not significant
- **Mitigation**: Run statistical tests (t-test, Cohen's d), report confidence intervals

---

This architecture is designed to be:
1. **Novel**: Hierarchical LoRA is new in music generation
2. **Efficient**: 99.6% fewer parameters than full fine-tuning
3. **Interpretable**: Each LoRA module controls specific musical aspect
4. **Rigorous**: Comprehensive evaluation with baselines and ablations
5. **Publishable**: Clear contributions, strong experiments, ready for top-tier venue
