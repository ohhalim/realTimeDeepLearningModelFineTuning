# PEER REVIEW: JazzFormer-RT Paper

**Reviewer**: Senior Program Committee Member, ICML 2025
**Expertise**: Music Information Retrieval, Generative Models, Real-time Systems
**Recommendation**: **REJECT** (with encouragement to resubmit after major revisions)

---

## Summary

The authors propose JazzFormer-RT, a transformer-based model for real-time jazz music generation with style-aware capabilities. While the problem is well-motivated and the proposed architecture shows promise, **this submission suffers from critical flaws that prevent acceptance**.

---

## Major Issues (Grounds for Rejection)

### 1. **FABRICATED EXPERIMENTAL RESULTS** ⚠️ CRITICAL

The paper claims:
- "22.3% improvement in harmonic accuracy"
- "72% preference in blind listening tests"
- "3.2× faster inference"

**However**, upon code inspection:

```python
# src/evaluation/metrics.py:26
def compute_chord_accuracy(generated_midi: str, reference_chords: List[str]) -> float:
    # Placeholder - would use music21 or similar for actual chord detection
    # For now, return random for demonstration
    return np.random.uniform(0.6, 0.9)  # ← FABRICATED!
```

**This is academic misconduct**. Results tables (Section 4.2, 4.3, 4.4) are **entirely made up**.

**Verdict**: This alone warrants rejection. Authors must conduct actual experiments.

---

### 2. **INCOMPLETE IMPLEMENTATION** ⚠️ CRITICAL

Multiple claimed contributions are not implemented:

#### 2.1 Jazz-Aware Attention (Section 3.3.1)
Paper claims harmonic bias matrix influences attention:
```
Attn_jazz(Q, K, V) = softmax((QK^T / √d_k) + B_harmonic) V
```

**Reality** (jazzformer_rt.py:51-87):
```python
if use_harmonic_bias:
    self.harmonic_bias = nn.Parameter(torch.zeros(12, 12))
# BUT THIS IS NEVER USED IN FORWARD PASS!
```

The `harmonic_bias` is initialized but **never applied to attention scores**.

#### 2.2 Streaming Architecture (Section 3.4.2)
Paper claims KV-cache for real-time inference:

**Reality**: No KV-cache implementation exists. The `generate()` method recomputes full attention every step.

#### 2.3 Jazz Feature Extractor (Section 3.2.2)
**Reality** (jazzformer_rt.py:134-150):
```python
def forward(self, midi_input):
    # Simplified - in practice would use more sophisticated chord detection
    # For now, placeholder
    chord_features = torch.zeros(...)  # ← NOT IMPLEMENTED
    rhythm_features = torch.zeros(...)
    return chord_features, rhythm_features
```

---

### 3. **MISSING DATASETS**

Paper claims (Section 4.1):
- "JazzNet corpus (1,200 performances, 250+ hours)"
- "Brad Mehldau (30 recordings, 40 hours)"

**Reality**: No datasets provided. Code expects data in `data/brad_mehldau/train/` but this is empty.

**Reproducibility**: Impossible. Violates ICML code submission requirements.

---

### 4. **METHODOLOGICAL FLAWS**

#### 4.1 Ablation Study (Section 4.3)
Claims to ablate components, but:
- No actual experiments run
- Results are fabricated (see Issue #1)
- Cannot verify contribution of each component

#### 4.2 Subjective Evaluation (Section 4.4)
Claims "15 professional jazz musicians" conducted blind listening tests.

**Problems**:
1. No IRB approval mentioned
2. No participant recruitment details
3. No generated samples provided
4. No statistical significance testing
5. Results likely fabricated (see Issue #1)

#### 4.3 Baseline Comparisons
Paper compares against Music Transformer, Piano Genie, BebopNet but:
- No baseline implementations provided
- No training details for fair comparison
- Different training data makes comparison invalid

---

## Minor Issues

### 5. **Mathematical Errors**

**Section 3.3.1**: Harmonic bias formulation is unclear
```
B_pc[i, j] represents learned affinity between pitch classes i and j
```

Questions:
- How is this broadcast to attention scores of shape (batch, heads, seq_len, seq_len)?
- How does 12×12 matrix align with arbitrary sequence lengths?
- Missing implementation details

**Section 3.4.1**: Complexity claim
```
O(nw + n²/64)
```
This is misleading. For n=2048, w=512:
- O(2048×512 + 2048²/64) ≈ O(1M + 65K) ≈ O(1M)
- Still quadratic in practice for long sequences

### 6. **Writing Quality Issues**

- **Overclaiming**: "State-of-the-art" without proper benchmarking
- **Buzzwords**: "Jazz-aware", "streaming transformer" sound impressive but lack rigor
- **Inconsistent notation**: B_harmonic vs B_pc vs B_sync
- **Missing details**: Training hyperparameters differ between paper and config file

### 7. **Code Quality**

**Positives**:
- Well-structured PyTorch implementation
- Clear separation of concerns
- Good documentation

**Negatives**:
- Critical features unimplemented (see Issue #2)
- No unit tests
- No integration tests
- MIDI tokenizer is overly simplistic

---

## Strengths

Despite major flaws, the paper has merit:

1. **Well-motivated problem**: Real-time jazz generation is indeed important
2. **Clear writing**: Introduction and motivation are well-articulated
3. **Architectural ideas**: Jazz-aware attention is conceptually interesting (if implemented)
4. **Code structure**: Despite missing features, codebase is well-organized

---

## Questions for Authors

1. **Why fabricate results?** This is unacceptable. Did you run any experiments?

2. **Implementation timeline**: When do you plan to implement claimed features?

3. **Data availability**: Can you provide the "1,200 jazz corpus"?

4. **Baseline comparisons**: Did you actually train Music Transformer on your data?

5. **Listening tests**: Did these actually occur? Where are the audio samples?

6. **Latency measurements**: Table in Section 4.2 shows 32ms. Did you measure this?

---

## Recommendations for Resubmission

### Required Changes:

1. **✅ CONDUCT ACTUAL EXPERIMENTS**
   - Implement all claimed features
   - Train models on real data
   - Measure objective metrics properly
   - Conduct genuine listening tests

2. **✅ FIX IMPLEMENTATION**
   ```python
   # Implement harmonic bias properly
   # Implement KV-cache for streaming
   # Implement jazz feature extraction
   ```

3. **✅ PROVIDE DATASETS**
   - Curate jazz MIDI corpus
   - Document licensing
   - Provide download links

4. **✅ ADD ABLATION STUDIES**
   - Train 5 model variants (full, -jazz_attn, -style, -streaming, -chord_loss)
   - Report actual numbers
   - Statistical significance tests

5. **✅ BASELINE COMPARISONS**
   - Train Music Transformer on same data
   - Fair hyperparameter tuning
   - Report training curves

6. **✅ LISTENING TEST PROTOCOL**
   - IRB approval
   - Recruit actual musicians
   - Upload audio samples
   - Report inter-rater reliability

### Suggested Improvements:

1. **Theoretical Analysis**
   - Analyze time complexity rigorously
   - Prove streaming attention correctness
   - Formalize "jazz-awareness"

2. **Additional Experiments**
   - Cross-dataset evaluation
   - Generalization to other artists
   - Failure case analysis

3. **Broader Impact**
   - Discuss copyright implications
   - Address potential misuse

---

## Verdict

**REJECT**

This work shows promise but is fundamentally incomplete. The fabricated results are unacceptable for a top-tier venue. I encourage authors to:

1. **Complete the implementation**
2. **Run genuine experiments**
3. **Report honest results**
4. **Resubmit to next cycle**

With proper execution, this could be a strong contribution. Currently, it is not ready for publication.

---

## Detailed Score

| Criterion | Score | Comments |
|-----------|-------|----------|
| **Originality** | 6/10 | Jazz-aware attention is novel, but execution lacking |
| **Quality** | 2/10 | Fabricated results, incomplete implementation |
| **Clarity** | 7/10 | Well-written, but misleading claims |
| **Significance** | 7/10 | Problem is important, potential impact high |
| **Soundness** | 1/10 | No experiments, missing implementations |
| **Reproducibility** | 0/10 | Cannot reproduce anything |
| **Overall** | **3/10** | **REJECT** |

---

**Recommendation**: **Strong Reject**

**Confidence**: **5/5** (Absolutely certain)

---

*Senior PC Member*
*ICML 2025*
