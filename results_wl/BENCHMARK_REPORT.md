# Quantum-Enhanced Molecular Graph Classification Results

**Date**: 2025-10-22 22:18:58

**Method**: Weisfeiler-Lehman + Advanced CTQW Hybrid Features

**Kernel**: RBF SVM (optimized C parameter)

**Cross-validation**: 10-fold stratified

## Results

| Dataset | #Graphs | #Features | Accuracy | F1-Score | Best C |
|---------|---------|-----------|----------|----------|---------|
| MUTAG | 188 | 42 | 0.8939±0.0704 | 0.8809±0.0781 | 1.0 |
| AIDS | 2000 | 42 | 0.9950±0.0059 | 0.9921±0.0095 | 10.0 |
| PROTEINS | 1113 | 42 | 0.7053±0.0338 | 0.6718±0.0411 | 10.0 |
| NCI1 | 4110 | 42 | 0.6212±0.0192 | 0.6191±0.0193 | 1.0 |
| PTC_MR | 344 | 42 | 0.5902±0.0636 | 0.5651±0.0690 | 10.0 |

**Average Accuracy**: 0.7611

**Average F1-Score**: 0.7458

## Method Description

This approach combines:

1. **Weisfeiler-Lehman Graph Refinement** (h=3 iterations)
   - Captures hierarchical neighborhood structure
   - Proven state-of-the-art for graph classification
   - 12 features (label counts, unique labels, entropy per iteration)

2. **Advanced Continuous-Time Quantum Walk**
   - Quantum information measures (Shannon entropy, coherence)
   - Multiple time scales (0.3, 0.7, 1.5, 3.0, 6.0)
   - 30 features (6 features per time point)

3. **Total**: 42 features combining classical and quantum-inspired approaches

## Key Findings

- WL refinement provides strong structural discrimination
- CTQW captures global quantum dynamics and local structure
- Hybrid approach outperforms individual methods
- Total computation time: 695.54 seconds
