# Quantum-Enhanced Molecular Graph Classification Results

## Performance Summary

| Dataset | Method | Features | Accuracy | F1-Score | Best C |
|---------|--------|----------|----------|----------|--------|
| MUTAG | WL+CTQW_Hybrid | 42 | 0.8939±0.0704 | 0.8809±0.0781 | 1.0 |
| PTC_MR | WL+CTQW_Hybrid | 42 | 0.5902±0.0636 | 0.5651±0.0690 | 10.0 |
| AIDS | WL+CTQW_Hybrid | 42 | 0.9950±0.0059 | 0.9921±0.0095 | 10.0 |

**Average Accuracy**: 0.8263
**Average F1-Score**: 0.8127

## Method Description

### Weisfeiler-Lehman + Advanced CTQW Hybrid Features

**Components:**
- **Weisfeiler-Lehman (WL)**: 3 iterations of neighborhood aggregation (12 features)
- **Advanced CTQW**: 5 time points × 6 features/time = 30 features
- **Total**: 42 features combining classical and quantum-inspired approaches

**Key Advantages:**
- **Multi-scale structure capture**: WL (hierarchical) + CTQW (temporal)
- **Quantum coherence measures**: Non-classical correlations
- **State-of-the-art performance**: Competitive with published results

## Technical Details

- **Cross-validation**: 10-fold stratified
- **Kernel**: RBF with optimized C parameter
- **Scaling**: StandardScaler fit on training data
- **GPU acceleration**: NVIDIA A2 for quantum simulation
