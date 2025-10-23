# 🎯 QPoland Quantum Hackathon - START HERE

## ⚡ Quick Start (5 minutes)

```bash
cd /scratch/cbjp404/qpoland

# Test your setup
python quickstart.py
```

This will:
- ✅ Load MUTAG dataset
- ✅ Extract topological, CTQW, and hybrid features
- ✅ Train SVM with 10-fold CV
- ✅ Report accuracy and F1-scores

**Expected output**: 75-85% accuracy on MUTAG

---

## 📁 Project Structure

```
/scratch/cbjp404/qpoland/
├── README.md                 # Project overview
├── START_HERE.md            # This file
├── TECHNICAL_GUIDE.md       # Deep technical details (35 pages)
├── ATTACK_STRATEGY.md       # Day-by-day winning strategy
├── quickstart.py            # Quick validation script
├── requirements.txt         # Dependencies (already installed)
│
├── qkernels/                # Core implementation
│   ├── datasets.py          # Dataset loading (TUDataset + GraKeL)
│   ├── features.py          # Feature extraction (Topo + CTQW)
│   ├── kernels.py           # Kernel SVMs (RBF, linear, LCTK)
│   ├── quantum.py           # Quantum kernels (Quri Parts)
│   ├── eval.py              # Cross-validation & metrics
│   ├── viz.py               # Visualizations
│   └── utils.py             # Helper functions
│
├── experiments/
│   ├── run_cv.py            # Single experiment runner
│   └── run_full_benchmark.py # Full benchmark suite
│
├── data/                    # Auto-created for datasets
├── results/                 # Auto-created for outputs
└── cache/                   # Auto-created for caching
```

---

## 🚀 Three Ways to Run Experiments

### Option 1: Quick Test (Recommended First)
```bash
python quickstart.py
```
- Tests all features on MUTAG
- Takes ~2 minutes
- Perfect for validating setup

### Option 2: Single Experiment
```bash
python experiments/run_cv.py \
    --dataset MUTAG \
    --feature hybrid \
    --kernel rbf \
    --cv-folds 10
```

### Option 3: Full Benchmark (Production)
```bash
python experiments/run_full_benchmark.py \
    --datasets MUTAG AIDS PROTEINS NCI1 PTC_MR \
    --features topological ctqw hybrid \
    --kernels rbf linear \
    --cv-folds 10 \
    --output-dir results
```
- Runs all combinations (5×3×2 = 30 experiments)
- Takes ~30-60 minutes depending on hardware
- Generates comprehensive results

---

## 📊 Key Files to Read

### 1. **TECHNICAL_GUIDE.md** (Must Read!)
35 pages covering:
- Why quantum methods work for molecular graphs
- CTQW mathematical foundation
- Feature engineering best practices
- Hyperparameter optimization
- Debugging common issues
- Expected performance benchmarks

### 2. **ATTACK_STRATEGY.md** (Your Playbook!)
Day-by-day strategy:
- Day 1: Foundation & baselines
- Day 2: Quantum enhancements
- Day 3: Optimization & quantum kernels
- Day 4: Analysis & report writing

### 3. **Implementation Files**
- `qkernels/features.py`: See how CTQW works
- `qkernels/quantum.py`: See quantum kernel implementation
- `experiments/run_full_benchmark.py`: See complete pipeline

---

## 🎯 Success Metrics

### Minimum Viable Product (Day 1)
- ✅ Load all 5 datasets
- ✅ Extract topological features
- ✅ Achieve ≥70% average accuracy
- ✅ Generate results CSV

### Competitive Performance (Day 2)
- ✅ Implement CTQW features
- ✅ Achieve ≥80% average accuracy
- ✅ Beat topological baseline by ≥5%

### Winning Submission (Day 3-4)
- ✅ Implement quantum kernel
- ✅ Achieve ≥85% average accuracy
- ✅ Generate all visualizations
- ✅ Write 4-page report
- ✅ Demonstrate quantum advantage

---

## 🔥 Pro Tips

### 1. Start Simple, Scale Up
```python
# Day 1: Topological only
python experiments/run_full_benchmark.py \
    --features topological --kernels rbf

# Day 2: Add CTQW
python experiments/run_full_benchmark.py \
    --features hybrid --kernels rbf

# Day 3: Add quantum
python experiments/run_cv.py \
    --dataset MUTAG --feature quantum --kernel quantum
```

### 2. Use Caching
First run is slow (downloads datasets), subsequent runs are fast.

### 3. Monitor Progress
```bash
# Watch results directory
watch -n 10 ls -lh results/

# Check logs
tail -f qpoland.log
```

### 4. GPU Acceleration
Your NVIDIA A2 is detected and ready:
- Quri Parts will use GPU for quantum simulation
- CTQW matrix operations can use CuPy (if installed)

---

## 📈 Expected Results

| Dataset | #Graphs | Topological | CTQW | Hybrid | Quantum | SOTA |
|---------|---------|-------------|------|--------|---------|------|
| MUTAG | 188 | 75% | 78% | 82% | 88% | 89% |
| AIDS | 2000 | 96% | 97% | 98% | 99% | 99% |
| PROTEINS | 1113 | 72% | 74% | 75% | 77% | 77% |
| NCI1 | 4110 | 80% | 82% | 84% | 86% | 87% |
| PTC_MR | 344 | 56% | 58% | 59% | 62% | 63% |

**Target**: Match or exceed SOTA on at least 2-3 datasets

---

## 🐛 Troubleshooting

### Issue: "No module named 'grakel'"
```bash
pip install grakel
```

### Issue: "Dataset download failed"
Datasets download automatically on first use. If it fails:
```python
from qkernels.datasets import MolecularGraphDataset
dataset = MolecularGraphDataset('MUTAG', use_grakel=False)
```

### Issue: "CTQW too slow"
Reduce number of graphs or time points:
```python
extractor = CTQWFeatureExtractor(time_points=[0.5, 1.0])  # Instead of 4
```

### Issue: "Low accuracy (<60%)"
Check:
1. Features are normalized (should happen automatically)
2. CV is stratified (should happen automatically)
3. Hyperparameters: Try C=10, gamma='scale'

---

## 📝 Report Checklist

### Page 1: Introduction
- [ ] Problem statement
- [ ] Why quantum methods
- [ ] CTQW mathematical foundation

### Page 2: Methodology
- [ ] Feature extraction pipeline
- [ ] Quantum circuit design
- [ ] Kernel formulation

### Page 3: Results
- [ ] Performance table (all datasets)
- [ ] Comparison with baselines
- [ ] Statistical significance

### Page 4: Analysis
- [ ] Visualizations (kernel matrices, t-SNE)
- [ ] Feature importance
- [ ] Quantum advantage discussion

---

## 🏆 Winning Strategy

**Best approach**: Hybrid topological + CTQW features with optimized RBF kernel

**Why**:
1. ✅ Competitive performance (80-85% average)
2. ✅ Fast to compute (~5 minutes for all datasets)
3. ✅ Strong theoretical foundation
4. ✅ Easy to explain and visualize
5. ✅ Clear quantum inspiration

**Quantum kernel**: Use as bonus/comparison, not primary method
- Harder to optimize
- More computational cost
- But great for "quantum advantage" narrative

---

## 🚦 Your Next Steps

### Right Now (5 min):
```bash
cd /scratch/cbjp404/qpoland
python quickstart.py
```

### Next Hour (60 min):
```bash
# Read technical guide
less TECHNICAL_GUIDE.md

# Run full benchmark on 2 datasets first
python experiments/run_full_benchmark.py \
    --datasets MUTAG AIDS \
    --features topological ctqw hybrid \
    --kernels rbf
```

### Today (4-6 hours):
Follow **ATTACK_STRATEGY.md** Day 1 plan

### This Week:
- Day 1: Baselines (4-6 hours)
- Day 2: CTQW enhancements (6-8 hours)  
- Day 3: Optimization (6-8 hours)
- Day 4: Report & visualizations (6-8 hours)

**Total time**: 22-30 hours spread over 4 days

---

## 💡 Key Insights from Research

1. **CTQW captures what matters**: Combines spectral properties (global) with node visitation (local)

2. **Small datasets favor kernels**: SVM with good kernel > deep learning on <5000 samples

3. **Hybrid wins**: Topological (interpretable) + CTQW (powerful) = best of both

4. **Time scales matter**: Use multiple time points [0.3, 0.7, 1.5, 3.0] to capture dynamics

5. **Normalization is critical**: Always use RobustScaler, fit only on training data

---

## 📚 Essential Reading

**Before starting**:
- `TECHNICAL_GUIDE.md` - Sections 1-3 (problem understanding, why quantum, implementation)

**While coding**:
- `TECHNICAL_GUIDE.md` - Section 4 (optimization techniques)
- `ATTACK_STRATEGY.md` - Current day's plan

**For report**:
- `TECHNICAL_GUIDE.md` - Section 5 (performance benchmarks)
- Example visualizations in `qkernels/viz.py`

---

## 🎓 Learning Resources

**Quantum Walks**:
- Original CTQW paper: Farhi & Gutmann (1998)
- Graph kernel survey: https://arxiv.org/abs/1903.11835

**Quantum ML**:
- Qiskit tutorials: https://qiskit.org/textbook/ch-machine-learning/
- Quri Parts docs: https://quri-parts.qunasys.com/

**Graph Kernels**:
- TUDataset benchmark: https://chrsmrrs.github.io/datasets/

---

## ✅ Final Checklist Before Starting

- [ ] `python quickstart.py` runs successfully
- [ ] Read TECHNICAL_GUIDE.md sections 1-3
- [ ] Read ATTACK_STRATEGY.md Day 1
- [ ] Understand your target: 80-85% average accuracy
- [ ] Know your timeline: 4 days, ~25 hours total
- [ ] Have coffee/energy drinks ready ☕

---

## 🚀 Ready to Win?

```bash
cd /scratch/cbjp404/qpoland
python quickstart.py
```

**After it runs successfully**, you're ready to follow the attack strategy!

**Good luck! You've got all the tools to win this! 🏆**

---

## 📞 Quick Reference

**Key commands**:
```bash
# Quick test
python quickstart.py

# Single experiment  
python experiments/run_cv.py --dataset MUTAG --feature hybrid --kernel rbf

# Full benchmark
python experiments/run_full_benchmark.py --datasets MUTAG AIDS --features hybrid --kernels rbf

# Check results
cat results/benchmark_summary.csv
```

**Key files**:
- Features: `qkernels/features.py`
- Kernels: `qkernels/kernels.py`
- Quantum: `qkernels/quantum.py`
- Visualization: `qkernels/viz.py`

**Key concepts**:
- CTQW: Quantum walk on graph adjacency matrix
- Fidelity kernel: K(i,j) = |⟨ψ_i|ψ_j⟩|²
- Hybrid features: Topological + CTQW concatenated

**Performance targets**:
- Baseline: 70-75%
- CTQW: 80-85%
- Quantum: 85-90%
- SOTA: 85-90%

---

**NOW GO WIN THIS HACKATHON! 🚀🏆**
