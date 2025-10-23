# References

This document lists the key references that informed the design, implementation, and physics validation of the WL+CTQW hybrid framework, SVM training, datasets, and Qiskit-based quantum circuits.

---

## Qiskit Documentation & Tutorials (Time Evolution, Feature Maps)

- Quantum Real Time Evolution using Trotterization (Qiskit Algorithms Tutorial)
  - URL: https://qiskit-community.github.io/qiskit-algorithms/tutorials/13_trotterQRTE.html
- Approximate quantum compilation for time evolution (IBM Quantum Tutorial)
  - URL: https://quantum.cloud.ibm.com/docs/en/tutorials/approximate-quantum-compilation-for-time-evolution
- Improved Trotterized Time Evolution with AQC-Tensor (IBM Quantum Learning)
  - URL: https://learning.quantum.ibm.com/tutorial/improved-trotterized-time-evolution-with-approximate-quantum-compilation
- ZZFeatureMap (Qiskit Circuit Library)
  - URL: https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.library.ZZFeatureMap
- Qiskit Aer Simulator (documentation hub)
  - URL: https://qiskit.org/ecosystem/aer/

---

## Continuous-Time Quantum Walk (CTQW): Theory and Mixing

- Farhi, E., & Gutmann, S. (1998). Quantum computation and decision trees. Physical Review A, 58(2), 915–928.
  - arXiv: https://arxiv.org/abs/quant-ph/9706062
- Childs, A. M., Cleve, R., Deotto, E., Farhi, E., Gutmann, S., & Spielman, D. A. (2003). Exponential algorithmic speedup by a quantum walk.
  - arXiv: https://arxiv.org/abs/quant-ph/0209131
- Godsil, C. (2011). Average mixing matrix of a quantum walk.
  - arXiv: https://arxiv.org/abs/1103.2578
- Kendon, V. (2007). Decoherence in quantum walks — a review. Mathematical Structures in Computer Science, 17(6), 1169–1220.
  - DOI: https://doi.org/10.1017/S0960129507006354

---

## Quantum-Enhanced Graph Kernels via CTQW

- Cui, L., Li, M., Wang, Y., Bai, L., & Hancock, E. R. (2023). AERK: Aligned Entropic Reproducing Kernels through Continuous-time Quantum Walks.
  - arXiv: https://arxiv.org/abs/2303.03396
  - PDF: https://arxiv.org/pdf/2303.03396
- Bai, L., et al. (2024). AEGK: Aligned Entropic Graph Kernels Through Continuous-Time Quantum Walks.
  - IEEE Xplore: https://ieeexplore.ieee.org/document/10844511/
- Cui, L., Bai, L., et al. (2022). QESK: Quantum-based Entropic Subtree Kernels for Graph Classification.
  - arXiv: https://arxiv.org/abs/2212.05228

---

## Graph Kernels and Weisfeiler–Lehman (WL)

- Shervashidze, N., Schweitzer, P., van Leeuwen, E. J., Mehlhorn, K., & Borgwardt, K. M. (2011). Weisfeiler–Lehman Graph Kernels. Journal of Machine Learning Research, 12, 2539–2561.
  - PDF: https://www.jmlr.org/papers/volume12/shervashidze11a/shervashidze11a.pdf
- Kriege, N. M., Giscard, P.-L., & Wilson, R. C. (2020). A survey on graph kernels. Applied Network Science, 5, 6.
  - Open access: https://appliednetsci.springeropen.com/articles/10.1007/s41109-019-0195-3
  - arXiv: https://arxiv.org/abs/1903.11835

---

## Datasets and Libraries

- TUDataset: Morris, C., Kriege, N. M., Bause, F., Kersting, K., Mutzel, P., & Neumann, M. (2020). TUDataset: A collection of benchmark datasets for graph classification.
  - arXiv: https://arxiv.org/abs/2007.08663
  - Dataset index: https://chrsmrrs.github.io/datasets
- GraKeL: Siglidis, G., Nikolentzos, G., Limnios, S., Giatsidis, C., Skianis, K., & Vazirgiannis, M. (2020). GraKeL: A Graph Kernel Library in Python. Journal of Machine Learning Research, 21(54), 1–5.
  - JMLR: https://www.jmlr.org/papers/v21/19-322.html
- PyTorch Geometric: Fey, M., & Lenssen, J. E. (2019). Fast Graph Representation Learning with PyTorch Geometric.
  - arXiv: https://arxiv.org/abs/1903.02428
  - Project: https://pytorch-geometric.readthedocs.io/
- NetworkX: Hagberg, A. A., Schult, D. A., & Swart, P. J. (2008). Exploring network structure, dynamics, and function using NetworkX.
  - Proceedings of the 7th Python in Science Conference (SciPy2008), 11–15.
  - Site: https://networkx.org/
- Quri Parts (QunaSys): Quantum computing library used for kernel experimentation.
  - Docs: https://quri-parts.qunasys.com/
- Qulacs: High-performance quantum circuit simulator.
  - Site: https://github.com/qulacs/qulacs

---

## Topological Indices and Spectral Features

- Wiener, H. (1947). Structural determination of paraffin boiling points. Journal of the American Chemical Society, 69(1), 17–20.
  - DOI: https://doi.org/10.1021/ja01193a005
- Randić, M. (1975). Characterization of molecular branching. Journal of the American Chemical Society, 97(23), 6609–6615.
  - DOI: https://doi.org/10.1021/ja00856a001
- Estrada, E. (2000). Characterization of 3D molecular structure. Chemical Physics Letters, 319(5–6), 713–718.
  - DOI: https://doi.org/10.1016/S0009-2614(00)00158-5
- Estrada, E., & Hatano, N. (2008). Communicability in complex networks. Physical Review E, 77(3), 036111.
  - DOI: https://doi.org/10.1103/PhysRevE.77.036111

---

## Optional Background (useful context)

- Continuous-time quantum walk — Physics overview and definitions.
  - Review (Physics Reports) overview: https://www.sciencedirect.com/science/article/abs/pii/S0370157311000184

---

## How these references were used

- **CTQW + AERK/QESK/AEGK** informed the selection of quantum features (entropy, average mixing matrix, coherence) and the physics plots.
- **WL Graph Kernels + Survey** grounded the WL side of the hybrid features and the SVM kernel choices.
- **Qiskit tutorials and docs** guided the Trotterization circuits and feature-map/ansatz circuit designs used in `experiments/qiskit_circuits.py` and `experiments/qiskit_analysis.py`.
- **TUDataset, GraKeL, PyG** references justify dataset handling and library choices in `qkernels/datasets.py`.
- **Topological indices** justify classical descriptors included in the hybrid feature vector.
