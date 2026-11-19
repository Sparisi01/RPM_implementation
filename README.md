# RPM implementation

![Build](https://img.shields.io/badge/build-passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)

**Authors**: _Parisi Simone_

**Keywords**: _Tikhonov Regularization, Inverse Conduction Problem, LMMSE, Soft Sensing_

**Paper**: [Efficient Physics-
Informed Inverse Heat Conduction Model for Temperature Sensing of Inaccessible Surface](https://www.preprints.org/manuscript/202611.1768).

---

## Description

This code implements the RPM model for solving the 1D IHCP

$
\Lambda_{R,\phi} = \frac{\hat H_0^\dagger e^{i\phi}}{|\hat H_0|^2+R}
$

where both $\phi$ and $R$ are real vectors. The model is optimized minimizing the trimmed MSE

MSE $(\mathcal{T}_{\mathrm{mid}}) = \frac{1}{|\mathcal{D}_{\mathrm{train}}|}\sum_{i \in \mathcal{D}_{\mathrm{train}}} \frac{1}{|\mathcal{J}_{\mathrm{mid}}|}\sum_{j \in \mathcal{J}_{\mathrm{mid}}} \left| \mathcal{F}^{-1}\left[\Lambda_{R,\phi}\hat{g}^{(i)}_{\mathrm{obs}}\right](t_j) - f^{(i)}(t_j) \right|^2$,

via Gradient Descent, implemented in PyTorch (src/RPM.py). 

The model is tested on a synthetic dataset generated via Gaussian Processes (dataset_generator.ipynb) and acquired data. 

For more detailed informations we refer you to the paper.