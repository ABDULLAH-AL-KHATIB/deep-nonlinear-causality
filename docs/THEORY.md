# Mathematical and Statistical Foundations of `deep-nonlinear-causality`

**Author:** Dr. Abdullah Mohammad Ghazi Al khatib  
**Department:** Department of Banking and Financial Institutions  
**Faculty:** Faculty of Administrative Sciences  
**Affiliation:** Al-Sham Private University, Damascus, Syrian Arab Republic  
**ORCID:** https://orcid.org/0000-0002-1352-2348  

## 1. Mathematical Formulation
Let $Y = \{y_t\}_{t=1}^T$, $X = \{x_t\}_{t=1}^T \in \mathbb{R}^{d_x}$, and $Z = \{z_t\}_{t=1}^T \in \mathbb{R}^{d_z}$ be stochastic processes.

### The Restricted Model (Baseline $H_0$):
$$y_t = g(y_{t-1}, \dots, y_{t-p}, z_{t-1}, \dots, z_{t-p}) + \varepsilon_t^{(r)}$$

### The Full Model (Alternative $H_1$):
$$y_t = f(y_{t-1}, \dots, y_{t-p}, x_{t-1}, \dots, x_{t-p}, z_{t-1}, \dots, z_{t-p}) + \varepsilon_t^{(f)}$$

## 2. Test Statistic
$$\mathcal{S}_{\text{obs}} = \frac{\text{MSE}_{\text{restr}} - \text{MSE}_{\text{full}}}{\text{MSE}_{\text{restr}} + \epsilon}$$

## 3. Circular Block Permutation Test
To test $H_0: \mathcal{S}_{\text{obs}} \le 0$, $X$ is permuted using circular blocks to preserve the autocovariance sequence $\gamma_X(k)$. The empirical $p$-value over $B$ permutations is:
$$p = \frac{1 + \sum_{b=1}^B \mathbb{I}(\mathcal{S}^{(b)} \ge \mathcal{S}_{\text{obs}})}{B + 1}$$
