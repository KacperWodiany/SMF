# Stochastic Financial Mathematics

The following project contains source code of the library that was created as a part of academic project during Stochastic Financial Mathematics course. It implements Monte Carlo Methods for pricing options in the following market model. We consider two assets $S^1_t$, $S^2_t$, such that

$$
\log\Big(\frac{S^i_{t+1}}{S^i_{t}}\Big) = \alpha_i N_t^i + \beta_i, \ i=1, 2,
$$

where $\alpha_i, \beta_i \in \mathbb{R}$, and $N_t^i$ are standard normal random variables $N(0, 1)$. Moreover:
- $N_t^i$, $N_s^i$ are independent for all $i, j$ and $t\neq s$,
- $N_t^1$, $N_t^2$ are correlated with constant correlation coefficient $\rho$.

We also consider risk-free instrument $S^0_t$, which represents time value of money.
