# Formal Hypothesis

Let `theta_base` denote the parameters learned by a strong baseline pipeline and `theta_gb` denote the parameters learned by the same pipeline augmented with Exposure-Debt Ghost Bank.

Because prior work shows that imbalance can sometimes be handled well by tuned standard components, the baseline in this project should not be weak by construction. It should include a reasonable training setup, such as appropriate augmentation and optimization choices, before the ghost bank is added.

The null and alternative hypotheses are:

- `H_0`: Exposure-Debt Ghost Bank does not improve minority-class performance relative to the strong baseline or to static memory variants, after controlling for training budget and randomness.
- `H_1`: Exposure-Debt Ghost Bank improves minority-class performance relative to the strong baseline and static memory variants, while preserving overall performance within an acceptable tolerance.

For a minority-sensitive metric `M_min` and an overall metric `M_all`, one formal statement is:

`E[M_min(theta_gb)] <= E[M_min(theta_base)]`

under `H_0`, and

`E[M_min(theta_gb)] > E[M_min(theta_base)]`

and

`E[M_min(theta_gb)] > E[M_min(theta_static_bank)]`

with the side constraint

`E[M_all(theta_gb)] >= E[M_all(theta_base)] - epsilon`

under `H_1`, for a small `epsilon >= 0`.

The hypothesis is intentionally conservative. The method should be treated as a candidate mechanism for improving minority-class learning, not as a presumed success. The static-bank comparison is mandatory because it tests whether exposure-debt control is a real contribution.
