# Formal Mathematical Definition

This document defines the ghost bank idea in a mechanism-agnostic way. The goal is to formalize the concept without committing to a specific implementation before the experiments are run.

The term "ghost bank" is a working project name for a persistent auxiliary memory that stores minority-class information in some retrievable form. It may store raw examples, features, prototypes, or other compressed representations. The mathematics below is written so it remains valid across those design choices.

## 1. Data and Model

Let the labeled training dataset be

`D = {(x_i, y_i)}_{i=1}^N`

where:

- `x_i` is an input sample
- `y_i \in Y = {1, 2, ..., K}` is its class label
- `K` is the number of classes

Let `n_c` be the number of training examples in class `c`, so that:

`sum_{c=1}^K n_c = N`

We say the dataset is imbalanced when the class counts are highly unequal. A common imbalance ratio is:

`rho = max_c n_c / min_c n_c`

where larger `rho` means stronger imbalance.

Let the model be parameterized by `theta` and define:

`f_theta : X -> R^K`

where `f_theta(x)` returns the class logits. The predicted class distribution is:

`p_theta(y | x) = softmax(f_theta(x))_y`

Let `ell(p, y)` be a standard supervised loss, such as cross-entropy.

## 2. Standard Training Objective

Under ordinary empirical risk minimization, the objective is:

`L_base(theta) = (1/N) sum_{i=1}^N ell(f_theta(x_i), y_i)`

When minibatch stochastic gradient descent is used, the gradient estimate at step `t` is computed from a minibatch `B_t \subset D`.

Because minibatches are drawn from the empirical data distribution, a minority class with small `n_c` contributes fewer updates in expectation. This is the optimization imbalance that the ghost bank is intended to address.

## 3. Ghost Bank State

Define a ghost bank at iteration `t` as a memory structure:

`G_t = {g_j}_{j=1}^{m_t}`

where each stored element `g_j` carries at least:

- a value representation `v_j`
- an associated label `y_j`

Depending on the implementation, `v_j` may be:

- a raw example `x`
- a latent feature vector
- a prototype
- another stored representation derived from training

The formalism below does not require a specific choice of `v_j`.

## 4. Query and Update Operators

Let the minibatch at step `t` be:

`B_t = {(x_i, y_i)}_{i=1}^{b_t}`

Let the bank retrieval operator be:

`R_t = Q(G_t, B_t, theta_t)`

where:

- `Q` selects a subset or transformation of the bank
- `R_t` is the bank-sourced auxiliary set used at step `t`

Let the bank update operator be:

`G_{t+1} = U(G_t, B_t, theta_t)`

where `U` defines how the bank is refreshed, replaced, compressed, or expanded after observing the current batch.

## 5. Ghost Bank Training Objective

A generic ghost bank objective can be written as:

`L_gb(theta; B_t, G_t) = L_sup(theta; B_t) + lambda * L_bank(theta; R_t)`

where:

`L_sup(theta; B_t) = (1 / |B_t|) sum_{(x,y) in B_t} ell(f_theta(x), y)`

and `L_bank` is an auxiliary term computed from the bank-retrieved set `R_t`.

`lambda >= 0` controls the strength of the ghost-bank contribution.

The parameter update is then:

`theta_{t+1} = theta_t - eta_t * nabla_theta L_gb(theta_t; B_t, G_t)`

where `eta_t` is the learning rate.

## 6. Minority-Class Exposure Interpretation

Let `C_min \subseteq Y` be the set of minority classes. Define the expected minority exposure per training step as the expected number of minority-labeled items used in optimization.

Without a bank, this exposure is determined by the sampling distribution of `B_t`.

With a ghost bank, if the retrieval policy increases the probability that items from `C_min` appear in `R_t`, then the effective minority exposure becomes:

`E[exposure_min^gb] = E[exposure_min^batch] + E[exposure_min^bank]`

where `E[exposure_min^bank] >= 0`.

If the bank is designed so that minority examples are retrieved more often than they would be under plain empirical sampling, then the model receives more minority-class optimization signal over the same number of updates.

This is the formal sense in which the ghost bank helps: it increases the effective presence of rare-class information in the training objective.

## 7. Exposure Debt

A stronger ghost bank should not retrieve entries only because a class is rare globally. It should retrieve entries because the class is currently underexposed relative to a target schedule.

Let `a_c(t)` be the effective exposure received by class `c` at step `t`. The simplest count-based version is:

`a_c(t) = sum_{(x,y) in B_t} 1[y = c]`

The accumulated exposure is:

`A_c(t) = sum_{s=1}^t a_c(s)`

Let `T_c(t)` be the target exposure for class `c` at step `t`. The exposure debt is:

`D_c(t) = max(0, T_c(t) - A_c(t))`

The retrieval budget for class `c` can then be assigned as:

`r_c(t) = floor(R * D_c(t) / sum_{j in C_min} D_j(t))`

where `R` is the total bank retrieval budget.

This turns the bank into a closed-loop controller: the bank retrieves minority information in response to measured underexposure, not only because a class has few samples in the dataset.

## 8. Relation to Existing Methods

The ghost bank is not mathematically equivalent to every existing imbalance method:

- If `Q` retrieves raw stored samples, the method resembles a memory-augmented replay or tail-enrichment strategy.
- If `Q` retrieves embeddings or prototypes, the method resembles a prototype bank or memory-based classifier support.
- If `L_bank` only reweights class contributions, the method becomes close to class-balanced reweighting.
- If `Q` changes the class composition of minibatches, the method partially overlaps with resampling.
- If `Q` is controlled by exposure debt, the method becomes a dynamic memory controller rather than a static bank.

The research contribution therefore depends on the exact choice of `Q`, `U`, and `L_bank`, and on whether the resulting optimization improves minority-class metrics beyond strong baselines.

## 9. Research Claim

The research claim can be stated as:

For sufficiently imbalanced data, there exists an exposure-debt controlled ghost bank design such that training with `L_gb` improves minority-class performance relative to strong baselines and static memory variants, while preserving overall performance within an acceptable tolerance.

This claim is what the experiments must test.
