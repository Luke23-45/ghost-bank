# Method Specification

## Method Name

Exposure-Debt Ghost Bank, abbreviated as `ED-GB`.

## Definitions

Let `C_min` be the set of minority classes.

Let `A_c(t)` be the accumulated effective exposure for class `c` up to step `t`.

Let `T_c(t)` be the target exposure for class `c` up to step `t`.

The exposure debt for class `c` is:

`D_c(t) = max(0, T_c(t) - A_c(t))`

The bank retrieval budget for class `c` is proportional to its exposure debt:

`r_c(t) = floor(R * D_c(t) / sum_j D_j(t))`

where `R` is the total number of bank entries retrieved at step `t`.

## Exposure Signal

The simplest exposure signal is count-based:

`a_c(t) = sum_{(x,y) in B_t} 1[y = c]`

The accumulated exposure is:

`A_c(t) = A_c(t - 1) + a_c(t)`

For a stronger version, exposure can be loss-weighted:

`a_c(t) = sum_{(x,y) in B_t} 1[y = c] * ell(f_theta(x), y)`

The preliminary experiment should start with count-based exposure because it is simple and stable. Loss-weighted or gradient-norm-weighted exposure can be tested as an ablation.

## Target Exposure

The target exposure can be uniform across classes:

`T_c(t) = t * beta`

where `beta` is the desired per-class exposure rate.

Alternatively, the target can use a smoothed class prior:

`T_c(t) = t * q_c`

where `q_c` is a target distribution over classes, such as a uniform distribution or a square-root-smoothed distribution.

For the first paper version, use a uniform target over minority classes and leave smoothed targets as an ablation.

## Bank State

Each bank item should contain:

- sample identifier
- class label
- stored representation
- loss or hardness score
- insertion step

For preliminary implementation, the stored representation can be a raw sample reference. For a stronger method, the bank can store both raw sample references and detached embeddings.

## Bank Update Rule

For each minority-class example observed in a minibatch:

1. Compute its current loss or confidence.
2. Insert it into the class-specific bank if the class bank is not full.
3. If the class bank is full, replace an existing item only if the new item improves diversity or hardness coverage.

The bank should avoid storing only the hardest examples, because that can preserve mislabeled or noisy samples. Use a mixture of hard examples and diverse examples.

## Retrieval Rule

At step `t`, retrieve entries from class `c` according to `r_c(t)`.

Within each class, prioritize:

1. high exposure debt
2. feature diversity
3. moderate-to-high loss
4. recency control, so the bank does not become stale

The retrieved bank set is `R_t`.

## Training Loss

The total loss is:

`L_total = L_base(B_t) + lambda * L_bank(R_t)`

For preliminary experiments:

`L_bank` can be cross-entropy on retrieved raw samples.

For stronger experiments:

`L_bank` can combine cross-entropy with a prototype consistency term:

`L_bank = L_ce + alpha * L_proto`

where `L_proto` encourages retrieved minority features to remain close to their class prototype and separated from competing prototypes.

## Training Schedule

Use a deferred schedule:

1. Warmup stage: train the representation with the baseline method.
2. Ghost-bank stage: activate exposure-debt retrieval and bank loss.

This follows the empirical lesson from decoupled long-tailed recognition: early representation learning can be harmed by aggressive balancing, while later classifier or tail correction can be beneficial.

## Mandatory Ablations

The method must be compared against:

1. Baseline cross-entropy.
2. Class-balanced loss.
3. Focal loss.
4. LDAM-DRW.
5. Logit adjustment.
6. Static minority replay bank.
7. Static prototype bank if features are used.
8. Exposure-Debt Ghost Bank.

The critical ablation is:

`static bank vs exposure-debt controlled bank`

This is the ablation that proves whether the proposed contribution is real.
