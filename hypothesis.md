# Hypothesis

## Research claim

Under a fixed backbone, fixed task split, fixed optimizer, and fixed total replay budget, **herding-based fixed-total replay** (`uniform_herding`) will produce a more representative memory than static per-class replay, and therefore achieve **higher average accuracy** on class-incremental CIFAR-100.

## Primary hypothesis

If replay budget is scarce, then exemplar **selection quality** matters more than naive storage policy:

- `uniform_herding` should outperform `static_bank` on average accuracy.
- The gain should come from better buffer representativeness, not from changing the model architecture or training schedule.
- The strongest effect should be visible in final average accuracy and NME-style evaluation, with moderate changes in forgetting.

## Secondary hypothesis

Full iCaRL-style exemplar replay is expected to reduce forgetting more than `uniform_herding`, but it may not improve average accuracy in this codebase. This means the main result is likely a **stability-plasticity tradeoff**, not a universal SOTA win.

## Null hypothesis

If the multi-seed confirmation sweep does not show a consistent gain for `uniform_herding` over `static_bank` and probe-guided variants, then the probe signal is not a reliable replay-allocation signal in this setting. In that case, the correct paper claim is a **negative result / tradeoff analysis**, not a stronger method claim.

## Success criteria

The hypothesis is supported if:

1. `uniform_herding` improves mean average accuracy over the fixed-budget baselines across multiple seeds.
2. The effect survives the same training schedule, same memory budget, and same backbone.
3. Probe-guided allocation does not consistently beat `uniform_herding`.

## References

- iCaRL: [Rebuffi et al., 2017](https://openaccess.thecvf.com/content_cvpr_2017/papers/Rebuffi_iCaRL_Incremental_Classifier_CVPR_2017_paper.pdf)
- BiC: [Wu et al., 2019](https://arxiv.org/pdf/1905.13260)
- DER++: [Buzzega et al., 2020](https://proceedings.neurips.cc/paper/2020/file/b704ea2c39778f07c617f6b7ce480e9e-Paper.pdf)
