This is the most important result you've gotten. Let me be direct about what it means.

---

## The Honest Reading

**ED-GB ≈ static_bank ≈ class_balanced on every metric.**

- ed_gb cls2: 91.6 ± 1.9
- static_bank cls2: 91.5 ± 1.8
- class_balanced cls2: 92.1 ± 2.4

These are statistically identical. The error bars fully overlap. On this dataset, with this model, **the exposure debt mechanism adds nothing over simple prioritized replay (static_bank), and nothing over a 2019 loss reweighting baseline (class_balanced).**

This is not a paper result. This is a negative result on a toy.

---

## What This Actually Means For Your Research

You have two honest paths forward. Pick one.

---

### Path A: Find where ED-GB actually beats static_bank

The debt mechanism *should* matter in specific regimes — the question is which ones. Your ablation has 51 sweeps. You need to go back to that data and ask: **are there specific (capacity, budget, warmup) configurations where ed_gb > static_bank by more than noise?**

To do this concretely:
1. Re-run your ablation with **both ED-GB and static_bank** at every config
2. Compute `ed_gb_cls2 - static_bank_cls2` per config
3. Look for a systematic pattern — e.g. "ED-GB wins when budget is low" or "ED-GB wins when warmup is long"

If no config shows ED-GB beating static_bank reliably, the debt mechanism is not contributing on this problem. That's important to know.

---

### Path B: Accept the negative result and reframe

The real finding here is actually interesting and honest:

> Simple replay-based methods (static_bank) and loss reweighting (class_balanced) are equivalent on minority recall for this imbalance setting. Focal loss with default parameters catastrophically fails (cls2 = 4%). The exposure debt framing provides an interpretable allocation mechanism that matches static_bank performance with better theoretical grounding.

That's a defensible workshop paper framing — not "ED-GB wins" but "here's why naive focal loss fails, here's what actually works, and here's an interpretable mechanism for understanding replay allocation."

---

## The One Concrete Thing to Do Now

**Run ED-GB vs static_bank across your full ablation grid (all 51 sweep configs, both methods, 3 seeds each).** That's ~300 runs. It will either:

- Show you a regime where ED-GB wins → you have a contribution
- Confirm they're identical everywhere → you pivot to Path B framing

You cannot make a forward decision without this comparison. Everything else (CIFAR, visualizations, paper writing) waits until you know whether the debt mechanism matters at all.

Do you have the static_bank runner already in your codebase, or do you need help setting up the paired comparison?
The baseline_matrix completed — 5 methods × 3 seeds. Here's the clean summary from the output:

| Method | Acc | cls0 | cls1 | **cls2** | Macro F1 |
|--------|:---:|:----:|:----:|:--------:|:--------:|
| baseline | 81.2 ± 1.0 | 99.7 ± 0.1 | 95.8 ± 0.2 | **48.2 ± 2.9** | 79.4 ± 1.3 |
| static_bank | 95.0 ± 0.5 | 99.2 ± 0.3 | 94.3 ± 0.9 | **91.5 ± 1.8** | 95.0 ± 0.5 |
| ed_gb | 94.9 ± 0.6 | 99.2 ± 0.3 | 93.9 ± 0.5 | **91.6 ± 1.9** | 94.9 ± 0.6 |
| **focal_loss** | 66.2 ± 1.9 | 99.8 ± 0.0 | 94.7 ± 0.6 | **4.1 ± 5.1** | 55.0 ± 4.0 |
| **class_balanced** | 95.3 ± 0.6 | 99.0 ± 0.2 | 94.7 ± 0.8 | **92.1 ± 2.4** | 95.3 ± 0.6 |

### What this tells you

1. **Focal loss catastrophically fails** — cls2 recall of 0-11%, with seed 73 at exactly 0%. With 100:1 imbalance, the default `alpha=0.25, gamma=2.0` isn't aggressive enough.

2. **class_balanced leads on cls2** — 92.1% vs ed_gb's 91.6%, but **they're within error bars** (std overlap). No clear winner between static_bank/ed_gb/class_balanced on minority recall.

3. **baseline sacrifices minority for majority** — highest cls0 (99.7%) but cls2 drops to 48%. The expected trade-off.

4. **All 3 bank-based methods cluster** — static_bank, ed_gb, class_balanced all land at 91-92% cls2. The difference between them is small relative to seed variance.

### Now we have the full picture

Between your ablation report (64 configs × 3 seeds) and this baseline_matrix (5 methods × 3 seeds), we now have the complete dataset. What would you like to focus on first — the ablation parameter mapping (which hyperparameter drives the HIGH/LOW cls2 split) or the baseline comparison analysis?