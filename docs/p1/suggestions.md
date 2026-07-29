Looking at this report, my honest read as your advisor: **you haven't actually found a dead end yet — you've found a dead end for one specific, non-standard experimental setup.** Before you write the "fundamental paradox" paper, there are several confounds in the design that would make a good reviewer (or your future self) throw this back at you. Let me walk through why, then give you a concrete, cheap path to a real answer.

## What's genuinely good here

Credit where due: fixing FC-freezing as a variable, isolating KL-vs-BCE with the α=0 ablation, and the honest cost/failure-mode logging are exactly what rigorous negative-result work should look like. Most people don't do this. The problem isn't the effort — it's that a few design choices upstream of all nine architectures may have guaranteed the "paradox" before a single line of PID code ran.

## The five issues that need to be resolved before you can trust "dead end"

**1. Your probe measures memorization, not forgetting — this is probably the whole story.**
The ghost-bank probe computes loss on the *same exemplars* used for replay. For any method that replays those exemplars in the gradient (iCaRL, PID-iCaRL, PID-GDR), of course the probe finds near-zero debt — the model is directly optimized on exactly those images every batch. That's not evidence of "no forgetting," it's evidence you're grading a model on its own training set. This is a well-documented failure mode in the replay literature: buffer/memory overfitting, where a model fits its tiny exemplar set almost perfectly while true generalization to the class still degrades. Several recent papers build entire methods around exactly this gap. Your finding that debt is near-zero specifically in the replay-using methods, and non-trivial specifically in the no-replay methods (DRKD, PID-DDC), is the signature of this artifact, not of a "measurement paradox." Fix: hold out a separate probe set per class (e.g., 30–50 images never placed in the replay buffer) and recompute debts. If debts stay near zero even on held-out data, the paradox is real. If they don't, you've been measuring the wrong thing for 45,000 GPU-minutes.

**2. Your memory budget removes the exact scarcity the controller is supposed to arbitrate.**
iCaRL's actual protocol (I checked the original paper to be sure) uses a *fixed total* budget K, with per-class allocation m = K/t that *shrinks* as more classes accumulate — so class 0 gets squeezed hard by task 10. Your setup fixes 200/class forever, growing the total unboundedly to 20,000. That's generous even by inflated standards in the literature (the original paper used 20/class; other work shows returns flatten well before 200). Under your protocol there's no real competition for storage — everyone always has plenty. A controller whose job is "who deserves more of the scarce resource" has nothing to arbitrate when the resource isn't scarce. This is a second, independent reason the debt signal could look flat.

**3. Your "no replay" experiments are confounded with frozen FC — the ~19% ceiling claim isn't isolated.**
DRKD and PID-DDC combine *no replay in gradient* **and** *frozen old-class FC rows*. But you already showed, twice, that frozen FC alone is catastrophic even *with* replay (FC-GM: 1.8% vs 37.4%; PID-GDR frozen vs trainable: 17.4% vs 32.5%, a 15-point swing from FC alone). So when DRKD lands at 18.1%, you don't actually know how much of that gap is "no replay" versus "frozen FC" — you never ran the one experiment that would tell you: decoupled distillation **with a trainable FC**. Section 8.2's claim that decoupled methods are "fundamentally capped at ~19%" is not yet supported by an isolated test.

**4. The PID looks untuned, and "neglect cycle" reads like a control-engineering bug, not a proof of concept failure.**
Oscillation and starvation from proportional-ish feedback on a noisy, small-sample signal (loss over 200 exemplars) is the textbook symptom of an unfiltered derivative term and/or no anti-windup on the integral term — not evidence that feedback control can't work here. There's no mention of gain tuning, output clamping, EMA-smoothing the debt signal, or a minimum allocation floor per class. Any of these standard fixes might turn "neglect cycle" into "converges."

**5. Single run per config, no seeds, no class-order variation.**
Several of your headline deltas (35.0% vs 37.4%, 32.5% vs 30.5%) are within the kind of run-to-run noise class-incremental learning is known for. Before "KL is 2.4% worse than BCE" becomes a claim in a paper, it needs 3+ seeds.

There's also the acknowledged 5-point bug in the iCaRL baseline itself — if a bug that size slipped into the reference implementation, I wouldn't trust any of the other eight without an audit for the same class of off-by-one error.

## So: dead end, or not?

Neither of your two framings is quite right. The correct third option is: **you don't know yet, and finding out is cheap relative to what you've already spent.** Here's the decision process I'd run:## The pilot: what I'd actually run

You don't need to repeat the full 47,000-minute sweep. A decisive pilot is a fraction of that, because you're not trying to publish yet — you're trying to find out if the phenomenon is real. I'd scope it to 3–4 configs, at reduced scale (e.g. 5 tasks instead of 10, 20–30 epochs/task instead of 70), with these fixes applied simultaneously:

1. **Fix the exemplar-storage bug first** and audit the other seven implementations for the same class of off-by-one error before trusting any comparison.
2. **Held-out probe set.** Reserve extra per-class images that never enter the replay buffer; recompute all debts against those instead of the buffer.
3. **Standard shrinking-budget memory.** Fixed total K (say 2,000, matching the literature), m = K/t per class, instead of a flat 200/class. This is where real memory pressure — and a real reason to prefer one class over another — would show up.
4. **A properly isolated no-replay condition.** DRKD-style decoupling *with a trainable FC*, so you can finally tell whether the ~19% ceiling is about replay or about the frozen classifier.
5. **A minimally sane PID**: small gain grid, EMA-smoothed debt signal, integral clamping, and a floor so no class's allocation goes to zero. Also throw in one non-adaptive baseline — a fixed recency-weighted schedule (older classes get systematically more weight, no feedback loop at all). If a tuned PID can't beat that simple static heuristic, that's a real strike against the adaptive-control framing specifically, independent of the other fixes.
6. **3 seeds minimum** on whichever 2–3 configs come out on top, before any comparative claim.

Rough budget: 6–10 short runs, on the order of 5,000–10,000 GPU-minutes total — a fifth of what's already spent, for a conclusion you can actually defend.

**The decision rule:** if after all five fixes the debts are still near-zero and no variant beats a well-tuned static baseline, you now have a genuinely rigorous negative result — confounds ruled out, not just asserted — and that's a much stronger paper than the current draft (reviewers in this area will ask exactly the five questions above; better to have already answered them). If the signal appears under any of these corrected conditions, you've found the actual regime where ghost-bank-style sensing has a job to do, and that's a new, narrower, more defensible research direction — not a retreat from the original idea, a correction to where it applies.

If it's useful, I can help you spec the exact pilot configs (gains, held-out probe sizes, the shrinking-budget schedule) or start drafting the negative-result writeup in parallel, since you'll want that either way.