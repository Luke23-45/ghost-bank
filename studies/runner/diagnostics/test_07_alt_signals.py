"""H1: Loss is the wrong control signal — compare alternatives.

Hypothesis: Per-class cross-entropy loss is a poor control signal
because after a label swap, BOTH swapped classes show equally high
loss.  The controller can't distinguish which class needs more help.

Alternative signals to evaluate:
  1. Cross-entropy loss (current) — [L_c]
  2. Prediction confidence (gap between top-1 and top-2 logits) — [conf_c]
  3. Forgetting events (prediction flips from correct→incorrect)
  4. Gradient norm (||dL/dW|| per class)
  5. Prediction entropy (uncertainty)
  6. Accuracy (1 - acc)
  7. Loss * (1 / class_freq) — frequency-adjusted loss
  8. Dataless gradient estimate (cosine similarity of class means)

This script compares these signals analytically by computing what
each would look like in the post-shift scenario.
"""

import math

import numpy as np


def confidence_signal(logits, targets):
    """Return 1 - (prob of correct class) per sample."""
    probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = probs / probs.sum(axis=1, keepdims=True)
    # Get probability of correct class
    correct_probs = probs[np.arange(len(targets)), targets]
    return 1.0 - correct_probs


def entropy_signal(logits):
    """Return per-sample prediction entropy."""
    probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = probs / probs.sum(axis=1, keepdims=True)
    return -np.sum(probs * np.log(probs + 1e-10), axis=1)


def margin_signal(logits):
    """Return 1 - (top1_prob - top2_prob). Low margin = uncertain."""
    probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = probs / probs.sum(axis=1, keepdims=True)
    sorted_probs = np.sort(probs, axis=1)[:, ::-1]
    margins = sorted_probs[:, 0] - sorted_probs[:, 1]
    return 1.0 - margins


def accuracy_signal(predictions, targets):
    """Return 1 - acc (per class)."""
    return (predictions != targets).astype(float)


def gradient_norm_signal(logits, targets, features):
    """Estimate per-class gradient norm from cross-entropy."""
    # Gradient of CE w.r.t. logits = softmax - one_hot
    probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = probs / probs.sum(axis=1, keepdims=True)
    one_hot = np.eye(logits.shape[1])[targets]
    dlogits = probs - one_hot  # [N, C]
    # Gradient norm approximated as |dL/dz| (norm of logit gradient)
    grad_norms = np.linalg.norm(dlogits, axis=1)
    return grad_norms


def compute_per_class_signals(signals, class_ids, num_classes=3):
    """Aggregate per-sample signals into per-class means."""
    per_class = {}
    for c in range(num_classes):
        mask = class_ids == c
        if mask.sum() > 0:
            per_class[c] = float(np.mean(signals[mask]))
        else:
            per_class[c] = 0.0
    return per_class


def simulate_post_shift_model_output(num_samples_poshift=1000, seed=42):
    """Simulate model outputs after label swap.

    After swap (0↔2):
    - Inputs originally from class 0 now have target=2
    - Model was trained on pre-swap data, so it predicts old labels
    - For class 0 inputs: model predicts ~0 (old mapping), target=2 (new)
    - For class 2 inputs: model predicts ~2 (old mapping), target=0 (new)
    - Class 1: unchanged
    """
    rng = np.random.RandomState(seed)

    # Generate logits for each class
    # Class 0 inputs: model still predicts class 0 (logits = [2.5, 0.0, 0.2])
    # Class 2 inputs: model still predicts class 2 (logits = [0.2, 0.0, 2.5])
    # Class 1 inputs: model predicts class 1 (logits = [0.0, 2.5, 0.0])

    results = []
    class_configs = [
        (0, 2, [2.5, 0.0, 0.2]),   # class 0 input, target=2
        (1, 1, [0.0, 2.5, 0.0]),   # class 1 input, target=1
        (2, 0, [0.2, 0.0, 2.5]),   # class 2 input, target=0
    ]

    for input_class, target, mean_logits in class_configs:
        for _ in range(num_samples_poshift // 3):
            logits = np.array(mean_logits) + rng.randn(3) * 0.5
            results.append((input_class, target, logits, rng.randn(2)))  # features

    rng.shuffle(results)
    return results


if __name__ == "__main__":
    print("=" * 70)
    print("  ALTERNATIVE CONTROL SIGNALS (H1)")
    print("=" * 70)

    samples = simulate_post_shift_model_output()

    inputs = np.array([s[0] for s in samples])
    targets = np.array([s[1] for s in samples])
    logits = np.array([s[2] for s in samples])
    preds = np.argmax(logits, axis=1)

    print(f"\n  Simulated {len(samples)} post-shift samples")
    print(f"  Label swap: class 0 ↔ class 2")

    # Compute all signals
    signals = {
        "Cross-entropy loss": np.array([
            -np.log(np.exp(logits[i, targets[i]]) / np.sum(np.exp(logits[i])))
            for i in range(len(logits))
        ]),
        "Accuracy (1-acc)": accuracy_signal(preds, targets),
        "Prediction confidence (1-p_correct)": confidence_signal(logits, targets),
        "Prediction margin (1-margin)": margin_signal(logits),
        "Prediction entropy": entropy_signal(logits),
        "Gradient norm": gradient_norm_signal(logits, targets, inputs),
    }

    # Compute per-class means
    print(f"\n  {'Signal':<40} {'Class 0':>10} {'Class 1':>10} {'Class 2':>10} {'C0/C2 ratio':>15}")
    print("  " + "-" * 85)

    for name, signal in signals.items():
        per_class = compute_per_class_signals(signal, inputs, num_classes=3)
        c0 = per_class[0]
        c1 = per_class[1]
        c2 = per_class[2]
        ratio = c0 / c2 if c2 > 0 else float('inf')
        print(f"  {name:<40} {c0:>10.4f} {c1:>10.4f} {c2:>10.4f} {ratio:>14.4f}")

    # Which signal best distinguishes struggling class 2 from adapted class 0?
    print("\n" + "-" * 70)
    print("  SIGNAL DISCRIMINATION: Can the controller tell class 0 and 2 apart?")
    print("-" * 70)
    print("""
    After the swap, class 0 (majority, 2000 samples) adapts quickly.
    Class 2 (minority, 20 samples) adapts slowly.

    A good control signal has:
      - HIGH value for class 2 (needs replay)
      - LOW value for class 0 (already adapted)
      - Large C2/C0 ratio

    Current cross-entropy loss has C0 ≈ C2 initially → can't discriminate.
    """)

    # Rank signals by C2/C0 ratio
    print(f"  {'Rank':>6} {'Signal':<40} {'C2/C0 Ratio':>15}")
    print("  " + "-" * 61)
    ranked = []
    for name, signal in signals.items():
        per_class = compute_per_class_signals(signal, inputs, num_classes=3)
        c0 = per_class[0]
        c2 = per_class[2]
        ratio = c2 / c0 if c0 > 0 else float('inf')
        ranked.append((ratio, name, c0, c2))
    ranked.sort(key=lambda x: x[0], reverse=True)

    for rank, (ratio, name, c0, c2) in enumerate(ranked, 1):
        print(f"  {rank:>6} {name:<40} {ratio:>14.4f}")

    # --- Per-class accuracy confusion ---
    print("\n" + "-" * 70)
    print("  CONFUSION MATRIX (what does the model predict post-shift?)")
    print("-" * 70)
    confusion = np.zeros((3, 3), dtype=float)
    for i in range(len(inputs)):
        confusion[targets[i], preds[i]] += 1
    row_sums = confusion.sum(axis=1, keepdims=True)
    confusion_pct = np.divide(confusion, row_sums, where=row_sums > 0) * 100
    print(f"  Actual ↓ / Pred →")
    print(f"  {'':>8} {'Pred 0':>10} {'Pred 1':>10} {'Pred 2':>10}")
    for c in range(3):
        print(f"  {'True ' + str(c):>8} {confusion_pct[c, 0]:>9.1f}% {confusion_pct[c, 1]:>9.1f}% {confusion_pct[c, 2]:>9.1f}%")

    # --- VERDICT ---
    print("\n\n" + "=" * 70)
    print("  VERDICT")
    print("=" * 70)
    print("""
  H1 (Loss is the wrong signal):
    If ALL signals show similar C0 ≈ C2 values, then no signal can
    discriminate post-shift — the problem is not the signal but the
    fundamental symmetry of the swap (both classes look equally
    confused to the model).

    If some signals show C2 >> C0, then we could improve PID-CR by
    switching to that signal.  Candidates like 'accuracy' or 'margin'
    might differentiate because class 0 has more data → adapts faster.
""")
