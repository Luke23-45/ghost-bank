Here's a synthesis of the literature search. I've found where our methods fit, and it's actually quite interesting.

## Where ED-GB / PID-GB Fit in the Literature

The landscape of replay-based CL has **two independent decisions**:

| Decision | What it is | Existing work |
|---|---|---|
| **WHAT to store** | Buffer update policy | Reservoir, CBRS, KLRS (NeurIPS'22), Rainbow Memory (CVPR'21), GDumb (ECCV'20), CORE (2024), RMM (ICLR'23) |
| **HOW MANY to retrieve** | Per-class retrieval allocation | **Almost no one addresses this** — nearly all use uniform retrieval |

The literature is filled with sophisticated **storage** strategies but almost everyone retrieves uniformly. The few exceptions:

- **MIR** (NeurIPS'19): retrieves samples with maximal loss increase — focuses on *which samples*, not *how many per class*
- **AdaER** (2023): contextually-cued recall — again, *which samples*
- **CORE** (2024): allocates buffer space by forgetting rate — this is **storage**, not retrieval
- **RMM** (ICLR'23): RL-based per-class memory allocation — again, **storage** allocation
- **"Watch Your Step"** (2024): specifically studies selective **retrieval** — class-selective and sample-selective primitives, but tested on pretrained object detection models

**ED-GB is genuinely novel**: it's the first method to dynamically adjust per-class **retrieval quantity** based on class exposure. The key distinction from all storage-side methods is that ED-GB controls the *replay distribution at training time*, not the *buffer composition*.

**PID-GB extends this** with control-theoretic loss feedback. CORE's forgetting-rate-based allocation is the closest analogue, but CORE operates on storage, not retrieval.

## What This Means for Our Benchmark

The standard bench in every paper:
- **Split CIFAR-100** (10 tasks × 10 classes, Class-IL)
- Buffer 2000, ResNet-18/32
- Compare: ER, GDumb, DER++, iCaRL, MIR, ER-ACE
- Metric: average accuracy across ALL classes after each task, final avg accuracy

**Our contribution**: show that uniform retrieval is suboptimal under imbalanced forgetting, and ED-GB/PID-GB's adaptive retrieval allocation consistently beats uniform retrieval across standard benchmarks.

---

What would you like to do? We have enough grounding now to:
1. **Design the experimental protocol** based on the standard Split CIFAR-100 Class-IL benchmark
2. **Implement the necessary infrastructure** (task-based dataset splitting, multi-task training loop, per-class metrics)
3. **Run the comparison** against uniform retrieval with the same storage strategy (isolating the retrieval variable)
4. These are **research designs** — we can discuss and decide without writing code yet