# Manuscript Structure

## Scope

This document is the source of truth for the manuscript outline.

- It is written to be venue-agnostic at the section level.
- If the target venue is NeurIPS, ICML, or ICLR, the main paper should follow the compact "paper + references + appendix" format.
- For NeurIPS-style submissions, the checklist is a required submission component, but it is not a narrative section of the paper.

## Main Paper

1. Title
2. Abstract
3. Introduction
   - Problem setup: class-incremental learning on CIFAR-100
   - Why replay and memory selection matter
   - What the paper studies: reference method, baselines, ablations, and resource sensitivity
   - Main claims and contributions
4. Related Work
   - Class-incremental learning
   - Replay and exemplar selection
   - Distillation and classifier/readout choices
   - Forgetting and stability metrics
5. Method and Experimental Setup
   - Dataset and task protocol
   - Backbone and training setup
   - Memory bank and retrieval budgets
   - Reference method definition
   - Baselines and ablations
   - Evaluation protocol
     - NME vs head-logit
     - Accuracy, forgetting, and BWT
   - Statistical setup
     - Seeds
     - Mean +/- std
     - Matched per-seed deltas for ablations
6. Results
   - Main comparison
     - Baseline vs reference
     - Use Table T1 and Figure 1
   - Component attribution
     - KD, readout choice, head type, and selection rule
     - Use Table T2 and Figure 2
   - Resource sensitivity
     - Memory budget and retrieval budget
     - Use Table T3 and Figure 3
   - Trade-off summary
     - Accuracy vs forgetting
     - Use Figure 4
   - Failure-mode analysis
     - Forgetting by task age
     - Use Figure 5
7. Discussion
   - What actually drives performance
     - NME/readout pairing
     - Herding vs random selection
     - KD as stability, not accuracy
   - Why memory matters more than retrieval
   - What the results imply for method design
   - Limitations
     - One dataset
     - Three seeds
     - No saturation check beyond the current memory range
   - Practical recommendation
8. Conclusion
   - One-paragraph summary of the main result
   - Final takeaway on replay, readout, and memory budget
9. References

## Appendix

The appendix follows the references.

1. Protocol and reproducibility details
   - [A1_protocol.md](C:/Users/Hellx/Documents/Programming/python/Project/iron/ghost-bank/studies/analysis/outputs/paper/tables/A1_protocol.md)
2. Per-task accuracies
   - [A2_per_task_accuracies.md](C:/Users/Hellx/Documents/Programming/python/Project/iron/ghost-bank/studies/analysis/outputs/paper/tables/A2_per_task_accuracies.md)
3. Per-task forgetting
   - [A3_per_task_forgetting.md](C:/Users/Hellx/Documents/Programming/python/Project/iron/ghost-bank/studies/analysis/outputs/paper/tables/A3_per_task_forgetting.md)
4. Per-seed metrics
   - [A4_per_seed_metrics.md](C:/Users/Hellx/Documents/Programming/python/Project/iron/ghost-bank/studies/analysis/outputs/paper/tables/A4_per_seed_metrics.md)
5. Compute cost
   - [A5_compute_cost.md](C:/Users/Hellx/Documents/Programming/python/Project/iron/ghost-bank/studies/analysis/outputs/paper/tables/A5_compute_cost.md)
6. Exemplar bank sizes
   - [A6_bank_sizes.md](C:/Users/Hellx/Documents/Programming/python/Project/iron/ghost-bank/studies/analysis/outputs/paper/tables/A6_bank_sizes.md)
7. Appendix figures
   - Forgetting heatmap
   - Evolution plots
   - Stability slopes

## Venue-Specific Notes

- NeurIPS-style order: main paper, references, appendix, checklist.
- ICML/ICLR-style order is similar, but checklist handling depends on the venue template.
- Keywords are omitted unless the target venue explicitly asks for them.
- Broader impact text should be included if the venue requires it or if the work has a clear societal risk or benefit to discuss.

## Recommended Narrative Order

1. Establish the problem and protocol.
2. Show the main ranking.
3. Explain which components matter.
4. Show how budget changes the result.
5. Close with the trade-off and failure modes.

## Final Working Outline

If the goal is a single finalized paper outline, use this order:

1. Title
2. Abstract
3. Introduction
4. Related Work
5. Method and Experimental Setup
6. Results
7. Discussion
8. Conclusion
9. References
10. Appendix
11. Submission checklist
    - Venue artifact, not a narrative section of the paper
