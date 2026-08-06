

**Front Matter**
1. Title
2. Abstract
3. Keywords, if required by the venue

**1. Introduction**
1. Problem setup: class-incremental learning on CIFAR-100
2. Why replay/memory selection matters
3. What the paper studies: reference method, baselines, ablations, resource sensitivity
4. Main claims and contributions

**2. Related Work**
1. Class-incremental learning
2. Replay and exemplar selection
3. Distillation and classifier/readout choices
4. Forgetting and stability metrics

**3. Method / Experimental Setup**
1. Dataset and task protocol
2. Backbone and training setup
3. Memory bank and retrieval budgets
4. Reference method definition
5. Baselines and ablations
6. Evaluation protocol
   - NME vs head-logit
   - Accuracy, forgetting, BWT
7. Statistical setup
   - seeds
   - mean ± std
   - matched per-seed deltas for ablations

**4. Results**
1. Main comparison
   - baseline vs reference
   - use [Table T1](C:/Users/Hellx/Documents/Programming/python/Project/iron/ghost-bank/studies/analysis/outputs/paper/tables/T1_master_results.md)
   - use [Figure 1](C:/Users/Hellx/Documents/Programming/python/Project/iron/ghost-bank/studies/analysis/outputs/paper/main/figures/fig1_per_task_accuracy.png)
2. Component attribution
   - KD, readout choice, head type, selection rule
   - use [Table T2](C:/Users/Hellx/Documents/Programming/python/Project/iron/ghost-bank/studies/analysis/outputs/paper/tables/T2_component_ablations.md)
   - use [Figure 2](C:/Users/Hellx/Documents/Programming/python/Project/iron/ghost-bank/studies/analysis/outputs/paper/main/figures/fig2_component_attribution.png)
3. Resource sensitivity
   - memory budget
   - retrieval budget
   - use [Table T3](C:/Users/Hellx/Documents/Programming/python/Project/iron/ghost-bank/studies/analysis/outputs/paper/tables/T3_resource_sensitivity.md)
   - use [Figure 3](C:/Users/Hellx/Documents/Programming/python/Project/iron/ghost-bank/studies/analysis/outputs/paper/main/figures/fig3_resource_sensitivity.png)
4. Trade-off summary
   - accuracy vs forgetting
   - use [Figure 4](C:/Users/Hellx/Documents/Programming/python/Project/iron/ghost-bank/studies/analysis/outputs/paper/main/figures/fig4_acc_forgetting_scatter.png)
5. Failure-mode analysis
   - forgetting by task age
   - use [Figure 5](C:/Users/Hellx/Documents/Programming/python/Project/iron/ghost-bank/studies/analysis/outputs/paper/main/figures/fig5_forgetting_by_age.png)

**5. Discussion**
1. What actually drives performance
   - NME/readout pairing
   - herding vs random selection
   - KD as stability, not accuracy
2. Why memory matters more than retrieval
3. What the results imply for method design
4. Limitations
   - one dataset
   - 3 seeds
   - no saturation check beyond current memory range
5. Practical recommendation

**6. Conclusion**
1. One-paragraph summary of the main result
2. Final takeaway on replay, readout, and memory budget

**Appendix**
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
   - forgetting heatmap
   - evolution plots
   - stability slopes

**Recommended ordering of the main narrative**
1. Establish the problem and the protocol.
2. Show the main ranking.
3. Explain which components matter.
4. Show how budget changes the result.
5. Close with the trade-off and failure modes.

If you want the most defensible manuscript structure, I would keep it to:
1. Abstract
2. Introduction
3. Related Work
4. Method and Experimental Setup
5. Results
6. Discussion
7. Conclusion
8. Appendix

