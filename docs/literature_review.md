# Literature Review

This project sits at the intersection of class-imbalanced learning, long-tailed recognition, and memory-based representation methods. The literature suggests three important points.

## 1. Imbalanced learning is usually addressed by reweighting, resampling, or loss reshaping

Class imbalance is commonly handled by changing how examples contribute to the loss or to minibatch construction rather than by changing the model alone.

- Focal Loss down-weights easy examples so that abundant, well-classified negatives do not dominate training.
  - Source: [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
- Class-Balanced Loss reweights classes using the effective number of samples, motivated by the diminishing marginal value of additional samples from already frequent classes.
  - Source: [Class-Balanced Loss Based on Effective Number of Samples](https://arxiv.org/abs/1901.05555)
- Balanced Group Softmax shows that even when extra tail sampling is avoided, class-specific classifier training can help address imbalance in detection.
  - Source: [Balanced Group Softmax](https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_Overcoming_Classifier_Imbalance_for_Long-Tail_Object_Detection_With_Balanced_Group_CVPR_2020_paper.pdf)

## 2. Memory-based methods already exist and are relevant to the ghost-bank idea

Several long-tailed recognition papers use memory banks, prototypes, or historical embeddings to enrich tail-class learning.

- Memory-based Jitter stores historical embeddings or prototypes in a memory bank and uses them to improve tail diversity.
  - Source: [Memory-based Jitter](https://www.arxiv.org/pdf/2008.09809v1)
- Inflated Episodic Memory introduces a memory mechanism for long-tailed visual recognition and argues that a single prototype is often insufficient because tail classes can have high intra-class variance.
  - Source: [Inflated Episodic Memory](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhu_Inflated_Episodic_Memory_With_Region_Self-Attention_for_Long-Tailed_Visual_Recognition_CVPR_2020_paper.pdf)

These papers are the closest conceptual neighbors to the current project. They support the idea that a persistent memory structure can help minority or tail classes by preserving useful class information over time.

## 3. Strong baselines matter

Recent work shows that some gains attributed to specialized imbalance methods can also be obtained by tuning ordinary pipeline components such as batch size, augmentation, optimizer choice, and label smoothing.

- Source: [How to Train Under Class Imbalance](https://proceedings.neurips.cc/paper_files/paper/2023/file/6ea69f8116b7c01e3c3e43b62e6868fc-Paper-Conference.pdf)

This matters for the paper because the ghost bank must be evaluated against strong, well-tuned baselines, not only against plain cross-entropy.

## Implication for This Project

The literature supports the following framing:

- The problem is minority-class underexposure under severe imbalance.
- Existing solutions mostly manipulate the loss, the sampling distribution, the classifier training schedule, or a memory/prototype mechanism.
- The ghost bank should therefore be framed as an exposure-debt controlled memory mechanism, not merely as another static bank.
- The experimental claim should be that exposure-debt controlled retrieval improves minority-class metrics beyond strong baselines and static memory variants under the same budget.

This review is the basis for the cleaned problem statement, hypothesis, and mathematical definition used in the rest of the project.
