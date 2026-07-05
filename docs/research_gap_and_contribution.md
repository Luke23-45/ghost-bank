# Research Gap and Contribution

## 1. What the literature already covers

The current literature already addresses class imbalance through several strong families of methods:

- Loss reshaping and reweighting, such as focal loss and class-balanced loss.
  - [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
  - [Class-Balanced Loss Based on Effective Number of Samples](https://arxiv.org/abs/1901.05555)
- Balanced classifier training and decoupled optimization, especially in long-tailed recognition and detection.
  - [Balanced Group Softmax](https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_Overcoming_Classifier_Imbalance_for_Long-Tail_Object_Detection_With_Balanced_Group_CVPR_2020_paper.pdf)
  - [How to Train Under Class Imbalance](https://proceedings.neurips.cc/paper_files/paper/2023/file/6ea69f8116b7c01e3c3e43b62e6868fc-Paper-Conference.pdf)
- Memory- and prototype-based tail enrichment.
  - [Memory-based Jitter](https://www.arxiv.org/pdf/2008.09809v1)
  - [Inflated Episodic Memory](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhu_Inflated_Episodic_Memory_With_Region_Self-Attention_for_Long-Tailed_Visual_Recognition_CVPR_2020_paper.pdf)

A recent survey also confirms that long-tailed learning already spans data balancing, neural architecture, feature enrichment, logits adjustment, loss functions, network optimization, and post hoc processing. In other words, the field is broad and crowded.

Source:
- [A Systematic Review on Long-Tailed Learning](https://arxiv.org/html/2408.00483v1)

## 2. What gap remains

The gap is not "class imbalance exists." That is already well known.

The gap is that existing methods often address imbalance indirectly:

- by changing the loss
- by changing the sampling distribution
- by decoupling classifier training
- by using a generic memory or prototype structure

What is less directly isolated in the literature is a training-time mechanism that measures whether a class has received enough effective optimization exposure and then uses that measurement to control minority-memory retrieval.

This is the practical gap this project should target. The important shift is from a static memory bank to a closed-loop exposure controller.

## 3. The research idea in one sentence

Ghost Bank is an exposure-debt controlled minority memory that stores rare-class information and retrieves it when a class has received less training exposure than a target schedule.

## 4. What would make this publishable

This project is worth publishing only if at least one of the following is true:

1. The ghost bank introduces a genuinely new retrieval or update mechanism, not just replay or prototype storage with a new name.
2. The method consistently improves minority-class metrics at severe imbalance ratios where strong baselines fail.
3. The method is simpler or more stable than existing memory- or prototype-based long-tailed methods.
4. The method provides a clear empirical or theoretical insight about why minority-class learning fails under minibatch optimization.

If the final method is only "store rare examples and reuse them," then it overlaps heavily with replay, resampling, and memory-based tail enrichment. That would be weak as a paper contribution unless the experimental setting or theory is unusually strong.

## 5. Best defensible contribution claim

The safest contribution claim is:

> We introduce an exposure-debt controlled memory mechanism for extreme class imbalance. Instead of relying only on class frequency, the method tracks how much effective training exposure each class receives and dynamically retrieves stored minority information when a class becomes underexposed.

That is defensible because it is narrower than claiming a universal new answer to imbalance, and it gives the method a concrete distinction from ordinary replay, resampling, and static memory banks.

## 6. Practical recommendation

At this stage, the paper should be positioned as one of two things:

- a method paper, if you can define a new ghost-bank mechanism with a distinct update/retrieval policy
- an empirical research paper, if the main contribution is a careful study showing when and why such a bank helps

If no new mechanism is introduced, the paper should not oversell novelty. In that case, the value comes from a rigorous experimental study and a clear formalization, not from claiming a brand-new imbalance paradigm.
