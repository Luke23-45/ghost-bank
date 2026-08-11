# A1_protocol

| Setting | Value |
| --- | ---: |
| Dataset | CIFAR-100, 10 tasks x 10 classes, split seed 13, probe/val splits 30/20 |
| Backbone | ResNet-18, base filters 64, dropout 0.0 |
| Head | cosine_margin (scale 30.0, margin 0.35, first-task imprinting); linear for a3 |
| Optimizer | SGD lr 0.1 / momentum 0.9 / weight decay 0.0005, grad clip 1.0, no LR schedule, warmup 0 |
| Epochs per task | 70 configured; 71 recorded for all 36 seeds (off-by-one in the epoch counter) |
| Batch size | 128 |
| Precision | 16-mixed |
| Seeds | 1993, 2023, 42 |
| Exemplar budgets | 500 (s1); 2000 (B0, B1, B2, B3, a1, a2, a3, a4, s3, s4); 4000 (s2) |
| Retrieval budgets | 32 (s3); 64 (B1, B2, B3, a1, a2, a3, a4, s1, s2); 128 (s4) |
| Knowledge distillation | weight 1.0, temperature 2.0; disabled for a1 (kd_weight 0.0) |
| Evaluation protocol | NME for B0/B1/B3/a1-a4; head-logit for B2 (native protocol) |
| Hardware | Tesla T4 |
| Software | torch 2.11.0+cu128, pytorch-lightning 2.6.5, python 3.12.13 |
| Git commits | B0, B1, B2 `327652e1` (dirty); B3, a1, a2, a3, a4, s1, s2, s3, s4 `4a47f6e5` (dirty) |
| Wall time | see Table A5 (compute cost) |
