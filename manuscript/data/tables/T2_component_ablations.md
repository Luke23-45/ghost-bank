# T2_component_ablations

| # | Experiment | avg_acc (%) | forgetting (%) | Δ acc (pp) | Δ forgetting (pp) | Sig (acc) | Sig (fgt) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| a1 | Ref. without KD | 44.79 ± 0.31 | 24.78 ± 0.81 | -0.21 | +10.93 | n.s. | sig |
| a2 | Ref. head-logit eval | 34.61 ± 1.13 | 46.36 ± 1.63 | -10.38 | +32.50 | sig | sig |
| a3 | Ref. linear head | 43.57 ± 0.47 | 14.68 ± 0.48 | -1.42 | +0.83 | marginal | n.s. |
| a4 | Ref. random selection | 40.17 ± 0.17 | 18.35 ± 0.61 | -4.83 | +4.50 | sig | sig |
