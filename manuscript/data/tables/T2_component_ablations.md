# T2_component_ablations

| # | Experiment | avg_acc (%) | forgetting (%) | Δ acc (pp) | Δ forgetting (pp) | Sig (acc) | Sig (fgt) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| a1 | Without KD | 42.52 ± 0.61 | 28.88 ± 0.19 | -1.48 | +11.66 | n.s. | sig |
| a2 | Head-logit evaluation | 33.54 ± 0.39 | 47.58 ± 0.84 | -10.46 | +30.36 | sig | sig |
| a3 | Linear head | 43.22 ± 0.36 | 17.34 ± 0.68 | -0.79 | +0.12 | marginal | n.s. |
| a4 | Random selection | 41.61 ± 0.53 | 18.38 ± 0.47 | -2.39 | +1.16 | sig | n.s. |
