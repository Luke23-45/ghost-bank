# A5_compute_cost

| # | Experiment | wall_time_s | seed 1993 | seed 2023 | seed 42 | device |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B1 | iCaRL | 3754.8 ± 38.0 | 3709.5 | 3802.5 | 3752.4 | Tesla T4 |
| B2 | Static bank | 2716.3 ± 91.6 | 2619.8 | 2689.7 | 2839.4 | Tesla T4 |
| B3 | Uniform Herding | 4493.2 ± 52.5 | 4514.2 | 4421.1 | 4544.4 | Tesla T4 |
| a1 | Without KD | 3307.7 ± 54.0 | 3242.5 | 3306.0 | 3374.7 | Tesla T4 |
| a2 | Head-logit evaluation | 4279.5 ± 93.4 | 4148.5 | 4359.7 | 4330.3 | Tesla T4 |
| a3 | Linear head | 3732.1 ± 42.7 | 3673.7 | 3748.1 | 3774.5 | Tesla T4 |
| a4 | Random selection | 4555.9 ± 37.1 | 4507.9 | 4561.7 | 4598.2 | Tesla T4 |
| s1 | Active budget 500 | 4292.4 ± 54.2 | 4227.7 | 4289.3 | 4360.2 | Tesla T4 |
| s2 | Active budget 4000 | 4351.7 ± 26.0 | 4318.3 | 4355.1 | 4381.8 | Tesla T4 |
| s3 | Retrieval 32 | 3958.1 ± 104.2 | 3810.7 | 4030.0 | 4033.5 | Tesla T4 |
| s4 | Retrieval 128 | 5919.2 ± 49.9 | 5890.3 | 5877.9 | 5989.4 | Tesla T4 |
