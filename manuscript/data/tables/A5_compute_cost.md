# A5_compute_cost

| # | Experiment | wall_time_s | seed 1993 | seed 2023 | seed 42 | device |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 | No replay | 2022.4 ± 79.4 | 1913.1 | 2098.9 | 2055.3 | Tesla T4 |
| B1 | iCaRL | 3649.3 ± 26.1 | 3612.8 | 3662.5 | 3672.6 | Tesla T4 |
| B2 | Static bank | 2756.4 ± 109.3 | 2638.2 | 2729.1 | 2901.8 | Tesla T4 |
| B3 | Uniform herding (Reference) | 4635.4 ± 47.5 | 4570.0 | 4654.3 | 4681.7 | Tesla T4 |
| a1 | Ref. without KD | 3446.0 ± 43.7 | 3401.3 | 3431.6 | 3505.3 | Tesla T4 |
| a2 | Ref. head-logit eval | 4332.4 ± 33.3 | 4289.1 | 4338.1 | 4370.0 | Tesla T4 |
| a3 | Ref. linear head | 3957.4 ± 17.8 | 3934.0 | 3960.8 | 3977.3 | Tesla T4 |
| a4 | Ref. random selection | 4336.8 ± 18.5 | 4314.1 | 4336.7 | 4359.5 | Tesla T4 |
| s1 | Memory 500 | 4237.0 ± 55.4 | 4181.5 | 4216.7 | 4312.7 | Tesla T4 |
| s2 | Memory 4000 | 4309.4 ± 15.3 | 4325.7 | 4288.9 | 4313.6 | Tesla T4 |
| s3 | Retrieval 32 | 3516.5 ± 25.3 | 3508.8 | 3490.1 | 3550.7 | Tesla T4 |
| s4 | Retrieval 128 | 5285.7 ± 40.7 | 5311.7 | 5317.1 | 5228.3 | Tesla T4 |
