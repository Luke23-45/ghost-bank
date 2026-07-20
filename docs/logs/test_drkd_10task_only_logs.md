Device: cuda
Loading CIFAR-100 data...
[CIFAR100Ingestor] Downloading parquet from Hugging Face (uoft-cs/cifar100) ...
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.

cifar100/train/0000.parquet: downloading bytes:  95% 112M/119M [00:03<00:00, 28.0MB/s, 9.34MB/s  ]
cifar100/train/0000.parquet: downloading bytes: 100% 115M/115M [00:04<00:00, 27.9MB/s, 9.42MB/s  ]
cifar100/train/0000.parquet: reconstructing file: 100% 119M/119M [00:04<00:00, 28.7MB/s, 10.8MB/s  ]

cifar100/test/0000.parquet: downloading bytes:  32% 7.57M/23.8M [00:01<00:04, 3.83MB/s]
cifar100/test/0000.parquet: downloading bytes:  92% 21.9M/23.8M [00:02<00:00, 13.2MB/s,  723kB/s  ]
cifar100/test/0000.parquet: reconstructing file:   9% 2.07M/23.8M [00:02<00:16, 1.29MB/s, 18.8kB/s  ]
cifar100/test/0000.parquet: reconstructing file:  54% 12.8M/23.8M [00:02<00:01, 9.39MB/s,  197kB/s  ]
cifar100/test/0000.parquet: downloading bytes: 100% 23.1M/23.1M [00:02<00:00, 8.79MB/s, 2.09MB/s  ]
cifar100/test/0000.parquet: reconstructing file: 100% 23.8M/23.8M [00:02<00:00, 9.04MB/s, 2.20MB/s  ]
[CIFAR100Ingestor] Converting Hugging Face parquet to tensors ...
Train images: 100% 50000/50000 [00:06<00:00, 7521.34it/s]
Test images: 100% 10000/10000 [00:01<00:00, 9186.59it/s]
[CIFAR100Ingestor] Done — train=50000, test=10000
Loaded 50000 images in 17.2s
Train per task: [4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000]

======================================================================
  DRKD 10-task benchmark (DRKD only)
  lam=1.0, tau=2.0, epochs=70
======================================================================
    Task  1/10 done in 225s
    Task  2/10 done in 297s
    Task  3/10 done in 296s
    Task  4/10 done in 294s
    Task  5/10 done in 293s
    Task  6/10 done in 291s
    Task  7/10 done in 291s
    Task  8/10 done in 290s
    Task  9/10 done in 289s
    Task 10/10 done in 284s
    Calibrating classifier across all 100 classes...
    Evaluating...

  3. DRKD (lam=1.0)
  --------------------------------------------------
    task 0: 0.132  (0.670, 0.200, 0.040, 0.010, 0.080, 0.020, 0.040, 0.010, 0.120, 0.130)
    task 1: 0.130  (0.050, 0.000, 0.260, 0.130, 0.040, 0.080, 0.270, 0.210, 0.260, 0.000)
    task 2: 0.174  (0.610, 0.140, 0.050, 0.320, 0.130, 0.070, 0.000, 0.050, 0.260, 0.110)
    task 3: 0.070  (0.070, 0.150, 0.080, 0.110, 0.000, 0.020, 0.000, 0.010, 0.170, 0.090)
    task 4: 0.169  (0.020, 0.180, 0.090, 0.000, 0.010, 0.060, 0.320, 0.420, 0.260, 0.330)
    task 5: 0.179  (0.000, 0.010, 0.580, 0.450, 0.090, 0.010, 0.230, 0.120, 0.170, 0.130)
    task 6: 0.258  (0.510, 0.290, 0.200, 0.120, 0.010, 0.060, 0.000, 0.370, 0.660, 0.360)
    task 7: 0.186  (0.400, 0.500, 0.270, 0.120, 0.000, 0.320, 0.200, 0.020, 0.010, 0.020)
    task 8: 0.238  (0.030, 0.240, 0.470, 0.060, 0.140, 0.140, 0.240, 0.250, 0.600, 0.210)
    task 9: 0.273  (0.150, 0.490, 0.000, 0.220, 0.740, 0.430, 0.190, 0.050, 0.130, 0.330)
    avg overall: 0.181  [2866s]

======================================================================
  RESULT vs published baselines
======================================================================
  Method                              Avg
  ----------------------------------------
  Baseline (no replay)                7.8% (from results.md)
  StaticBank                         13.1% (from results.md)
  PID-GB                             10.9% (from results.md)
  DRKD (lam=1.0)                     18.1%
======================================================================