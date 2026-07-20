Device: cuda
Loading CIFAR-100 data...
[CIFAR100Ingestor] Downloading parquet from Hugging Face (uoft-cs/cifar100) ...
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.

cifar100/train/0000.parquet: downloading bytes:  97% 115M/119M [00:04<00:00, 26.7MB/s, 8.91MB/s  ]
cifar100/train/0000.parquet: downloading bytes: 100% 115M/115M [00:04<00:00, 24.0MB/s, 9.05MB/s  ]
cifar100/train/0000.parquet: reconstructing file: 100% 119M/119M [00:04<00:00, 24.7MB/s, 10.6MB/s  ]

cifar100/test/0000.parquet: downloading bytes:  90% 21.4M/23.8M [00:02<00:00, 14.9MB/s, 1.30MB/s  ]
cifar100/test/0000.parquet: reconstructing file:  38% 8.94M/23.8M [00:02<00:03, 4.11MB/s]
cifar100/test/0000.parquet: reconstructing file:  50% 12.0M/23.8M [00:02<00:02, 5.71MB/s,  849kB/s  ]
cifar100/test/0000.parquet: reconstructing file:  68% 16.1M/23.8M [00:02<00:00, 8.62MB/s, 1.13MB/s  ]
cifar100/test/0000.parquet: downloading bytes: 100% 23.1M/23.1M [00:02<00:00, 9.33MB/s, 2.11MB/s  ]
cifar100/test/0000.parquet: reconstructing file: 100% 23.8M/23.8M [00:02<00:00, 9.59MB/s, 2.22MB/s  ]
[CIFAR100Ingestor] Converting Hugging Face parquet to tensors ...
Train images: 100% 50000/50000 [00:06<00:00, 7971.80it/s]
Test images: 100% 10000/10000 [00:01<00:00, 6572.61it/s]
[CIFAR100Ingestor] Done — train=50000, test=10000
Loaded 50000 images in 17.5s
Train per task: [4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000]

======================================================================
  PID-DDC 10-task benchmark
  lam0=1.0, alpha=1.0, tau=2.0
  epochs=70
======================================================================
    Task  1/10 done in 226s
    Task  2 — PID debts: max=0.14 mean=0.06 lam range=[1.01, 1.14]
    Task  2/10 done in 294s
    Task  3 — PID debts: max=0.65 mean=0.22 lam range=[1.04, 1.65]
    Task  3/10 done in 294s
    Task  4 — PID debts: max=1.48 mean=0.55 lam range=[1.02, 2.48]
    Task  4/10 done in 293s
    Task  5 — PID debts: max=2.44 mean=1.21 lam range=[1.03, 3.44]
    Task  5/10 done in 293s
    Task  6 — PID debts: max=3.23 mean=1.49 lam range=[1.01, 4.23]
    Task  6/10 done in 293s
    Task  7 — PID debts: max=5.20 mean=2.82 lam range=[1.02, 6.20]
    Task  7/10 done in 293s
    Task  8 — PID debts: max=6.91 mean=3.59 lam range=[1.00, 7.91]
    Task  8/10 done in 293s
    Task  9 — PID debts: max=9.66 mean=4.60 lam range=[1.03, 10.66]
    Task  9/10 done in 293s
    Task 10 — PID debts: max=10.01 mean=5.67 lam range=[1.01, 11.01]
    Task 10/10 done in 293s
    Calibrating classifier across all 100 classes...
    Evaluating...

  PID-DDC (lam0=1.0, alpha=1.0)
  --------------------------------------------------
    task 0: 0.197  (0.860, 0.340, 0.020, 0.030, 0.240, 0.010, 0.060, 0.020, 0.190, 0.200)
    task 1: 0.159  (0.030, 0.000, 0.330, 0.220, 0.000, 0.100, 0.200, 0.420, 0.290, 0.000)
    task 2: 0.171  (0.540, 0.050, 0.030, 0.230, 0.200, 0.190, 0.000, 0.060, 0.410, 0.000)
    task 3: 0.099  (0.130, 0.240, 0.080, 0.190, 0.010, 0.000, 0.010, 0.000, 0.250, 0.080)
    task 4: 0.165  (0.020, 0.190, 0.150, 0.000, 0.010, 0.020, 0.410, 0.290, 0.270, 0.290)
    task 5: 0.151  (0.010, 0.020, 0.480, 0.070, 0.180, 0.000, 0.280, 0.030, 0.250, 0.190)
    task 6: 0.258  (0.440, 0.340, 0.220, 0.170, 0.140, 0.040, 0.000, 0.340, 0.670, 0.220)
    task 7: 0.152  (0.000, 0.360, 0.260, 0.040, 0.050, 0.350, 0.400, 0.000, 0.010, 0.050)
    task 8: 0.260  (0.000, 0.200, 0.410, 0.190, 0.320, 0.120, 0.320, 0.230, 0.560, 0.250)
    task 9: 0.293  (0.120, 0.520, 0.000, 0.170, 0.700, 0.440, 0.410, 0.080, 0.210, 0.280)
    avg overall: 0.191  [2881s]

======================================================================
  COMPARISON
======================================================================
  Method                              Avg
  ----------------------------------------
  Baseline (no replay)                7.8%
  StaticBank                         13.1%
  PID-GB                             10.9%
  DRKD (lam=1.0)                     18.1%
  PID-DDC (lam0=1.0, α=1.0)          19.1%
======================================================================