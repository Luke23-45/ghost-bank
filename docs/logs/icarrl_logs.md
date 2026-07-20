Device: cuda
Loading CIFAR-100 data...
[CIFAR100Ingestor] Downloading parquet from Hugging Face (uoft-cs/cifar100) ...

cifar100/train/0000.parquet: downloading bytes:  94% 111M/119M [00:01<00:00, 153MB/s, 7.02MB/s  ]  
cifar100/train/0000.parquet: downloading bytes: 100% 115M/115M [00:01<00:00, 87.3MB/s, 11.0MB/s  ]
cifar100/train/0000.parquet: reconstructing file: 100% 119M/119M [00:01<00:00, 89.7MB/s, 11.5MB/s  ]

cifar100/test/0000.parquet: downloading bytes:  74% 17.6M/23.8M [00:01<00:00, 22.7MB/s, 1.07MB/s  ]
cifar100/test/0000.parquet: reconstructing file:  49% 11.7M/23.8M [00:01<00:01, 9.47MB/s]
cifar100/test/0000.parquet: reconstructing file:  56% 13.3M/23.8M [00:01<00:01, 10.2MB/s, 1.14MB/s  ]
cifar100/test/0000.parquet: downloading bytes: 100% 23.1M/23.1M [00:01<00:00, 15.7MB/s, 2.18MB/s  ]
cifar100/test/0000.parquet: reconstructing file: 100% 23.8M/23.8M [00:01<00:00, 16.1MB/s, 2.27MB/s  ]
[CIFAR100Ingestor] Converting Hugging Face parquet to tensors ...
Train images: 100% 50000/50000 [00:05<00:00, 8342.44it/s]
Test images: 100% 10000/10000 [00:01<00:00, 8811.75it/s]
[CIFAR100Ingestor] Done — train=50000, test=10000
Loaded 50000 images in 12.2s
Train per task: [4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000]

======================================================================
  iCaRL 10-task benchmark
  lam=1.0, epochs=70
======================================================================
    Task  1/10 done in 235s
    Task  2/10 done in 511s
    Task  3/10 done in 582s
    Task  4/10 done in 611s
    Task  5/10 done in 585s
    Task  6/10 done in 613s
    Task  7/10 done in 632s
    Task  8/10 done in 648s
    Task  9/10 done in 650s
    Task 10/10 done in 652s
    Evaluating with NME...

  iCaRL (lam=1.0)
  --------------------------------------------------
    task 0: 0.000  (0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000)
    task 1: 0.514  (0.500, 0.460, 0.610, 0.570, 0.490, 0.320, 0.520, 0.690, 0.540, 0.440)
    task 2: 0.534  (0.750, 0.610, 0.310, 0.660, 0.810, 0.410, 0.340, 0.510, 0.550, 0.390)
    task 3: 0.354  (0.610, 0.330, 0.190, 0.540, 0.280, 0.220, 0.480, 0.280, 0.180, 0.430)
    task 4: 0.392  (0.250, 0.530, 0.390, 0.440, 0.220, 0.240, 0.180, 0.610, 0.610, 0.450)
    task 5: 0.406  (0.280, 0.330, 0.580, 0.800, 0.450, 0.120, 0.360, 0.420, 0.390, 0.330)
    task 6: 0.423  (0.730, 0.570, 0.600, 0.340, 0.300, 0.120, 0.120, 0.230, 0.690, 0.530)
    task 7: 0.342  (0.460, 0.610, 0.080, 0.330, 0.300, 0.470, 0.480, 0.170, 0.250, 0.270)
    task 8: 0.374  (0.140, 0.290, 0.700, 0.420, 0.250, 0.480, 0.390, 0.510, 0.290, 0.270)
    task 9: 0.399  (0.390, 0.480, 0.270, 0.230, 0.730, 0.470, 0.350, 0.430, 0.280, 0.360)
    avg overall: 0.374  [6097s]

======================================================================
  COMPARISON
======================================================================
  Method                              Avg
  ----------------------------------------
  Baseline (no replay)                7.8%
  StaticBank                         13.1%
  PID-GB                             10.9%
  iCaRL (ours)                       37.4%
  DRKD (lam=1.0)                     18.1%
  PID-DDC (lam0=1.0, α=1.0)          19.1%
======================================================================