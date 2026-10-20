| configuration | peak seq/s | at concurrency | p50 / p99 at peak (ms) | best seq/s, p99 <= 50 ms | best seq/s, p99 <= 100 ms | best seq/s, p99 <= 200 ms | best seq/s, p99 <= 500 ms |
|---|---|---|---|---|---|---|---|
| inf2.8xlarge native | **727** | 2 | 52 / 73 | — | 727 (C=2) | 727 (C=2) | 727 (C=2) |
| inf2.8xlarge trace | **761** | 2 | 51 / 73 | — | 761 (C=2) | 761 (C=2) | 761 (C=2) |
| inf2.xlarge native | **762** | 3 | 68 / 130 | — | 732 (C=2) | 762 (C=3) | 762 (C=3) |
| inf2.xlarge trace | **792** | 3 | 66 / 126 | — | 771 (C=2) | 792 (C=3) | 792 (C=3) |
| trn2.3xlarge native LNC=1 x8 | **3396** | 48 | 236 / 306 | — | 3215 (C=8) | 3376 (C=16) | 3396 (C=48) |
| trn2.3xlarge native LNC=2 x4 | **2977** | 8 | 44 / 72 | 2652 (C=4) | 2977 (C=8) | 2977 (C=8) | 2977 (C=8) |
| trn2.3xlarge native + NKI kernels LNC=1 x8 | **5384** | 32 | 99 / 133 | 5215 (C=8) | 5361 (C=16) | 5384 (C=32) | 5384 (C=32) |
| trn2.3xlarge native + NKI kernels LNC=2 x4 | **4606** | 96 | 336 / 391 | 4571 (C=8) | 4589 (C=16) | 4594 (C=24) | 4606 (C=96) |
| trn2.3xlarge trace LNC=1 x8 | **3662** | 32 | 145 / 197 | — | 3405 (C=8) | 3662 (C=32) | 3662 (C=32) |
