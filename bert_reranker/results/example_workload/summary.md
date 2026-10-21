| configuration | peak seq/s | at concurrency | p50 / p99 at peak (ms) | best seq/s, p99 <= 50 ms | best seq/s, p99 <= 100 ms | best seq/s, p99 <= 200 ms | best seq/s, p99 <= 500 ms |
|---|---|---|---|---|---|---|---|
| inf2.8xlarge native | **727** | 2 | 52 / 73 | — | 727 (C=2) | 727 (C=2) | 727 (C=2) |
| inf2.8xlarge trace | **761** | 2 | 51 / 73 | — | 761 (C=2) | 761 (C=2) | 761 (C=2) |
| inf2.xlarge native | **817** | 8 | 161 / 244 | — | 759 (C=2) | 816 (C=4) | 817 (C=8) |
| inf2.xlarge native + NKI kernels | **1085** | 3 | 47 / 94 | 524 (C=1) | 1085 (C=3) | 1085 (C=3) | 1085 (C=3) |
| inf2.xlarge trace | **792** | 3 | 66 / 126 | — | 771 (C=2) | 792 (C=3) | 792 (C=3) |
| trn2.3xlarge native LNC=1 x8 | **3396** | 48 | 236 / 306 | — | 3215 (C=8) | 3376 (C=16) | 3396 (C=48) |
| trn2.3xlarge native LNC=2 x4 | **2977** | 8 | 44 / 72 | 2652 (C=4) | 2977 (C=8) | 2977 (C=8) | 2977 (C=8) |
| trn2.3xlarge native + NKI kernels LNC=1 x8 | **5381** | 16 | 50 / 79 | 5197 (C=8) | 5381 (C=16) | 5381 (C=16) | 5381 (C=16) |
| trn2.3xlarge native + NKI kernels LNC=2 x4 | **4629** | 64 | 230 / 271 | 4548 (C=8) | 4548 (C=8) | 4548 (C=8) | 4629 (C=64) |
| trn2.3xlarge trace LNC=1 x8 | **3662** | 32 | 145 / 197 | — | 3405 (C=8) | 3662 (C=32) | 3662 (C=32) |
