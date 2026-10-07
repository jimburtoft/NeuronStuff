| configuration | peak seq/s | at concurrency | p50 / p99 at peak (ms) | best seq/s, p99 <= 50 ms | best seq/s, p99 <= 100 ms | best seq/s, p99 <= 200 ms | best seq/s, p99 <= 500 ms |
|---|---|---|---|---|---|---|---|
| inf2.8xlarge native | **727** | 2 | 52 / 73 | — | 727 (C=2) | 727 (C=2) | 727 (C=2) |
| inf2.8xlarge trace | **761** | 2 | 51 / 73 | — | 761 (C=2) | 761 (C=2) | 761 (C=2) |
| inf2.xlarge native | **762** | 3 | 68 / 130 | — | 732 (C=2) | 762 (C=3) | 762 (C=3) |
| inf2.xlarge trace | **792** | 3 | 66 / 126 | — | 771 (C=2) | 792 (C=3) | 792 (C=3) |
| trn2.3xlarge native LNC=1 x8 | **3293** | 32 | 163 / 220 | — | 3101 (C=8) | 3251 (C=16) | 3293 (C=32) |
| trn2.3xlarge native LNC=2 x4 | **2910** | 8 | 45 / 74 | 2599 (C=4) | 2910 (C=8) | 2910 (C=8) | 2910 (C=8) |
| trn2.3xlarge trace LNC=1 x8 | **3662** | 32 | 145 / 197 | — | 3405 (C=8) | 3662 (C=32) | 3662 (C=32) |
