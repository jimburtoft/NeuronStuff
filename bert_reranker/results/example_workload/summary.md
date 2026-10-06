| configuration | peak seq/s | at concurrency | p50 / p99 at peak (ms) | best seq/s, p99 <= 50 ms | best seq/s, p99 <= 100 ms | best seq/s, p99 <= 200 ms | best seq/s, p99 <= 500 ms |
|---|---|---|---|---|---|---|---|
| inf2.8xlarge native | **486** | 12 | 410 / 617 | — | — | 427 (C=2) | 486 (C=4) |
| inf2.8xlarge trace | **761** | 2 | 51 / 73 | — | 761 (C=2) | 761 (C=2) | 761 (C=2) |
| inf2.xlarge native | **485** | 12 | 414 / 597 | — | — | 418 (C=2) | 478 (C=4) |
| inf2.xlarge trace | **792** | 3 | 66 / 126 | — | 771 (C=2) | 792 (C=3) | 792 (C=3) |
| trn2.3xlarge native LNC=1 x8 | **1999** | 96 | 786 / 958 | — | — | 1885 (C=8) | 1990 (C=32) |
| trn2.3xlarge native LNC=2 x4 | **1899** | 64 | 540 / 680 | — | 1749 (C=4) | 1896 (C=8) | 1898 (C=24) |
| trn2.3xlarge trace LNC=1 x8 | **3662** | 32 | 145 / 197 | — | 3405 (C=8) | 3662 (C=32) | 3662 (C=32) |
