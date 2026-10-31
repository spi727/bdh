| Model | Val loss (mean ± std) | Bits/byte | ms/step | Train time (s) | Peak MB | Seeds |
|---|---|---|---|---|---|---|
| bdh | 1.519 ± 0.009 | 2.192 | 494 | 989 | 3355 | 3 |
| bdh-ckpt | 1.513 ± nan | 2.183 | 683 | 1366 | 976 | 1 |
| mamba | 1.531 ± 0.009 | 2.209 | 961 | 1922 | 1528 | 3 |
| transformer | 1.808 ± 0.024 | 2.608 | 47 | 94 | 622 | 3 |

Transformer with more steps (seed 1337):

| Steps | Val loss | Train time (s) |
|---|---|---|
| 2000 | 1.830 | 94 |
| 4000 | 1.573 | 187 |
| 8000 | 1.509 | 369 |