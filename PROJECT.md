# BDH: memory-efficient training and recurrent inference

Machine Learning course project, K Praveen Sagar (CS26M1002), M.Tech CSE (Data Science and AI), IIITDM Kancheepuram.

This repository compares the Dragon Hatchling (BDH) language model with a Transformer and a
Mamba-style state-space model under one controlled protocol, and adds two things to BDH:

1. **Activation checkpointing** for training: 71% less peak GPU memory, identical results.
2. **Exact recurrent (Hebbian synapse) inference**: constant time and memory per generated token.

Everything lives in the fork https://github.com/spi727/bdh:
- branch `experiments`: this harness, all scripts, raw results, samples and figures;
- branch `checkpoint-layers`: the checkpointing flag (pull request #13 to pathwaycom/bdh);
- branch `hebbian-recurrent`: the recurrent inference mode.

## Results

All models about 25M parameters, Tiny Shakespeare as bytes, batch 4, context 256, 2,000 steps,
AdamW 3e-4 with warm-up and cosine decay, bf16, seeds 1337 / 1 / 2, one RTX 3050 Laptop GPU (4 GB).

| Model | Val loss (mean ± sd) | Time / step | Peak GPU memory |
| --- | --- | --- | --- |
| BDH | 1.519 ± 0.009 | 494 ms | 3,355 MB |
| Mamba-style SSM | 1.531 ± 0.009 | 961 ms | 1,528 MB |
| Transformer | 1.808 ± 0.024 | 47 ms | 622 MB |

The Transformer reaches 1.509 at 8,000 steps.

**Checkpointing.** Peak memory 3,355 MB → 976 MB; validation loss identical at every seed
(1.513 / 1.530 / 1.515); training time +34% (seed 1337).

**Recurrent inference.** Matches the normal forward pass on the trained model (val loss 1.5277 both
ways, max logit difference 9e-6, identical generated text). About 13 ms per token at any context length,
against 1,542 ms for re-reading 2,048 tokens without a cache; fixed 192 MB state.

**Length generalisation.** A BDH trained on 256-byte windows: loss 1.496 within 256 bytes,
1.568 at 256–511, 1.985 at 512–1,023, 2.565 at 1,024–2,047.
Trained on 512-byte windows with checkpointing (same 2,000 steps, RTX 5060 Laptop GPU): 1.474, 1.411,
1.691 and 2.180, lower at every range; validation loss 1.506 at 1,485 MB peak.

## Files

| File | Purpose |
| --- | --- |
| `compare.py` | Shared harness: data, Transformer, Mamba-style SSM, BDH, training, timing, memory, results.csv |
| `bdh.py` | BDH model (from pathwaycom/bdh, MIT licence) |
| `bdh_ckpt.py` | Checkpointed BDH subclass used for the training runs (same change as the fork's flag) |
| `bdh_recurrent.py` | Recurrent Hebbian-state inference (`BDHState`, `step`, `generate_recurrent`) |
| `test_scan.py` | Checks the chunked SSM scan against a step-by-step loop |
| `test_ckpt.py`, `check_equal.py` | Checkpointing equivalence checks |
| `analyze.py`, `loadres.py` | Mean and spread over seeds |
| `plot_curves.py`, `plot_context.py` | Figures |
| `generate.py` | Same-prompt generation and the real-word measure |
| `check_trained.py` | Recurrent vs normal forward on the trained model |
| `check_long.py`, `check_long_equal.py` | Loss by position past the training length |
| `results_main.csv`, `results.csv`, `curve_*.csv` | Raw results |
| `*.png`, `samples.txt` | Figures and generated text |

`test_checkpoint.py`, `test_recurrent.py` and `bench_recurrent.py` are on the `checkpoint-layers` and `hebbian-recurrent` branches.

## Reproduce

```bash
pip install -r requirements.txt
# put Tiny Shakespeare at input.txt (https://github.com/karpathy/char-rnn, data/tinyshakespeare)
for m in transformer mamba bdh bdh-ckpt; do
  for s in 1337 1 2; do LR=3e-4 ITERS=2000 SEED=$s python compare.py $m; done
done
python analyze.py
```

Trained weights (`*.pt`) are not committed.

## References

- Kosowski et al., The Dragon Hatchling, arXiv:2509.26507 (2025); code: https://github.com/pathwaycom/bdh
- Gu and Dao, Mamba, arXiv:2312.00752 (2023)
- Chen et al., Training Deep Nets with Sublinear Memory Cost, arXiv:1604.06174 (2016)
