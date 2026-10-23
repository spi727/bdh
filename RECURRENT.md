# Recurrent (Hebbian synapse) inference for BDH

`bdh_recurrent.py` runs BDH one token at a time with a fixed-size state instead of
re-reading the whole text for every new token.

## Idea

BDH attention has no softmax:

    y_t = sum over s<t of (QR_t . QR_s) x_s

where QR is the sparse neuron activity after RoPE. Regrouping the sum:

    y_t = QR_t @ S_t,    S_t = sum over s<t of QR_s^T x_s

`S` has shape (N x D) per head, per layer. After each token, `QR^T x` is added to it:
the connection between a neuron and a value strengthens when that neuron fires for
that token. This is the Hebbian synapse view described in the BDH paper
(arXiv 2509.26507). The result is mathematically identical to the full forward pass.
RoPE only needs the token's absolute position, which the state tracks.

## What is here

- `bdh_recurrent.py`: `BDHState`, `step` (one token), `generate_recurrent` (same interface as `BDH.generate`).
- `test_recurrent.py`: checks that logits from the recurrent route match `model(idx)` at every position,
  and that both generate routes give the same text with the same sampling seed.
- `bench_recurrent.py`: time and memory for producing one token at different context lengths.

No existing file is modified.

## Memory

The state holds n_layer x n_head x N x D numbers. For the 25M model (6 layers, 4 heads,
N = 8192, D = 256) that is about 50M numbers (about 200 MB in fp32), and it does not grow with context length.
A cache of past tokens needs about 33,000 numbers per token per layer, so the two cost the same
at roughly 254 tokens; beyond that the state is smaller.

## Benchmark (RTX 3050 4 GB, fp32, random weights, one next token)

| Context T | Full forward ms | Full extra MB | Recurrent ms | Recurrent extra MB |
| --- | --- | --- | --- | --- |
| 256 | 179.5 | 224 | 33.2 (includes GPU warm-up) | 33 |
| 512 | 192.1 | 432 | 19.8 | 33 |
| 1024 | 495.0 | 865 | 12.7 | 33 |
| 2048 | 1542.2 | 1730 | 12.9 | 33 |

"Full forward" is what `BDH.generate` does today (no cache). A fair comparison with a
token cache is not measured here.

## Limits

- Inference only; training still uses the parallel form.
- The prompt is read one token at a time, so reading a long prompt is slower than one parallel pass.
- The state is kept in fp32; lower precision was not tested.
- Dropout is off (evaluation mode).
- Verified on small random models and one real model forward pass; not yet measured against trained-weight perplexity.

The recurrent state itself is 192 MB (fp32) at every context length and is not included in
the "extra MB" columns. Single run per cell; the first recurrent row includes GPU warm-up.
At T=256 the recurrent total (192 + 33 MB) is about equal to the full forward (224 MB), as predicted;
at longer contexts the full forward keeps growing and the recurrent version does not.
