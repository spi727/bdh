"""Recurrent ("Hebbian synapse") inference for BDH.

BDH's attention is  y_t = sum_{s<t} (QR_t . QR_s) x_s  with no softmax, so it can be
rewritten as  y_t = QR_t @ S_t,  S_t = sum_{s<t} QR_s^T x_s.
S is a fixed-size grid of synapses (N x D per head). Each new token adds QR^T x to it
(neurons that fire together strengthen their connection to the token's value).
This gives exactly the same numbers as the normal forward pass, up to float rounding.
"""
import torch
import torch.nn.functional as F
from bdh import Attention


class BDHState:
    def __init__(self, model, batch=1, dtype=torch.float32):
        C = model.config
        nh, D = C.n_head, C.n_embd
        N = C.mlp_internal_dim_multiplier * D // nh
        dev = model.decoder.device
        self.S = [torch.zeros(batch, nh, N, D, device=dev, dtype=dtype) for _ in range(C.n_layer)]
        self.t = 0  # absolute position of the next token (RoPE needs it)

    def nbytes(self):
        return sum(s.numel() * s.element_size() for s in self.S)


@torch.no_grad()
def step(model, tok, state):
    """tok: LongTensor (B,) = the token at position state.t. Returns logits (B, vocab)."""
    C = model.config
    B = tok.size(0)
    x = model.ln(model.embed(tok).view(B, 1, 1, -1))            # B,1,1,D
    phases = float(state.t) * model.attn.freqs                    # 1,1,1,N   (same as arange*freqs)
    for l in range(C.n_layer):
        x_sparse = F.relu(x @ model.encoder)                      # B,nh,1,N
        QR = Attention.rope(phases, x_sparse)                     # B,nh,1,N
        S = state.S[l]
        yKV = QR @ S                                              # B,nh,1,D  (past tokens only, like tril(-1))
        S += QR.transpose(-1, -2) @ x                             # add this token for the future
        yKV = model.ln(yKV)
        y_sparse = F.relu(yKV @ model.encoder_v)
        xy_sparse = x_sparse * y_sparse                           # (dropout is off in eval)
        yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, 1, -1) @ model.decoder
        x = model.ln(x + model.ln(yMLP))
    state.t += 1
    return x.view(B, -1) @ model.lm_head


@torch.no_grad()
def generate_recurrent(model, idx, max_new_tokens, temperature=1.0, top_k=None):
    """Same interface as BDH.generate, but keeps the synapse state instead of re-reading the text."""
    model.eval()
    state = BDHState(model, batch=idx.size(0))
    for i in range(idx.size(1)):                                  # read the prompt
        logits = step(model, idx[:, i], state)
    for _ in range(max_new_tokens):
        lg = logits / temperature
        if top_k is not None:
            v, _ = torch.topk(lg, min(top_k, lg.size(-1)))
            lg[lg < v[:, [-1]]] = float("-inf")
        nxt = torch.multinomial(F.softmax(lg, dim=-1), 1)
        idx = torch.cat((idx, nxt), dim=1)
        logits = step(model, nxt[:, 0], state)
    return idx
