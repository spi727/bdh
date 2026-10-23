"""Checks that recurrent (Hebbian-state) BDH gives the same logits as the normal forward pass. CPU only."""
import torch
from bdh import BDH, BDHConfig
from bdh_recurrent import BDHState, step, generate_recurrent

torch.manual_seed(0)
cfg = BDHConfig(n_layer=3, n_embd=64, n_head=2, mlp_internal_dim_multiplier=16, vocab_size=256, dropout=0.0)
model = BDH(cfg).eval()
for p in model.parameters():                      # make activations non-trivial
    p.data.normal_(0, 0.1) if p.dim() > 1 else None

idx = torch.randint(0, 256, (2, 48))
with torch.no_grad():
    full, _ = model(idx)                          # normal forward, all positions at once
    state = BDHState(model, batch=2)
    rec = torch.stack([step(model, idx[:, t], state) for t in range(idx.size(1))], dim=1)

diff = (full - rec).abs().max().item()
print(f"max |full - recurrent| over all positions: {diff:.2e}  (logit scale {full.abs().max().item():.2f})")
assert torch.allclose(full, rec, atol=1e-4, rtol=1e-4), "MISMATCH"

# generation: same sampling seed -> same text from both routes
prompt = idx[:, :5]
torch.manual_seed(1); a = model.generate(prompt.clone(), 20)
torch.manual_seed(1); b = generate_recurrent(model, prompt.clone(), 20)
print("generate() == generate_recurrent():", torch.equal(a, b))
print("state size: %.2f MB" % (state.nbytes() / 2**20))
print("PASS")
