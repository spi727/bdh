"""Run from ~/projects/bdh. Is the loss rise past 256 bytes a property of the model, or a bug in the recurrent route?
Runs ONE 1024-byte validation snippet through both routes and compares them."""
import os, sys, torch
import torch.nn.functional as F
import compare as C
import bdh
sys.path.append(os.path.expanduser("~/projects/bdh_upstream"))
from bdh_recurrent import BDHState, step

T = 1024
C.BLOCK_SIZE = T
dev = C.device
model = bdh.BDH(bdh.BDHConfig()).to(dev)
sd = torch.load("bdh_seed1337.pt", map_location=dev)
model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in sd.items()})
model.eval()

g = torch.Generator().manual_seed(0)
x, y = C.get_batch("val", g)
assert x.size(1) == T
x, y = x[:1], y[:1]
with torch.no_grad():
    full, _ = model(x)
    st = BDHState(model, batch=1)
    rec = torch.stack([step(model, x[:, t], st) for t in range(T)], dim=1)
print(f"max |logit difference| over {T} positions: {(full - rec).abs().max().item():.2e}")
lf = F.cross_entropy(full[0], y[0], reduction="none").cpu()
lr = F.cross_entropy(rec[0], y[0], reduction="none").cpu()
for a, b in [(0, 256), (256, 512), (512, 1024)]:
    print(f"  positions {a:>4}-{b-1:<4}: normal forward {lf[a:b].mean():.3f}   recurrent {lr[a:b].mean():.3f}")
