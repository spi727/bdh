"""Run from ~/projects/bdh. Does the trained BDH keep working past its 256-byte training length?
Reads 2048-byte validation snippets with the recurrent route and reports loss by position range."""
import os, sys, torch
import torch.nn.functional as F
import compare as C
import bdh
sys.path.append(os.path.expanduser("~/projects/bdh_upstream"))
from bdh_recurrent import BDHState, step

T, NSEQ = 2048, 6
C.BLOCK_SIZE = T                                         # make get_batch return T-byte slices
dev = C.device
model = bdh.BDH(bdh.BDHConfig()).to(dev)
sd = torch.load("bdh_seed1337.pt", map_location=dev)
model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in sd.items()})
model.eval()

g = torch.Generator().manual_seed(0)
loss_at = torch.zeros(T, device=dev)
for i in range(NSEQ):
    x, y = C.get_batch("val", g)
    assert x.size(1) == T, f"get_batch gave length {x.size(1)}, expected {T}"
    x, y = x[:1], y[:1]
    st = BDHState(model, batch=1)
    for t in range(T):
        loss_at[t] += F.cross_entropy(step(model, x[:, t], st), y[:, t])
    print(f"sequence {i+1}/{NSEQ} done", flush=True)
loss_at = (loss_at / NSEQ).cpu()

print("\nmean loss by position range (%d sequences):" % NSEQ)
for a, b in [(0, 256), (256, 512), (512, 1024), (1024, 2048)]:
    print(f"  positions {a:>4}-{b-1:<4}: {loss_at[a:b].mean():.3f}")
