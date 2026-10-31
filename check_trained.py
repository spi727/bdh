"""Run from ~/projects/bdh. Checks recurrent BDH against the normal forward pass on the TRAINED model."""
import os, sys, torch
import torch.nn.functional as F
import compare as C
import bdh                                              # this folder's bdh.py (the one that trained the model)
sys.path.append(os.path.expanduser("~/projects/bdh_upstream"))
from bdh_recurrent import BDHState, step, generate_recurrent

dev = C.device
model = bdh.BDH(bdh.BDHConfig()).to(dev)
sd = torch.load("bdh_seed1337.pt", map_location=dev)
sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
model.load_state_dict(sd)
model.eval()

# 1) validation loss both ways, same data, fp32
g = torch.Generator().manual_seed(0)
worst, lf, lr_ = 0.0, [], []
for _ in range(5):
    x, y = C.get_batch("val", g)
    with torch.no_grad():
        full, _ = model(x)
        st = BDHState(model, batch=x.size(0))
        rec = torch.stack([step(model, x[:, t], st) for t in range(x.size(1))], dim=1)
    worst = max(worst, (full - rec).abs().max().item())
    lf.append(F.cross_entropy(full.reshape(-1, full.size(-1)), y.reshape(-1)).item())
    lr_.append(F.cross_entropy(rec.reshape(-1, rec.size(-1)), y.reshape(-1)).item())
print(f"val loss, normal forward : {sum(lf)/len(lf):.4f}")
print(f"val loss, recurrent      : {sum(lr_)/len(lr_):.4f}")
print(f"max |logit difference|   : {worst:.2e}")

# 2) generation both ways, same sampling seed
prompt = torch.tensor([list("ROMEO:\n".encode())], device=dev)
torch.manual_seed(1); a = model.generate(prompt.clone(), 300, temperature=0.8)
torch.manual_seed(1); b = generate_recurrent(model, prompt.clone(), 300, temperature=0.8)
same = torch.equal(a, b)
print("generated text identical :", same)
if not same:
    first = (a != b).nonzero()[0, 1].item()
    print("first differing position :", first)
print("---- recurrent sample ----")
print(bytes(b[0].tolist()).decode("utf-8", errors="replace"))
