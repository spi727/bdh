import math
import torch
import compare as C
import bdh

SEED = 1337
torch.manual_seed(SEED)
g = torch.Generator().manual_seed(SEED)
model = bdh.BDH(bdh.BDHConfig()).to(C.device)
opt = torch.optim.AdamW(model.parameters(), lr=C.LR, weight_decay=C.WD)
iters = C.MAX_ITERS
model.train()
for step in range(iters):
    x, y = C.get_batch("train", g)
    warm = min(1.0, (step + 1) / 50)
    cos = 0.1 + 0.45 * (1 + math.cos(math.pi * step / iters))
    for pg in opt.param_groups:
        pg["lr"] = C.LR * warm * cos
    with C.ctx:
        _, loss = model(x, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    if step % 100 == 0:
        print(f"step {step}/{iters} loss {loss.item():.3f}")
print("val loss:", round(C.evaluate(model), 3))
torch.save(model.state_dict(), "bdh_seed1337.pt")
print("saved bdh_seed1337.pt")