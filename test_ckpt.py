import torch, compare as C, bdh, bdh_ckpt

def run(cls):
    m = cls(bdh.BDHConfig()).to(C.device)
    m.load_state_dict(torch.load("bdh_seed1337.pt", map_location=C.device))
    m.train()
    x, y = C.get_batch("train", torch.Generator().manual_seed(0))
    torch.manual_seed(1)
    with C.ctx:
        _, loss = m(x, y)
    loss.backward()
    g = torch.cat([p.grad.flatten().float() for p in m.parameters()])
    return loss.item(), g

l1, g1 = run(bdh.BDH)
l2, g2 = run(bdh_ckpt.BDHCkpt)
print("loss:", l1, l2)
print("relative gradient difference:", ((g1 - g2).abs().max() / g1.abs().max()).item())