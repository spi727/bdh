import torch
import compare as C
import bdh_flag

def run(flag):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    m = bdh_flag.BDH(bdh_flag.BDHConfig(checkpoint_layers=flag)).to(C.device)
    m.load_state_dict(torch.load("bdh_seed1337.pt", map_location=C.device))
    m.train()
    x, y = C.get_batch("train", torch.Generator().manual_seed(0))
    torch.manual_seed(1)
    with C.ctx:
        _, loss = m(x, y)
    loss.backward()
    g = torch.cat([p.grad.flatten().float() for p in m.parameters()])
    mem = torch.cuda.max_memory_allocated() / 2**20
    return loss.item(), g, mem

l0, g0, m0 = run(False)
l1, g1, m1 = run(True)
print(f"flag off: loss {l0:.6f}, peak {m0:.0f} MB")
print(f"flag on : loss {l1:.6f}, peak {m1:.0f} MB")
print("relative gradient difference:", ((g0 - g1).abs().max() / g0.abs().max()).item())