import importlib.util, torch

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

orig = load("/home/prave/projects/bdh/bdh.py", "orig")
new = load("/home/prave/projects/bdh_upstream/bdh.py", "new")
sd = torch.load("/home/prave/projects/bdh/bdh_seed1337.pt", map_location="cuda")
g = torch.Generator().manual_seed(0)
x = torch.randint(0, 256, (1, 128), generator=g).cuda()
y = torch.randint(0, 256, (1, 128), generator=g).cuda()

outs = []
for mod in (orig, new):
    m = mod.BDH(mod.BDHConfig()).cuda()
    m.load_state_dict(sd)
    m.eval()
    with torch.no_grad():
        outs.append(m(x, y))
print("loss:", outs[0][1].item(), outs[1][1].item())
print("max logit difference:", (outs[0][0] - outs[1][0]).abs().max().item())