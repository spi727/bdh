import math, os, sys, time, csv
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import numpy as np
import bdh

# ---- settings: identical for every model ----
BLOCK_SIZE, BATCH_SIZE, MAX_ITERS = 256, 4, 500
LR, WD, LOG_FREQ, EVAL_BATCHES, SEED, WARMUP = 1e-3, 0.1, 50, 50, 1337, 10
LR = float(os.environ.get("LR", LR))
MAX_ITERS = int(os.environ.get("ITERS", MAX_ITERS))
SEED = int(os.environ.get("SEED", SEED))
BLOCK_SIZE = int(os.environ.get("BLOCK", BLOCK_SIZE))
COMPILE = os.environ.get("COMPILE") == "1"
CFG = {
    "transformer": dict(d_model=512, n_layer=8, n_head=8),
    "mamba": dict(d_model=704, n_layer=8, ckpt=True),
}

device = torch.device("cuda")
ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "input.txt")
data = torch.from_numpy(np.fromfile(DATA, dtype=np.uint8).astype(np.int64))
n = int(0.9 * len(data))
splits = {"train": data[:n], "val": data[n:]}

def get_batch(split, g):
    d = splits[split]
    ix = torch.randint(len(d) - BLOCK_SIZE - 1, (BATCH_SIZE,), generator=g)
    x = torch.stack([d[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([d[i + 1:i + 1 + BLOCK_SIZE] for i in ix])
    return x.to(device), y.to(device)

def init(m):
    if isinstance(m, (nn.Linear, nn.Embedding)):
        nn.init.normal_(m.weight, std=0.02)

def lm_loss(logits, targets):
    if targets is None:
        return None
    return F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.view(-1))

# ---------------- Transformer ----------------
class TBlock(nn.Module):
    def __init__(s, d, h):
        super().__init__()
        s.h = h
        s.ln1, s.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        s.qkv = nn.Linear(d, 3 * d, bias=False)
        s.proj = nn.Linear(d, d, bias=False)
        s.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
    def forward(s, x):
        B, T, C = x.shape
        q, k, v = s.qkv(s.ln1(x)).split(C, -1)
        q, k, v = [t.view(B, T, s.h, C // s.h).transpose(1, 2) for t in (q, k, v)]
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + s.proj(a.transpose(1, 2).reshape(B, T, C))
        return x + s.mlp(s.ln2(x))

class GPT(nn.Module):
    def __init__(s, d_model, n_layer, n_head):
        super().__init__()
        s.tok, s.pos = nn.Embedding(256, d_model), nn.Embedding(BLOCK_SIZE, d_model)
        s.blocks = nn.ModuleList([TBlock(d_model, n_head) for _ in range(n_layer)])
        s.ln_f = nn.LayerNorm(d_model)
        s.head = nn.Linear(d_model, 256, bias=False)
        s.apply(init)
        s.head.weight = s.tok.weight
    def forward(s, idx, targets=None):
        x = s.tok(idx) + s.pos(torch.arange(idx.size(1), device=idx.device))
        for b in s.blocks:
            x = b(x)
        logits = s.head(s.ln_f(x))
        return logits, lm_loss(logits, targets)

# ---------------- Mamba (pure PyTorch, no CUDA kernels) ----------------
def scan_slow(a, dBx, Cm):
    # a, dBx: (B,T,di,ds)  Cm: (B,T,ds)  ->  y: (B,T,di)
    B, T, di, ds = a.shape
    dA = torch.exp(a)
    h = a.new_zeros(B, di, ds)
    ys = []
    for t in range(T):
        h = dA[:, t] * h + dBx[:, t]
        ys.append((h * Cm[:, t].unsqueeze(1)).sum(-1))
    return torch.stack(ys, 1)

def scan_fast(a, dBx, Cm, L=16):
    B, T, di, ds = a.shape
    pad = (-T) % L
    if pad:
        a = F.pad(a, (0, 0, 0, 0, 0, pad))
        dBx = F.pad(dBx, (0, 0, 0, 0, 0, pad))
        Cm = F.pad(Cm, (0, 0, 0, pad))
    Tp = T + pad
    nc = Tp // L
    a = a.reshape(B, nc, L, di, ds)
    dBx = dBx.reshape(B, nc, L, di, ds)
    Cc = Cm.reshape(B, nc, L, 1, ds)
    S = torch.cumsum(a, 2)                       # log-decay inside each chunk (<= 0)
    h_loc = torch.exp(S) * torch.cumsum(torch.exp(-S) * dBx, 2)
    D = torch.exp(S[:, :, -1])                   # total decay of each chunk
    end = h_loc[:, :, -1]
    H = a.new_zeros(B, di, ds)
    prev = []
    for c in range(nc):                          # only nc = T/L sequential steps
        prev.append(H)
        H = D[:, c] * H + end[:, c]
    Hprev = torch.stack(prev, 1)
    h = h_loc + torch.exp(S) * Hprev.unsqueeze(2)
    y = (h * Cc).sum(-1)
    return y.reshape(B, Tp, di)[:, :T]

SCAN = scan_fast

class MambaBlock(nn.Module):
    def __init__(s, d, d_state=16, d_conv=4, expand=2):
        super().__init__()
        s.di, s.ds, s.dtr = expand * d, d_state, math.ceil(d / 16)
        s.in_proj = nn.Linear(d, 2 * s.di, bias=False)
        s.conv = nn.Conv1d(s.di, s.di, d_conv, groups=s.di, padding=d_conv - 1)
        s.x_proj = nn.Linear(s.di, s.dtr + 2 * d_state, bias=False)
        s.dt_proj = nn.Linear(s.dtr, s.di)
        dt = torch.exp(torch.rand(s.di) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        s.dt_proj.bias.data = dt + torch.log(-torch.expm1(-dt))
        s.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).float().repeat(s.di, 1)))
        s.D = nn.Parameter(torch.ones(s.di))
        s.out_proj = nn.Linear(s.di, d, bias=False)
    def forward(s, x):
        B, T, _ = x.shape
        xi, z = s.in_proj(x).chunk(2, -1)
        xi = F.silu(s.conv(xi.transpose(1, 2))[..., :T].transpose(1, 2)).float()
        dt, Bm, Cm = s.x_proj(xi).float().split([s.dtr, s.ds, s.ds], -1)
        dt = F.softplus(s.dt_proj(dt).float())
        A = -torch.exp(s.A_log.float())
        a = (dt.unsqueeze(-1) * A).clamp(min=-4.0)
        dBx = dt.unsqueeze(-1) * Bm.unsqueeze(2) * xi.unsqueeze(-1)
        y = SCAN(a, dBx, Cm) + xi * s.D
        return s.out_proj((y * F.silu(z.float())).to(x.dtype))

class MambaLM(nn.Module):
    def __init__(s, d_model, n_layer, ckpt=False):
        super().__init__()
        s.ckpt = ckpt
        s.tok = nn.Embedding(256, d_model)
        s.blocks = nn.ModuleList([MambaBlock(d_model) for _ in range(n_layer)])
        s.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layer)])
        s.ln_f = nn.LayerNorm(d_model)
        s.head = nn.Linear(d_model, 256, bias=False)
        s.apply(init)
        s.head.weight = s.tok.weight
    def forward(s, idx, targets=None):
        x = s.tok(idx)
        for blk, nm in zip(s.blocks, s.norms):
            if s.ckpt and s.training:
                x = x + checkpoint(blk, nm(x), use_reentrant=False)
            else:
                x = x + blk(nm(x))
        logits = s.head(s.ln_f(x))
        return logits, lm_loss(logits, targets)

def build(name):
    if name == "bdh":
        return bdh.BDH(bdh.BDHConfig())
    if name == "bdh-ckpt":
        import bdh_ckpt
        return bdh_ckpt.BDHCkpt(bdh.BDHConfig())
    if name == "transformer":
        return GPT(**CFG[name])
    return MambaLM(**CFG[name])

@torch.no_grad()
def evaluate(model):
    model.eval()
    g = torch.Generator().manual_seed(0)
    tot = 0.0
    for _ in range(EVAL_BATCHES):
        x, y = get_batch("val", g)
        with ctx:
            tot += model(x, y)[1].item()
    return tot / EVAL_BATCHES

def run(name):
    torch.manual_seed(SEED)
    g = torch.Generator().manual_seed(SEED)
    model = build(name).to(device)
    tag = name + ("-compiled" if COMPILE else "")
    if COMPILE:
        model = torch.compile(model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n=== {name}: {n_params/1e6:.2f}M parameters ===")
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    torch.cuda.reset_peak_memory_stats()
    model.train()
    torch.cuda.synchronize(); t_start = time.time()
    eval_s=0.0
    for step in range(MAX_ITERS):
        if step == WARMUP:
            torch.cuda.synchronize(); t0 = time.time()
        x, y = get_batch("train", g)
        warm = min(1.0, (step + 1) / 50)
        cos = 0.1 + 0.45 * (1 + math.cos(math.pi * step / MAX_ITERS))
        for pg in opt.param_groups:
            pg["lr"] = LR * warm * cos
        with ctx:
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % LOG_FREQ == 0:
            print(f"step {step}/{MAX_ITERS} train loss {loss.item():.3f}")
        if (step > 0 and step % 250 == 0) or step == MAX_ITERS - 1:
            torch.cuda.synchronize(); te = time.time()
            v = evaluate(model)
            model.train()
            eval_s += time.time() - te
            with open(f"curve_{name}_s{SEED}.csv", "a") as f:
                f.write(f"{name},{LR},{step},{time.time()-t_start-eval_s:.1f},{v:.4f}\n")
    torch.cuda.synchronize()
    total_s = time.time() - t_start-eval_s
    ms = (time.time() - t0-eval_s) / (MAX_ITERS - WARMUP) * 1000
    val = evaluate(model)
    torch.save(model.state_dict(), f"{name}_seed{SEED}.pt")
    mem = torch.cuda.max_memory_allocated() / 2**20
    row = [tag, LR, MAX_ITERS, SEED, BLOCK_SIZE, f"{n_params/1e6:.2f}", f"{val:.3f}", f"{val/math.log(2):.3f}",
       f"{ms:.0f}", f"{BATCH_SIZE*BLOCK_SIZE/(ms/1000):.0f}", f"{total_s:.0f}", f"{mem:.0f}"]
    print("RESULT (name, M params, val loss, val bpc, ms/step, peak MB):", row)
    new = not os.path.exists("results.csv")
    with open("results.csv", "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["model", "lr","Max_Iters", "Seed","block", "M_params", "val_loss", "val_bpc",
            "ms_per_step", "tokens_per_sec", "total_train_s", "peak_MB"])
        w.writerow(row)

if __name__ == "__main__":
    names = ["bdh", "transformer", "mamba"] if sys.argv[1] == "all" else [sys.argv[1]]
    for n_ in names:
        run(n_)
        torch.cuda.empty_cache()