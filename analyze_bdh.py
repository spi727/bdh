import torch, torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import compare as C
import bdh

model = bdh.BDH(bdh.BDHConfig()).to(C.device)
model.load_state_dict(torch.load("bdh_seed1337.pt", map_location=C.device))
model.eval()
cfg = model.config
D, nh, L = cfg.n_embd, cfg.n_head, cfg.n_layer
N = D * cfg.mlp_internal_dim_multiplier // nh
W = nh * N
dev = C.device

sums = [torch.zeros(256, W, device=dev) for _ in range(L)]   # activation mass per (current byte, neuron)
nz = [torch.zeros(W, device=dev) for _ in range(L)]          # how often each neuron fires
cnt = torch.zeros(256, device=dev)                           # how often each byte occurs
frac = torch.zeros(L, 3)                                     # active fraction: x_sparse, y_sparse, xy_sparse
NB = 100
g = torch.Generator().manual_seed(0)

with torch.no_grad(), C.ctx:
    for _ in range(NB):
        idx, _y = C.get_batch("val", g)
        B, T = idx.shape
        onehot = F.one_hot(idx.reshape(-1), 256).float()
        cnt += onehot.sum(0)
        x = model.ln(model.embed(idx).unsqueeze(1))
        for l in range(L):
            xs = F.relu(x @ model.encoder)
            yKV = model.ln(model.attn(Q=xs, K=xs, V=x))
            ys = F.relu(yKV @ model.encoder_v)
            xy = xs * ys
            frac[l] += torch.tensor([xs.gt(0).float().mean().item(),
                                     ys.gt(0).float().mean().item(),
                                     xy.gt(0).float().mean().item()])
            with torch.autocast("cuda", enabled=False):
                a = xs.permute(0, 2, 1, 3).reshape(B * T, W).float()
                sums[l] += onehot.T @ a
                nz[l] += a.gt(0).sum(0)
            yMLP = xy.transpose(1, 2).reshape(B, 1, T, W) @ model.decoder
            x = model.ln(x + model.ln(yMLP))
frac /= NB

n_tot = cnt.sum()
ar = torch.arange(W, device=dev)
print(f"{'layer':>5} {'active %':>9} {'dead %':>7} {'median top-byte share':>22} {'% neurons share>0.5':>20}")
shares = []
for l in range(L):
    S = sums[l]
    tot = S.sum(0)
    mean_b = S / cnt.clamp(min=1).unsqueeze(1)
    mean_b[cnt < 20] = 0
    top = mean_b.argmax(0)
    share = S[top, ar] / tot.clamp(min=1e-9)
    ok = nz[l] >= 50
    shares.append(share[ok].cpu())
    print(f"{l:5d} {100*frac[l,0]:9.1f} {100*(nz[l]==0).float().mean():7.1f} "
          f"{share[ok].median():22.3f} {100*(share[ok]>0.5).float().mean():20.1f}")
    if l == L - 1:
        base = cnt[top] / n_tot
        order = torch.argsort(torch.where(ok, share, torch.zeros_like(share)), descending=True)[:10]
        print("\nMost byte-selective neurons in the last layer:")
        for j in order.tolist():
            print(f"  head {j // N}, neuron {j % N}: byte {chr(int(top[j]))!r}, "
                  f"{share[j]:.2f} of its activation (byte is {base[j]:.3f} of text), fires {int(nz[l][j])} times")

print("\nActive fraction per layer (x_sparse, y_sparse, xy_sparse):")
for l in range(L):
    print(l, [round(100 * v.item(), 1) for v in frac[l]])

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for k, name in enumerate(["x_sparse", "y_sparse", "x*y"]):
    ax[0].plot(range(L), 100 * frac[:, k], marker="o", label=name)
ax[0].set_xlabel("layer"); ax[0].set_ylabel("% of neurons active"); ax[0].legend(); ax[0].grid(alpha=0.3)
ax[1].hist(shares[-1].numpy(), bins=40)
ax[1].set_xlabel("share of a neuron's activation on its top byte (last layer)")
ax[1].set_ylabel("neurons")
plt.tight_layout(); plt.savefig("bdh_sparsity.png", dpi=150)
print("saved bdh_sparsity.png")