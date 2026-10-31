import time, torch
import compare as C

def make(B, T, di, ds, scale, dtype, device):
    g = torch.Generator().manual_seed(0)
    a = (-scale * torch.rand(B, T, di, ds, generator=g, dtype=dtype)).clamp(min=-4.0)
    d = torch.randn(B, T, di, ds, generator=g, dtype=dtype)
    c = torch.randn(B, T, ds, generator=g, dtype=dtype)
    return [t.to(device).requires_grad_() for t in (a, d, c)]

def rel(x, y):
    return ((x - y).abs().max() / (y.abs().max() + 1e-12)).item()

def check(B, T, di, ds, scale, dtype, device):
    a, d, c = make(B, T, di, ds, scale, dtype, device)
    w = torch.randn(B, T, di, dtype=dtype).to(device)
    ys = C.scan_slow(a, d, c)
    gs = torch.autograd.grad((ys * w).sum(), [a, d, c])
    yf = C.scan_fast(a, d, c)
    gf = torch.autograd.grad((yf * w).sum(), [a, d, c])
    print(f"{str(dtype):14s} T={T:3d} decay scale={scale:4.1f}: "
          f"output err {rel(yf, ys):.1e}, gradient err {max(rel(f, s) for f, s in zip(gf, gs)):.1e}")

def bench(fn, device):
    a, d, c = make(4, 256, 1408, 16, 1.0, torch.float32, device)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t = time.time()
    for _ in range(3):
        fn(a, d, c).sum().backward()
    torch.cuda.synchronize()
    return (time.time() - t) / 3, torch.cuda.max_memory_allocated() / 2**20

if __name__ == "__main__":
    print("--- correctness, float64 on CPU (expect about 1e-10 or smaller) ---")
    for T, s in [(100, 0.5), (100, 10.0), (256, 1.0)]:
        check(2, T, 8, 4, s, torch.float64, "cpu")
    print("--- correctness, float32 on GPU, real shape (expect below 1e-3) ---")
    check(4, 256, 1408, 16, 1.0, torch.float32, "cuda")
    print("--- speed for one layer, forward + backward ---")
    for name, fn in [("slow loop", C.scan_slow), ("chunked", C.scan_fast)]:
        s, mb = bench(fn, "cuda")
        print(f"{name:10s} {s*1000:8.0f} ms   peak {mb:6.0f} MB")