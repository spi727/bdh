"""Cost of producing ONE next token at context length T:
   full  = what BDH.generate does today (re-run the whole text)
   recur = Hebbian synapse state (constant work per token).
Weights are random: speed and memory do not depend on trained values.  fp32 for both."""
import os, time, torch
from bdh import BDH, BDHConfig
from bdh_recurrent import BDHState, step

dev = "cuda" if torch.cuda.is_available() else "cpu"
small = os.environ.get("BENCH_SMALL") == "1"          # quick CPU logic check only
cfg = (BDHConfig(n_layer=2, n_embd=64, n_head=2, mlp_internal_dim_multiplier=16, vocab_size=256, dropout=0.0)
       if small else
       BDHConfig(n_layer=6, n_embd=256, n_head=4, mlp_internal_dim_multiplier=128, vocab_size=256, dropout=0.0))
model = BDH(cfg).to(dev).eval()
sync = torch.cuda.synchronize if dev == "cuda" else (lambda: None)
MB = 2**20

def peak_extra(fn, reps):
    if dev == "cuda":
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
    sync(); t = time.time()
    for _ in range(reps): fn()
    sync(); ms = (time.time() - t) / reps * 1000
    extra = (torch.cuda.max_memory_allocated() - base) / MB if dev == "cuda" else float("nan")
    return ms, extra

print(f"{'T':>6} | {'full ms/token':>13} {'full extra MB':>13} | {'recur ms/token':>14} {'recur extra MB':>14} {'state MB':>9}")
for T in ([32, 64, 128] if small else [256, 512, 1024, 2048]):
    idx = torch.randint(0, 256, (1, T), device=dev)
    try:
        with torch.no_grad():
            f_ms, f_mb = peak_extra(lambda: model(idx), 3)
        f = f"{f_ms:13.1f} {f_mb:13.0f}"
    except torch.cuda.OutOfMemoryError:
        f = f"{'OOM':>13} {'OOM':>13}"; torch.cuda.empty_cache()
    st = BDHState(model); st.t = T
    tok = torch.randint(0, 256, (1,), device=dev)
    def one():
        st.t = T; step(model, tok, st)
    r_ms, r_mb = peak_extra(one, 10)
    print(f"{T:>6} | {f} | {r_ms:14.1f} {r_mb:14.0f} {st.nbytes()/MB:9.0f}")
