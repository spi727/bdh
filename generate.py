import sys, re, torch
import torch.nn.functional as F
import compare as C

text = open(sys.argv[1] if len(sys.argv) > 1 else "input.txt",
            encoding="utf-8", errors="ignore").read()
known = set(re.findall(r"[a-z']+", text.lower()))   # words that exist in the training text

PROMPT, NEW, TEMP, SEEDS = "ROMEO:\n", 300, 0.8, [1, 2, 3]
report, log = [], []

for name in ["transformer", "mamba", "bdh"]:
    model = C.build(name).to(C.device)
    model.load_state_dict(torch.load(f"{name}_seed{C.SEED}.pt", map_location=C.device))
    model.eval()
    hit = tot = 0
    for s in SEEDS:
        g = torch.Generator(device=C.device).manual_seed(s)
        idx = torch.tensor([list(PROMPT.encode())], device=C.device)
        with torch.no_grad():
            for _ in range(NEW):
                with C.ctx:
                    logits, _ = model(idx[:, -C.BLOCK_SIZE:])
                p = F.softmax(logits[:, -1, :].float() / TEMP, dim=-1)
                idx = torch.cat([idx, torch.multinomial(p, 1, generator=g)], dim=1)
        out = bytes(idx[0].tolist()).decode("utf-8", errors="replace")
        words = re.findall(r"[a-z']+", out.lower())
        hit += sum(w in known for w in words); tot += len(words)
        log.append(f"===== {name} | sample seed {s} =====\n{out}\n")
    report.append(f"{name}: {hit}/{tot} generated words are real training words ({100*hit/max(tot,1):.1f}%)")
    del model; torch.cuda.empty_cache()

open("samples.txt", "w").write("\n".join(log))
print("\n".join(report))