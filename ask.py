"""Type a prompt, get a continuation from all three trained models.
Run from ~/projects/bdh:   python ask.py          (interactive)
                           python ask.py "ROMEO:" (one prompt)
Options via env: N=300 (bytes to generate)  TEMP=0.8  SEED=1"""
import os, sys, torch
import torch.nn.functional as F
import compare as C

N, TEMP, SEED = int(os.environ.get("N", 300)), float(os.environ.get("TEMP", 0.8)), int(os.environ.get("SEED", 1))
MODELS = [("transformer", "transformer_seed1337.pt"), ("mamba", "mamba_seed1337.pt"), ("bdh", "bdh_seed1337.pt")]

loaded = []
for name, path in MODELS:
    m = C.build(name).to(C.device)
    sd = torch.load(path, map_location=C.device)
    m.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in sd.items()})
    m.eval()
    loaded.append((name, m))
print(f"loaded {len(loaded)} models; generating {N} bytes, temperature {TEMP}, sampling seed {SEED}")

@torch.no_grad()
def continue_text(model, prompt):
    idx = torch.tensor([list(prompt.encode("utf-8"))], device=C.device)
    g = torch.Generator(device=C.device).manual_seed(SEED)       # same randomness for every model
    for _ in range(N):
        with C.ctx:
            logits, _ = model(idx[:, -C.BLOCK_SIZE:])                # models were trained on 256-byte windows
        p = F.softmax(logits[:, -1, :].float() / TEMP, dim=-1)
        idx = torch.cat([idx, torch.multinomial(p, 1, generator=g)], dim=1)
    return bytes(idx[0].tolist()).decode("utf-8", errors="replace")

def run(prompt):
    prompt = prompt.replace("\\n", "\n")
    for name, m in loaded:
        out = continue_text(m, prompt)
        print(f"\n===== {name} =====\n{out}")
    print()

if len(sys.argv) > 1:
    run(" ".join(sys.argv[1:]))
else:
    print('Type a prompt (use \\n for a new line, e.g. ROMEO:\\n). Empty line to quit.')
    while True:
        try:
            p = input("> ")
        except EOFError:
            break
        if not p.strip():
            break
        run(p)
