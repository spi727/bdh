import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import bdh

class BDHCkpt(bdh.BDH):
    def _layer(self, x):
        x_sparse = F.relu(x @ self.encoder)
        yKV = self.ln(self.attn(Q=x_sparse, K=x_sparse, V=x))
        y_sparse = F.relu(yKV @ self.encoder_v)
        xy = self.drop(x_sparse * y_sparse)
        B, _, T, _ = x.shape
        N = xy.shape[-1]
        yMLP = xy.transpose(1, 2).reshape(B, 1, T, N * self.config.n_head) @ self.decoder
        return self.ln(x + self.ln(yMLP))

    def forward(self, idx, targets=None):
        x = self.ln(self.embed(idx).unsqueeze(1))
        for _ in range(self.config.n_layer):
            if self.training:
                x = checkpoint(self._layer, x, use_reentrant=False)
            else:
                x = self._layer(x)
        B, T = idx.shape
        logits = x.view(B, T, self.config.n_embd) @ self.lm_head
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss