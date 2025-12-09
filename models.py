import torch
import torch.nn as nn
import torch.nn.functional as F

from model_helpers import get_batch, estimate_loss


class MidiTextTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256,
                 n_head=4, n_layer=6, dim_ff=512, block_size=512):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=dim_ff,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layer)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.shape
        assert T <= self.block_size
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = self.tok_emb(x) + self.pos_emb(pos)

        # causal mask: prevent attention to future positions
        mask = torch.triu(torch.ones(T, T, device=x.device) * float("-inf"), diagonal=1)
        h = self.encoder(h, mask)
        logits = self.lm_head(h)
        return logits

def train_midi_text_transformer(
    model: MidiTextTransformer,
    train_ids,
    val_ids,
    vocab_size: int,
    device: str | torch.device = None,
    max_iters: int = 6000,
    eval_interval: int = 250,
    lr: float = 3e-4,
):
    """
    Train a MozartTransformer using get_batch / estimate_loss from model_helpers.

    Args:
        model:        instantiated MozartTransformer
        train_ids:    tensor or array of token ids for the training corpus
        vocab_size:   size of the vocabulary (should match model.vocab_size)
        device:       "cuda", "cpu", or torch.device (defaults to auto)
        max_iters:    number of training steps
        eval_interval: steps between eval logging
        lr:           learning rate
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for step in range(max_iters):
        # get_batch should return xb, yb of shape [B, T], dtype long
        xb, yb = get_batch(train_ids, device=device)

        # Forward pass
        logits = model(xb)  # [B, T, vocab_size]

        B, T, C = logits.shape
        assert C == vocab_size, f"Logits last dim {C} != vocab_size {vocab_size}"

        loss = F.cross_entropy(
            logits.view(B * T, C),
            yb.view(B * T)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % eval_interval == 0:
            losses = estimate_loss(model, train_ids, val_ids, vocab_size, device=device)
            train_loss, train_acc = losses["train"]
            val_loss, val_acc = losses["val"]
            print(
                f"step {step}: "
                f"train loss {train_loss:.3f}, acc {train_acc:.3f} | "
                f"val loss {val_loss:.3f}, acc {val_acc:.3f}"
            )

    return model

@torch.no_grad()
def generate_midi_tokens_with_transformer(
    model: MidiTextTransformer,
    sos_id: int,
    eos_id: int,
    start_tokens=None,
    max_new_tokens: int = 1000,
    device: str | torch.device | None = None,
):
    """
    Autoregressively generate token IDs from a trained model.

    Args:
        model:          trained MidiTextTransformer
        sos_id:         integer ID for SOS token
        eos_id:         integer ID for EOS token
        start_tokens:   list of int to seed generation (optional)
        max_new_tokens: max number of new tokens to sample
        device:         device to run on (defaults to model's device)

    Returns:
        List[int]: the full token sequence (including seed & generated)
    """
    model.eval()

    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)

    block_size = model.block_size

    if start_tokens is None:
        x = torch.tensor([[sos_id]], dtype=torch.long, device=device)
    else:
        x = torch.tensor([start_tokens], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        # only feed the last block_size tokens
        x_cond = x[:, -block_size:]

        logits = model(x_cond)          # [1, T, vocab_size]
        logits = logits[:, -1, :]       # [1, vocab_size] – last time step
        probs = torch.softmax(logits, dim=-1)

        next_id = torch.multinomial(probs, num_samples=1)  # [1, 1]
        x = torch.cat([x, next_id], dim=1)

        # stop at EOS
        if int(next_id[0, 0]) == eos_id:
            break

    return x[0].tolist()