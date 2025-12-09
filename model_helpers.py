import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt

def get_batch(data_ids, block_size=128, batch_size=32, device="cpu"):
    """
    data_ids: 1D LongTensor of token ids.
    Returns:
        x, y: [batch_size, block_size] on the given device.
    """
    ix = torch.randint(0, len(data_ids) - block_size - 1, (batch_size,))

    x = torch.stack([data_ids[i : i + block_size] for i in ix])
    y = torch.stack([data_ids[i + 1 : i + block_size + 1] for i in ix])

    return x.to(device).long(), y.to(device).long()

def estimate_loss(
    model,
    train_ids,
    val_ids,
    vocab_size,
    block_size=128,
    device="cpu",
):
    """
    Compute average loss & accuracy on train/val splits.

    Returns:
        out = {
            "train": (avg_loss, accuracy),
            "val":   (avg_loss, accuracy),
        }
    """
    model.eval()
    out = {}

    with torch.no_grad():
        for split, data_ids in (("train", train_ids), ("val", val_ids)):
            losses = []
            correct = 0
            total = 0

            for _ in range(10):
                xb, yb = get_batch(data_ids, block_size=block_size, device=device)
                logits = model(xb)  # [B, T, vocab_size]

                # cross-entropy expects [N, C] and [N]
                loss = F.cross_entropy(
                    logits.view(-1, vocab_size),
                    yb.view(-1),
                )
                losses.append(loss.item())

                preds = torch.argmax(logits, dim=-1)  # [B, T]
                correct += (preds == yb).float().sum().item()
                total += yb.numel()

            avg_loss = sum(losses) / len(losses)
            accuracy = correct / total

            out[split] = (avg_loss, accuracy)

    model.train()
    return out

class ResidualBlock(nn.Module):
    """Residual block with time embedding."""

    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t):
        h = F.relu(self.conv1(x))
        time_emb = F.relu(self.time_mlp(t))
        h = h + time_emb[:, :, None, None]  # Broadcast to (B, C, H, W)
        h = F.relu(self.conv2(h))
        return h + self.shortcut(x)


class SinusoidalPositionEmbedding(nn.Module):
    """Timestep embedding using sinusoidal functions."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings

def prepare_noise_schedule(device, timesteps=200, beta_start=0.0001, beta_end=0.02):
    """
    Precompute all values needed for the forward diffusion process.

    Args:
        timesteps: Total number of diffusion steps
        beta_start: Starting noise level
        beta_end: Ending noise level

    Returns (betas, alphas):
        - betas: Noise schedule
        - alphas: Cumulative product of (1 - betas)
    """
    betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
    alphas = torch.cumprod(1.0 - betas, dim=0)

    return betas, alphas

def show_image_tensor(tensor, ax=None, title=None):
    """
    tensor: [C, H, W] in [-1,1], with C=1 or 3
    """
    tensor = ((tensor + 1) / 2).clamp(0, 1)

    if tensor.shape[0] == 1:
        img_np = tensor.squeeze(0).cpu().numpy()       # [H, W]
        cmap = "gray"
    else:
        img_np = tensor.permute(1, 2, 0).cpu().numpy() # [H, W, C]
        cmap = None

    if ax is None:
        plt.figure(figsize=(5, 3))
        plt.imshow(img_np, cmap=cmap, aspect="auto")
        plt.axis('off')
        if title:
            plt.title(title)
        plt.tight_layout()
        plt.show()
    else:
        ax.imshow(img_np, cmap=cmap, aspect="auto")
        ax.axis('off')
        if title:
            ax.set_title(title)
