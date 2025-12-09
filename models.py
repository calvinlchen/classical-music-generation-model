import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil
from tqdm import tqdm


from model_helpers import SinusoidalPositionEmbedding, ResidualBlock, prepare_noise_schedule, show_image_tensor
from model_helpers import get_batch, estimate_loss


class MidiTextTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256,
                 n_head=4, n_layer=6, dim_ff=512, block_size=512,
                 dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=dim_ff,
            dropout=dropout,
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
        mask = torch.triu(torch.ones(T, T, device=x.device) * float("-inf"),
                          diagonal=1)
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
            losses = estimate_loss(model, train_ids, val_ids, vocab_size,
                                   device=device)
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

# ----- IMAGE-RELATED DIFFUSION MODEL CLASSES AND METHODS -----

def train_diffusion_model(model, dataloader, timesteps, num_epochs=500,
                          lr=1e-4, gen_freq=50, weight_decay=1e-4,
                          device="cpu"):
    """
    Train the diffusion model to predict noise.
    
    Args:
        model: SimpleUNet model
        dataloader: DataLoader with training images
        num_epochs: Total number of training epochs
        lr: Learning rate
        gen_freq: Generate sample image every N epochs
    
    Returns:
        losses: List of average loss per epoch
    """
    _, alphas = prepare_noise_schedule(device, timesteps=timesteps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)
    losses = []
    
    num_epoch_groups = ceil(num_epochs / gen_freq)
    print(f"\nGenerating example every {gen_freq} epochs.")
    print("="*70)
    
    for epoch_group in range(num_epoch_groups):
        pbar = tqdm(range(gen_freq),
                    desc=f"Group {epoch_group+1}/{num_epoch_groups}")
        
        for epoch in pbar:
            total_loss = 0
            
            for batch in dataloader:
                batch = batch.to(device)
                batch_size = batch.size(0)

                t = torch.randint(0, timesteps, (batch_size,), device=device)
                noise = torch.randn_like(batch)

                sqrt_alphas_t = torch.sqrt(alphas[t]).view(-1, 1, 1, 1)
                sqrt_one_minus_alphas_t = torch.sqrt(1 - alphas[t]).view(-1, 1, 1, 1)
                x_t = sqrt_alphas_t * batch + sqrt_one_minus_alphas_t * noise

                predicted_noise = model(x_t, t)
                loss = F.mse_loss(predicted_noise, noise)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            
            avg_loss = total_loss / len(dataloader)
            losses.append(avg_loss)
            pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
        
        # Generate and display sample image
        print(f"\nEpoch {(epoch_group+1)*gen_freq}: Loss = {avg_loss:.4f}")
        sample_img = sample_image(model, alphas, device)
        show_image_tensor(sample_img, title=f"Generated at epoch {(epoch_group+1)*gen_freq}")
        print()
    
    print("="*70)
    print("Training complete!")
    
    return losses


class SimpleUNet(nn.Module):
    def __init__(self, channels=[16, 32, 64, 64], time_emb_dim=64):
        super().__init__()

        # Time embedding (unchanged)
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        # Initial conv: 1 input channel instead of 3
        self.conv_in = nn.Conv2d(1, channels[0], 3, padding=1)

        # Encoder (downsampling)
        self.down1 = ResidualBlock(channels[0], channels[1], time_emb_dim)
        self.down2 = ResidualBlock(channels[1], channels[2], time_emb_dim)
        self.down3 = ResidualBlock(channels[2], channels[3], time_emb_dim)

        # Bottleneck
        self.bottleneck = ResidualBlock(channels[3], channels[3], time_emb_dim)

        # Decoder (upsampling)
        self.up3 = ResidualBlock(channels[3] + channels[2], channels[2], time_emb_dim)
        self.up2 = ResidualBlock(channels[2] + channels[1], channels[1], time_emb_dim)
        self.up1 = ResidualBlock(channels[1] + channels[0], channels[0], time_emb_dim)

        # Output: 1 channel instead of 3
        self.conv_out = nn.Conv2d(channels[0], 1, 3, padding=1)

    def forward(self, x, t):
        """
        x: [B, 1, 88, 1024]
        t: [B]
        """
        t_emb = self.time_mlp(t)        # [B, time_emb_dim]

        x0 = self.conv_in(x)

        # Encoder
        x1 = self.down1(F.max_pool2d(x0, 2), t_emb)  # [B, c1, 44, 512]
        x2 = self.down2(F.max_pool2d(x1, 2), t_emb)  # [B, c2, 22, 256]
        x3 = self.down3(F.max_pool2d(x2, 2), t_emb)  # [B, c3, 11, 128]

        # Bottleneck
        x = self.bottleneck(x3, t_emb)

        # Decoder with skip connections
        x = F.interpolate(x, scale_factor=2, mode='nearest')    # [B, c3, 22, 256]
        x = self.up3(torch.cat([x, x2], dim=1), t_emb)

        x = F.interpolate(x, scale_factor=2, mode='nearest')    # [B, c2, 44, 512]
        x = self.up2(torch.cat([x, x1], dim=1), t_emb)

        x = F.interpolate(x, scale_factor=2, mode='nearest')    # [B, c1, 88, 1024]
        x = self.up1(torch.cat([x, x0], dim=1), t_emb)

        return self.conv_out(x)  # [B, 1, 88, 1024]

@torch.no_grad()
def sample_image(model, alphas, device):
    """
    Generate one image through reverse diffusion.
    
    Args:
        model: Trained UNet model
        alphas: Cumulative noise schedule from prepare_noise_schedule
        device: torch device
    
    Returns:
        x: Generated image tensor (1, 88, 1024) normalized to [-1, 1]
    """
    alphas = alphas.to(device)
    t = len(alphas)
    
    # Start from pure noise
    x = torch.randn(1, 88, 1024, device=device)   # 1-channel piano-roll image

    for step in reversed(range(t)):
        a = alphas[step]
        sqrt_a = torch.sqrt(a)
        sqrt_a_diff = torch.sqrt(1 - a).to(device)
        
        x_batch = x.unsqueeze(0)          # [1, 1, 88, 1024]
        t_batch = torch.tensor([step], device=device)

        noise_pred = model(x_batch, t_batch).squeeze(0)  # [1, 88, 1024]

        pred_x0 = (x - sqrt_a_diff * noise_pred) / sqrt_a
        pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

        if step > 0:
            z = torch.randn_like(x)
            a_prev = alphas[step - 1]
            x = torch.sqrt(a_prev) * pred_x0 + torch.sqrt(1 - a_prev) * z
        else:
            x = pred_x0

    return x  # [1, 88, 1024]
