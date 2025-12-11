import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import matplotlib.pyplot as plt
from math import ceil
from tqdm import tqdm
import util
import os

from transformers import (
    GPT2LMHeadModel,
    get_linear_schedule_with_warmup
)

from model_helpers import SinusoidalPositionEmbedding, ResidualBlock
from model_helpers import prepare_noise_schedule, show_image_tensor
from model_helpers import get_batch, estimate_loss
from text_processing import MidiTokenizer


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
    Train a MozartTransformer using get_batch / estimate_loss from
    model_helpers.

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
        device = util.get_best_device()
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
        assert C == vocab_size, f"Logits last dim {C} != "\
            f"vocab_size {vocab_size}"

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

def train_diffusion_with_early_stopping(
        model, train_loader, val_loader, timesteps, num_epochs=200, lr=1e-4,
        gen_freq=10, patience=5, weight_decay=1e-4, device="cpu",
        save_checkpoints=True, alpha_start=1.1, alpha_end=0.5,
        save_dir="models/diffusion_checkpoints"):

    os.makedirs(save_dir, exist_ok=True)

    # Setup
    _, alphas = prepare_noise_schedule(device, timesteps=timesteps)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay)

    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    patience_counter = 0

    num_epoch_groups = ceil(num_epochs / gen_freq)
    print(f"\nTraining for {num_epochs} epochs. "
          f"Validation & checkpoint every {gen_freq} epochs.")
    print(f"Early Stopping Patience: {patience} checks "
          f"(≈{patience * gen_freq} epochs).")
    print("=" * 70)

    # Training Loop
    for epoch_group in range(num_epoch_groups):
        pbar = tqdm(range(gen_freq),
                    desc=f"Block {epoch_group+1}/{num_epoch_groups}")

        # --- TRAIN BLOCK ---
        for epoch_in_group in pbar:
            global_epoch = epoch_group * gen_freq + epoch_in_group
            if global_epoch >= num_epochs:
                break  # safety in case num_epochs not multiple of gen_freq

            total_train_loss = 0.0

            for batch in train_loader:
                batch = batch.to(device)
                batch_size = batch.size(0)

                t = torch.randint(0, timesteps, (batch_size,), device=device)
                noise = torch.randn_like(batch)

                sqrt_alphas_t = torch.sqrt(alphas[t]).view(-1, 1, 1, 1)
                sqrt_one_minus_alphas_t = torch.sqrt(1 - alphas[t]).view(
                    -1, 1, 1, 1)
                x_t = sqrt_alphas_t * batch + sqrt_one_minus_alphas_t * noise

                predicted_noise = model(x_t, t)

                # Custom weighted loss
                note_mask = (batch > -0.5).float()
                progress = global_epoch / num_epochs
                alpha_weight = alpha_start + (
                    alpha_end - alpha_start) * progress
                weight = 1.0 + alpha_weight * note_mask

                mse = (predicted_noise - noise) ** 2
                loss = (weight * mse).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            pbar.set_postfix({'loss': f'{avg_train_loss:.4f}'})

        # If we exited because we hit num_epochs exactly,
        # make sure global_epoch exists
        if global_epoch >= num_epochs:
            global_epoch = num_epochs - 1

        # --- VALIDATION & EARLY STOPPING BLOCK ---
        current_val_loss = validate(
            model, val_loader, alphas, timesteps, device)
        val_losses.append(current_val_loss)

        print(f"\nEpoch {global_epoch+1}: "
              f"Train Loss={avg_train_loss:.4f}, "
              f"Val Loss={current_val_loss:.4f}")

        # Visualization
        samples = [sample_image(model, alphas, device) for _ in range(4)]
        fig, axes = plt.subplots(1, 4, figsize=(12, 3))
        for ax, img in zip(axes.flat, samples):
            show_image_tensor(img, ax=ax)
        plt.show()

        # --- REGULAR CHECKPOINT (every gen_freq epochs) ---
        if save_checkpoints:
            ckpt_path = os.path.join(
                save_dir, f"diffusion_epoch_{global_epoch+1:04d}.pt"
            )
            torch.save(
                {
                    "epoch": global_epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": avg_train_loss,
                    "val_loss": current_val_loss,
                },
                ckpt_path,
            )
            print(f"Saved checkpoint to {ckpt_path}")

        # --- BEST-MODEL CHECKPOINT + EARLY STOPPING ---
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            patience_counter = 0  # Reset counter

            # Save best model weights separately
            if save_checkpoints:
                best_path = os.path.join(save_dir, "best_model.pt")
                torch.save(model.state_dict(), best_path)
                print(f"New best model, saved to {best_path}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience or global_epoch + 1 >= num_epochs:
            print(f"\nStopping early -- validation loss hasn't improved for "
                  f"{patience} checks or max epochs reached.")
            break

    print("=" * 70)
    return train_losses, val_losses


class SimpleUNet(nn.Module):
    def __init__(self, channels=[16, 32, 64, 64], time_emb_dim=64):
        super().__init__()

        # Time embedding (unchanged)
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        # Initial conv: 1 input channel
        self.conv_in = nn.Conv2d(1, channels[0], 3, padding=1)

        # Encoder (downsampling)
        self.down1 = ResidualBlock(channels[0], channels[1], time_emb_dim)
        self.down2 = ResidualBlock(channels[1], channels[2], time_emb_dim)
        self.down3 = ResidualBlock(channels[2], channels[3], time_emb_dim)

        # Bottleneck
        self.bottleneck = ResidualBlock(channels[3], channels[3], time_emb_dim)

        # Decoder (upsampling)
        self.up3 = ResidualBlock(
            channels[3] + channels[2], channels[2], time_emb_dim)
        self.up2 = ResidualBlock(
            channels[2] + channels[1], channels[1], time_emb_dim)
        self.up1 = ResidualBlock(
            channels[1] + channels[0], channels[0], time_emb_dim)

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
        x = F.interpolate(x, scale_factor=2, mode='nearest')    # [B,c3,22,256]
        x = self.up3(torch.cat([x, x2], dim=1), t_emb)

        x = F.interpolate(x, scale_factor=2, mode='nearest')    # [B,c2,44,512]
        x = self.up2(torch.cat([x, x1], dim=1), t_emb)

        x = F.interpolate(x, scale_factor=2, mode='nearest')   # [B,c1,88,1024]
        x = self.up1(torch.cat([x, x0], dim=1), t_emb)

        return self.conv_out(x)  # [B, 1, 88, 1024]


def validate(model, val_loader, alphas, timesteps, device):
    """Computes validation loss (Pure MSE) to check for overfitting."""
    model.eval()
    total_val_loss = 0

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            batch_size = batch.size(0)

            # Same noise process as training
            t = torch.randint(0, timesteps, (batch_size,), device=device)
            noise = torch.randn_like(batch)

            sqrt_alphas_t = torch.sqrt(alphas[t]).view(-1, 1, 1, 1)
            sqrt_one_minus_alphas_t = torch.sqrt(1 - alphas[t]).view(
                -1, 1, 1, 1)
            x_t = sqrt_alphas_t * batch + sqrt_one_minus_alphas_t * noise

            predicted_noise = model(x_t, t)

            # Use standard MSE for validation to get a stable metric
            loss = torch.nn.functional.mse_loss(predicted_noise, noise)
            total_val_loss += loss.item()

    model.train()  # Switch back to train mode
    return total_val_loss / len(val_loader)


@torch.no_grad()
def sample_image(model, alphas, device, img_size=[88, 1024]):
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
    x = torch.randn(1, img_size[0], img_size[1], device=device)  # 1 channel

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


# ----- PRETRAINED GPT-2 TRANSFORMER MODEL CLASSES AND METHODS -----

def train_gpt_2(model, train_loader, val_loader, num_epochs=5, lr=3e-4,
                weight_decay=0.01, device: str | torch.device = None,
                model_save_dir: str | None = None):
    """
    Train GPT-2.

    Args:
        model: GPT2LMHeadModel
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: maximum number of epochs
        lr: learning rate
        weight_decay: weight decay for AdamW
        device: torch.device or string; if None, uses util.get_best_device()
        model_save_dir: if not None, best model is saved in this directory with
                        .save_pretrained()
    """
    if device is None:
        device = util.get_best_device()

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = num_epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps,
    )

    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}",
                    leave=True)

        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()

            pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                total_val_loss += outputs.loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{num_epochs} | train loss: \
              {avg_train_loss:.4f} | val loss: {avg_val_loss:.4f}")

        if model_save_dir is not None:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                print("  -> saving best model")
                os.makedirs(model_save_dir, exist_ok=True)
                model.save_pretrained(model_save_dir)


def generate_midi_tokens_with_gpt_model(
        prompt_text: str,
        vocab_file: str,
        model_save_dir: str,
        max_new_tokens: int = 1024,
        temp: float = 0.5,
        top_k: int = 50,
        device: str | torch.device = None
) -> str:

    if device is None:
        device = util.get_best_device()

    tok = MidiTokenizer(vocab_file)
    mdl = GPT2LMHeadModel.from_pretrained(model_save_dir).to(device)
    mdl.eval()

    prompt_ids = tok.encode(prompt_text, add_special_tokens=False)

    # Build input: [BOS] + prompt tokens (no EOS)
    input_ids = torch.tensor([[tok.bos_token_id] + prompt_ids],
                             dtype=torch.long).to(device)
    attention_mask = (input_ids != tok.pad_token_id).long()

    with torch.no_grad():
        output_ids = mdl.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temp,
            top_k=top_k,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )

    generated_ids = output_ids[0].tolist()
    generated_text = tok.decode(generated_ids, skip_special_tokens=True)
    return generated_text
