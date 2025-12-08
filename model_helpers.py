import torch
import torch.nn.functional as F


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