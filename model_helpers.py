import torch

def get_batch(split, data, block_size=128, batch_size=32, device="cpu"):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size]     for i in ix]).to(device)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]).to(device)
    return x, y

def estimate_loss(model, vocab_size, device="cpu"):
    model.eval()
    out = {}
    with torch.no_grad():
        for split in ["train", "val"]:
            losses = []
            correct = 0
            total = 0
            for _ in range(10):
                xb, yb = get_batch(split, device=device)
                logits = model(xb)
                loss = F.cross_entropy(
                    logits.view(-1, vocab_size),
                    yb.view(-1)
                )
                losses.append(loss.item())

                # Accuracy
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == yb).float().sum().item()
                total += yb.numel()
                
            avg_loss = sum(losses) / len(losses)
            accuracy = correct / total
            out[split] = (avg_loss, accuracy)
    model.train()
    return out