from collections import Counter
from torch.utils.data import Dataset
import torch
import os


class MidiTokenizer:
    def __init__(self, vocab_file: str):
        with open(vocab_file, "r", encoding="utf-8") as f:
            tokens = [line.strip() for line in f if line.strip()]

        self.tokens = tokens
        self.token_to_id = {tok: i for i, tok in enumerate(tokens)}
        self.id_to_token = {i: tok for tok, i in self.token_to_id.items()}

        # Special tokens
        self.pad_token = "<pad>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"

        self.pad_token_id = self.token_to_id[self.pad_token]
        self.bos_token_id = self.token_to_id[self.bos_token]
        self.eos_token_id = self.token_to_id[self.eos_token]
        self.unk_token_id = self.token_to_id[self.unk_token]

    def get_vocab(self):
        return self.token_to_id

    def encode(self, text: str, add_special_tokens: bool = True):
        tokens = text.strip().split()
        ids = []
        if add_special_tokens:
            ids.append(self.bos_token_id)
        for t in tokens:
            ids.append(self.token_to_id.get(t, self.unk_token_id))
        if add_special_tokens:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, token_ids, skip_special_tokens: bool = True):
        toks = []
        special_ids = {
            self.pad_token_id,
            self.bos_token_id,
            self.eos_token_id,
            self.unk_token_id,
        }
        for i in token_ids:
            i = int(i)
            if skip_special_tokens and i in special_ids:
                continue
            tok = self.id_to_token.get(i, self.unk_token)
            toks.append(tok)
        return " ".join(toks)


class MidiTextDataset(Dataset):
    def __init__(
        self, data_dir: str, tokenizer: MidiTokenizer,
        block_size: int = 512
    ):
        self.samples = []
        self.tokenizer = tokenizer
        self.block_size = block_size

        if not os.path.isdir(data_dir):
            raise ValueError(f"Data directory {data_dir} does not exist")

        for root, _, files in os.walk(data_dir):
            for fname in files:
                if not fname.endswith(".txt"):
                    continue
                path = os.path.join(root, fname)
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        text = line.strip()
                        if not text:
                            continue
                        ids = tokenizer.encode(text, add_special_tokens=True)
                        # break long sequences into fixed chunks
                        for i in range(0, len(ids), block_size):
                            chunk = ids[i: i + block_size]
                            self.samples.append(chunk)

        print(f"Loaded {len(self.samples)} sequences from {data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids = self.samples[idx]
        pad_id = self.tokenizer.pad_token_id

        if len(ids) < self.block_size:
            ids = ids + [pad_id] * (self.block_size - len(ids))
        else:
            ids = ids[: self.block_size]

        input_ids = torch.tensor(ids, dtype=torch.long)
        labels = input_ids.clone()
        return {"input_ids": input_ids, "labels": labels}


def build_vocab_from_dir(data_dir):
    counter = Counter()
    for split in ["train", "val"]:
        split_dir = os.path.join(data_dir, split)
        if not os.path.isdir(split_dir):
            continue
        for root, _, files in os.walk(split_dir):
            for fname in files:
                if not fname.endswith(".txt"):
                    continue
                path = os.path.join(root, fname)
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        tokens = line.strip().split()
                        counter.update(tokens)
    return counter
