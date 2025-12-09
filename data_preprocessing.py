from pathlib import Path
from torch.utils.data import Dataset
from midi_conversion import midi_to_text, midi_to_pianoroll_images
import mido
from PIL import Image
import numpy as np
import torch

# Constants
train_path = Path("data/train")
val_path = Path("data/val")
test_path = Path("data/test")
    
SEQ_SOS = "<SOS>"
SEQ_EOS = "<EOS>"


def get_midis_by_composer(composer):
    """
    Get all MIDIs for one or more composers within default /data filepath.

    Returns [[train midis], [val midis], [test midis]], where within each
    train/val/test midi list at each index is (midi, composer)

    :param composer: either a single lowercase composer string 
                     or a list of lowercase composer strings
    """
    # Normalize to list
    if isinstance(composer, str):
        composers = [composer]
    elif isinstance(composer, (list, tuple, set)):
        composers = [c.lower() for c in composer]
    else:
        raise ValueError("composer must be a string or list/tuple/set of strings")
    
    paths = [train_path, val_path, test_path]
    # Index 0 contains train midis, 1 contains val midis, 2 contains test midis.
    train_val_test_midis = [[], [], []]
    total_midis = 0

    for i, path in enumerate(paths):
        print(f"Now loading MIDIs from {path}.")
        for midi_file in path.glob("*.mid"):
            filename = midi_file.name.lower()
            for c in composers:
                if c in filename:
                    try:
                        midi_obj = mido.MidiFile(midi_file)
                        train_val_test_midis[i].append((midi_obj, c))
                    except Exception as e:
                        print(f"Could not load {midi_file}: {e}")    
        total_midis += len(train_val_test_midis[i])        
        print(f"Loaded {len(train_val_test_midis[i])} MIDI files from {path}")

    print(f"{total_midis} MIDI files retrieved.")
    return train_val_test_midis

def process_midis_to_text(midis):
    """
    Process a series of MIDI files into text by calling the
    midi_to_text method from midi_conversion.py
    
    :param midis: List of midis in (mido object, composer) format to convert
    """
    texts = []

    for i, (midi, composer) in enumerate(midis):
        texts.append(midi_to_text(midi, composer))
        print(f"Processed {i}/{len(midis)} files", end="\r")

    texts = [f"{SEQ_SOS} {text} {SEQ_EOS}" for text in texts]

    print(f"Successfully processed {len(midis)} MIDIs into text.")

    return texts

def process_midis_to_images(midis):
    """
    Process a series of MIDI files into images by calling the
    midi_to_pianoroll_images method from midi_conversion.py
    
    :param midis: List of individual mido.MidiObject objects to convert.
                  (No composer info included)
    """
    images = []

    for i, midi in enumerate(midis):
        midi_images = midi_to_pianoroll_images(midi)
        for image in midi_images:
            images.append(image)
        print(f"Processed {i}/{len(midis)} MIDI files", end="\r")

    print(f"Successfully processed {len(midis)} MIDIs into {len(images)} images.")

    return images


class VocabBuilder:
    def __init__(self, train_seqs, add_unknown_token=True):
        """
        Build vocabulary from training sequences only.
        
        :param train_seqs: list of strings (training text sequences)
        :param add_unknown_token: whether to include <UNK>
        """
        all_tokens = []
        for s in train_seqs:
            all_tokens.extend(s.split())

        self.vocab = sorted(set(all_tokens))
        
        if add_unknown_token:
            self.vocab.extend(["<UNK>"])
        
        self.stoi = {t: i for i, t in enumerate(self.vocab)}
        self.itos = {i: t for t, i in self.stoi.items()}
        self.vocab_size = len(self.vocab)

        print(f"Vocabulary size (train only): {self.vocab_size}")

        # Store training IDs
        self.train_ids = torch.tensor([self.stoi[t] for t in all_tokens], dtype=torch.long)

    def encode(self, text: str):
        """Convert text string → list of token IDs, map unknowns to <UNK>"""
        return [self.stoi.get(t, self.stoi["<UNK>"]) for t in text.split()]

    def decode(self, ids):
        """Convert list of token IDs → text string"""
        return " ".join(self.itos[int(i)] for i in ids)


class PianoRollDataset(Dataset):
    """
    Dataset for piano-roll images saved as grayscale PNGs.
    Each sample: tensor [1, 88, x] normalized to [-1, 1].
    x should be divisible by 8, for UNet pooling.
    All samples must be the same image size.
    """

    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.image_files = sorted(self.data_dir.glob("*.png"))
        if not self.image_files:
            raise ValueError(f"No images found in {data_dir}")

        print(f"Found {len(self.image_files)} images in {data_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img = Image.open(self.image_files[idx]).convert("L")  # grayscale
        arr = np.array(img, dtype=np.float32)                 # [H, W]
        H, W = arr.shape

        # Make sure size works with 3 levels of 2x2 pooling
        assert H % 8 == 0 and W % 8 == 0, f"Bad size {H}x{W} for UNet"

        arr = arr / 255.0          # [0,1]
        arr = arr * 2.0 - 1.0      # [-1,1]

        # [1, 88, 1024] using default sizes
        x = torch.from_numpy(arr).unsqueeze(0)
        return x
