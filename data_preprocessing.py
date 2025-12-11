from pathlib import Path
from torch.utils.data import Dataset
from midi_conversion import (
    midi_to_text,
    midi_to_pianoroll_images,
    extract_notes_and_meta
)
import util
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
        raise ValueError(
            "composer must be a string or list/tuple/set of strings")

    paths = [train_path, val_path, test_path]
    # Index 0 contains train midis, 1 contains val midis,
    # 2 contains test midis.
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


def midi_split_to_text_split(
        midis: list[list[tuple[mido.MidiFile, str]]],
        save_to_directory: str = None,
        splits: list[str] = ["train", "val", "test"],
        verbose: bool = True
) -> list[list[str]]:
    """
    :param midis: lists within a larger list; each list
        contains (MIDI file, composer) tuples for the respective section.
        The sections are "train", "val", and "test" by default.
    :param save_to_directory: Leave as None to not save the output as .txt
        files. Otherwise, saves text files to section subfolders of the given
        path, with section names matching the terms of param splits.
    :param splits: defines the # and labels of sections.
    :type midis: list[ list[ tuple[mido.MidiFile, str] ] ]
    :type save_to_directory: str
    :return: list[ list[train texts], list[val texts], list[test texts] ]
    :rtype: list[list[str]]
    """
    text_list = []

    for i in range(max(len(midis), len(splits))):
        text_list.append([])

    for i in range(len(midis)):
        text_list[i] = process_midis_to_text(midis[i])

    if save_to_directory is not None:

        # If not enough split names defined in param 'splits'
        if len(splits) < len(midis):
            print(f"Error: could not save, since splits param defines \
                  [{len(splits)}] split names, while midis param list has \
                  [{len(midis)}] splits.")
            return text_list

        util.mkdir(save_to_directory)

        for split_idx, split_name in enumerate(splits):
            split_dir = util.path_join(save_to_directory, split_name)
            util.mkdir(split_dir)

            for i, text in enumerate(text_list[split_idx]):
                util.write_txt_file(
                    split_dir, f"{split_name}_{i:04d}.txt", text)

            if verbose:
                print(f"Saved {len(text_list[split_idx])} files to \
                      {split_dir}")

    return text_list


def process_midis_to_text(midis: list[tuple[mido.MidiFile, str]]):
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


def process_midis_to_images(midis: list[mido.MidiFile]):
    """
    Process a series of MIDI files into images by calling the
    midi_to_pianoroll_images method from midi_conversion.py

    :param midis: List of individual mido.MidiFile objects to convert.
                  (No composer info included)
    """
    images = []

    for i, midi in enumerate(midis):
        midi_images = midi_to_pianoroll_images(midi)
        for image in midi_images:
            images.append(image)
        print(f"Processed {i}/{len(midis)} MIDI files", end="\r")

    print(
        f"Successfully processed {len(midis)} MIDIs into {len(images)} images."
    )

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
        self.train_ids = torch.tensor(
            [self.stoi[t] for t in all_tokens], dtype=torch.long)

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


class PianoRollDatasetWithMetadata(Dataset):
    """
    Dataset for piano-roll images saved as grayscale PNGs.
    Includes piece metadata, which is used for
    classifier-free guidance training for diffusion models.
    Each sample: tensor [1, 88, x] normalized to [-1, 1].
    x should be divisible by 8, for UNet pooling.
    All samples must be the same image size.
    """

    def __init__(self, data_dir, composer_map, key_map):
        self.data_dir = Path(data_dir)
        self.image_files = sorted(self.data_dir.glob("*.png"))
        if not self.image_files:
            raise ValueError(f"No images found in {data_dir}")

        print(f"Found {len(self.image_files)} images in {data_dir}")

        self.composer_map = composer_map  # e.g. {'haydn': 0, 'mozart': 1...}
        self.key_map = key_map            # e.g. {'C': 0, 'Am': 1...}

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

        # Parse filename: haydn_C_120_train_0_0.png
        fname = self.image_files[idx].name
        parts = fname.split('_')

        composer_label = self.composer_map.get(parts[0], 0)
        key_label = self.key_map.get(parts[1], 0)

        # Tempo is continuous, normalize it (e.g., 0 to 1)
        tempo_val = float(parts[2]) / 200.0

        return {
            "x": x,
            "composer": torch.tensor(composer_label).long(),
            "key": torch.tensor(key_label).long(),
            "tempo": torch.tensor(tempo_val).float()
        }


def midi_objs_to_images(
    midi_objs: list[mido.MidiFile],
    data_dir: str,
    subfolder_name: str
):
    """
    Converts a list of mido.MidiFile objects into images, which are saved in
    a subfolder of the given directory.
    """
    images = process_midis_to_images(midi_objs)
    output_path = f"{data_dir}/{subfolder_name}"
    util.mkdir(output_path)
    for i, img in enumerate(images):
        img.save(
            util.path_join(output_path, f"{subfolder_name}_window_{i:03d}.png")
        )


def midi_objs_to_images_with_metadata(
    midi_objs: list,
    data_dir: str,
    subfolder_name: str
):
    output_path = f"{data_dir}/{subfolder_name}"
    util.mkdir(output_path)

    # Extract metadata
    for i, (midi, composer) in enumerate(midi_objs):
        _, meta = extract_notes_and_meta(midi)
        key_str = meta["key"] if meta["key"] else "Unknown"
        tempo_val = meta["tempo"] if meta["tempo"] else 120

        # Get images for this specific midi
        images = midi_to_pianoroll_images(midi)

        fname_base = f"{composer}_{key_str}_{tempo_val}_{subfolder_name}_{i}_"

        for j, img in enumerate(images):
            # Filename format
            fname = f"{fname_base}{j}.png"
            img.save(util.path_join(output_path, fname))
