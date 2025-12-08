from pathlib import Path
from midi_conversion import midi_to_text
import mido

# Constants
train_path = Path("data/train")
val_path = Path("data/val")
test_path = Path("data/test")
    
SEQ_SOS = "<SOS>"
SEQ_EOS = "<EOS>"


def get_midis_by_composer(composer):
    """
    Get all midis for a particular composer within default /data filepath.

    :param composer: lowercase composer name at the start of each file
    """
    paths = [train_path, val_path, test_path]
    # Index 0 contains train midis, 1 contains val midis, 2 contains test midis.
    train_val_test_midis = [[], [], []]
    total_midis = 0

    for i, path in enumerate(paths):
        print(f"Now loading MIDIs from {path}.")
        for midi_file in path.glob("*.mid"):
            if composer in midi_file.name.lower():
                try:
                    midi_obj = mido.MidiFile(midi_file)
                    train_val_test_midis[i].append(midi_obj)
                except Exception as e:
                    print(f"Could not load {midi_file}: {e}")    
        total_midis += len(train_val_test_midis[i])        
        print(f"Loaded {len(train_val_test_midis[i])} MIDI files from {path}")

    print(f"{total_midis} MIDI files retrieved.")
    return train_val_test_midis


def process_midis_to_text(midis, composer=None):
    """
    Process a series of MIDI files into text by calling the
    midi_to_text method from midi_conversion.py
    
    :param midis: List of mido MIDI objects to convert
    :param composer: Composer name
    """
    texts = []

    for i, midi in enumerate(midis):
        texts.append(midi_to_text(midi, composer))
        print(f"Processed {i}/{len(midis)} files", end="\r")

    texts = [f"{SEQ_SOS} {text} {SEQ_EOS}" for text in texts]

    print(f"Successfully processed {len(midis)} MIDIs into text.")

    return texts

