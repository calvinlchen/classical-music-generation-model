import mido

# Given a MIDI file through mido, return a list of its messages in chronological order
# Return format: [(abs_time, track_index, msg), ...]
def get_chronological_messages(mido_file):
    events = []

    for track_idx, track in enumerate(mido_file.tracks):
        abs_time = 0
        for msg in track:
            abs_time += msg.time
            events.append((abs_time, track_idx, msg))

    # Sort messages by absolute time
    events.sort(key=lambda x: (x[0], x[1]))

    return events

def midi_to_text(mido_file, composer=None):
    mid = mido_file
    ticks_per_beat = mid.ticks_per_beat
    events = get_chronological_messages(mid)
    
    text_tokens = []
    current_pos = None
    current_notes = []
    current_beat = None

    if composer:
        try:
            text_tokens.append(f"COMPOSER_{composer}")
        except:
            print("Error: Could not add composer to text.")

    text_tokens.append(f"TICKS_PER_BEAT_{ticks_per_beat}")
    
    for abs_ticks, track_idx, msg in events:
        if msg.type in ['note_on', 'note_off']:
            beat_number = abs_ticks // ticks_per_beat
    
            if beat_number != current_beat:
                # flush previous beat's notes
                if current_notes:
                    text_tokens.append(f"POS_{current_pos} " + " ".join(current_notes))
                    current_notes = []
    
                # emit POS_0 only for *fully empty* beats between old and new
                if current_beat is not None:
                    for empty_beat in range(current_beat + 1, beat_number):
                        text_tokens.append("POS_0")
    
                current_beat = beat_number
                current_pos = None  # important: don't pre-set to "0"
            
            pos = (abs_ticks % ticks_per_beat) / ticks_per_beat
            pos_str = f"{pos:.3f}".rstrip("0").rstrip(".")
    
            note_name = f"NOTE{msg.note}"
            suffix = "ON" if (msg.type == 'note_on' and msg.velocity > 0) else "OFF"
    
            if pos_str != current_pos:
                if current_notes:
                    text_tokens.append(f"POS_{current_pos} " + " ".join(current_notes))
                current_pos = pos_str
                current_notes = [f"{note_name}_{suffix}"]
            else:
                current_notes.append(f"{note_name}_{suffix}")
        elif msg.type == 'set_tempo':
            text_tokens.append(f"TEMPO_BPM_{round(mido.tempo2bpm(msg.tempo))}")
        elif msg.type == 'key_signature':
            text_tokens.append(f"KEY_{msg.key}")
        elif msg.type == 'time_signature':
            text_tokens.append(f"TIME_SIGNATURE_{msg.numerator}/{msg.denominator}")
                
    # Flush last group
    if current_notes:
        text_tokens.append(f"POS_{current_pos} " + " ".join(current_notes))

    # TODO: convert instruments
    
    return " ".join(text_tokens)


def text_to_midi(text: str) -> mido.MidiFile:
    tokens = text.split()

    # ---------- 1) First pass: extract global metadata ----------
    ticks_per_beat = 480
    tempo = mido.bpm2tempo(120)  # default 120 bpm
    time_sig = (4, 4)
    key = None
    composer = None

    for tok in tokens:
        if tok.startswith("TICKS_PER_BEAT_"):
            try:
                ticks_per_beat = int(tok[len("TICKS_PER_BEAT_"):])
            except ValueError:
                print("ValueError: Ticks per beat")
                pass
        elif tok.startswith("TEMPO_BPM_"):
            # just use the last one we see as "global"
            try:
                bpm = float(tok[len("TEMPO_BPM_"):])
                tempo = mido.bpm2tempo(bpm)
            except ValueError:
                print("ValueError: Tempo")
                pass
        elif tok.startswith("TIME_SIGNATURE_"):
            sig = tok[len("TIME_SIGNATURE_"):]  # "4/4"
            if "/" in sig:
                num_str, den_str = sig.split("/", 1)
                try:
                    time_sig = (int(num_str), int(den_str))
                except ValueError:
                    print("ValueError: Time signature")
                    pass
        elif tok.startswith("KEY_"):
            key = tok[len("KEY_"):]
        elif tok.startswith("COMPOSER_"):
            composer = tok[len("COMPOSER_"):]

    # ---------- 2) Second pass: parse POS groups & notes ----------
    events = []  # (abs_ticks, mido.Message)
    current_beat = 0
    last_pos = None  # last position within current beat

    meta_prefixes = (
        "TICKS_PER_BEAT_",
        "TEMPO_BPM_",
        "TIME_SIGNATURE_",
        "KEY_",
        "COMPOSER_",
    )

    i = 0
    while i < len(tokens):
        tok = tokens[i]

        if tok.startswith("POS_"):
            # Start of a POS group
            pos_str = tok[4:]  # after "POS_"
            pos_str = pos_str if pos_str != "" else "0"
            i += 1

            group_notes = []
            # Collect note tokens until next POS_ or meta token
            while (
                i < len(tokens)
                and not tokens[i].startswith("POS_")
                and not any(tokens[i].startswith(p) for p in meta_prefixes)
            ):
                group_notes.append(tokens[i])
                i += 1

            # Empty beat marker: "POS_0" with no notes
            if not group_notes:
                current_beat += 1
                last_pos = None
                continue

            # Non-empty group: notes at some position within a beat
            pos_val = float(pos_str) if pos_str != "" else 0.0

            if last_pos is None:
                # first position in this beat
                pass
            else:
                # if position goes backwards, we started a new beat
                if pos_val < last_pos:
                    current_beat += 1
            last_pos = pos_val

            tick_in_beat = int(round(pos_val * ticks_per_beat))
            abs_ticks = current_beat * ticks_per_beat + tick_in_beat

            for nt in group_notes:
                if not nt.startswith("NOTE"):
                    continue
                base, suffix = nt.split("_", 1)  # e.g. "NOTE71", "ON"
                note_num = int(base[4:])        # strip "NOTE"

                if suffix == "ON":
                    msg = mido.Message("note_on", note=note_num, velocity=64, time=0)
                else:
                    msg = mido.Message("note_off", note=note_num, velocity=64, time=0)

                events.append((abs_ticks, msg))

        else:
            # skip meta tokens here (already handled)
            i += 1

    # ---------- 3) Build the MIDI file ----------
    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    # Optional: put meta messages at time 0
    if composer is not None:
        track.append(mido.MetaMessage("text", text=f"COMPOSER_{composer}", time=0))

    track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))

    num, den = time_sig
    track.append(
        mido.MetaMessage(
            "time_signature",
            numerator=num,
            denominator=den,
            clocks_per_click=24,
            notated_32nd_notes_per_beat=8,
            time=0,
        )
    )

    if key is not None:
        track.append(mido.MetaMessage("key_signature", key=key, time=0))

    # Sort events and convert to delta times
    events.sort(key=lambda x: x[0])
    last_tick = 0
    for abs_ticks, msg in events:
        delta = abs_ticks - last_tick
        msg.time = delta
        last_tick = abs_ticks
        track.append(msg)

    return mid


def midi_to_note_name(midi_note_number):
    """Converts a MIDI note number to its corresponding note name (e.g., C4)."""
    octave = (midi_note_number // 12) - 1  # MIDI note 0 is C-1, so adjust octave
    note_index = midi_note_number % 12
    note_name = NOTE_NAMES[note_index]
    return f"{note_name}{octave}"
    