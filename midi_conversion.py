import mido

# Constants
TARGET_TPB = 480          # standardized ticks per beat
GRID_RESOLUTION = 48      # grid slots per beat (480 / 48 = 10 ticks per grid)
MAX_DURATION_BEATS = 8    # cap duration at this many beats
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def midi_to_note_name(midi_note_number: int) -> str:
    """Converts a MIDI note number to its corresponding note name (e.g., C4)."""
    octave = (midi_note_number // 12) - 1  # MIDI note 0 is C-1
    note_index = midi_note_number % 12
    note_name = NOTE_NAMES[note_index]
    return f"{note_name}{octave}"


def normalize_midi(mid: mido.MidiFile) -> mido.MidiFile:
    """
    Return a copy of `mid` whose ticks_per_beat is TARGET_TPB.
    Rescales all delta times to preserve musical timing.
    """
    if mid.ticks_per_beat == TARGET_TPB:
        return mid

    scale = TARGET_TPB / mid.ticks_per_beat
    new_mid = mido.MidiFile(ticks_per_beat=TARGET_TPB)

    for track in mid.tracks:
        new_track = mido.MidiTrack()
        abs_scaled = 0
        last_scaled = 0
        for msg in track:
            abs_scaled += msg.time * scale
            new_msg = msg.copy()
            if hasattr(new_msg, "time"):
                # Convert from absolute scaled time back to delta
                scaled_int = int(round(abs_scaled))
                new_msg.time = scaled_int - last_scaled
                last_scaled = scaled_int
            new_track.append(new_msg)
        new_mid.tracks.append(new_track)

    return new_mid


def extract_notes_and_meta(mid: mido.MidiFile):
    """
    From a (normalized) MidiFile, return:
      notes: list of dicts {start, duration, note, velocity}
      meta:  dict with optional 'tempo', 'key', 'time_sig'
    """
    mid = normalize_midi(mid)

    active_notes = {}  # (note, channel) -> list[(start_tick, velocity)]
    notes = []
    meta = {"tempo": None, "key": None, "time_sig": None}

    for track in mid.tracks:
        abs_ticks = 0
        for msg in track:
            abs_ticks += msg.time

            if msg.type == "set_tempo" and meta["tempo"] is None:
                meta["tempo"] = round(mido.tempo2bpm(msg.tempo))

            elif msg.type == "key_signature" and meta["key"] is None:
                meta["key"] = msg.key

            elif msg.type == "time_signature" and meta["time_sig"] is None:
                meta["time_sig"] = (msg.numerator, msg.denominator)

            elif msg.type == "note_on" and msg.velocity > 0:
                key = (msg.note, getattr(msg, "channel", 0))
                active_notes.setdefault(key, []).append((abs_ticks, msg.velocity))

            elif msg.type in ("note_off", "note_on"):  # note_on with vel=0 = note_off
                if msg.type == "note_on" and msg.velocity > 0:
                    # already handled as note_on above
                    continue
                key = (msg.note, getattr(msg, "channel", 0))
                if key in active_notes and active_notes[key]:
                    start_tick, vel = active_notes[key].pop()
                    duration_ticks = max(1, abs_ticks - start_tick)
                    notes.append({
                        "start": start_tick,
                        "duration": duration_ticks,
                        "note": msg.note,
                        "velocity": vel,
                    })

    # Sort by start time then pitch
    notes.sort(key=lambda n: (n["start"], n["note"]))
    return notes, meta


def midi_to_text(mido_file: mido.MidiFile, composer: str | None = None) -> str:
    """
    Convert a MIDI file to a token sequence with:
    - COMPOSER_x, KEY_x, TIME_SIGNATURE_n/d, TEMPO_BPM_x
    - MEASURE, BEAT, POS_grid
    - NOTE_pitch, DUR_grids, VEL_bin
    """
    notes, meta = extract_notes_and_meta(mido_file)

    ticks_per_grid = TARGET_TPB // GRID_RESOLUTION
    max_duration_grids = MAX_DURATION_BEATS * GRID_RESOLUTION

    # Beats per bar from time signature (default 4/4 if missing)
    if meta["time_sig"] is not None:
        beats_per_bar = meta["time_sig"][0]
    else:
        beats_per_bar = 4

    tokens: list[str] = []

    # Global metadata tokens
    if composer:
        tokens.append(f"COMPOSER_{composer}")
    if meta["key"] is not None:
        tokens.append(f"KEY_{meta['key']}")
    if meta["time_sig"] is not None:
        num, den = meta["time_sig"]
        tokens.append(f"TIME_SIGNATURE_{num}/{den}")
    if meta["tempo"] is not None:
        tokens.append(f"TEMPO_BPM_{meta['tempo']}")

    # Time tracking (global beat index, measure index, sub-beat position)
    current_global_beat = -1  # increments for every BEAT
    current_measure_idx = -1
    last_grid_index = -1

    for n in notes:
        start_ticks = n["start"]

        # Global beat index & sub-beat offset
        target_beat = start_ticks // TARGET_TPB
        offset_within_beat = start_ticks % TARGET_TPB

        # Quantize to nearest grid (0..GRID_RESOLUTION-1)
        grid_index = (offset_within_beat + ticks_per_grid // 2) // ticks_per_grid
        if grid_index >= GRID_RESOLUTION:
            # rounding pushed us past the last grid, move to next beat
            target_beat += 1
            grid_index = 0

        # Emit MEASURE / BEAT tokens to "walk" from current_global_beat → target_beat
        while current_global_beat < target_beat:
            current_global_beat += 1
            measure_idx = current_global_beat // beats_per_bar

            # New bar?
            if measure_idx != current_measure_idx:
                tokens.append("MEASURE")
                current_measure_idx = measure_idx

            tokens.append("BEAT")
            last_grid_index = -1  # reset POS for new beat

        # Now we are at the correct beat; emit POS if grid changed
        if grid_index != last_grid_index:
            tokens.append(f"POS_{grid_index}")
            last_grid_index = grid_index

        # Duration (in grid units)
        dur_grids = max(1, int(round(n["duration"] / ticks_per_grid)))
        dur_grids = min(dur_grids, max_duration_grids)

        # Velocity binned into 8 levels (0–7)
        vel_bin = n["velocity"] // 16

        tokens.append(f"NOTE_{n['note']}")
        tokens.append(f"DUR_{dur_grids}")
        tokens.append(f"VEL_{vel_bin}")

    return " ".join(tokens)


def text_to_midi(text: str) -> mido.MidiFile:
    """
    Convert a token sequence with MEASURE/BEAT/POS/NOTE/DUR/VEL back into a MidiFile.
    Assumes the format generated by `midi_to_text`.
    """
    tokens = text.split()

    ticks_per_beat = TARGET_TPB
    ticks_per_grid = ticks_per_beat // GRID_RESOLUTION

    # First pass: Extract global metadata
    tempo = mido.bpm2tempo(120.0)   # default 120 bpm
    time_sig = (4, 4)               # default
    key = None
    composer = None

    for tok in tokens:
        if tok.startswith("TEMPO_BPM_"):
            try:
                bpm = float(tok[len("TEMPO_BPM_"):])
                tempo = mido.bpm2tempo(bpm)
            except ValueError:
                pass
        elif tok.startswith("TIME_SIGNATURE_"):
            sig = tok[len("TIME_SIGNATURE_"):]  # e.g. "4/4"
            if "/" in sig:
                num_str, den_str = sig.split("/", 1)
                try:
                    time_sig = (int(num_str), int(den_str))
                except ValueError:
                    pass
        elif tok.startswith("KEY_"):
            key = tok[len("KEY_"):]
        elif tok.startswith("COMPOSER_"):
            composer = tok[len("COMPOSER_"):]

    beats_per_bar = time_sig[0]
    meta_prefixes = ("TEMPO_BPM_", "TIME_SIGNATURE_", "KEY_", "COMPOSER_")

    # Second pass: Timeline & note events
    events: list[tuple[int, mido.Message]] = []  # (abs_ticks, message)

    global_beat = -1      # advanced by BEAT tokens
    grid_index = 0        # last POS within the current beat

    i = 0
    while i < len(tokens):
        tok = tokens[i]

        if tok == "MEASURE":
            # Structural marker only; timing is driven by BEAT tokens
            i += 1
            continue

        if tok == "BEAT":
            global_beat += 1
            grid_index = 0
            i += 1
            continue

        if tok.startswith("POS_"):
            # Set sub-beat position (grid index)
            grid_str = tok[4:]
            try:
                grid_index = int(grid_str)
            except ValueError:
                grid_index = 0

            i += 1

            # Read NOTE/DUR/VEL triples until POS/MEASURE/BEAT/meta or end
            while (
                i < len(tokens)
                and tokens[i] not in ("MEASURE", "BEAT")
                and not tokens[i].startswith("POS_")
                and not any(tokens[i].startswith(p) for p in meta_prefixes)
            ):
                if not tokens[i].startswith("NOTE_"):
                    # Skip unexpected tokens inside POS group
                    i += 1
                    continue

                # NOTE
                note_tok = tokens[i]
                try:
                    note_num = int(note_tok[len("NOTE_"):])
                except ValueError:
                    note_num = 60  # default middle C on error
                i += 1

                # DUR
                dur_grids = 1
                if i < len(tokens) and tokens[i].startswith("DUR_"):
                    try:
                        dur_grids = int(tokens[i][len("DUR_"):])
                    except ValueError:
                        dur_grids = 1
                    i += 1

                # VEL
                vel_bin = 4  # mid-level default
                if i < len(tokens) and tokens[i].startswith("VEL_"):
                    try:
                        vel_bin = int(tokens[i][len("VEL_"):])
                    except ValueError:
                        vel_bin = 4
                    i += 1

                # Ensure we have at least beat 0 if no BEAT was seen
                if global_beat < 0:
                    global_beat = 0

                base_tick = global_beat * ticks_per_beat + grid_index * ticks_per_grid
                duration_ticks = max(1, dur_grids * ticks_per_grid)

                # Map velocity bin 0–7 back to 1–127
                velocity = max(1, min(127, vel_bin * 16 + 8))

                start_tick = base_tick
                end_tick = base_tick + duration_ticks

                events.append(
                    (start_tick, mido.Message("note_on", note=note_num, velocity=velocity, time=0))
                )
                events.append(
                    (end_tick, mido.Message("note_off", note=note_num, velocity=0, time=0))
                )

            continue  # done with this POS group; move to next token

        # Any other token here is meta or unknown; just skip
        i += 1

    # Build the MidiFile
    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    if composer is not None:
        track.append(mido.MetaMessage("text", text=f"COMPOSER_{composer}", time=0))

    # Meta events at time 0
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

    # Sort note events and convert absolute ticks -> delta times
    events.sort(key=lambda x: (x[0], 0 if x[1].type == "note_off" else 1, getattr(x[1], "note", 0)))

    last_tick = 0
    for abs_ticks, msg in events:
        delta = abs_ticks - last_tick
        msg.time = delta
        last_tick = abs_ticks
        track.append(msg)

    return mid
