# backend/main.py
import io
import base64
import torch
import numpy as np
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from pydantic import BaseModel

import os
from openai import OpenAI
from dotenv import load_dotenv

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent  # already implicitly used
TEXT_DIR = ROOT_DIR / "data" / "example_midi_texts"

# Add parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from data_preprocessing import VocabBuilder
from models import SimpleUNet, sample_image, MidiTextTransformer
from models import generate_midi_tokens_with_transformer
from model_helpers import prepare_noise_schedule
from midi_conversion import pianoroll_images_to_midi, text_to_midi

app = FastAPI()


class TransformerRequest(BaseModel):
    start_text: str


# Allow React dev server
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained diffusion model checkpoint once at startup
CHECKPOINT_PATH = "../models/diffusion_12000/diffusion_epoch_0010.pt"

diff_model = SimpleUNet().to(DEVICE)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
diff_model.load_state_dict(checkpoint["model_state_dict"])
diff_model.eval()

# Load vocab for transformer model
vocab_data = torch.load("../models/transformer/vocab.pt")

# Reconstruct VocabBuilder
vb = VocabBuilder(train_seqs=[])
vb.stoi = vocab_data["stoi"]
vb.itos = vocab_data["itos"]
vb.vocab_size = vocab_data["vocab_size"]
vb.train_ids = vocab_data["train_ids"]

# Load trained transformer model
transformer_model = MidiTextTransformer(
    vocab_size=vb.vocab_size, d_model=512, n_head=8, n_layer=8,
    dim_ff=1024, block_size=1024)
transformer_model.to(DEVICE)
transformer_weights = torch.load(
    "../models/transformer/transformer_weights.pt", map_location=DEVICE)
transformer_model.load_state_dict(transformer_weights)


def generate_midi_and_image_bytes_from_diffusion(
    T: int = 100,
    height: int = 176,
    width: int = 256,
) -> tuple[bytes, bytes]:
    """
    Runs the diffusion model once, converts the generated piano-roll image
    to MIDI bytes, and also returns the PNG bytes of the image.
    """
    with torch.no_grad():
        _, alphas = prepare_noise_schedule(DEVICE, timesteps=T)
        sample = sample_image(
            diff_model,
            alphas,
            device=DEVICE,
            img_size=[height, width],
        )

    # Map from [-1, 1] to [0, 255] and to a PIL image
    sample_01 = (sample + 1.0) / 2.0
    sample_01 = sample_01.squeeze(0).cpu().numpy()
    sample_img = (sample_01 * 255).astype(np.uint8)
    img = Image.fromarray(sample_img, mode="L")

    # ---- MIDI bytes ----
    midi_file = pianoroll_images_to_midi([img])
    midi_buf = io.BytesIO()
    midi_file.save(file=midi_buf)
    midi_buf.seek(0)
    midi_bytes = midi_buf.getvalue()

    # ---- PNG image bytes ----
    img_buf = io.BytesIO()
    img.save(img_buf, format="PNG")
    img_buf.seek(0)
    img_bytes = img_buf.getvalue()

    return midi_bytes, img_bytes


def generate_midi_bytes_from_text(start_text, max_new_tokens=500):
    # Encode start text into token IDs
    start_ids = torch.tensor(
        [vb.encode(start_text)], dtype=torch.long).to(DEVICE)

    transformer_model.eval()
    with torch.no_grad():
        generated_ids = generate_midi_tokens_with_transformer(
            transformer_model,
            sos_id=vb.stoi["<SOS>"],
            eos_id=vb.stoi["<EOS>"],
            start_tokens=start_ids[0].tolist(),
            max_new_tokens=max_new_tokens,
        )

    # Decode back to text
    generated_text = vb.decode(generated_ids)

    # Convert text → MIDI (using your midi_conversion helpers)
    midi_file = text_to_midi(generated_text)
    buf = io.BytesIO()
    midi_file.save(file=buf)
    buf.seek(0)
    return buf.getvalue()


@app.post("/generate-midi-from-diffusion")
def generate_midi_from_diffusion():
    midi_bytes, img_bytes = generate_midi_and_image_bytes_from_diffusion()

    midi_b64 = base64.b64encode(midi_bytes).decode("ascii")
    img_b64 = base64.b64encode(img_bytes).decode("ascii")

    # Return both as base64; frontend will reconstruct Blob/Data URLs
    return JSONResponse(
        {
            "midi_base64": midi_b64,
            "image_base64": img_b64,
            "filename": "generated.mid",
        }
    )


@app.post("/generate-midi-from-transformer")
def generate_midi_from_transformer(
    payload: dict = Body(
        default={"start_text": "<SOS>", "max_new_tokens": 500}
    )
):
    """
    Request body (JSON):
    {
      "start_text": "COMPOSER_mozart KEY_D ...",   # optional
      "max_new_tokens": 500                        # optional
    }
    """
    start_text = payload.get("start_text", "<SOS>")
    max_new_tokens = int(payload.get("max_new_tokens", 500))

    midi_bytes = generate_midi_bytes_from_text(start_text, max_new_tokens)
    midi_b64 = base64.b64encode(midi_bytes).decode("ascii")

    return JSONResponse(
        {
            "midi_base64": midi_b64,
            "filename": "transformer_sample.mid",
        }
    )


# Initialize OpenAI Client
load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")
client = None
if api_key:
    try:
        client = OpenAI(api_key=api_key)
        print("OpenAI Client initialized successfully.")
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")
        client = None
else:
    print("WARNING: OpenAI API key not found in .env. AI features will be disabled.")

SYSTEM_PROMPT = """
You are a music composer assistant. Your goal is to convert natural language descriptions 
into a sequence of musical tokens that will be used to seed a music transformer.

The valid tokens which the transformer model understands are in this format (examples):
- NOTE_72, NOTE_20, etc. (MIDI note pitch values)
- DUR_250, DUR_1, etc. (Note duration)
- VEL_6, VEL_2, etc. (Note volume)
- <SOS> (at the start of each sequence)
- MEASURE (at the start of new measures)
- BEAT (at the start of each beat)
- ONLY 3 composer options: COMPOSER_haydn, COMPOSER_mozart, COMPOSER_beethoven
- ONLY certain key options: KEY_D, KEY_Dm, KEY_A, KEY_Ab, KEY_B, KEY_Bb, KEY_C, KEY_C#m, KEY_Db, KEY_E, KEY_Eb, KEY_F, KEY_Fm, KEY_G, KEY_Gm
- ONLY certain time signatures: TIME_SIGNATURE_6/8, TIME_SIGNATURE_9/8, TIME_SIGNATURE_12/8, TIME_SIGNATURE_2/2, TIME_SIGNATURE_2/4, TIME_SIGNATURE_3/4, TIME_SIGNATURE_4/4
- TEMPO_BPM_60, TEMPO_BPM_120, etc. (these specify tempo)

For example, this is the beginning of a real Haydn piano piece sequence:
<SOS> COMPOSER_haydn KEY_G TIME_SIGNATURE_6/8 TEMPO_BPM_120 MEASURE BEAT BEAT BEAT POS_24 NOTE_74 DUR_22 VEL_4 BEAT POS_0 NOTE_55 DUR_20 VEL_3 NOTE_59 DUR_20 VEL_3 NOTE_74 DUR_36 VEL_5 POS_36 NOTE_73 DUR_12 VEL_4 BEAT
And this is the beginning of a real Mozart symphony:
<SOS> COMPOSER_mozart KEY_C TIME_SIGNATURE_4/4 TEMPO_BPM_250 MEASURE BEAT POS_0 NOTE_60 DUR_20 VEL_2 NOTE_79 DUR_8 VEL_2 POS_8 NOTE_81 DUR_8 VEL_2 POS_16 NOTE_79 DUR_8 VEL_2 POS_24 NOTE_62 DUR_20 VEL_2 NOTE_77 DUR_12

You will be seeding the music transformer by providing the initial metadata, as follows:

<SOS> [composer] [key] [time signature] [tempo] MEASURE BEAT

You should end your output there, unless the user very specifically wants you to add pitch tokens. However, you should generally assume that you are incapable of adding NOTE/DUR/VEL tokens.

Try to cater your metadata choices to the user's wishes. For example, a livelier/faster user request should use a higher BPM. A waltz should use TIME_SIGNATURE_3/4 rather than 4/4. A sadder song should use a minor key like KEY_Dm or KEY_Gm rather than KEY_D or KEY_G, Etc.

OUTPUT RULES:
1. Output ONLY the space-separated tokens. No explanations.
2. Keep the sequence as short as possible to complete the request.
3. Start with <SOS>.
5. Do not type more than one or two "BEAT" tokens at the end of your output.
4. Do NOT venture outside of the keys, composers, and time signatures I listed -- maintain all given constraints.
"""


# Add this to the other Pydantic models
class SeedRequest(BaseModel):
    user_prompt: str


# Add this new endpoint
@app.post("/generate-seed-text")
def generate_seed_text(request: SeedRequest):
    # 1. Check if AI is available
    if not client:
        return JSONResponse(
            content={
                "error": "OpenAI API Key is missing on the server. AI features are disabled."
            },
            status_code=503 # Service Unavailable
        )

    try:
        # 2. Proceed if client exists
        chat_completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Generate a valid token sequence for: {request.user_prompt}"}
            ],
        )

        seed_text = chat_completion.choices[0].message.content.strip()
        print(f"ChatGPT generated seed: {seed_text}")

        return JSONResponse({"seed_text": seed_text})

    except Exception as e:
        print(f"Error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)