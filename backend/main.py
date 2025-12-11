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
from transformers import GPT2LMHeadModel

ROOT_DIR = Path(__file__).resolve().parent.parent  # already implicitly used
TEXT_DIR = ROOT_DIR / "data" / "example_midi_texts"

# Add parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from data_preprocessing import VocabBuilder
from models import (
    SimpleUNet,
    sample_image,
    MidiTextTransformer,
    generate_midi_tokens_with_transformer,
    ConditionedUNet,
    sample_image_guided
)
from model_helpers import prepare_noise_schedule
from midi_conversion import pianoroll_images_to_midi, text_to_midi
from text_processing import MidiTokenizer

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

COMPOSER_MAP = {
    "haydn": 0,
    "mozart": 1,
    "beethoven": 2,
    "null": 3
}
KEY_MAP = {
    "C": 0, "Cm": 1, "Db": 2, "C#m": 3, "D": 4, "Dm": 5,
    "Eb": 6, "Ebm": 7, "E": 8, "Em": 9, "F": 10, "Fm": 11, "Gb": 12,
    "F#m": 13, "G": 14, "Gm": 15, "Ab": 16, "G#m": 17, "A": 18,
    "Am": 19, "Bb": 20, "Bbm": 21, "B": 22, "Bm": 23, "Unknown": 24,
    "null": 25  # NULL token
}

GPT2_MODEL_DIR = ROOT_DIR / "models" / "midi_gpt2_model"
GPT2_VOCAB_PATH = ROOT_DIR / "data" / "midi_text_exports" / "midi_vocab.txt"
GPT2_MAX_CONTEXT_TOKENS = 1024

# Lazy globals for GPT-2 tuned model
gpt2_tokenizer: MidiTokenizer | None = None
gpt2_model: GPT2LMHeadModel | None = None

try:
    if GPT2_MODEL_DIR.exists() and GPT2_VOCAB_PATH.exists():
        gpt2_tokenizer = MidiTokenizer(str(GPT2_VOCAB_PATH))
        gpt2_model = GPT2LMHeadModel.from_pretrained(
            str(GPT2_MODEL_DIR)
        ).to(DEVICE)
        gpt2_model.eval()
        print("GPT-2 tuned model loaded successfully.")
    else:
        print(
            "GPT-2 tuned model files not found; GPT-2 mode will be disabled."
        )
except Exception as e:
    print(f"Error loading GPT-2 tuned model: {e}")
    gpt2_model = None
    gpt2_tokenizer = None

# Load trained diffusion model checkpoints once at startup
UNCOND_DIFF_MODEL_PATH = "../models/diffusion_unconditional_12000/\
best_model.pt"
COND_DIFF_MODEL_PATH = "../models/diffusion_conditional_20000/best_model.pt"

diff_unconditional_model = SimpleUNet().to(DEVICE)
checkpoint = torch.load(UNCOND_DIFF_MODEL_PATH, map_location=DEVICE)
if "model_state_dict" in checkpoint:
    diff_unconditional_model.load_state_dict(checkpoint["model_state_dict"])
else:
    diff_unconditional_model.load_state_dict(checkpoint)
diff_unconditional_model.eval()

diff_cond_model = ConditionedUNet(
    num_composers=len(COMPOSER_MAP),
    num_keys=len(KEY_MAP)
).to(DEVICE)
checkpoint = torch.load(COND_DIFF_MODEL_PATH, map_location=DEVICE)
if "model_state_dict" in checkpoint:
    diff_cond_model.load_state_dict(checkpoint["model_state_dict"])
else:
    diff_cond_model.load_state_dict(checkpoint)
diff_cond_model.eval()

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


# for UNconditional model
def generate_midi_bytes_with_unconditional_diffusion(
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
            diff_unconditional_model,
            alphas,
            device=DEVICE,
            img_size=[height, width],
        )

    midi_bytes, img_bytes = convert_pianoroll_sample_to_bytes(sample)

    return midi_bytes, img_bytes


# for CONDITIONAL model
def generate_midi_bytes_with_conditional_diffusion(
    composer: str,
    key: str,
    bpm: int,
    guidance: float = 3.0,
    T: int = 100
):
    # Map text to indices
    c_idx = COMPOSER_MAP.get(composer.lower(), COMPOSER_MAP["null"])
    k_idx = KEY_MAP.get(key, KEY_MAP["Unknown"])

    # Normalize BPM (matches training logic)
    tempo_val = float(bpm) / 200.0

    with torch.no_grad():
        _, alphas = prepare_noise_schedule(DEVICE, timesteps=T)

        # Use the guided sampler from models.py
        sample = sample_image_guided(
            diff_cond_model,
            alphas,
            DEVICE,
            composer_idx=c_idx,
            key_idx=k_idx,
            tempo_val=tempo_val,
            num_composers=len(COMPOSER_MAP),
            num_keys=len(KEY_MAP),
            guidance_scale=guidance
        )

    midi_bytes, img_bytes = convert_pianoroll_sample_to_bytes(sample)

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


def generate_midi_bytes_from_gpt2(
    prompt_text: str,
    max_new_tokens: int = 500,
    temperature: float = 0.8,
    top_k: int = 50,
):
    if gpt2_model is None or gpt2_tokenizer is None:
        raise RuntimeError("GPT-2 tuned model is not available on the server.")

    prompt_ids = gpt2_tokenizer.encode(
        prompt_text, add_special_tokens=False)
    input_ids = torch.tensor(
        [[gpt2_tokenizer.bos_token_id] + prompt_ids],
        dtype=torch.long
    ).to(DEVICE)
    attention_mask = (input_ids != gpt2_tokenizer.pad_token_id).long()

    gpt2_model.eval()
    with torch.no_grad():
        output_ids = gpt2_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            pad_token_id=gpt2_tokenizer.pad_token_id,
            eos_token_id=gpt2_tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0].tolist()
    generated_text = gpt2_tokenizer.decode(
        generated_ids, skip_special_tokens=True)

    midi_file = text_to_midi(generated_text)
    buf = io.BytesIO()
    midi_file.save(file=buf)
    buf.seek(0)
    return buf.getvalue()


# Helper wrappers that also return the generated token text
def generate_midi_and_text_inhouse(start_text: str, max_new_tokens: int = 500):
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

    generated_text = vb.decode(generated_ids)
    midi_file = text_to_midi(generated_text)
    buf = io.BytesIO()
    midi_file.save(file=buf)
    buf.seek(0)
    return buf.getvalue(), generated_text


def generate_midi_and_text_gpt2(
    prompt_text: str,
    max_new_tokens: int = 500,
    temperature: float = 0.8,
    top_k: int = 50,
):
    if gpt2_model is None or gpt2_tokenizer is None:
        raise RuntimeError("GPT-2 tuned model is not available on the server.")

    prompt_ids = gpt2_tokenizer.encode(
        prompt_text, add_special_tokens=False)
    input_ids = torch.tensor(
        [[gpt2_tokenizer.bos_token_id] + prompt_ids],
        dtype=torch.long
    ).to(DEVICE)
    attention_mask = (input_ids != gpt2_tokenizer.pad_token_id).long()

    gpt2_model.eval()
    with torch.no_grad():
        output_ids = gpt2_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            pad_token_id=gpt2_tokenizer.pad_token_id,
            eos_token_id=gpt2_tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0].tolist()
    generated_text = gpt2_tokenizer.decode(
        generated_ids, skip_special_tokens=True)

    midi_file = text_to_midi(generated_text)
    buf = io.BytesIO()
    midi_file.save(file=buf)
    buf.seek(0)
    return buf.getvalue(), generated_text


# Helper method
def convert_pianoroll_sample_to_bytes(sample):

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


@app.post("/generate-midi-from-diffusion")
def generate_midi_from_diffusion():
    midi_bytes, img_bytes = generate_midi_bytes_with_unconditional_diffusion()

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


class ConditionalDiffusionRequest(BaseModel):
    composer: str = "mozart"
    key: str = "C"
    bpm: int = 120
    guidance: float = 3.0


@app.post("/generate-midi-from-diffusion-conditional")
def generate_midi_conditional(req: ConditionalDiffusionRequest):
    midi_bytes, img_bytes = generate_midi_bytes_with_conditional_diffusion(
        req.composer, req.key, req.bpm, req.guidance
    )

    midi_b64 = base64.b64encode(midi_bytes).decode("ascii")
    img_b64 = base64.b64encode(img_bytes).decode("ascii")

    return JSONResponse({
        "midi_base64": midi_b64,
        "image_base64": img_b64
    })


@app.post("/generate-midi-from-transformer")
def generate_midi_from_transformer(
    payload: dict = Body(
        default={
            "start_text": "<SOS>",
            "max_new_tokens": 500,
            "model_type": "inhouse"
        }
    )
):
    """
    Request body (JSON):
    {
      "start_text": "COMPOSER_mozart KEY_D ...",   # optional
      "max_new_tokens": 500,                       # optional
      "model_type": "inhouse" | "gpt2"             # optional
    }
    """
    start_text = payload.get("start_text", "<SOS>")
    max_new_tokens = int(payload.get("max_new_tokens", 500))
    model_type = payload.get("model_type", "inhouse")
    temperature = float(payload.get("temperature", 0.8))
    top_k = int(payload.get("top_k", 50))

    if model_type not in {"inhouse", "gpt2"}:
        return JSONResponse(
            {"error": "Invalid model_type. Use 'inhouse' or 'gpt2'."},
            status_code=400
        )

    generated_text = None

    try:
        if model_type == "gpt2":
            if gpt2_model is None or gpt2_tokenizer is None:
                return JSONResponse(
                    {
                        "error": (
                            "GPT-2 tuned model is not available on the server."
                        )
                    },
                    status_code=503,
                )

            prompt_tokens = gpt2_tokenizer.encode(
                start_text, add_special_tokens=False)
            total_tokens = len(prompt_tokens) + max_new_tokens
            if total_tokens > GPT2_MAX_CONTEXT_TOKENS:
                allowed = max(GPT2_MAX_CONTEXT_TOKENS - len(prompt_tokens), 0)
                return JSONResponse(
                    {
                        "error": (
                            "Token limit exceeded for GPT-2 tuned model: "
                            f"{len(prompt_tokens)} prompt tokens + "
                            f"{max_new_tokens} max_new_tokens > "
                            f"{GPT2_MAX_CONTEXT_TOKENS}. "
                            "Reduce max_new_tokens or shorten the prompt."
                        ),
                        "max_new_tokens_allowed": allowed,
                    },
                    status_code=400,
                )

            midi_bytes, generated_text = generate_midi_and_text_gpt2(
                prompt_text=start_text,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )
            filename = "gpt2_transformer_sample.mid"
        else:
            midi_bytes, generated_text = generate_midi_and_text_inhouse(
                start_text=start_text,
                max_new_tokens=max_new_tokens
            )
            filename = "transformer_sample.mid"
    except Exception as e:
        print(f"Error generating transformer MIDI: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

    midi_b64 = base64.b64encode(midi_bytes).decode("ascii")

    return JSONResponse(
        {
            "midi_base64": midi_b64,
            "filename": filename,
            "generated_text": generated_text,
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
            status_code=503  # Service Unavailable
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
