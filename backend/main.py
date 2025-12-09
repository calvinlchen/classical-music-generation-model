# backend/main.py
import io
import torch
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image

import sys
from pathlib import Path

# Add parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from models import SimpleUNet, sample_image
from model_helpers import prepare_noise_schedule
from midi_conversion import pianoroll_images_to_midi

app = FastAPI()

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

# ---- load your trained model checkpoint once at startup ----
CHECKPOINT_PATH = "../models/diffusion_checkpoints/diffusion_epoch_0200.pt"

model = SimpleUNet().to(DEVICE)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


def generate_midi_bytes(
    T: int = 100,
    height: int = 88,
    width: int = 256,
) -> bytes:
    """
    Runs the diffusion model once, converts the generated piano-roll image
    to MIDI bytes.
    """
    with torch.no_grad():
        _, alphas = prepare_noise_schedule(DEVICE, timesteps=T)
        sample = sample_image(
            model,
            alphas,
            device=DEVICE,
            img_size=[88, 256]
        )

    sample_01 = (sample + 1.0) / 2.0
    sample_01 = sample_01.squeeze(0).cpu().numpy()
    sample_img = (sample_01 * 255).astype(np.uint8)
    img = Image.fromarray(sample_img, mode="L")

    midi_file = pianoroll_images_to_midi([img])

    buf = io.BytesIO()
    midi_file.save(file=buf)
    buf.seek(0)
    return buf.getvalue()


@app.post("/generate-midi-from-diffusion")
def generate_midi_from_diffusion():
    midi_bytes = generate_midi_bytes()
    return StreamingResponse(
        io.BytesIO(midi_bytes),
        media_type="audio/midi",
        headers={
            "Content-Disposition": 'attachment; filename="generated.mid"'
        },
    )
