# SETUP.md

## Overview
Step-by-step instructions for installing dependencies and running the Classical Music Generation web app (transformer + diffusion).

The project consists of:
- Backend: loads trained models and generates MIDI files.
- Frontend: lets you generate, preview, and download MIDI.
- Jupyter notebooks: training diffusion models and transformers.
- Utility scripts: MIDI preprocessing, piano-roll conversion, model helpers.

## Prerequisites
- Python 3.11.x
- Node.js 18+ (https://nodejs.org/en/download)
- Git
- Optional: NVIDIA GPU with CUDA 12.1 for faster training/inference.

## 1) Clone the repository
```
git clone git@github.com:calvinlchen/classical-music-generation-model.git
cd classical-music-generation-model
```

## 2) Create and activate a Python environment
macOS/Linux:
```
python3.11 -m venv .venv
source .venv/bin/activate
```
Windows (PowerShell):
```
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 3) Install Python requirements
```
pip install --upgrade pip
pip install -r requirements.txt
```

CUDA option (if you have a compatible NVIDIA GPU):
```
pip uninstall -y torch torchvision torchaudio
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --extra-index-url https://download.pytorch.org/whl/cu121
```

## 4) Run the backend API
Make sure your virtual environment is active:
```
cd backend
uvicorn main:app --reload --port 8000
```

## 5) Run the frontend
Use a separate terminal (outside the Python venv):
```
cd frontend
npm install
npm run dev
```
Open the app at http://localhost:5173.

## 6) Optional: enable OpenAI-assisted prompts
Create `backend/.env` and add:
```
OPENAI_API_KEY=<your key>
```
The app still works without this key; the OpenAI prompt helper will just be disabled.
