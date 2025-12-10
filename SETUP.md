# SETUP.md

## Overview

We be going over how to set up the environment, install requirment packages, and run the backend and frontend for thr Music Diffusion + Transformer Web App.

The project consists of:
- Backend: loads trained diffusion model and generates MIDI files
- Frontend: plays and downloads generated MIDI files
- Jupyter Notebooks - training diffusion models and transformers
- Utility Scripts - MIDI preprocessing, piano rol conversion, model helpers

### Clone the Repository
To clone the repository, run these commends on the terminal:
```
git clone git@github.com:calvinlchen/classical-music-generation-model.git
cd classical-music-generation-model
```

### Create a Python Environment
This project was tested with Python 3.11.9, so ensure that Python 3.11.x is installed on your system.

For macOS users, run these commands to create a python environemnt on your terminal:
```
python3.11 -m venv .venv
source .venv/bin/activate
```

For Windows users:
```
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Install Requirements.txt
Inside your environment, run these comments to install the packages from requirements.txt file:
```
pip install --upgrade pip
pip install -r requirements.txt
```

If your system is CUDA-compatible (NVIDIA GPUs), please run the following commands to install CUDA-compatible versions of PyTorch:
```
pip uninstall torch torchvision torchaudio
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --extra-index-url https://download.pytorch.org/whl/cu121
```

### Runing the Backend
To run the backend, make sure that your virtual enviornment is activated using these commands:
```
cd backend
uvicorn main:app --reload --port 8000
```

### Running the Frontend
Ensure that you have Node.js installed globally on your device (https://nodejs.org/en/download).

To run the frontend, run these commands globally (not from your virtual machine environment):
```
cd frontend
npm install (If you have not installed Node.js before)
npm run dev
```

If run sucessfully, you should see the webapp displayed locally on:
```
http://localhost:5173
```

You should open that link in your browser to take you to the web app.
