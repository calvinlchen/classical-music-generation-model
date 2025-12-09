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
'''bash
git clone git@github.com:calvinlchen/classical-music-generation-model.git
cd classical-music-generation-model
'''

### Create a Python Environment
This project was tested with Python 3.11.5, so ensure that you are running this verison of python.

For macOS users, run these commands to create a python environemnt on your terminal:
'''bash
python3 -m venv .venv
source .venv/bin/activate
'''

For Windows users:
'''bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
'''

### Install Requirements.txt
Inside your environment, run these comments to install the packages from requirements.txt file:
'''bash
pip install --upgrade pip
pip install -r requirements.txt
'''

If you have an NVIDI GPU on your device (https://download.pytorch.org/whl/cu121), you can change
'''bash
pip install torch==2.5.1
torchvision==0.20.1 
torchaudio==2.5.1
'''

To this on the requirements.txt file:
'''bash
pip install torch==2.5.1+cu121 
torchvision==0.20.1+cu121 
torchaudio==2.5.1+cu121
'''

### Runing the Backend
To run the backend, make sure that your virtual enviornment is activated using these commands:
'''bash
cd backend
uvicorn main:app --reload --port 8000
'''

If successful, you should see:
'''bash
Uvicorn running on http://localhost:8000
'''

### Running the Frontend
Ensure that you have Node.js installed globally on your device (https://nodejs.org/en/download).

To run the frontend, run these commends globally (not from your virtual machine environment):
'''bash
cd frontend
npm install (If you have not installed Node.js before)
npm run dev
'''

If run sucessfully, you should see:
'''bash
Local http://localhost:5173
'''

You should open that link in your browser to take you to the web app.
