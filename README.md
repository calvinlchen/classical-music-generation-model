# classical-music-generation-model
Machine learning experiment for generating classical-style music.

====== INSTALLATION ======

To get started, please install the dependencies listed in requirements.txt, such as via the command:
pip install -r requirements.txt

If CUDA is available in your machine (NVIDIA GPU), please install the CUDA versions of torch with the following:
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121


====== RUNNING THE WEB APP ======

- Start the backend by running the following commands within your virtual environment:
cd backend
uvicorn main:app --reload --port 8000

- Ensure you have Node.js installed (globally): https://nodejs.org/en/download

- Outside of the venv, start the app by running:
cd frontend
npm run dev

Tested in December 2025 with Python 3.11.5
