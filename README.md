# Classical Music Generation Model

## What It Does

What our project does is that we generated new classical-style piano music using a Transformer model and a Diffusion Model. We trained both models on a Mozart and Haydn MIDI dataset which is part of the dataset we found from Hugging Face (https://huggingface.co/datasets/drengskapur/midi-classical-music) which allowed our models to learn musical patterns directly from real compsitions. Our Transformer model generates music as sequences of musical tokens while our Diffusion model produces piano roll images to represent the generated music visually. We have built a web app where users can generate music, listen to them directly in the browser, can view the generated piano roll images, and are able to download the resulting MIDI files. With the Transformer portion of our web app, we also included an integration with OpenAI's ChatGPT Mini so that users can describe the type of misc they want to generate such as if the user wants to generate a sad song, a slow song, or an energetic song for example. Our main goal of this project is to create an accessible AI tool that can create coherent and expressive classical music in a way that is easy for anyone to use. With this goal in mind, we want our AI tool to help musicians, students, and musical hobbyists to experiment with music generation without requiring an technical or musical expertise.



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
