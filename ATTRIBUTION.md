## Datasets

Our model were trained by using the public avaliable MIDI Classical Music dataset that we have found here: https://huggingface.co/datasets/drengskapur/midi-classical-music . The dataset provides us with MIDI files for composers such as Mozart, Haydn, Beethoven, and more. We specifically utilized the Haydn, Mozart, and Beethoven MIDI files for training our Transformer and Diffusion models. The data that we used were accessed in solely for academic and research purposes.

## Third-Party Libraries and Frameworks

### Backend
- FastAPI, Uvicorn, PyTorch, mido, Pillow, NumPy, OpenAI API  
- See `backend/requirements.txt` for exact versions.

### Frontend
- React, Vite, Tone.js, @tonejs/midi, and supporting Node.js tooling  
- See `frontend/package.json` for exact versions.

## AI Assistance

We used OpenAI's ChatGPT 5.1 and Google Gemini in December 2025 to:
- Debug diffusion and transformer training notebooks and Python files.
- Suggest improvements to model design and tuning.
- Improve code structure and readability.
- Assist with frontend/backend app code and deployment tasks.
