# Classical Music Generation Model

## Project Purpose and Motivation
This project explores whether accessible machine learning methods can generate coherent classical-style music. We focus on classical-era composers (roughly 1725–1800 A.D.), training on Haydn, Mozart, and Beethoven to keep a consistent tonal style. The browser app lets musicians, students, and hobbyists experiment with these models to generate MIDIs without deep technical or music-theory expertise.

## What It Does
Two complementary generators power the app: a transformer that produces musical token sequences, and a diffusion model that synthesizes piano-roll images. Both outputs are converted to MIDI for playback. Users can generate pieces, preview them in-browser, view the piano-roll image, download the MIDI, and optionally draft prompts with an OpenAI-assisted helper.

Within each model type exist two implementations: for transformers, an in-house model and a tuned version of OpenAI GPT-2; for diffusion, an unconditional model and a conditional model trained with classifier-free guidance. Each of these have their strengths and weaknesses, allowing the user to choose a model which best suits their music generation goals. For instance, GPT-2 achieves the best musical results of all the models but is limited in its context window to 1,024 tokens, while our custom in-house model can produce an endless context window but procures more inconsistent results.

## Quick Start
1) Backend (Python 3.11+, virtual environment activated):
```
cd backend
uvicorn main:app --reload --port 8000
```
2) Frontend (Node.js 18+):
```
cd frontend
npm install
npm run dev
```
Then open http://localhost:5173 in your browser.
3) Optional: enable the OpenAI prompt helper by adding `OPENAI_API_KEY=<your key>` to `backend/.env`.

See `SETUP.md` for full installation details, CUDA notes, and troubleshooting.

## Video Links
- Demo Video:
- Technical Walkthrough Video:

## Evaluation

Our project, we want to evaluate our Transformer model and Diffusion model. Since musical quality cannot be captured by a single numerical score, we combined quantitative metrics such as loss and accuracy with qualitative listening tests and visual comparisons against real classical music.

### Transformer Model Evaluation

Our Transformer model is trained to predict the next musical token in a sequence, so we want to evaluate it using training statistcs and musical analysis.

What have we have evaluated:

- Step during training
- Training loss and training accuracy
- Validation loss and validation accuracy on held-out Mozart/Haydn sequences
- Sound quality testing by loading the generated MIDI file into a MIDI software
- Look at the visiul comparison of piano roll images to real Mozart/Haydn compositions to check for tempo drift, repeated looping patterns, overlapping notes, and gaps or unnatural silence

Our training results:

| Step | Train Loss | Train Accuracy | Validation Loss | Validation Accuracy|
|---|---|---|---|---|
| 0 | 5.663 | 0.075 | 5.738 | 0.073 |
| 500 | 1.514 | 0.577 | 1.714 | 0.542 |
| 1000 | 1.329 | 0.619 | 1.575 | 0.566 |
| 1500 | 1.246 | 0.645 | 1.487 | 0.591 |
| 2000 | 1.206 | 0.653 | 1.438 | 0.612 |
| 2500 | 1.136 | 0.672 | 1.435 | 0.610 |
| 3000 | 1.075 | 0.684 | 1.437 | 0.608 |
| 3500 | 1.074 | 0.687 | 1.339 | 0.635 |
| 4000 | 1.087 | 0.689 | 1.310 | 0.640 |
| 4500 | 1.047 | 0.698 | 1.315 | 0.641 |
| 5000 | 1.077 | 0.688 | 1.282 | 0.650 |
| 5500 | 0.960 | 0.720 | 1.324 | 0.639 |
| 6000 | 0.984 | 0.714 | 1.340 | 0.638 |
| 6500 | 0.960 | 0.720 | 1.282 | 0.656 |
| 7000 | 0.927 | 0.730 | 1.293 | 0.652 |
| 7500 | 0.982 | 0.713 | 1.306 | 0.654 |
| 8000 | 0.917 | 0.731 | 1.303 | 0.651 |
| 8500 | 0.910 | 0.734 | 1.301 | 0.652 |
| 9000 | 0.936 | 0.726 | 1.291 | 0.657 |
| 9500 | 0.924 | 0.732 | 1.255 | 0.660 |

### Diffusion Model Evaluation

Our Diffusion model generates piano roll images in which we converted it to MIDI. So we want to evaluate the model mainly by watching how the loss changes during training and by visually inspecting the generated piano roll images and listening to the output.

What have we evaluated:

- Epoch number during training
- Loss curve over epochs
- Visual inspection of generated piano roll images to see any realistic clustering of notes, look at spacing across pitch and time, and check for all-white or all-black images to avoid
- Listening test using MIDI software to check for rhythmic consistency

Our training results:

| Epoch | Training Loss | Validation Loss |
|---|---|---|
| 20 | 0.0779 | 0.0772 |
| 40 | 0.0476 | 0.0405 |
| 60 | 0.0357 | 0.0342 |
| 80 | 0.0290 | 0.0302 |
| 100 | 0.0254 | 0.0251 |
| 120 | 0.0215 | 0.0207 |
| 140 | 0.0201 | 0.0186 |
| 160 | 0.0190 | 0.0213 |
| 180 | 0.0165 | 0.0164 |
| 200 | 0.0154 | 0.0153 |

The 200-epoch model is saved as "best_model.pt" in models\diffusion_unconditional_1200\.

## Individual Contributions

### Calvin Chen

- Collected the dataset from Hugging Face and was able to performed data cleaning, preprocessing, and spliting the data for model training.
- Built custom tokenizer and conversion methods for MIDI files to/from text and image.
- Desgined and trained the Transformer model for music generation.
- Designed and trained the Diffusion models to generate piano roll images.
- Implemented tuning of GPT-2 for music generation.
- Implemented the backend API for running the models and generating the MIDI files.
- Helped build and deployed the React web application, and helped build and test features such as the audio playback, MIDI downloading, and the OpenAI integration feature.

### Johan Nino Espino
- Tested the entire project acorss different environments such as MacOS to ensure full compatibility for users.
- Helped developed all deliverables which includes the README.md file, the SETUP.md file, the documentation, and the evaluation summaries.
- Contributed by debugging the model training notebooks and confirmed that the diffusion and transformer ppelines works well from start to finish.
