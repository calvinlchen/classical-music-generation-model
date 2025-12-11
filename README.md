# Classical Music Generation Model

## What It Does

What our project does is that we generated new classical-style piano music using a Transformer model and a Diffusion Model. We trained both models on a Mozart and Haydn MIDI dataset which is part of the dataset we found from Hugging Face (https://huggingface.co/datasets/drengskapur/midi-classical-music) which allowed our models to learn musical patterns directly from real compsitions. Our Transformer model generates music as sequences of musical tokens while our Diffusion model produces piano roll images to represent the generated music visually. We have built a web app where users can generate music, listen to them directly in the browser, can view the generated piano roll images, and are able to download the resulting MIDI files. With the Transformer portion of our web app, we also included an integration with OpenAI's ChatGPT Mini so that users can describe the type of misc they want to generate such as if the user wants to generate a sad song, a slow song, or an energetic song for example. Our main goal of this project is to create an accessible AI tool that can create coherent and expressive classical music in a way that is easy for anyone to use. With this goal in mind, we want our AI tool to help musicians, students, and musical hobbyists to experiment with music generation without requiring an technical or musical expertise.

## Quick Start

Follwing these steps to run the project after completing the installation instructions in the SETUP.md file (We went over how to run the Web App there as well).

### Starting the Backend Server

To run the backend, make sure that your Python virtual enviornment is activated using these commands on the terminal:

```
cd backend
uvicorn main:app --reload --port 8000
```

### Starting the Frontend Web App

Ensure that you have Node.js installed globally on your device (https://nodejs.org/en/download).

In a separate terminal window which is outside the Python virtual environment, run these commands:
```
cd frontend
npm install (If you have not installed Node.js before)
npm run dev
```

If run sucessfully, you should see:

```
Local http://localhost:5173
You should open that link in your browser to take you to the web app.
```

### Using the OPENAI API Feature

If yopu want to use the text-prompt interface of our web app, you do need to add your own OpenAI API key.

In the backend folder, create the ```.env``` file.

Inside the ```.env``` file, type:

```
OPENAI_API_KEY = [your own key]
```

This enables the AI-assisted music generation feature. If you do not have an OpenAI API key, you can still be able to generate music, can view piano roll images, and the rest of the app should still work, it is just that the OpenAI text-prompt feature will be disabled.

## Video Links

Demo Video:

Technical Walkthrough Video:

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
| 0 | 5.663 | 0.075 |
| 500 | 1.514 | 0.577 |
| 1000 | 1.329 | 0.619 |
| 1500 | 1.246 | 0.645 |
| 2000 | 1.206 | 0.653 |
| 2500 | 1.136 | 0.672 |
| 3000 | 1.075 | 0.684 |
| 3500 | 1.074 | 0.687 |
| 4000 | 1.087 | 0.689 |
| 4500 | 1.047 | 0.698 |

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
