# DeepFake Audio Detection System

A Streamlit-based web app for detecting whether an uploaded audio clip is real or spoofed/deepfake. The project uses a trained Convolutional Neural Network (CNN) model that classifies audio from MFCC features extracted with Librosa.

## Project Overview

This repository contains a trained fake-audio detection model and a simple web interface for testing audio files. The app accepts common audio formats, converts the uploaded file to FLAC internally, splits the signal into 3-second chunks, extracts MFCC features, averages the model predictions across chunks, and displays the final label.

The model was trained using the ASVspoof 2019 Logical Access (LA) dataset.

## Features

- Upload audio files through a Streamlit interface.
- Supports `.flac`, `.wav`, `.mp3`, `.ogg`, `.m4a`, and `.aac` uploads.
- Converts uploaded audio to FLAC before prediction.
- Extracts 13 MFCC features at a 22,050 Hz sample rate.
- Uses a saved CNN model (`cnn_audio.h5`) for classification.
- Displays average prediction probabilities and the final result as:
  - `Real Audio`
  - `Spoofed Audio`

## Repository Contents

| File | Description |
| --- | --- |
| `app.py` | Main Streamlit application used to upload audio and run deepfake detection. |
| `cnn_audio.h5` | Trained CNN model used by the Streamlit app. |
| `final_code.ipynb` | Notebook containing dataset preparation, MFCC extraction, CNN training, evaluation, and model saving code. |
| `data.json` | Pre-extracted MFCC feature data used for model training/testing. |
| `requirements.txt` | Python dependencies required to run the app and notebook. |
| `webpage.py` | Earlier placeholder Streamlit script. Use `app.py` for the working application. |

Generated runtime files such as `temp_uploaded_audio.*` and `converted_audio.flac` may appear after running the app. They are temporary files created from uploaded audio.

## Model Workflow

1. Load ASVspoof 2019 LA protocol metadata.
2. Organize audio files into class folders.
3. Extract MFCC features from the audio files using Librosa.
4. Save extracted features and labels to `data.json`.
5. Train a CNN with TensorFlow/Keras.
6. Evaluate the model using accuracy, loss, classification report, and confusion matrix.
7. Save the trained model as `cnn_audio.h5`.
8. Use the saved model inside `app.py` for real-time predictions.

## Prediction Workflow in `app.py`

1. The user uploads an audio file.
2. The app saves the upload temporarily.
3. Librosa loads the audio at 22,050 Hz.
4. SoundFile writes the processed audio as `converted_audio.flac`.
5. The audio is split into 3-second chunks.
6. MFCC features are extracted from each chunk.
7. The CNN predicts each chunk.
8. Predictions are averaged and mapped to the final class label.

Class mapping used by the app:

```python
{0: "Spoofed Audio", 1: "Real Audio"}
```

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## How to Run

Start the Streamlit app:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal, upload an audio file, and click **Detect**.

## Dataset

The model training notebook uses the ASVspoof 2019 Logical Access dataset. The notebook expects the dataset to be available locally and uses hard-coded paths, so update the dataset paths in `final_code.ipynb` before retraining on another machine.

Important notebook path variables to update:

```python
BASE_PATH = "E:/Coding/Projects/asvspoof2019/LA/LA"
DATASET_PATH = "/Coding/Projects/audio"
DATA_PATH = "/Coding/Projects/audioforlstm/data.json"
```

## Notes and Limitations

- `app.py` is the recommended entry point for the project.
- `webpage.py` is a placeholder/older experiment and does not contain the complete prediction pipeline.
- The trained model file `cnn_audio.h5` must be present in the project root for the app to run.
- The current app writes temporary audio files into the project directory during prediction.
- Prediction quality depends on the training data, preprocessing choices, and audio quality.
- This tool should be used as a research/demo system, not as the only source of truth for high-stakes decisions.

## Tech Stack

- Python
- Streamlit
- TensorFlow/Keras
- Librosa
- SoundFile
- NumPy
- Scikit-learn
- Pandas
- Matplotlib

