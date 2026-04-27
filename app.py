import streamlit as st
import numpy as np

st.title("Deepfake Audio Detection ")
st.write("Upload an audio file to check if it's a deepfake. ")

uploaded_file = st.file_uploader("Choose an audio file", type=["flac", "wav", "mp3", "ogg", "m4a", "aac"])

if uploaded_file is not None:
  
  from keras.models import load_model
  import librosa
  import soundfile as sf
  import os

  # Load the model
  model = load_model('cnn_audio.h5')

  def predict_voice(model, audio_file_path, genre_mapping):
    
    signal, sample_rate = librosa.load(audio_file_path, sr=22050)
    target_length = 66150

    # Split signal into chunks of 66150 samples (3 seconds)
    chunks = []
    for i in range(0, len(signal), target_length):
        chunk = signal[i:i + target_length]
        if len(chunk) < target_length:
            # Ignore trailing chunk if it's too small and we already have chunks
            if len(chunks) > 0 and len(chunk) < target_length / 2:
                continue
            chunk = np.pad(chunk, (0, target_length - len(chunk)))
        chunks.append(chunk)

    if len(chunks) == 0:
        return "Unknown"

    predictions = []
    for chunk in chunks:
        mfcc = librosa.feature.mfcc(y=chunk, sr=sample_rate, n_mfcc=13, n_fft=2048, hop_length=512)
        mfcc = mfcc.T
        mfcc = mfcc[np.newaxis, ..., np.newaxis]
        pred = model.predict(mfcc, verbose=0)
        predictions.append(pred[0])

    # Average the predictions across all chunks
    avg_prediction = np.mean(predictions, axis=0)
    predicted_index = np.argmax(avg_prediction)
    
    genre_label = genre_mapping[predicted_index]
    st.write("Average prediction probabilities:", avg_prediction)

    return genre_label


  if st.button("Detect"):
    # Save the uploaded file temporarily
    temp_ext = os.path.splitext(uploaded_file.name)[1]
    temp_input_path = "temp_uploaded_audio" + temp_ext
    with open(temp_input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Convert to FLAC using librosa and soundfile
    st.write("Processing and converting audio to FLAC format...")
    signal, sample_rate = librosa.load(temp_input_path, sr=22050)
    flac_path = "converted_audio.flac"
    sf.write(flac_path, signal, sample_rate, format='FLAC')
    st.success("Successfully converted to .flac format!")

    audio_file_path = flac_path

    genre_mapping = {0: "Spoofed Audio", 1: "Real Audio"}


    predicted_voice = predict_voice(model, audio_file_path, genre_mapping)

    st.write("Predicted label:")
    if predicted_voice == "Real Audio":
        st.success(f"**{predicted_voice}**")
    else:
        st.error(f"**{predicted_voice}**")
else:
  st.info("Upload an audio file to proceed.")
