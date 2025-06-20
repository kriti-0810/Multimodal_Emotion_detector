import streamlit as st
import os
import cv2
import tempfile
import numpy as np
from utils.text_emotion import predict_emotion as predict_text
from utils.speech_emotion import predict_emotion_from_audio
from utils.facial_predictor import predict_emotion_from_image
import sounddevice as sd
from scipy.io.wavfile import write
import time

st.set_page_config(page_title="Unified Emotion Detector", layout="centered")
st.title("🧠 Unified Emotion Detection System")

st.markdown("Detect emotions from **Text**, **Voice (Mic or Upload)**, and **Facial Expressions (Image or Webcam)** 🎯")

# ---------- TEXT EMOTION ----------
st.header("📄 Text Input")
text_input = st.text_area("Type your sentence here:")
text_emotion = None
if st.button("Predict Text Emotion"):
    if text_input.strip():
        text_emotion = predict_text(text_input)
        st.success(f"**Text Emotion:** {text_emotion} 😄")

# ---------- SPEECH EMOTION ----------
st.header("🎙️ Speech Input")

# Upload audio
uploaded_audio = st.file_uploader("Upload an audio file (.wav)", type=["wav"])
if uploaded_audio and st.button("Predict Emotion from Uploaded Audio"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
        tmp_audio.write(uploaded_audio.read())
        speech_emotion = predict_emotion_from_audio(tmp_audio.name)
        st.success(f"**Speech Emotion:** {speech_emotion} 🎧")

# Record live audio
st.subheader("Or Record Audio via Microphone")
duration = st.slider("Recording Duration (seconds)", 1, 10, 3)
if st.button("🎙 Record from Mic"):
    st.info("Recording... Speak now!")
    fs = 44100  # Sampling rate
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    temp_wav = os.path.join(tempfile.gettempdir(), "mic_input.wav")
    write(temp_wav, fs, recording)
    st.success("Recording complete!")
    mic_emotion = predict_emotion_from_audio(temp_wav)
    st.success(f"**Mic Emotion:** {mic_emotion} 🎤")

# ---------- FACIAL EMOTION ----------
st.header("📷 Facial Input")

# Upload image
uploaded_img = st.file_uploader("Upload an image (jpg/jpeg/png)", type=["jpg", "jpeg", "png"])
if uploaded_img and st.button("Predict Emotion from Uploaded Image"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_img:
        tmp_img.write(uploaded_img.read())
        facial_emotion = predict_emotion_from_image(tmp_img.name)
        st.success(f"**Facial Emotion:** {facial_emotion} 🧑‍🦱")

# Webcam prediction
st.subheader("Or Use Webcam")
if st.button("📸 Start Webcam Detection"):
    st.info("Webcam starting...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not access webcam.")
    else:
        st.info("Press 'q' on the webcam window to stop.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Save frame temporarily
            tmp_img_path = os.path.join(tempfile.gettempdir(), "live_cam.jpg")
            cv2.imwrite(tmp_img_path, frame)

            # Predict emotion
            emotion = predict_emotion_from_image(tmp_img_path)

            # Show live webcam with emotion label
            cv2.putText(frame, f"Emotion: {emotion}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Live Facial Emotion Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
