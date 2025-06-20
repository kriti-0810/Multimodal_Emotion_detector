import librosa
import numpy as np
import joblib

# Load model and label encoder
model = joblib.load("models/speech_emotion_model.pkl")
label_encoder = joblib.load("models/speech_label_encoder.pkl")

def predict_emotion_from_audio(audio_path):
    try:
        audio, sample_rate = librosa.load(audio_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        prediction = model.predict([mfccs_mean])[0]
        emotion = label_encoder.inverse_transform([prediction])[0]
        return emotion
    except Exception as e:
        print(f"Error predicting emotion from {audio_path}: {e}")
        return "Error"
