import cv2
import numpy as np
import tensorflow as tf
import joblib
from keras.models import load_model

# Load the trained model and label encoder
model = load_model('models/facial_emotion_model.h5')
label_encoder = joblib.load('models/facial_label_encoder.pkl')

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier('utils/haarcascade_frontalface_default.xml')

# Define a function to preprocess face image
def preprocess_face(face_img):
    face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_resized = cv2.resize(face_gray, (48, 48))
    face_normalized = face_resized / 255.0
    face_reshaped = np.reshape(face_normalized, (1, 48, 48, 1))
    return face_reshaped

# Emoji map (optional)
emoji_map = {
    "angry": "😠", "disgust": "🤢", "fear": "😱",
    "happy": "😄", "sad": "😢", "surprise": "😲", "neutral": "😐"
}

# Start webcam
cap = cv2.VideoCapture(0)

print("🔴 Press 'q' to quit...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        processed_face = preprocess_face(face)

        # Predict emotion
        predictions = model.predict(processed_face)
        predicted_label = np.argmax(predictions)
        emotion = label_encoder.inverse_transform([predicted_label])[0]

        emoji = emoji_map.get(emotion, "")
        label = f"{emotion} {emoji}"

        # Draw bounding box and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Show video
    cv2.imshow('Real-Time Facial Emotion Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
