import cv2
import numpy as np
import tensorflow as tf
import joblib

# Load trained model and label encoder
model = tf.keras.models.load_model("models/facial_emotion_model.h5")
label_encoder = joblib.load("models/facial_label_encoder.pkl")

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Emoji mapping
emoji_map = {
    "angry": "😠",
    "disgust": "🤢",
    "fear": "😱",
    "happy": "😄",
    "neutral": "😐",
    "sad": "😢",
    "surprise": "😲"
}

def predict_emotion_from_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        print("No face detected.")
        return

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        face = cv2.resize(roi_gray, (48, 48))
        face = face.astype("float32") / 255.0
        face = np.expand_dims(face, axis=-1)
        face = np.expand_dims(face, axis=0)

        prediction = model.predict(face)
        predicted_class = label_encoder.inverse_transform([np.argmax(prediction)])[0]
        emoji = emoji_map[predicted_class]

        print(f"Predicted Emotion: {predicted_class} {emoji}")

        # Optional: Draw on image
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(img, f"{predicted_class} {emoji}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show the image
    cv2.imshow("Prediction", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    predict_emotion_from_image(r"C:\Users\Admin\Desktop\Multimodal_Emotion_Detector\test_images\images.jpeg")  # Replace with your image path
