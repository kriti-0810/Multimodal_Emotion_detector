import joblib
from sklearn.preprocessing import LabelEncoder

classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

label_encoder = LabelEncoder()
label_encoder.fit(classes)

joblib.dump(label_encoder, "models/facial_label_encoder.pkl")
print("✅ Label encoder saved successfully!")
