from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib

def train_text_emotion_model():
    # Load dataset
    dataset = load_dataset("emotion")
    train_texts = dataset['train']['text']
    train_labels = dataset['train']['label']

    # Create pipeline: TF-IDF + Logistic Regression
    text_clf = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LogisticRegression(max_iter=1000)),
    ])

    # Train the model
    text_clf.fit(train_texts, train_labels)

    # Save model
    joblib.dump(text_clf, 'models/text_emotion_model.pkl')

    print("✅ Text emotion model trained and saved!")

if __name__ == "__main__":
    train_text_emotion_model()

def predict_emotion(text):
    model = joblib.load('models/text_emotion_model.pkl')
    prediction = model.predict([text])[0]

    label_map = {
        0: "sadness",
        1: "joy",
        2: "love",
        3: "anger",
        4: "fear",
        5: "surprise"
    }

    return label_map[prediction]

