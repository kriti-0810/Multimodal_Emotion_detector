# Multimodal Emotion Detector

An AI-powered application that detects human emotions through:
- 📝 Text (typed or uploaded)
- 🔊 Speech (microphone or audio file)
- 📷 Facial expression (image or webcam)

This project uses machine learning and deep learning techniques to classify emotions like happy, sad, angry, fear, neutral, and more.

More details coming soon!
# 🧠 Multimodal Emotion Detector

An intelligent system that detects emotions from **text**, **speech**, and **facial expressions**, then combines the results into a unified emotion output with confidence and emojis! 🎭🔊📷✍️

---

## 🚀 Features

- 🔤 Text Emotion Detection (BERT/LSTM)
- 🔊 Speech Emotion Detection (RAVDESS Dataset, MFCC + RandomForest)
- 😄 Facial Emotion Recognition (FER-2013 + CNN)
- 📸 Webcam & file-based detection
- 🎙 Live mic recording
- 📂 Upload image/audio/text files
- ✅ Unified interface using Streamlit
- 🧠 Displays detected emotion + emoji + confidence

---

## 📁 Project Structure

```
Multimodal_Emotion_Detector/
├── app.py                  # Streamlit frontend
├── requirements.txt        # Python dependencies
├── models/                 # Trained model files (.pkl, .h5)
├── utils/                  # Text/speech/facial modules
│   ├── text_emotion.py
│   ├── speech_emotion.py
│   ├── facial_emotion.py
├── datasets/               # Datasets (you must download manually)
├── static/                 # Icons, emojis, images
├── test_images/            # Sample image inputs
└── README.md               # Project instructions
```

---

## 🔧 Installation

1. **Clone the repo**
   ```bash
   git clone https://github.com/kriti-0810/Multimodal_Emotion_detector.git
   cd Multimodal_Emotion_detector
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv emotion_env
   emotion_env\Scripts\activate   # On Windows
   ```

3. **Install requirements**
   ```bash
   pip install -r requirements.txt
   ```

---

## 📥 Download Datasets (Required!)

This project requires two datasets. **They are not included in the repo**. Download them manually:

| Dataset | Purpose | Download |
|--------|--------|----------|
| FER-2013 | Facial emotion detection | https://www.kaggle.com/datasets/msambare/fer2013 |
| RAVDESS | Speech emotion detection | https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio |

After download, place them like this:
```
datasets/
├── FER-2013/
├── RAVDESS/
```

---

## 🏃‍♀️ Running the App

```bash
streamlit run app.py
```

Visit [http://localhost:8501](http://localhost:8501) in your browser.

---

## 💡 Demo Inputs (Optional)

Put test files in `test_images/` or use the upload buttons inside the app.

---

## 📌 Notes

- The project is modular; each component can be run/tested separately.
- Ensure your system has `ffmpeg`, `OpenCV`, and compatible Python version (3.7+).
- If model files are missing, you may need to retrain using the scripts in `utils/`.

---

## 👩‍💻 Author

> 🔗 [Kriti](https://github.com/kriti-0810)  
> 🎓 BTech CSE | Passionate about AI/ML | VIT Vellore  
> 🤝 Open to collaboration and learning!

---

## 📜 License

This project is for educational purposes.
