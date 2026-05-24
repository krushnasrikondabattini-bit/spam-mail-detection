# Spam Mail Detection

A machine learning based web application that detects whether an email or message is spam or not spam using Natural Language Processing (NLP).

## Features
- Spam email/message detection
- Machine learning prediction model
- Simple Flask web interface
- Fast text classification

## Tech Stack
- Python
- Flask
- Scikit-learn
- NLP
- HTML/CSS

## Project Files
- `app.py` → Flask web application
- `train_model.py` → Model training script
- `model.pkl` → Trained ML model
- `vectorizer.pkl` → Text vectorizer
- `spam.csv` → Dataset

## How to Run

```bash
git clone https://github.com/krushnasrikondabattini-bit/spam-mail-detection.git
cd spam-mail-detection
pip install flask pandas scikit-learn
python app.py
```

## Output

The system predicts whether the entered message is:

- Spam
- Not Spam

### Spam Message Detection

![Spam Detection](assets/images/spam-detection.jpg)

---

### Not Spam Message Detection

![Not Spam Detection](assets/images/not-spam-detection.jpg)

## Author
KRUSHNA SRI KONDABATTINI
