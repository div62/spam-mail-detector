@"
# Spam Mail Detector

A machine learning project that classifies SMS messages as **Spam** or **Ham** using Natural Language Processing (NLP).

## Features

- Text preprocessing
- TF-IDF text vectorization
- Machine learning classification
- Spam/Ham prediction
- Spam probability
- Model evaluation
- Confusion matrix
- Saved trained model and TF-IDF vectorizer

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Matplotlib
- Joblib

## Project Structure

```text
Spam Mail Detector/
│
├── artifacts/
│   ├── spam_model.pkl
│   ├── tfidf_vectorizer.pkl
│   └── confusion_matrix.png
│
├── data/
│   └── SMSSpamCollection
│
├── evaluation/
│   └── evaluate_model.py
│
├── models/
│   ├── predict.py
│   ├── save_model.py
│   ├── spam_models.py
│   ├── tfidf_vectorizer.py
│   ├── train_model.py
│   └── train_test_split.py
│
├── preprocessing/
│   ├── prepare_data.py
│   └── text_preprocessor.py
│
├── .gitignore
├── requirements.txt
└── README.md