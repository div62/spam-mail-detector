import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from preprocessing.text_preprocessor import clean_text


# Load dataset
data = pd.read_csv(
    "data/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"]
)

# Clean messages
data["cleaned_message"] = data["message"].apply(clean_text)

X = data["cleaned_message"]
y = data["label"]


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# TF-IDF
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2
)

X_train_tfidf = vectorizer.fit_transform(X_train)


# Train model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)


# Save model and vectorizer
joblib.dump(model, "artifacts/spam_model.pkl")
joblib.dump(vectorizer, "artifacts/tfidf_vectorizer.pkl")


print("Model saved: artifacts/spam_model.pkl")
print("Vectorizer saved: artifacts/tfidf_vectorizer.pkl")