import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import joblib
from preprocessing.text_preprocessor import clean_text
model = joblib.load("artifacts/spam_model.pkl")
vectorizer = joblib.load("artifacts/tfidf_vectorizer.pkl")


# Get message from user
message = input("\nEnter a message: ")

# Clean message
cleaned_message = clean_text(message)

# Convert message using saved TF-IDF vectorizer
message_tfidf = vectorizer.transform([cleaned_message])

# Predict
prediction = model.predict(message_tfidf)[0]

# Get spam probability
probabilities = model.predict_proba(message_tfidf)[0]
spam_index = list(model.classes_).index("spam")
spam_probability = probabilities[spam_index]


print("\nPrediction:", prediction.upper())
print(f"Spam Probability: {spam_probability * 100:.2f}%")

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
vectorizer = TfidfVectorizer(    ngram_range=(1, 2),
    min_df=2
)

X_train_tfidf = vectorizer.fit_transform(X_train)


# Train model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)


# Custom message
message = input("\nEnter a message: ")

cleaned_message = clean_text(message)

message_tfidf = vectorizer.transform([cleaned_message])


# Prediction
prediction = model.predict(message_tfidf)[0]

# Probability
probabilities = model.predict_proba(message_tfidf)[0]

# Find probability belonging to spam
spam_index = list(model.classes_).index("spam")
spam_probability = probabilities[spam_index]


print("\nPrediction:", prediction.upper())
print(f"Spam Probability: {spam_probability * 100:.2f}%")