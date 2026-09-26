import pandas as pd
from sklearn.model_selection import train_test_split
from preprocessing.text_preprocessor import clean_text


data = pd.read_csv(
    "data/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"]
)

data["cleaned_message"] = data["message"].apply(clean_text)


X = data["cleaned_message"]
y = data["label"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("Training messages:", len(X_train))
print("Testing messages:", len(X_test))

print("\nTraining labels:")
print(y_train.value_counts())

print("\nTesting labels:")
print(y_test.value_counts())