import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    tokens = word_tokenize(text)

    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]

    return " ".join(tokens)
message = "Congratulations! You WON a FREE Prize!"

print(clean_text(message))    