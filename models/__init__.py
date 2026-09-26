from sklearn.feature_extraction.text import TfidfVectorizer


vectorizer = TfidfVectorizer()

messages = [
    "free prize win",
    "free offer today",
    "hello how are you"
]

tfidf_matrix = vectorizer.fit_transform(messages)

print("Shape:", tfidf_matrix.shape)

print("\nVocabulary:")
print(vectorizer.get_feature_names_out())

print("\nTF-IDF Matrix:")
print(tfidf_matrix.toarray())
