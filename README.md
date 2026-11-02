# Spam Mail Detector

A small machine learning project that reads an SMS message and decides whether it's spam or ham (ham is just the word for "not spam"). It cleans up the text with NLP, turns the words into numbers with TF-IDF, and uses a Multinomial Naive Bayes classifier to make the call.

I built it to go through the whole loop once: clean the data, train a model, measure how well it works, save it, and use it on new messages.

## How it works

1. **Clean the text.** Everything is lowercased, numbers and punctuation are removed, the message is split into words with NLTK's tokenizer, and common English stopwords like "the" and "is" are dropped.
2. **Split the data.** 80% goes to training and 20% to testing. The split is stratified so both sets keep the same spam/ham ratio, and it uses a fixed random seed so results are repeatable.
3. **Vectorize.** TF-IDF turns the messages into features using single words and two-word phrases. Anything that appears in only one message is ignored.
4. **Classify.** A Multinomial Naive Bayes model learns from the training set.
5. **Predict.** Give it a new message and it returns spam or ham, along with the probability that the message is spam.

## Dataset

The data is in `data/SMSSpamCollection`, a tab-separated file with 5,572 labeled messages: 4,825 ham and 747 spam. Each line is a label, a tab, and then the message text.

## Results

These numbers come from the 1,115 messages held out for testing:

| Metric | Score |
|---|---|
| Accuracy | 96.59% |
| Precision (spam) | 100% |
| Recall (spam) | 74.50% |
| F1 score (spam) | 0.854 |

Confusion matrix:

| | Predicted ham | Predicted spam |
|---|---|---|
| **Actual ham** | 966 | 0 |
| **Actual spam** | 38 | 111 |

In plain terms, when the model calls a message spam, it has been right every time in this test set. The weak spot is recall: it catches about three out of four spam messages and lets the other quarter through. That's the first thing I'd work on.

## Project structure

- `main.py` loads the dataset and prints a preview and the label counts.
- `preprocessing/text_preprocessor.py` holds the `clean_text` function that every other script uses.
- `preprocessing/prepare_data.py` cleans the whole dataset and shows a sample.
- `models/train_test_split.py` prints the training and testing sizes.
- `models/train_model.py` trains a model and prints the first 10 predictions next to the real labels.
- `models/save_model.py` trains the final model and saves it, along with the vectorizer, to `artifacts/`.
- `models/predict.py` asks for a message and prints the prediction and the spam probability.
- `evaluation/evaluate_model.py` prints accuracy, precision, recall, and F1, and saves the confusion matrix image to `artifacts/`.
- `artifacts/` holds the saved model, the saved vectorizer, and the confusion matrix image.

## Setup

You'll need Python 3. Clone the repo and install the requirements:

```
git clone https://github.com/div62/spam-mail-detector.git
cd spam-mail-detector
pip install -r requirements.txt
```

The cleaning step uses NLTK's tokenizer and stopword list, so download those once:

```
python -c "import nltk; nltk.download(['punkt', 'punkt_tab', 'stopwords'])"
```

## Usage

Run everything from the project root. The `-m` flag makes sure the imports between folders work.

Check out the dataset:

```
python main.py
```

Try a message yourself:

```
python -m models.predict
```

For example, the message "WINNER!! You have won a free ticket. Text CLAIM to 81010 now" comes back as:

```
Prediction: SPAM
Spam Probability: 67.15%
```

See the evaluation metrics and save the confusion matrix:

```
python -m evaluation.evaluate_model
```

Retrain the model and save it to `artifacts/`:

```
python -m models.save_model
```

The trained model and vectorizer are already in `artifacts/`, so you can skip the retraining step and go straight to `models.predict`.

## Limitations

- The model misses about a quarter of spam messages in the test set (38 out of 149).
- It's trained on one SMS dataset, so I wouldn't expect it to work well on email or on newer scam styles without retraining on more data.
- Only English is handled. The stopword list and the cleaning step both assume English text.

## Dataset credit

The data is the public SMS Spam Collection dataset.
