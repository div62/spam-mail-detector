import pandas as pd
from text_preprocessor import clean_text


data = pd.read_csv(
    "data/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"]
)

data["cleaned_message"] = data["message"].apply(clean_text)

print(data[["label", "message", "cleaned_message"]].head())