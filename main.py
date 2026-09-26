import pandas as pd

data = pd.read_csv(
    "data/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"]
)

print(data.head())

print("\nShape:", data.shape)

print("\nLabels:")
print(data["label"].value_counts())