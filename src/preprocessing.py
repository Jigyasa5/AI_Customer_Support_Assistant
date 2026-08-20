import re
import nltk
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Download NLTK resources
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")


stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def load_dataset(path):

    df = pd.read_csv(path)

    print("Dataset shape:", df.shape)
    print("Columns:", df.columns.tolist())

    print("\nMissing values:")
    print(df.isnull().sum())

    return df


def preprocess_text(text):

    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove special characters
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Tokenization
    words = text.split()

    # Stopword removal
    words = [
        word for word in words
        if word not in stop_words
    ]

    # Lemmatization
    words = [
        lemmatizer.lemmatize(word)
        for word in words
    ]

    return " ".join(words)


def prepare_dataset(df):

    # Remove missing utterances/intents
    df = df.dropna(
        subset=["utterance", "intent"]
    ).copy()

    # Preprocess utterance
    df["processed_utterance"] = (
        df["utterance"]
        .apply(preprocess_text)
    )

    return df