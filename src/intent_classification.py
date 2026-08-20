import numpy as np
import torch

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)


class CustomerSupportDataset(
    torch.utils.data.Dataset
):

    def __init__(self, encodings, labels):

        self.encodings = encodings
        self.labels = list(labels)

    def __getitem__(self, idx):

        item = {
            key: torch.tensor(value[idx])
            for key, value in self.encodings.items()
        }

        item["labels"] = torch.tensor(
            self.labels[idx],
            dtype=torch.long
        )

        return item

    def __len__(self):

        return len(self.labels)


def prepare_intent_data(df):

    label_encoder = LabelEncoder()

    df["label"] = label_encoder.fit_transform(
        df["intent"]
    )

    X = df["utterance"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        label_encoder
    )


def train_baseline(
    X_train,
    X_test,
    y_train,
    y_test,
    label_encoder
):

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2)
    )

    X_train_tfidf = vectorizer.fit_transform(
        X_train
    )

    X_test_tfidf = vectorizer.transform(
        X_test
    )

    model = LogisticRegression(
        max_iter=1000
    )

    model.fit(
        X_train_tfidf,
        y_train
    )

    predictions = model.predict(
        X_test_tfidf
    )

    print("\n========== TF-IDF + LOGISTIC REGRESSION ==========")

    print(
        "Accuracy:",
        accuracy_score(
            y_test,
            predictions
        )
    )

    print(
        classification_report(
            y_test,
            predictions,
            target_names=label_encoder.classes_
        )
    )

    return model, vectorizer


def train_bert(
    X_train,
    X_test,
    y_train,
    y_test,
    num_labels
):

    model_name = "bert-base-uncased"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    train_encodings = tokenizer(
        X_train.tolist(),
        truncation=True,
        padding=True,
        max_length=128
    )

    test_encodings = tokenizer(
        X_test.tolist(),
        truncation=True,
        padding=True,
        max_length=128
    )

    train_dataset = CustomerSupportDataset(
        train_encodings,
        y_train
    )

    test_dataset = CustomerSupportDataset(
        test_encodings,
        y_test
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )

    def compute_metrics(eval_pred):

        logits, labels = eval_pred

        predictions = np.argmax(
            logits,
            axis=-1
        )

        return {
            "accuracy": accuracy_score(
                labels,
                predictions
            )
        }

    training_args = TrainingArguments(
        output_dir="./models/intent_model",
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()

    results = trainer.evaluate()

    print("\n========== BERT RESULTS ==========")
    print(results)

    return model, tokenizer, trainer