from pathlib import Path
import re

import pandas as pd
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "data" / "spam.csv"

app = Flask(__name__)

STOP_WORDS = set(ENGLISH_STOP_WORDS)


def preprocess_text(text):
    text = re.sub(r"[^a-z\s]", "", str(text).lower())
    return " ".join(
        word for word in text.split() if word not in STOP_WORDS
    )


def load_model():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_PATH}. Add spam.csv to the project folder."
        )

    data = pd.read_csv(DATASET_PATH, encoding="latin1")
    required_columns = {"v1", "v2"}
    if not required_columns.issubset(data.columns):
        raise ValueError("spam.csv must contain the columns 'v1' and 'v2'.")

    messages = data["v2"].fillna("").map(preprocess_text)
    labels = data["v1"].map(lambda label: 1 if label == "spam" else 0)

    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=3000)),
            ("classifier", LinearSVC()),
        ]
    )
    model.fit(messages, labels)
    return model


model = load_model()


@app.route("/", methods=["GET", "POST"])
def index():
    message = ""
    result = None
    result_class = ""

    if request.method == "POST":
        message = request.form.get("message", "").strip()
        if message:
            prediction = model.predict([preprocess_text(message)])[0]
            result = "Spam detected" if prediction == 1 else "Looks like ham"
            result_class = "spam" if prediction == 1 else "ham"
        else:
            result = "Enter a message to check"
            result_class = "error"

    return render_template(
        "index.html",
        message=message,
        result=result,
        result_class=result_class,
    )


if __name__ == "__main__":
    app.run(debug=True)
