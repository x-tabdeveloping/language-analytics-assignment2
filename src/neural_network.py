from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline


def main():
    out_path = Path("out")
    model_path = Path("models")
    out_path.mkdir(exist_ok=True)
    model_path.mkdir(exist_ok=True)

    print("Loading data")
    data = pd.read_csv("data/fake_or_real_news.csv")
    X_train, X_test, y_train, y_test = train_test_split(
        data["text"], data["label"], test_size=0.2, random_state=42
    )

    print("Fitting classifier.")
    classifier = make_pipeline(TfidfVectorizer(stop_words="english"), MLPClassifier())
    classifier.fit(X_train, y_train)

    print("Evaluating.")
    report = classification_report(y_test, classifier.predict(X_test))
    print(report)

    print("Saving model and report.")
    joblib.dump(classifier, model_path.joinpath("neural_network.joblib"))
    with open(out_path.joinpath("neural_network.txt"), "w") as report_file:
        report_file.write(report)


if __name__ == "__main__":
    main()
