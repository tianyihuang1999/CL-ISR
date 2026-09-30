"""SVM baseline; train transformer baselines with model_kind: transformer."""

from __future__ import annotations

import argparse

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from .metrics import compute_metrics
from .data import validate_frame


def svm_tfidf(train_path: str, test_path: str) -> dict:
    train = validate_frame(pd.read_csv(train_path))
    test = validate_frame(pd.read_csv(test_path))
    vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
    x_train = vec.fit_transform(train["text"].astype(str))
    x_test = vec.transform(test["text"].astype(str))
    clf = LinearSVC(random_state=42)
    clf.fit(x_train, train["label"].astype(int))
    pred = clf.predict(x_test)
    return compute_metrics(test["label"].astype(int), pred)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/sample/train.csv")
    parser.add_argument("--test", default="data/sample/test.csv")
    args = parser.parse_args()
    print(svm_tfidf(args.train, args.test))


if __name__ == "__main__":
    main()
