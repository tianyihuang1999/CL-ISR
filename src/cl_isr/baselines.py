"""Baselines reported in Table 1: SVM (TF-IDF), BERT, RoBERTa."""

from __future__ import annotations

import argparse

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from .metrics import compute_metrics


def svm_tfidf(train_path: str, test_path: str) -> dict:
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
    x_train = vec.fit_transform(train["text"].astype(str))
    x_test = vec.transform(test["text"].astype(str))
    clf = LinearSVC()
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
