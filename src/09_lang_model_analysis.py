import random
import pathlib

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


def load_sample(path, seed=42, max_examples_per_class=20000):
    df = pd.read_parquet(path, columns=["text_orig", "lang_group"])
    df = df.dropna(subset=["text_orig", "lang_group"]).copy()
    groups = []
    for label in ["ko", "en", "other"]:
        group = df[df["lang_group"] == label]
        if len(group) > max_examples_per_class:
            group = group.sample(max_examples_per_class, random_state=seed)
        groups.append(group)
    return pd.concat(groups).sample(frac=1, random_state=seed).reset_index(drop=True)


def train_and_evaluate(path):
    print("Loading sample data...")
    df = load_sample(path, max_examples_per_class=20000)
    print(f"Sample size: {len(df)}")
    print(df["lang_group"].value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        df["text_orig"], df["lang_group"], test_size=0.2, random_state=42, stratify=df["lang_group"]
    )

    print("Extracting TF-IDF text features...")
    vect = TfidfVectorizer(
        min_df=5,
        max_df=0.9,
        ngram_range=(1, 2),
        max_features=25000,
        lowercase=True,
        strip_accents="unicode",
    )
    X_train_tfidf = vect.fit_transform(X_train)
    X_test_tfidf = vect.transform(X_test)

    print("Training logistic regression classifier...")
    clf = LogisticRegression(
        solver="saga",
        penalty="l2",
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )
    clf.fit(X_train_tfidf, y_train)

    print("Evaluating model...")
    y_pred = clf.predict(X_test_tfidf)
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred, labels=["ko", "en", "other"]))

    top_features = get_top_features(vect, clf, labels=["ko", "en", "other"], n=15)
    print("Top predictor tokens per class:")
    for label, terms in top_features.items():
        print(f"{label}: {', '.join(terms)}")

    sample_results = pd.DataFrame({"text": X_test, "true": y_test, "pred": y_pred})
    sample_hits = sample_results[sample_results["true"] == sample_results["pred"]].sample(5, random_state=42)
    sample_misses = sample_results[sample_results["true"] != sample_results["pred"]].sample(5, random_state=42)
    print("\nExample correct predictions:")
    print(sample_hits.to_string(index=False, max_colwidth=120))
    print("\nExample errors:")
    print(sample_misses.to_string(index=False, max_colwidth=120))


def get_top_features(vect, clf, labels, n=15):
    feature_names = np.array(vect.get_feature_names_out())
    coefs = clf.coef_
    top = {}
    for idx, label in enumerate(labels):
        top_indices = np.argsort(coefs[idx])[-n:][::-1]
        top[label] = feature_names[top_indices].tolist()
    return top


if __name__ == "__main__":
    path = pathlib.Path(__file__).resolve().parents[1] / "data" / "processed" / "reviews_final_th03.parquet"
    train_and_evaluate(str(path))
