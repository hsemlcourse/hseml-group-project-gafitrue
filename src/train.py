import pandas as pd
import joblib
import scipy.sparse as sp
import os
import time

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

from preprocessing import load_data, preprocess


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    df = load_data("data/cve.csv")
    df = preprocess(df)

    print("\nClass distribution:")
    print(df["severity"].value_counts(normalize=True))

    tfidf = TfidfVectorizer(max_features=3000)
    X_text = tfidf.fit_transform(df["summary"])

    extra_cols = [col for col in df.columns if col.startswith("has_")] + ["desc_len", "year"]
    X_extra = df[extra_cols].values

    X = sp.hstack([X_text, X_extra])
    y = df["severity"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "LogReg (baseline)": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Naive Bayes": MultinomialNB(),
        "RandomForest": RandomForestClassifier(n_estimators=150, random_state=42),
    }

    results = []

    best_model = None
    best_score = 0

    for name, model in models.items():
        start = time.time()

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        f1 = f1_score(y_test, preds, average="macro")
        acc = accuracy_score(y_test, preds)
        duration = time.time() - start

        results.append({
            "model": name,
            "f1_macro": round(f1, 4),
            "accuracy": round(acc, 4),
            "train_time_sec": round(duration, 2)
        })

        print(f"{name}: F1={f1:.4f}, Acc={acc:.4f}")

        if f1 > best_score:
            best_score = f1
            best_model = model

    results_df = pd.DataFrame(results).sort_values(by="f1_macro", ascending=False)
    results_df.to_csv("reports/experiments.csv", index=False)

    joblib.dump(best_model, "models/model.pkl")
    joblib.dump(tfidf, "models/tfidf.pkl")

    print("\nFinal table:\n", results_df)


if __name__ == "__main__":
    main()