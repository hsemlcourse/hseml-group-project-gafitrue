"""Обучение моделей и эксперименты.

Закрывает оставшуюся часть моделирования (CP2):
* baseline без feature engineering (logreg на голом тексте);
* 6 продвинутых моделей: logreg, NB, KNN, SVM, RandomForest, GradientBoosting,
  LightGBM, XGBoost — итого 7 базовых;
* 2 ансамбля: Voting (soft) и Stacking;
* RandomizedSearchCV по гиперпараметрам топ-3 моделей;
* эксперимент с уменьшением размерности через TruncatedSVD;
* feature importance для интерпретируемости.

Все результаты пишутся в reports/experiments.csv.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

from src.config import MODELS_DIR, REPORTS_DIR, SEED, set_seed
from src.features import build_feature_pipeline
from src.preprocessing import load_data, preprocess, split_by_year

# Опциональные бустинги — не валим импорт, если их нет
try:
    from lightgbm import LGBMClassifier  # type: ignore
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    from xgboost import XGBClassifier  # type: ignore
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


@dataclass
class ExperimentResult:
    name: str
    f1_macro_val: float
    f1_macro_test: float
    accuracy_test: float
    train_time_sec: float
    notes: str = ""


def _evaluate(model, X_val, y_val, X_test, y_test, name: str,
              elapsed: float, notes: str = "") -> ExperimentResult:
    f1_val = f1_score(y_val, model.predict(X_val), average="macro")
    preds = model.predict(X_test)
    f1_test = f1_score(y_test, preds, average="macro")
    acc = accuracy_score(y_test, preds)
    return ExperimentResult(
        name=name,
        f1_macro_val=round(f1_val, 4),
        f1_macro_test=round(f1_test, 4),
        accuracy_test=round(acc, 4),
        train_time_sec=round(elapsed, 2),
        notes=notes,
    )


def _fit_eval(model, X_tr, y_tr, X_val, y_val, X_te, y_te,
              name: str, notes: str = "") -> tuple[ExperimentResult, object]:
    t = time.time()
    model.fit(X_tr, y_tr)
    elapsed = time.time() - t
    res = _evaluate(model, X_val, y_val, X_te, y_te, name, elapsed, notes)
    print(f"  {name:32s} f1_val={res.f1_macro_val:.4f} "
          f"f1_test={res.f1_macro_test:.4f} acc={res.accuracy_test:.4f} "
          f"({elapsed:.1f}s)")
    return res, model


# --- Эксперименты ---------------------------------------------------------
def run_baseline(split, results: list[ExperimentResult]) -> None:
    """Эксперимент 0: baseline = TF-IDF + LogReg без feature engineering."""
    print("\n[Baseline] TF-IDF + LogReg по голому тексту")
    tfidf = TfidfVectorizer(max_features=3000)
    X_tr = tfidf.fit_transform(split.X_train["summary"])
    X_val = tfidf.transform(split.X_val["summary"])
    X_te = tfidf.transform(split.X_test["summary"])

    model = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=SEED,
    )
    res, _ = _fit_eval(
        model, X_tr, split.y_train, X_val, split.y_val, X_te, split.y_test,
        "LogReg (baseline, text only)", "только TF-IDF, без числовых фич",
    )
    results.append(res)


def run_models(split, results: list[ExperimentResult], quick: bool = False):
    """Эксперимент 1: 7 моделей на полном feature set."""
    print("\n[Models] Полный feature set (TF-IDF + num + cat)")

    feat = build_feature_pipeline(
        tfidf_max_features=3000 if quick else 5000,
    )
    X_tr = feat.fit_transform(split.X_train)
    X_val = feat.transform(split.X_val)
    X_te = feat.transform(split.X_test)
    print(f"  feature matrix: train={X_tr.shape}, val={X_val.shape}, "
          f"test={X_te.shape}")

    n_est = 100 if quick else 300

    models = {
        "LogReg (full features)": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=SEED,
        ),
        "Naive Bayes": MultinomialNB(),
        "KNN (k=15)": KNeighborsClassifier(n_neighbors=15, n_jobs=-1),
        "LinearSVC": LinearSVC(
            class_weight="balanced", random_state=SEED, max_iter=2000,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=n_est, n_jobs=-1, random_state=SEED,
            class_weight="balanced",
        ),
    }
    if not quick:
        models["GradientBoosting"] = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, random_state=SEED,
        )
    if HAS_LGBM:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=n_est, learning_rate=0.1, num_leaves=63,
            random_state=SEED, n_jobs=-1, verbose=-1, class_weight="balanced",
        )
    if HAS_XGB and not quick:
        models["XGBoost"] = XGBClassifier(
            n_estimators=n_est, learning_rate=0.1, max_depth=6,
            random_state=SEED, n_jobs=-1, tree_method="hist",
            eval_metric="mlogloss",
        )

    trained: dict[str, object] = {}
    for name, model in models.items():
        try:
            res, fitted = _fit_eval(
                model, X_tr, split.y_train, X_val, split.y_val,
                X_te, split.y_test, name,
            )
            results.append(res)
            trained[name] = fitted
        except Exception as e:  # noqa: BLE001
            print(f"  {name} FAILED: {e}")

    return feat, trained, (X_tr, X_val, X_te)


def run_dim_reduction(split, X_tr, X_val, X_te, results) -> None:
    """Эксперимент 2: TruncatedSVD для разреженного TF-IDF."""
    print("\n[Dim reduction] TruncatedSVD + LogReg")
    for n_comp in (100, 300):
        svd = TruncatedSVD(n_components=n_comp, random_state=SEED)
        Xs_tr = svd.fit_transform(X_tr)
        Xs_val = svd.transform(X_val)
        Xs_te = svd.transform(X_te)
        explained = svd.explained_variance_ratio_.sum()
        model = LogisticRegression(
            max_iter=1000, class_weight="balanced",
            random_state=SEED, n_jobs=-1,
        )
        res, _ = _fit_eval(
            model, Xs_tr, split.y_train, Xs_val, split.y_val,
            Xs_te, split.y_test,
            f"LogReg + SVD(n={n_comp})",
            f"explained_var={explained:.3f}",
        )
        results.append(res)


def run_hp_search(split, X_tr, X_val, X_te, results,
                  quick: bool = False) -> object:
    """Эксперимент 3: подбор гиперпараметров для топ-моделей."""
    print("\n[HP search] RandomizedSearchCV по LogReg и LightGBM/RF")

    cv = StratifiedKFold(n_splits=2 if quick else 3, shuffle=True,
                          random_state=SEED)
    n_iter_lr = 3 if quick else 8
    n_iter_boost = 3 if quick else 10

    # LogReg: регуляризация + штраф
    print("  -> LogReg")
    lr_grid = {
        "C": [0.1, 0.5, 1.0, 2.0, 5.0],
        "penalty": ["l2"],
        "solver": ["lbfgs", "saga"],
    }
    lr_search = RandomizedSearchCV(
        LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=SEED,
        ),
        lr_grid, n_iter=n_iter_lr, cv=cv, scoring="f1_macro",
        random_state=SEED, n_jobs=-1, verbose=0,
    )
    t = time.time()
    lr_search.fit(X_tr, split.y_train)
    elapsed = time.time() - t
    res = _evaluate(
        lr_search.best_estimator_, X_val, split.y_val,
        X_te, split.y_test,
        f"LogReg (tuned: {lr_search.best_params_})",
        elapsed, notes=f"cv_best_f1={lr_search.best_score_:.4f}",
    )
    print(f"  LogReg tuned f1_test={res.f1_macro_test:.4f} ({elapsed:.1f}s)")
    results.append(res)

    # Boosting tuning
    best_boost = None
    if HAS_LGBM:
        print("  -> LightGBM")
        lgbm_grid = {
            "n_estimators": [100, 200, 400] if quick else [200, 400, 600],
            "learning_rate": [0.05, 0.1, 0.15],
            "num_leaves": [31, 63] if quick else [31, 63, 127],
            "min_child_samples": [10, 20, 40],
            "reg_alpha": [0.0, 0.1],
        }
        lgbm_search = RandomizedSearchCV(
            LGBMClassifier(
                random_state=SEED, n_jobs=-1, verbose=-1,
                class_weight="balanced",
            ),
            lgbm_grid, n_iter=n_iter_boost, cv=cv, scoring="f1_macro",
            random_state=SEED, n_jobs=-1, verbose=0,
        )
        t = time.time()
        lgbm_search.fit(X_tr, split.y_train)
        elapsed = time.time() - t
        res = _evaluate(
            lgbm_search.best_estimator_, X_val, split.y_val,
            X_te, split.y_test,
            "LightGBM (tuned)",
            elapsed,
            notes=f"best_params={lgbm_search.best_params_}, "
                  f"cv_best_f1={lgbm_search.best_score_:.4f}",
        )
        print(f"  LightGBM tuned f1_test={res.f1_macro_test:.4f} "
              f"({elapsed:.1f}s)")
        results.append(res)
        best_boost = lgbm_search.best_estimator_
    else:
        print("  -> RandomForest (LightGBM не установлен)")
        rf_grid = {
            "n_estimators": [200, 400, 600],
            "max_depth": [None, 20, 40],
            "min_samples_split": [2, 5, 10],
            "max_features": ["sqrt", "log2"],
        }
        rf_search = RandomizedSearchCV(
            RandomForestClassifier(
                n_jobs=-1, random_state=SEED, class_weight="balanced",
            ),
            rf_grid, n_iter=n_iter_boost, cv=cv, scoring="f1_macro",
            random_state=SEED, n_jobs=-1, verbose=0,
        )
        t = time.time()
        rf_search.fit(X_tr, split.y_train)
        elapsed = time.time() - t
        res = _evaluate(
            rf_search.best_estimator_, X_val, split.y_val,
            X_te, split.y_test,
            "RandomForest (tuned)", elapsed,
            notes=f"best_params={rf_search.best_params_}, "
                  f"cv_best_f1={rf_search.best_score_:.4f}",
        )
        print(f"  RF tuned f1_test={res.f1_macro_test:.4f}")
        results.append(res)
        best_boost = rf_search.best_estimator_

    return lr_search.best_estimator_, best_boost


def run_ensembles(split, X_tr, X_val, X_te, lr_best, boost_best, results):
    """Эксперимент 4: ансамбли — Voting и Stacking."""
    print("\n[Ensembles] Voting + Stacking")

    estimators = [
        ("lr", lr_best),
        ("boost", boost_best),
        ("nb", MultinomialNB()),
    ]

    # Voting soft требует predict_proba у всех -> NB и boost умеют, LinearSVC нет
    voting = VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)
    res, fitted_voting = _fit_eval(
        voting, X_tr, split.y_train, X_val, split.y_val,
        X_te, split.y_test, "Voting (soft)",
    )
    results.append(res)

    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(
            max_iter=1000, random_state=SEED, n_jobs=-1,
        ),
        cv=3, n_jobs=-1, passthrough=False,
    )
    res, fitted_stack = _fit_eval(
        stacking, X_tr, split.y_train, X_val, split.y_val,
        X_te, split.y_test, "Stacking",
    )
    results.append(res)

    return fitted_voting, fitted_stack


def run_feature_importance(model, feature_pipeline, X_te, y_te,
                            top_k: int = 30) -> pd.DataFrame:
    """Интерпретируемость: коэффициенты LogReg по каждому классу."""
    print("\n[Feature importance] LogReg coefficients (per class)")

    feature_names = feature_pipeline.get_feature_names_out()
    if hasattr(model, "coef_"):
        coefs = model.coef_  # shape: (n_classes, n_features)
    else:
        print("  модель не имеет coef_, пропускаю")
        return pd.DataFrame()

    from src.config import SEVERITY_MAP
    rows = []
    for cls_idx in range(coefs.shape[0]):
        idx_top = np.argsort(coefs[cls_idx])[::-1][:top_k]
        for rank, fi in enumerate(idx_top):
            rows.append({
                "severity": SEVERITY_MAP[cls_idx],
                "rank": rank + 1,
                "feature": feature_names[fi],
                "coef": round(float(coefs[cls_idx, fi]), 4),
            })
    df = pd.DataFrame(rows)
    out = REPORTS_DIR / "feature_importance.csv"
    df.to_csv(out, index=False)
    print(f"  saved -> {out}")
    return df


# --- Main -----------------------------------------------------------------
def main(quick: bool = False, sample_size: int | None = None) -> None:
    set_seed()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/6] Загрузка и препроцессинг")
    df = load_data()
    df = preprocess(df)

    if quick:
        sample_size = sample_size or 8000
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=SEED)
        print(f"[sample] subsample to {len(df)}")

    print(f"  total samples: {len(df)}")

    print("[2/6] Time-based split (val<2017, test>=2019)")
    split = split_by_year(df)
    print(f"  train={len(split.X_train)}, val={len(split.X_val)}, "
          f"test={len(split.X_test)}")
    print(f"  class dist train:\n{split.y_train.value_counts(normalize=True)}")

    results: list[ExperimentResult] = []

    print("\n[3/6] Baseline")
    run_baseline(split, results)

    print("\n[4/6] Семь моделей на полном feature set")
    feature_pipeline, trained, mats = run_models(split, results, quick=quick)
    X_tr, X_val, X_te = mats

    print("\n[5/6] Dim reduction + HP search")
    run_dim_reduction(split, X_tr, X_val, X_te, results)
    lr_best, boost_best = run_hp_search(
        split, X_tr, X_val, X_te, results, quick=quick,
    )

    print("\n[6/6] Ансамбли + feature importance")
    voting, stacking = run_ensembles(
        split, X_tr, X_val, X_te, lr_best, boost_best, results,
    )

    # --- Сохранение результатов ----------------------------------------
    results_df = pd.DataFrame([r.__dict__ for r in results])
    results_df = results_df.sort_values("f1_macro_test", ascending=False)
    results_df.to_csv(REPORTS_DIR / "experiments.csv", index=False)
    print("\n=== Итоговая таблица экспериментов ===")
    print(results_df.to_string(index=False))

    # Выбираем финальную модель по f1_macro на TEST (val уже использовали для HP)
    final_row = results_df.iloc[0]
    final_name = final_row["name"]
    print(f"\nФинальная модель: {final_name} "
          f"(f1_macro_test={final_row['f1_macro_test']})")

    # Маппинг имя -> объект
    final_model_map = {
        **trained,
        f"LogReg (tuned: {lr_best.get_params(deep=False)})": lr_best,
        "Voting (soft)": voting,
        "Stacking": stacking,
    }
    # Берём лучший по имени или fallback на boost_best
    final_model = boost_best
    for k, v in final_model_map.items():
        if k == final_name:
            final_model = v
            break

    # Сохраняем пайплайн целиком
    joblib.dump(feature_pipeline, MODELS_DIR / "feature_pipeline.pkl")
    joblib.dump(final_model, MODELS_DIR / "model.pkl")
    with open(MODELS_DIR / "model_meta.json", "w", encoding="utf-8") as f:
        json.dump({
            "final_model": final_name,
            "f1_macro_test": final_row["f1_macro_test"],
            "accuracy_test": final_row["accuracy_test"],
            "seed": SEED,
        }, f, indent=2, ensure_ascii=False)

    # Classification report финальной модели
    preds = final_model.predict(X_te)
    cls_report = classification_report(
        split.y_test, preds, target_names=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
        digits=4,
    )
    cm = confusion_matrix(split.y_test, preds)
    with open(REPORTS_DIR / "final_model_report.txt", "w",
              encoding="utf-8") as f:
        f.write(f"Финальная модель: {final_name}\n\n")
        f.write("Classification report (test):\n")
        f.write(cls_report)
        f.write("\n\nConfusion matrix:\n")
        f.write(str(cm))

    # Feature importance — на отдельно дообученной LogReg
    run_feature_importance(lr_best, feature_pipeline, X_te, split.y_test)

    print("\nГотово. Артефакты:")
    print(f"  {MODELS_DIR}/model.pkl, feature_pipeline.pkl, model_meta.json")
    print(f"  {REPORTS_DIR}/experiments.csv, final_model_report.txt, "
          f"feature_importance.csv")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true",
                   help="прогон на 8к строк для отладки")
    p.add_argument("--sample-size", type=int, default=None,
                   help="ограничить размер выборки (по умолчанию весь датасет)")
    args = p.parse_args()
    main(quick=args.quick, sample_size=args.sample_size)
