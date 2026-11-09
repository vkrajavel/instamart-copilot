"""Part B - customer segments (RFM + K-means) and lapse prediction.

Run:  python -m copilot.part_b        (reads Part A's data/clean/*.csv)
"""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, silhouette_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CLEAN, OUT, SEED

SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")
CUTOFF = pd.Timestamp("2025-09-01 00:00:00")
FINAL_K = 4
NUMERIC = ["recency", "frequency", "monetary", "avg_discount", "cancel_rate", "avg_delivery_minutes", "age", "tenure_days"]
NOMINAL = ["city", "gender", "favourite_category"]


def load():
    orders = pd.read_csv(CLEAN / "orders.csv", parse_dates=["order_ts"])
    customers = pd.read_csv(CLEAN / "customers.csv", parse_dates=["signup_date"])
    products = pd.read_csv(CLEAN / "products.csv")
    orders = orders.merge(products[["product_id", "category"]], on="product_id")
    return orders, customers


# ------------------------------------------------------------------ task 1: RFM
def build_rfm(orders):
    comp = orders[orders["status"] == "completed"]
    g = comp.groupby("customer_id")
    return pd.DataFrame({"recency": (SNAPSHOT - g["order_ts"].max()).dt.days,
                         "frequency": g["order_id"].nunique(),
                         "monetary": g["order_amount"].sum()})


# ------------------------------------------------------------------ tasks 2-3: K-means, profiles, PCA
SEGMENT_PLAYBOOK = {
    "Premium": "VIP treatment: early access to electronics launches and a dedicated support line.",
    "Loyal Regulars": "Protect the habit: subscription / free-delivery bundles and a loyalty streak reward.",
    "Dormant": "Win-back: personalised coupon plus a reminder push before they churn for good.",
    "Occasional Shoppers": "Grow frequency: basket-building offers on staples and a second-order discount.",
}


def name_segments(profile):
    """profile: per-cluster mean recency/frequency/monetary. Premium = highest mean monetary (as required)."""
    names = {}
    premium = profile["monetary"].idxmax()
    names[premium] = "Premium"
    rest = profile.drop(index=premium)
    loyal = rest["frequency"].idxmax()
    names[loyal] = "Loyal Regulars"
    rest = rest.drop(index=loyal)
    names[rest["recency"].idxmax()] = "Dormant"
    for c in rest.index:
        names.setdefault(c, "Occasional Shoppers")
    return names


def segmentation(rfm):
    cols = ["recency", "frequency", "monetary"]
    X = StandardScaler().fit_transform(rfm[cols])
    sweep = []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X)
        sweep.append({"K": k, "inertia": round(float(km.inertia_), 1), "silhouette": round(float(silhouette_score(X, km.labels_)), 3)})
    km = KMeans(n_clusters=FINAL_K, n_init=10, random_state=SEED).fit(X)
    seg = rfm.copy()
    seg["segment_id"] = km.labels_
    profile = seg.groupby("segment_id")[cols].mean()
    names = name_segments(profile)
    seg["segment_name"] = seg["segment_id"].map(names)
    seg["is_premium"] = seg["segment_name"] == "Premium"
    prof = seg.groupby(["segment_id", "segment_name"]).agg(customers=("frequency", "size"), mean_recency=("recency", "mean"),
                                                          mean_frequency=("frequency", "mean"), mean_monetary=("monetary", "mean")).round(2).reset_index()
    prof["recommended_action"] = prof["segment_name"].map(SEGMENT_PLAYBOOK)
    pca = PCA(n_components=2).fit(X)
    out = seg.reset_index().rename(columns={"index": "customer_id"})
    out[["customer_id", "recency", "frequency", "monetary", "segment_id", "segment_name", "is_premium"]].to_csv(OUT / "customer_segments.csv", index=False)
    return seg, sweep, prof, [round(float(v), 3) for v in pca.explained_variance_ratio_]


# ------------------------------------------------------------------ tasks 4-5: lapse dataset (no look-ahead)
def build_lapse_dataset(orders, customers):
    before = orders[orders["order_ts"] < CUTOFF]
    comp = before[before["status"] == "completed"]
    g = comp.groupby("customer_id")
    f = pd.DataFrame({"recency": (CUTOFF - g["order_ts"].max()).dt.days, "frequency": g["order_id"].nunique(),
                      "monetary": g["order_amount"].sum(), "avg_discount": g["discount_pct"].mean(),
                      "avg_delivery_minutes": g["delivery_minutes"].mean(),
                      "favourite_category": g["category"].agg(lambda s: s.value_counts().sort_index().idxmax())})
    final = before[before["status"].isin(["completed", "cancelled"])]
    f["cancel_rate"] = (final["status"] == "cancelled").groupby(final["customer_id"]).mean().reindex(f.index)
    f = f.join(customers.set_index("customer_id")[["city", "age", "gender", "signup_date"]])
    f["tenure_days"] = (CUTOFF - f["signup_date"]).dt.days
    f = f.drop(columns="signup_date")
    sept = orders[(orders["status"] == "completed") & (orders["order_ts"] >= CUTOFF) & (orders["order_ts"] < pd.Timestamp("2025-10-01"))]
    y = (~f.index.isin(sept["customer_id"].unique())).astype(int)
    return f[NUMERIC + NOMINAL], pd.Series(y, index=f.index, name="lapsed")


def make_pipeline(model):
    prep = ColumnTransformer([("num", StandardScaler(), NUMERIC),
                              ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), NOMINAL)])
    return Pipeline([("prep", prep), ("model", model)])


def lapse_metrics(y, pred):
    return {"precision": round(float(precision_score(y, pred, zero_division=0)), 3), "recall": round(float(recall_score(y, pred, zero_division=0)), 3),
            "f1": round(float(f1_score(y, pred, zero_division=0)), 3), "confusion_matrix[[tn,fp],[fn,tp]]": confusion_matrix(y, pred).tolist()}


def oversample(X, y):
    """Keep every training row and add random duplicates of lapsed rows (with replacement) until classes are equal."""
    rng = np.random.RandomState(SEED)
    pos, neg = y[y == 1].index, y[y == 0].index
    extra = rng.choice(pos, size=len(neg) - len(pos), replace=True) if len(pos) < len(neg) else rng.choice(neg, size=len(pos) - len(neg), replace=True)
    idx = list(X.index) + list(extra)
    return X.loc[idx], y.loc[idx]


def main():
    orders, customers = load()
    report = {}

    # 1. RFM
    rfm = build_rfm(orders)
    report["rfm"] = {"customers": len(rfm), "recency_median": float(rfm["recency"].median()), "recency_min": int(rfm["recency"].min()),
                     "recency_max": int(rfm["recency"].max()), "frequency_mean": round(float(rfm["frequency"].mean()), 2),
                     "frequency_max": int(rfm["frequency"].max()), "monetary_mean": round(float(rfm["monetary"].mean()), 2),
                     "monetary_max": round(float(rfm["monetary"].max()), 2)}

    # 2-3. segments
    seg, sweep, profile, pca_ratio = segmentation(rfm)
    report["kmeans_sweep"] = sweep
    report["segment_profile"] = profile.to_dict(orient="records")
    report["pca_explained_variance_ratio"] = pca_ratio + [round(sum(pca_ratio), 3)]

    # 4. lapse problem
    X, y = build_lapse_dataset(orders, customers)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    base_pred = np.zeros(len(yte), dtype=int)
    report["lapse_dataset"] = {"cohort": int(len(y)), "lapsed": int(y.sum()), "lapse_rate_pct": round(100 * float(y.mean()), 2),
                               "train_rows": int(len(ytr)), "test_rows": int(len(yte)),
                               "baseline_accuracy_pct": round(100 * float((base_pred == yte).mean()), 1),
                               "baseline_lapsed_f1": round(float(f1_score(yte, base_pred, zero_division=0)), 3)}
    report["features_missing_values"] = int(X.isna().sum().sum())
    assert X.isna().sum().sum() == 0, "a feature column contains missing values"

    # 5-6. grid searches (preprocessing lives inside the pipeline, so it is refit on training folds only)
    lr_gs = GridSearchCV(make_pipeline(LogisticRegression(max_iter=1000)), {"model__C": [0.01, 0.1, 1, 10, 100]}, cv=5, scoring="f1").fit(Xtr, ytr)
    rf_gs = GridSearchCV(make_pipeline(RandomForestClassifier(n_estimators=100, random_state=SEED)),
                         {"model__max_depth": [3, 5, 8], "model__min_samples_leaf": [5, 10, 20]}, cv=5, scoring="f1").fit(Xtr, ytr)
    scaler = lr_gs.best_estimator_.named_steps["prep"].named_transformers_["num"]
    report["scaler_check"] = {"fitted_train_mean_frequency": round(float(scaler.mean_[NUMERIC.index("frequency")]), 3),
                              "full_cohort_mean_frequency": round(float(X["frequency"].mean()), 3)}
    lr_pred, rf_pred = lr_gs.predict(Xte), rf_gs.predict(Xte)
    report["logistic_regression"] = {"grid": "C in {0.01, 0.1, 1, 10, 100}, cv=5, f1", "best_params": lr_gs.best_params_,
                                     "best_cv_f1": round(float(lr_gs.best_score_), 3), "test": lapse_metrics(yte, lr_pred)}
    report["random_forest"] = {"grid": "max_depth in {3, 5, 8} x min_samples_leaf in {5, 10, 20}, cv=5, f1 (45 fits)",
                               "best_params": rf_gs.best_params_, "best_cv_f1": round(float(rf_gs.best_score_), 3), "test": lapse_metrics(yte, rf_pred)}
    best_name = "random_forest" if report["random_forest"]["test"]["f1"] >= report["logistic_regression"]["test"]["f1"] else "logistic_regression"
    best = rf_gs if best_name == "random_forest" else lr_gs
    report["selected_model"] = best_name
    assert report[best_name]["test"]["f1"] >= 0.55

    # 7. class balance (training rows only)
    Xb, yb = oversample(Xtr, ytr)
    bal = make_pipeline(LogisticRegression(max_iter=1000, C=lr_gs.best_params_["model__C"])).fit(Xb, yb)
    report["oversampling"] = {"train_class_counts_before": {int(k): int(v) for k, v in ytr.value_counts().sort_index().items()},
                              "train_class_counts_after": {int(k): int(v) for k, v in yb.value_counts().sort_index().items()},
                              "unbalanced_logreg_test": lapse_metrics(yte, lr_pred), "balanced_logreg_test": lapse_metrics(yte, bal.predict(Xte))}

    # 8. interpretation, persistence, retention list
    names = best_rf_names = rf_gs.best_estimator_.named_steps["prep"].get_feature_names_out()
    imp = rf_gs.best_estimator_.named_steps["model"].feature_importances_
    report["rf_top5_importances"] = {str(n).replace("num__", "").replace("cat__", ""): round(float(v), 3)
                                     for n, v in sorted(zip(names, imp), key=lambda t: -t[1])[:5]}
    coef = dict(zip(lr_gs.best_estimator_.named_steps["prep"].get_feature_names_out(), lr_gs.best_estimator_.named_steps["model"].coef_[0]))
    report["logreg_coefficients"] = {k: round(float(coef[f"num__{k}"]), 3) for k in ("recency", "frequency", "avg_delivery_minutes", "cancel_rate", "avg_discount")}
    joblib.dump(best.best_estimator_, OUT / "lapse_model.joblib")
    reloaded = joblib.load(OUT / "lapse_model.joblib")
    report["reload_reproduces_test_predictions"] = bool(np.array_equal(reloaded.predict(Xte), best.predict(Xte)))
    assert report["reload_reproduces_test_predictions"]

    premium_ids = set(seg.index[seg["is_premium"]])
    test_prem = Xte[Xte.index.isin(premium_ids)]
    proba = pd.Series(reloaded.predict_proba(test_prem)[:, 1], index=test_prem.index)
    ret = pd.DataFrame({"customer_id": proba.index, "lapse_probability": proba.values.round(4),
                        "actually_lapsed": yte.loc[proba.index].values}).sort_values("lapse_probability", ascending=False).head(10)
    ret.to_csv(OUT / "retention_list.csv", index=False)
    report["retention_list"] = {"premium_test_customers": int(len(test_prem)), "listed": int(len(ret)), "truly_lapsed": int(ret["actually_lapsed"].sum())}

    (OUT / "part_b_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
