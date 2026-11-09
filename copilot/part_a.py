"""Part A - audit, clean, store and query the raw exports.

Run:  python -m copilot.part_a            (then: python -m copilot.validate_a)
"""
import json
import re
import sqlite3
import numpy as np
import pandas as pd

from .config import RAW, CLEAN, DB_PATH, OUT, SQL_FILE

CITY_RE = re.compile(r"\s+")


# ------------------------------------------------------------------ audit
def table_profile(df: pd.DataFrame) -> dict:
    return {"shape": list(df.shape),
            "dtypes": {c: str(t) for c, t in df.dtypes.items()},
            "missing": {c: {"count": int(df[c].isna().sum()), "pct": round(100 * float(df[c].isna().mean()), 2)} for c in df.columns}}


def audit(customers_raw, products_raw, orders_raw) -> dict:
    amt = pd.to_numeric(orders_raw["order_amount"], errors="coerce")
    dm = pd.to_numeric(orders_raw["delivery_minutes"], errors="coerce")
    age = pd.to_numeric(customers_raw["age"], errors="coerce")
    gender = customers_raw["gender"].fillna("")
    figures = {
        "orders_raw_rows": len(orders_raw),
        "orders_exact_duplicate_rows": int(orders_raw.duplicated().sum()),
        "orders_status_pending": int((orders_raw["status"] == "pending").sum()),
        "orders_distinct_order_id": int(orders_raw["order_id"].nunique()),
        "orders_non_numeric_amounts": int(amt.isna().sum()),
        "orders_negative_delivery_minutes": int((dm < 0).sum()),
        "orders_blank_delivery_minutes": int(dm.isna().sum()),
        "customers_blank_age": int(age.isna().sum()),
        "customers_age_outside_10_100": int(((age < 10) | (age > 100)).sum()),
        "customers_distinct_raw_city_spellings": int(customers_raw["city"].nunique()),
        "customers_distinct_nonblank_gender_spellings": int(gender[gender.str.strip() != ""].nunique()),
        "customers_blank_gender": int((gender.str.strip() == "").sum()),
    }
    return {"tables": {"customers": table_profile(customers_raw), "products": table_profile(products_raw),
                       "orders": table_profile(orders_raw)}, "figures": figures}


# ------------------------------------------------------------------ cleaning
def clean_customers(df: pd.DataFrame):
    df = df.copy()
    log = {}
    before = df["city"].copy()
    df["city"] = df["city"].str.strip().str.title()
    log["city_values_changed"] = int((before != df["city"]).sum())
    g = df["gender"].fillna("").str.strip().str.lower().str[:1]
    df["gender"] = g.map({"m": "male", "f": "female"}).fillna("unknown")
    age = pd.to_numeric(df["age"], errors="coerce")
    bad = age.notna() & ~age.between(10, 100)
    log["ages_outside_10_100_set_missing"] = int(bad.sum())
    age = age.mask(bad)
    log["ages_missing_before_fill"] = int(age.isna().sum())
    df["age"] = age.fillna(age.groupby(df["city"]).transform("median"))      # city median via groupby + transform
    df["signup_date"] = pd.to_datetime(df["signup_date"])
    return df, log


def clean_orders(orders, products, customers=None):
    log = {"rows_raw": len(orders)}
    df = orders.copy()
    df["order_ts"] = pd.to_datetime(df["order_ts"])                                  # 1. datetime
    n = len(df); df = df.drop_duplicates()                                           # 2. exact duplicates
    log["exact_duplicates_removed"] = n - len(df)
    n = len(df)                                                                      # 3. latest record per order_id
    df = df.sort_values(["order_id", "order_ts"]).drop_duplicates("order_id", keep="last")
    log["superseded_records_removed"] = n - len(df)
    log["pending_rows_left"] = int((df["status"] == "pending").sum())

    price = df["product_id"].map(products.set_index("product_id")["price_inr"])      # 4. amounts
    amount = pd.to_numeric(df["order_amount"], errors="coerce")
    log["unparseable_amounts_filled_from_price"] = int(amount.isna().sum())
    fill = (price * df["quantity"] * (1 - df["discount_pct"] / 100)).round(2)
    df["order_amount"] = amount.fillna(fill)

    dm = pd.to_numeric(df["delivery_minutes"], errors="coerce")                      # 5. delivery minutes
    neg = dm < 0
    log["negative_delivery_set_missing"] = int(neg.sum())
    dm = dm.mask(neg)
    cat = df["product_id"].map(products.set_index("product_id")["category"])
    completed = df["status"] == "completed"
    cat_median = dm.where(completed).groupby(cat).transform("median")               # category median, completed only
    to_fill = completed & dm.isna()
    log["completed_delivery_filled_with_category_median"] = int(to_fill.sum())
    log["cancelled_delivery_left_missing"] = int((~completed & dm.isna()).sum())
    df["delivery_imputed"] = to_fill.astype(int)          # extra flag: lets the refund agent distrust estimated times
    df["delivery_minutes"] = dm.fillna(cat_median.where(completed))
    df["delivery_minutes"] = df["delivery_minutes"].round().astype("Int64")
    df = df.sort_values("order_id").reset_index(drop=True)
    log["rows_clean"] = len(df)
    return df, log


# ------------------------------------------------------------------ SQL
def load_queries(path=SQL_FILE):
    queries, name, buf = {}, None, []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"--\s*name:\s*(\S+)", line)
        if m:
            if name:
                queries[name] = "\n".join(buf).strip().rstrip(";")
            name, buf = m.group(1), []
        elif name:
            buf.append(line)
    if name:
        queries[name] = "\n".join(buf).strip().rstrip(";")
    return queries


def run_queries(con):
    return {name: pd.read_sql_query(q, con) for name, q in load_queries().items()}


# ------------------------------------------------------------------ pandas cross-check + EDA
def pandas_analysis(customers, products, orders):
    comp = orders[orders["status"] == "completed"].merge(customers[["customer_id", "city"]], on="customer_id") \
                                                  .merge(products[["product_id", "category"]], on="product_id")
    pivot = pd.pivot_table(comp, values="order_amount", index="city", columns="category", aggfunc="sum", margins=True, margins_name="Total")
    by_city = comp.groupby("city").agg(total_inr=("order_amount", "sum"), mean_inr=("order_amount", "mean"),
                                       orders=("order_amount", "count")).round(2)
    x = comp["order_amount"]
    q1, q3 = x.quantile(.25), x.quantile(.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    stats = {"mean": round(x.mean(), 2), "median": round(x.median(), 2), "skewness": round(float(x.skew()), 2),
             "q1": round(q1, 2), "q3": round(q3, 2), "lower_fence": round(lo, 2), "upper_fence": round(hi, 2),
             "outliers_above": int((x > hi).sum()), "outliers_below": int((x < lo).sum())}
    num = comp[["quantity", "discount_pct", "order_amount", "delivery_minutes"]].astype(float)
    corr = num.corr()
    pairs = [(a, b, float(corr.loc[a, b])) for i, a in enumerate(corr.columns) for b in corr.columns[i + 1:]]
    pairs.sort(key=lambda t: -abs(t[2]))
    return pivot, by_city, stats, pairs


def main():
    customers_raw = pd.read_csv(RAW / "customers.csv", keep_default_na=False, dtype=str).replace("", np.nan)
    customers_raw["gender"] = customers_raw["gender"].fillna("")
    products = pd.read_csv(RAW / "products.csv")
    orders_raw = pd.read_csv(RAW / "orders.csv", keep_default_na=False, dtype=str)
    orders_raw_typed = orders_raw.copy()
    for c in ("quantity", "discount_pct"):
        orders_raw_typed[c] = pd.to_numeric(orders_raw_typed[c])
    orders_raw_typed["delivery_minutes"] = orders_raw_typed["delivery_minutes"].replace("", np.nan)

    report = {"audit": audit(customers_raw.replace("", np.nan), products, orders_raw_typed.replace("", np.nan))}
    customers, clog = clean_customers(customers_raw)
    orders, olog = clean_orders(orders_raw_typed, products)
    report["customer_cleaning"], report["order_cleaning"] = clog, olog

    # ---- store: CSV + SQLite (order_ts as text YYYY-MM-DD HH:MM:SS)
    CLEAN.mkdir(parents=True, exist_ok=True)
    cust_out = customers.copy(); cust_out["signup_date"] = cust_out["signup_date"].dt.strftime("%Y-%m-%d")
    ord_out = orders.copy(); ord_out["order_ts"] = ord_out["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
    cust_out.to_csv(CLEAN / "customers.csv", index=False)
    products.to_csv(CLEAN / "products.csv", index=False)
    ord_out.to_csv(CLEAN / "orders.csv", index=False)
    if DB_PATH.exists():
        DB_PATH.unlink()
    with sqlite3.connect(DB_PATH) as con:
        cust_out.to_sql("customers", con, index=False)
        products.to_sql("products", con, index=False)
        ord_out.to_sql("orders", con, index=False)
        con.execute("CREATE INDEX idx_orders_customer ON orders(customer_id)")
        con.execute("CREATE INDEX idx_orders_ts ON orders(order_ts)")
        results = run_queries(con)

    pivot, by_city, stats, pairs = pandas_analysis(customers, products, orders)
    report["pandas"] = {"pivot_grand_total": round(float(pivot.loc["Total", "Total"]), 2), "by_city": by_city.reset_index().to_dict(orient="records"),
                        "amount_stats": stats, "top_correlations": [(a, b, round(r, 3)) for a, b, r in pairs]}
    lines = ["PART A - SQL RESULTS", ""]
    for name, res in results.items():
        lines += [f"## {name} ({len(res)} rows)", res.head(12).to_string(index=False), ""]
    lines += ["## pandas pivot (completed revenue, INR)", pivot.round(2).to_string(), "", "## by city", by_city.to_string(), ""]
    OUT.mkdir(exist_ok=True)
    (OUT / "part_a_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (OUT / "part_a_queries.txt").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "audit"}, indent=2, default=str))
    print(json.dumps(report["audit"]["figures"], indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
