"""Part A validation: asserts the reference checkpoints and prints ALL CHECKPOINTS PASSED if every one holds.

Run:  python -m copilot.validate_a
"""
import sqlite3
import sys
import pandas as pd

from .config import CLEAN, DB_PATH, RAW
from .part_a import load_queries


def main():
    cust = pd.read_csv(CLEAN / "customers.csv")
    orders = pd.read_csv(CLEAN / "orders.csv")
    raw = pd.read_csv(RAW / "orders.csv", dtype=str, keep_default_na=False)
    comp = orders[orders["status"] == "completed"]
    canc = orders[orders["status"] == "cancelled"]
    con = sqlite3.connect(DB_PATH)
    q = {k: pd.read_sql_query(v, con) for k, v in load_queries().items()}
    q6 = q["q6_rank_within_city"]
    checks = {
        "raw order rows = 13084": len(raw) == 13084,
        "raw exact duplicates = 119": int(raw.duplicated().sum()) == 119,
        "cities = 6 with reference counts": cust["city"].value_counts().to_dict() ==
            {"Pune": 227, "Delhi": 210, "Bengaluru": 197, "Chennai": 191, "Hyderabad": 191, "Mumbai": 184},
        "genders female 610 / male 560 / unknown 30": cust["gender"].value_counts().to_dict() == {"female": 610, "male": 560, "unknown": 30},
        "0 missing ages, all between 18 and 60": cust["age"].isna().sum() == 0 and cust["age"].between(18, 60).all(),
        "clean orders = 12309 unique order_id": len(orders) == 12309 and orders["order_id"].is_unique,
        "11110 completed and 1199 cancelled": len(comp) == 11110 and len(canc) == 1199,
        "no pending rows": (orders["status"] == "pending").sum() == 0,
        "0 missing order_amount": orders["order_amount"].isna().sum() == 0,
        "0 missing delivery (completed), 1199 missing (cancelled)": comp["delivery_minutes"].isna().sum() == 0 and canc["delivery_minutes"].isna().sum() == 1199,
        "completed revenue = INR 2998668.80": round(comp["order_amount"].sum(), 2) == 2998668.80,
        "database has customers, products, orders": {"customers", "products", "orders"} <= set(
            pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", con)["name"]),
        "Q1 = 30 customers": len(q["q1_big_orders_having"]) == 30,
        "Q2 = 30 rows summing to INR 2998668.80": len(q["q2_revenue_city_category"]) == 30 and round(q["q2_revenue_city_category"]["revenue_inr"].sum(), 2) == 2998668.80,
        "Q3 = 11 products": len(q["q3_products_above_average_price"]) == 11,
        "Q4 = 25 customers": len(q["q4_customers_without_completed_order"]) == 25,
        "Q5 = 550 customers, average 2552.06": len(q["q5_above_average_spenders"]) == 550 and q["q5_above_average_spenders"]["average_spend_inr"].iloc[0] == 2552.06,
        "Q6: 8 rows with RANK != ROW_NUMBER": int((q6["spend_rank"] != q6["spend_row_number"]).sum()) == 8,
        "Q7 = 183 days, last cumulative INR 2998668.80": len(q["q7_daily_revenue_running_total"]) == 183 and q["q7_daily_revenue_running_total"]["cumulative_revenue_inr"].iloc[-1] == 2998668.80,
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        print("FAILED: " + "; ".join(failed))
        sys.exit(1)
    print("ALL CHECKPOINTS PASSED")


if __name__ == "__main__":
    main()
