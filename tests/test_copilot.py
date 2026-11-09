"""Fast unit tests. Run: python -m unittest discover -s tests -v   (Part A-D outputs not required)"""
import unittest
import pandas as pd

from copilot.part_a import clean_customers, clean_orders
from copilot.part_c import redact, contains_pii, validate, route, ZERO_SHOT, FEW_SHOT
from copilot.part_d import chunk_documents


class TestRedaction(unittest.TestCase):
    def test_redact_counts_and_order(self):
        text, counts = redact("Mail user7@example.com or call 9876543210 about ORD-100001, also 6123456789.")
        self.assertEqual(counts, {"EMAIL": 1, "PHONE": 2, "ORDER_ID": 1})
        self.assertEqual(text, "Mail [EMAIL] or call [PHONE] about [ORDER_ID], also [PHONE].")
        self.assertFalse(contains_pii(text))


class TestValidation(unittest.TestCase):
    def test_json(self):
        self.assertEqual(validate('{"category": "fraud", "reason": "x"}'), ("classified_json", "fraud"))

    def test_chatty_keyword(self):
        self.assertEqual(validate("The customer appears to have a billing-related problem."), ("classified_keyword", "billing"))

    def test_invalid_never_guessed(self):
        self.assertEqual(validate('{"category": "shipping", "reason": "x"}'), ("flagged_invalid_output", None))
        self.assertEqual(validate("maybe billing or delivery"), ("flagged_invalid_output", None))

    def test_prompts(self):
        self.assertIn("EXAMPLES:", FEW_SHOT)
        self.assertNotIn("EXAMPLES:", ZERO_SHOT)


class TestRouting(unittest.TestCase):
    def test_rules(self):
        base = {"status": "classified_json", "category": "billing", "customer_id": "C1"}
        self.assertEqual(route(base, {"C1"}), ("premium_desk", "HIGH"))
        self.assertEqual(route(base, set()), ("support_queue", "NORMAL"))
        self.assertEqual(route({**base, "category": "fraud"}, {"C1"}), ("security_team", "HIGH"))
        self.assertEqual(route({**base, "status": "budget_exhausted"}, {"C1"}), ("human_review", "NORMAL"))


class TestChunking(unittest.TestCase):
    def test_reference_counts(self):
        chunks = chunk_documents()                     # needs data/policies from generate_data.py
        counts = pd.Series([c["source"] for c in chunks]).value_counts().to_dict()
        self.assertEqual(len(chunks), 15)
        self.assertEqual(counts["refund_policy.txt"], 4)
        self.assertEqual(len(chunk_documents(overlap=0)), 13)


class TestCleaning(unittest.TestCase):
    def test_customers(self):
        df = pd.DataFrame({"customer_id": ["C1", "C2", "C3"], "city": [" CHENNAI ", "chennai", "pune"],
                           "signup_date": ["2024-01-01"] * 3, "age": ["280", "30", "40"], "gender": ["F", "", "m"]})
        out, log = clean_customers(df)
        self.assertEqual(list(out["city"]), ["Chennai", "Chennai", "Pune"])
        self.assertEqual(out.loc[0, "age"], 30)                      # 280 -> missing -> median of Chennai (30)
        self.assertEqual(list(out["gender"]), ["female", "unknown", "male"])
        self.assertEqual(log["ages_outside_10_100_set_missing"], 1)

    def test_orders(self):
        products = pd.DataFrame({"product_id": ["P001"], "price_inr": [100], "category": ["Grocery"]})
        base = dict(customer_id="C1", product_id="P001", quantity=2, discount_pct=10)
        raw = pd.DataFrame([
            dict(order_id="A", order_ts="2025-04-01 10:00:00", status="pending", order_amount="180", delivery_minutes=None, **base),
            dict(order_id="A", order_ts="2025-04-01 10:05:00", status="completed", order_amount="N/A", delivery_minutes=-10.0, **base),
            dict(order_id="A", order_ts="2025-04-01 10:05:00", status="completed", order_amount="N/A", delivery_minutes=-10.0, **base),
            dict(order_id="B", order_ts="2025-04-02 10:00:00", status="completed", order_amount="180", delivery_minutes=20.0, **base),
            dict(order_id="C", order_ts="2025-04-03 10:00:00", status="cancelled", order_amount="180", delivery_minutes=None, **base)])
        out, log = clean_orders(raw, products)
        self.assertEqual(list(out["order_id"]), ["A", "B", "C"])
        a = out.set_index("order_id")
        self.assertEqual(a.loc["A", "order_amount"], 180.0)          # filled from price x qty x (1 - discount)
        self.assertEqual(a.loc["A", "delivery_minutes"], 20)         # negative -> missing -> category median of completed
        self.assertTrue(pd.isna(a.loc["C", "delivery_minutes"]))     # cancelled stays missing
        self.assertEqual(log["exact_duplicates_removed"], 1)
        self.assertEqual(log["superseded_records_removed"], 1)


if __name__ == "__main__":
    unittest.main()
