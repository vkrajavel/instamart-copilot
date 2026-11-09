"""Part C - support-ticket triage: guardrails, PII redaction, resilient + validated LLM calls, budget, routing.

Run:  python -m copilot.part_c        (needs Part B's outputs/customer_segments.csv and the supplied mock_llm.py)
"""
import json
import random
import re
import time
import pandas as pd

import mock_llm
from mock_llm import TransientLLMError, est_tokens

from .config import RAW, OUT

CATEGORIES = ["billing", "delivery", "technical", "fraud"]
MAX_INPUT_TOKENS = 400
BASE_DELAY = 0.05
MAX_RETRIES = 3
DEMO_TOKEN_BUDGET = 6000
FINAL_TOKEN_BUDGET = 100000

# ---------------------------------------------------------------- task 2: redaction (exact patterns, exact order)
PATTERNS = [("EMAIL", r"[\w.+-]+@[\w-]+\.[\w.]+", "[EMAIL]"),
            ("PHONE", r"\b[6-9]\d{9}\b", "[PHONE]"),
            ("ORDER_ID", r"ORD-\d{6}", "[ORDER_ID]")]


def redact(text):
    counts = {}
    for name, pattern, placeholder in PATTERNS:
        text, n = re.subn(pattern, placeholder, text)
        counts[name] = n
    return text, counts


def contains_pii(text):
    return any(re.search(p, text) for _, p, _ in PATTERNS)


# ---------------------------------------------------------------- task 3: prompts
ZERO_SHOT = ("You are a support ticket CLASSIFY assistant for a grocery delivery app. "
             "Classify the ticket into exactly one category: billing, delivery, technical or fraud. "
             'Reply with JSON only, in the form {"category": "...", "reason": "..."}.')
FEW_SHOT = (ZERO_SHOT + "\nEXAMPLES:\n"
            'Ticket: I was billed twice for one order. -> {"category": "billing", "reason": "double charge"}\n'
            'Ticket: The rider never reached my home. -> {"category": "delivery", "reason": "delivery problem"}\n'
            'Ticket: The app freezes when I pay for my cart. -> {"category": "technical", "reason": "app failure"}\n'
            'Ticket: Someone logged in and spent my wallet money. -> {"category": "fraud", "reason": "account misuse"}')
assert "CLASSIFY" in ZERO_SHOT and "EXAMPLES:" not in ZERO_SHOT and "EXAMPLES:" in FEW_SHOT


# ---------------------------------------------------------------- task 4: resilient call
def call_with_retry(messages, ticket_id, rng, log):
    for n in range(MAX_RETRIES + 1):
        try:
            return mock_llm.chat(messages)
        except TransientLLMError:
            if n == MAX_RETRIES:
                raise
            wait = BASE_DELAY * 2 ** n * rng.uniform(0.9, 1.15)
            log.append(f"retry ticket={ticket_id} retry_number={n} wait={wait:.4f}s")
            time.sleep(wait)


# ---------------------------------------------------------------- task 5: validation
def validate(content):
    """Return (status, category). Never guesses: anything unclear is flagged for human review."""
    try:
        data = json.loads(content)
        cat = data.get("category") if isinstance(data, dict) else None
        if isinstance(cat, str) and cat in CATEGORIES:
            return "classified_json", cat
        return "flagged_invalid_output", None
    except (json.JSONDecodeError, TypeError):
        pass
    found = {w for w in re.findall(r"[a-z]+", (content or "").lower()) if w in CATEGORIES}
    if len(found) == 1:
        return "classified_keyword", next(iter(found))
    return "flagged_invalid_output", None


# ---------------------------------------------------------------- pipeline
def run_triage(tickets, system_prompt, token_budget, label):
    mock_llm.reset()                                   # required before each full run
    rng = random.Random(42)
    log, rows, total_tokens, transient = [], [], 0, 0
    for _, t in tickets.iterrows():
        text = t["text"]
        row = {"ticket_id": t["ticket_id"], "customer_id": t["customer_id"], "true_category": t["true_category"],
               "status": None, "category": None, "tokens_used": 0}
        if not text.strip():
            row["status"] = "skipped_empty"
        elif est_tokens(text) > MAX_INPUT_TOKENS:
            row["status"] = "flagged_long"
        elif total_tokens >= token_budget:
            row["status"] = "budget_exhausted"
        else:
            clean, _ = redact(text)
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": clean}]
            assert not any(contains_pii(m["content"]) for m in messages), "PII reached a prompt"
            before = len(log)
            resp = call_with_retry(messages, t["ticket_id"], rng, log)
            row["tokens_used"] = resp.usage.total_tokens
            total_tokens += resp.usage.total_tokens
            row["status"], row["category"] = validate(resp.choices[0].message.content)
        rows.append(row)
    df = pd.DataFrame(rows)
    return df, total_tokens, [f"[{label}] {l}" for l in log]


def accuracy(df):
    c = df[df["status"].isin(["classified_json", "classified_keyword"])]
    return len(c), int((c["category"] == c["true_category"]).sum())


def route(row, premium_ids):
    if row["status"] in ("skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"):
        return "human_review", "NORMAL"
    if row["category"] == "fraud":
        return "security_team", "HIGH"
    if row["customer_id"] in premium_ids:
        return "premium_desk", "HIGH"
    return "support_queue", "NORMAL"


def main():
    tickets = pd.read_csv(RAW / "tickets.csv", keep_default_na=False)       # empty texts stay ""
    report = {"tickets": len(tickets)}

    # task 1 + 2: guardrail counts and redaction counts across all 164 tickets
    est = tickets["text"].apply(est_tokens)
    report["empty_tickets"] = int((tickets["text"].str.strip() == "").sum())
    report["long_tickets"] = int((est > MAX_INPUT_TOKENS).sum())
    report["long_ticket_estimated_tokens"] = sorted(set(int(v) for v in est[est > MAX_INPUT_TOKENS]))
    totals = {n: 0 for n, _, _ in PATTERNS}
    redacted_texts = []
    for text in tickets["text"]:
        r, c = redact(text)
        redacted_texts.append(r)
        for k, v in c.items():
            totals[k] += v
    report["redaction"] = {**totals, "total": sum(totals.values()),
                           "left_after_redaction": int(sum(contains_pii(r) for r in redacted_texts))}
    assert report["redaction"]["left_after_redaction"] == 0

    # task 6: zero-shot vs few-shot (no budget)
    results, logs = {}, []
    for label, prompt in (("zero_shot", ZERO_SHOT), ("few_shot", FEW_SHOT)):
        df, tokens, log = run_triage(tickets, prompt, float("inf"), label)
        n, correct = accuracy(df)
        results[label] = df
        logs += log
        report[label] = {"status_counts": df["status"].value_counts().to_dict(), "classified": n, "correct": correct,
                         "accuracy_pct": round(100 * correct / n, 2), "total_tokens": tokens,
                         "tokens_per_classified_ticket": round(tokens / n, 1), "retries": len(log),
                         "system_prompt_estimated_tokens": est_tokens(prompt),
                         "permanent_errors": 0}
    z = results["zero_shot"]
    zc = z[z["status"].isin(["classified_json", "classified_keyword"])]
    report["zero_shot_confusion"] = pd.crosstab(zc["true_category"], zc["category"]).to_dict(orient="index")

    # task 7a: budget demo (zero-shot, TOKEN_BUDGET = 6000)
    demo, demo_tokens, log = run_triage(tickets, ZERO_SHOT, DEMO_TOKEN_BUDGET, "budget_demo")
    logs += log
    demo[["ticket_id", "customer_id", "status", "category", "tokens_used"]].to_csv(OUT / "triage_budget_demo.csv", index=False)
    report["budget_demo"] = {"token_budget": DEMO_TOKEN_BUDGET, "tokens_used": demo_tokens, "status_counts": demo["status"].value_counts().to_dict(),
                             "budget_exhausted": int((demo["status"] == "budget_exhausted").sum())}

    # task 7b: final run (few-shot, TOKEN_BUDGET = 100000) + routing with Part B's segments
    final, final_tokens, log = run_triage(tickets, FEW_SHOT, FINAL_TOKEN_BUDGET, "final")
    logs += log
    seg = pd.read_csv(OUT / "customer_segments.csv")
    premium_ids = set(seg.loc[seg["is_premium"], "customer_id"])
    names = seg.set_index("customer_id")["segment_name"]
    routed = final.apply(lambda r: route(r, premium_ids), axis=1, result_type="expand")
    final[["route", "priority"]] = routed
    final["segment_name"] = final["customer_id"].map(names).fillna("")
    final["category"] = final["category"].fillna("")
    final[["ticket_id", "customer_id", "status", "category", "segment_name", "priority", "route", "tokens_used"]].to_csv(OUT / "triage_results.csv", index=False)
    report["final_run"] = {"token_budget": FINAL_TOKEN_BUDGET, "total_tokens": final_tokens, "rows": len(final),
                           "status_counts": final["status"].value_counts().to_dict(), "route_counts": final["route"].value_counts().to_dict(),
                           "priority_counts": final["priority"].value_counts().to_dict(),
                           "fraud_to_security": int(((final["category"] == "fraud") & (final["route"] == "security_team")).sum()),
                           "premium_desk": int((final["route"] == "premium_desk").sum()),
                           "customers_without_segment_row": int((final["segment_name"] == "").sum())}

    waits = [float(re.search(r"wait=([\d.]+)s", l).group(1)) for l in logs]
    retry_numbers = {int(re.search(r"retry_number=(\d)", l).group(1)) for l in logs}
    report["retry_log"] = {"lines": len(logs), "retry_numbers_seen": sorted(retry_numbers), "min_wait_s": min(waits), "max_wait_s": max(waits),
                           "per_full_run": report["zero_shot"]["retries"]}
    (OUT / "triage_retry_log.txt").write_text("\n".join(logs) + "\n", encoding="utf-8")
    (OUT / "part_c_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
