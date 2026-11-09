# Instamart Customer Intelligence & Support Copilot

## Overview

One repository, four connected parts, all on branch `main`. Raw exports are audited and cleaned into a SQLite database (Part A), the clean
orders feed customer segments and a lapse-prediction model (Part B), Part B's segments route support tickets that have been privacy-redacted,
validated and budget-controlled (Part C), and the same database plus the policy documents power a policy Q&A assistant and a tool-using refund
agent (Part D). All data is synthetic; all money is in INR. All model calls use the supplied offline `mock_llm.py` (no network, no API key).
Libraries: Python standard library, pandas, numpy, scikit-learn, joblib.

| Path | What it is |
|---|---|
| `generate_data.py`, `mock_llm.py` | supplied files, unmodified |
| `copilot/part_a.py`, `validate_a.py` | audit, cleaning, SQLite, SQL, pandas analysis; validation script |
| `copilot/part_b.py`, `part_c.py`, `part_d.py` | segments + lapse model; ticket triage; RAG + refund agent |
| `sql/queries.sql` | the seven Part A queries |
| `data/clean/*.csv`, `data/instamart.db` | Part A outputs (tables `customers`, `products`, `orders`) |
| `outputs/` | `customer_segments.csv`, `lapse_model.joblib`, `retention_list.csv`, `triage_results.csv`, `triage_budget_demo.csv`, `agent_runs.jsonl`, `refund_ledger.jsonl` and report files (`part_*_report.json`, `part_a_queries.txt`, `triage_retry_log.txt`, `agent_demos.jsonl`) |
| `state/` | agent checkpoints (one JSON file per run) |
| `tests/test_copilot.py`, `run_all.py` | 9 unit tests; runs Parts A-D in order |

## Reproduce

1. `git clone <repo-url> && cd instamart-copilot` (branch `main`).
2. `python -m venv .venv` then `source .venv/bin/activate` (Windows: `.venv\Scripts\activate`).
3. `pip install -r requirements.txt`, then `pip freeze > requirements.txt` to pin the exact versions you tested with (the committed file lists the four libraries).
4. `python generate_data.py` - last line must be `generated: customers=1200 products=40 orders_raw=13084 tickets=164`.
5. Part A: `python -m copilot.part_a` then `python -m copilot.validate_a` - the validation script prints `ALL CHECKPOINTS PASSED`.
6. Parts B-D: `python -m copilot.part_b`, `python -m copilot.part_c`, `python -m copilot.part_d` (steps 5-6 together: `python run_all.py`).
7. Optional: `python -m unittest discover -s tests`.

All randomness uses seed 42. Re-running regenerates every output identically except the timestamps inside `outputs/refund_ledger.jsonl`.
Part C waits about 0.05 s per simulated retry (a few seconds in total).

## Part A - Data foundation

**Audit of the raw data** (`outputs/part_a_report.json` has shape, dtypes and per-column missing counts for all three tables).
customers 1,200 x 5 (missing: age 61 = 5.08%, gender 30 = 2.50%); products 40 x 4 (none missing); orders 13,084 x 9
(missing: delivery_minutes 2,541 = 19.42%; amount problems are text, not blanks).

| Audit figure | Value |
|---|---|
| raw `orders.csv` rows / exact duplicate rows / `pending` rows / distinct `order_id` | 13,084 / 119 / 656 / 12,309 |
| non-numeric `order_amount` / negative `delivery_minutes` / blank `delivery_minutes` | 120 / 106 / 2,541 |
| customers: blank age / age outside 10-100 | 61 / 6 |
| customers: distinct raw city spellings / distinct non-blank gender spellings / blank gender | 24 / 10 / 30 |

**Customer cleaning rules** (rows or values changed): city stripped and title-cased - 476 values changed, exactly six cities remain
(Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, Mumbai 184); gender mapped to male / female / unknown (blank -> unknown): female 610,
male 560, unknown 30; ages outside 10-100 set to missing (6), then all 67 missing ages (61 blank + 6) filled with the median age of the customer's city
via `groupby().transform("median")` - 21 of the filled values are x.5 medians; 0 missing, all ages between 18 and 60; `signup_date` converted to datetime (1,200 values).

**Order cleaning rules**, applied in this order:

| # | Rule | Rows / values changed |
|---|---|---|
| 1 | `order_ts` converted to datetime | 13,084 |
| 2 | exact duplicate rows removed | 119 removed |
| 3 | per `order_id` keep the record with the latest `order_ts` (drops the earlier `pending` records) | 656 removed -> 12,309 rows, 0 pending |
| 4 | unparseable `order_amount` (`N/A`, `error`) -> missing -> `round(price_inr x quantity x (1 - discount_pct/100), 2)` | 112 filled |
| 5a | negative `delivery_minutes` -> missing | 105 set missing (106 in the raw file; one sat in a removed duplicate row) |
| 5b | completed orders: missing `delivery_minutes` -> median of the product category among completed orders (`groupby().transform`) | 774 filled |
| 5c | cancelled orders keep missing `delivery_minutes` (a cancelled order was never delivered) | 1,199 left missing |

Result: 12,309 rows (11,110 completed, 1,199 cancelled), 0 missing `order_amount`, 0 missing `delivery_minutes` among completed orders, completed revenue INR 2,998,668.80.
The cleaned `orders` table carries one extra column, `delivery_imputed` (1 for the 774 estimated delivery times), so later code can tell estimates from real times.

**Storage and validation.** `data/clean/{customers,products,orders}.csv` and `data/instamart.db` (tables `customers`, `products`, `orders`; `order_ts` stored as text `YYYY-MM-DD HH:MM:SS`).
`python -m copilot.validate_a` asserts 19 reference checkpoints (cleaning counts, revenue, city/gender counts, all seven SQL results) and prints `ALL CHECKPOINTS PASSED`.

**SQL results** (`sql/queries.sql`, run with `pd.read_sql_query`; full output in `outputs/part_a_queries.txt`). Only completed orders count.

| Query | Result |
|---|---|
| Q1 WHERE + HAVING (orders > INR 500, customer total > INR 8,000) | 30 customers |
| Q2 revenue by city x category (3-table join) | 30 rows, sum INR 2,998,668.80, largest cell Bengaluru / Electronics INR 233,909.45 |
| Q3 products priced above the average catalogue price (scalar subquery) | 11 products |
| Q4 customers with no completed order (LEFT JOIN) | 25 customers |
| Q5 CTE: customers above the average customer spend | 550 customers; average INR 2,552.06 |
| Q6 RANK / DENSE_RANK / ROW_NUMBER per city | top per city: Bengaluru C0082 12,856.20; Chennai C0930 10,151.40; Delhi C0291 13,742.60; Hyderabad C0743 12,730.05; Mumbai C0693 14,766.55; Pune C0426 10,377.10; 8 rows have RANK different from ROW_NUMBER |
| Q7 daily revenue with running total | 183 days; last cumulative value INR 2,998,668.80 |

In Q6, customers with identical total spend share the same RANK but receive consecutive ROW_NUMBERs, so every tied customer after the first gets a ROW_NUMBER larger than its RANK.

**pandas cross-check.** `pd.pivot_table(..., margins=True)` grand total INR 2,998,668.80 equals the SQL total. Completed order_amount by city (sum / mean / count):
Bengaluru 523,643.40 / 276.47 / 1,894; Chennai 463,629.15 / 272.72 / 1,700; Delhi 502,829.80 / 243.62 / 2,064; Hyderabad 528,589.90 / 284.95 / 1,855;
Mumbai 479,058.50 / 297.00 / 1,613; Pune 500,918.05 / 252.48 / 1,984.
Completed `order_amount`: mean 269.91, median 135.00, skewness 4.61, Q1 64.80, Q3 297.00, fences -283.50 and 645.30, outliers 897 above and 0 below.
Correlations (completed orders): order_amount vs discount_pct -0.095 (largest in absolute value), order_amount vs quantity +0.087, all others between -0.013 and 0.

**What the data says** (under 200 words). *About:* six months (April-September 2025) of quick-commerce orders from 1,200 customers in six cities across 40 products; 11,110 completed orders worth INR 2,998,668.80.
*Distribution:* amounts are right-skewed (mean 269.91 against median 135.00, skewness 4.61); the middle half lies between INR 64.80 and 297.00.
*Outliers:* 897 orders exceed the INR 645.30 fence and none fall below the lower fence; 635 of them are electronics, so they are genuine expensive items and were kept. The real anomalies
(placeholder amounts, negative delivery times, impossible ages) were fixed in cleaning. *Relationships:* the correlations among amount, discount, quantity and delivery time are all weak; the largest in absolute value is order_amount
with discount_pct (-0.095), just ahead of quantity (+0.087). A correlation this small does not establish causation, and says nothing about whether discounts change basket value.

## Part B - Customer segments and lapse prediction

No encoder, scaler or resampler saw test rows (preprocessing sits inside a scikit-learn `Pipeline` fitted on training rows only, and oversampling touches training rows only);
no data from 2025-09-01 onward entered the features; and the segment labels from the clustering task were not used as features.

**RFM** (snapshot 2025-10-01, completed orders): 1,175 customers; recency median 19, min 0, max 180 days; frequency mean 9.46, max 33; monetary mean INR 2,552.06, max INR 14,766.55.

**K-means** on standardised RFM (`n_init=10`, seed 42):

| K | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|
| inertia | 1820.8 | 1268.2 | 743.1 | 617.4 | 503.2 | 408.5 | 366.5 |
| silhouette | 0.465 | 0.445 | 0.526 | 0.480 | 0.482 | 0.416 | 0.412 |

I chose **K = 4**: it has the best silhouette (0.526), the elbow bends there (inertia falls 41% from K=3 to K=4 but only 17% and 18% for the next two steps), and four segments map onto four distinct campaigns.
Inertia always falls as K grows, because extra centroids can only move every point closer to a centre, so K=8 beats K=2 automatically; that alone cannot pick K, which is why the silhouette, the elbow and business use also count.
Cluster numbers are only labels with no ordering.

| segment_id | Segment | Customers | Mean recency (days) | Mean frequency | Mean monetary (INR) | Recommended action |
|---|---|---|---|---|---|---|
| 1 | **Premium** (highest mean monetary) | 138 | 24.12 | 6.77 | 7,129.70 | VIP treatment: early access to electronics launches, dedicated support line |
| 2 | Loyal Regulars | 419 | 8.77 | 19.40 | 3,410.02 | protect the habit: free-delivery bundle, streak reward |
| 0 | Occasional Shoppers | 436 | 28.86 | 3.77 | 1,057.62 | grow frequency: basket-building offers, second-order discount |
| 3 | Dormant | 182 | 109.04 | 2.20 | 685.99 | win-back coupon and reminder push |

PCA (2 components) explains 0.634 and 0.218 of the variance (0.852 in total). `outputs/customer_segments.csv` has `customer_id, recency, frequency, monetary, segment_id, segment_name, is_premium`.

**Lapse problem** (cutoff 2025-09-01): cohort = customers with a completed order before the cutoff = 1,156; `lapsed` = no completed order in 2025-09-01..2025-09-30: 443 lapsed (38.32%).
Features use only orders before the cutoff: recency, frequency, monetary, average discount, cancellation rate, average delivery minutes, favourite category, city, age, gender, tenure. Split (`test_size=0.2`, stratified, seed 42): 924 train / 232 test.
Always predicting "not lapsed" gives accuracy 61.6% and lapsed-class F1 0.
Preprocessing: `OneHotEncoder(drop="first")` for city, gender and favourite category and `StandardScaler` for numeric features, inside the Pipeline; no feature has a missing value (checked in code).
The scaler's learned mean of `frequency` is 8.115 against 8.029 for the full cohort because it is fitted on the 924 training rows only, not on all 1,156.

| Model (lapsed class, test set) | Best params (5-fold CV F1) | Precision | Recall | F1 | Confusion [[TN, FP], [FN, TP]] |
|---|---|---|---|---|---|
| Baseline (never lapsed) | - | 0 | 0 | 0 | accuracy 61.6% |
| Logistic regression, grid C in {0.01, 0.1, 1, 10, 100} | C = 100 (0.651) | 0.644 | 0.652 | 0.648 | [[111, 32], [31, 58]] |
| Random forest, grid max_depth {3, 5, 8} x min_samples_leaf {5, 10, 20} (45 fits) | depth 8, leaf 10 (0.683) | 0.667 | 0.652 | 0.659 | [[114, 29], [31, 58]] |

**Selected model: random forest** (test F1 0.659, above the 0.55 requirement). Missing a lapsing customer loses their future orders (an average customer spends about INR 2,552 in six months), while wrongly targeting a loyal customer costs one coupon,
so recall matters more than precision. Both tuned models catch the same 58 of 89 lapsing customers, but the forest wrongly targets 29 loyal customers instead of 32 and has the higher cross-validated and test F1, so it is the better choice. (My logistic regression picked C = 100 where the reference run picked C = 1.)

**Class balance.** Oversampling the lapsed training rows from 354 to 570 (570 non-lapsed) and refitting the logistic regression moves recall from 0.652 to 0.854 while precision falls from 0.644 to 0.589 (F1 0.648 -> 0.697; confusion [[90, 53], [13, 76]]).
Recall moved most: the retention team would reach far more customers who are about to leave, at the price of contacting more who would have stayed.

**Interpretation.** Top-5 forest importances: frequency 0.281, monetary 0.270, recency 0.114, avg_delivery_minutes 0.062, cancel_rate 0.053 - how much and how recently someone buys predicts lapse far better than anything else.
Logistic coefficients (per standard deviation): frequency -1.698 (frequent buyers rarely lapse), recency +0.039 (a longer gap since the last order slightly raises risk), avg_delivery_minutes +0.079 (slower deliveries slightly raise risk).
The selected pipeline is saved to `outputs/lapse_model.joblib`; reloading it reproduces the original test predictions exactly (checked in code).
`outputs/retention_list.csv`: 40 test-set customers are Premium; the 10 with the highest predicted lapse probability are listed and 2 of them truly lapsed.
The Premium flag comes from whole-window RFM that includes the September window defining the label, so it is a post-hoc label: the retention list is illustrative, not a leakage-free evaluation.

## Part C - Support ticket triage with guardrails

Tickets are processed in file order. **Input guardrails:** 4 empty tickets get `skipped_empty` and 4 tickets estimated at 734 tokens (> 400) get `flagged_long`; neither is ever sent to the model.
**Redaction** (exact patterns, in order email -> phone -> order id): 55 emails, 91 phone numbers, 132 order ids = 278 substitutions across all 164 tickets; re-applying the patterns to the redacted texts finds 0.
The user message is exactly the redacted ticket text, and an assertion in the code checks that no prompt contains an email, phone or order id.
**Prompts:** a zero-shot and a few-shot system prompt (both contain `CLASSIFY`, the four categories and a JSON-only reply format; only the few-shot one has `EXAMPLES:` with four labelled examples including fraud).
**Resilience:** each call retries up to 3 times with wait `0.05 x 2**n x uniform(0.9, 1.15)` seconds and one log line per retry (`outputs/triage_retry_log.txt`); `mock_llm.reset()` runs before every full run.
A full run over the 156 eligible tickets hits exactly 19 `TransientLLMError`s, all recovered with no permanent error; every retry is retry number 0, with waits between 0.0453 and 0.0562 s.
**Validation:** JSON with an allowed category -> `classified_json`; otherwise `re.findall(r"[a-z]+", text.lower())` and accept if exactly one category word appears -> `classified_keyword`; anything else -> `flagged_invalid_output` (never guessed).

| Full run, 156 eligible tickets | Zero-shot | Few-shot |
|---|---|---|
| classified_json / classified_keyword / flagged_invalid_output | 135 / 12 / 9 | 135 / 12 / 9 |
| tickets classified | 147 | 147 |
| accuracy over classified tickets | 85.71% (126 of 147) | 100% (147 of 147) |
| total tokens / tokens per classified ticket | 11,584 / 78.8 | 23,596 / 160.5 |
| system prompt (estimated tokens) | 46 | 123 |

Zero-shot confusion (rows true, columns predicted): billing 38 billing; delivery 36 delivery; **fraud 23 fraud, 11 billing**; **technical 29 technical, 10 billing**; no other errors.
Every error is a fraud or technical ticket that mentions charges or payments and was read as billing. The few-shot prompt costs about 82 more tokens per ticket (12,012 extra tokens in the run, about 572 for each of the 21 extra correct answers).
That is clearly worth paying for fraud, where a missed ticket is a security problem and each mistake costs far more than the few extra tokens; for low-stakes categories zero-shot would be adequate.

**Budget demo** (zero-shot, `TOKEN_BUDGET = 6000`, `outputs/triage_budget_demo.csv`): the run stopped calling the model after 6,029 tokens, 75 tickets ended `budget_exhausted` (71 json + 6 keyword + 4 invalid classified or flagged before that point).
**Final run** (few-shot, `TOKEN_BUDGET = 100000`, 23,596 tokens, `outputs/triage_results.csv`, 164 rows): routing reads `outputs/customer_segments.csv`.
Fraud -> `security_team` HIGH (34 tickets); premium customers -> `premium_desk` HIGH (11); other classified tickets -> `support_queue` NORMAL (102); `human_review` NORMAL (17 = 4 empty + 4 long + 9 invalid output). Customers missing from the segment file count as not premium (0 such tickets).
**Rule-validated vs human:** 147 of 164 tickets were classified and validated by rules (135 JSON, 12 keyword); the other 17 went to humans, and none were guessed.

## Part D - Policy RAG assistant and refund agent

**Chunking:** 40-word chunks with 10-word overlap give 15 chunks (refund 4, delivery 3, escalation 3, payment 3, privacy 2). **Retrieval:** `TfidfVectorizer()` plus cosine similarity; for "refund late delivery" the top three chunks all come from `refund_policy.txt`.
TF measures how often a word occurs in a chunk and IDF down-weights words that appear in many chunks, so rare, distinctive words dominate the score; a limitation is that two phrasings with the same meaning but different words ("money back" vs "refund") score poorly.

| `policy_eval.csv` | Overlap 10 | Overlap 0 (13 chunks) |
|---|---|---|
| answerable: expected_doc in top 3 | 12/12 | 12/12 |
| answerable: answer contains expected_fact | 12/12 | 10/12 |
| unanswerable: answer begins "I could not find" | 2/2 | 2/2 |

With zero overlap two questions break: "What delivery time makes a completed order eligible for a full refund?" (the fact "more than 45 minutes" sits across a chunk boundary, so the model got only the second half of the sentence) and "How soon is money returned when a payment fails but is deducted?" (the "48 hours" part fell into the next chunk).

**Tools** (`TOOLS_SPEC`, `TOOL_MAP`): `get_order` (reads `data/instamart.db`), `search_policy`, `calculate_refund` (100% of the amount if delivered in more than 45 minutes, else 0), `issue_refund` and `escalate_to_human` (append a timestamped JSON line to `outputs/refund_ledger.jsonl`).
The ledger is idempotent: asking again for an order that was already refunded or escalated returns the original amount (`replayed: true`) without a second ledger line, so no order is paid twice.
**ReAct loop** (`run_agent`, `max_iterations=6`, `token_limit=6000`, `time_limit_seconds=10`): one JSON line per run in `outputs/agent_runs.jsonl`; checkpoints in `state/`.

| Scenario | Status | Refund (INR) | Model calls | Tokens used |
|---|---|---|---|---|
| S1 | REFUND | 128.25 | 5 | 1,127 |
| S2 | REFUND | 135.00 | 5 | 1,127 |
| S3 | ESCALATE | 1,999.00 | 5 | 1,114 |
| S4 | DENIED | 0.00 | 4 | 789 |
| S5 | DENIED | 0.00 | 3 | 475 |
| S6 | NOT_FOUND | 0.00 | 2 | 164 |
| S7 | FALLBACK_MAX_ITERATIONS | 0.00 | 6 | 3,276 |

All seven match the expected outcome and refund amount; S7 (the model keeps calling `search_policy`) finishes all six model calls under the default token limit and falls back.
**Safe stops, on S1:** `token_limit=200` -> `FALLBACK_TOKEN_LIMIT` (stopped after the third model call, 492 tokens, with only `get_order` and `search_policy` executed; the tool requested by the third call was not run);
`time_limit_seconds=0` -> `FALLBACK_TIMEOUT` (0 model calls, 0 tools). Each fallback returns the completed steps and a human-handoff summary in `final_answer`.
**Crash and resume:** after each tool observation a checkpoint `state/<run_id>.json` is written. A simulated crash right after the second tool (checkpoint saved), then `resume_agent("S1-crash-demo")`: tools executed in order across crash and resume =
`get_order`, `search_policy`, `calculate_refund`, `issue_refund` (four executions, none repeated), final status REFUND with INR 128.25, 5 model calls in total. Because scenario S1 already refunded this order, the ledger replay rule meant the demo did not pay it a second time.

**ReAct versus plan-and-execute.** ReAct took 5 model calls and 1,127 tokens for S1, 3 calls and 475 tokens for S5 and 2 calls and 164 tokens for S6; it re-sends the growing message history on every call.
A plan-and-execute design would ask the model for the whole step list once and then reply once, so S1 would need about 2 calls and fewer tokens. But the plan is written before any observation: for S6 (`NOT_FOUND`) or S5 (cancelled order) the remaining steps become pointless and a replanning call is needed,
which erases most of the saving, and branches such as "refund above INR 1000 -> escalate" depend on the order amount that only an observation reveals. For this short, branching refund workflow I would choose ReAct: the extra cost is about 1,000 tokens at worst and every step is checked against real data before the next one.

## Limitations and Next Steps

- The Premium flag is a post-hoc whole-window label (its RFM includes the September window that defines `lapsed`), so the retention list is illustrative and not a leakage-free evaluation; the model's own features use only orders before the cutoff. With only 40 Premium test customers the list is also tiny (2 of 10 truly lapsed).
- 774 completed orders have estimated (category-median) delivery times, which would never exceed 45 minutes; a real refund workflow should treat those orders as "delivery time unknown" and escalate. `delivery_imputed` marks them, but `get_order` follows the specified four-field output and does not expose it.
- `run_agent` receives only the message text, so it cannot check that the order belongs to the customer asking, and the 7-day refund window is assumed (requests carry no date). Both need real identity and timestamp data in production.
- Ages filled with city medians include 21 values ending in .5; the synthetic data has no real age signal, and a fixed grid of tuned models was used, so the lapse F1 (about 0.65) is modest. My logistic regression chose C = 100 rather than 1, which does not change the conclusions.
- The mock LLM is keyword-based, so classification and policy answers show the engineering pattern (guardrails, retries, validation, budgets) rather than real-model quality; zero-shot vs few-shot differences reflect its rules.
- Next steps: add real timestamps and identity to the agent; compare TF-IDF retrieval with embeddings; validate the lapse model on a rolling window; measure retention-campaign uplift; and review the 17 human-review tickets to learn which validation failures recur.
# instamart-copilot
