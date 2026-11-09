"""Part D - policy RAG assistant and ReAct refund agent (all model calls go through the supplied mock_llm.py).

Run:  python -m copilot.part_d        (needs Part A's data/instamart.db and generate_data.py's data/policies)
"""
import json
import sqlite3
import time
from datetime import datetime, timezone

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import mock_llm

from .config import RAW, POLICIES, OUT, STATE, DB_PATH

LEDGER = OUT / "refund_ledger.jsonl"
LATE_MINUTES = 45
AUTO_REFUND_LIMIT_INR = 1000


# ================================================================ tasks 1-2: chunking and retrieval
def chunk_documents(size=40, overlap=10):
    """40-word windows, 10-word overlap (each chunk starts 30 words after the previous one)."""
    chunks, step = [], size - overlap
    for f in sorted(POLICIES.glob("*.txt")):
        words = f.read_text(encoding="utf-8").split()
        start = 0
        while True:
            end = start + size
            chunks.append({"source": f.name, "text": " ".join(words[start:end])})
            if end >= len(words):                       # first chunk whose end reaches the last word
                break
            start += step
    return chunks


class Index:
    def __init__(self, chunks):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform([c["text"] for c in chunks])

    def retrieve(self, query, k=3):
        sims = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        top = sims.argsort()[::-1][:k]
        return [{"source": self.chunks[i]["source"], "score": round(float(sims[i]), 4), "text": self.chunks[i]["text"]} for i in top]


_index = None


def default_index():
    global _index
    if _index is None:
        _index = Index(chunk_documents())
    return _index


def retrieve(query, k=3, index=None):
    return (index or default_index()).retrieve(query, k)


# ================================================================ task 3: augment and answer
SYSTEM_POLICY = ("You are a POLICY ASSISTANT for a grocery delivery company. Answer only from the supplied context. "
                 "If the context does not contain the answer, say that you could not find it.")


def answer(question, index=None):
    hits = retrieve(question, 3, index)
    context = "\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)
    user = f"CONTEXT:\n{context}\nQUESTION:\n{question}"
    resp = mock_llm.chat([{"role": "system", "content": SYSTEM_POLICY}, {"role": "user", "content": user}])
    return {"answer": resp.choices[0].message.content, "sources": [h["source"] for h in hits], "tokens": resp.usage.total_tokens}


# ================================================================ task 4: evaluation
def evaluate(index):
    ev = pd.read_csv(RAW / "policy_eval.csv", keep_default_na=False)
    rows = []
    for _, r in ev.iterrows():
        res = answer(r["question"], index)
        if r["expected_doc"] == "NONE":
            rows.append({"question": r["question"], "kind": "unanswerable", "fallback": res["answer"].startswith("I could not find"), "answer": res["answer"]})
        else:
            rows.append({"question": r["question"], "kind": "answerable", "doc_in_top3": r["expected_doc"] in res["sources"],
                         "fact_in_answer": r["expected_fact"] in res["answer"], "expected_fact": r["expected_fact"], "answer": res["answer"]})
    df = pd.DataFrame(rows)
    a, u = df[df["kind"] == "answerable"], df[df["kind"] == "unanswerable"]
    return {"chunks": len(index.chunks),
            "chunks_per_source": pd.Series([c["source"] for c in index.chunks]).value_counts().sort_index().to_dict(),
            "answerable": len(a), "doc_in_top3": int(a["doc_in_top3"].sum()), "fact_in_answer": int(a["fact_in_answer"].sum()),
            "unanswerable": len(u), "fallback_answers": int(u["fallback"].sum()),
            "failed_questions": a.loc[~a["fact_in_answer"].astype(bool), ["question", "expected_fact", "answer"]].to_dict(orient="records")}


# ================================================================ task 5: tools and tool map
def _db():
    return sqlite3.connect(DB_PATH)


def get_order(order_id):
    with _db() as con:
        row = con.execute("SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ?", (order_id,)).fetchone()
    if row is None:
        return {"error": "order not found"}
    return {"order_id": row[0], "status": row[1], "order_amount": float(row[2]),
            "delivery_minutes": None if row[3] is None or pd.isna(row[3]) else int(row[3])}


def search_policy(query):
    return {"results": retrieve(query)}


def calculate_refund(order_amount, delivery_minutes):
    """Late (> 45 min) completed orders get 100% of order_amount; otherwise 0. (The agent checks status first.)"""
    late = delivery_minutes is not None and delivery_minutes > LATE_MINUTES
    return {"refund_inr": round(float(order_amount), 2) if late else 0.0}


def _ledger_write(tool, order_id, refund_inr):
    """Append one JSON line. Idempotent: the same order is never paid or escalated twice (a replay returns the
    original amount, flagged, without adding a second ledger line)."""
    LEDGER.touch(exist_ok=True)
    for line in LEDGER.read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        if entry["tool"] == tool and entry["order_id"] == order_id:
            return {"refund_inr": entry["refund_inr"], "replayed": True}
    entry = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), "tool": tool,
             "order_id": order_id, "refund_inr": float(refund_inr)}
    with LEDGER.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return {"refund_inr": float(refund_inr)}


def issue_refund(order_id, refund_inr):
    return _ledger_write("issue_refund", order_id, refund_inr)


def escalate_to_human(order_id, refund_inr):
    return _ledger_write("escalate_to_human", order_id, refund_inr)


def _spec(name, description, props, required):
    return {"name": name, "description": description,
            "parameters": {"type": "object", "properties": props, "required": required}}


TOOLS_SPEC = [
    _spec("get_order", "Look up an order in the database by its order id.", {"order_id": {"type": "string"}}, ["order_id"]),
    _spec("search_policy", "Search the company policy documents and return the three most relevant passages.", {"query": {"type": "string"}}, ["query"]),
    _spec("calculate_refund", "Compute the refund in INR from the order amount and the delivery time in minutes.",
          {"order_amount": {"type": "number"}, "delivery_minutes": {"type": "integer"}}, ["order_amount", "delivery_minutes"]),
    _spec("issue_refund", "Issue an automatic refund (only for refunds up to INR 1000) and record it in the ledger.",
          {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}, ["order_id", "refund_inr"]),
    _spec("escalate_to_human", "Escalate a refund above INR 1000 to a human support lead and record it in the ledger.",
          {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}, ["order_id", "refund_inr"]),
]
TOOL_MAP = {"get_order": get_order, "search_policy": search_policy, "calculate_refund": calculate_refund,
            "issue_refund": issue_refund, "escalate_to_human": escalate_to_human}


# ================================================================ tasks 6-8: ReAct loop, safe stops, checkpoints
AGENT_SYSTEM = ("You are a refund AGENT for a grocery delivery company. Use the tools to look up the order, check the "
                "refund policy, calculate the refund, then issue it or escalate it to a human. Finish with one line "
                "starting with REFUND:, ESCALATE:, DENIED: or NOT_FOUND:.")


class SimulatedCrash(Exception):
    pass


def _save_checkpoint(run_id, scenario_id, user_message, messages, tokens_used, llm_calls, steps, executed):
    STATE.mkdir(parents=True, exist_ok=True)
    cp = {"run_id": run_id, "scenario_id": scenario_id, "user_message": user_message, "messages": messages,
          "tokens_used": tokens_used, "llm_calls": llm_calls, "step": len(executed), "tools_executed": executed, "steps": steps}
    (STATE / f"{run_id}.json").write_text(json.dumps(cp, indent=1), encoding="utf-8")


def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10,
              run_id=None, crash_after_tools=None, _state=None):
    run_id = run_id or scenario_id
    started = time.monotonic()
    if _state:
        messages, tokens_used, llm_calls = _state["messages"], _state["tokens_used"], _state["llm_calls"]
        steps, executed = _state["steps"], _state["tools_executed"]
    else:
        messages = [{"role": "system", "content": AGENT_SYSTEM}, {"role": "user", "content": user_message}]
        tokens_used, llm_calls, steps, executed = 0, 0, [], []

    def refund_amount():
        for s in reversed(steps):
            if s["tool"] in ("issue_refund", "escalate_to_human") and "refund_inr" in s["observation"]:
                return float(s["observation"]["refund_inr"])
        return 0.0

    def finish(status, final_answer):
        return {"scenario_id": scenario_id, "run_id": run_id, "status": status, "final_answer": final_answer, "steps": steps,
                "llm_calls": llm_calls, "tokens_used": tokens_used, "refund_inr": refund_amount() if status in ("REFUND", "ESCALATE") else 0.0}

    def fallback(status):
        done = ", ".join(s["tool"] for s in steps) or "none"
        return finish(status, f"{status}: automated handling stopped after {len(steps)} step(s) ({done}). "
                              f"Human agent please review request: {user_message}")

    while True:
        if time.monotonic() - started >= time_limit_seconds:
            return fallback("FALLBACK_TIMEOUT")
        if llm_calls >= max_iterations:
            return fallback("FALLBACK_MAX_ITERATIONS")
        resp = mock_llm.chat(messages, TOOLS_SPEC)
        llm_calls += 1
        tokens_used += resp.usage.total_tokens
        if tokens_used > token_limit:                       # stop BEFORE executing the tool just requested
            return fallback("FALLBACK_TOKEN_LIMIT")
        msg = resp.choices[0].message
        if not msg.tool_calls:
            text = (msg.content or "").strip()
            return finish(text.split(":")[0].strip().upper(), text)
        messages.append({"role": "assistant", "content": None,
                         "tool_calls": [{"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in msg.tool_calls]})
        for tc in msg.tool_calls:
            fn = TOOL_MAP.get(tc.name)
            try:
                obs = fn(**tc.arguments) if fn else {"error": f"unknown tool {tc.name}"}
            except TypeError as exc:
                obs = {"error": f"bad arguments: {exc}"}
            messages.append({"role": "tool", "tool_call_id": tc.id, "name": tc.name, "content": json.dumps(obs)})
            steps.append({"step": len(steps) + 1, "tool": tc.name, "arguments": tc.arguments, "observation": obs})
            executed.append(tc.name)
            _save_checkpoint(run_id, scenario_id, user_message, messages, tokens_used, llm_calls, steps, executed)
            if crash_after_tools is not None and len(executed) == crash_after_tools:
                raise SimulatedCrash(f"simulated crash after tool {len(executed)} ({tc.name})")


def resume_agent(run_id, **limits):
    cp = json.loads((STATE / f"{run_id}.json").read_text(encoding="utf-8"))
    return run_agent(cp["user_message"], cp["scenario_id"], run_id=run_id, _state=cp, **limits)


def _jsonable(res):
    return json.loads(json.dumps(res, default=str))


def main():
    mock_llm.reset()
    # ---- fresh, reproducible ledger and checkpoint folder
    LEDGER.write_text("", encoding="utf-8")
    STATE.mkdir(parents=True, exist_ok=True)
    for f in STATE.glob("*.json"):
        f.unlink()
    report = {}

    # ---- tasks 1-4
    idx10 = default_index()
    report["retrieval_demo_sources"] = [h["source"] for h in retrieve("refund late delivery")]
    report["overlap_10"] = evaluate(idx10)
    report["overlap_0"] = evaluate(Index(chunk_documents(overlap=0)))

    # ---- task 6: the seven scenarios
    scenarios = pd.read_csv(RAW / "agent_scenarios.csv", keep_default_na=False)
    rows, lines = [], []
    for _, s in scenarios.iterrows():
        res = run_agent(s["message"], s["scenario_id"])
        ok = res["status"].startswith(s["expected_outcome"]) and abs(res["refund_inr"] - float(s["expected_refund_inr"])) <= 0.01
        rows.append({"scenario_id": s["scenario_id"], "expected": s["expected_outcome"], "status": res["status"], "refund_inr": res["refund_inr"],
                     "llm_calls": res["llm_calls"], "tokens_used": res["tokens_used"], "steps": len(res["steps"]), "correct": bool(ok)})
        lines.append({k: res[k] for k in ("scenario_id", "status", "final_answer", "llm_calls", "tokens_used", "steps")})
    (OUT / "agent_runs.jsonl").write_text("\n".join(json.dumps(_jsonable(l)) for l in lines) + "\n", encoding="utf-8")
    report["scenarios"] = rows

    # ---- tasks 7-8: demos on S1
    s1 = scenarios.loc[scenarios["scenario_id"] == "S1", "message"].iloc[0]
    tok = run_agent(s1, "S1", token_limit=200, run_id="S1-token-limit-demo")
    tmo = run_agent(s1, "S1", time_limit_seconds=0, run_id="S1-timeout-demo")
    try:
        run_agent(s1, "S1", run_id="S1-crash-demo", crash_after_tools=2)
        crashed = False
    except SimulatedCrash:
        crashed = True
    cp = json.loads((STATE / "S1-crash-demo.json").read_text(encoding="utf-8"))
    resumed = resume_agent("S1-crash-demo")
    report["demos"] = {
        "token_limit_200": {"status": tok["status"], "llm_calls": tok["llm_calls"], "tokens_used": tok["tokens_used"], "tools_executed": [s["tool"] for s in tok["steps"]], "final_answer": tok["final_answer"]},
        "time_limit_0": {"status": tmo["status"], "llm_calls": tmo["llm_calls"], "tools_executed": [s["tool"] for s in tmo["steps"]], "final_answer": tmo["final_answer"]},
        "crash_and_resume": {"crashed": crashed, "checkpoint_step_at_crash": cp["step"], "tools_before_crash": cp["tools_executed"],
                             "tools_executed_total": [s["tool"] for s in resumed["steps"]], "final_status": resumed["status"],
                             "refund_inr": resumed["refund_inr"], "llm_calls": resumed["llm_calls"],
                             "ledger_replayed": resumed["steps"][-1]["observation"].get("replayed", False)}}
    (OUT / "agent_demos.jsonl").write_text("\n".join(json.dumps(_jsonable(r)) for r in (tok, tmo, resumed)) + "\n", encoding="utf-8")
    report["ledger_lines"] = len([l for l in LEDGER.read_text(encoding="utf-8").splitlines() if l.strip()])
    report["state_files"] = sorted(p.name for p in STATE.glob("*.json"))

    (OUT / "part_d_report.json").write_text(json.dumps(_jsonable(report), indent=2), encoding="utf-8")
    print(json.dumps(_jsonable(report), indent=2))


if __name__ == "__main__":
    main()
