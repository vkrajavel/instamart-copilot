"""Shared paths, seeds and constants for the Copilot."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
POLICIES = ROOT / "data" / "policies"
CLEAN = ROOT / "data" / "clean"
DB_PATH = ROOT / "data" / "instamart.db"
STATE = ROOT / "state"
OUT = ROOT / "outputs"
SQL_FILE = ROOT / "sql" / "queries.sql"

SEED = 42
CITIES = ["Chennai", "Mumbai", "Bengaluru", "Delhi", "Hyderabad", "Pune"]
WINDOW_START = pd.Timestamp("2025-04-01")
SNAPSHOT = pd.Timestamp("2025-10-01")      # "today" for recency (day after last order date)
LAPSE_CUTOFF = pd.Timestamp("2025-08-01")  # features use orders before this; label looks after it

for d in (OUT, CLEAN, STATE):
    d.mkdir(parents=True, exist_ok=True)
