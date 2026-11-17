"""Step 3: load clean data into SQLite and run the queries in sql/analysis_queries.sql"""
import sqlite3
from pathlib import Path

import pandas as pd

DB = Path("data/food_delivery.db")
df = pd.read_csv("data/processed/clean.csv")

with sqlite3.connect(DB) as con:
    df.to_sql("orders", con, if_exists="replace", index=False)
    text = Path("sql/analysis_queries.sql").read_text(encoding="utf-8")
    for block in [b.strip() for b in text.split(";") if b.strip()]:
        title = block.splitlines()[0].lstrip("- ").strip()
        print(f"\n=== {title} ===")
        print(pd.read_sql_query(block, con).to_string(index=False))