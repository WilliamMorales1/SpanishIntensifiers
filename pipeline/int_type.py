# Int_type classes for output tables, looked up from Int after the model runs; the model never predicts them, so a
# different classification is a different int_type.json, not a retrained model (python pipeline/int_type.py --help)

import argparse
import json
import os
import sqlite3

import pandas as pd

from output import to_sqlite

DEFAULT_MAP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "int_type.json")


def load_map(path=None):
    """load_map reads an Int -> Int_type JSON object (default pipeline/int_type.json)."""
    with open(path or DEFAULT_MAP, encoding="utf-8") as f:
        return json.load(f)


def add_int_type(df, mapping):
    """add_int_type sets Int_type from Int, placed right after Int; an Int missing from mapping gets "?"."""
    df = df.drop(columns="Int_type", errors="ignore")
    ints = df["Int"].astype(str)
    types = ints.map(lambda i: "/" if i == "/" else mapping.get(i.lower(), "?"))
    missing = sorted(set(ints[types == "?"]))
    if missing:
        print(f"Int_type: no class for {', '.join(missing)}; set to ? (add them to the map)")
    df.insert(df.columns.get_loc("Int") + 1, "Int_type", types)
    return df


def read_table(path):
    if path.endswith(".sqlite"):
        with sqlite3.connect(path) as con:
            return pd.read_sql("SELECT * FROM intensifier_data", con)
    if path.endswith(".xlsx"):
        return pd.read_excel(path, dtype=str, keep_default_na=False)
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def write_table(df, path):
    if path.endswith(".sqlite"):
        to_sqlite(df, path)
    elif path.endswith(".xlsx"):
        df.to_excel(path, index=False)
    else:
        df.to_csv(path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="(re)fill Int_type of an output table from an Int -> class map")
    parser.add_argument("table", help="output table (.sqlite, .xlsx or .csv)")
    parser.add_argument("--map", help=f"JSON object of Int -> Int_type (default {DEFAULT_MAP})")
    parser.add_argument("--output", help="where to write (default: overwrite table)")
    args = parser.parse_args()
    df = add_int_type(read_table(args.table), load_map(args.map))
    write_table(df, args.output or args.table)
    print(
        f"Saved {args.output or args.table}: " + ", ".join(f"{t} {n}" for t, n in df["Int_type"].value_counts().items())
    )
