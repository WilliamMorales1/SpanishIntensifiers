# output tables for predict.py: Number and Syllable columns, speaker metadata, the normalized SQLite database

import os
import re
import sqlite3

import pandas as pd


def drop_invariant_hosts(df):
    """drop_invariant_hosts removes every row of an adjective lemma or adverb that is never intensified in df."""
    if df.empty:
        return df
    host = df["Lex_Adj"].where(df["Adv"] == "/", df["Adv"].str.lower())
    varies = (df["Int"] != "/").groupby(host).transform("any")
    print(f"Dropped {(~varies).sum()} rows of {host[~varies].nunique()} hosts never intensified")
    return df[varies].reset_index(drop=True)


def numero(row, count_dict=None):
    # Split the original label to get the speaker number
    if count_dict is None:
        count_dict = {}
    speaker_number = row["Speaker"]

    # Update the count for this speaker number
    count_dict[speaker_number] = count_dict.get(speaker_number, 0) + 1

    # Generate the new label
    new_label = f"{speaker_number}_{count_dict[speaker_number]:02d}"
    return new_label


# add syl col
def count_syllables(word):
    # Remove silent 'u' in gue/gui/que/qui unless it has a diaeresis (ü)
    word = re.sub(r"gue|gui|que|qui", lambda m: m.group(0).replace("u", ""), word)

    # Count vowel groups (as a simple approximation of syllables)
    syllables = re.findall(r"[aeiouáéíóúü]+", word)

    return len(syllables)


def merge_speaker_data(id_df, int_df):
    transformed_data = []

    # Iterate through each row in df
    for _, row in int_df.iterrows():
        Speaker = row["Speaker"]
        id_row = id_df[id_df["Speaker"] == Speaker]

        if not id_row.empty:
            # Convert id_row to a dict, excluding the Speaker column
            row_data = row.to_dict()
            id_data = id_row.iloc[0].to_dict()
            id_data.pop("Speaker", None)

            # Combine row data with id_data
            merged_row = {**row_data}
            for key, value in id_data.items():
                if pd.notna(value):  # Only use id_data value if it's not null/NaN
                    merged_row[key] = value

            transformed_data.append(merged_row)

    transformed_df = pd.DataFrame(transformed_data)
    print("Speaker data merged into intensifier data")
    return transformed_df


# columns with few distinct words, stored once in a shared term table
term_cols = [
    "Double",
    "Int",
    "Int_type",
    "Adv",
    "Adj",
    "Lex_Adj",
    "Adj_type",
    "Noun",
    "Negation",
    "Verb",
    "Inf_Verb",
]
phrase_cols = ["Phr_tagged", "Phr"]
token_cols = ["Number", "Syllable", "Token"]


def _quote(name):
    return '"' + str(name).replace('"', '""') + '"'


def _sql_values(df):
    # sqlite3 can't bind numpy scalars or NaN
    return [
        tuple(None if pd.isna(v) else v.item() if hasattr(v, "item") else v for v in row)
        for row in df.itertuples(index=False)
    ]


def to_sqlite(df, output_filename):
    """to_sqlite writes df as a normalized SQLite database whose intensifier_data view rebuilds df."""
    terms = [c for c in term_cols if c in df]
    phrases = [c for c in phrase_cols if c in df]
    tokens = [c for c in token_cols if c in df]
    # whatever is left is speaker metadata, which repeats on every row of a speaker
    sources = [c for c in df if c not in terms + phrases + tokens]

    source_id = df.groupby(sources, dropna=False, sort=False).ngroup() + 1
    phrase_id = df.groupby(phrases, dropna=False, sort=False).ngroup() + 1
    term_codes, term_text = pd.factorize(pd.concat([df[c] for c in terms]), use_na_sentinel=False)
    term_ids = pd.DataFrame(term_codes.reshape(len(terms), len(df)).T + 1, columns=terms, index=df.index)

    if os.path.exists(output_filename):
        os.remove(output_filename)
    con = sqlite3.connect(output_filename)

    # untyped columns so ints, floats and text all come back as they went in
    def create(table, cols, data):
        con.execute(f"CREATE TABLE {table} (id INTEGER PRIMARY KEY, {', '.join(_quote(c) for c in cols)})")
        con.executemany(
            f"INSERT INTO {table} VALUES ({', '.join('?' * (len(cols) + 1))})",
            _sql_values(data),
        )

    create(
        "source",
        sources,
        df[sources].assign(id=source_id)[["id"] + sources].drop_duplicates("id"),
    )
    create(
        "phrase",
        phrases,
        df[phrases].assign(id=phrase_id)[["id"] + phrases].drop_duplicates("id"),
    )
    create(
        "term",
        ["text"],
        pd.DataFrame({"id": range(1, len(term_text) + 1), "text": term_text}),
    )
    token = pd.concat(
        [
            pd.Series(range(1, len(df) + 1), index=df.index, name="id"),
            source_id.rename("source_id"),
            phrase_id.rename("phrase_id"),
            term_ids.add_suffix("_id"),
            df[tokens],
        ],
        axis=1,
    )
    create("token", list(token.columns[1:]), token)

    select, joins = [], []
    for c in df:
        if c in terms:
            alias = _quote("t_" + c)
            joins.append(f"JOIN term {alias} ON {alias}.id = k.{_quote(c + '_id')}")
            select.append(f"{alias}.text AS {_quote(c)}")
        else:
            table = "s" if c in sources else "p" if c in phrases else "k"
            select.append(f"{table}.{_quote(c)} AS {_quote(c)}")
    con.execute(f"""CREATE VIEW intensifier_data AS
        SELECT {", ".join(select)}
        FROM token k
        JOIN source s ON s.id = k.source_id
        JOIN phrase p ON p.id = k.phrase_id
        {" ".join(joins)}
        ORDER BY k.id""")

    # no indexes: a full scan of ~86k tokens takes milliseconds, and they cost 11% of the file
    con.commit()
    con.execute("VACUUM")
    con.close()
    print(f"Saved SQLite database to {output_filename}")


def write_outputs(df, output, speaker_path=None, xlsx=False, csv=False):
    """write_outputs adds Number and Syllable, merges speaker metadata if given, and saves output.sqlite and optional xlsx and csv."""
    number_counts = {}
    df["Number"] = df.apply(numero, axis=1, count_dict=number_counts)
    df["Syllable"] = df["Adj"].apply(count_syllables)
    if speaker_path:
        df = merge_speaker_data(pd.read_excel(speaker_path), df)

    for col in df.select_dtypes(include=["object", "string"]).columns:
        # commas would split cells in tools that re-read the csv naively
        df[col] = df[col].str.replace(",", "", regex=False)

    to_sqlite(df, output + ".sqlite")
    if xlsx:
        df.to_excel(output + ".xlsx", index=False)
        print(f"Saved {output}.xlsx")
    if csv:
        df.to_csv(output + ".csv", index=False)
        print(f"Saved {output}.csv")
    return df
