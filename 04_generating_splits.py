import pandas as pd
import re

INPUT_CSV = "/serverdata/ccshome/sid/subsampled_val.csv"
OUTPUT_CSV = "/serverdata/ccshome/sid/subsampled_val_data_pop.csv"
SOURCE_CSV = "/serverdata/ccshome/sid/Final Pipeline/final_outputs/02_valid_files.csv"

source_df = pd.read_csv(SOURCE_CSV)
df = pd.read_csv(INPUT_CSV)

def extract_subject_id(filepath):
    m = re.search(r"/\d{3}/([^/]+)/", str(filepath))
    return m.group(1)

source_df["subject_id"] = source_df["filepath"].apply(extract_subject_id)

source_df["age"]= pd.to_numeric(source_df["age"], errors="coerce").astype("Int64")
df["age"]= pd.to_numeric(df["age"], errors="coerce").astype("Int64")

match =  source_df.merge(df[["subject_id", "age"]].drop_duplicates(), 
                         on = ["subject_id", "age"],
                         how = "inner")

match = match[["filepath", "age", "gender"]]

match.to_csv(OUTPUT_CSV, index = False)

print(f"Saved {len(match)} rows to {OUTPUT_CSV}")
