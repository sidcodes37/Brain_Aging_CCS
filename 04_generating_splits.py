'''
Once the train, test, val split is done, run each file in this code to populate the specific files for every subject in the file. 
'''
import pandas as pd
import re

INPUT_CSV = # Input path of subject wise splits (Run the code for each file separately)
OUTPUT_CSV = # Output file path 04_final_train_data, 04_final_test_data, 04_final_val_data
SOURCE_CSV = # Input path of file 02_valid_files.csv. This is the source csv using which the code will populate the output csv. 

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
