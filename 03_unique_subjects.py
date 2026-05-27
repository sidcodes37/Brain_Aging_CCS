import pandas as pd

INPUT_CSV = "/serverdata/ccshome/sid/Final Pipeline/final_outputs/02_valid_files.csv"
OUTPUT_CSV = "/serverdata/ccshome/sid/Final Pipeline/final_outputs/03_unique_subjects.csv"

df = pd.read_csv(INPUT_CSV)

def extract_subject_id(filepath):
    split = filepath.split('/')
    return split[7]

# df['subject_id'] = extract_subject_id(df['filepath'])
df['subject_id'] = df['filepath'].apply(extract_subject_id)

result = df.groupby(['subject_id','age']).size().reset_index(name = 'count')

result.to_csv(OUTPUT_CSV, index = False)

print(f"Done. Output saved to {OUTPUT_CSV}")