import re
import json
from collections import Counter, defaultdict

"""
Stream-parse headers.txt and produce JSON summary.

- fs_all                : { '<full_filepath>': <hdr_sample_frequency or None>, ... }
- FS_NOT_SAME           : int (files with per-Block6 EEG/ECG per-channel fs mismatch)
- FS_NOT_SAME_LIST      : [ '<full_filepath>', ... ]
- electrodes_all        : { '<LABEL>': global_count, ... }  (labels from Block 6 only; preserved as-is except uppercased/trimmed)
- ELECTRODES_NOT_UNIQUE : int (files with duplicate labels according to chan_labels block)
- ELECTRODES_NOT_UNIQUE_LIST : [ '<full_filepath>', ... ]

Notes:
- Labels used for electrodes_all are taken only from Block 6 channel parentheticals.
- Only Block 6 channels whose parenthetical contains EEG/ECG/EKG (case-insensitive)
  and that do not match the EXCLUDE_KEYWORDS are counted.
- Leading "EEG"/"ECG"/"EKG" and trailing "-REF" are retained in the label keys.
"""

INPUT_PATH = # Input the headers.txt file here. Enter the location of the file.
OUT_PATH = # Output is a json file. Enter the desired location here.

# regexes
re_file_start = re.compile(r'^\s*\d+:\s*(\S.*)$')
re_block_start = re.compile(r'^\s*Block\s+(\d+)\s*:', re.IGNORECASE)
re_hdr_fs = re.compile(r'^\s*hdr_sample_frequency\s*=\s*([0-9]+(?:\.[0-9]+)?)', re.IGNORECASE)
re_channel_fs_with_label = re.compile(r'^\s*channel\[\s*\d+\]\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*Hz\s*\((.*?)\)', re.IGNORECASE)
re_channel_fs = re.compile(r'^\s*channel\[\s*\d+\]\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*Hz', re.IGNORECASE)
re_chan_labels_start = re.compile(r'^\s*chan_labels\s*\(\s*\d+\s*\)\s*=\s*(.*)', re.IGNORECASE)
re_bracket = re.compile(r'\[([^\]]+)\]')
re_chan_trans_type_start = re.compile(r'^\s*chan_trans_type', re.IGNORECASE)

# outputs / counters
fs_all = {}                                      # filepath -> hdr_sample_frequency (float) or None
per_file_block6_channel_fs = defaultdict(list)   # filepath -> list of block6 EEG/ECG per-channel fs floats
per_file_labels = {}                             # used for ELECTRODES_NOT_UNIQUE check (labels preserved)
electrodes_all = Counter()                       # global counts (only FROM Block 6 and only EEG/ECG/EKG channels)

FS_NOT_SAME = 0
FS_NOT_SAME_LIST = []

ELECTRODES_NOT_UNIQUE = 0
ELECTRODES_NOT_UNIQUE_LIST = []

# helpers
def normalize_label_preserve(raw: str) -> str:
    """
    Preserve label content but trim and uppercase.
    Do NOT remove leading EEG/ECG/EKG or trailing -REF per user instruction.
    """
    return raw.strip().upper()

def floats_all_equal(lst, tol=1e-6):
    if not lst:
        return True
    first = lst[0]
    for x in lst:
        if abs(x - first) > tol:
            return False
    return True

# Exclude tokens that should not be counted as electrodes even if in Block 6
EXCLUDE_KEYWORDS = [
    r'PULSE', r'PULSE\s+RATE', r'IBI', r'BURST', r'BURSTS',
    r'SUPPR', r'SUPPRESSION', r'RESP', r'PHOTIC', r'DC', r'LOC'
]
re_exclude = re.compile(r'(?i)\b(?:' + '|'.join(k for k in EXCLUDE_KEYWORDS) + r')\b')

# streaming parse state
current_fp = None
in_block6 = False
collecting_chan_labels = False
pending_chan_labels = []

try:
    with open(INPUT_PATH, 'r', encoding='utf-8', errors='replace') as fh:
        for raw in fh:
            line = raw.rstrip('\n')

            # detect new file
            mfile = re_file_start.match(line)
            if mfile:
                # finalize pending chan_labels for previous file
                if current_fp is not None:
                    if pending_chan_labels:
                        processed = [normalize_label_preserve(t) for t in pending_chan_labels]
                        per_file_labels[current_fp] = processed
                        c = Counter(processed)
                        if any(v > 1 for v in c.values()):
                            ELECTRODES_NOT_UNIQUE += 1
                            ELECTRODES_NOT_UNIQUE_LIST.append(current_fp)
                        pending_chan_labels = []
                    else:
                        per_file_labels.setdefault(current_fp, [])
                # start new file block
                current_fp = mfile.group(1).strip()
                fs_all.setdefault(current_fp, None)
                per_file_block6_channel_fs[current_fp] = []
                in_block6 = False
                collecting_chan_labels = False
                pending_chan_labels = []
                continue

            if current_fp is None:
                continue

            # detect block starts; update in_block6 on block number
            mb = re_block_start.match(line)
            if mb:
                block_num = int(mb.group(1))
                in_block6 = (block_num == 6)

            # also mark block6 if line contains 'derived values' or 'per channel sample frequencies'
            low = line.lower()
            if 'derived values' in low or 'per channel sample frequencies' in low:
                in_block6 = True

            # hdr_sample_frequency anywhere -> fs_all
            mh = re_hdr_fs.match(line)
            if mh:
                try:
                    fs_val = float(mh.group(1))
                except Exception:
                    fs_val = None
                fs_all[current_fp] = fs_val
                continue

            # channel lines with labels: only Block6 EEG/ECG/EKG channels contribute to electrodes_all and FS_NOT_SAME
            mch_label = re_channel_fs_with_label.match(line)
            if mch_label:
                try:
                    ch_fs = float(mch_label.group(1))
                except Exception:
                    ch_fs = None
                label_text = mch_label.group(2).strip()

                # If in Block 6 and label contains EEG/ECG/EKG and not excluded, use it.
                if in_block6 and re.search(r'(?i)\b(EEG|ECG|EKG)\b', label_text) and not re_exclude.search(label_text):
                    if ch_fs is not None:
                        per_file_block6_channel_fs[current_fp].append(ch_fs)
                    norm = normalize_label_preserve(label_text)
                    # require at least one alphabetic char to avoid numeric noise
                    if norm and re.search(r'[A-Z]', norm):
                        electrodes_all[norm] += 1

                continue

            # fallback channel fs without label (ignored for Block6-specific collections)
            mch = re_channel_fs.match(line)
            if mch:
                continue

            # collect chan_labels (unchanged behavior used for ELECTRODES_NOT_UNIQUE check)
            mcl = re_chan_labels_start.match(line)
            if mcl:
                collecting_chan_labels = True
                rest = mcl.group(1)
                found = re_bracket.findall(rest)
                if found:
                    pending_chan_labels.extend(found)
                continue

            if collecting_chan_labels:
                if re_chan_trans_type_start.match(line):
                    collecting_chan_labels = False
                    if pending_chan_labels:
                        processed = [normalize_label_preserve(t) for t in pending_chan_labels]
                        per_file_labels[current_fp] = processed
                        c = Counter(processed)
                        if any(v > 1 for v in c.values()):
                            ELECTRODES_NOT_UNIQUE += 1
                            ELECTRODES_NOT_UNIQUE_LIST.append(current_fp)
                        pending_chan_labels = []
                    continue
                if '[' in line:
                    found = re_bracket.findall(line)
                    if found:
                        pending_chan_labels.extend(found)
                    continue
                # otherwise keep waiting

    # finalize last file's pending chan_labels if any
    if current_fp is not None and pending_chan_labels:
        processed = [normalize_label_preserve(t) for t in pending_chan_labels]
        per_file_labels[current_fp] = processed
        c = Counter(processed)
        if any(v > 1 for v in c.values()):
            ELECTRODES_NOT_UNIQUE += 1
            ELECTRODES_NOT_UNIQUE_LIST.append(current_fp)
        pending_chan_labels = []

except FileNotFoundError:
    # Input missing; proceed with whatever was gathered (likely empty).
    pass

# Compute FS_NOT_SAME using only Block6 EEG/ECG per-channel fs
for fp, ch_list in per_file_block6_channel_fs.items():
    if not ch_list:
        # no Block6 EEG/ECG channel fs available -> do not flag
        continue
    if not floats_all_equal(ch_list):
        FS_NOT_SAME += 1
        FS_NOT_SAME_LIST.append(fp)

# prepare final dicts
electrodes_all = dict(electrodes_all)

summary = {
    "fs_all": fs_all,
    "FS_NOT_SAME": FS_NOT_SAME,
    "FS_NOT_SAME_LIST": FS_NOT_SAME_LIST,
    "electrodes_all": electrodes_all,
    "ELECTRODES_NOT_UNIQUE": ELECTRODES_NOT_UNIQUE,
    "ELECTRODES_NOT_UNIQUE_LIST": ELECTRODES_NOT_UNIQUE_LIST
}

# write JSON summary
try:
    with open(OUT_PATH, 'w', encoding='utf-8') as outf:
        json.dump(summary, outf, indent=2, sort_keys=True)
except Exception as e:
    print("Failed to write summary file:", e)

if __name__ == "__main__":
    print("Wrote JSON summary to:", OUT_PATH)
    print("Files seen:", len(fs_all))
    print("FS_NOT_SAME:", FS_NOT_SAME)
    print("ELECTRODES_NOT_UNIQUE:", ELECTRODES_NOT_UNIQUE)
