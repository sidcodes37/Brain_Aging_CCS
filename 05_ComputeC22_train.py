import os
import gc
import mne
import numpy as np
import pandas as pd
import time
from autoreject import AutoReject,get_rejection_threshold
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
from utils import compute_catch22, clean_channel_name

INPUT_CSV = "/serverdata/ccshome/sid/final_train_data_unshuffled.csv"
OUTPUT_CSV = "/serverdata/ccshome/sid/C22_train_features.csv"
THRESH_CSV = "/serverdata/ccshome/sid/C22_train_thresh_log.csv"
REJECT_CSV = "/serverdata/ccshome/sid/C22_train_reject_log.csv"

# Parallelism
N_JOBS = -1   # set >1 for parallel feature extraction (joblib)

SEED = 37

# Preprocessing params 
EPOCH_DURATION = 30.0
TARGET_SFREQ = 200.0
f_L = 0.5
f_H = 45.0
THRESHOLD_UV = 500
THRESHOLD_V = THRESHOLD_UV * 1e-6
TARGET_ELECTRODES = [
    "EEG FP1-REF", "EEG FP2-REF", "EEG F3-REF", "EEG F4-REF",
    "EEG C3-REF", "EEG C4-REF", "EEG P3-REF", "EEG P4-REF",
    "EEG O1-REF", "EEG O2-REF", "EEG F7-REF", "EEG F8-REF",
    "EEG T3-REF", "EEG T4-REF", "EEG T5-REF", "EEG T6-REF"
]
# prepare channel names cleaned (same order as TARGET_ELECTRODES)
clean_ch_names = [clean_channel_name(c) for c in TARGET_ELECTRODES]
thresh_ch_names = clean_ch_names.copy()
thresh_ch_names[0]=f"{thresh_ch_names[0]}/global_thresh"

def acquire_lock(lockfile, timeout=30.0, poll=0.05):
    start = time.time()
    while True:
        try:
            # O_CREAT | O_EXCL ensures atomic creation; will raise FileExistsError if exists
            fd = os.open(lockfile, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return
        except FileExistsError:
            if (time.time() - start) > timeout:
                raise TimeoutError(f"Timeout acquiring lock {lockfile}")
            time.sleep(poll)

def release_lock(lockfile):
    os.remove(lockfile)
    

def append_dict_to_csv(row_dict, csv_path, lock_timeout=30):
    """
    Append a single-row dict to csv_path safely (creates parent dir).
    Uses simple lockfile mechanism so concurrent processes don't clobber the file.
    """
    os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)
    lockfile = csv_path + ".lock"
    try:
        acquire_lock(lockfile, timeout=lock_timeout)
        try:
            pd.DataFrame([row_dict]).to_csv(csv_path, mode='a', header= not os.path.exists(csv_path), index=False)
        finally:
            release_lock(lockfile)
    except TimeoutError:
        print(f"[ERROR] Timeout acquiring lock for {csv_path}", flush = True)
        raise
    
def append_rows_to_csv(rows_list, csv_path, lock_timeout=30):
    if not rows_list:
        return
    os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)
    lockfile = csv_path + ".lock"
    acquire_lock(lockfile, timeout=lock_timeout)
    try:
        header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
        pd.DataFrame(rows_list).to_csv(csv_path, mode='a', header=header, index=False)
    finally:
        release_lock(lockfile)

# PER-FILE FUNCtion

filelist = pd.read_csv(INPUT_CSV)
total_files = len(filelist)

def ComputeCatch22(i, filepath, age, gender):
    file_start_time = time.time()
    preprocess_start_time = time.time()
    epoch_count = 0
    print(f"[MAIN] Preprocessing file {i}/{total_files}: {filepath}", flush=True)
    raw_edf = mne.io.read_raw_edf(filepath, preload=True, verbose=False)

    # filtering / resample
    raw_edf.filter(l_freq=f_L, h_freq=f_H, picks='all', verbose=False)
    sfreq = raw_edf.info.get('sfreq')
    if sfreq != TARGET_SFREQ:
        raw_edf.resample(TARGET_SFREQ, npad='auto')
        sfreq = TARGET_SFREQ

    # pick the target electrodes and reorder
    edf_sel = raw_edf.copy().pick(picks=TARGET_ELECTRODES, verbose=False)
    edf_sel.reorder_channels(TARGET_ELECTRODES)

   # rename channels to cleaned standard names and set montage
    rename_map = {orig: clean for orig, clean in zip(TARGET_ELECTRODES, clean_ch_names)}
    edf_sel.rename_channels(rename_map)  # now channels are e.g. 'Fp1','Fp2','F3',...

    # attach a standard 10-20 montage so channels have 3D positions
    montage = mne.channels.make_standard_montage('standard_1020')
    edf_sel.set_montage(montage, match_case=False)  # on_missing='ignore' not needed with match_case=False

    # extract coordinates
    coords = np.array([ch['loc'][:3] for ch in edf_sel.info['chs']])

    # convert to DataFrame aligned with channel order
    coords_df = pd.DataFrame(coords, columns=['x','y','z'])
    coords_df['Chan'] = edf_sel.ch_names

    # make fixed-length epochs
    raw_epochs = mne.make_fixed_length_epochs(edf_sel, duration=EPOCH_DURATION, preload=True, verbose=False)
    data = raw_epochs.get_data()  # shape: (n_epochs, n_channels, n_times)
    n_epochs, n_channels, n_times = data.shape
    epoch_count += n_epochs

    # Removing epochs with Amplitudes greater than 500 uV
    print("Level 1 of preprocessing - Removing epochs with amplitude greater than 500 uV")
    reject = {"eeg": THRESHOLD_V}
    epochs_clean = raw_epochs.drop_bad(reject=reject)
    # dropped_epochs = [i for i, log in enumerate(epochs_clean.drop_log) if len(log) > 0]
    # l1_reject_count = len(dropped_epochs)
    l1_reject_count = n_epochs - len(epochs_clean)
    print("Epochs dropped at Level 1: ", l1_reject_count)
    print(f"Level 1 preprocessing done. Rejected {l1_reject_count} epochs out of {n_epochs}")

    if len(epochs_clean) == 0:
        print(f"[REJECT] All epochs dropped for {filepath} at Level 1 (amplitude >{THRESHOLD_UV}). Skipping file.")
        result = {
        "success": False,
        "filepath": filepath,
        'global_thresh': None,
        'thresh_channels': None,
        'features': None,
        'n_epochs_original': n_epochs,
        'n_epochs_after_L1': 0,
        'n_epochs_after_L2': None,
        'preprocess_time': time.time() - preprocess_start_time,
        'total_time': time.time() - file_start_time
        }

        reject_log = {
                "filepath": filepath,
                "L1_reject": l1_reject_count,
                "L2_reject": None,
                "skipped": 1,
                "method": None, 
                "error_reason": 'All dropped at L1'
            }
        append_dict_to_csv(reject_log, REJECT_CSV)
        return result

    print("Level 2 of preprocessing...")
    # Determining CV
    if len(epochs_clean) < 4:
        print(f"[REJECT] Skipping {filepath}: Too few epochs ({len(epochs_clean)} < 4)", flush = True)
        result = {
        "success": False,
        "filepath": filepath,
        'global_thresh': None,
        'thresh_channels': None,
        'features': None,
        'n_epochs_original': n_epochs,
        'n_epochs_after_L1': len(epochs_clean),
        'n_epochs_after_L2': len(epochs_clean),
        'preprocess_time': time.time() - preprocess_start_time,
        'total_time': time.time() - file_start_time
        }

        reject_log = {
                "filepath": filepath,
                "L1_reject": l1_reject_count,
                "L2_reject": None,
                "skipped": 1,
                "method": None, 
                "error_reason": 'Number of epochs smaller than CV'
            }
        append_dict_to_csv(reject_log, REJECT_CSV)
        return result
    
    elif 4 <= len(epochs_clean) <= 10:
        print(f"[INFO] Only {len(epochs_clean)} after L1. Using get_rejection_threshold fallback (No Autoreject).", flush = True)
        cv = max(4, min(5, len(epochs_clean)))
        thres_dict = get_rejection_threshold(epochs_clean,decim=1, random_state = SEED, cv = cv)
        epochs_ar = epochs_clean.copy().drop_bad(reject=thres_dict)
        l2_reject_count = len(epochs_clean) - len(epochs_ar)
        reject_log = None
        global_thresh = thres_dict        # {'eeg': float}
        thresh_channels = None
        
        result = {
            'success':True,
            'global_thresh': thres_dict,
            'thresh_channels' : None,
            'n_epochs_original': n_epochs,
            'n_epochs_after_L1': len(epochs_clean),
            'n_epochs_after_L2': len(epochs_ar),
            'preprocess_time': time.time() - preprocess_start_time,
            'total_time': time.time() - file_start_time
            }
        reject_log = {
                "filepath": filepath,
                "L1_reject": n_epochs - len(epochs_clean),
                "L2_reject": l2_reject_count,
                "skipped": 0,
                "method": "get_rejection_threshold"
            }
        
    
    else:
        cv = max(4, min(10, len(epochs_clean)))
        cv_splitter = KFold(n_splits = cv, shuffle = True, random_state = SEED) 
        ar = AutoReject(n_interpolate=[1,2,3,4], consensus = [0.25], cv = cv_splitter, random_state = SEED, thresh_method = 'random_search', n_jobs=1, verbose=False)
        ar.fit(epochs_clean)
        epochs_ar, __ = ar.transform(epochs_clean, return_log = True)
        l1_reject_count = n_epochs - len(epochs_clean)
        l2_reject_count = len(epochs_clean) - len(epochs_ar)
        print(f"Level 2 preprocessing done! Rejected {len(epochs_clean) - len(epochs_ar)} epochs out of {len(epochs_clean)}")
        global_thresh = None
        thresh_channels = ar.threshes_

        result = {
            'success': True,
            'global_thresh': None,
            'thresh_channels': thresh_channels,
            'n_epochs_original': n_epochs,
            'n_epochs_after_L1': len(epochs_clean),
            'n_epochs_after_L2': len(epochs_ar),
            'preprocess_time': time.time() - preprocess_start_time,
            'total_time': time.time() - file_start_time
            }
        reject_log = {
                "filepath": filepath,
                "L1_reject": l1_reject_count,
                "L2_reject": l2_reject_count,
                "skipped": 0,
                "method": "autoreject"
            }
    
    preprocess_end_time = time.time()
    preprocess_time = preprocess_end_time - preprocess_start_time
    print(f"[TIME] Preprocessing time: {preprocess_time:.2f} seconds for {filepath}", flush=True)

    print(f"Computing Catch22 feature for {filepath}")

    # Catch-22
    # get epoch data shape (n_epochs, n_channels, n_times)
    data = epochs_ar.get_data()
    n_epochs, n_channels, n_times = data.shape
    epoch_count += n_epochs  # update epoch count

    ts_list = [data[e, c, :] for e in range(n_epochs) for c in range(n_channels)]
    results = Parallel(n_jobs=-1, backend="loky")(delayed(compute_catch22)(ts) for ts in ts_list)

    # shape: (n_epochs, n_channels, 22)
    catch22_all = np.asarray(results, dtype=np.float32).reshape(n_epochs, n_channels, 22)

    # Catch22 fixed feature names
    feature_names = compute_catch22(np.random.randn(2000), get_only_names=True)

    # Compute mean and std across epochs -> both (n_channels, 22)
    catch22_mean = np.nanmean(catch22_all, axis=0)  # (n_channels, 22)
    catch22_std  = np.nanstd(catch22_all,  axis=0)  # (n_channels, 22)

    # Build one row per channel
    for ch_idx, ch_name in enumerate(clean_ch_names):
        row = {
            'filepath':      filepath,
            'age':           age,
            'gender_male':   1 if gender == 'Male'   else 0,
            'gender_female': 1 if gender == 'Female' else 0,
            'Chan':          ch_name,
        }

        for feat_idx, fn in enumerate(feature_names):
            mean_val = catch22_mean[ch_idx, feat_idx]
            std_val  = catch22_std[ch_idx,  feat_idx]
            row[f"{fn}_mean"] = float(mean_val) if np.isfinite(mean_val) else np.nan
            row[f"{fn}_std"]  = float(std_val)  if np.isfinite(std_val)  else np.nan

        append_dict_to_csv(row, OUTPUT_CSV)

    print(f"catch22 features extracted from file {i} (epochs extracted: {n_epochs}).")

    if global_thresh is not None:
            rec = {}
            rec[thresh_ch_names[0]] = global_thresh['eeg']
            for ch in thresh_ch_names[1:]:
                rec[ch] = np.nan
            rec['filepath'] = filepath
            rec['type'] = 'global_eeg'
            append_dict_to_csv(rec, THRESH_CSV)
    if thresh_channels is not None:
            # thresh_channels may be a dict-like of per-channel thresholds
            # rec = dict(thresh_channels)
            rec = {}
            for orig_ch, out_ch in zip(clean_ch_names, thresh_ch_names):
                rec[out_ch] = thresh_channels.get(orig_ch, np.nan)
            rec['filepath'] = filepath
            rec['type'] = 'per_channel'
            append_dict_to_csv(rec, THRESH_CSV)

    # Reject log (if present from AutoReject branch)
    if reject_log is not None:
        append_dict_to_csv(reject_log, REJECT_CSV)

    print(f"[MAIN] Features and logs appended to CSVs for {filepath}", flush=True)
    print(f"{i}/{total_files} files processed.\n")

    file_end_time = time.time()
    total_time = file_end_time - file_start_time
    return {
        'success': True,
        'filepath': filepath,
        'preprocess_time': preprocess_time,
        'total_time': total_time
    }

    if 'raw_edf' in locals():
        del raw_edf
    if 'edf_sel' in locals():
        del edf_sel
    if 'raw_epochs' in locals():
        del raw_epochs
    if 'epochs_clean' in locals():
        del epochs_clean
    if 'epochs_ar' in locals():
        del epochs_ar

if __name__ == "__main__":
    # Checking for already processed files
    filelist = pd.read_csv(INPUT_CSV)
    total_files = len(filelist)
    
    if os.path.exists(OUTPUT_CSV):
        processed_edf = pd.read_csv(OUTPUT_CSV)
        if 'filepath' in processed_edf.columns:
            processed_files = set(processed_edf['filepath'].dropna().unique())
            
            filelist = filelist[~filelist['filepath'].isin(processed_files)].reset_index(drop=True)
            skipped_count = total_files - len(filelist)
            print(f"[MAIN] Filtered out {skipped_count} already-processed files", flush=True)
            
    if os.path.exists(REJECT_CSV):
        reject_df = pd.read_csv(REJECT_CSV)
        if 'filepath' in reject_df.columns:
            reject_files = set(reject_df['filepath'].dropna().unique())
            
            # Check for duplicates in REJECT_CSV
            total_reject_rows = len(reject_df)
            unique_reject = len(reject_files)
            if total_reject_rows != unique_reject:
                print(f"[WARNING] REJECT_CSV has {total_reject_rows - unique_reject} duplicate entries!", flush=True)
            
            processed_files.update(reject_files)
            reject_count = len(reject_files)
            print(f"[MAIN] Rejected files (REJECT_CSV): {reject_count}", flush=True)
            print(f"[MAIN] Sample reject path: {list(reject_files)[0] if reject_files else 'NONE'}", flush=True)
     
    
    if len(filelist) == 0:
        print("[MAIN] Nothing to do - all files already processed!", flush=True)
        exit(0)
    
    jobs = []
    # skipped_count = 0
    for i, (_, row) in enumerate(filelist.iterrows(), start=1):
        # filepath = row['filepath']
        # if filepath in processed_files:
        #     skipped_count += 1
        #     print(f"Skipping coz done {filepath}.")
        #     continue
        
        jobs.append((i, row['filepath'], row.get('age'), row.get('gender')))
    
    
    results = Parallel(n_jobs=N_JOBS, backend = 'loky', verbose=5)(
    delayed(ComputeCatch22)(i, filepath, age, gender) for (i, filepath, age, gender) in jobs
    )
    gc.collect()    

    # Summary
    success_count = sum(1 for r in results if r.get('success'))
    failed_count = len(results) - success_count
    print(f"[MAIN] Done. Successes: {success_count}, Failures: {failed_count}", flush=True)
    print(f"[MAIN] Feature CSV: {os.path.abspath(OUTPUT_CSV)}", flush=True)
    print(f"[MAIN] Threshold CSV: {os.path.abspath(THRESH_CSV)}", flush=True)
    print(f"[MAIN] Reject log CSV: {os.path.abspath(REJECT_CSV)}", flush=True)
