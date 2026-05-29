"""
Preprocess EDFs, then call CCS features to compute per-epoch channel features. Averages
features per-channel and writes CSV.
Columns - 4 (ID, Age, Gender_male, Gender_female, channel_name, x,y,z) + 70 (features)
Rows - 16 rows per file (Values per channel)
"""

import os
import gc
import traceback
import mne
from joblib import Parallel, delayed
import time
import numpy as np
import pandas as pd
from utils import clean_channel_name 
from eegfeatures_fast import generate_multieegfeatures
from sklearn.model_selection import KFold
from autoreject import AutoReject, get_rejection_threshold

# -----------------------
# CONFIG SECTION (edit these)
# -----------------------
INPUT_CSV =  # Input path of file 04_final_val_data.csv
OUTPUT_CSV = # Output path for file 05_CCS_val_features.csv
THRESH_CSV = # Output path for file 05_CCS_val_thresh.csv
REJECT_CSV = # Output path for file 05_CCS_val_reject_log.csv
FAILED_LOG_CSV = # Output path for file 05_CCS_val_failed_log.csv
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
thresh_ch_names[0]=f"global_thresh/{thresh_ch_names[0]}"


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
    try:
        os.remove(lockfile)
    except FileNotFoundError:
        pass

def append_dict_to_csv(row_dict, csv_path, lock_timeout=30):
    """
    Append a single-row dict to csv_path safely (creates parent dir).
    Uses simple lockfile mechanism so concurrent processes don't clobber the file.
    """
    os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)
    lockfile = csv_path + ".lock"
    acquire_lock(lockfile, timeout=lock_timeout)
    try:
        pd.DataFrame([row_dict]).to_csv(csv_path, mode='a', header= not os.path.exists(csv_path), index=False)
    finally:
        release_lock(lockfile)

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

def ComputeCCS(i, filepath, age, gender):
    try:
        
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
        
        if len(epochs_ar) == 0:
            print(f"[REJECT] All epochs dropped for {filepath} at Level 2. Skipping file.")
            result = {
            "success": False,
            "filepath": filepath,
            'global_thresh': None,
            'thresh_channels': None,
            'features': None,
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
                    "skipped": 1,
                    "method": None, 
                    "error_reason": 'All dropped at L2'
                }
            append_dict_to_csv(reject_log, REJECT_CSV)
            return result
    
        preprocess_end_time = time.time()
        preprocess_time = preprocess_end_time - preprocess_start_time
        print(f"[TIME] Preprocessing time: {preprocess_time:.2f} seconds for {filepath}", flush=True)
        
        print(f"Calling generate_multieegfeatures() for {filepath} ...", flush = True)
       
        # Calling multieegfeatures
        feature_data = epochs_ar.get_data()
        multidf = generate_multieegfeatures(
            feature_data, sfreq, clean_ch_names,
            featurelist=['psd', 'fooof', 'irasa', 'nonlinear', 'acw'],
            psdtype='welch',
            kwargs_psd=dict(scaling='density', average='median', window="hamming", nperseg=None),
            freq_range=[1, 40], 
            bands=[(1, 4, 'Delta'), (4, 8, 'Theta'), (6, 10, 'ThetaAlpha'), 
                   (8, 12, 'Alpha'), (12, 18, 'Beta1'),(18, 30, 'Beta2'), 
                   (30, 40, 'Gamma1')]
        )
    
        exclude = {'Chan', 'epoch', 'filepath', 'age', 'gender_male', 'gender_female',
           'x', 'y', 'z', 'epochs_avg'}
        feature_cols = [c for c in multidf.columns if c not in exclude]
        # Average across epochs per channel
        # multidf has rows = n_epoch * n_chan and has column 'Chan' and numeric feature columns
        # we group by 'Chan' and take mean (averaging across epochs)
        grouped_mean = multidf.groupby('Chan')[feature_cols].mean().reindex(clean_ch_names)  # index will be Chan, columns are numeric features
        grouped_std = multidf.groupby('Chan')[feature_cols].std().reindex(clean_ch_names)
    
        grouped_mean = grouped_mean.add_suffix('_mean')
        grouped_std = grouped_std.add_suffix('_std')
        grouped = pd.concat([grouped_mean, grouped_std], axis=1)
        
        cols = []
        for feat in feature_cols:
            cols.append(feat + '_mean')
            cols.append(feat + '_std')
    
        grouped = grouped.reindex(columns=cols)
        per_channel = grouped.reset_index()
    
        # Grouped per-channel features into a single row in the order of clean_ch_names
        per_channel['filepath'] = filepath
        per_channel['age'] = age
        per_channel['gender_male'] = 1 if (str(gender).lower() == 'male') else 0
        per_channel['gender_female'] = 1 if (str(gender).lower() == 'female') else 0
        
        # number of epochs averaged
        per_channel['epochs_avg'] = n_epochs
    
        #coords 
        per_channel = per_channel.merge(coords_df, on='Chan', how='left')
    
        cols_order = ['filepath','age','gender_male','gender_female','Chan','x','y','z','epochs_avg'] + cols
        per_channel = per_channel[cols_order]
    
        print(f"[MAIN] Features for file {i} done (epochs={n_epochs}). for {filepath}", flush=True)

        file_end_time = time.time()
        total_time = file_end_time - file_start_time
        print(f"[TIME] Total time for file {filepath}: {total_time:.2f} seconds", flush=True)
        
        # Immediate writing
        append_rows_to_csv(per_channel.to_dict(orient='records'), OUTPUT_CSV)
    
        if global_thresh is not None:
                rec = {}
                rec['filepath'] = filepath
                rec['type'] = 'global_eeg'
                rec[thresh_ch_names[0]] = global_thresh['eeg']
                for ch in thresh_ch_names[1:]:
                    rec[ch] = np.nan
                append_dict_to_csv(rec, THRESH_CSV)
        if thresh_channels is not None:
                # thresh_channels may be a dict-like of per-channel thresholds
                rec = {}
                rec['filepath'] = filepath
                rec['type'] = 'per_channel'
                for orig_ch, out_ch in zip(clean_ch_names, thresh_ch_names):
                    rec[out_ch] = thresh_channels.get(orig_ch, np.nan)
                append_dict_to_csv(rec, THRESH_CSV)
    
        # Reject log (if present from AutoReject branch)
        if reject_log is not None:
            append_dict_to_csv(reject_log, REJECT_CSV)
    
        print(f"[MAIN] Features and logs appended to CSVs for {filepath}", flush=True)
    
        file_end_time = time.time()
        total_time = file_end_time - file_start_time
        return {
            'success': True,
            'filepath': filepath,
            'preprocess_time': preprocess_time,
            'total_time': total_time
        }
    
    except Exception as e:
        print(f"\n{'='*80}", flush=True)
        print(f"[ERROR] File {i}/{total_files}: {filepath}", flush=True)
        print(f"[ERROR] Exception: {type(e).__name__}: {str(e)}", flush=True)
        print("[ERROR] Full traceback:", flush=True)
        print(traceback.format_exc(), flush=True)
        print(f"{'='*80}\n", flush=True)
        
        return {
            'success': False,
            'error': True,
            'filepath': filepath,
            'error_type': type(e).__name__,
            'error_message': str(e),
            'error_traceback': traceback.format_exc()
        }
    

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
    for i, (_, row) in enumerate(filelist.iterrows(), start=1):
        jobs.append((i, row['filepath'], row.get('age'), row.get('gender')))
    
    
    results = Parallel(n_jobs=N_JOBS, backend = 'loky', verbose=5)(
    delayed(ComputeCCS)(i, filepath, age, gender) for (i, filepath, age, gender) in jobs
    )
    gc.collect()    

    # Summary
    success_count = sum(1 for r in results if r.get('success'))
    failed_files = [r for r in results if r.get('error')]
    failed_count = len(failed_files)
    
    # Saving failed files log
    error_df = pd.DataFrame(failed_files)
    error_df.to_csv(FAILED_LOG_CSV, index=False)
    print(f"\n[ERROR SUMMARY] {failed_count} files failed. Details saved to: {FAILED_LOG_CSV}", flush=True)
    print("[ERROR SUMMARY] Failed files:", flush=True)
    for f in failed_files:
        print(f"  - {f['filepath']}: {f.get('error_type', 'Unknown')}", flush=True)
        
    print(f"[MAIN] Done. Successes: {success_count}, Failures: {failed_count}", flush=True)
    print(f"[MAIN] Feature CSV: {os.path.abspath(OUTPUT_CSV)}", flush=True)
    print(f"[MAIN] Threshold CSV: {os.path.abspath(THRESH_CSV)}", flush=True)
    print(f"[MAIN] Reject log CSV: {os.path.abspath(REJECT_CSV)}", flush=True)
