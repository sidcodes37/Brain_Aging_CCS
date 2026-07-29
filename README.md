### This is an end-to-end pipeline for computing EEG based Brain Age Index developed at the Centre for Consciousness Studies, NIMHANS, Bangalore. 

Dataset Used - TUH-EEG Corpus (Obeid and Picone, 2016) 

# What does the pipeline involve?
1) Data Engineering 
2) EEG signal Preprocessing
3) Feature Extraction & Engineering
    a) Catch22
    b) In-house built feature set (CCS Features) 
4) Machine Learning Models Training
5) Feature Importance - SHapley Additive exPlanations (SHAP)


# Dataset 
This project uses the TUH EEG Corpus (TUEG). This corpus is an archive of 14987 patients’ resting state EEG recordings collected at the Temple University Hospital. There are a total of 69670 resting state EEG recordings. 

The dataset is not an open-source dataset and thus is not included in this repository. Dataset can be requested from [here](https://isip.piconepress.com/projects/nedc/html/tuh_eeg/)

The dataset includes a readme file, a headers file and all the corresponding edf files. In our pipeline, data engineering steps were performed exclusively on the metadata contained within the headers text file, ensuring no EEG signal data were read, modified, or distorted at this stage. At each step, a CSV file was generated retaining a filepath column as a constant identifier, which was later used to call the relevant EDF files during pre-processing and feature extraction.



# Data Pre-processing 

## 1. Metadata Extraction 

The pipeline begins with a headers.txt file generated from the EDF collection. Recording metadata including file path, sampling frequency, recording duration, subject age, and sex are extracted into a structured CSV file. During this stage, recordings can optionally be filtered to retain only those containing a predefined set of sixteen common EEG electrodes.

The selected electrode configuration consists of: FP1, FP2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T3, T4, T5, T6. 
Only recordings containing all sixteen electrodes are retained to ensure a consistent spatial representation across the dataset.

## 2. Dataset Cleaning

Extracted metadata undergoes quality control before feature extraction. Recordings are retained only if they satisfy all of the following criteria:
- file exists locally
- valid subject age is available
- age lies between 18 and 90 years
- valid sex information is present
- recording duration exceeds the specified minimum threshold

Subject identifiers are then extracted from file paths, allowing recordings from the same participant to be grouped together.

## 3. Subject-Level Dataset Splitting

To prevent data leakage, training, validation, and test datasets are created at the subject level rather than the recording level.
Each subject is assigned to only one dataset split. All EEG recordings belonging to that participant are placed into the same partition, ensuring that recordings from the same individual never appear simultaneously in different datasets.

# Signal Preprocessing

Each EEG recording underwent a standardized preprocessing pipeline prior to feature extraction to ensure consistency across recordings originating from different datasets and acquisition systems.

## 1. EEG Loading and Channel Selection

Raw EEG recordings were imported from European Data Format (EDF) files using the MNE-Python library. Only the sixteen target electrodes (FP1, FP2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T3, T4, T5, and T6) were retained. Channels were subsequently reordered into a fixed sequence to ensure identical feature ordering across all recordings.

## 2. Temporal Filtering

Each recording was band-pass filtered between 0.5 Hz and 45 Hz using MNE's finite impulse response (FIR) filtering implementation. This step removed slow baseline drift while attenuating high-frequency noise outside the frequency range relevant for conventional scalp EEG analysis.

## 3. Resampling

To standardize recordings acquired at different sampling frequencies, all EEG signals were resampled to 200 Hz. This ensured a uniform temporal resolution for all subsequent preprocessing and feature extraction procedures.

## 4. Electrode Standardization

A standard International 10–20 electrode montage was assigned to every recording after channel selection. This provided standardized three-dimensional electrode coordinates for each channel and ensured consistent electrode identification across recordings from different sources.

## 5. Epoch Segmentation

Continuous EEG recordings were segmented into consecutive, non-overlapping 30-second epochs. All subsequent artefact rejection procedures and feature extraction were performed independently on each epoch.

## 6. Initial Artefact Rejection

A preliminary quality control step was performed by rejecting epochs containing excessive signal amplitudes. Any epoch in which the absolute voltage exceeded ±500 μV in any channel was discarded. Recordings with no remaining epochs following this procedure were excluded from further analysis.

## 7. Adaptive Artefact Rejection

Following amplitude-based rejection, a second stage of automated artefact rejection was applied to identify residual noisy epochs.
Recordings containing fewer than four remaining epochs were excluded because insufficient data were available for reliable threshold estimation. For recordings containing between four and ten epochs, global rejection thresholds were estimated using MNE's ```get_rejection_threshold()``` function. Recordings containing more than ten epochs were processed using the AutoReject algorithm, which estimates channel-specific rejection thresholds through cross-validation and performs channel interpolation when appropriate. Recordings with no clean epochs remaining after this stage were excluded from downstream analysis.

Rejection thresholds, preprocessing statistics, and details of excluded recordings were recorded in dedicated quality-control logs to facilitate reproducibility and subsequent inspection.

# Feature Extraction

Following preprocessing, features are computed independently for every retained epoch and channel.
Two independent feature extraction pipelines are provided.

## Catch22 Feature Set

The first pipeline computes the Catch22 feature set.
Twenty-two canonical time-series descriptors are extracted from every epoch for each of the sixteen EEG channels. This feature set is derived from hctsa feature library (Lubba et al., 2019). 

This produces one feature vector per channel consisting of:
- subject metadata
- channel identity
- Catch22 feature means
- Catch22 feature standard deviations

## CCS Feature Set

The second pipeline is an in-house feature set that extracts a broader collection of EEG biomarkers, including:
- Power spectral density (PSD)
- Fitting Oscillations & One Over F (FOOOF) spectral parameterisation
- Irregular Resampling Auto-Spectral Analysis (IRASA) oscillatory and fractal decomposition
- Non-linear signal complexity measures - Entropy, Fractal Complexity, Lempel–Ziv Complexity, Multifractal Detrended Fluctuation Analysis (DFA)
- Autocorrelation window (ACW) measures

Extracted using ccstools (Sasidharan, 2026)
Feature extraction is performed independently for each epoch before averaging across retained epochs.

The resulting feature matrix additionally includes:
- participant age
- sex
- channel identity
- electrode coordinates
- number of epochs contributing to each feature estimate

For every feature in both the feature sets, the mean and standard deviation across epochs are computed for each EEG channel. 

# Machine Learning Pipeline

Feature matrices are used to train multiple supervised regression models for brain age prediction. The implemented models include:
- Linear Regression
- Decision Tree Regression
- Random Forest Regression
- Support Vector Regression (SVR)
- Relevance Vector Regression (RVR)
- Gaussian Process Regression (GPR)

Where applicable, model hyperparameters are optimised using Optuna with five-fold cross-validation on the training set. 
After optimization, each model is trained using the complete training dataset before generating predictions for the validation and independent test sets.
Predicted brain age (BA) and Brain Age Index (BAI = BA − chronological age) are computed for every subject.

To reduce age-related prediction bias, a post hoc correction is applied using mean prediction error estimated within consecutive 10-year age bins derived from the training data. Corrected brain age estimates are subsequently evaluated using:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Coefficient of Determination (R²)

Model interpretability is assessed using SHAP, with the appropriate explainer selected according to the regression model (LinearSHAP, TreeSHAP, or KernelSHAP). SHAP values, summary plots, and ranked feature importance tables are generated for all trained models.

This pipeline produces fully reproducible brain age predictions together with comprehensive preprocessing logs, feature matrices, trained models, evaluation metrics, and model interpretation outputs suitable for benchmarking and downstream analysis.

# Requirements

All the libraries and their versions used in this repo are listed in requirements.txt. 

# Code descriptions

## ```00_get_summary_from_txt.py```

Parses a ```headers.txt``` file generated from EDF header extraction and creates a JSON summary containing each file's header sampling frequency, a global count of EEG/ECG electrode labels, and quality-control checks for inconsistent per-channel sampling frequencies and duplicate channel labels. The input is a ```headers.txt``` file, and the output is a JSON file listing sampling frequencies, electrode statistics, and the file paths of recordings that fail the consistency or uniqueness checks.

## ```01_selecting_16chans.py```

Parses a ```headers.txt``` file and extracts the filepath, age, gender, recording duration, and header sampling frequency for each EEG recording into a CSV file. When ```SELECTIVE_ELECTRODES``` is enabled, only recordings containing all specified ```TARGET_ELECTRODES``` are included in the output; otherwise, metadata for every recording is written.


## ```02_valid_age_gender_files.py```

Reads the metadata CSV generated in the previous step, converts each relative filepath to a local absolute path, and filters the dataset to retain only existing EEG files with valid age, gender, and recording duration information. The output is a cleaned CSV containing recordings with ages between 18–90 years, non-missing gender values, durations longer than the specified minimum, and verified file paths.


## ```03_unique_subjects.py```

Reads the cleaned metadata CSV and extracts a unique subject identifier from each file path to identify individual participants. The output is a CSV containing one row per unique subject–age combination, along with the number of recordings available for that subject, which can be used for subject-level train, validation, and test dataset splitting.


## ```04_generating_splits.py```

Reads a subject-level train, validation, or test split and matches it back to the cleaned metadata CSV to retrieve all EEG recordings belonging to each selected subject. The output is a dataset-specific CSV (train, validation, or test) containing the filepaths, ages, and genders of all recordings assigned to that split, ensuring subject-level separation is preserved.

## ```05_ComputeC22_train.py, 05_ComputeC22_val.py, 05_ComputeC22_test.py```

Preprocesses each EEG recording listed in the input CSV by filtering, resampling, epoching, rejecting noisy epochs using amplitude thresholding and AutoReject (or a fallback thresholding method), and extracting Catch22 time-series features for each of the 16 target electrodes. The input is a CSV containing EEG file paths and participant metadata, and the outputs are (1) a feature CSV containing the mean and standard deviation of all 22 Catch22 features for every channel, (2) a threshold log recording the preprocessing rejection thresholds used, and (3) a rejection log documenting preprocessing outcomes and any files that were skipped.

## ```05_ComputeCCS_train.py, 05_ComputeCCS_val.py, 05_ComputeCCS_test.py```

Preprocesses each EEG recording listed in the input CSV by filtering, resampling, epoching, rejecting noisy epochs using amplitude thresholding and AutoReject (or a fallback thresholding method), and extracting a comprehensive set of EEG features including power spectral density (PSD), FOOOF, IRASA, nonlinear, and autocorrelation window (ACW) features for each of the 16 target electrodes. The input is a CSV containing EEG file paths and participant metadata, and the outputs are (1) a feature CSV containing the mean and standard deviation of all extracted features for every channel, (2) a threshold log recording the preprocessing rejection thresholds used, (3) a rejection log documenting preprocessing outcomes and skipped files, and (4) a failed-file log containing detailed error information for any recordings that could not be processed.

## ```Machine Learning Models - 06_RVR_trainer.py, 06_SVR_trainer.py, 06_decision_tree_trainer.py, 06_gpy_gpr_trainer.py, 06_linreg_trainer.py, 06_random_forest_trainer.py, 06_xgb_trainer.py```

Trains and evaluates the respective brain-age prediction model using the extracted EEG features, including (where applicable) hyperparameter optimization, feature scaling, age-bias correction, and SHAP-based model interpretation. The inputs are the train, validation, and test feature CSVs, and the outputs include the trained model, prediction files, performance metrics, learning curve plots, bias-corrected predictions, optimized hyperparameters (if applicable), and SHAP feature importance tables and visualizations. 

# References

Lubba, Carl H., et al. “Catch22: CAnonical Time-Series CHaracteristics.” Data Mining and Knowledge Discovery, vol. 33, no. 6, 9 Aug. 2019, pp. 1821–1852, 10.1007/s10618-019-00647-x.
Obeid, Iyad, and Joseph Picone. “The Temple University Hospital EEG Data Corpus.” Frontiers in Neuroscience, vol. 10, 13 May 2016, 10.3389/fnins.2016.00196. 
Sasidharan, A. (2026, May 15). ccs_toolbox (v0.1.2) (R. Venugopal, Ed.). Github. https://github.com/arunsasidharan84/ccs_toolbox.


