### This is an end-to-end pipeline for computing EEG based Brain Age Index developed at the Centre for Consciousness Studies, NIMHANS, Bangalore. 

![Dataset Used - TUH-EEG Corpus](https://isip.piconepress.com/projects/nedc/html/tuh_eeg/)

What does the pipeline involve?
- Data Engineering 
- EEG signal Preprocessing
- Feature Extraction & Engineering
- Catch22 based features
- In-house built feature set (CCS Features) 
- Machine Learning Models Training
- Feature Importance - SHapley Additive exPlanations (SHAP)

#### Dataset 
This project uses the TUH EEG Corpus (TUEG). This corpus is an archive of 14987 patients’ resting state EEG recordings collected at the Temple University Hospital. There are a total of 69670 resting state EEG recordings. 

The dataset is not an open-source dataset and thus is not included in this repository. Dataset can be requested from [here](https://isip.piconepress.com/projects/nedc/html/tuh_eeg/)

The dataset includes a readme file, a headers file and all the corresponding edf files. In our pipeline, data engineering steps were performed exclusively on the metadata contained within the headers text file, ensuring no EEG signal data were read, modified, or distorted at this stage. At each step, a CSV file was generated retaining a filepath column as a constant identifier, which was later used to call the relevant EDF files during pre-processing and feature extraction.

#### Feature Extraction
For this project, we used 2 feature sets for model training and prediction.
Feature set 1) Catch22 - 22 CAnonical Time-series CHaracteristics typically found in time-series data derived from hctsa feature [library](https://link.springer.com/article/10.1007/s10618-019-00647-x)

Feature set 2) In-house Feature set (CCS Features) -  Comprises Power-Spectra Density (PSD), Fitting Oscillations & One Over F (FOOOF), Irregular Resampling Auto-Spectral Analysis (IRASA), Auto-Correlation Window (ACW), Non-linear (entropy, fractal complexity, Lempel–Ziv complexity), Multifractal detrended fluctuation analysis). Extracted using [ccstools](https://github.com/arunsasidharan84/ccs_toolbox) 

#### Machine Learning Models

7 models were trained
- Simple Linear Regression
- Support Vector Machine
- Relevant Vector Machine
- Decision Tree
- Random Forest 
- XgBoost
- Gaussian Process Regression 

#### Feature Importance
SHAP feature importance was used. 

#### Requirements
All the libraries and their versions used in this repo are listed in requirements.txt. 
