# Senior Thesis: Predicting Alzheimer’s Disease Progression using the ADNI Dataset

## Data

Data used for this project is from the Alzheimer's Disease Neuroimaging Initiative (ADNI). The following curated datasets were downloaded from [ADNI](https://ida.loni.usc.edu/login.jsp):
  * ADNI1: Complete 1Y 1.5T
  * ADNIMerge tablular data

## Software and Packages

* Python 3.12
* FMRIB Software Library (FSL)
* HD-BET (see packages folder)
* ANTs (see packages folder)

## Scan Preprocessing

1. Run Scan_Preprocessing.py (Ensure paths to imaging data is updated before running script)
2. Run tran_val_test_split.py

## Analysis

There are two notebooks to run for exploratory analyses and volumetric analyses. Update the paths to where your data is stored

## CNN Model

1. Run the pretraining script first, which pretrains the CNN models on the IXI dataset (https://brain-development.org/ixi-dataset/).
2. Run the CNN training notebook (This takes a considerable amount of time). If not not running on apple silicon, update "mps" to either "cpu" or "CUDA" to indicate what pytorch should use.
3. Run the CNN explanation notebook.
