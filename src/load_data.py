import pandas as pd
import os

# change this to your local path
DATA_DIR = r'C:\Users\amari\OneDrive\Desktop\readmission-predictor\data\mimic-iv-3.1\hosp'


def load_table(filename):
    return pd.read_csv(os.path.join(DATA_DIR, filename), compression='gzip')


def load_data():
    admissions = load_table('admissions.csv.gz')
    patients = load_table('patients.csv.gz')
    diagnoses = load_table('diagnoses_icd.csv.gz')

    return admissions, patients, diagnoses