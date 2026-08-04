import pandas as pd
import os

# change this to your local path
DATA_DIR = r'C:\Users\amari\OneDrive\Desktop\readmission-predictor\data\mimic-iv-3.1\hosp'

# simple function to load the table using the data directory
def load_table(filename, usecols=None):
    return pd.read_csv(
        os.path.join(DATA_DIR, filename),
        compression='gzip',
        usecols=usecols
    )


def load_data():
    admissions = load_table('admissions.csv.gz')
    patients = load_table('patients.csv.gz')
    diagnoses = load_table('diagnoses_icd.csv.gz')

    #the following tbales are too large to load fully so we only select the columns we need
    procedures = load_table('procedures_icd.csv.gz', usecols=['hadm_id', 'icd_code','seq_num'])
    prescriptions = load_table('prescriptions.csv.gz', usecols=['hadm_id', 'poe_id'])

    #lab data too large to load in at one time also calculates the number of abnormal lab findings per admission
    lab_chunks = []
    for chunk in pd.read_csv(
            r'C:\Users\amari\OneDrive\Desktop\readmission-predictor\data\mimic-iv-3.1\hosp\labevents.csv.gz',
            compression='gzip',
            usecols=['hadm_id', 'flag'],  # only load columns you need
            chunksize=500000
    ):
        # count abnormal flags per admission in this chunk
        abnormal = chunk[chunk['flag'] == 'abnormal'].groupby('hadm_id').size()
        lab_chunks.append(abnormal)

    # combine all chunks
    lab = pd.concat(lab_chunks).groupby('hadm_id').sum().reset_index()
    lab.columns = ['hadm_id', 'num_abnormal_labs']

    return admissions, patients, diagnoses, procedures, prescriptions, lab