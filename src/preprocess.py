import pandas as pd
import polars as pl
from comorbidipy import comorbidity


def build_readmission_label(admissions):
    admissions['admittime'] = pd.to_datetime(admissions['admittime'])
    admissions['dischtime'] = pd.to_datetime(admissions['dischtime'])
    admissions = admissions.sort_values(['subject_id', 'admittime'])
    admissions['next_admittime'] = admissions.groupby('subject_id')['admittime'].shift(-1)
    admissions['days_to_next'] = (admissions['next_admittime'] - admissions['dischtime']).dt.days
    admissions['readmitted_30'] = (admissions['days_to_next'] <= 30).astype(int)
    admissions = admissions[admissions['discharge_location'] != 'DIED']
    return admissions


def build_comorbidities(diagnoses):
    di_9 = diagnoses[diagnoses['icd_version'] == 9]
    di_10 = diagnoses[diagnoses['icd_version'] == 10]

    result_9 = comorbidity(
        pl.from_pandas(di_9[['hadm_id', 'icd_code']]),
        id_col='hadm_id',
        code_col='icd_code',
        score='elixhauser',
        weighting='van_walraven',
        icd='icd9'
    ).to_pandas()

    result_10 = comorbidity(
        pl.from_pandas(di_10[['hadm_id', 'icd_code']]),
        id_col='hadm_id',
        code_col='icd_code',
        score='elixhauser',
        weighting='van_walraven',
        icd='icd10'
    ).to_pandas()

    return pd.concat([result_9, result_10]).groupby('hadm_id').max().reset_index()


def preprocess(admissions, patients, diagnoses):
    df = build_readmission_label(admissions)
    df = df.merge(patients, on='subject_id', how='left')
    comorbidities = build_comorbidities(diagnoses)
    df = df.merge(comorbidities, on='hadm_id', how='left')
    return df