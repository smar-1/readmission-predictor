import pandas as pd
import polars as pl
from comorbidipy import comorbidity


def admission_features(admissions):
    #Readmission label
    admissions['admittime'] = pd.to_datetime(admissions['admittime'])
    admissions['dischtime'] = pd.to_datetime(admissions['dischtime'])
    admissions = admissions.sort_values(['subject_id', 'admittime'])
    admissions['next_admittime'] = admissions.groupby('subject_id')['admittime'].shift(-1)
    admissions['days_to_next'] = (admissions['next_admittime'] - admissions['dischtime']).dt.days
    admissions['readmitted_30'] = (admissions['days_to_next'] <= 30).astype(int)

    # Remove admissions who die as they will not be readmitted and will cause data leak (model gaining access to information it won't have in a real environment)
    admissions = admissions[admissions['discharge_location'] != 'DIED']

    # length of stay
    admissions["length"] = admissions["dischtime"] - admissions["admittime"]
    admissions["length"] = admissions["length"] / pd.Timedelta(days=1)

    # number of previous admissions
    admissions['No_of_admission'] = admissions.groupby('subject_id')['hadm_id'].transform('count')

    #date since last admission
    admissions = admissions.sort_values(['subject_id', 'admittime'])
    admissions['last_admission'] = admissions.groupby('subject_id')['admittime'].shift(1)

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
    df = admission_features(admissions)
    df = df.merge(patients, on='subject_id', how='left')
    comorbidities = build_comorbidities(diagnoses)
    df = df.merge(comorbidities, on='hadm_id', how='left')
    return df