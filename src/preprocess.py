import pandas as pd
import polars as pl
from comorbidipy import comorbidity


def admission_features(admissions):
    admissions['admittime'] = pd.to_datetime(admissions['admittime'])
    admissions['dischtime'] = pd.to_datetime(admissions['dischtime'])
    admissions = admissions.sort_values(['subject_id', 'admittime'])

    # readmission label
    admissions['next_admittime'] = admissions.groupby('subject_id')['admittime'].shift(-1)
    admissions['days_to_next'] = (admissions['next_admittime'] - admissions['dischtime']).dt.days
    admissions['readmitted_30'] = (admissions['days_to_next'] <= 30).astype(int)

    # remove deaths
    admissions = admissions[admissions['discharge_location'] != 'DIED']

    # length of stay — calculate then shift to get previous stay's length
    admissions['length'] = (admissions['dischtime'] - admissions['admittime']) / pd.Timedelta(days=1)
    admissions['prev_length'] = admissions.groupby('subject_id')['length'].shift(1)

    # number of previous admissions
    admissions['No_of_admission'] = admissions.groupby('subject_id').cumcount()

    # days since last admission
    admissions['last_admittime'] = admissions.groupby('subject_id')['admittime'].shift(1)
    admissions['days_since_last'] = (admissions['admittime'] - admissions['last_admittime']).dt.days

    return admissions


def features(procedures, prescriptions, lab, admissions):
    # procedures per admission
    proc_counts = procedures.groupby('hadm_id')['seq_num'].count().reset_index()
    proc_counts.columns = ['hadm_id', 'num_procedures']

    # medications per admission
    med_counts = prescriptions.groupby('hadm_id')['poe_id'].count().reset_index()
    med_counts.columns = ['hadm_id', 'num_medications']

    # abnormal labs per admission
    lab.columns = ['hadm_id', 'num_abnormal_labs']

    # merge counts onto admissions so we can shift them
    adm = admissions[['subject_id', 'hadm_id', 'admittime']].copy()
    adm = adm.merge(proc_counts, on='hadm_id', how='left')
    adm = adm.merge(med_counts, on='hadm_id', how='left')
    adm = adm.merge(lab, on='hadm_id', how='left')

    # sort and shift to get previous admission's values
    adm = adm.sort_values(['subject_id', 'admittime'])
    adm['prev_num_procedures'] = adm.groupby('subject_id')['num_procedures'].shift(1)
    adm['prev_num_medications'] = adm.groupby('subject_id')['num_medications'].shift(1)
    adm['prev_num_abnormal_labs'] = adm.groupby('subject_id')['num_abnormal_labs'].shift(1)

    return adm[['hadm_id', 'prev_num_procedures', 'prev_num_medications', 'prev_num_abnormal_labs']]



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


def preprocess(admissions, patients, diagnoses, procedures, prescriptions, lab):
    df = admission_features(admissions)
    df = df.merge(patients, on='subject_id', how='left')

    comorbidities = build_comorbidities(diagnoses)
    df = df.merge(comorbidities, on='hadm_id', how='left')

    prev_features = features(procedures, prescriptions, lab, admissions)
    df = df.merge(prev_features, on='hadm_id', how='left')

    return df