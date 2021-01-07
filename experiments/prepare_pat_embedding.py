import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import copy

import itertools
import collections
import time
import random

from src.utils import PickleUtils
from stellargraph import IndexedArray, StellarGraph

##############################################################
# Prepare data for patient embedding
##############################################################

# load data
spec_dict = pd.read_csv('saved_data/spec_dict.csv')
svc_dict = pd.read_csv('saved_data/svc_dict.csv')
med_jny = pd.read_parquet('saved_data/med_jny_dedup_lean.parquet')
enc_spec = pd.read_csv('saved_data/enc_spec.csv')
pat_table = pd.read_parquet('saved_data/pat_table.parquet')

# convert patient specialty service to id
med_jny = med_jny.merge(enc_spec, on='patientunitstayid', how='inner')
med_jny = med_jny.merge(spec_dict, on='specialty', how='inner')
med_jny = med_jny.sort_values(by=['patientunitstayid','offset'])

pat_num = len(med_jny.patientunitstayid.unique())
pat_dict = pd.DataFrame({'patientunitstayid':med_jny.patientunitstayid.unique(), 'pat_id':list(range(pat_num))})
pat_dict.to_csv('saved_data/pat_dict.csv', index=False)

med_jny = med_jny.merge(pat_dict, on='patientunitstayid', how='inner')
med_jny = med_jny.sort_values(by=['patientunitstayid','offset'])
med_jny = med_jny.drop(columns=['offset','PPD name','specialty'])
med_jny = med_jny[['patientunitstayid','pat_id','spec_id','svc_id']]

# prepare med_jny for sequence modeling
stay_svc_cnts = med_jny.groupby('patientunitstayid').size().reset_index(name='cnt')
stay_svc_cnts = stay_svc_cnts[(stay_svc_cnts.cnt <= 400) & (stay_svc_cnts.cnt >= 10)]

med_jny = med_jny[med_jny.patientunitstayid.isin(stay_svc_cnts.patientunitstayid)]
pat_seq = med_jny.groupby('patientunitstayid')['svc_id'].apply(lambda x: x.values).reset_index(name='seq')

def pad_seq(x):
    seq = np.ones((400,), dtype=int) * 3157
    seq[-len(x):] = x
    return seq
pat_seq['seq'] = pat_seq['seq'].apply(pad_seq)

pat_seq = pat_seq.merge(pat_table[['patientunitstayid','readmission']], on='patientunitstayid', how='inner')
pat_seq.to_parquet('saved_data/pat_seq_readmission_v2.parquet', index=False)

# get weights for each unique combo of service and specialty
# weight of a unique combo = # of the combo for one patient / total # of combos for one patient
data = med_jny.groupby(['patientunitstayid','pat_id','spec_id','svc_id']).size() \
    .groupby(level=0).apply(lambda x: x / float(x.sum())).reset_index(name='count')

# get negative weights of unique specialty-service combo
neg_weights = med_jny[['spec_id','svc_id']].value_counts().reset_index(name='freq')
neg_weights['freq'] = neg_weights['freq'] ** 0.75
neg_weights['freq'] = neg_weights['freq'] / sum(neg_weights['freq'])

data.to_csv('saved_data/data_ready_for_pat_emb.csv', index=False)
neg_weights.to_csv('saved_data/neg_weights.csv', index=False)



