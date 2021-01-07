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
# Create doctor-service table
##############################################################

doc_table = pd.read_csv('eICU_data/carePlanCareProvider.csv')
doc_table = doc_table.dropna(subset=['specialty'])

enc_spec = doc_table.groupby('patientunitstayid')['specialty'] \
    .apply(lambda x: x.value_counts().index[0]) \
    .reset_index(name='specialty')
enc_spec.to_csv('saved_data/enc_spec.csv', index=False)

med_jny = pd.read_parquet('saved_data/med_jny_dedup_lean.parquet')
med_jny = med_jny.merge(enc_spec, on='patientunitstayid', how='inner')

spec_dict = pd.DataFrame({'specialty':med_jny.specialty.unique(), 'spec_id':list(range(med_jny.specialty.nunique()))})
spec_dict.to_csv('saved_data/spec_dict.csv', index=False)

# load trained svc embedding
X = np.loadtxt('node2vec/emb/ppd_eICU.emd', skiprows=1)
X_coor = np.array([x[1:] for x in X])
X_id = np.array([int(x[0]) for x in X])
ii = np.argsort(X_id)
X_coor = X_coor[ii]
PickleUtils.saver('saved_data/svc_emb.pkl', X_coor)

# initialize doctor embedding
def get_init_emb(x, emb):
    return np.mean(emb[x['svc_id']], axis=0)

spec_init_emb = med_jny.groupby('patientunitstayid').apply(get_init_emb, emb=X_coor).reset_index(name='embedding')
spec_svc = med_jny.groupby('patientunitstayid')['svc_id'].apply(lambda x: list(x.unique())) \
    .reset_index(name='svc_id')
spec_init_emb = spec_init_emb.merge(spec_svc, on='patientunitstayid', how='inner')
spec_init_emb = spec_init_emb.merge(enc_spec, on='patientunitstayid', how='inner') \
    .merge(spec_dict, on='specialty', how='inner') \
    .drop(columns='specialty')

spec_init_emb.to_parquet('saved_data/spec_init_emb.parquet', index=False)
