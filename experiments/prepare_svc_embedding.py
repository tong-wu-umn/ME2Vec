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
# Create patient journeys (PPD sequences)
##############################################################

# load all necessary tables
pat_table_orig = pd.read_csv('saved_data/patient.csv')
admissionDx_table_orig = pd.read_csv('saved_data/admissionDx.csv')
diag_table_orig = pd.read_csv('saved_data/diagnosis.csv')
treatment_table_orig = pd.read_csv('saved_data/treatment.csv')

# merge pat diag and treatment tables
pat_table = pat_table_orig[['uniquepid','patientunitstayid','patienthealthsystemstayid','gender','age','ethnicity',
                            'hospitaladmitoffset','unitvisitnumber','unitdischargestatus']].drop_duplicates()

admissionDx_table = admissionDx_table_orig[['patientunitstayid','admitdxenteredoffset','admitdxpath']] \
    .rename(columns={'admitdxpath':'PPD name', 'admitdxenteredoffset':'offset'})
med_jny = admissionDx_table[admissionDx_table.patientunitstayid.isin(pat_table.patientunitstayid.to_list())].sort_values(by='patientunitstayid')

diag_table = diag_table_orig[['patientunitstayid','diagnosisoffset','diagnosisstring']] \
    .rename(columns={'diagnosisstring':'PPD name', 'diagnosisoffset':'offset'})
med_jny = pd.concat([med_jny, diag_table]).sort_values(by=['patientunitstayid', 'offset'])

treatment_table = treatment_table_orig[['patientunitstayid','treatmentoffset','treatmentstring']] \
    .rename(columns={'treatmentstring':'PPD name', 'treatmentoffset':'offset'})
med_jny = pd.concat([med_jny, treatment_table]).sort_values(by=['patientunitstayid', 'offset'])

# add labels for readmission
last_units = pat_table.groupby('patienthealthsystemstayid')['unitvisitnumber'].max().reset_index(name='unitvisitnumber')
last_units['readmission'] = 0
pat_table = pat_table.merge(last_units, on=['patienthealthsystemstayid','unitvisitnumber'], how='outer')
pat_table['readmission'] = pat_table['readmission'].fillna(1)

# remove adjacent duplicate codes within one encounter
def remove_adj_duplicate(x):
    col = x['PPD name'].to_list()
    ii = [i for i, n in enumerate(col) if i==0 or n != col[i-1]]
    return x.iloc[ii,:]

med_jny_dedup = med_jny.groupby('patientunitstayid').apply(remove_adj_duplicate).reset_index(drop=True)

# remove unit stays with less than 5 services
pat_rec_cnt = med_jny_dedup.patientunitstayid.value_counts().reset_index(name='count') \
    .rename(columns={'index':'patientunitstayid'})
pat_rec_cnt = pat_rec_cnt[pat_rec_cnt['count'] >= 5]
med_jny_dedup = med_jny_dedup[med_jny_dedup['patientunitstayid'].isin(pat_rec_cnt.patientunitstayid.to_list())]
pat_table = pat_table[pat_table['patientunitstayid'].isin(pat_rec_cnt.patientunitstayid.to_list())]
pat_table = pat_table.sort_values(by=['patienthealthsystemstayid','unitvisitnumber'])

# replace ppd sequences with their IDs
# the larger IDs are, the more infrequent
svc_cnts = med_jny_dedup['PPD name'].value_counts().reset_index(name='count') \
    .rename(columns={'index':'PPD name'})
svc_cnts['svc_id'] = list(range(len(svc_cnts)))
med_jny_dedup = med_jny_dedup.merge(svc_cnts, on=['PPD name'], how='inner').drop(columns='count')

# save data
pat_table.to_parquet('saved_data/pat_table.parquet', index=False)
med_jny_dedup.to_parquet('saved_data/med_jny_dedup.parquet', index=False)
svc_cnts.to_csv('saved_data/svc_dict.csv', index=False)

##############################################################
# Load ppd sequences data
##############################################################

data = pd.read_parquet('saved_data/med_jny_dedup.parquet')
data = data.sort_values(by=['patientunitstayid','offset','PPD name'])

pat_table = pd.read_parquet('saved_data/pat_table.parquet')
svc_dict = pd.read_csv('saved_data/svc_dict.csv')

# remove svc whose occurrences are less than 100
svc_dict_lean = svc_dict[svc_dict['count'] >= 100]
data_lean = data[data['svc_id'].isin(svc_dict_lean.svc_id.to_list())]
pat_table_lean = pat_table[pat_table.patientunitstayid.isin(data_lean.patientunitstayid.unique())]

data_lean.to_parquet('saved_data/med_jny_dedup_lean.parquet', index=False)
pat_table_lean.to_parquet('saved_data/pat_table_lean.parquet', index=False)

##############################################################
# Create numpy version of patient journey and adjacency matrix
##############################################################

pat_enc_cnts = data_lean.groupby('patientunitstayid')['offset'].count().reset_index(name='count')
pat_ind_split = np.cumsum(pat_enc_cnts['count'].to_numpy())
pat_ind_split = np.concatenate(([0], pat_ind_split))
data_lean_np = data_lean[['patientunitstayid','offset','svc_id']].to_numpy()

win_len = 60 # one hour
adj_mat = np.zeros((len(svc_dict_lean), len(svc_dict_lean)), dtype=int)

for i in tqdm(range(len(pat_enc_cnts))):

    pat_journey = data_lean_np[pat_ind_split[i]:pat_ind_split[i + 1]]
    ii = np.floor_divide(pat_journey[:, 1], win_len)
    ii_uni = np.unique(ii)
    for k in ii_uni:
        serv_win = pat_journey[ii == k, 2]
        if len(serv_win) == 1:
            continue
        indx = np.array(list(combinations(serv_win, 2)), dtype=int).T
        adj_mat[indx[0, :], indx[1, :]] += 1
        adj_mat[indx[1, :], indx[0, :]] += 1

adj_mat[np.diag_indices(len(svc_dict_lean))] = 0

# generate edge list
fh = open('node2vec/graph/ppd_eICU.edgelist', 'w')
for x, y in combinations(range(len(adj_mat)), 2):
    if adj_mat[x, y] > 0:
        fh.write(str(x + 1) + ' ' + str(y + 1) + ' ' + str(adj_mat[x, y]) + '\n')
fh.close()
