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

# load data
spec_dict = pd.read_csv('saved_data/spec_dict.csv')
svc_dict = pd.read_csv('saved_data/svc_dict.csv')
med_jny = pd.read_parquet('saved_data/med_jny_dedup_lean.parquet')
enc_spec = pd.read_csv('saved_data/enc_spec.csv')

# convert patient specialty service to id
med_jny = med_jny.merge(enc_spec, on='patientunitstayid', how='inner')
med_jny = med_jny.merge(spec_dict, on='specialty', how='inner')
med_jny = med_jny.sort_values(by=['patientunitstayid','offset'])

pat_num = len(med_jny.patientunitstayid.unique())
pat_dict = pd.DataFrame({'patientunitstayid':med_jny.patientunitstayid.unique(), 'pat_id':list(range(pat_num))})
pat_dict.to_csv('saved_data/pat_dict.csv', index=False)

med_jny = med_jny.merge(pat_dict, on='patientunitstayid', how='inner')
med_jny = med_jny.drop(columns=['offset','PPD name','specialty'])
med_jny = med_jny[['patientunitstayid','pat_id','spec_id','svc_id']]

##############################################################
# Prepare data for LINE
##############################################################

# get weights for each unique pat_id and svc_id
num_pats = len(med_jny.patientunitstayid.unique())
med_jny['svc_id'] = med_jny['svc_id'] + num_pats

data_pat_svc = med_jny.groupby(['pat_id','svc_id']).size() \
    .groupby(level=0).apply(lambda x: x / float(x.sum())).reset_index(name='count')
data_svc_pat = med_jny.groupby(['svc_id','pat_id']).size() \
    .groupby(level=0).apply(lambda x: x / float(x.sum())).reset_index(name='count')

data_pat_svc.columns = ['input_node','neighbor_node','weight']
data_svc_pat.columns = ['input_node','neighbor_node','weight']
data = pd.concat([data_pat_svc, data_svc_pat], axis=0,ignore_index=True)

data.to_csv('saved_data/baseline/data_for_baseline_LINE.csv', index=False)

# get negative weights
neg_weights = med_jny.groupby('svc_id').size().reset_index(name='freq').rename(columns={'svc_id':'node_id'})
neg_weights['freq'] = neg_weights['freq'] ** 0.75
neg_weights['freq'] = neg_weights['freq'] / sum(neg_weights['freq'])
svc_neg_weights = neg_weights.copy()

neg_weights = med_jny.groupby('pat_id').size().reset_index(name='freq').rename(columns={'pat_id':'node_id'})
neg_weights['freq'] = neg_weights['freq'] ** 0.75
neg_weights['freq'] = neg_weights['freq'] / sum(neg_weights['freq'])

neg_weights = neg_weights.append(svc_neg_weights).reset_index(drop=True)
neg_weights.to_csv('saved_data/baseline/neg_weights_baseline_LINE.csv', index=False)

##############################################################
# Prepare data for node2vec
##############################################################

num_pats = len(med_jny.patientunitstayid.unique())
med_jny['svc_id'] = med_jny['svc_id'] + num_pats

data = med_jny.groupby(['pat_id','svc_id']).size().reset_index(name='cnt')
data.to_csv('saved_data/baseline/baseline_node2vec.edgelist', header=None, index=None, sep=' ', mode='a')

##############################################################
# Prepare data for metapath2vec
##############################################################

med_jny['pat_id'] = med_jny['pat_id'].astype(str)
med_jny['spec_id'] = med_jny['spec_id'].astype(str)
med_jny['svc_id'] = med_jny['svc_id'].astype(str)

pat_spec_edges = med_jny[['pat_id','spec_id']].drop_duplicates() \
    .groupby(['pat_id','spec_id']).size() \
    .reset_index(name='weight') \
    .rename(columns={'pat_id':'source', 'spec_id':'target'})
pat_svc_edges = med_jny.groupby(['pat_id','svc_id']).size() \
    .reset_index(name='weight') \
    .rename(columns={'pat_id':'source', 'svc_id':'target'})
spec_svc_edges = med_jny[['spec_id','svc_id']].drop_duplicates() \
    .groupby(['spec_id','svc_id']).size() \
    .reset_index(name='weight') \
    .rename(columns={'spec_id':'source', 'svc_id':'target'})

all_edges = pd.concat([pat_spec_edges, pat_svc_edges, spec_svc_edges], axis=0, ignore_index=True)

pat_nodes = pd.DataFrame(index=med_jny.pat_id.unique().tolist())
spec_nodes = pd.DataFrame(index=med_jny.spec_id.unique().tolist())
svc_nodes = pd.DataFrame(index=med_jny.svc_id.unique().tolist())

graph_metapath = StellarGraph({"patient":pat_nodes, "specialty":spec_nodes, "service":svc_nodes}, all_edges)
PickleUtils.saver('saved_data/baseline/graph_metapath.pkl', graph_metapath)
