import random
import numpy as np
import pandas as pd
from src.utils import PickleUtils

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.metrics import roc_auc_score, f1_score

##############################################################
# functions
##############################################################

def LogisticRegression_cv(data, labels, folds):
    nmf_perf = np.zeros((folds, 2), dtype=float)
    kf = StratifiedKFold(n_splits=folds, random_state=42)

    i = 0
    for train_idx, test_idx in kf.split(data, labels):
        
        x_train = data[train_idx]
        y_train = labels[train_idx]

        x_test = data[test_idx]
        y_test = labels[test_idx]

        clf = LogisticRegression(random_state=0).fit(x_train, y_train)
        y_pred = clf.predict_proba(x_test)[:, 1]

        nmf_perf[i, 0] = average_precision_score(y_test, y_pred, average='micro') # pr-auc
        nmf_perf[i, 1] = roc_auc_score(y_test, y_pred, average='micro') # roc-auc

        i += 1
    
    return np.mean(nmf_perf, axis=0)

##############################################################
# load data
##############################################################

# patient medical journey
med_jny = pd.read_parquet('saved_data/med_jny_dedup_lean.parquet')

# ppd embedding
svc_dict = pd.read_csv('saved_data/svc_dict.csv')
svc_emb = PickleUtils.loader('saved_data/svc_emb.pkl')

# spec embedding
spec_dict = pd.read_csv('saved_data/spec_dict.csv')
spec_emb = PickleUtils.loader('saved_data/spec_emb.pkl')

# patient dictionary
pat_dict = pd.read_csv('saved_data/pat_dict.csv')

pat_table = pd.read_parquet('saved_data/pat_table_lean.parquet')
pat_table = pat_table[pat_table.patientunitstayid.isin(pat_dict.patientunitstayid.values)]
pat_table = pat_table.sort_values(by='patientunitstayid')
pat_table = pat_table.reset_index(drop=True)
labels = pat_table.readmission.values

##############################################################
# ME2Vec
##############################################################

pat_emb_me2vec = PickleUtils.loader('saved_data/pat_emb.pkl')
pat_dict = pd.read_csv('saved_data/pat_dict.csv')

me2vec_perf = LogisticRegression_cv(pat_emb_me2vec, labels, 10)
print(me2vec_perf)

##############################################################
# metapath2vec
##############################################################

X = np.loadtxt('saved_data/baseline/baseline_emb_metapath2vec.emd', skiprows=1)
X_coor = np.array([x[1:] for x in X])
X_id = np.array([int(x[0]) for x in X])
ii = np.argsort(X_id)
X_coor = X_coor[ii]
PickleUtils.saver('saved_data/baseline/pat_metapath_emb.pkl', X_coor)

pat_metapath_emb = PickleUtils.loader('saved_data/baseline/pat_metapath_emb.pkl')
pat_metapath_emb = pat_metapath_emb[:141666]

metapath_perf = LogisticRegression_cv(pat_metapath_emb, labels, 10)
print(metapath_perf)

##############################################################
# node2vec
##############################################################

pat_node2vec_emb = PickleUtils.loader('saved_data/baseline/pat_node2vec_emb.pkl')
pat_node2vec_emb = pat_node2vec_emb[:141666]

# Readmission
node2vec_perf = LogisticRegression_cv(pat_node2vec_emb, labels, 10)
print(node2vec_perf)

##############################################################
# LINE
##############################################################

pat_line_emb = PickleUtils.loader('saved_data/baseline/pat_svc_emb_baseline_LINE.pkl')

line_perf = LogisticRegression_cv(pat_line_emb, labels, 10)
print(line_perf)

##############################################################
# nonnegative matrix factorization
##############################################################

# patient-ppd matrix
pat_svc_mat = pd.crosstab(med_jny.patientunitstayid, med_jny.svc_id)
pat_svc_sparse = csr_matrix(pat_svc_mat.values)

# nmf
nmf = NMF(n_components=128, random_state=0, init='random', verbose=True)
pat_nmf_emb = nmf.fit_transform(pat_svc_sparse)

nmf_perf = LogisticRegression_cv(pat_nmf_emb, labels, 10)
print(nmf_perf)

##############################################################
# spectral clustering
##############################################################

def spec_embed(G, num_k, edge_str):
    node_list = [f for f in G.nodes()]
    node_list.sort()
    L_norm = nx.normalized_laplacian_matrix(G, nodelist=node_list, weight=edge_str)
    L_norm = L_norm.astype('float')
    
    ev, v = eigs(L_norm, k=num_k, which='SM', maxiter=50)
    ev = ev.astype('float')
    v_norm = normalize(v.astype('float'), norm='l2')
    spec_df = pd.DataFrame(v_norm, columns=['eigvec_dim_{}'.format(f) for f in range(num_k)], 
                           dtype='float', index=node_list)
    return spec_df

pat_svc_cnts = med_jny.groupby(['patientunitstayid','svc_id'])['svc_id'].count().reset_index(name='cnt')
G = nx.from_pandas_edgelist(pat_svc_cnts, source='patientunitstayid', target='svc_id', edge_attr='cnt')
pat_sc_emb = spec_embed(G, 128, 'cnt')
pat_sc_emb = pat_sc_emb.values[-len(pat_table):]

sc_perf = LogisticRegression_cv(pat_sc_emb, labels, 10)
print(sc_perf)
