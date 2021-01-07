# ME2Vec: A Graph-Based Hierarchical Medical Embedding Framework for Healthcare Applications

## Authors
Tong Wu<sup>1</sup>, Yunlong Wang<sup>1*</sup>, Yue Wang<sup>1</sup>, Emily Zhao<sup>1</sup>, Yilian Yuan<sup>1</sup>

<sup>1</sup> Advanced Analytics, IQVIA Inc., Plymouth Meeting, Pennsylvania, USA

## Dataset

[**The eICU Collaborative Research Database**](https://eicu-crd.mit.edu/)

Once obtaining the permission, download `patient.csv, admissionDx.csv, diagnosis.csv, treatment.csv, carePlanCareProvider.csv` to the folder `eICU_data`.

## How to use

### Generate ME2Vec embeddings

To generate service embedding, run
```python
python experiments/prepare_svc_embedding.py
python node2vec/src/main.py --input graph/ppd_eICU.edgelist --output emb/ppd_eICU.emd
```

To generate doctor embedding, run
```python
python experiments/prepare_doc_embedding.py
python experiments/doc_emb_train.py
```

To generate patient embedding, run
```python
python experiments/prepare_pat_embedding.py
python experiments/pat_emb_train.py
```

### Generate baseline embeddings

*node2vec*:
```python
python node2vec/src/main.py --input graph/baseline_node2vec.edgelist --output emb/baseline_node2vec_emb.emd
```

*LINE*:
```python
python experiments/line_emb_train.py
```

*metapath2vec*:
```python
python experiments/metapath2vec_emb_train.py --input saved_data/baseline/graph_metapath.pkl --output saved_data/baseline/baseline_emb_metapath2vec.emd
```

### Experiment: Readmission prediction

```python
python experiments/readmission_prediction.py
```

### Experiment: Sequential learning using pretrained emebeddings

```python
python experiments/sequential_learning_finetune.py 
```

