# ME2Vec: A Graph-Based Hierarchical Medical Embedding Framework for Healthcare Applications

## Authors
Tong Wu<sup>1</sup>, Yunlong Wang<sup>1*</sup>, Yue Wang<sup>1</sup>, Emily Zhao<sup>1</sup>, Yilian Yuan<sup>1</sup>

<sup>1</sup> Advanced Analytics, IQVIA Inc., Plymouth Meeting, Pennsylvania, USA

## Dataset

[**The eICU Collaborative Research Database**](https://eicu-crd.mit.edu/)

Once obtaining the permission, download `patient.csv`, `admissionDx.csv`, `diagnosis.csv`, `treatment.csv`, and `carePlanCareProvider.csv` to the folder `saved_data`.

## How to use

### Dependencies

```
tqdm==4.51.0
gensim==3.8.3
matplotlib==3.3.3
networkx==2.5
numpy==1.19.5
pandas==1.2.0
scikit_learn==0.24.0
stellargraph==1.2.1
torch==1.7.1
```

### Generate ME2Vec embeddings

The implementation of *node2vec* is from [aditya-grover](https://github.com/aditya-grover)/**[node2vec](https://github.com/aditya-grover/node2vec)**.

To generate service embedding, run
```python
python experiments/prepare_svc_embedding.py
python node2vec/src/main.py --input graph/ppd_eICU.edgelist --output emb/ppd_eICU.emd --dimensions 128 --walk-length 100 --num-walks 10 --window-size 20 --iter 150 --workers 8 --p 4 --q 1
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

Run the follwing script first:
```python
python experiments/prepare_baseline_embedding.py
```

*node2vec*:
```python
python node2vec/src/main.py --input graph/baseline_node2vec.edgelist --output emb/baseline_node2vec_emb.emd --dimensions 128 --walk-length 100 --num-walks 10 --window-size 20 --iter 150 --workers 8 --p 4 --q 1
```

*LINE*:
```python
python experiments/line_emb_train.py
```

*metapath2vec*:
```python
python experiments/metapath2vec_emb_train.py --input saved_data/baseline/graph_metapath.pkl --output saved_data/baseline/baseline_emb_metapath2vec.emd
```

For *nonnegative matrix factorization (NMF)* and *spectral clustering (SC)*, their embeddings are generated in the experiment code.

### Experiment: Readmission prediction

```python
python experiments/readmission_prediction.py
```

### Experiment: Sequential learning using pretrained emebeddings

```python
python experiments/sequential_learning_finetune.py 
```

