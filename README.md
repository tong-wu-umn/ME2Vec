# ME2Vec: A Graph-Based Hierarchical Medical Embedding Framework for Healthcare Applications

## Authors
Tong Wu<sup>1</sup>, Yunlong Wang<sup>1*</sup>, Yue Wang<sup>1</sup>, Emily Zhao<sup>1</sup>, Yilian Yuan<sup>1</sup>

<sup>1</sup> Advanced Analytics, IQVIA Inc., Plymouth Meeting, Pennsylvania, USA

## Dataset

[**The eICU Collaborative Research Database**](https://eicu-crd.mit.edu/)

## How to use

### Generate ME2Vec embeddings

To generate service embedding, run

``
python experiments/prepare_svc_embedding.py
python node2vec/src/main.py --input graph/ppd_eICU.edgelist --output emb/ppd_eICU.emd
``

To generate doctor embedding, run

