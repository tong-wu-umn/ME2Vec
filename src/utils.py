import numpy as np
import torch
import pickle
from torch.utils.data import Dataset


def get_weights(self):
    pos_cnt = sum(self.train_label == 1)
    neg_cnt = len(self.train_label) - pos_cnt
    class_sample_count = [neg_cnt, pos_cnt]

    weights = [1000.0 / class_sample_count[self.train_label[idx]] for idx in range(len(self.train_label))]
    return torch.from_numpy(np.asarray(weights))


class PickleUtils(object):
    """
    Pickle file loader/saver utility functions
    """
    def __init__(self):
        pass
    
    @staticmethod
    def loader(directory):
        with open(directory, 'rb') as f:
            data = pickle.load(f)
        print("load pickle from {}".format(directory))
        return data
    
    @staticmethod
    def saver(directory, data):
        with open(directory, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("save pickle to {}".format(directory))


class DocDataset(Dataset):
    def __init__(self, doc_emb, doc_spec, doc_svc):
        self.doc_emb = torch.tensor(doc_emb, dtype=torch.float)
        self.doc_spec = torch.tensor(doc_spec, dtype=torch.int)
        self.doc_svc = doc_svc
    
    def __getitem__(self, index):
        slt_doc_emb = self.doc_emb[index]
        slt_doc_spec = self.doc_spec[index]
        slt_doc_svc = self.doc_svc[index]
        return slt_doc_emb, slt_doc_spec, slt_doc_svc
    
    def __len__(self):
        return len(self.doc_emb)



