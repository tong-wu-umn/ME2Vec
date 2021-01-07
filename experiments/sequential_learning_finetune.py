import random
import pandas as pd
import numpy as np
from tqdm import tqdm
import datetime as dt
from itertools import combinations
import matplotlib.pyplot as plt
import collections
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from gensim.models import Word2Vec
from src.utils import PickleUtils

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import BatchSampler, RandomSampler, SequentialSampler

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class VanillaLSTM(nn.Module):
    def __init__(self, vocab, lstm_layers, lstm_units, embed_dim, drop_rate):
        super(VanillaLSTM, self).__init__()
        
        # initializatioin
        self.vocab = vocab
        self.lstm_layers = lstm_layers
        self.lstm_units = lstm_units
        self.embed_dim = embed_dim
        self.drop_rate = drop_rate
        
        # embedding layer
        vocab_size = len(self.vocab)
        
        padding_idx = self.vocab['<PAD>']
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=self.embed_dim,
            padding_idx=padding_idx
        )
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=self.embed_dim,
            hidden_size=self.lstm_units,
            num_layers=self.lstm_layers,
            batch_first=True,
            dropout=self.drop_rate
        )
        
        self.fc = nn.Sequential(
            nn.Linear(self.lstm_units, int(self.lstm_units / 2.0)),
            nn.ReLU(),
            nn.Dropout(p=self.drop_rate),
            nn.Linear(int(self.lstm_units / 2.0), 1)
        )
        
    def forward(self, jny, hid_init):
        
        # embedding
        jny_embed = self.embedding(jny)
        
        # LSTM
        x, _ = self.lstm(jny_embed, hid_init) # dim of x is (Batch, len, feat)
        (x, _) = torch.max(x, dim=1)
        
        return self.fc(x)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=1986,
                        help='global random seed number')

    parser.add_argument('--epochs', type=int, default=20,
                        help='number of epochs of training')

    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate')

    parser.add_argument('--drop-rate', type=float, default=0.5,
                        help='dropout rate')

    parser.add_argument('--clip', type=float, default=0.25)

    parser.add_argument('--embed-dim', type=int, default=128)

    parser.add_argument('--lstm-layers', type=int, default=2)

    parser.add_argument('--lstm-units', type=int, default=256)

    parser.add_argument('--batch-size', type=int, default=64)

    parser.add_argument('--log-interval', type=int, default=100)

    parser.add_argument('--embedding', type=int, default=0, help='0: me2vec; 1: metapath2vec; 2: node2vec; 3: word2vec; 4: random initialization.')

    parser.add_argument('--checkpoint', dest='checkpoint', action='store_true')
    parser.set_defaults(weighted=True)

    return parser.parse_args()

def set_rnd_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def load_emb(ppd_path):
    ppd_emb = np.loadtxt(ppd_path, skiprows=1)
    ppd_coor = np.array([x[1:] for x in ppd_emb])
    ppd_id = [int(x[0]) for x in ppd_emb]
    svc_emb = ppd_coor[np.argsort(ppd_id), :]
    
    return np.sort(ppd_id), svc_emb

def train(epoch, model, optimizer, args, padded_jny, pat_lbls):
    '''
    padded_jny: padded and tokenized patient journey
    jny_lens: lengths of each patient's journey
    pat_lbls: binary outcome of each patient
    '''

    # set the model in train mode
    model.train()

    train_loss = 0

    idx_list = list(BatchSampler(RandomSampler(range(len(padded_jny))), args.batch_size, drop_last=False))

    padded_jny_ts = torch.tensor(padded_jny, device=args.dev, dtype=torch.long)
    pat_lbls_ts = torch.tensor(pat_lbls, device=args.dev, dtype=torch.float)

    for i in range(len(idx_list)):

        # load current batch into tensor
        cur_batch_jnys = padded_jny_ts[idx_list[i]]
        cur_batch_lbls = pat_lbls_ts[idx_list[i]]

        # train model
        h_0 = torch.randn(args.lstm_layers, len(cur_batch_jnys), args.lstm_units, device=args.dev)
        c_0 = torch.randn(args.lstm_layers, len(cur_batch_jnys), args.lstm_units, device=args.dev)
        
        optimizer.zero_grad()
        y_pred = model(cur_batch_jnys, (h_0, c_0)).squeeze()
        loss = F.binary_cross_entropy_with_logits(y_pred, cur_batch_lbls)
        train_loss += loss.item() * len(cur_batch_jnys)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        # display running loss
        if i % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}'.format(
                epoch, (i + 1) * args.batch_size, len(padded_jny),
                100. * (i + 1) * args.batch_size / len(padded_jny), loss.item()))

    train_loss /= len(padded_jny)
    print('Average train loss of epoch {} is {:.4f}.'.format(epoch, train_loss))

    return train_loss

def test(epoch, model, args, padded_jny, pat_lbls):

    # set the mode in testing mode
    model.eval()

    test_loss = 0

    idx_list = list(BatchSampler(SequentialSampler(range(len(padded_jny))), args.batch_size, drop_last=False))

    padded_jny_ts = torch.tensor(padded_jny, device=args.dev, dtype=torch.long)
    pat_lbls_ts = torch.tensor(pat_lbls, device=args.dev, dtype=torch.float)

    y_pred_total = torch.zeros(1,)

    with torch.no_grad():
        for i in range(len(idx_list)):

            # load current batch into tensor
            cur_batch_jnys = padded_jny_ts[idx_list[i]]
            cur_batch_lbls = pat_lbls_ts[idx_list[i]]

            # test model
            h_0 = torch.randn(args.lstm_layers, len(cur_batch_jnys), args.lstm_units, device=args.dev)
            c_0 = torch.randn(args.lstm_layers, len(cur_batch_jnys), args.lstm_units, device=args.dev)
        
            y_pred = model(cur_batch_jnys, (h_0, c_0)).squeeze()
            y_pred_total = torch.cat((y_pred_total, torch.sigmoid(y_pred).detach().cpu()))
            loss = F.binary_cross_entropy_with_logits(y_pred, cur_batch_lbls)
            test_loss += loss.item() * len(cur_batch_jnys)

    test_loss /= len(padded_jny)
    print('Average test loss of epoch {} is {:.4f}.'.format(epoch, test_loss))

    return y_pred_total[1:].numpy(), test_loss

def save_best_model(model, PATH):    
    torch.save({'model_state_dict': model.state_dict()}, PATH)

def main(args):

    med_seq = pd.read_parquet('saved_data/pat_seq_readmission_v2.parquet')
    data_jny_np = np.stack(med_seq.seq.values, axis=0)
    labels = med_seq.readmission.values
    svc_dict = pd.read_csv('saved_data/svc_dict.csv')
    svc_dict = dict(zip(svc_dict['PPD name'], svc_dict['svc_id']))
    svc_dict['<PAD>'] = 3157

    if args.embedding == 0:
        svc_emb = PickleUtils.loader('saved_data/svc_emb.pkl')
    elif args.embedding == 1:
        svc_emb = PickleUtils.loader('saved_data/baseline/pat_metapath_emb.pkl')
        svc_emb = svc_emb[141666:(141666 + 3157)]
    elif args.embedding == 2:
        svc_emb = PickleUtils.loader('saved_data/baseline/pat_node2vec_emb.pkl')
        svc_emb = svc_emb[141666:(141666 + 3157)]
    svc_id = np.asarray(list(range(3157)))

    svc_emb_ts = torch.randn(len(svc_id)+1, args.embed_dim, dtype=torch.float)
    svc_emb_ts[-1] = torch.zeros(args.embed_dim, dtype=torch.float)
    svc_emb_ts[svc_id] = torch.FloatTensor(svc_emb)

    pr_logs = np.zeros((2, 10))

    skf = StratifiedShuffleSplit(train_size=0.8, random_state=0, n_splits=10)
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(data_jny_np, labels)):
        
        print('=' * 70)
        print('Fold={}'.format(fold_idx))

        # data split
        train_jny_np = data_jny_np[train_idx]
        train_labels = labels[train_idx]
        
        test_jny_np = data_jny_np[test_idx]
        test_labels = labels[test_idx]
        
        ss = StratifiedShuffleSplit(train_size=0.5)
        ii = next(ss.split(test_jny_np, test_labels))
        
        val_jny_np = test_jny_np[ii[0]]
        val_labels = test_labels[ii[0]]
        
        test_jny_np = test_jny_np[ii[1]]
        test_labels = test_labels[ii[1]]
        
        train_jny_list = train_jny_np.tolist()
        
        # train model (with pretrained emb)
        set_rnd_seed(1986)

        model = VanillaLSTM(
            vocab=svc_dict,
            lstm_layers=args.lstm_layers,
            lstm_units=args.lstm_units,
            embed_dim=args.embed_dim,
            drop_rate=args.drop_rate
        )
        
        if args.embedding == 3:
            # train word2vec embedding
            walks = [list(map(str, walk)) for walk in train_jny_list]
            wv_model = Word2Vec(walks, size=args.embed_dim, window=20, min_count=0, sg=1, 
                             workers=8, iter=150)
            wv_model.wv.save_word2vec_format('saved_data/baseline/wv2.emd')
            svc_id, svc_emb = load_emb('saved_data/baseline/wv2.emd')
            svc_emb_ts = torch.randn(len(svc_dict), args.embed_dim, dtype=torch.float)
            svc_emb_ts[-1] = torch.zeros(args.embed_dim, dtype=torch.float)
            svc_emb_ts[svc_id] = torch.FloatTensor(svc_emb)
        
        if args.embedding != 4:
            model.embedding = nn.Embedding.from_pretrained(svc_emb_ts, freeze=False)

        model = model.to(args.dev)
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        best_pr = 0
        for epoch in range(1, args.epochs + 1):
            train_loss = train(epoch, model, opt, args, train_jny_np, train_labels)
            y_pred, val_loss = test(epoch, model, args, val_jny_np, val_labels)

            pr_auc = average_precision_score(val_labels, y_pred, average='micro')
            if pr_auc > best_pr:
                best_pr = pr_auc
                save_best_model(model, 'saved_models/best_pretrain_sequential')
            print('PR AUC of epoch {} is {:.4f}.\n'.format(epoch, pr_auc))
        
        # load model and evaluate on the test set
        checkpoint = torch.load('saved_models/best_pretrain_sequential')
        model.load_state_dict(checkpoint['model_state_dict'])
        y_pred, _ = test(1, model, args, test_jny_np, test_labels)
        pr_logs[0, fold_idx] = average_precision_score(test_labels, y_pred, average='micro')
        pr_logs[1, fold_idx] = roc_auc_score(test_labels, y_pred, average='micro')

    print(pr_logs)

if __name__ == "__main__":
    args = parse_args()
    main(args)

