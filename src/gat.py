import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import GraphAttentionLayer

class GAT_3L(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, drop_enc, alpha, nheads):
        """Dense version of GAT."""
        super(GAT_3L, self).__init__()
        self.dropout = dropout
        self.drop_enc = drop_enc
        self.nclass = nclass

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) 
            for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.mid_att = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) 
            for _ in range(nheads)]
        for i, attention in enumerate(self.mid_att):
            self.add_module('att_mid_{}'.format(i), attention)

        self.out_att = [GraphAttentionLayer(nfeat, nclass, dropout=dropout, alpha=alpha, concat=False) 
            for _ in range(nheads)]
        for i, attention in enumerate(self.out_att):
            self.add_module('att_out_{}'.format(i), attention)

    def forward(self, x, adj, svc_emb):
        # input layer
        if self.drop_enc:
            x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj, svc_emb) for att in self.attentions], dim=1)

        # middle layer
        if self.drop_enc:
            x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj, svc_emb) for att in self.mid_att], dim=1)

        x_prime = x.detach().clone()

        # output classification layer
        if self.drop_enc:
            x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj, svc_emb).view(1, -1, self.nclass) for att in self.out_att], dim=0)
        x = F.elu(torch.mean(x, dim=0))

        return x, x_prime


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, drop_enc, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.drop_enc = drop_enc
        self.nclass = nclass

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) 
            for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = [GraphAttentionLayer(nfeat, nclass, dropout=dropout, alpha=alpha, concat=False) 
            for _ in range(nheads)]
        for i, attention in enumerate(self.out_att):
            self.add_module('att_out_{}'.format(i), attention)

    def forward(self, x, adj, svc_emb):
        # input layer
        if self.drop_enc:
            x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj, svc_emb) for att in self.attentions], dim=1)

        x_prime = x.detach().clone()

        # output classification layer
        if self.drop_enc:
            x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj, svc_emb).view(1, -1, self.nclass) for att in self.out_att], dim=0)
        x = F.elu(torch.mean(x, dim=0))

        return x, x_prime
    


