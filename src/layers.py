import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True, drop_enc=False):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.drop_enc = drop_enc

        # dimension of W is (d, d')
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj, serv_emb):
        """
        input is initialized doctor embedding, dim = (N, d)
        adj is the list of services done by each doctor, dim = (N,)
        serv_emb is the embedding matrix of all services, dim = (K, d)

        h_prime is the new doctor embedding, dim = (N, d)
        """

        N, d = input.size()
        num_svc = serv_emb.size()[0]

        h_doc = torch.mm(input, self.W)
        h_svc = torch.mm(serv_emb, self.W)

        a_input = torch.cat((h_doc.repeat(1, num_svc).view(N * num_svc, -1), 
            h_svc.repeat(N, 1)), dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        if self.drop_enc:
            attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h_svc)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime
