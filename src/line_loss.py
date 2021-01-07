import torch
import torch.nn as nn
# import torch.nn.functional as F

class line_loss(nn.Module):
    """docstring for line_loss"nn.Modulef __init__(self, arg):
        super(line_loss,nn.Module.__init__()
        self.arg = arg
    """
    def __init__(self, num_pat, embed_dim):
        super(line_loss, self).__init__()
        self.num_pat = num_pat
        self.embed_dim = embed_dim
        self.pat_emb = nn.Embedding(num_embeddings=self.num_pat, embedding_dim=embed_dim)
        self.pat_emb.weight.data.uniform_(-1, 1)
        self.svc_map = nn.Linear(2 * embed_dim, embed_dim)

    def forward(self, input_labels, input_weights, pos_svcs, neg_svcs):
        """
            input_labels is tensor, dim = [batch_size * window_size,]
            input_weights is tensor, same dim as input_labels
        """
        input = self.pat_emb(input_labels)
        output = self.svc_map(pos_svcs)
        neg_svcs = self.svc_map(neg_svcs)

        log_target = (input * output).sum(1).squeeze().sigmoid().log()

        ''' ∑[batch_size * window_size, num_sampled, embed_size] * [batch_size * window_size, embed_size, 1] ->
            ∑[batch_size, num_sampled, 1] -> [batch_size]
        '''
        neg_svcs = neg_svcs.view(len(input), -1, self.embed_dim)
        sum_log_sampled = torch.bmm(neg_svcs.neg(), input.unsqueeze(2)).sigmoid().log().sum(1).squeeze()

        loss = torch.mul(log_target + sum_log_sampled, input_weights)

        return -loss.sum()

    def input_embeddings(self):
        return self.pat_emb.weight.data.cpu().numpy()


class line_loss_homo(nn.Module):
    def __init__(self, num_node, embed_dim, weights):
        super(line_loss_homo, self).__init__()
        self.num_node = num_node
        self.embed_dim = embed_dim
        self.weights = weights

        self.in_node_emb = nn.Embedding(num_embeddings=self.num_node, embedding_dim=self.embed_dim)
        self.out_node_emb = nn.Embedding(num_embeddings=self.num_node, embedding_dim=self.embed_dim)

        self.in_node_emb.weight.data.uniform_(-1, 1)
        self.out_node_emb.weight.data.uniform_(-1, 1)

    def sample(self, num_sample):
        """
        draws a sample from classes based on weights
        """
        return torch.multinomial(self.weights, num_sample, True)

    def forward(self, input_labels, output_labels, input_weights, num_sampled):
        """
            input_labels is tensor, dim = [batch_size * window_size,]
            input_weights is tensor, same dim as input_labels
        """
        input = self.in_node_emb(input_labels)
        output = self.out_node_emb(output_labels)

        noise_sample_count = len(input_labels) * num_sampled
        draw = self.sample(noise_sample_count)
        noise = draw.view(-1, num_sampled)
        noise = self.out_node_emb(noise).neg()

        log_target = (input * output).sum(1).squeeze().sigmoid().log()

        ''' ∑[batch_size * window_size, num_sampled, embed_size] * [batch_size * window_size, embed_size, 1] ->
            ∑[batch_size, num_sampled, 1] -> [batch_size]
        '''
        sum_log_sampled = torch.bmm(noise, input.unsqueeze(2)).sigmoid().log().sum(1).squeeze()

        loss = torch.mul(log_target + sum_log_sampled, input_weights)

        return -loss.sum()

    def input_embeddings(self):
        return self.in_node_emb.weight.data.cpu().numpy()

