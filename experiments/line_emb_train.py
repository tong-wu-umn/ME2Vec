import random
import argparse
import numpy as np
import pandas as pd
from src.utils import PickleUtils

import torch
import torch.optim as optim
from torch.utils.data import BatchSampler, RandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.line_loss import line_loss_homo

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=1986,
                        help='global random seed number')

    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs of training')

    parser.add_argument('--lr', type=float, default=0.0025,
                        help='learning rate')

    parser.add_argument('--lr-factor', type=float, default=0.25,
                        help='rate of reducing learning rate')

    parser.add_argument('--lr-patience', type=int, default=5,
                        help='number of epochs validation loss not improving')

    parser.add_argument('--neg-samples', type=int, default=10)

    parser.add_argument('--embed-dim', type=int, default=128)

    parser.add_argument('--batch-size', type=int, default=512)

    parser.add_argument('--log-interval', type=int, default=20)

    parser.add_argument('--checkpoint', dest='checkpoint', action='store_true')
    parser.set_defaults(weighted=True)

    return parser.parse_args()

def train(epoch, model, optimizer, args, pat_journey, num_node):
    '''
        pat_journey contains three columns: input_node, neighbor_node, weight
        pat IDs and svc IDs are non-overlapping.
    '''

    model.train()
    train_loss = 0
    
    idx_list = list(BatchSampler(RandomSampler(pat_journey['input_node'].unique()), args.batch_size, drop_last=False))
    
    for i in range(len(idx_list)):
        
        cur_batch_journey = pat_journey[pat_journey['input_node'].isin(idx_list[i])]
        
        input_labels = torch.tensor(cur_batch_journey['input_node'].to_numpy(), dtype=torch.long, device=args.dev)
        output_labels = torch.tensor(cur_batch_journey['neighbor_node'].to_numpy(), dtype=torch.long, device=args.dev)
        input_weights = torch.tensor(cur_batch_journey['weight'].to_numpy(), dtype=torch.float, device=args.dev)
                
        optimizer.zero_grad()
        loss = model(input_labels, output_labels, input_weights, args.neg_samples)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        
        if i % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (i+1) * args.batch_size, num_node,
                100. * (i+1) * args.batch_size / num_node, loss.item()/len(idx_list[i])))
        
    train_loss /= num_node
    print('Average train loss per patient of epoch {} is {:.4f}.\n'.format(epoch, train_loss))
    
    return train_loss

def set_rnd_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

def main(args):

    # load data
    data = pd.read_csv('saved_data/baseline/data_for_baseline_LINE.csv')
    neg_weights = pd.read_csv('saved_data/baseline/neg_weights_baseline_LINE.csv')
    neg_weights = torch.tensor(neg_weights.freq.values, dtype=torch.float, device=args.dev)

    # define model
    num_node = len(data.input_node.unique())
    set_rnd_seed(args)

    model = line_loss_homo(num_node, args.embed_dim, neg_weights).to(args.dev)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(opt, factor=args.lr_factor, patience=args.lr_patience, verbose=True)

    best_loss = 100.

    for epoch in range(1, args.epochs + 1):
            train_loss = train(epoch, model, opt, args, data, num_node)
            scheduler.step(train_loss)

            if train_loss < best_loss:
                best_loss = train_loss

                if args.check_point:
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': opt.state_dict(),
                        'train_loss': train_loss,
                        'epoch': epoch
                        }, 'saved_models/pat_emb_baseline_LINE')

    # load best model
    checkpoint = torch.load('saved_models/pat_emb_baseline_LINE')
    model.load_state_dict(checkpoint['model_state_dict'])

    pat_emb = model.input_embeddings()
    PickleUtils.saver('saved_data/baseline/pat_svc_emb_baseline_LINE.pkl', pat_emb[:141666])

if __name__ == "__main__":
    args = parse_args()
    main(args)
