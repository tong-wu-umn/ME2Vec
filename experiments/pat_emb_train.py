import argparse
import random
import numpy as np
import pandas as pd
from src.utils import PickleUtils

import torch
import torch.optim as optim
from torch.utils.data import BatchSampler, RandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.line_loss import line_loss

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

    parser.add_argument('--lr-patience', type=int, default=5, help='number of epochs validation loss not improving')

    parser.add_argument('--neg-samples', type=int, default=10)

    parser.add_argument('--batch-size', type=int, default=512)

    parser.add_argument('--log-interval', type=int, default=20)

    parser.add_argument('--checkpoint', dest='checkpoint', action='store_true')
    parser.set_defaults(weighted=True)

    return parser.parse_args()

def train(epoch, model, optimizer, args, pat_journey, neg_weights, spec_emb, svc_emb):
    model.train()
    train_loss = 0

    num_pat = len(pat_journey.pat_id.unique())
    neg_weights_np = neg_weights.freq.values
    
    idx_list = list(BatchSampler(RandomSampler(range(num_pat)), args.batch_size, drop_last=False))
    
    for i in range(len(idx_list)):
        
        cur_batch_journey = pat_journey[pat_journey.pat_id.isin(idx_list[i])]
        
        input_labels = torch.tensor(cur_batch_journey['pat_id'].to_numpy(), dtype=torch.long, device=dev)
        input_weights = torch.tensor(cur_batch_journey['count'].to_numpy(), dtype=torch.float, device=dev)
        
        spec_emb_pos = spec_emb[cur_batch_journey['spec_id'].to_list()]
        svc_emb_pos = svc_emb[cur_batch_journey['svc_id'].to_list()]
        pos_svcs = torch.tensor(np.concatenate((spec_emb_pos, svc_emb_pos), axis=1), dtype=torch.float, device=dev)
        
        num_neg_samples = len(pos_svcs) * args.neg_samples
        neg_draw = torch.multinomial(torch.tensor(neg_weights_np, dtype=torch.float), num_neg_samples, True).squeeze()
        neg_journey = neg_weights.iloc[neg_draw.tolist()]
        spec_emb_neg = spec_emb[neg_journey['spec_id'].to_list()]
        svc_emb_neg = svc_emb[neg_journey['svc_id'].to_list()]
        neg_svcs = torch.tensor(np.concatenate((spec_emb_neg, svc_emb_neg), axis=1), dtype=torch.float, device=dev)
        
        optimizer.zero_grad()
        loss = model(input_labels, input_weights, pos_svcs, neg_svcs)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        
        if i % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (i+1) * args.batch_size, num_pat,
                100. * (i+1) * args.batch_size / num_pat, loss.item()/len(idx_list[i])))
        
    train_loss /= num_pat
    print('Average train loss per patient of epoch {} is {:.4f}.\n'.format(epoch, train_loss))
    
    return train_loss

def set_rnd_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

def main(args):
    data = pd.read_csv('saved_data/data_ready_for_pat_emb.csv')
    neg_weights = pd.read_csv('saved_data/neg_weights.csv')
    spec_emb = PickleUtils.loader('saved_data/spec_emb.pkl')
    svc_emb = PickleUtils.loader('saved_data/svc_emb.pkl')

    num_pat = len(data.patientunitstayid.unique())
    embed_dim = spec_emb.shape[1]
    set_rnd_seed(args)

    model = line_loss(num_pat, embed_dim).to(dev)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(opt, factor=args.lr_factor, patience=args.lr_patience, verbose=True)

    best_loss = 100.

    for epoch in range(1, args.epochs + 1):
            train_loss = train(epoch, model, opt, args, data, neg_weights, spec_emb, svc_emb)
            scheduler.step(train_loss)

            if train_loss < best_loss:
                best_loss = train_loss

                if args.check_point:
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': opt.state_dict(),
                        'train_loss': train_loss,
                        'epoch': epoch
                        }, 'saved_models/pat_emb_best')

    # load best model
    checkpoint = torch.load('saved_models/pat_emb_best')
    model.load_state_dict(checkpoint['model_state_dict'])

    pat_emb = model.input_embeddings()
    PickleUtils.saver('saved_data/pat_emb.pkl', pat_emb)

if __name__ == "__main__":
    args = parse_args()
    main(args)
