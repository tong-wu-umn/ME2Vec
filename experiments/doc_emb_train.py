import random
import argparse
import numpy as np
import pandas as pd
from src.utils import PickleUtils
from src.gat import GAT

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import BatchSampler, RandomSampler, SequentialSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

##############################################################
# Define parameters and load data
##############################################################

def parse_args():
	parser = argparse.ArgumentParser()

	parser.add_argument('--seed', type=int, default=1986,
	                    help='global random seed number')

	parser.add_argument('--epochs', type=int, default=100,
	                    help='number of epochs of training')

	parser.add_argument('--lr', type=float, default=0.01,
	                    help='learning rate')

	parser.add_argument('--lr-factor', type=float, default=0.2,
	                    help='rate of reducing learning rate')

	parser.add_argument('--lr-patience', type=int, default=3,
	                    help='number of epochs validation loss not improving')

	parser.add_argument('--batch-size', type=int, default=512)

	parser.add_argument('--log-interval', type=int, default=20)

	parser.add_argument('--weight-decay', type=float, default=0.)

	parser.add_argument('--nb-heads', type=int, default=4, 
						help='number of attention heads')

	parser.add_argument('--dropout', type=float, default=0.6)

	parser.add_argument('--alpha', type=float, default=0.2,
						help='parameters of GAT')

	parser.add_argument('--checkpoint', dest='checkpoint', action='store_true')
	parser.set_defaults(weighted=True)

	return parser.parse_args()

##############################################################
# Define train and test functions
##############################################################

def train(epoch, model, optimizer, args, doc_emb, doc_spec, doc_svc, svc_emb):

    model.train()
    train_loss = 0

    idx_list = list(BatchSampler(RandomSampler(range(len(doc_emb))), args.batch_size, drop_last=False))
    doc_emb_ts = torch.tensor(doc_emb, dtype=torch.float, device=dev)
    doc_spec_ts = torch.tensor(doc_spec, dtype=torch.long, device=dev)
    doc_svc_ts = torch.tensor(doc_svc, dtype=torch.long, device=dev)

    for i in range(len(idx_list)):
        x = doc_emb_ts[idx_list[i]]
        y = doc_spec_ts[idx_list[i]]
        adj = doc_svc_ts[idx_list[i]]

        optimizer.zero_grad()
        pred_y, _ = model(x, adj, svc_emb)
        loss = F.cross_entropy(pred_y, y)
        train_loss += loss.item() * len(x)
        loss.backward()
        optimizer.step()

        if i % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (i+1) * args.batch_size, len(doc_emb),
                100. * (i+1) * args.batch_size / len(doc_emb), loss.item()))

    train_loss /= len(doc_emb)
    print('Average train loss of epoch {} is {:.4f}.\n'.format(epoch, train_loss))

    return train_loss

def test(epoch, model, optimizer, args, doc_emb, doc_spec, doc_svc, svc_emb):
    
    model.eval()
    test_loss = 0
    correct = 0

    idx_list = list(BatchSampler(SequentialSampler(range(len(doc_emb))), args.batch_size, drop_last=False))
    doc_emb_ts = torch.tensor(doc_emb, dtype=torch.float, device=dev)
    doc_spec_ts = torch.tensor(doc_spec, dtype=torch.long, device=dev)
    doc_svc_ts = torch.tensor(doc_svc, dtype=torch.long, device=dev)
    
    with torch.no_grad():
        for i in range(len(idx_list)):
            x = doc_emb_ts[idx_list[i]]
            y = doc_spec_ts[idx_list[i]]
            adj = doc_svc_ts[idx_list[i]]

            pred_y, _ = model(x, adj, svc_emb)
            test_loss += F.cross_entropy(pred_y, y, reduction='sum')
            pred = F.log_softmax(pred_y, dim=1).argmax(dim=1, keepdim=True)
            correct += pred.eq(y.view_as(pred)).sum().item()

            if i % args.log_interval == 0:
                print('Test Epoch: {} [{}/{} ({:.0f}%)]'.format(
                    epoch, (i+1) * args.batch_size, len(doc_emb),
                    100. * (i+1) * args.batch_size / len(doc_emb)))

    test_loss /= len(doc_emb)
    accu = 100. * correct / len(doc_emb)

    print('Average test loss of epoch {} is {:.4f}, accuracy is {:.2f}%.\n'.format(epoch, test_loss, accu))

    return test_loss, accu

def test_batch(model, args, doc_emb, doc_svc, svc_emb):
    
    model.eval()

    idx_list = list(BatchSampler(SequentialSampler(range(len(doc_emb))), args.batch_size, drop_last=False))

    with torch.no_grad():
        for i in range(len(idx_list)):
            x = doc_emb[idx_list[i]]
            adj = doc_svc[idx_list[i]]
            _, x_prime = model(x, adj, svc_emb)

            if i == 0:
                doc_emb_prime = x_prime.detach().cpu().numpy()
            else:
                doc_emb_prime = np.concatenate((doc_emb_prime, x_prime.detach().cpu().numpy()), axis=0)

            if i % args.log_interval == 0:
                print('Processed: [{}/{} ({:.0f}%)]'.format(
                    (i+1) * args.batch_size, len(doc_emb),
                    100. * (i+1) * args.batch_size / len(doc_emb)))

    return doc_emb_prime

def set_rnd_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

def get_doc_emb(args):
	# load prepared init data for training doctor embedding
	doc_emb_data = pd.read_parquet('saved_data/spec_init_emb.parquet')

	doc_emb = np.stack(doc_emb_data.embedding.to_list(), axis=0)
	doc_spec = doc_emb_data.spec_id.values
	doc_svc = doc_emb_data.svc_id.values

	# load service embedding matrix
	ppd_emb = np.loadtxt('node2vec/emb/ppd_eICU.emd', skiprows=1)
	ppd_coor = np.array([x[1:] for x in ppd_emb])
	ppd_id = [int(x[0]) for x in ppd_emb]

	# sort ppd_emb in the ascending order of node id
	svc_emb = ppd_coor[np.argsort(ppd_id), :]
	svc_emb_ts = torch.tensor(svc_emb, dtype=torch.float, device=dev)

	# prepare dataset
	num_doc = len(doc_emb_data) # len(doc_emb)
	num_svc = len(svc_emb)
	adj_mat = np.zeros((num_doc, num_svc), dtype=int)
	for i in range(num_doc):
	    adj_mat[i, doc_svc[i]] = 1

	doc_emb_ts = torch.tensor(doc_emb, dtype=torch.float, device=dev)
	doc_svc_ts = torch.tensor(adj_mat, dtype=torch.long, device=dev)

	# define and load models
	model = GAT(
	    nfeat=doc_emb.shape[1], 
	    nhid=int(doc_emb.shape[1] / args.nb_heads), 
	    nclass=len(np.unique(doc_spec)), 
	    dropout=args.dropout, 
	    drop_enc=args.drop_enc, 
	    alpha=args.alpha, 
	    nheads=args.nb_heads).to(dev)

	# load model parameters
	checkpoint = torch.load('saved_models/doc_emb_best')
	model.load_state_dict(checkpoint['model_state_dict'])

	# inference
	doc_emb_prime = test_batch(model, args, doc_emb_ts, doc_svc_ts, svc_emb_ts)
	PickleUtils.saver('saved_data/doc_emb_prime.pkl', doc_emb_prime)

	doc_emb = np.stack(doc_emb_data.embedding.to_list(), axis=0)
	doc_spec = doc_emb_data.spec_id.values
	doc_svc = doc_emb_data.svc_id.values

	spec_emb = np.zeros((49, 128), dtype=float)
	for i in range(49):
	    spec_emb[i] = np.mean(doc_emb_prime[np.where(doc_spec == i)], axis=0)
	PickleUtils.saver('saved_data/spec_emb.pkl', spec_emb)

def main(args):
	# load prepared init data for training doctor embedding
	doc_emb_data = pd.read_parquet('saved_data/spec_init_emb.parquet')

	doc_emb = np.stack(doc_emb_data.embedding.to_list(), axis=0)
	doc_spec = doc_emb_data.spec_id.values
	doc_svc = doc_emb_data.svc_id.values

	# load service embedding matrix
	ppd_emb = np.loadtxt('node2vec/emb/ppd_eICU.emd', skiprows=1)
	ppd_coor = np.array([x[1:] for x in ppd_emb])
	ppd_id = [int(x[0]) for x in ppd_emb]

	# sort ppd_emb in the ascending order of node id 
	svc_emb = ppd_coor[np.argsort(ppd_id), :]
	svc_emb_ts = torch.tensor(svc_emb, dtype=torch.float, device=dev)

	# prepare dataset
	num_doc = len(doc_emb_data)
	num_svc = len(svc_emb)
	set_rnd_seed(args)
	rndx = np.random.permutation(range(num_doc))

	doc_emb = doc_emb[rndx]
	doc_spec = doc_spec[rndx]
	doc_svc = doc_svc[rndx]

	adj_mat = np.zeros((num_doc, num_svc), dtype=int)
	for i in range(num_doc):
	    adj_mat[i, doc_svc[i]] = 1
	    
	doc_emb_train = doc_emb[:round(num_doc * args.train_ratio)]
	doc_emb_test = doc_emb[round(num_doc * args.train_ratio):]

	doc_spec_train = doc_spec[:round(num_doc * args.train_ratio)]
	doc_spec_test = doc_spec[round(num_doc * args.train_ratio):]

	doc_svc_train = adj_mat[:round(num_doc * args.train_ratio)]
	doc_svc_test = adj_mat[round(num_doc * args.train_ratio):]

	# define model
	set_rnd_seed(args)
	model = GAT(
	    nfeat=doc_emb.shape[1], 
	    nhid=int(doc_emb.shape[1] / args.nb_heads), 
	    nclass=len(np.unique(doc_spec)), 
	    dropout=args.dropout, 
	    drop_enc=args.drop_enc, 
	    alpha=args.alpha, 
	    nheads=args.nb_heads).to(dev)

	optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	scheduler = ReduceLROnPlateau(optimizer, factor=args.lr_factor, patience=args.lr_patience, verbose=True)

	# train and validation
	best_loss = 100.

	for epoch in range(1, args.epochs + 1):
	    train_loss = train(epoch, model, optimizer, args, doc_emb_train, doc_spec_train, doc_svc_train, svc_emb_ts)
	    test_loss, accu = test(epoch, model, optimizer, args, doc_emb_test, doc_spec_test, doc_svc_test, svc_emb_ts)
	    scheduler.step(test_loss)

	    if test_loss < best_loss:
	        best_loss = test_loss

	        if args.checkpoint:
	            torch.save({
	                'model_state_dict': model.state_dict(),
	                'optimizer_state_dict': optimizer.state_dict(),
	                'train_loss': train_loss,
	                'test_loss': test_loss,
	                'accu': accu,
	                'epoch': epoch
	                }, 'saved_models/doc_emb_best')

	get_doc_emb(args)

if __name__ == "__main__":
	args = parse_args()
	main(args)

