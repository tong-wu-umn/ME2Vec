import numpy as np
import networkx as nx
from stellargraph.data import UniformRandomMetaPathWalk
from stellargraph import StellarGraph
from gensim.models import Word2Vec
from src.utils import PickleUtils

def parse_args():
    '''
    Parses the node2vec arguments.
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', nargs='?', default='graph/ppd_eICU.edgelist',
                        help='Input graph path')

    parser.add_argument('--output', nargs='?', default='emb/ppd_eICU.emd',
                        help='Embeddings path')

    parser.add_argument('--dimensions', type=int, default=128,
                        help='Number of dimensions. Default is 128.')

    parser.add_argument('--walk-length', type=int, default=100,
                        help='Length of walk per source. Default is 80.')

    parser.add_argument('--num-walks', type=int, default=10,
                        help='Number of walks per source. Default is 10.')

    parser.add_argument('--window-size', type=int, default=20,
                        help='Context size for optimization. Default is 10.')

    parser.add_argument('--iter', default=150, type=int,
                      help='Number of epochs in SGD')

    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers. Default is 8.')

    return parser.parse_args()

def main(args):

    # load stellar graph
    G = PickleUtils.loader(args.input)

    # generate walks
    rw = UniformRandomMetaPathWalk(graph=G)

    walks = rw.run(
        nodes=list(G.nodes()),  # root nodes
        length=args.walk_length,  # maximum length of a random walk
        n=args.num_walks,  # number of random walks per root node
        metapaths=[['patient','service','specialty','service','patient']]
    )

    model = Word2Vec(
        walks, 
        size=args.dimensions,
        window=args.window_size,
        min_count=0,
        sg=1,
        iter=args.iter,
        workers=args.workers)

    model.wv.save_word2vec_format(args.output)

if __name__ == "__main__":
    args = parse_args()
    main(args)

