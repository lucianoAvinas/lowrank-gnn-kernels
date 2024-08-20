import torch
import argparse
import torch_geometric

from datasets import get_dataset, sparse_svd


DATA_NAMES = ['cora', 'citeseer', 'pubmed', 'chameleon', 'squirrel', 
              'actor', 'cornell', 'texas', 'wisconsin', 'penn94', 'reed98', 
              'amherst41', 'cornell5', 'johnshopkins55', 'genius', 'csbm']
GRAPH_DICT = dict(none=(False,False), shift=(False, True), norm=(True, False), normshift=(True, True))

parser = argparse.ArgumentParser(description='Test low rank svd')


parser.add_argument('--n_vecs', type=int, default=1000, 
                    help='Number of eigenvectors to keep in reduction.')

parser.add_argument('--max_iter_power', type=int, default=6, 
                    help='Defines max iteration as 2**"max_iter"')

parser.add_argument('--graphs', type=str, default='none',
                    help='Determines graph matrix. "none" corresponds to adjacency: A. "norm" corresponds to '
                         'normalized adjacency, where for D_ii = sum_j A_ij, we have: D^{-1/2} A D^{-1/2} for '
                         'undirected A and D^{-1} A for directed A. "shift" corresponds to graph Laplacian: '
                         'D - A. "normshift" corresponds to normalized graph Laplacian: I - D^{-1/2} A D^{-1/2} '
                         'for undirected A and I - D^{-1} A for directed A.')

parser.add_argument('--datasets', type=str, default=DATA_NAMES, 
                    help='Names of graph datasets to test and sweep over')


if __name__ == '__main__':
    args = parser.parse_args()

    data = get_dataset(args.datasets, 'balanced')
    #A = torch_geometric.utils.to_torch_sparse_tensor(data.edge_index).coalesce()
    #print(torch.equal(A.indices(), A.T.coalesce().indices()))

    n = data.edge_index.max() + 1
    #D = torch.sparse.spdiags(torch.mv(A, torch.ones(n)), torch.zeros(1, dtype=int), A.shape)
    g_norm, g_shift = GRAPH_DICT[args.graphs]

    S_prev = torch.inf
    for m in range(1, args.max_iter_power+1):
        with torch.no_grad():
            S = sparse_svd((args.datasets, data.edge_index), g_norm, g_shift, args.n_vecs, 2**m)[1]
            print(f'niters {2**m}, l1 err:', torch.linalg.norm(S - S_prev, ord=1))
            S_prev = S
