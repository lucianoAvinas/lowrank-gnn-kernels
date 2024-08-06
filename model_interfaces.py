import torch
import numpy as np
import torch.nn as nn
import scipy.sparse as sp

from argparse import Namespace
from acm_gnn.models import GCN
from gpr_gnn.models import GPRGNN
from jacobi_conv.models import Gmodel, Combination, Seq, TensorMod
from jacobi_conv import PolyConv
from np_gnn.models import NPGNN
from abc import ABC, abstractmethod
from functools import partial
from torch_geometric.utils.convert import to_scipy_sparse_matrix


class ModelInterface(ABC):
    @staticmethod
    @abstractmethod
    def get_param_opts():
        # accounts for multiple hyperparam configs when testing
        # different model variants 
        pass

    @staticmethod
    @abstractmethod
    def get_model_inputs(data):
        # conforms data to input signature expected by the model
        pass

    @staticmethod
    @abstractmethod
    def suggest_values(trial):
        # suggests hyperparameters for optuna trial
        pass


def normalize_tensor(mx, eqvar = None):
    """Row-normalize sparse matrix"""
    mx = sp.csr_matrix(mx)
    rowsum = np.array(mx.sum(1))
    if eqvar:
        r_inv = np.power(rowsum, -1.0/eqvar).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv, 0)
        mx = r_mat_inv.dot(mx)    
    else:
        r_inv = np.power(rowsum, -1.0).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv, 0)
        mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


class NPGNN_AB(ModelInterface, NPGNN):
    adj = None
    n_feats = None
    n_out = None
    def __init__(self, hyper_params = {'spec_train':True, 'kern_fn':None, 'norm':False, 'shift':False, 'pct':1, 'use_sqrt_K':False}):
        NPGNN.__init__(self, NPGNN_AB.adj, NPGNN_AB.n_feats, NPGNN_AB.n_out, **hyper_params)

    @staticmethod
    def get_param_opts():
        return dict()

    @staticmethod
    def get_model_inputs(data):
        NPGNN_AB.adj = data.get_adjacency_matrix().to_dense()
        NPGNN_AB.n_feats = data.x.shape[1]
        NPGNN_AB.n_out = len(data.y.unique())

        return [data.x]

    @staticmethod
    def suggest_values(trial):
        return dict()

class ACM_GCNP(ModelInterface, GCN):
    n = None
    d = None
    c = None
    def __init__(self, hyper_params):
        GCN.__init__(self, ACM_GCNP.d, 64, ACM_GCNP.c, None, 
                     ACM_GCNP.n, hyper_params['dropout'], model_type='acmgcnp', 
                     structure_info=True, variant=True)

    @staticmethod
    def get_param_opts():
        return dict()

    @staticmethod
    def get_model_inputs(data):
        edge_data, X, y = data.edge_index, data.x, data.y

        ACM_GCNP.n = len(y)
        ACM_GCNP.d = X.shape[1]
        ACM_GCNP.c = len(y.unique())

        device = y.device

        adj_low_unnormalized = to_scipy_sparse_matrix(edge_data.to('cpu'), num_nodes=ACM_GCNP.n)
        adj_low = normalize_tensor(sp.identity(ACM_GCNP.n) + adj_low_unnormalized)
        adj_high = sp.identity(ACM_GCNP.n) - adj_low
        
        adj_low = sparse_mx_to_torch_sparse_tensor(adj_low).to(device)
        adj_high = sparse_mx_to_torch_sparse_tensor(adj_high).to(device)
        adj_low_unnormalized = sparse_mx_to_torch_sparse_tensor(adj_low_unnormalized).to(device)

        return X, adj_low, adj_high, adj_low_unnormalized

    @staticmethod
    def suggest_values(trial):
        #nhid = trial.suggest_categorical('nhid', [])
        dropout = trial.suggest_float('dropout', 0, 0.9, step=0.1)

        return dict(dropout=dropout)


class GPR_GNN(ModelInterface, GPRGNN):
    data = None
    def __init__(self, hyper_params):
        args = Namespace(**hyper_params)
        args.ppnp = 'GPR_prop'
        args.Init = 'PPR'
        args.K = 10
        args.dropout = 0.5
        args.hidden = 64
        args.Gamma = None

        GPRGNN.__init__(self, GPR_GNN.data, args)

    @staticmethod
    def get_param_opts():
        return dict()

    @staticmethod
    def get_model_inputs(data):
        GPR_GNN.data = data
        GPR_GNN.data.num_features = data.x.shape[1]
        GPR_GNN.data.num_classes = len(data.y.unique())
        return (data,)

    @staticmethod
    def suggest_values(trial):
        alpha = trial.suggest_categorical('alpha', [0.1, 0.2, 0.5, 0.9])
        dprate = trial.suggest_float('dprate', 0, 0.9, step=0.1)

        return dict(alpha=alpha, dprate=dprate)


class Jacobi_Conv(ModelInterface, Gmodel):
    x = None
    output_channels = None
    def __init__(self, hyper_params):
        emb = Seq([
            TensorMod(Jacobi_Conv.x),
            nn.Dropout(p=hyper_params['dpb']),
            nn.Linear(Jacobi_Conv.x.shape[1], Jacobi_Conv.output_channels),
            nn.Dropout(hyper_params['dpt'], inplace=True)
        ])

        conv_fn = partial(PolyConv.JacobiConv, a=hyper_params['a'], b=hyper_params['b'])
        conv = PolyConv.PolyConvFrame(conv_fn, depth=10, aggr='gcn', 
                                      alpha=hyper_params['alpha'], fixed=False)
        comb = Combination(Jacobi_Conv.output_channels, 11, sole=False)
   

        Gmodel.__init__(self, emb, conv, comb)

    @staticmethod
    def get_param_opts():
        return dict()

    @staticmethod
    def get_model_inputs(data):
        Jacobi_Conv.x = data.x
        Jacobi_Conv.output_channels = len(data.y.unique())

        ei = data.edge_index
        ea = torch.ones(ei.shape[1]).to(data.y.device)
        msk = torch.ones_like(data.y, dtype=bool)

        return (ei, ea, msk)

    @staticmethod
    def suggest_values(trial):
        alpha = trial.suggest_float('alpha', 0.5, 2.0, step=0.5)
        a = trial.suggest_float('a', -1.0, 2.0, step=0.25)
        b = trial.suggest_float('b', -0.5, 2.0, step=0.25)
        dpb = trial.suggest_float("dpb", 0.0, 0.9, step=0.1)
        dpt = trial.suggest_float("dpt", 0.0, 0.9, step=0.1)

        return dict(alpha=alpha, a=a, b=b, dpb=dpb, dpt=dpt)
