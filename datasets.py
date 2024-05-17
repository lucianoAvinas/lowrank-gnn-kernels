import os
import torch
import torch_geometric

from math import log
from pathlib import Path


def get_dataset(dataset_nm, mask_type='geom_gcn'):
    mask_type = mask_type.lower()
    dataset_nm = dataset_nm.lower()

    if dataset_nm in ['cora', 'citeseer', 'pubmed']:
        data = torch_geometric.datasets.Planetoid('graph_data', dataset_nm, 'geom-gcn')[0]
    elif dataset_nm in ['chameleon', 'squirrel']:
        data = torch_geometric.datasets.WikipediaNetwork('graph_data', dataset_nm)[0]
    elif dataset_nm in ['actor']:
        data = torch_geometric.datasets.Actor('graph_data')[0]
    elif dataset_nm in ['cornell', 'texas', 'wisconsin']:
        data = torch_geometric.datasets.WebKB('graph_data', dataset_nm)[0]
        data.edge_index = torch_geometric.utils.to_undirected(data.edge_index)
    elif 'csbm' in dataset_nm:
        if mask_type == 'geom_gcn':
            raise ValueError('CSBM data does not have pre-defined split.')

        _, a, b, cls_sep = dataset_nm.split('_')
        a,b,cls_sep = float(a), float(b), float(cls_sep)

        torch.manual_seed(47)
        n = 1000
        p,q = a*log(n)/n, b*log(n)/n

        d = 1
        x = torch.stack((cls_sep+torch.randn(2*n,d), -cls_sep+torch.randn(2*n,d)))
        
        data = torch_geometric.datasets.StochasticBlockModelDataset(os.path.join('graph_data', dataset_nm), 
                                                                    [n,n], [[p, q],[q, p]], 
                                                                    num_channels=1, n_clusters_per_class=1,
                                                                    class_sep=cls_sep)[0]
        data.x = x[data.y, torch.arange(2*n)]
    else:
        raise NotImplementedError(f'Dataset {dataset_nm} not yet implemented')

    n = len(data.y)
    torch.manual_seed(10)

    if mask_type == 'random':
        data.train_mask, data.val_mask, data.test_mask = torch.zeros((3,n,10), dtype=bool)
        for i in range(10):
            inds = torch.randperm(n)
            data.train_mask[inds[:int(0.6*n)],i] = True
            data.val_mask[inds[int(0.6*n):int(0.8*n)],i] = True
            data.test_mask[inds[int(0.8*n):],i] = True

    elif mask_type == 'balanced':
        C = len(data.y.unique())
        bnd = int(n * 0.6 / C)
        all_inds = [(data.y == c).nonzero() for c in range(C)]
        data.train_mask, data.val_mask, data.test_mask = torch.zeros((3,n,10), dtype=bool)

        for i in range(10):
            eval_inds = []
            for c in range(C):
                cls_inds = all_inds[c]
                cls_inds = cls_inds[torch.randperm(cls_inds.shape[0])]
                data.train_mask[cls_inds[:bnd],i] = True
                eval_inds.append(cls_inds[bnd:])
                
            eval_inds = torch.cat(eval_inds)
            eval_inds = eval_inds[torch.randperm(eval_inds.shape[0])]

            data.val_mask[eval_inds[:int(n*0.2)],i] = True
            data.test_mask[eval_inds[int(n*0.2):],i] = True

    else:
        # mask_type == 'geom_gcn'
        data.train_mask = data.train_mask.bool()
        data.val_mask = data.val_mask.bool()
        data.test_mask = data.test_mask.bool()    

    return data


def normalize_adjacency(A, D, is_symm):
    mask = (D != 0)
    Dinv = torch.ones_like(D)

    if is_symm:
        Dinv[mask] = 1/torch.sqrt(D[mask])
        A = Dinv[:,None] * (A * Dinv[None])
    else:
        Dinv[mask] = 1/D[mask]
        A = Dinv[:,None] * A 

    return A


def spectral_decomp(A, data_nm, norm, shift, is_symm):
    device = A.device
    spec_path = Path('spectral_data')

    spec_path = spec_path / f'{data_nm}{"_symm" if norm else ""}{"_shift" if shift else ""}.pt'

    if is_symm:
        try:
            eigh_dict = torch.load(spec_path)
            M, U = eigh_dict['M'].to(device), eigh_dict['U'].to(device)
        except FileNotFoundError:
            M, U = torch.linalg.eigh(A)
            torch.save(dict(M=M.cpu(), U=U.cpu()), spec_path)
        Vh = U.T
    else:
        try:
            svd_dict = torch.load(spec_path)
            U, M, Vh = svd_dict['U'].to(device), svd_dict['M'].to(device), svd_dict['Vh'].to(device)
        except FileNotFoundError:
            U, M, Vh = torch.linalg.svd(A)  
            torch.save(dict(U=U.cpu(), M=M.cpu(), Vh=Vh.cpu()), spec_path)
    return U, M, Vh


def adjacency_svd(edge_data, norm, shift, pct):
    data_nm, edges = edge_data
    A = torch_geometric.utils.to_dense_adj(edges).squeeze()

    D = A.sum(1)
    is_symm = torch.all(A == A.T)

    if norm:
        A = normalize_adjacency(A, D, is_symm)
        A = torch.eye(A.shape[0]).to(A.device) - A if shift else A
    else:
        A = torch.diag(D) - A if shift else A

    U, M, Vh = spectral_decomp(A, data_nm, norm, shift, is_symm)

    eig_qt = torch.quantile(abs(M), 1 - pct)
    eig_mask = (abs(M) >= eig_qt)

    if eig_mask.mean(dtype=float).item() < 1:
        U, M, Vh = U[:,eig_mask], M[eig_mask], Vh[eig_mask]
        A = U @ torch.diag(M) @ Vh

    return A, (U, M, Vh)
