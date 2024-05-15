import torch
import optuna
import argparse
import itertools
import numpy as np
import xarray as xr

from tqdm import tqdm
from pathlib import Path
from functools import reduce

from datasets import get_dataset, adjacency_svd
from optimize import evaluate_model, evaluate_params
from kernels import sobolev_cmpct, sobolev_reals, gaussian_rbf, linear_reals


DATA_NAMES = ['cora', 'citeseer', 'pubmed', 'chameleon', 'squirrel', 
              'actor', 'cornell', 'texas', 'wisconsin', 'csbm']

MASKS_OPTS = ['geom_gcn', 'random', 'balanced']

MODEL_OPTS = ['linear', 'adj', 'free', 'kernel']

GRAPH_OPTS = ['none', 'shift', 'norm', 'normshift']
GRAPH_DICT = dict(none=(False,False), shift=(False, True), norm=(True, False), normshift=(True, True))

KERNL_OPTS = ['sob_cmpct', 'sob_reals', 'gauss_rbf', 'lin_reals']
KERNL_DICT = dict(sob_cmpct=sobolev_cmpct, sob_reals=sobolev_reals, gauss_rbf=gaussian_rbf, 
                  lin_reals=linear_reals)


parser = argparse.ArgumentParser(description='Set up low-rank model sweep')

parser.add_argument('--csbm_params', type=float, nargs=3, 
                    help='Sets (p,q)-CSBM parameters (a,b,cls_sep) where p=a(log n)/n, q=b(log n)/n,'
                         'and cls_sep is mean separation between cluster features.')

parser.add_argument('--reduction_range', type=int, default=(0,19,20), nargs=3,
                    help='Expects three arguments non-negative integers, ex: "--reduction_range a b c", '
                         'with "a" <= "b" < "c". Roughly equivalent to "linspace(a/c, b/c, c)" where both '
                         'are included if c > 1. Each element in range determines a rank reduction to apply '
                         'on graph matrix.')

parser.add_argument('--datasets', type=str, default=DATA_NAMES, nargs='+',
                    help='Names of graph datasets to test and sweep over.')

parser.add_argument('--masks', type=str, default=MASKS_OPTS, nargs='+',
                    help='Different masking options to consider when training. "geom-gcn" uses the '
                         'original Geom-GCN masking splits.')

parser.add_argument('--models', type=str, default=MODEL_OPTS, nargs='+',
                    help='Different models to consider and test. Let X be features, W a weight matrix, and '
                         'G a graph matrix, then: "linear" = XW and "adj" = GXW. Next let (U,S,Vh) be a '
                         'spectral decomposition of G, K(S) a kernel matrix defined on S and "a" a learnable '
                         'parameter, then: "free" = (U diag(a) Vh)XW and "kernel" = U(Ka)Vh XW')

parser.add_argument('--graphs', type=str, default=GRAPH_OPTS, nargs='+',
                    help='Determines graph matrix. "none" corresponds to adjacency: A. "norm" corresponds to '
                         'normalized adjacency, where for D_ii = sum_j A_ij, we have: D^{-1/2} A D^{-1/2} for '
                         'undirected A and D^{-1} A for directed A. "shift" corresponds to graph Laplacian: '
                         'D - A. "normshift" corresponds to normalized graph Laplacian: I - D^{-1/2} A D^{-1/2} '
                         'for undirected A and I - D^{-1} A for directed A.')

parser.add_argument('--kernels', type=str, default=KERNL_OPTS, nargs='+',
                    help='Selects kernel function to use and define the spectral interaction matrix.')

parser.add_argument('--save_name', type=str, default='run_results',
                    help='Name to save sweep validation and test results under.')

parser.add_argument('--overwrite', action='store_true',
                    help='Overwrites results if "save_name" already exists. Else starts from last non-nan entry.')


def run_and_record(save_params, data, model_params, use_gamma):
    result_xr, opt_tuple, save_name = save_params
    if not np.isnan(result_xr.loc[opt_tuple]).all().item():
        return  # skip if entry already filled

    try:
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=1), direction='maximize',
                                    study_name=str(opt_tuple))
        study.optimize(evaluate_model(data, model_params, use_gamma), n_trials=100)

        results = evaluate_params(data, model_params, study.best_trial.params)

        # save current results
        result_xr.loc[opt_tuple] = np.array(results)
        result_xr.to_netcdf(save_name)
    except Exception as e:
        print(f'Error at {opt_tuple} with: {e}')


if __name__ == '__main__':
    args = parser.parse_args()

    rnk_st, rnk_en, rnk_ln = args.reduction_range
    rank_steps = list(range(rnk_st, rnk_en+1))
    all_options = (rank_steps, args.datasets, args.masks, args.models, args.graphs, args.kernels)

    save_name = Path('sweep_results') / f'{args.save_name}.nc'

    try:
        if args.overwrite:
            raise FileNotFoundError
        else:
            result_xr = xr.load_dataset(save_name)['__xarray_dataarray_variable__']

    except FileNotFoundError:
        result_xr = xr.DataArray(np.full([len(opt) for opt in all_options] + [2, 10], np.nan), 
                                 dims=('rank', 'data', 'mask', 'model', 'graph', 'kernel', 'result', 'split'),
                                 coords=dict(rank=rank_steps, data=args.datasets, mask=args.masks, model=args.models, 
                                             graph=args.graphs, kernel=args.kernels,result=['val', 'test'], 
                                             split=list(range(10))))
        
    last_opts = (None,) * len(all_options)
    N_total = reduce(lambda u,v: u * len(v), all_options, 1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for opt_tuple in tqdm(itertools.product(*all_options), total=N_total):
        i, data_nm, mask_type, model_type, graph_type, kern_nm = opt_tuple

        if data_nm == 'csbm':
            if args.csbm_params is None:
                continue
            else:
                a,b,cls_sep = args.csbm_params
                data_nm = f'{data_nm}_{a}_{b}_{cls_sep}'

        pct = 1.0 - i/rnk_ln
        g_norm, g_shift = GRAPH_DICT[graph_type]

        new_data = last_opts[1] != data_nm or last_opts[2] != mask_type
        if new_data:    
            data = get_dataset(data_nm, mask_type).to(device)

            X,y = data.x, data.y
            n_feats = X.shape[1]
            n_out = len(y.unique())
            data_mask = (data.train_mask.T, data.val_mask.T, data.test_mask.T)

        new_graph = last_opts[0] != i or last_opts[1] != data_nm or last_opts[4] != graph_type
        if new_graph:
            A, (U,M,Vh) = adjacency_svd((data_nm, data.edge_index), g_norm, g_shift, pct)

        save_params = (result_xr, opt_tuple, save_name)

        if model_type == 'linear':
            if i == rank_steps[0] and graph_type == args.graphs[0] and kern_nm == args.kernels[0]:
                run_and_record(save_params, (X, y, data_mask, None), (n_feats, n_out, None, None), False)
        elif model_type == 'adj':
            if kern_nm == args.kernels[0]:
                run_and_record(save_params, (A @ X, y, data_mask, None), (n_feats, n_out, None, None), False)
        elif model_type == 'free':
            if kern_nm == args.kernels[0]:
                run_and_record(save_params, (X, y, data_mask, None), (n_feats, n_out, U, Vh), False)
        else:
            new_kernel = last_opts[5] != kern_nm or last_opts[3] != 'kern' or new_graph
            if new_kernel:
                kern_fn = KERNL_DICT[kern_nm]

                if kern_nm == 'sob_cmpct':
                    kern = kern_fn(M.min().item(), M.max().item())
                    use_gamma = False
                elif kern_nm == 'lin_reals':
                    kern = kern_fn()
                    use_gamma = False
                else:
                    kern = kern_fn(1)
                    use_gamma = True

                K = kern(M, M)

            run_and_record(save_params, (X, y, data_mask, K), (n_feats, n_out, U, Vh), use_gamma)

        last_opts = opt_tuple
