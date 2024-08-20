import torch
import optuna
import argparse
import itertools
import numpy as np
import xarray as xr
import torch.nn.functional as F

from tqdm import tqdm
from pathlib import Path
from functools import reduce

from statistics import mean
from optuna.trial import TrialState

from datasets import get_dataset

from model_interfaces import ACM_GCNP, GPR_GNN, Jacobi_Conv


DATA_NAMES = ['cora', 'citeseer', 'pubmed', 'chameleon', 'squirrel', 
              'actor', 'cornell', 'texas', 'wisconsin', 'csbm', 'rdpg']

MASKS_OPTS = ['geom_gcn', 'random', 'balanced']

MODEL_OPTS = ['acmgcnp', 'jacobi', 'gprgnn']

MODEL_DICT = dict(acmgcnp=ACM_GCNP, jacobi=Jacobi_Conv, gprgnn=GPR_GNN)


parser = argparse.ArgumentParser(description='Set up hyperparmeter sweep for other models')

parser.add_argument('--csbm_params', type=float, nargs=3, 
                    help='Sets (p,q)-CSBM parameters (a,b,cls_sep) where p=a(log n)/n, q=b(log n)/n,'
                         'and cls_sep is mean separation between cluster features.')

parser.add_argument('--datasets', type=str, default=DATA_NAMES, nargs='+',
                    help='Names of graph datasets to test and sweep over.')

parser.add_argument('--masks', type=str, default=MASKS_OPTS, nargs='+',
                    help='Different masking options to consider when training. "geom-gcn" uses the '
                         'original Geom-GCN masking splits.')

parser.add_argument('--model', type=str, choices=MODEL_OPTS, help='')

parser.add_argument('--save_name', type=str, default='run_results',
                    help='Name to save sweep validation and test results under.')

parser.add_argument('--overwrite', action='store_true',
                    help='Overwrites results if "save_name" already exists. Else starts from last non-nan entry.')


def accuracy(pred, label):
    return (pred.argmax(axis=1) == label).mean(dtype=float)


def evaluate_params(data, model_cls, hyper_params, trial=None):
    inp, y, masks = data
    device = y.device

    hyper_params = dict(hyper_params)
    lr = hyper_params.pop('lr')
    wd = hyper_params.pop('wd')

    val_results = []
    tst_results = []

    for rep, (tr_msk, vl_msk, tst_msk) in enumerate(zip(*masks)):
        torch.manual_seed(rep+1)
        model = model_cls(hyper_params).to(device)

        y_tr, y_vl, y_tst = y[tr_msk], y[vl_msk], y[tst_msk]

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        early_it, best_acc, test_acc = 0, 0, 0
        for i in range(1500):
            optimizer.zero_grad()
            model.train()

            out = model(*inp)
            F.cross_entropy(out[tr_msk], y_tr).backward()
            optimizer.step()

            with torch.no_grad():
                model.eval()
                out = model(*inp)

                val_acc = accuracy(out[vl_msk], y_vl).item()

                if val_acc >= best_acc:
                    early_it = 0
                    best_acc = val_acc
                    test_acc = accuracy(out[tst_msk], y_tst).item() if trial is None else None
                else:
                    early_it += 1

            if early_it == 200:
                break

        val_results.append(best_acc)
        tst_results.append(test_acc)

        if trial is not None:
            trial.report(mean(val_results), rep)
            if trial.should_prune():
                raise optuna.TrialPruned()

    if trial is None:
        res = (val_results, tst_results)
    else:
        res = mean(val_results)        

    return res


def evaluate_model(data, model_cls):
    def trial_runner(trial):
        # (lr,wd): has complicated multi-modal relationship w/ Adam
        # use categorical to uncouple (lr,wd) relation from optuna numerical modeling
        lr = trial.suggest_categorical('lr', [5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1e-0])
        wd = trial.suggest_categorical('wd', [0.0, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1])

        hyper_params = dict(lr=lr, wd=wd) | model_cls.suggest_values(trial)  ###

        for previous_trial in trial.study.trials:
            if previous_trial.state == TrialState.COMPLETE and trial.params == previous_trial.params:
                print(f'DUPLICATED TRIAL: {trial.params}')
                return previous_trial.value    

        return evaluate_params(data, model_cls, hyper_params, trial)
    return trial_runner


def run_and_record(save_params, data, model_cls):
    result_xr, opt_tuple, save_name = save_params
    if not np.isnan(result_xr.loc[opt_tuple]).all().item():
        return  # skip if entry already filled

    try:
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=1), direction='maximize',
                                    study_name=str(opt_tuple))
        study.optimize(evaluate_model(data, model_cls), n_trials=100)

        results = evaluate_params(data, model_cls, study.best_trial.params)

        # save current results
        result_xr.loc[opt_tuple] = np.array(results)
        result_xr.to_netcdf(save_name)
    except Exception as e:
        print(f'Error at {opt_tuple} with: {e}')

if __name__ == '__main__':
    args = parser.parse_args()

    try:
        model_cls = MODEL_DICT[args.model]
    except KeyError:
        raise NotImplementedError(f'Interface not yet implemented for {args.model}')
        

    option_dict = dict(data=args.datasets, mask=args.masks)

    option_dict = option_dict | model_cls.get_param_opts() ####

    save_name = Path('sweep_results') / f'{args.save_name}.nc'

    try:
        if args.overwrite:
            raise FileNotFoundError
        else:
            result_xr = xr.load_dataset(save_name)['__xarray_dataarray_variable__']

    except FileNotFoundError:
        super_dict = option_dict | dict(result=['val', 'test'], split=list(range(10)))
        result_xr = xr.DataArray(np.full([len(opt) for opt in option_dict.values()] + [2, 10], np.nan), 
                                 dims=tuple(super_dict.keys()), coords=super_dict)
    
    last_opts = (None,) * len(option_dict.values())
    N_total = reduce(lambda u,v: u * len(v), option_dict.values(), 1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for opt_tuple in tqdm(itertools.product(*option_dict.values()), total=N_total):    
        data_nm, mask_type = opt_tuple[0], opt_tuple[1]
        if data_nm == 'csbm':
            if args.csbm_params is None:
                continue
            else:
                a,b,cls_sep = args.csbm_params
                data_nm = f'{data_nm}_{a}_{b}_{cls_sep}'

        new_data = last_opts[0] != data_nm or last_opts[1] != mask_type
        if new_data:    
            data = get_dataset(data_nm, mask_type).to(device)

            data_mask = (data.train_mask.T, data.val_mask.T, data.test_mask.T)


        save_params = (result_xr, opt_tuple, save_name)
        inp = model_cls.get_model_inputs(data)   ####

        run_and_record(save_params, (inp, data.y, data_mask), model_cls)

        last_opts = opt_tuple
