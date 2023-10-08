import torch
import optuna
import torch.nn as nn
import numpy as np

from tqdm import tqdm

from datasets import get_dataset

import torch.nn.functional as F

from statistics import mean
from models import LinearGNN
from optuna.trial import TrialState


class MLP2(nn.Module):
    def __init__(self, n_feats, n_out, n_hidden):
        super().__init__()
        self.W_in = nn.Linear(n_feats, n_hidden)
        self.W_out = nn.Linear(n_hidden, n_out)

    def forward(self, X):
        return self.W_out(F.relu(self.W_in(X)))


def accuracy(pred, label):
    return (pred.argmax(axis=1) == label).mean(dtype=float)


def evaluate_model(data, model_params):
    def trial_runner(trial):
        # (lr,wd): has complicated multi-modal relationship w/ Adam
        # use categorical to uncouple (lr,wd) relation from optuna numerical modeling
        lr = trial.suggest_categorical('lr', [5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1e-0])
        wd = trial.suggest_categorical('wd', [0.0, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1])
        n_hidden = trial.suggest_categorical('nh', [16, 32, 64, 128, 256, 512])
        hyper_params = dict(lr=lr, wd=wd, nh=n_hidden)

        for previous_trial in trial.study.trials:
            if previous_trial.state == TrialState.COMPLETE and trial.params == previous_trial.params:
                print(f'DUPLICATED TRIAL: {trial.params}')
                return previous_trial.value    

        return evaluate_params(data, model_params, hyper_params, trial)
    return trial_runner


def evaluate_params(data, model_params, hyper_params, trial=None):
    X, y, masks, K = data
    device = X.device

    lr, wd, nh = hyper_params['lr'], hyper_params['wd'], hyper_params['nh']

    val_results = []
    tst_results = []

    for rep, (tr_msk, vl_msk, tst_msk) in enumerate(zip(*masks)):
        torch.manual_seed(rep+1)
        model = MLP2(model_params[0], model_params[1], nh).to(device)

        y_tr, y_vl, y_tst = y[tr_msk], y[vl_msk], y[tst_msk]

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        early_it, best_acc, test_acc = 0, 0, 0
        for i in range(1500):
            optimizer.zero_grad()
            model.train()

            out = model(X)
            F.cross_entropy(out[tr_msk], y_tr).backward()
            optimizer.step()

            with torch.no_grad():
                model.eval()
                out = model(X)

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


if __name__ == '__main__':
    DATA_NAMES = ['cora', 'citeseer', 'pubmed', 'chameleon', 'squirrel', 
                  'actor', 'cornell', 'texas', 'wisconsin']

    results = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
    for data_nm in tqdm(DATA_NAMES):
        data = get_dataset(data_nm, 'balanced').to(device)
        X,y = data.x, data.y
        n_feats = X.shape[1]
        n_out = len(y.unique())
        data_mask = (data.train_mask.T, data.val_mask.T, data.test_mask.T)

        data = (X, y, data_mask, None)
        model_params = (n_feats, n_out)

        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=1), direction='maximize',
                                    study_name='tmp_run')
        study.optimize(evaluate_model(data, model_params), n_trials=100)

        results.append(evaluate_params(data, model_params, study.best_trial.params))

    results = np.array(results)
    np.save('../mlp_run.npy', results)
