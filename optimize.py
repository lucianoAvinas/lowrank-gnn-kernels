import torch
import optuna
import torch.nn.functional as F

from statistics import mean
from models import LinearGNN
from optuna.trial import TrialState


def accuracy(pred, label):
    return (pred.argmax(axis=1) == label).mean(dtype=float)


def evaluate_model(data, model_params, use_gamma):
    def trial_runner(trial):
        # (lr,wd): has complicated multi-modal relationship w/ Adam
        # use categorical to uncouple (lr,wd) relation from optuna numerical modeling
        lr = trial.suggest_categorical('lr', [5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1e-0])
        wd = trial.suggest_categorical('wd', [0.0, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1])
        hyper_params = dict(lr=lr, wd=wd)

        if use_gamma:  # bandwidth hyperparameter for non-compact eigen-domains
            gamma = trial.suggest_float('gamma', -2, 2, step=0.5)
            hyper_params['gamma'] = gamma

        for previous_trial in trial.study.trials:
            if previous_trial.state == TrialState.COMPLETE and trial.params == previous_trial.params:
                print(f'DUPLICATED TRIAL: {trial.params}')
                return previous_trial.value    

        return evaluate_params(data, model_params, hyper_params, trial)
    return trial_runner


def evaluate_params(data, model_params, hyper_params, trial=None):
    X, y, masks, K = data
    device = X.device

    lr, wd = hyper_params['lr'], hyper_params['wd']
    if 'gamma' in hyper_params:
        gamma = 10**hyper_params['gamma']
        K_sc = K**gamma
    else:
        K_sc = K

    val_results = []
    tst_results = []

    for rep, (tr_msk, vl_msk, tst_msk) in enumerate(zip(*masks)):
        torch.manual_seed(rep+1)
        model = LinearGNN(*model_params).to(device)

        y_tr, y_vl, y_tst = y[tr_msk], y[vl_msk], y[tst_msk]

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        early_it, best_acc, test_acc = 0, 0, 0
        for i in range(1500):
            optimizer.zero_grad()
            model.train()

            out = model(X, K_sc)
            F.cross_entropy(out[tr_msk], y_tr).backward()
            optimizer.step()

            with torch.no_grad():
                model.eval()
                out = model(X, K_sc)

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
