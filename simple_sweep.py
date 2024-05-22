import torch
import numpy as np
from torch_geometric.data import Data
from sklearn.datasets import make_blobs
from extensions import ExtendedData
from tqdm import tqdm
import optuna
from optuna.trial import TrialState
import json

from nett import sample_dcsbm
from model_interfaces import ACM_GCNP, NPGNN_AB
from basics import train_model_class
from kernels import sobolev_reals, gaussian_rbf

torch.manual_seed(123)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_hyper_params(model_class, trial):
    if model_class == NPGNN_AB:
        kern_type = trial.suggest_categorical('kern_type', ['sobolev', 'gaussian'])
        gamma = trial.suggest_float('gamma', 0.2, 1.5)
        kern_fn = sobolev_reals(gamma) if kern_type == 'sobolev' else gaussian_rbf(gamma)
        hyper_params = {
            'spec_train': True,
            'kern_fn': kern_fn,
            'norm': False,
            'shift': False,
            'pct': 1,
            'use_sqrt_K': False
        }
    elif model_class == ACM_GCNP:
        dropout = trial.suggest_float('dropout', 0, 0.75)
        hyper_params = {'dropout': dropout}
    return hyper_params

def train_model_with_reps(model_class, hyper_params, data, n_rep, n_iter=1500, lr=1e-2, wd=5e-4):
    val_results = []
    tst_results = []
    for rep in range(n_rep):
        best_acc, test_acc, _, _ = train_model_class(model_class, hyper_params, data, rep, n_iter, lr, wd)
        val_results.append(best_acc)
        tst_results.append(test_acc)
    mean_val = np.mean(val_results)
    std_val = np.std(val_results)
    mean_tst = np.mean(tst_results)
    std_tst = np.std(tst_results)
    return mean_val, std_val, mean_tst, std_tst

def objective(trial, model_class, data, n_iter, lr, wd):
    hyper_params = get_hyper_params(model_class, trial)
    mean_val, _, _, _ = train_model_with_reps(model_class, hyper_params, data, n_rep=1, n_iter=n_iter, lr=lr, wd=wd)
    return mean_val

n_values = [300, 600, 1200, 1500]
lr = 1e-2
wd = 5e-4
n_iter = 1500
final_n_rep = 10
results = {}

for n in n_values:
    print(f'\n---\nSimulations for n = {n}\n---\n')

    p = 0.015
    q = 0.002
    K = 3
    B = torch.full((K, K), q)
    for i in range(K):
        B[i, i] = p

    X, y, centers = make_blobs(
        n_samples=n,
        n_features=K,
        centers=K,
        cluster_std=10,
        random_state=42,
        return_centers=True
    )

    data = Data(edge_index=sample_dcsbm(y, B), x=torch.tensor(X).float(), y=torch.tensor(y).long())
    data = ExtendedData.from_dict(data.to_dict())
    data.create_masks('balanced')

    models = [{'model_class': NPGNN_AB}, {'model_class': ACM_GCNP}]
    
    results[n] = {}

    for model in models:
        model_class = model['model_class']
        study = optuna.create_study(direction='maximize')
        try:
            study.optimize(lambda trial: objective(trial, model_class, data, n_iter, lr, wd), n_trials=50)
        except Exception as e:
            print(f"Error during optimization for {model_class.__name__}")
            print(f"Exception: {e}")
            continue

        pruned_trials = [t for t in study.trials if t.state == TrialState.PRUNED]
        complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]

        print(f"Study statistics: ")
        print(f"  Number of finished trials: {len(study.trials)}")
        print(f"  Number of pruned trials: {len(pruned_trials)}")
        print(f"  Number of complete trials: {len(complete_trials)}")

        print(f"Best trial for {model_class.__name__}:")
        trial = study.best_trial

        print(f"  Value: {trial.value}")
        print(f"  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

        best_hyper_params = get_hyper_params(model_class, trial)
        mean_val, std_val, mean_tst, std_tst = train_model_with_reps(
            model_class, best_hyper_params, data, n_rep=final_n_rep, n_iter=n_iter, lr=lr, wd=wd
        )

        results[n][model_class.__name__] = {
            'best_hyper_params': trial.params,
            'mean_val': mean_val,
            'std_val': std_val,
            'mean_tst': mean_tst,
            'std_tst': std_tst
        }

        print(f"\nBest {model_class.__name__} - val = {mean_val:.3f} ± {std_val:.3f}, tst = {mean_tst:.3f} ± {std_tst:.3f}")

# Save results to a JSON file
with open('results2.json', 'w') as f:
    json.dump(results, f, indent=4)

print("\nResults saved to 'results.json'")
