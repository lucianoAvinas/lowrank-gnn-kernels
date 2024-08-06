import torch
import numpy as np
from torch_geometric.data import Data
from extensions import ExtendedData
from tqdm import tqdm
import optuna
from optuna.trial import TrialState
import json
import networkx as nx
import time
import os

from model_interfaces import ACM_GCNP, NPGNN_AB
from basics import train_model_class
from kernels import sobolev_reals, gaussian_rbf
from torch_geometric.datasets import LINKXDataset

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
    total_time = 0
    for rep in range(n_rep):
        start_time = time.time()
        best_acc, test_acc, _, _ = train_model_class(model_class, hyper_params, data, rep, n_iter, lr, wd)
        end_time = time.time()
        total_time += end_time - start_time
        val_results.append(best_acc)
        tst_results.append(test_acc)
    mean_val = np.mean(val_results)
    std_val = np.std(val_results)
    mean_tst = np.mean(tst_results)
    std_tst = np.std(tst_results)
    avg_time = total_time / n_rep
    return mean_val, std_val, mean_tst, std_tst, avg_time

def objective(trial, model_class, data, n_iter, lr, wd):
    hyper_params = get_hyper_params(model_class, trial)
    mean_val, _, _, _, _ = train_model_with_reps(model_class, hyper_params, data, n_rep=1, n_iter=n_iter, lr=lr, wd=wd)
    return mean_val

def subsample_graph(data, num_nodes):
    G = nx.Graph()
    G.add_edges_from(data.edge_index.t().tolist())
    subG = nx.Graph(G.subgraph(np.random.choice(G.nodes(), num_nodes, replace=False)))
    
    node_map = {old: new for new, old in enumerate(subG.nodes())}
    edge_index = torch.tensor([[node_map[u], node_map[v]] for u, v in subG.edges()]).t()
    
    x = data.x[list(subG.nodes())]
    y = data.y[list(subG.nodes())]
    
    return Data(x=x, edge_index=edge_index, y=y)

def run_experiment(dataset_name, n, lr, wd, n_iter, final_n_rep):
    # Load LINKX dataset
    dataset = LINKXDataset(root='./data', name=dataset_name)
    full_data = dataset[0]

    # Subsample the graph
    data = subsample_graph(full_data, n)

    # Create masks for train/val/test split
    data = ExtendedData.from_dict(data.to_dict())
    data.create_masks('balanced')

    # Move data to device
    data = data.to(device)

    results = {}
    models = [{'model_class': NPGNN_AB}, {'model_class': ACM_GCNP}]

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
        mean_val, std_val, mean_tst, std_tst, avg_time = train_model_with_reps(
            model_class, best_hyper_params, data, n_rep=final_n_rep, n_iter=n_iter, lr=lr, wd=wd
        )

        results[model_class.__name__] = {
            'best_hyper_params': trial.params,
            'mean_val': mean_val,
            'std_val': std_val,
            'mean_tst': mean_tst,
            'std_tst': std_tst,
            'avg_training_time': avg_time
        }

        print(f"\nBest {model_class.__name__} - val = {mean_val:.3f} ± {std_val:.3f}, tst = {mean_tst:.3f} ± {std_tst:.3f}, time = {avg_time:.2f} s")

    return results

if __name__ == '__main__':
    lr = 1e-2
    wd = 5e-4
    n_iter = 1500
    final_n_rep = 10
    
    datasets = ['genius', 'penn94', 'cornell5','johnshopkins55']  # Add or remove datasets as needed
    n_values = [2000, 5000]  # Add or remove sample sizes as needed

    results_file = 'linkx_results.json'

    # Load existing results if file exists
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            all_results = json.load(f)
    else:
        all_results = {}

    for dataset_name in datasets:
        if dataset_name not in all_results:
            all_results[dataset_name] = {}
        
        for n in n_values:
            if str(n) not in all_results[dataset_name]:
                print(f"\nRunning experiment for {dataset_name} with n = {n}")
                results = run_experiment(dataset_name, n, lr, wd, n_iter, final_n_rep)
                all_results[dataset_name][str(n)] = results
                
                # Save results after each experiment
                with open(results_file, 'w') as f:
                    json.dump(all_results, f, indent=4)
                
                print(f"\nResults for {dataset_name} with n = {n} saved to '{results_file}'")
            else:
                print(f"\nExperiment for {dataset_name} with n = {n} already exists in results file. Skipping.")

    print(f"\nAll experiments completed. Results saved to '{results_file}'")