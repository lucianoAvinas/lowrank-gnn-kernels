import torch
import argparse
import torch_geometric
import torch.nn.functional as F

from optimize import accuracy
from datasets import get_dataset
from statistics import mean, stdev

parser = argparse.ArgumentParser(description='Model and data selection for evaluation comparison.')

parser.add_argument('--model', default='linear', choices=['linear', 'adj'], help='Selects model XW '
                    '(linear) or AXW (adj).')
parser.add_argument('--dataset', default='cora', choices=['cora', 'citeseer', 'pubmed', 'chameleon', 
                    'squirrel', 'actor', 'cornell', 'texas', 'wisconsin'], 
                    help='Select dataset to evaluate.')
parser.add_argument('--split', default='sparse', choices=['sparse', 'public', 'geom_gcn', 'balanced'],
                    help='Select dataset split to evaluate.')


if __name__ == '__main__':
    args = parser.parse_args()

    split = args.split
    dataset_nm = args.dataset

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if dataset_nm in ['cora', 'citeseer', 'pubmed'] and split in ['sparse', 'public']:
        data = torch_geometric.datasets.Planetoid('graph_data', dataset_nm, 'public')[0]
        masks = (data.train_mask.bool()[None], data.val_mask.bool()[None], 
                 data.test_mask.bool()[None])
    else:
        assert split not in ['sparse', 'public']
        data = get_dataset(dataset_nm, split)
        masks = (data.train_mask.bool().T, data.val_mask.bool().T, 
                 data.test_mask.bool().T)

    data = data.to(device)
    X,y = data.x, data.y

    if args.model == 'adj':
        A = torch_geometric.utils.to_dense_adj(data.edge_index).squeeze()
        X = A @ X

    tst_results = []
    for rep, (tr_msk, vl_msk, tst_msk) in enumerate(zip(*masks)):
        torch.manual_seed(rep+1)
        model = torch.nn.Linear(X.shape[1], len(y.unique())).to(device)

        X_tr, X_vl, X_tst = X[tr_msk], X[vl_msk], X[tst_msk]
        y_tr, y_vl, y_tst = y[tr_msk], y[vl_msk], y[tst_msk]

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)

        early_it, best_acc, test_acc = 0, 0, 0
        for i in range(1500):
            optimizer.zero_grad()

            out = model(X_tr)
            F.cross_entropy(out, y_tr).backward()
            optimizer.step()

            if split != 'sparse':
                with torch.no_grad():
                    out = model(X_vl)

                    val_acc = accuracy(out, y_vl).item()

                    if val_acc >= best_acc:
                        early_it = 0
                        best_acc = val_acc
                        test_acc = accuracy(model(X_tst), y_tst).item()
                    else:
                        early_it += 1
            else:
                with torch.no_grad():
                    out = model(X_tst)
                    test_acc = accuracy(out, y_tst).item()

            if early_it == 200:
                break

        tst_results.append(test_acc)
    std_dev = 0.0 if len(tst_results) == 1 else stdev(tst_results)
    print(f'Final Accuracy: {mean(tst_results):.4f} +/- {std_dev:.4f}')
