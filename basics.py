import torch
import torch.nn.functional as F
import torch_geometric

def get_data(dataset_nm, mask_type='geom_gcn'):
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
    else:
        raise NotImplementedError(f'Dataset {dataset_nm} not yet implemented')
    
    return data


def accuracy(pred, label):
    return (pred.argmax(axis=1) == label).mean(dtype=float).item()

def train(model, data, rep, n_iter=1500, lr=1e-2, wd=5e-4):
    device = data.get_device()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    tr_msk = data.masks['train'][:, rep]
    vl_msk = data.masks['val'][:, rep]
    tst_msk = data.masks['test'][:, rep]
    y_tr, y_vl, y_tst = data.y[tr_msk], data.y[vl_msk], data.y[tst_msk]

    early_it, best_acc, test_acc = 0, 0, 0

    for i in range(n_iter):
        optimizer.zero_grad()
        model.train()

        out = model(data.x)
        F.cross_entropy(out[tr_msk], y_tr).backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            out = model(data.x)

            val_acc = accuracy(out[vl_msk], y_vl)

            if val_acc >= best_acc:
                early_it = 0
                best_acc = val_acc
                test_acc = accuracy(out[tst_msk], y_tst)
            else:
                early_it += 1

        if early_it == 200:
            break

    return best_acc, test_acc, i
