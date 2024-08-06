import torch_geometric
import torch
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
import plotly.graph_objects as go
from sklearn.decomposition import TruncatedSVD


class ExtendedData(torch_geometric.data.Data):
    def get_device(self):
        return self.edge_index.device if self.edge_index is not None else torch.device('cpu')
    
    def get_adjacency_matrix(self, format='torch_sparse'):
        device = self.get_device()
        edge_index = self.edge_index
        num_nodes = self.num_nodes
        n_edges = edge_index.shape[1]

        # Ensure edge_index is on the correct device
        edge_index = edge_index.to(device) if device else edge_index

        # Create adjacency matrix
        if format == 'torch_sparse':
            adj = torch.sparse_coo_tensor(
                edge_index, 
                torch.ones(n_edges).to(device), 
                (num_nodes, num_nodes)
            )
        # elif format == 'torch_dense':
        #     # adj = torch.zeros(num_nodes, num_nodes, device=device)
        #     # adj[edge_index[0], edge_index[1]] = 1
        #     adj = torch_geometric.utils.to_dense_adj(edge_index).squeeze()

        elif format == 'scipy':
            row, col = edge_index.cpu()
            # Create a SciPy sparse matrix
            adj = sparse.coo_matrix(
                (np.ones(n_edges), (row.numpy(), col.numpy())),
                shape=(num_nodes, num_nodes)
            )
        else:
            raise ValueError(f'Unknown format: {format}')

        return adj
    
    def to_networkx(self):
        adj = self.get_adjacency_matrix(format='scipy')
        return nx.from_scipy_sparse_array(adj)

    def plot_adjacency_matrix(self):
        adj = self.get_adjacency_matrix('scipy')
        plt.spy(adj, markersize=2)
        # plt.spy(adj.cpu().to_dense(), markersize=2)
        plt.show()

    def plot_network(self):
        adj = self.get_adjacency_matrix(format='scipy')
        G = nx.from_scipy_sparse_array(adj)
        G.remove_edges_from(nx.selfloop_edges(G))
        pos = nx.spring_layout(G, iterations=500)
        node_sizes = 20 * np.log(np.array(G.degree())[:, 1] + 1) ** 1.3
        nx.draw(
            G, pos, with_labels=False,
            node_color=self.y.cpu(),
            node_size=node_sizes,
            edge_color='gray',
            linewidths=1, font_size=15,
            arrows=True, connectionstyle='arc3,rad=0'
        )
        plt.show()

    def get_degrees(self):
        adj = self.get_adjacency_matrix(format='scipy')
        # return np.array(adj.sum(axis=0)).squeeze()
        return torch.tensor(adj.sum(axis=0)).squeeze().to(self.get_device())
    

    def plot_features_3d(self, use_svd=False):
        X = self.x.cpu().numpy()
        y = self.y.cpu().numpy()

        if X.shape[1] > 3 and use_svd:
            svd = TruncatedSVD(n_components=3)
            X = svd.fit_transform(X)
        elif X.shape[1] > 3:
            X = X[:, :3]

        fig = go.Figure()

        for class_value in np.unique(y):
            fig.add_trace(go.Scatter3d(
                x=X[y == class_value, 0],
                y=X[y == class_value, 1],
                z=X[y == class_value, 2],
                mode='markers',
                marker=dict(size=5),
                name=f'Class {class_value}'
            ))

        fig.update_layout(
            width=700,
            height=600,
            margin=dict(l=0, r=0, b=0, t=0),
            scene=dict(
                xaxis=dict(title='X1'),
                yaxis=dict(title='Y1'),
                zaxis=dict(title='Z1')
            )
        )
        # fig.show()
        return fig


    def create_masks(self, mask_type='geom_gcn'):
        mask_type = mask_type.lower()
        n = len(self.y)
        torch.manual_seed(10)

        masks = {'train': None, 'val': None, 'test': None}

        if mask_type == 'random':
            train_mask, val_mask, test_mask = torch.zeros((n, 10), dtype=bool), torch.zeros((n, 10), dtype=bool), torch.zeros((n, 10), dtype=bool)
            for i in range(10):
                inds = torch.randperm(n)
                train_mask[inds[:int(0.6 * n)], i] = True
                val_mask[inds[int(0.6 * n):int(0.8 * n)], i] = True
                test_mask[inds[int(0.8 * n):], i] = True
            masks['train'], masks['val'], masks['test'] = train_mask, val_mask, test_mask

        elif mask_type == 'balanced':
            C = len(self.y.unique())
            bnd = int(n * 0.6 / C)
            all_inds = [(self.y == c).nonzero() for c in range(C)]
            train_mask, val_mask, test_mask = torch.zeros((n, 10), dtype=bool), torch.zeros((n, 10), dtype=bool), torch.zeros((n, 10), dtype=bool)

            for i in range(10):
                eval_inds = []
                for c in range(C):
                    cls_inds = all_inds[c]
                    cls_inds = cls_inds[torch.randperm(cls_inds.shape[0])]
                    train_mask[cls_inds[:bnd], i] = True
                    eval_inds.append(cls_inds[bnd:])

                eval_inds = torch.cat(eval_inds)
                eval_inds = eval_inds[torch.randperm(eval_inds.shape[0])]

                val_mask[eval_inds[:int(n * 0.2)], i] = True
                test_mask[eval_inds[int(n * 0.2):], i] = True
            masks['train'], masks['val'], masks['test'] = train_mask, val_mask, test_mask

        else:
            # mask_type == 'geom_gcn'
            masks['train'] = self.train_mask.bool()
            masks['val'] = self.val_mask.bool()
            masks['test'] = self.test_mask.bool()

        self.masks = masks

# # Example usage
# extended_data = ExtendedData()
# extended_data.create_masks(mask_type='random')
# print(extended_data.masks['train'])


# Usage:
# extended_data = ExtendedData.from_dict(data.to_dict())
# extended_data.get_adjacency_matrix('scipy')
# extended_data.plot_adjacency_matrix()
# extended_data.plot_network()