import torch
import torch.nn as nn


class LinearGNN(nn.Module):
    def __init__(self, n_feats, n_out, U=None, Vh=None):
        super().__init__()

        if (U is not None) and (Vh is not None):
            assert all(e1 == e2 for e1,e2 in zip(U.shape, Vh.T.shape))

            k = U.shape[1]
            self.alpha = nn.Parameter(torch.randn(k, 1) / torch.sqrt(torch.tensor(k)))
        else:
            self.alpha = None

        self.U, self.Vh = U, Vh
        self.W_in = nn.Linear(n_feats, n_out)

    def forward(self, X, K):
        out = self.W_in(X)

        if self.alpha is not None:
            eigs = self.alpha if K is None else K @ self.alpha
            out = self.U @ (eigs * (self.Vh @ out))

        return out
