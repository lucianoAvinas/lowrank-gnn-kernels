import torch
import torch.nn as nn
import torch.nn.functional as F


def get_network_matrix(A, norm=False, shift=False, is_symm=False):
# A is the adjacency matrix
    D = A.sum(1)
    
    if norm:
        mask = (D != 0)
        Dinv = torch.ones_like(D)
        if is_symm:
            Dinv[mask] = 1/torch.sqrt(D[mask])
            A = Dinv[:,None] * (A * Dinv[None])
        else:
            Dinv[mask] = 1/D[mask]
            A = Dinv[:,None] * A 

        A = torch.eye(A.shape[0]).to(A.device) - A if shift else A
    else:
        A = torch.diag(D) - A if shift else A

    return A

def get_spectral_decomp(A, is_symm=False):
    # Spectral decomp
    if is_symm:
        S, U = torch.linalg.eigh(A)
        Vh = U.T
    else:
        U, S, Vh = torch.linalg.svd(A)  
    return U, S, Vh
            

import torch
import torch.nn as nn
import torch.nn.functional as F

class NPGNN(nn.Module):
    def __init__(self, A, n_feats, n_out, 
                 spec_train=True, kern_fn=None, 
                 norm=False, shift=False, pct=1,
                 use_sqrt_K=False):
        super().__init__()

        # Get the network matrix and its spectral decomposition
        is_symm = torch.all(A == A.T)
        self.M = get_network_matrix(A, norm=norm, shift=shift, is_symm=is_symm)
        self.U, self.S, self.Vh = get_spectral_decomp(self.M, is_symm=is_symm)

        # Truncate the spectral decomposition
        if pct < 1:
            abs_S = abs(self.S)
            eig_mask = abs_S >= torch.quantile(abs_S, 1 - pct)
            self.U, self.S, self.Vh = self.U[:, eig_mask], self.S[eig_mask], self.Vh[eig_mask]

        # Should we train the spectral filter?
        r = self.U.shape[1]
        if spec_train:   
            self.alpha = nn.Parameter(torch.randn(r, 1) / torch.sqrt(torch.tensor(r)))
        else:
            self.alpha = None  # effectively the all-ones vector

        if kern_fn is not None:
            self.K = kern_fn(self.S, self.S)  # will be an r x r matrix
            # self.K = kern_fn(self.S, self.S) / r  # will be an r x r matrix
            if use_sqrt_K:
               # Compute the matrix square root of K
                eigvals, eigvecs = torch.linalg.eigh(self.K)
                self.K = eigvecs @ torch.diag(torch.sqrt(eigvals)) @ eigvecs.T
        else:
            self.K = None  # effectively identity

        self.W_in = nn.Linear(n_feats, n_out)

    def forward(self, X):
        out = self.W_in(X)

        if self.alpha is not None:
            eigs = self.alpha if self.K is None else self.K @ self.alpha
            out = self.U @ (eigs * (self.Vh @ out))
        else:
            out = self.M @ out 

        return out
