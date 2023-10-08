import torch


def min_kernel(s,t):
    return torch.minimum(s[None].T, t)


def sobolev_cmpct(a, b):
    assert b > a

    lngth = b - a
    kern_fn = lambda s,t: 1 + min_kernel((s - a)/lngth, (t - a)/lngth)
    return kern_fn


def sobolev_reals(gamma):
    kern_fn = lambda s,t: torch.exp(-gamma*torch.cdist(s[:,None], t[:,None], p=1))
    return kern_fn


def gaussian_rbf(gamma):
    kern_fn = lambda s,t: torch.exp(-gamma*torch.cdist(s[:,None], t[:,None], p=2)**2)
    return kern_fn

def linear_reals():
    kern_fn = lambda s,t: torch.outer(s,t)
    return kern_fn
    