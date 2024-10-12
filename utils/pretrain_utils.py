"""
The code is sourced with some modifications made, from
https://github.com/r-pad/taxpose/blob/0c4298fa0486fd09e63bf24d618a579b66ba0f18/taxpose/utils/emb_losses.py.
"""

import torch
import torch.nn.functional as F


def dist2weight(xyz, func=None):
    d = (xyz.unsqueeze(1) - xyz.unsqueeze(2)).norm(dim=-1)
    if func is not None:
        d = func(d)
    w = d / d.max(dim=-1, keepdims=True)[0]
    w = w + torch.eye(d.shape[-1], device=d.device).unsqueeze(0).tile(
        [d.shape[0], 1, 1]
    )
    return w


def infonce_loss(phi_1, phi_2, weights=None, temperature=0.1):
    B, N, D = phi_1.shape

    # cosine similarity
    phi_1 = F.normalize(phi_1, dim=2)
    phi_2 = F.normalize(phi_2, dim=2)
    similarity = phi_1 @ phi_2.mT

    target = torch.arange(N, device=similarity.device).tile([B, 1])
    if weights is None:
        weights = 1.0
    loss = F.cross_entropy(torch.log(weights) + (similarity / temperature), target)

    return loss, similarity


def mean_order(similarity):
    order = (similarity > similarity.diagonal(dim1=-2, dim2=-1).unsqueeze(-1)).sum(-1)
    return order.float().mean() / similarity.shape[-1]
