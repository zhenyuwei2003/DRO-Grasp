import typing
from typing import Optional

import torch


@typing.no_type_check
def estimate_p(
    P: torch.FloatTensor, R: torch.FloatTensor, W: Optional[torch.FloatTensor] = None
) -> torch.FloatStorage:
    assert P.ndim == 3  # N x D x 1
    assert R.ndim == 1  # N
    assert P.shape[0] == R.shape[0]
    assert P.shape[1] in {2, 3}

    N, D, _ = P.shape

    if W is None:
        W = torch.ones(N, device=P.device)
    assert W.ndim == 1  # N
    W = W[:, None, None]

    # Shared stuff.
    Pt = P.permute(0, 2, 1)
    PPt = P @ Pt
    PtP = (Pt @ P).squeeze()
    I = torch.eye(D, device=P.device)
    NI = I[None].repeat(N, 1, 1)
    PtP_minus_r2 = (PtP - R**2)[:, None, None]

    # These are ripped straight from the paper, with weighting passed through.
    a = (W * (PtP_minus_r2 * P)).mean(dim=0)
    B = (W * (-2 * PPt - PtP_minus_r2 * NI)).mean(dim=0)
    c = (W * P).mean(dim=0)
    f = a + B @ c + 2 * c @ c.T @ c
    H = -2 * PPt.mean(dim=0) + 2 * c @ c.T
    q = -torch.linalg.inv(H) @ f
    p = q + c

    return p


def multilateration(dro, fixed_pc):
    """
    Compute the target point cloud described by D(R,O) matrix & fixed_pc

    :param dro: (B, N, N), point-wise relative distance matrix between target point cloud & fixed point cloud
    :param fixed_pc: (B, N, 3), point cloud as a reference for relative distance
    :return: (B, N, 3), the target point cloud
    """
    assert dro.ndim == 3 and fixed_pc.ndim == 3, "multilateration() requires batch data."
    v_est_p = torch.vmap(torch.vmap(estimate_p, in_dims=(None, 0)))
    target_pc = v_est_p(fixed_pc.unsqueeze(-1), dro)[..., 0]
    return target_pc