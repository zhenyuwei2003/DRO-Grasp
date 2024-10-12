import os
import sys
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from utils.rotation import *


def process_transform(pk_chain, transform, device=None):
    """ Compute extra link transform, and convert SE3 transform to only translation. """
    new_transform = transform.copy()
    for name in pk_chain.get_frame_names(exclude_fixed=False):
        if name.startswith('extra'):
            frame = pk_chain.find_frame(name)
            parent_name = pk_chain.idx_to_frame[pk_chain.parents_indices[pk_chain.frame_to_idx[name]][-2].item()]
            new_transform[name] = new_transform[parent_name] @ frame.joint.offset.get_matrix()[0]
    for name, se3 in new_transform.items():
        new_transform[name] = se3[:, :3, 3]
        if device is not None:
            new_transform[name] = new_transform[name].to(device)

    return new_transform


def jacobian(pk_chain, q, frame_X_dict, frame_names):
    """
    Calculate Jacobian (dX/dq) of all frames

    Notation: (similar as https://manipulation.csail.mit.edu/pick.html#monogram)
        J: jacobian, X: transform, R: rotation, p: position, v: velocity, w: angular velocity
        <>_BA_C: <X, R, p, w> of frame A measured from frame B expressed in frame C
        W: world frame, J: joint frame, F: link frame

    :param pk_chain: get from pk.build_chain_from_urdf()
    :param q: (6 + DOF,) or (B, 6 + DOF), joint values (euler representation)
    :return: Jacobian: {frame_name: (B, 6, num_joints)}
    """
    jacobian_dict = {}

    q = torch.atleast_2d(q)
    batch_size = q.shape[0]
    joint_names = pk_chain.get_joint_parameter_names()
    num_joints = len(joint_names)
    joint_name2idx = {name: idx for idx, name in enumerate(joint_names)}

    frames = [pk_chain.find_frame(name) for name in pk_chain.get_joint_parent_frame_names()]
    idx = lambda frame: joint_name2idx[frame.joint.name]

    transfer_X = {}
    for frame in frames:
        q_frame = q[:, idx(frame)]
        if frame.joint.joint_type == 'prismatic':
            q_frame = q_frame.unsqueeze(-1)
        transfer_X[idx(frame)] = frame.get_transform(q_frame).get_matrix()

    frame_X_dict = {f: frame_X_dict[f] for f in frame_X_dict if f in frame_names}

    for frame_name, frame_X in frame_X_dict.items():
        jacobian = torch.zeros((batch_size, 6, num_joints), dtype=pk_chain.dtype, device=pk_chain.device)

        R_WF = frame_X.get_matrix()[:, :3, :3]
        X_JF = torch.eye(4, dtype=pk_chain.dtype, device=pk_chain.device).repeat(batch_size, 1, 1)
        for frame_idx in reversed(pk_chain.parents_indices[pk_chain.frame_to_idx[frame_name]].tolist()):
            frame = pk_chain.find_frame(pk_chain.idx_to_frame[frame_idx])
            joint = frame.joint
            if joint.joint_type == 'fixed':
                if joint.offset is not None:
                    X_JF = joint.offset.get_matrix() @ X_JF
                continue

            R_FJ = X_JF[:, :3, :3].mT
            R_WJ = R_WF @ R_FJ
            p_JF_J = X_JF[:, :3, 3][:, :, None]
            w_WJ_J = joint.axis[None, :, None].repeat(batch_size, 1, 1)
            if joint.joint_type == 'revolute':
                jacobian_v = R_WJ @ torch.cross(w_WJ_J, p_JF_J, dim=1)
                jacobian_w = R_WJ @ w_WJ_J
            elif joint.joint_type == 'prismatic':
                jacobian_v = R_WJ @ w_WJ_J
                jacobian_w = torch.zeros([batch_size, 3, 1], dtype=jacobian_v.dtype, device=jacobian_v.device)
            else:
                raise NotImplementedError(f"Unknown joint_type: {joint.joint_type}")

            joint_idx = joint_name2idx[joint.name]
            X_JF = transfer_X[joint_idx] @ X_JF
            jacobian[:, :, joint_idx] = torch.cat([jacobian_v[..., 0], jacobian_w[..., 0]], dim=1)

        jacobian_dict[frame_name] = jacobian
    return jacobian_dict


def create_problem(pk_chain, frame_names):
    """
    Only use all frame positions (ignore rotation) to optimize joint values.

    :param pk_chain: get from pk.build_chain_from_urdf()
    :param frame_names: list of frame names to optimize
    :return: CvxpyLayer()
    """
    n_joint = len(pk_chain.get_joint_parameter_names())

    delta_q = cp.Variable(n_joint)

    q = cp.Parameter(n_joint)
    jacobian = {}
    frame_xyz = {}
    target_frame_xyz = {}

    objective_expr = 0
    for link_name in frame_names:
        frame_xyz[link_name] = cp.Parameter(3)
        target_frame_xyz[link_name] = cp.Parameter(3)

        jacobian[link_name] = cp.Parameter((3, n_joint))
        delta_frame_xyz = jacobian[link_name] @ delta_q

        predict_frame_xyz = frame_xyz[link_name] + delta_frame_xyz
        objective_expr += cp.norm2(predict_frame_xyz - target_frame_xyz[link_name])
    objective = cp.Minimize(objective_expr)

    lower_joint_limits, upper_joint_limits = pk_chain.get_joint_limits()
    upper_limit = cp.minimum(0.5, upper_joint_limits - q)
    lower_limit = cp.maximum(-0.5, lower_joint_limits - q)
    constraints = [delta_q <= upper_limit, delta_q >= lower_limit]
    problem = cp.Problem(objective, constraints)

    layer = CvxpyLayer(
        problem,
        parameters=[
            q,
            *frame_xyz.values(),
            *target_frame_xyz.values(),
            *jacobian.values()
        ],
        variables=[delta_q]
    )
    return layer


def optimization(pk_chain, layer, initial_q, transform, n_iter=64, verbose=False):
    if initial_q.shape[-1] != len(pk_chain.get_joint_parameter_names()):
        initial_q = q_rot6d_to_q_euler(initial_q)
    q = initial_q.clone()

    for i in range(n_iter):
        status = pk_chain.forward_kinematics(q)
        jacobians = jacobian(pk_chain, q, status, transform.keys())

        frame_xyz = {}
        target_frame_xyz = {}
        jacobians_xyz = {}
        for link_name, link_jacobian in jacobians.items():
            frame_xyz[link_name] = status[link_name].get_matrix()[:, :3, 3]
            target_frame_xyz[link_name] = transform[link_name]
            jacobians_xyz[link_name] = link_jacobian[:, :3, :]

        delta_q = layer(
            q,
            *list(frame_xyz.values()),
            *list(target_frame_xyz.values()),
            *list(jacobians_xyz.values()),
        )
        q += delta_q[0]
        if verbose:
            print(f'[Step {i}], delta_q norm: {delta_q[0].norm()}')
        if delta_q[0].norm() < 0.3:
            if verbose:
                print("Converged at iteration:", i)
            break
    return q
