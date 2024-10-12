import torch


def compute_se3_transform(P, Q):
    """
    Compute SE3 transform between two point clouds.

    :param P: (N, 3) or (B, N, 3), point cloud (w/ or w/o batch)
    :param Q: same as P
    :return: SE3 transform between P and Q, (4, 4) or (B, 4, 4)
    """
    assert P.shape == Q.shape

    if P.ndim == 2:  # (N, 3)
        P_mean = torch.mean(P, dim=0)
        Q_mean = torch.mean(Q, dim=0)
        P_prime = P - P_mean
        Q_prime = Q - Q_mean
        H = P_prime.T @ Q_prime
        U, _, Vt = torch.linalg.svd(H)
        V = Vt.T
        R = V @ U.T
        if torch.linalg.det(R) < 0:
            V[:, -1] *= -1
            R = V @ U.T
        t = Q_mean - R @ P_mean

        T = torch.eye(4).to(P.device)
        T[:3, :3] = R
        T[:3, 3] = t
    elif P.ndim == 3:  # (B, N, 3)
        P_mean = torch.mean(P, dim=1, keepdim=True)
        Q_mean = torch.mean(Q, dim=1, keepdim=True)
        P_prime = P - P_mean
        Q_prime = Q - Q_mean
        H = P_prime.transpose(1, 2) @ Q_prime
        U, _, Vt = torch.linalg.svd(H)
        V = Vt.transpose(1, 2)
        R = V @ U.transpose(1, 2)
        det_R = torch.linalg.det(R)
        VV = V.clone()
        VV[:, :, -1] *= torch.where(det_R < 0, -1.0, 1.0).unsqueeze(-1)
        RR = VV @ U.transpose(1, 2)
        t = Q_mean.squeeze(1) - (RR @ P_mean.transpose(1, 2)).squeeze(-1)

        T = torch.eye(4).repeat(P.shape[0], 1, 1).to(P.device)
        T[:, :3, :3] = RR
        T[:, :3, 3] = t
    else:
        raise RuntimeError('Unexpected point cloud shape!')

    return T


def se3_transform_point_cloud(P, Transform):
    """
    Apply SE3 transform on point cloud.

    :param P: (N, 3) or (B, N, 3), point cloud (w/ or w/o batch)
    :param Transform: SE3 transform (w/ or w/o batch)
    :return: Point Cloud after SE3 transform, (N, 3) or (B, N, 3)
    """
    P_prime = torch.cat((P, torch.ones([*P.shape[:-1], 1], dtype=torch.float32, device=P.device)), dim=-1)
    P_transformed = P_prime @ Transform.mT
    return P_transformed[..., :3]


def compute_link_pose(robot_links_pc, predict_pcs, is_train=True):
    """
    Calculate link poses of the predicted pc.

    :param robot_links_pc: (train) [{link_name: (N_i, 3), ...}, ...], per link sampled points of batch robots
                           (validate) {link_name: (N_i, 3), ...}, per link sampled points of the same robot
    :param predict_pcs: (B, N, 3), point cloud to calculate SE3
    :return: link transforms, [{link_name: (4, 4)}, ...];
             transformed_pc, (B, N, 3)
    """
    if is_train:
        assert predict_pcs.ndim == 3, "compute_link_pose() requires batch data during training."
        batch_transform = []
        batch_transformed_pc = []
        for idx in range(len(robot_links_pc)):
            links_pc = robot_links_pc[idx]
            predict_pc = predict_pcs[idx]

            global_index = 0
            transform = {}
            transformed_pc = []
            for link_index, (link_name, link_pc) in enumerate(links_pc.items()):
                predict_pc_link = predict_pc[global_index: global_index + link_pc.shape[-2], :3]
                global_index += link_pc.shape[0]

                link_se3 = compute_se3_transform(link_pc.unsqueeze(0), predict_pc_link.unsqueeze(0))[0]  # (4, 4)
                link_transformed_pc = se3_transform_point_cloud(link_pc, link_se3)  # (N_link, 3)
                transform[link_name] = link_se3
                transformed_pc.append(link_transformed_pc)

            batch_transform.append(transform)
            batch_transformed_pc.append(torch.cat(transformed_pc, dim=0))

        return batch_transform, torch.stack(batch_transformed_pc, dim=0)
    else:
        batch_transform = {}
        batch_transformed_pc = []
        global_index = 0
        for link_index, (link_name, link_pc) in enumerate(robot_links_pc.items()):
            if predict_pcs.ndim == 3 and link_pc.ndim != 3:
                link_pc = link_pc.unsqueeze(0).repeat(predict_pcs.shape[0], 1, 1)
            predict_pc_link = predict_pcs[..., global_index: global_index + link_pc.shape[-2], :3]
            global_index += link_pc.shape[-2]
            batch_transform[link_name] = compute_se3_transform(link_pc, predict_pc_link)
            batch_transformed_pc.append(se3_transform_point_cloud(link_pc, batch_transform[link_name]))
        batch_transformed_pc = torch.cat(batch_transformed_pc, dim=-2)

        return batch_transform, batch_transformed_pc
