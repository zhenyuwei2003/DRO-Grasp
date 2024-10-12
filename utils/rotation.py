import torch
from scipy.spatial.transform import Rotation

def matrix_to_euler(matrix):
    device = matrix.device
    # forward_kinematics() requires intrinsic euler ('XYZ')
    euler = Rotation.from_matrix(matrix.cpu().numpy()).as_euler('XYZ')
    return torch.tensor(euler, dtype=torch.float32, device=device)

def euler_to_matrix(euler):
    device = euler.device
    matrix = Rotation.from_euler('XYZ', euler.cpu().numpy()).as_matrix()
    return torch.tensor(matrix, dtype=torch.float32, device=device)

def matrix_to_rot6d(matrix):
    return matrix.T.reshape(9)[:6]

def rot6d_to_matrix(rot6d):
    x = normalize(rot6d[..., 0:3])
    y = normalize(rot6d[..., 3:6])
    a = normalize(x + y)
    b = normalize(x - y)
    x = normalize(a + b)
    y = normalize(a - b)
    z = normalize(torch.cross(x, y, dim=-1))
    matrix = torch.stack([x, y, z], dim=-2).mT
    return matrix

def euler_to_rot6d(euler):
    matrix = euler_to_matrix(euler)
    return matrix_to_rot6d(matrix)

def rot6d_to_euler(rot6d):
    matrix = rot6d_to_matrix(rot6d)
    return matrix_to_euler(matrix)

def axisangle_to_matrix(axis, angle):
    (x, y, z), c, s = axis, torch.cos(angle), torch.sin(angle)
    return torch.tensor([
        [(1 - c) * x * x + c, (1 - c) * x * y - s * z, (1 - c) * x * z + s * y],
        [(1 - c) * x * y + s * z, (1 - c) * y * y + c, (1 - c) * y * z - s * x],
        [(1 - c) * x * z - s * y, (1 - c) * y * z + s * x, (1 - c) * z * z + c]
    ])

def euler_to_quaternion(euler):
    device = euler.device
    quaternion = Rotation.from_euler('XYZ', euler.cpu().numpy()).as_quat()
    return torch.tensor(quaternion, dtype=torch.float32, device=device)

def normalize(v):
    return v / torch.norm(v, dim=-1, keepdim=True)

def q_euler_to_q_rot6d(q_euler):
    return torch.cat([q_euler[..., :3], euler_to_rot6d(q_euler[..., 3:6]), q_euler[..., 6:]], dim=-1)

def q_rot6d_to_q_euler(q_rot6d):
    return torch.cat([q_rot6d[..., :3], rot6d_to_euler(q_rot6d[..., 3:9]), q_rot6d[..., 9:]], dim=-1)


if __name__ == '__main__':
    """ Test correctness of above functions, no need to compare euler angle due to singularity. """
    test_euler = torch.rand(3) * 2 * torch.pi

    test_matrix = euler_to_matrix(test_euler)
    test_euler_prime = matrix_to_euler(test_matrix)
    test_matrix_prime = euler_to_matrix(test_euler_prime)
    assert torch.allclose(test_matrix, test_matrix_prime), \
        f"Original Matrix: {test_matrix}, Converted Matrix: {test_matrix_prime}"

    test_rot6d = matrix_to_rot6d(test_matrix)
    test_matrix_prime = rot6d_to_matrix(test_rot6d)
    assert torch.allclose(test_matrix, test_matrix_prime),\
        f"Original Matrix: {test_matrix}, Converted Matrix: {test_matrix_prime}"

    test_rot6d_prime = matrix_to_rot6d(test_matrix_prime)
    assert torch.allclose(test_rot6d, test_rot6d_prime), \
        f"Original Rot6D: {test_rot6d}, Converted Rot6D: {test_rot6d_prime}"

    test_euler_prime = rot6d_to_euler(test_rot6d)
    test_rot6d_prime = euler_to_rot6d(test_euler_prime)
    assert torch.allclose(test_rot6d, test_rot6d_prime), \
        f"Original Rot6D: {test_rot6d}, Converted Rot6D: {test_rot6d_prime}"

    print("All Tests Passed！")
