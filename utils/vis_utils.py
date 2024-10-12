"""
Most of the visualization code has not been encapsulated into functions;
only the part for visualizing vectors is kept in this file, and the comment format is not consistent.
"""

import trimesh
import numpy as np
from scipy.spatial.transform import Rotation as R


def normalize(x):
    """
    Normalize the input vector. If the magnitude of the vector is zero, a small value is added to prevent division by zero.

    Parameters:
    - x (np.ndarray): Input vector to be normalized.

    Returns:
    - np.ndarray: Normalized vector.
    """
    if len(x.shape) == 1:
        mag = np.linalg.norm(x)
        if mag == 0:
            mag = mag + 1e-10
        return x / mag
    else:
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-10, norms)
        return x / norms


def sample_transform_w_normals(
    new_palm_center,
    new_face_vector,
    sample_roll,
    ori_face_vector=np.array([1.0, 0.0, 0.0]),
):
    """
    Compute the transformation matrix from the original palm pose to a new palm pose.

    Parameters:
    - new_palm_center (np.ndarray): The point of the palm center [x, y, z].
    - new_face_vector (np.ndarray): The direction vector representing the new palm facing direction.
    - sample_roll (float): The roll angle in range [0, 2*pi).
    - ori_face_vector (np.ndarray): The original direction vector representing the palm facing direction. Default is [1.0, 0.0, 0.0].

    Returns:
    - rst_transform (np.ndarray): A 4x4 transformation matrix.
    """

    rot_axis = np.cross(ori_face_vector, normalize(new_face_vector))
    rot_axis = rot_axis / (np.linalg.norm(rot_axis) + 1e-16)
    rot_ang = np.arccos(np.clip(np.dot(ori_face_vector, new_face_vector), -1.0, 1.0))

    if rot_ang > 3.1415 or rot_ang < -3.1415:
        rot_axis = (
            np.array([1.0, 0.0, 0.0])
            if not np.isclose(ori_face_vector, np.array([1.0, 0.0, 0.0])).all()
            else np.array([0.0, 1.0, 0.0])
        )

    rot = R.from_rotvec(rot_ang * rot_axis).as_matrix()
    roll_rot = R.from_rotvec(sample_roll * new_face_vector).as_matrix()

    final_rot = roll_rot @ rot
    rst_transform = np.eye(4)
    rst_transform[:3, :3] = final_rot
    rst_transform[:3, 3] = new_palm_center
    return rst_transform


def vis_vector(
    start_point,
    vector,
    length=0.1,
    cyliner_r=0.003,
    color=[255, 255, 100, 245],
    no_arrow=False,
):
    """
    start_points: np.ndarray, shape=(3,)
    vectors: np.ndarray, shape=(3,)
    length: cylinder length
    """
    normalized_vector = normalize(vector)
    end_point = start_point + length * normalized_vector

    # create a mesh for the force
    force_cylinder = trimesh.creation.cylinder(
        radius=cyliner_r, segment=np.array([start_point, end_point])
    )

    # create a mesh for the arrowhead
    cone_transform = sample_transform_w_normals(
        end_point, normalized_vector, 0, ori_face_vector=np.array([0.0, 0.0, 1.0])
    )
    arrowhead_cone = trimesh.creation.cone(
        radius=2 * cyliner_r, height=4 * cyliner_r, transform=cone_transform
    )
    # combine the two meshes into one
    if not no_arrow:
        force_mesh = force_cylinder + arrowhead_cone
    else:
        force_mesh = force_cylinder
    force_mesh.visual.face_colors = color

    return force_mesh
