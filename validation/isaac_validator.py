from isaacgym import gymapi
from isaacgym import gymtorch

import os
import sys
import time
import numpy as np
import torch
from scipy.spatial.transform import Rotation

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from utils.controller import controller


class IsaacValidator:
    def __init__(
        self,
        robot_name,
        joint_orders,
        batch_size,
        gpu=0,
        is_filter=False,
        use_gui=False,
        robot_friction=3.,
        object_friction=3.,
        steps_per_sec=100,
        grasp_step=100,
        debug_interval=0.01
    ):
        self.gym = gymapi.acquire_gym()
        
        self.robot_name = robot_name
        self.joint_orders = joint_orders
        self.batch_size = batch_size
        self.gpu = gpu
        self.is_filter = is_filter
        self.robot_friction = robot_friction
        self.object_friction = object_friction
        self.steps_per_sec = steps_per_sec
        self.grasp_step = grasp_step
        self.debug_interval = debug_interval

        self.envs = []
        self.robot_handles = []
        self.object_handles = []
        self.robot_asset = None
        self.object_asset = None
        self.rigid_body_num = None
        self.object_force = None
        self.urdf2isaac_order = None
        self.isaac2urdf_order = None

        self.sim_params = gymapi.SimParams()
        # set common parameters
        self.sim_params.dt = 1 / steps_per_sec
        self.sim_params.substeps = 2
        self.sim_params.gravity = gymapi.Vec3(0.0, 0.0, 0.0)
        #self.sim_params.use_gpu_pipeline = True
        # set PhysX-specific parameters
        self.sim_params.physx.use_gpu = True
        self.sim_params.physx.solver_type = 1
        self.sim_params.physx.num_position_iterations = 8
        self.sim_params.physx.num_velocity_iterations = 0
        self.sim_params.physx.contact_offset = 0.01
        self.sim_params.physx.rest_offset = 0.0

        self.sim = self.gym.create_sim(self.gpu, self.gpu, gymapi.SIM_PHYSX, self.sim_params)
        self._rigid_body_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self._dof_states = self.gym.acquire_dof_state_tensor(self.sim)

        self.viewer = None
        if use_gui:
            self.has_viewer = True
            self.camera_props = gymapi.CameraProperties()
            self.camera_props.width = 1920
            self.camera_props.height = 1080
            self.camera_props.use_collision_geometry = True
            self.viewer = self.gym.create_viewer(self.sim, self.camera_props)
            self.gym.viewer_camera_look_at(self.viewer, None, gymapi.Vec3(1, 0, 0), gymapi.Vec3(0, 0, 0))
        else:
            self.has_viewer = False

        self.robot_asset_options = gymapi.AssetOptions()
        self.robot_asset_options.disable_gravity = True
        self.robot_asset_options.fix_base_link = True
        self.robot_asset_options.collapse_fixed_joints = True

        self.object_asset_options = gymapi.AssetOptions()
        self.object_asset_options.override_com = True
        self.object_asset_options.override_inertia = True
        self.object_asset_options.density = 500

    def set_asset(self, robot_path, robot_file, object_path, object_file):
        self.robot_asset = self.gym.load_asset(self.sim, robot_path, robot_file, self.robot_asset_options)
        self.object_asset = self.gym.load_asset(self.sim, object_path, object_file, self.object_asset_options)
        self.rigid_body_num = (self.gym.get_asset_rigid_body_count(self.robot_asset)
                               + self.gym.get_asset_rigid_body_count(self.object_asset))
        # print_asset_info(gym, self.robot_asset, 'robot')
        # print_asset_info(gym, self.object_asset, 'object')

    def create_envs(self):
        for env_idx in range(self.batch_size):
            env = self.gym.create_env(
                self.sim,
                gymapi.Vec3(-1, -1, -1),
                gymapi.Vec3(1, 1, 1),
                int(self.batch_size ** 0.5)
            )
            self.envs.append(env)

            # draw world frame
            if self.has_viewer:
                x_axis_dir = np.array([0, 0, 0, 1, 0, 0], dtype=np.float32)
                x_axis_color = np.array([1, 0, 0], dtype=np.float32)
                self.gym.add_lines(self.viewer, env, 1, x_axis_dir, x_axis_color)
                y_axis_dir = np.array([0, 0, 0, 0, 1, 0], dtype=np.float32)
                y_axis_color = np.array([0, 1, 0], dtype=np.float32)
                self.gym.add_lines(self.viewer, env, 1, y_axis_dir, y_axis_color)
                z_axis_dir = np.array([0, 0, 0, 0, 0, 1], dtype=np.float32)
                z_axis_color = np.array([0, 0, 1], dtype=np.float32)
                self.gym.add_lines(self.viewer, env, 1, z_axis_dir, z_axis_color)

            # object actor setting
            object_handle = self.gym.create_actor(
                env,
                self.object_asset,
                gymapi.Transform(),
                f'object_{env_idx}',
                env_idx
            )
            self.object_handles.append(object_handle)

            object_shape_properties = self.gym.get_actor_rigid_shape_properties(env, object_handle)
            for i in range(len(object_shape_properties)):
                object_shape_properties[i].friction = self.object_friction
            self.gym.set_actor_rigid_shape_properties(env, object_handle, object_shape_properties)

            # robot actor setting
            robot_handle = self.gym.create_actor(
                env,
                self.robot_asset,
                gymapi.Transform(),
                f'robot_{env_idx}',
                env_idx
            )
            self.robot_handles.append(robot_handle)

            robot_properties = self.gym.get_actor_dof_properties(env, robot_handle)
            robot_properties["driveMode"].fill(gymapi.DOF_MODE_POS)
            robot_properties["stiffness"].fill(1000)
            robot_properties["damping"].fill(200)
            self.gym.set_actor_dof_properties(env, robot_handle, robot_properties)

            object_shape_properties = self.gym.get_actor_rigid_shape_properties(env, robot_handle)
            for i in range(len(object_shape_properties)):
                object_shape_properties[i].friction = self.robot_friction
            self.gym.set_actor_rigid_shape_properties(env, robot_handle, object_shape_properties)

            # print_actor_info(self.gym, env, robot_handle)
            # print_actor_info(self.gym, env, object_handle)

        # assume robots & objects in the same batch are the same
        obj_property = self.gym.get_actor_rigid_body_properties(self.envs[0], self.object_handles[0])
        object_mass = [obj_property[i].mass for i in range(len(obj_property))]
        object_mass = torch.tensor(object_mass)
        self.object_force = 0.5 * object_mass

        self.urdf2isaac_order = np.zeros(len(self.joint_orders), dtype=np.int32)
        self.isaac2urdf_order = np.zeros(len(self.joint_orders), dtype=np.int32)
        for urdf_idx, joint_name in enumerate(self.joint_orders):
            isaac_idx = self.gym.find_actor_dof_index(self.envs[0], self.robot_handles[0], joint_name, gymapi.DOMAIN_ACTOR)
            self.urdf2isaac_order[isaac_idx] = urdf_idx
            self.isaac2urdf_order[urdf_idx] = isaac_idx

    def set_actor_pose_dof(self, q):
        self.gym.prepare_sim(self.sim)

        # set all actors to origin
        _root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        root_state = gymtorch.wrap_tensor(_root_state)
        root_state[:] = torch.tensor([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=torch.float32)
        self.gym.set_actor_root_state_tensor(self.sim, _root_state)

        outer_q, inner_q = controller(self.robot_name, q)

        for env_idx in range(len(self.envs)):
            env = self.envs[env_idx]
            robot_handle = self.robot_handles[env_idx]

            dof_states_initial = self.gym.get_actor_dof_states(env, robot_handle, gymapi.STATE_ALL).copy()
            dof_states_initial['pos'] = outer_q[env_idx, self.urdf2isaac_order]
            self.gym.set_actor_dof_states(env, robot_handle, dof_states_initial, gymapi.STATE_ALL)

            dof_states_target = self.gym.get_actor_dof_states(env, robot_handle, gymapi.STATE_ALL).copy()
            dof_states_target['pos'] = inner_q[env_idx, self.urdf2isaac_order]
            self.gym.set_actor_dof_position_targets(env, robot_handle, dof_states_target["pos"])

    def run_sim(self):
        # controller phase
        for step in range(self.grasp_step):
            self.gym.simulate(self.sim)

            if self.has_viewer:
                if self.gym.query_viewer_has_closed(self.viewer):
                    break
                t = time.time()
                while time.time() - t < self.debug_interval:
                    self.gym.step_graphics(self.sim)
                    self.gym.draw_viewer(self.viewer, self.sim, render_collision=True)

        self.gym.refresh_rigid_body_state_tensor(self.sim)
        start_pos = gymtorch.wrap_tensor(self._rigid_body_states)[::self.rigid_body_num, :3].clone()

        force_tensor = torch.zeros([len(self.envs), self.rigid_body_num, 3])  # env, rigid_body, xyz
        x_pos_force = force_tensor.clone()
        x_pos_force[:, 0, 0] = self.object_force
        x_neg_force = force_tensor.clone()
        x_neg_force[:, 0, 0] = -self.object_force
        y_pos_force = force_tensor.clone()
        y_pos_force[:, 0, 1] = self.object_force
        y_neg_force = force_tensor.clone()
        y_neg_force[:, 0, 1] = -self.object_force
        z_pos_force = force_tensor.clone()
        z_pos_force[:, 0, 2] = self.object_force
        z_neg_force = force_tensor.clone()
        z_neg_force[:, 0, 2] = -self.object_force
        force_list = [x_pos_force, y_pos_force, z_pos_force, x_neg_force, y_neg_force, z_neg_force]

        # force phase
        for step in range(self.steps_per_sec * 6):
            self.gym.apply_rigid_body_force_tensors(self.sim,
                                                    gymtorch.unwrap_tensor(force_list[step // self.steps_per_sec]),
                                                    None,
                                                    gymapi.ENV_SPACE)
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)

            if self.has_viewer:
                if self.gym.query_viewer_has_closed(self.viewer):
                    break
                t = time.time()
                while time.time() - t < self.debug_interval:
                    self.gym.step_graphics(self.sim)
                    self.gym.draw_viewer(self.viewer, self.sim, render_collision=True)

        self.gym.refresh_rigid_body_state_tensor(self.sim)
        end_pos = gymtorch.wrap_tensor(self._rigid_body_states)[::self.rigid_body_num, :3].clone()

        distance = (end_pos - start_pos).norm(dim=-1)
    
        if self.is_filter:
            success = (distance <= 0.02) & (end_pos.norm(dim=-1) <= 0.05)
        else:
            success = (distance <= 0.02)

        # apply inverse object transform to robot to get new joint value
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        object_pose = gymtorch.wrap_tensor(self._rigid_body_states).clone()[::self.rigid_body_num, :7]  # batch_size, 7 (xyz + quat)
        object_transform = np.eye(4)[np.newaxis].repeat(self.batch_size, axis=0)
        object_transform[:, :3, 3] = object_pose[:, :3]
        object_transform[:, :3, :3] = Rotation.from_quat(object_pose[:, 3:7]).as_matrix()

        self.gym.refresh_dof_state_tensor(self.sim)
        dof_states = gymtorch.wrap_tensor(self._dof_states).clone().reshape(len(self.envs), -1, 2)[:, :, 0]  # batch_size, DOF (xyz + euler + joint)
        robot_transform = np.eye(4)[np.newaxis].repeat(self.batch_size, axis=0)
        robot_transform[:, :3, 3] = dof_states[:, :3]
        robot_transform[:, :3, :3] = Rotation.from_euler('XYZ', dof_states[:, 3:6]).as_matrix()

        robot_transform = np.linalg.inv(object_transform) @ robot_transform
        dof_states[:, :3] = torch.tensor(robot_transform[:, :3, 3])
        dof_states[:, 3:6] = torch.tensor(Rotation.from_matrix(robot_transform[:, :3, :3]).as_euler('XYZ'))
        q_isaac = dof_states[:, self.isaac2urdf_order].to(torch.device('cpu'))

        return success, q_isaac

    def reset_simulator(self):
        self.gym.destroy_sim(self.sim)
        if self.has_viewer:
            self.gym.destroy_viewer(self.viewer)
            self.viewer = self.gym.create_viewer(self.sim, self.camera_props)
        self.sim = self.gym.create_sim(self.gpu, self.gpu, gymapi.SIM_PHYSX, self.sim_params)
        for env in self.envs:
            self.gym.destroy_env(env)
        self.envs = []
        self.robot_handles = []
        self.object_handles = []
        self.robot_asset = None
        self.object_asset = None

    def destroy(self):
        for env in self.envs:
            self.gym.destroy_env(env)
        self.gym.destroy_sim(self.sim)
        if self.has_viewer:
            self.gym.destroy_viewer(self.viewer)
        del self.gym
