import sys, os, os.path as osp

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import scipy
import torch
from torch import nn
import torch.nn.functional as F
from transforms3d.euler import euler2mat

from optim_utils import *
from init_helpers import (
    get_near_mesh_init_geo_values,
    get_on_mesh_init_geo_values,
    get_inside_mesh_init_geo_values,
)
import logging
from utils.system_utils import mkdir_p
from utils.chamfer import ChamferDistance
from plyfile import PlyData, PlyElement
import open3d as o3d
from pytorch3d.transforms import (
    matrix_to_axis_angle,
    axis_angle_to_matrix,
    quaternion_to_matrix,
    matrix_to_quaternion,
)
from model_utils import sph_order2nfeat
from pytorch3d.ops import knn_points
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from voxel_deformer import VoxelDeformer, Deformer
from helpers import weighted_l2_loss_v1, weighted_l2_loss_v2, quat_mult, o3d_knn
from external import build_rotation
import math
from scipy.optimize import linear_sum_assignment

from torch.distributions import Categorical
import scipy.stats as stats
from sklearn.mixture import GaussianMixture
from sklearn.covariance import MinCovDet
from sklearn.decomposition import PCA
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2, zscore
from scipy.spatial.distance import cdist
from omegaconf import DictConfig, OmegaConf
from geomloss import SamplesLoss
import colorsys
from distinctipy import distinctipy

SMPL_JOINT_IDX = {
    "pelvis_root": 0,
    "left_hip": 1,
    "right_hip": 2,
    "belly_button": 3,
    "left_knee": 4,
    "right_knee": 5,
    "lower_chest": 6,
    "left_ankle": 7,
    "right_ankle": 8,
    "upper_chest": 9,
    "left_toe": 10,
    "right_toe": 11,
    "neck": 12,
    "left_clavicle": 13,
    "right_clavicle": 14,
    "head": 15,
    "left_shoulder": 16,
    "right_shoulder": 17,
    "left_elbow": 18,
    "right_elbow": 19,
    "left_wrist": 20,
    "right_wrist": 21,
    "left_thumb": 22,
    "right_thumb": 23,
}

SMPL_PARENT = {
    1: 0,
    2: 0,
    3: 0,
    4: 1,
    5: 2,
    6: 3,
    7: 4,
    8: 5,
    9: 6,
    10: 7,
    11: 8,
    12: 9,
    13: 9,
    14: 9,
    15: 12,
    16: 13,
    17: 14,
    18: 16,
    19: 17,
    20: 18,
    21: 19,
    22: 20,
    23: 21,
}


SMAL_PARENT = {
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
    7: 6,
    8: 7,
    9: 8,
    10: 9,
    11: 6,
    12: 11,
    13: 12,
    14: 13,
    15: 6,
    16: 15,
    17: 0,
    18: 17,
    19: 18,
    20: 19,
    21: 0,
    22: 21,
    23: 22,
    24: 23,
    25: 0,
    26: 25,
    27: 26,
    28: 27,
    29: 28,
    30: 29,
    31: 30,
    32: 16,
    33: 16,
    34: 16,
}


PARENT = {"human": SMPL_PARENT, "dog": SMAL_PARENT}

TORSO_JOINTS_NAME = [
    "pelvis_root",
    "belly_button",
    "lower_chest",
    "upper_chest",
    "left_clavicle",
    "right_clavicle",
]
SMPL_TORSO_JOINTS = [SMPL_JOINT_IDX[joint_name] for joint_name in TORSO_JOINTS_NAME]


SMAL_TORSO_JOINTS = [2, 5, 8, 11, 12, 23]


TORSO_JOINTS = {"human": SMPL_TORSO_JOINTS, "dog": SMAL_TORSO_JOINTS}


BONE_STDS = np.array([0.03, 0.06, 0.03])
HEAD_STDS = np.array([0.06, 0.06, 0.06])
JOINT_STDS = np.array([0.02, 0.02, 0.02])


class AdditionalBones(nn.Module):
    def __init__(
        self,  # additional bones
        num_bones: int = 0,
        total_t: int = 0,  # any usage of time should use this!
        mode="pose-mlp",
        pose_dim=23 * 3,
        mlp_hidden_dims=[256, 256, 256, 256],
        mlp_act=nn.LeakyReLU,
    ):
        super().__init__()
        self.num_bones = num_bones
        if self.num_bones == 0:
            return
        self.mode = mode
        assert self.mode in ["pose-mlp", "pose+t-mlp", "delta-list", "list"]
        self.total_t = total_t

        if self.mode == "pose-mlp":
            self.pose_dim = pose_dim
            self.mlp_layers = nn.ModuleList()
            c_in = self.pose_dim
            for c_out in mlp_hidden_dims:
                self.mlp_layers.append(nn.Sequential(nn.Linear(c_in, c_out), mlp_act()))
                c_in = c_out
            self.mlp_output_head = nn.Linear(c_in, 7 * self.num_bones, bias=False)
            with torch.no_grad():
                self.mlp_output_head.weight.data = (
                    torch.randn_like(self.mlp_output_head.weight.data) * 1e-3
                )
        elif self.mode == "delta-list":
            self.dr_list = nn.Parameter(torch.zeros(self.total_t, num_bones, 3))
            self.dt_list = nn.Parameter(torch.zeros(self.total_t, num_bones, 3))
        else:
            raise NotImplementedError()

        return


def aiap_loss(x_canonical, x_deformed, n_neighbors=5, nn_ix=None):
    if x_canonical.shape != x_deformed.shape:
        raise ValueError("Input point sets must have the same shape.")

    if nn_ix is None:
        _, nn_ix, _ = knn_points(
            x_canonical.unsqueeze(0),
            x_canonical.unsqueeze(0),
            K=n_neighbors + 1,
            return_sorted=True,
        )
        nn_ix = nn_ix.squeeze(0)

    dists_canonical = torch.cdist(x_canonical.unsqueeze(1), x_canonical[nn_ix])[
        :, 0, 1:
    ]
    dists_deformed = torch.cdist(x_deformed.unsqueeze(1), x_deformed[nn_ix])[:, 0, 1:]

    loss = F.l1_loss(dists_canonical, dists_deformed)

    return loss


def full_aiap_loss(xyz_can, xyz_obs, n_neighbors=5):

    xyz_obs = xyz_obs.detach().clone()
    _, nn_ix, _ = knn_points(
        xyz_can.unsqueeze(0), xyz_obs.unsqueeze(0), K=n_neighbors, return_sorted=True
    )
    nn_ix = nn_ix.squeeze(0)

    loss_xyz = aiap_loss(xyz_can, xyz_obs, nn_ix=nn_ix)

    return loss_xyz


def chamfer_loss(pc_trans_list, pc_list, chamfer_dist):
    # pc_trans_list, pc_list: [T-1, N, 3]
    # trans_list: [T-1, P, 4, 4]
    cd = chamfer_dist(pc_trans_list, pc_list)  # [T-1, N]
    dist_loss = torch.sum(cd)
    return dist_loss


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """

    device = xyz.device
    B, N, C = xyz.shape

    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points: indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]

    idx = idx.to(dtype=torch.long)

    batch_indices = torch.arange(B, dtype=torch.long).to(device).unsqueeze(1)

    batch_indices = batch_indices.expand(-1, idx.shape[1])

    new_points = points[batch_indices, idx, :]
    return new_points


def get_src_permutation_idx(indices):
    # permute predictions following indices
    batch_idx = torch.cat(
        [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
    )
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx


def get_tgt_permutation_idx(indices):
    # permute targets following indices
    batch_idx = torch.cat(
        [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]
    )
    tgt_idx = torch.cat([tgt for (_, tgt) in indices])
    return batch_idx, tgt_idx


class AABB(torch.nn.Module):
    def __init__(self, coord_max, coord_min):
        super().__init__()
        self.coord_max = coord_max.cuda()
        self.coord_min = coord_min.cuda()

    def normalize(self, x, sym=False):
        x = (x - self.coord_min) / (self.coord_max - self.coord_min)
        if sym:
            x = 2 * x - 1.0
        return x

    def unnormalize(self, x, sym=False):
        if sym:
            x = 0.5 * (x + 1)
        x = x * (self.coord_max - self.coord_min) + self.coord_min
        return x

    def clip(self, x):
        return x.clip(min=self.coord_min, max=self.coord_max)

    def volume_scale(self):
        return self.coord_max - self.coord_min

    def scale(self):
        return math.sqrt((self.volume_scale() ** 2).sum() / 3.0)


class GaussianTemplateModel(nn.Module):
    def __init__(
        self,
        template,
        add_bones: AdditionalBones,
        log_dir,
        w_correction_flag=True,
        f_localcode_dim=0,
        max_sph_order=0,
        w_memory_type="point",
        max_scale=0.1,  # use sigmoid activation, can't be too large
        min_scale=0.0,
        # geo init
        init_mode="random",
        opacity_init_value=0.9,  # the init value of opacity
        # on mesh init params
        onmesh_init_subdivide_num=0,
        onmesh_init_scale_factor=1.0,
        onmesh_init_thickness_factor=0.5,
        # near mesh init params
        scale_init_value=0.01,  # the init value of scale
        nearmesh_init_num=10000,
        nearmesh_init_std=0.1,
        eps=0.06,
        samples=80,
        use_l1=False,
        use_rotation=False,
        use_haus=False,
        use_emd=False,
        use_dynamic=False,
        dynamic_start_steps=20000,
        dynamic_end_steps=20000,
        inverse_start_steps=20000,
        inverse_end_steps=20000,
        emd_gap=10,
        lambda_l1=0,
        lambda_haus=0,
        lambda_emd=1,
        densify_interval=201,
        prune_interval=201,
        lambda_assign=3e-1,
        inverse_K=40,
        use_voxel=False,
        K=5,
        inverse_interval=50,
        threshold=0.5,
        delay=1000,
        lambda_skin=1,
        resolution=64,
        noise_step=2000,
        regaussian_steps=[],
        pose_noise=0.1,
        voxel=True,
        voxel_iterations=30,
        coe=0.7,
        dbs_steps=2000,
    ) -> None:
        super().__init__()

        self.log_dir = log_dir
        self.template = template

        self.add_bones = add_bones
        self.num_add_bones = add_bones.num_bones

        self.max_scale = max_scale
        self.min_scale = min_scale
        self._init_act(self.max_scale, self.min_scale)
        self.opacity_init_logit = self.o_inv_act(opacity_init_value)

        self.dbs_steps = dbs_steps
        self.eps = eps
        self.samples = samples
        self.use_l1 = use_l1
        self.use_rotation = use_rotation
        self.use_haus = use_haus
        self.use_emd = use_emd
        self.use_dynamic = use_dynamic
        self.dynamic_start_steps = dynamic_start_steps
        self.dynamic_end_steps = dynamic_end_steps
        self.inverse_start_steps = inverse_start_steps
        self.inverse_end_steps = inverse_end_steps
        self.lambda_l1 = lambda_l1
        self.lambda_haus = lambda_haus
        self.lambda_emd = lambda_emd
        self.densify_interval = densify_interval
        self.prune_interval = prune_interval
        self.lambda_assign = lambda_assign
        self.inverse_K = inverse_K
        self.use_voxel = use_voxel
        self.w_correction_flag = w_correction_flag
        self.emd_gap = emd_gap
        self.delay = delay
        self.lambda_skin = lambda_skin
        self.voxel = voxel

        self.INVERSE_INTERVAL = inverse_interval
        self.noise_step = noise_step
        self.regaussian_steps = regaussian_steps
        self.pose_noise = pose_noise

        self.threshold = threshold
        self.parents_idx = self.template._template_layer.parents
        self.joints = self.template.joints
        self.num_bones = self.joints.shape[1]

        if self.num_bones != 24:
            self.mode = "dog"
        else:
            self.mode = "human"

        self.PARENT_IDX = PARENT[self.mode]
        self.TORSO_JOINTS_IDX = TORSO_JOINTS[self.mode]

        self.gt_bone_points, self.gt_weights, _, _, _, _ = self.get_joints(
            self.joints, self.parents_idx, K=K
        )

        _, _, _, _, self.ellipsoid_samples, _ = self.get_joints(
            self.joints, self.parents_idx, K=1
        )

        self.connected_joints = {joint: set() for joint in range(self.num_bones)}
        for child, parent in self.PARENT_IDX.items():
            self.connected_joints[child].add(parent)
            self.connected_joints[parent].add(child)

        for joint in self.connected_joints:
            self.connected_joints[joint] = list(self.connected_joints[joint])

        # * init geometry
        if init_mode == "on_mesh":
            x, q, s, o = get_on_mesh_init_geo_values(
                template,
                on_mesh_subdivide=onmesh_init_subdivide_num,
                scale_init_factor=onmesh_init_scale_factor,
                thickness_init_factor=onmesh_init_thickness_factor,
                max_scale=max_scale,
                min_scale=min_scale,
                s_inv_act=self.s_inv_act,
                opacity_init_logit=self.opacity_init_logit,
            )
        elif init_mode == "near_mesh":
            self.scale_init_logit = self.s_inv_act(scale_init_value)
            x, q, s, o = get_near_mesh_init_geo_values(
                template,
                scale_base_logit=self.scale_init_logit,
                opacity_base_logit=self.opacity_init_logit,
                random_init_num=nearmesh_init_num,
                random_init_std=nearmesh_init_std,
            )
        elif init_mode == "in_mesh":
            self.scale_init_logit = self.s_inv_act(scale_init_value)
            x, q, s, o = get_inside_mesh_init_geo_values(
                template,
                scale_base_logit=self.scale_init_logit,
                opacity_base_logit=self.opacity_init_logit,
                random_init_num=nearmesh_init_num,
            )

        elif init_mode == "random":
            num_pts = 400
            print(f"Generating random point cloud ({num_pts})...")
            self.scale_init_logit = self.s_inv_act(scale_init_value)

            x = self.sample_gaussian_bone_points(self.joints[0], num_pts)
            range_mask = (torch.abs(x) <= 1).all(dim=1)
            x = x[range_mask]

            num_pts = x.shape[0]
            o = nn.Parameter(torch.ones(num_pts, 1) * self.opacity_init_logit)
            s = torch.ones(num_pts, 3) * self.scale_init_logit
            q = torch.zeros(num_pts, 4)
            q[:, 0] = 1.0

        elif init_mode == "random_space":
            num_pts = 20_000
            self.scale_init_logit = self.s_inv_act(scale_init_value)
            # We create random points inside the bounds of the synthetic Blender scenes
            x = np.random.random((num_pts, 3)) * 2.2 - 1.1
            o = nn.Parameter(torch.ones(num_pts, 1) * self.opacity_init_logit)
            s = torch.ones(num_pts, 3) * self.scale_init_logit
            q = torch.zeros(num_pts, 4)
            q[:, 0] = 1.0
        else:
            raise NotImplementedError(f"Unknown init_mode {init_mode}")

        x = torch.tensor(x).float().cuda()
        o = torch.tensor(o).float().cuda()
        s = torch.tensor(s).float().cuda()
        q = torch.tensor(q).float().cuda()

        self._xyz = nn.Parameter(x.requires_grad_(True))
        self._rotation = nn.Parameter(q.requires_grad_(True))
        self._scaling = nn.Parameter(s.requires_grad_(True))
        self._opacity = nn.Parameter(o.requires_grad_(True))

        self.tmp_x = x.detach().clone()
        self.w_memory_type = w_memory_type
        assert self.w_memory_type in ["point", "voxel"], f"Unknown {w_memory_type}"

        self.max_sph_order = max_sph_order
        self.w_dc_dim = self.template.dim if w_correction_flag else 0
        self.w_rest_dim = self.add_bones.num_bones
        self.f_localcode_dim = f_localcode_dim

        sph_rest_dim = 3 * (sph_order2nfeat(self.max_sph_order) - 1)
        self._features_dc = nn.Parameter(torch.zeros_like(self._xyz))
        self._features_rest = nn.Parameter(torch.zeros(self.N, sph_rest_dim))

        self._w_correction_dc = nn.Parameter(
            torch.zeros(self.N, self.w_dc_dim), requires_grad=False
        )
        self._w_correction_rest = nn.Parameter(
            torch.ones(self.N, self.w_rest_dim) * 1e-4
        )

        self._features_localcode = nn.Parameter(
            torch.zeros(self.N, self.f_localcode_dim)
        )

        assert (
            self.f_localcode_dim == 0
        ), "Local feature codes are not supported by this model"

        # warning, our code use N, instead of (N,1) as in GS code
        self.register_buffer("xyz_gradient_accum", torch.zeros(self.N).float())
        self.register_buffer("xyz_gradient_denom", torch.zeros(self.N).long())
        self.register_buffer("max_radii2D", torch.zeros(self.N).float())

        self.op_update_exclude = ["add_bones"]
        if self.w_memory_type != "point":
            self.op_update_exclude.extend(["w_dc_vox", "w_rest_vox", "mlp_params"])

        self.eps_s0 = 1e-8
        self.start_reg = False

        divisor = 8 if resolution == 128 else 4
        if self.use_voxel:
            self.voxel_deformer = VoxelDeformer(
                vtx=self.gt_bone_points,
                vtx_features=self.gt_weights,
                resolution_dhw=[
                    resolution // divisor,
                    resolution,
                    resolution,
                ],
                voxel=voxel,
                iterations=voxel_iterations,
                coe=coe,
            )
        else:
            self.voxel_deformer = Deformer(
                vtx=self.gt_bone_points,
                vtx_features=self.gt_weights,
                resolution_dhw=[
                    resolution // divisor,
                    resolution,
                    resolution,
                ],
            )
        self.bone_colors = self.get_distinct_colors(self.joints.shape[1])

        self.chamfer_dist = ChamferDistance()

        viz_base_R_opencv = np.asarray(euler2mat(np.pi, 0, 0, "sxyz"))

        viz_base_R_opencv = torch.from_numpy(viz_base_R_opencv).float()
        self.viz_base_R = viz_base_R_opencv.to(torch.device("cuda:0"))

        self.can_pose = self.template.real_canonical_pose.detach().to(
            torch.device("cuda:0")
        )

        if self.mode == "human":
            self.can_pose[0] = self.viz_base_R.to(self.can_pose.device)
            self.can_pose = matrix_to_axis_angle(self.can_pose)[None]
        else:
            self.can_pose[:3] = matrix_to_axis_angle(self.viz_base_R[None])[0]
            self.can_pose = self.can_pose[None]

        self.can_trans = torch.zeros(len(self.can_pose), 3).to(self.can_pose)
        self.can_trans[:, -1] = 3.0
        self.cnt = 0
        self.th = 0.3
        self.q = 0.85

        self.posed_mu = None
        self.posed_frame = None
        self.posed_W = None

        return

    def rotation_matrix_axis_angle(self, axis, angle):
        """
        根据旋转轴和角度计算旋转矩阵。
        """
        axis = axis / torch.norm(axis)
        K = torch.tensor(
            [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]],
            device=axis.device,
            dtype=axis.dtype,
        )
        R = (
            torch.eye(3, device=axis.device, dtype=axis.dtype)
            + torch.sin(angle) * K
            + (1 - torch.cos(angle)) * torch.matmul(K, K)
        )
        return R

    def rotation_from_y(self, v):
        """
        计算将 y 轴向量 [0, 1, 0] 旋转到向量 v 的旋转矩阵。
        """
        v = v / torch.norm(v)
        y_axis = torch.tensor([0, 1, 0], device=v.device, dtype=v.dtype)
        axis = torch.cross(y_axis, v)
        axis_norm = torch.norm(axis)
        if axis_norm < 1e-6:
            # y_axis 和 v 平行
            if torch.dot(y_axis, v) > 0:
                return torch.eye(3, device=v.device, dtype=v.dtype)
            else:
                # 180 度旋转
                return torch.tensor(
                    [[-1, 0, 0], [0, -1, 0], [0, 0, 1]], device=v.device, dtype=v.dtype
                )
        axis = axis / axis_norm
        angle = torch.acos(torch.clamp(torch.dot(y_axis, v), -1.0, 1.0))
        return self.rotation_matrix_axis_angle(axis, angle)

    def sample_gaussian_bone_points(self, joints, num_points_per_bone=300):
        """
        从近似的骨骼高斯分布中采样点云。

        Args:
            joints: 形状为 (Total_Joints, 3) 的张量，表示关节位置。
            num_points_per_bone: 每个骨骼采样的点数，默认 300。

        Returns:
            形状为 (Total_Samples, 3) 的张量，包含采样的点云。
        """
        total_joints = joints.shape[0]
        device = joints.device
        samples = []

        BONE_STDS_TENSOR = torch.from_numpy(BONE_STDS).to(
            device=device, dtype=joints.dtype
        )
        HEAD_STDS_TENSOR = torch.from_numpy(HEAD_STDS).to(
            device=device, dtype=joints.dtype
        )
        JOINT_STDS_TENSOR = torch.from_numpy(JOINT_STDS).to(
            device=device, dtype=joints.dtype
        )

        # 处理骨骼（父关节和子关节之间）
        for bone_idx, parent_idx in self.PARENT_IDX.items():
            parent_joint = joints[parent_idx]
            child_joint = joints[bone_idx]
            bone_center = (parent_joint + child_joint) / 2.0
            bone_direction = child_joint - parent_joint
            bone_length = torch.norm(bone_direction)
            bone_direction = bone_direction / bone_length

            # 生成高斯椭球的尺度因子
            bone_stds = BONE_STDS_TENSOR.clone()
            bone_stds[1] = bone_length / 2.0  # 将 y 轴尺度设置为骨骼长度的一半

            # 调整躯干关节的尺度
            if parent_idx in self.TORSO_JOINTS_IDX:
                bone_stds[0] *= 1 / 1.5
                bone_stds[2] *= 1 / 1.5

            scale_factors = bone_stds * 2.0  # 标准差乘以 2

            # 计算旋转矩阵，将 y 轴旋转到骨骼方向
            rotation_matrix = self.rotation_from_y(bone_direction)

            # 在局部坐标系中采样点
            local_samples = torch.randn(
                num_points_per_bone, 3, device=device
            ) * scale_factors.unsqueeze(0)

            # 应用旋转和平移，转换到全局坐标系
            world_samples = local_samples @ rotation_matrix.T + bone_center.unsqueeze(0)

            samples.append(world_samples)

        all_joint_indices = set(range(total_joints))
        parent_joint_indices = set(self.PARENT_IDX.values())
        # 末端关节索引为不在父关节索引集合中的关节
        end_joints = all_joint_indices - parent_joint_indices

        # 处理末端关节
        for joint_idx in end_joints:
            joint_center = joints[joint_idx]
            if self.mode == "human":
                if joint_idx == SMPL_JOINT_IDX["head"]:
                    joint_stds = HEAD_STDS_TENSOR
                else:
                    joint_stds = JOINT_STDS_TENSOR
            else:
                joint_stds = JOINT_STDS_TENSOR

            scale_factors = joint_stds * 2.0
            local_samples = torch.randn(
                num_points_per_bone, 3, device=device
            ) * scale_factors.unsqueeze(0)
            world_samples = local_samples + joint_center.unsqueeze(0)
            samples.append(world_samples)

        all_samples = torch.cat(samples, dim=0)
        return all_samples

    def get_joints(self, joints, parent_id, K=10):
        num_batch, num_joints, _ = joints.shape

        if not isinstance(parent_id, torch.Tensor):
            parent_id = torch.tensor(parent_id)

        samples = []
        weights = []
        new_joints = joints.clone()  # 复制原始joints用于添加虚拟节点
        new_parent_id = parent_id.clone()  # 复制原始parent_id用于添加虚拟节点的关系

        ellipsoid_samples = [None] * num_joints
        ellipsoid_weights = [None] * num_joints

        # 识别每个关节是否有子节点
        has_child = torch.zeros(num_joints, dtype=torch.bool, device=joints.device)
        for idx in range(num_joints):
            if parent_id[idx] != -1:
                has_child[parent_id[idx]] = True

        # 添加每个关节本身作为样本点，并且权重完全归自己所有
        for k in range(num_joints):
            samples.append(joints[:, k])
            weight = torch.zeros((num_batch, num_joints), device=joints.device)
            weight[:, k] = 1
            weights.append(weight)

            # 如果关节有有效的父关节并且不是根节点，则进行中间点采样
            if k == 0:
                ellipsoid_samples[k] = joints[:, k]
                weight = torch.zeros((num_batch, num_joints), device=joints.device)
                weight[:, k] = 1
                ellipsoid_weights[k] = weight
            else:
                parent_joint = joints[:, parent_id[k]]
                child_joint = joints[:, k]

                for i in range(1, K + 1):
                    t = i / (K + 1)
                    sample_point = (1 - t) * parent_joint + t * child_joint
                    samples.append(sample_point)

                    # 中间采样点的权重归父节点所有
                    weight = torch.zeros((num_batch, num_joints), device=joints.device)
                    weight[:, parent_id[k]] = 1
                    weights.append(weight)

                    if ellipsoid_samples[parent_id[k]] is None:
                        ellipsoid_samples[parent_id[k]] = sample_point
                        ellipsoid_weights[parent_id[k]] = weight

                # 检查当前节点是否为叶子节点
                if not has_child[k]:
                    # 添加虚拟子节点，此节点位于关节和父节点连线的延长线上
                    virtual_child_joint = (
                        child_joint + (child_joint - parent_joint) * 0.5
                    )

                    virtual_real_joint = child_joint + (child_joint - parent_joint)

                    samples.append(virtual_child_joint)

                    # 虚拟子节点的权重归当前节点所有
                    weight = torch.zeros((num_batch, num_joints), device=joints.device)
                    weight[:, k] = 1
                    weights.append(weight)

                    virtual_node_index = new_joints.shape[1]
                    new_joints = torch.cat(
                        [new_joints, virtual_real_joint.unsqueeze(1)], dim=1
                    )
                    new_parent_id = torch.cat(
                        [new_parent_id, torch.tensor([k], device=joints.device)]
                    )

                    ellipsoid_samples[k] = virtual_child_joint
                    ellipsoid_weights[k] = weight

        samples = torch.stack(samples, dim=1).cuda()
        weights = torch.stack(weights, dim=1).cuda()
        ellipsoid_samples = torch.stack(
            [s for s in ellipsoid_samples if s is not None], dim=1
        ).cuda()
        ellipsoid_weights = torch.stack(
            [w for w in ellipsoid_weights if w is not None], dim=1
        ).cuda()

        return (
            samples,
            weights,
            new_joints,
            new_parent_id,
            ellipsoid_samples,
            ellipsoid_weights,
        )

    def compute_distances_to_joints(self, samples, joints):
        """
        计算每个采样点到24个关节点的距离。

        参数:
        - samples: 采样点，形状为 (n * samples_per_segment, 3)。
        - joints: 关节点坐标，形状为 (24, 3)。

        返回:
        - 一个形状为 (n * samples_per_segment, 24) 的距离矩阵。
        """
        samples_expanded = samples.unsqueeze(
            1
        )  # 形状变为 (n * samples_per_segment, 1, 3)
        joints_expanded = joints.unsqueeze(0)  # 形状变为 (1, 24, 3)

        distances = torch.sqrt(((samples_expanded - joints_expanded) ** 2).sum(dim=2))

        return distances

    def get_posed(self, theta, trans):  # for template_model
        B = len(theta)

        mu_can = self._xyz[None].expand(B, -1, -1)  # canonical的顶点位置
        frame_can = quaternion_to_matrix(self._rotation)[None].expand(B, -1, -1, -1)

        _, A = self.template.forward(
            theta, mu_can
        )  # W 分配给每个gs权重矩阵[1,15550,35] A 当前pose相对于canonical pose的每个joint的变换矩阵，包含位移和旋转信息[1,35,4,4] theta 1,112

        W = torch.clamp(self._w_correction_dc[None], min=1e-8)  # 1,10000,24
        W = W / torch.sum(W, dim=2, keepdim=True)
        T = torch.einsum(
            "bnj, bjrc -> bnrc", W[..., : self.num_bones], A
        )  # [1,15550,4,4] ,lbs之后的每个gs的rotation矩阵

        R, t = (
            T[:, :, :3, :3],
            T[:, :, :3, 3],
        )  # B,N,3,3; B,N,3   Batch=1 得到对于每个gs的R和T

        mu = torch.einsum("bnij,bnj->bni", R, mu_can) + t  # B,N,3
        frame = torch.einsum("bnij,bnjk->bnik", R, frame_can)  # B,N,3,3

        mu = mu + trans[:, None, :]

        return mu[0], frame[0], R, t

    def poseCluster(self, index, pose, trans):
        xyz, rotations, _, _ = self.get_posed(pose, trans)
        for bone_index, vertex_indices in self.bone_vertex_dicts.items():
            self.bone_vertex_coordinates[bone_index] = xyz[vertex_indices]
            self.bone_vertex_rotations[bone_index] = rotations[vertex_indices]
            store_path = os.path.join(f"output/{index}_parts", f"part_{bone_index}.ply")
            self.save_ply(
                store_path,
                self.bone_vertex_coordinates[bone_index],
                self.bone_vertex_rotations[bone_index],
            )
        return

    def get_distinct_colors(self, num_colors):
        """
        生成指定数量的不同颜色。

        参数:
            num_colors (int): 需要生成的颜色数量。

        返回:
            np.ndarray: 形状为 (num_colors, 4) 的RGBA颜色数组。
        """
        if num_colors <= 32:
            colors_tab20 = plt.cm.get_cmap("tab20", 20)
            colors_set3 = plt.cm.get_cmap("Set3", 12)
            colors = np.vstack(
                (
                    colors_tab20(np.linspace(0, 1, 20)),
                    colors_set3(np.linspace(0, 1, 12)),
                )
            )
            return colors[:num_colors]
        else:
            # 使用 distinctipy 生成不同颜色
            rgb_colors = distinctipy.get_colors(
                num_colors, rng=42
            )  # rng 为随机种子，可选
            rgba_colors = np.array([(*color, 1.0) for color in rgb_colors])
            return rgba_colors

    def show_bone_masks(self, imgpath, pts, masks, export=False):
        bone_colors = self.bone_colors
        points_np = pts.detach().clone().cpu().numpy()  # n,3

        colors = np.zeros((points_np.shape[0], 4))
        colors[:, 3] = 1  # 设置alpha为1

        for j, mask in enumerate(masks):
            mask_np = mask.clone().cpu().numpy()  # 将mask转换为numpy数组
            colors[mask_np, :] = bone_colors[j]  # 分配关节颜色

        # 进行坐标转换
        points_np[:, [1, 2]] = points_np[:, [2, 1]]
        points_np[:, 1] = -points_np[:, 1]  # 翻转Y轴
        points_np[:, 2] = -points_np[:, 2]  # 翻转Z轴

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")
        sc = ax.scatter(
            points_np[:, 0], points_np[:, 1], points_np[:, 2], c=colors, s=0.5
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_box_aspect([1, 1, 1])  # 确保轴比例相同

        max_range = (
            np.array(
                [
                    points_np[:, 0].max() - points_np[:, 0].min(),
                    points_np[:, 1].max() - points_np[:, 1].min(),
                    points_np[:, 2].max() - points_np[:, 2].min(),
                ]
            ).max()
            / 2.0
        )
        mid_x = (points_np[:, 0].max() + points_np[:, 0].min()) * 0.5
        mid_y = (points_np[:, 1].max() + points_np[:, 1].min()) * 0.5
        mid_z = (points_np[:, 2].max() + points_np[:, 2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        fig.savefig(imgpath)

        if export:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_np)
            pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
            ply_path = imgpath.replace(".png", ".ply")
            o3d.io.write_point_cloud(ply_path, pcd)

    def show_bone_weights_th(self, imgpath, pts, weights, export=False, threshold=0.8):
        bone_colors = self.bone_colors
        points_np = pts.detach().clone().cpu().numpy()  # n,3
        weights_np = weights.detach().clone().cpu().numpy()  # n,24

        colors = np.zeros((points_np.shape[0], 4))
        colors[:, 3] = 1  # 设置alpha为1

        for j in range(weights_np.shape[1]):
            mask = weights_np[:, j] > threshold
            colors[mask, :] = bone_colors[j]  # 分配关节颜色

        # 进行坐标转换
        points_np[:, [1, 2]] = points_np[:, [2, 1]]
        points_np[:, 1] = -points_np[:, 1]  # 翻转Y轴
        points_np[:, 2] = -points_np[:, 2]  # 翻转Z轴

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")
        sc = ax.scatter(
            points_np[:, 0], points_np[:, 1], points_np[:, 2], c=colors, s=0.5
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_box_aspect([1, 1, 1])  # 确保轴比例相同

        max_range = (
            np.array(
                [
                    points_np[:, 0].max() - points_np[:, 0].min(),
                    points_np[:, 1].max() - points_np[:, 1].min(),
                    points_np[:, 2].max() - points_np[:, 2].min(),
                ]
            ).max()
            / 2.0
        )
        mid_x = (points_np[:, 0].max() + points_np[:, 0].min()) * 0.5
        mid_y = (points_np[:, 1].max() + points_np[:, 1].min()) * 0.5
        mid_z = (points_np[:, 2].max() + points_np[:, 2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        fig.savefig(imgpath)

        if export:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_np)
            pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
            ply_path = imgpath.replace(".png", ".ply")
            o3d.io.write_point_cloud(ply_path, pcd)

        plt.close(fig)

    def show_bone_weights_max(self, imgpath, pts, weights, export=False):
        bone_colors = self.bone_colors
        points_np = pts.detach().clone().cpu().numpy()  # n,3
        weights_np = weights.detach().clone().cpu().numpy()  # n,24

        colors = np.zeros((points_np.shape[0], 4))
        colors[:, 3] = 1  # 设置alpha为1

        max_weight_joints = np.argmax(weights_np, axis=1)

        for i, joint in enumerate(max_weight_joints):
            colors[i, :] = bone_colors[joint]

        # 进行坐标转换
        points_np[:, [1, 2]] = points_np[:, [2, 1]]
        points_np[:, 1] = -points_np[:, 1]  # 翻转Y轴
        points_np[:, 2] = -points_np[:, 2]  # 翻转Z轴

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")
        sc = ax.scatter(
            points_np[:, 0], points_np[:, 1], points_np[:, 2], c=colors, s=0.5
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_box_aspect([1, 1, 1])  # 确保轴比例相同

        max_range = (
            np.array(
                [
                    points_np[:, 0].max() - points_np[:, 0].min(),
                    points_np[:, 1].max() - points_np[:, 1].min(),
                    points_np[:, 2].max() - points_np[:, 2].min(),
                ]
            ).max()
            / 2.0
        )
        mid_x = (points_np[:, 0].max() + points_np[:, 0].min()) * 0.5
        mid_y = (points_np[:, 1].max() + points_np[:, 1].min()) * 0.5
        mid_z = (points_np[:, 2].max() + points_np[:, 2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        fig.savefig(imgpath)

        if export:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_np)
            pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
            ply_path = imgpath.replace(".png", ".ply")
            o3d.io.write_point_cloud(ply_path, pcd)

        plt.close(fig)

    def showTmpBoneWeight(
        self,
        imgpath,
        pts,
        weights,
        update_mask=None,
        export=False,
        blend=True,
        joints=None,
    ):
        bone_colors = self.bone_colors
        points_np = pts.detach().clone().cpu().numpy()  # n,3

        weights_np = weights.detach().clone().cpu().numpy()
        max_indices = np.argmax(weights_np[0], axis=1)

        if joints is not None:
            joints_np = joints.detach().clone().cpu().numpy()[0]
            joints_np[:, [1, 2]] = joints_np[:, [2, 1]]

            joints_np[:, 1] = -joints_np[:, 1]  # 翻转Y轴
            joints_np[:, 2] = -joints_np[:, 2]  # 翻转Z轴

        if blend:
            colors = np.dot(weights_np[0], bone_colors)
            colors = np.clip(colors, 0, 1)
        else:
            colors = bone_colors[
                max_indices
            ]  # Select colors directly based on max_indices
            colors = np.clip(colors, 0, 1)  # n,4

        points_np[:, [1, 2]] = points_np[:, [2, 1]]
        points_np[:, 1] = -points_np[:, 1]  # Flip Y axis
        points_np[:, 2] = -points_np[:, 2]  # Flip Z axis

        if update_mask is not None:
            mask = update_mask.clone().cpu()  # n
            colors[mask, :] = [0, 0, 0, 1]  # Set masked colors to black (RGB=0, A=1)

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")
        sc = ax.scatter(
            points_np[:, 0], points_np[:, 1], points_np[:, 2], c=colors, s=0.5
        )

        if joints is not None:
            ax.scatter(
                joints_np[:, 0], joints_np[:, 1], joints_np[:, 2], c="black", s=50
            )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_box_aspect([1, 1, 1])  # Ensure axis proportions are the same

        max_range = (
            np.array(
                [
                    points_np[:, 0].max() - points_np[:, 0].min(),
                    points_np[:, 1].max() - points_np[:, 1].min(),
                    points_np[:, 2].max() - points_np[:, 2].min(),
                ]
            ).max()
            / 2.0
        )
        mid_x = (points_np[:, 0].max() + points_np[:, 0].min()) * 0.5
        mid_y = (points_np[:, 1].max() + points_np[:, 1].min()) * 0.5
        mid_z = (points_np[:, 2].max() + points_np[:, 2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        fig.savefig(imgpath)

        def create_sphere_points(center, radius, n_points=50):
            """创建一个球体的点云"""
            phi = np.random.uniform(0, 2 * np.pi, n_points)
            costheta = np.random.uniform(-1, 1, n_points)
            theta = np.arccos(costheta)
            x = radius * np.sin(theta) * np.cos(phi) + center[0]
            y = radius * np.sin(theta) * np.sin(phi) + center[1]
            z = radius * np.cos(theta) + center[2]
            return np.column_stack((x, y, z))

        if export:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_np)
            pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
            pcd.normals = o3d.utility.Vector3dVector(
                np.vstack(
                    (max_indices.astype(float), np.zeros((2, len(max_indices))))
                ).T
            )

            if joints is not None:
                joint_points = []
                joint_colors = []
                joint_normals = []

                for joint in joints_np:
                    sphere_points = create_sphere_points(
                        joint, radius=0.01, n_points=100
                    )
                    joint_points.extend(sphere_points)
                    joint_colors.extend([[1, 0, 0]] * len(sphere_points))  # 红色
                    joint_normals.extend(
                        [[-1, -1, -1]] * len(sphere_points)
                    )  # 特殊标记

                all_points = np.vstack((points_np, joint_points))
                all_colors = np.vstack((colors[:, :3], joint_colors))
                all_normals = np.vstack((pcd.normals, joint_normals))

                pcd.points = o3d.utility.Vector3dVector(all_points)
                pcd.colors = o3d.utility.Vector3dVector(all_colors)
                pcd.normals = o3d.utility.Vector3dVector(all_normals)

            ply_path = imgpath.replace(".png", ".ply")
            o3d.io.write_point_cloud(ply_path, pcd)

    def showBoneWeight(
        self, imgpath, pts, step=50000, export=False, showPose=None, trans=None
    ):
        bone_colors = self.bone_colors
        points_np = pts.detach().clone().cpu().numpy()  # n,3

        grid = self.voxel_deformer.grid_denorm

        W = self.voxel_deformer(self.get_x[None].detach().clone())
        if self.w_correction_flag:
            W = W + self._w_correction_dc[None]
        W = torch.clamp(W, min=1e-8)
        W = W / torch.sum(W, dim=2, keepdim=True)
        weights_np = W.cpu().numpy()  # 1,n,24

        colors = np.dot(weights_np[0], bone_colors)
        colors = np.clip(colors, 0, 1)

        points_np[:, [1, 2]] = points_np[:, [2, 1]]

        points_np[:, 1] = -points_np[:, 1]  # 翻转Y轴
        points_np[:, 2] = -points_np[:, 2]  # 翻转Z轴

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")

        sc = ax.scatter(
            points_np[:, 0], points_np[:, 1], points_np[:, 2], c=colors, s=0.5
        )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # 设置轴的相对比例，使x、y、z轴在视觉上保持一致的刻度
        ax.set_box_aspect([1, 1, 1])  # 确保各轴比例相同

        max_range = (
            np.array(
                [
                    points_np[:, 0].max() - points_np[:, 0].min(),
                    points_np[:, 1].max() - points_np[:, 1].min(),
                    points_np[:, 2].max() - points_np[:, 2].min(),
                ]
            ).max()
            / 2.0
        )
        mid_x = (points_np[:, 0].max() + points_np[:, 0].min()) * 0.5
        mid_y = (points_np[:, 1].max() + points_np[:, 1].min()) * 0.5
        mid_z = (points_np[:, 2].max() + points_np[:, 2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        fig.savefig(imgpath)

        if export:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_np)
            pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
            ply_path = imgpath.replace(".png", ".ply")
            o3d.io.write_point_cloud(ply_path, pcd)

    def showSmplBoneWeight(self, imgpath, pts, step=50000):
        bone_colors = self.bone_colors
        points_np = pts.detach().clone().cpu().numpy()  # n,3

        W = self.template.smpl_voxel_deformer(self.get_x[None].detach().clone())
        weights_np = W.cpu().numpy()  # 1,n,24
        colors = np.dot(weights_np[0], bone_colors)
        colors = np.clip(colors, 0, 1)

        points_np[:, [1, 2]] = points_np[:, [2, 1]]

        points_np[:, 1] = -points_np[:, 1]  # 翻转Y轴
        points_np[:, 2] = -points_np[:, 2]  # 翻转Z轴

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")

        sc = ax.scatter(
            points_np[:, 0], points_np[:, 1], points_np[:, 2], c=colors, s=0.5
        )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # 设置轴的相对比例，使x、y、z轴在视觉上保持一致的刻度
        ax.set_box_aspect([1, 1, 1])  # 确保各轴比例相同

        max_range = (
            np.array(
                [
                    points_np[:, 0].max() - points_np[:, 0].min(),
                    points_np[:, 1].max() - points_np[:, 1].min(),
                    points_np[:, 2].max() - points_np[:, 2].min(),
                ]
            ).max()
            / 2.0
        )
        mid_x = (points_np[:, 0].max() + points_np[:, 0].min()) * 0.5
        mid_y = (points_np[:, 1].max() + points_np[:, 1].min()) * 0.5
        mid_z = (points_np[:, 2].max() + points_np[:, 2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        fig.savefig(imgpath)

    def save_ply(self, path, _xyz, _rotation):
        mkdir_p(os.path.dirname(path))

        xyz = _xyz.detach().cpu().numpy()
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(xyz)
        o3d.io.write_point_cloud(path, point_cloud)

        print(f"Point cloud saved to {path}")

    def summary(self):
        msg = ""
        for name, param in self.named_parameters():
            if name.startswith("add_bones"):
                continue  # compact print
            msg = msg + f"[{name}:{param.numel()/1e3:.1f}K] "
        logging.info(msg)
        return

    def _init_act(self, max_s_value, min_s_value):
        def s_act(x):
            if isinstance(x, float):
                x = torch.tensor(x).squeeze()
            return min_s_value + torch.sigmoid(x) * (max_s_value - min_s_value)

        def s_inv_act(x):
            if isinstance(x, float):
                x = torch.tensor(x).squeeze()
            y = (x - min_s_value) / (max_s_value - min_s_value) + 1e-5
            y = torch.logit(y)
            assert not torch.isnan(
                y
            ).any(), f"{x.min()}, {x.max()}, {y.min()}, {y.max()}"
            return y

        def o_act(x):
            if isinstance(x, float):
                x = torch.tensor(x).squeeze()
            return torch.sigmoid(x)

        def o_inv_act(x):
            if isinstance(x, float):
                x = torch.tensor(x).squeeze()
            return torch.logit(x)

        self.s_act = s_act
        self.s_inv_act = s_inv_act
        self.o_act = o_act
        self.o_inv_act = o_inv_act
        self.rotation_activation = torch.nn.functional.normalize
        return

    @property
    def N(self):
        return len(self._xyz)

    @property
    def get_x(self):
        return self._xyz

    @property
    def get_R(self):
        return quaternion_to_matrix(self._rotation)

    @property
    def get_o(self):
        return self.o_act(self._opacity)

    @property
    def get_s(self):
        return self.s_act(self._scaling)

    @property
    def get_c(self):
        return torch.cat([self._features_dc, self._features_rest], dim=-1)

    def voxel_downsample(self, points, features, voxel_size):
        points = points.detach().clone().cpu().numpy()
        features = features.detach().clone().cpu().numpy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # 执行体素下采样
        down_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

        down_points = np.asarray(down_pcd.points)

        # 寻找原始点云中每个下采样点的最近邻点，用以提取对应的特征
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        down_features = []
        for point in down_points:
            _, idx, _ = pcd_tree.search_knn_vector_3d(point, 1)
            down_features.append(features[idx[0]])

        down_features = np.array(down_features)
        down_points = torch.from_numpy(down_points).float().to(device="cuda")
        down_features = torch.from_numpy(down_features).float().to(device="cuda")
        return down_points, down_features

    def voxel_sample_points_and_weights(
        self, mu_can, weights, org_weights, num_samples=100
    ):
        sampled_points = []
        sampled_weights = []
        sampled_org_weights = []
        num_bones = weights.shape[1]

        mu_can_np = mu_can.detach().cpu().numpy()
        weights_np = weights.detach().cpu().numpy()
        org_weights_np = org_weights.detach().cpu().numpy()

        for bone in range(num_bones):
            bone_points_indices = np.where(weights_np[:, bone] == 1)[0]
            bone_points = mu_can_np[bone_points_indices]

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(bone_points)

            voxel_size = 0.01
            pcd_down = pcd.voxel_down_sample(voxel_size)

            if len(pcd_down.points) > num_samples:
                chosen_indices = np.random.choice(
                    len(pcd_down.points), num_samples, replace=False
                )
            else:
                chosen_indices = np.arange(len(pcd_down.points))

            sampled_points.extend(bone_points[chosen_indices])
            sampled_indices = bone_points_indices[
                chosen_indices
            ]  # 转换回原始点云中的索引
            sampled_weights.append(weights_np[sampled_indices])
            sampled_org_weights.append(org_weights_np[sampled_indices])

        merged_points = torch.tensor(sampled_points, device=mu_can.device)
        merged_weights = torch.tensor(np.vstack(sampled_weights), device=mu_can.device)
        merged_org_weights = torch.tensor(
            np.vstack(sampled_org_weights), device=mu_can.device
        )

        return merged_points, merged_weights, merged_org_weights

    def compute_hausdorff_loss(self, points1, points2):
        dist1 = torch.cdist(points1, points2)
        dist2 = torch.cdist(points2, points1)
        max_dist1 = torch.max(torch.min(dist1, dim=1).values)  # 最小距离中的最大值
        max_dist2 = torch.max(torch.min(dist2, dim=1).values)  # 最小距离中的最大值
        return max_dist1 + max_dist2

    def get_posed_pts(self, pts, theta, trans=None, step=50000):

        B = len(theta)

        xyz = pts.detach().clone().cuda()
        mu_can = xyz  # 15550,3

        _, A, joints = self.template.forward(
            theta, step
        )  # W 分配给每个gs权重矩阵[1,15550,35] A 当前pose相对于canonical pose的每个joint的变换矩阵，包含位移和旋转信息[1,35,4,4] theta 1,112

        mu_can = mu_can[None].expand(B, -1, -1)  # canonical的顶点位置

        W = self.voxel_deformer(mu_can)
        W = torch.clamp(W, min=1e-8)
        W = W / torch.sum(W, dim=2, keepdim=True)

        T = torch.einsum(
            "bnj, bjrc -> bnrc", W[..., : self.num_bones], A
        )  # [1,15550,4,4] ,lbs之后的每个gs的rotation矩阵

        R, t = (
            T[:, :, :3, :3],
            T[:, :, :3, 3],
        )  # B,N,3,3; B,N,3   Batch=1 得到对于每个gs的R和T

        mu = torch.einsum("bnij,bnj->bni", R, mu_can) + t  # B,N,3

        if trans is not None:
            mu = mu + trans[:, None, :]
        return mu

    def forward(
        self,
        theta,
        trans,
        additional_dict={},
        active_sph_order=None,
        fast=False,
        step=-1,
        is_noise=False,
    ):
        # * fast will use the cached per point attr, no query anymore

        # theta: B,24,3; trans: B,3.
        B = len(theta)
        if active_sph_order is None:
            active_sph_order = self.max_sph_order
        else:
            assert (
                active_sph_order <= self.max_sph_order
            ), "active_sph_order should be smaller"
        sph_dim = 3 * sph_order2nfeat(active_sph_order)

        xyz = self.get_x
        mu_can = xyz  # 15550,3
        frame_can = self.get_R  # 15550,3,3 canonical的旋转朝向
        s = self.get_s  # n,3
        o = self.get_o
        sph = self.get_c[:, :sph_dim]

        if self.mode == "human":
            if not (
                (step == self.inverse_start_steps)
                or (step % self.INVERSE_INTERVAL == 0)
                or (step % self.densify_interval == 1)
                or (step % self.prune_interval == 1)
            ):
                if (
                    step < (self.inverse_start_steps + self.noise_step)
                    and step > (self.inverse_start_steps)
                    and np.random.uniform() <= 0.5
                ):
                    pose_mat = axis_angle_to_matrix(theta).clone()

                    pose_mat = (
                        pose_mat
                        + torch.randn(pose_mat.shape, device=pose_mat.device)
                        * self.pose_noise
                    )

                    _u, _s, _v = torch.svd(pose_mat)
                    pose_mat = torch.matmul(_u, _v.transpose(-2, -1))

                    theta = matrix_to_axis_angle(pose_mat)

        if fast:
            # only forward skeleton, no query voxel
            _, A, _ = self.template.forward(theta, step)
            W = self._cached_W[None].expand(B, -1, -1)

        else:  # pose , can  --forward-wrap->
            _, A, joints = self.template.forward(
                theta, step
            )  # W 分配给每个gs权重矩阵[1,15550,35] A 当前pose相对于canonical pose的每个joint的变换矩阵，包含位移和旋转信息[1,35,4,4] theta 1,112

        mu_can = mu_can[None].expand(B, -1, -1)  # canonical的顶点位置
        frame_can = frame_can[None].expand(B, -1, -1, -1)  # 1,10000,3,3

        W = self.voxel_deformer(mu_can)
        if self.w_correction_flag:
            W = W + self._w_correction_dc[None]  # 1,10000,23

        W = torch.clamp(W, min=1e-8)
        W = W / torch.sum(W, dim=2, keepdim=True)

        W_joints = self.voxel_deformer(self.gt_bone_points)
        W_joints = torch.clamp(W_joints, min=1e-8)
        W_joints = W_joints / torch.sum(W_joints, dim=2, keepdim=True)
        loss_bone_w = torch.nn.functional.mse_loss(W_joints, self.gt_weights)

        T = torch.einsum(
            "bnj, bjrc -> bnrc", W[..., : self.num_bones], A
        )  # [1,15550,4,4] ,lbs之后的每个gs的rotation矩阵

        R, t = (
            T[:, :, :3, :3],
            T[:, :, :3, 3],
        )  # B,N,3,3; B,N,3   Batch=1 得到对于每个gs的R和T

        mu = torch.einsum("bnij,bnj->bni", R, mu_can) + t  # B,N,3
        frame = torch.einsum("bnij,bnjk->bnik", R, frame_can)  # B,N,3,3

        loss_l1_l2 = torch.tensor(0).cuda()
        loss_rotation = torch.tensor(0).cuda()
        loss_emd = torch.tensor(0).cuda()
        loss_haus = torch.tensor(0).cuda()

        if step >= self.inverse_start_steps and step < self.inverse_end_steps:
            for i in range(1):
                if (
                    (step == self.inverse_start_steps)
                    or (step % self.INVERSE_INTERVAL == 0)
                    or (step % self.densify_interval == 1)
                    or (step % self.prune_interval == 1)
                    or ((step) in self.regaussian_steps)
                ):

                    joints_skeleton = self.get_posed_pts(self.joints[0], theta)

                    for q in [self.q]:  # 分位数太低，都被排除了，太高没作用
                        for cov_th in [0.2]:  # cov 考虑权重的阈值
                            for threshold in [self.th]:  # 相似度距离的阈值
                                for t in [1]:
                                    for iter in [5]:
                                        if t:
                                            joints = self.get_posed_pts(
                                                self.joints[0], theta
                                            )

                                        else:
                                            joints = self.get_posed_pts(
                                                self.ellipsoid_samples[0], theta
                                            )

                                        if (self.cnt + 1) % 3 == 0 and (self.th > 0.19):
                                            self.th = self.th - 0.03
                                        if self.th < 0.2:
                                            self.q = 0.9

                                        device = mu.device
                                        self.bone_directions = (
                                            joints_skeleton[:, self.parents_idx]
                                            - joints_skeleton
                                        )
                                        bone_directions_norm = torch.norm(
                                            self.bone_directions, dim=2, keepdim=True
                                        )
                                        self.bone_directions = torch.where(
                                            bone_directions_norm > 1e-6,
                                            self.bone_directions / bone_directions_norm,
                                            torch.tensor(
                                                [0.0, 1.0, 0.0], device=device
                                            ),
                                        )
                                        (
                                            update_weight,
                                            mask,
                                            valid_mask,
                                            estimated_weight,
                                        ) = self.update_weights_js(
                                            mu.detach().clone(),
                                            W.detach().clone(),
                                            joints.clone(),
                                            js_threshold=threshold,
                                            cov_th=cov_th,
                                            q=q,
                                            iteration_threshold=iter,
                                        )
                                        self.mask = mask
                                        self.update_weight = update_weight

                                        self.inv_mu_can, self.inv_frame_can = (
                                            self.inverse_lbs(
                                                mu,
                                                frame,
                                                A.detach().clone(),
                                                update_weight.detach().clone(),
                                            )
                                        )

                                        self.est_inv_mu_can, self.est_inv_frame_can = (
                                            self.inverse_lbs(
                                                mu,
                                                frame,
                                                A.detach().clone(),
                                                estimated_weight.detach().clone(),
                                            )
                                        )

                                        # Optional inverse-skinning diagnostics (images and point clouds).
                                        if getattr(self, "debug_visualizations", False):
                                            new_can = self.get_posed_pts(
                                                self.inv_mu_can[0], self.can_pose
                                            )
                                            estimate_new_can = self.get_posed_pts(
                                                self.est_inv_mu_can[0], self.can_pose
                                            )
                                            log_dir = osp.join(
                                                self.log_dir,
                                                f"js_new_prob_{cov_th}_th_{threshold}_q_{q}_t_{t}_step_{step}_iter_{iter}",
                                            )
                                            if not os.path.exists(log_dir):
                                                os.mkdir(log_dir)
                                            self.showTmpBoneWeight(
                                                f"{log_dir}/gmm_obs_iter_{iter}_{cov_th}_org_{step}.png",
                                                mu[0].detach().clone(),
                                                W.detach().clone(),
                                                export=True,
                                            )
                                            self.showTmpBoneWeight(
                                                f"{log_dir}/gmm_obs_iter_{iter}_{cov_th}_correct_{step}.png",
                                                mu[0].detach().clone(),
                                                update_weight.detach().clone(),
                                                export=True,
                                            )
                                            self.showTmpBoneWeight(
                                                f"{log_dir}/gmm_obs_iter_{iter}_{cov_th}_estimate_{step}.png",
                                                mu[0].detach().clone(),
                                                estimated_weight.detach().clone(),
                                                export=True,
                                            )

                                            self.showTmpBoneWeight(
                                                f"{log_dir}/gmm_obs_iter_{iter}_{cov_th}_correct_mask_{step}.png",
                                                mu[0].detach().clone(),
                                                update_weight.detach().clone(),
                                                mask[0],
                                                export=True,
                                            )

                                            self.showTmpBoneWeight(
                                                f"{log_dir}/gmm_org_iter_{iter}_{cov_th}_can_{step}.png",
                                                mu_can[0].detach().clone(),
                                                W.detach().clone(),
                                                export=True,
                                            )
                                            self.showTmpBoneWeight(
                                                f"{log_dir}/gmm_org_iter_{iter}_{cov_th}_can_mask_{step}.png",
                                                mu_can[0].detach().clone(),
                                                W.detach().clone(),
                                                mask[0],
                                                export=True,
                                            )
                                            self.showTmpBoneWeight(
                                                f"{log_dir}/gmm_can_iter_{iter}_{cov_th}_correct_{step}.png",
                                                new_can[0].detach().clone(),
                                                update_weight,
                                                export=True,
                                            )
                                            self.showTmpBoneWeight(
                                                f"{log_dir}/gmm_can_iter_{iter}_{cov_th}_estimate_{step}.png",
                                                estimate_new_can[0].detach().clone(),
                                                estimated_weight,
                                                export=True,
                                            )
                                            self.showTmpBoneWeight(
                                                f"{log_dir}/gmm_can_iter_{iter}_{cov_th}_correct_mask_{step}.png",
                                                new_can[0].detach().clone(),
                                                update_weight,
                                                mask[0],
                                                export=True,
                                            )
                                            self.showTmpBoneWeight(
                                                f"{log_dir}/gmm_obs_iter_{iter}_{cov_th}_valid_mask_{step}.png",
                                                mu[0].detach().clone(),
                                                W.detach().clone(),
                                                valid_mask[0],
                                                export=True,
                                                joints=joints,
                                            )
                                            self.showTmpBoneWeight(
                                                f"{log_dir}/gmm_can_iter_{iter}_{cov_th}_valid_mask_{step}.png",
                                                mu_can[0].detach().clone(),
                                                W.detach().clone(),
                                                valid_mask[0],
                                                export=True,
                                                joints=joints,
                                            )

                                        self.cnt += 1

            if self.mask is not None and bool(self.mask.any()):
                if self.use_l1:
                    l1_loss_fn = nn.L1Loss()

                    loss_l1_l2 = l1_loss_fn(
                        mu_can[self.mask], self.inv_mu_can[self.mask].detach().clone()
                    )

                else:
                    l2_loss_fn = nn.MSELoss()

                    loss_l1_l2 = l2_loss_fn(
                        mu_can[self.mask], self.inv_mu_can[self.mask].detach().clone()
                    )

                inv_can = self.inv_mu_can[self.mask][None].detach().clone()
                inv_W_pseudo = self.update_weight[self.mask][None].detach().clone()

                inv_W = self.voxel_deformer(inv_can)
                inv_W = torch.clamp(inv_W, min=1e-8)
                inv_W = inv_W / torch.sum(inv_W, dim=2, keepdim=True)
                loss_bone_w += self.lambda_skin * torch.nn.functional.mse_loss(
                    inv_W, inv_W_pseudo
                )

        elif step > self.inverse_end_steps:
            self.mask = None
            self.update_weight = None
            self.inv_mu_can = None
            self.inv_frame_can = None
            self.est_inv_mu_can = None
            self.est_inv_frame_can = None
            self.bone_directions = None

        loss_l1_l2 = self.lambda_l1 * loss_l1_l2
        loss_emd = self.lambda_emd * loss_emd
        loss_haus = self.lambda_haus * loss_haus

        s = s[None].expand(B, -1, -1)  # B,N,1
        o = o[None].expand(B, -1, -1)  # B,N,1
        sph = sph[:, :sph_dim][None].expand(B, -1, -1)  # B,N,C

        mu = mu + trans[:, None, :]

        self.posed_mu = mu[0]
        self.posed_frame = matrix_to_quaternion(frame)[0]
        self.posed_W = W[0]

        loss_xyz = torch.tensor(0).cuda()

        loss_reg = {"nr_xyz": 0, "nr_scale": 0, "nr_rot": 0}

        return (
            mu,
            frame,
            s,
            o,
            sph,
            loss_reg,
            loss_xyz,
            loss_l1_l2,
            loss_haus,
            loss_emd,
            loss_rotation,
            loss_bone_w,
        )

    def sinkhorn_loss(self, x, y, epsilon=0.1, n_iters=50):
        """
        计算两个点云 x 和 y 之间的 Sinkhorn 损失。

        参数：
            x: 张量，形状为 (N, D)，第一个点云。
            y: 张量，形状为 (M, D)，第二个点云。
            epsilon: 正则化参数（熵正则化系数）。
            n_iters: Sinkhorn 算法的迭代次数。

        返回：
            Sinkhorn 损失（标量张量）。
        """
        x = x.float()
        y = y.float()

        N, D = x.shape
        M, _ = y.shape

        # 定义经验测度的权重（均匀分布）
        mu = torch.full((N,), 1.0 / N, dtype=torch.float, device=x.device)
        nu = torch.full((M,), 1.0 / M, dtype=torch.float, device=y.device)

        # 计算成本矩阵 C（欧氏距离的平方）
        C = torch.cdist(x, y, p=2) ** 2  # 形状为 [N, M]

        u = torch.zeros_like(mu)  # 形状为 [N]
        v = torch.zeros_like(nu)  # 形状为 [M]

        # Sinkhorn 迭代
        for _ in range(n_iters):
            u_prev = u  # 保存上一轮的 u 值，用于检查收敛性（可选）

            u = (
                epsilon
                * (
                    torch.log(mu + 1e-8)
                    - torch.logsumexp((-C + v.unsqueeze(0)) / epsilon, dim=1)
                )
                + u
            )

            v = (
                epsilon
                * (
                    torch.log(nu + 1e-8)
                    - torch.logsumexp((-C + u.unsqueeze(1)) / epsilon, dim=0)
                )
                + v
            )

        # 计算输运计划 pi
        pi = torch.exp((u.unsqueeze(1) + v.unsqueeze(0) - C) / epsilon)  # 形状为 [N, M]

        sinkhorn_loss = torch.sum(pi * C)

        return sinkhorn_loss

    def cosine_similarity_loss(self, x, y):

        # 计算余弦相似度，dim参数指定计算相似度的维度
        cosine_sim = F.cosine_similarity(x, y, dim=2)

        loss = 1 - cosine_sim.mean()

        return loss

    def median_absolute_deviation(x, scale=1.4826):
        median = torch.median(x)
        return scale * torch.median(torch.abs(x - median))

    # Estimate covariance after filtering points by axis-wise quantiles.
    def estimate_covariances(
        self, points, weights, joints, weight_threshold=0.3, q=0.9
    ):
        n_joints = joints.shape[1]
        n_points = points.shape[1]
        device = points.device

        weight_mask = weights > weight_threshold

        covariances = []
        global_invalid_mask = torch.ones(1, n_points, dtype=torch.bool, device=device)

        for j in range(n_joints):
            joint_position = joints[0, j]
            bone_direction = self.bone_directions[0, j]

            # 计算垂直于骨骼方向的两个正交向量
            v1 = torch.tensor([1.0, 0.0, 0.0], device=device)
            if torch.abs(torch.dot(v1, bone_direction)) > 0.9:
                v1 = torch.tensor([0.0, 1.0, 0.0], device=device)
            v2 = torch.cross(bone_direction, v1)
            v2 /= torch.norm(v2)
            v1 = torch.cross(v2, bone_direction)

            basis = torch.stack([bone_direction, v1, v2], dim=1)

            relevant_mask = weight_mask[0, :, j]
            relevant_points = points[0, relevant_mask]

            if relevant_points.shape[0] > 0:
                # 将点投影到主轴坐标系
                centered_points = relevant_points - joint_position
                projected_points = torch.matmul(centered_points, basis)

                # 在三个主轴方向上分别计算分位数
                scales = torch.quantile(torch.abs(projected_points), q, dim=0)

                valid_points_mask = torch.all(
                    torch.abs(projected_points) <= scales, dim=1
                )
                valid_points = relevant_points[valid_points_mask]

                if valid_points.shape[0] > 0:
                    # 计算协方差矩阵
                    centered_valid_points = valid_points - joint_position
                    covariance = (
                        torch.matmul(centered_valid_points.t(), centered_valid_points)
                        / centered_valid_points.shape[0]
                    )
                    covariance += torch.eye(3, device=device) * 1e-6
                else:
                    covariance = torch.eye(3, device=device)

                # 更新全局无效掩码
                joint_valid_mask = torch.zeros(
                    n_points, dtype=torch.bool, device=device
                )
                joint_valid_mask[relevant_mask] = valid_points_mask
                global_invalid_mask[0] &= ~joint_valid_mask
            else:
                covariance = torch.eye(3, device=device)

            covariances.append(covariance)

        return covariances, global_invalid_mask

    def find_k_nearest_neighbors(self, points, k=10):
        dist_matrix = torch.cdist(points[0], points[0])
        distances, neighbors = torch.topk(dist_matrix, k, largest=False, dim=1)
        return neighbors, distances

    def update_weights_with_neighbors(
        self,
        points,
        current_weights,
        updated_weights,
        anomaly_mask,
        anomalous_indices,
        neighbors,
        distances,
    ):
        """
        仅更新异常点的权重。

        :param points: Tensor of shape (batch_size, n_points, dims)
        :param current_weights: Tensor of shape (batch_size, n_points, weight_dims)
        :param updated_weights: Tensor of shape (batch_size, n_points, weight_dims)
        :param anomaly_mask: Tensor of shape (batch_size, n_points)
        :param anomalous_indices: Tensor of shape (num_anomalous_points,)
        :param neighbors: Tensor of shape (num_anomalous_points, k)
        :param distances: Tensor of shape (num_anomalous_points, k)
        :return: Tensor of updated_weights
        """
        num_anomalous = anomalous_indices.shape[0]
        for idx in range(num_anomalous):
            point_idx = anomalous_indices[idx].item()
            neighbor_indices = neighbors[idx]

            # 过滤掉异常邻居
            normal_neighbors = neighbor_indices[~anomaly_mask[0, neighbor_indices]]

            if len(normal_neighbors) > 0:
                normal_weights = current_weights[0, normal_neighbors]
                current_updated_weight = updated_weights[0, point_idx].unsqueeze(0)

                # 计算所有正常邻居的 JS 散度
                m = 0.5 * (current_updated_weight + normal_weights)
                kl_w1_m = torch.sum(
                    current_updated_weight
                    * torch.log((current_updated_weight + 1e-10) / (m + 1e-10)),
                    dim=1,
                )
                kl_w2_m = torch.sum(
                    normal_weights * torch.log((normal_weights + 1e-10) / (m + 1e-10)),
                    dim=1,
                )
                js_divergences = (
                    0.5 * kl_w1_m + 0.5 * kl_w2_m
                )  # Shape: (num_normal_neighbors,)

                # 选择 JS 散度最小的1个邻居（可以调整topk的值）
                top_k = 1
                topk_values, topk_indices = torch.topk(
                    js_divergences, top_k, largest=False
                )
                selected_neighbors = normal_neighbors[topk_indices]

                selected_weights = current_weights[0, selected_neighbors]
                updated_weights[0, point_idx] = selected_weights.mean(dim=0)
            else:
                updated_weights[0, point_idx] = current_weights[0, point_idx]
                # 如果没有正常邻居，保持原来的更新权重不变

        return updated_weights

    def compute_probabilities(self, points, covariances, joints):
        """
        计算每个点属于每个关节控制的高斯分布的概率，并根据所属关节仅考虑直接相连的关节重新计算概率。
        :param points: Tensor, 点云位置，形状为 (1, n, 3)
        :param covariances: list of Tensors, 每个关节的协方差矩阵
        :param joints: Tensor, 每个关节的位置，形状为 (1, n_joints, 3)
        :return: Tensor, 重新计算后的概率向量，形状为 (1, n, n_joints)
        """
        n_points = points.shape[1]
        n_joints = joints.shape[1]
        device = points.device

        # 初步概率计算（不归一化）
        probabilities = torch.zeros((n_points, n_joints), device=device)
        for j in range(n_joints):
            joint_position = joints[0, j, :]
            diff = points[0] - joint_position  # (n_points, 3)
            inv_cov = torch.inverse(covariances[j])  # (3, 3)
            mahalanobis_dist = torch.sum((diff @ inv_cov) * diff, dim=1)  # (n_points,)
            # 计算高斯概率密度函数
            det_cov = torch.det(covariances[j])
            denominator = torch.sqrt((2 * torch.pi) ** 3 * det_cov)  # 标量
            exponent = -0.5 * mahalanobis_dist  # (n_points,)
            probabilities[:, j] = torch.exp(exponent) / denominator  # (n_points,)

        assigned_joints = torch.argmax(probabilities, dim=1)  # (n_points,)

        # 为每个点获取关联骨骼（包括自身骨骼）
        connected_joints_per_point = []
        for i in range(n_points):
            assigned_joint = assigned_joints[i].item()
            connected = self.connected_joints.get(assigned_joint, [])
            connected_joints_per_point.append(connected + [assigned_joint])  # 包含自身

        # 重新计算概率，仅考虑关联骨骼
        new_probabilities = torch.zeros_like(probabilities)
        for i in range(n_points):
            connected = connected_joints_per_point[i]
            if not connected:
                connected = [assigned_joints[i].item()]
            probs = probabilities[i, connected]
            if probs.sum() > 0:
                new_probs = probs / probs.sum()

            new_probabilities[i, connected] = new_probs

        new_probabilities = new_probabilities.unsqueeze(0)  # (1, n_points, n_joints)
        return new_probabilities

    def update_weights(
        self,
        points,
        initial_weights,
        joints,
        threshold=0.8,
        cov_th=0.8,
        iteration_threshold=10,
        q=0.9,
    ):
        current_weights = initial_weights.clone()
        prev_anomaly_count = float("inf")
        cumulative_anomaly_mask = torch.zeros_like(
            initial_weights[:, :, 0], dtype=torch.bool
        )

        for i in range(iteration_threshold):
            masked_weights = current_weights.clone()
            masked_weights[
                cumulative_anomaly_mask.unsqueeze(-1).expand_as(masked_weights)
            ] = 0

            covariances, valid_mask = self.estimate_covariances(
                points, masked_weights, joints, cov_th, q=q
            )
            estimated_weights = self.compute_probabilities(points, covariances, joints)

            # 计算 Bhattacharyya 距离
            bc_distance = -torch.log(
                torch.sum(torch.sqrt(initial_weights * estimated_weights), dim=2)
            )
            anomaly_mask = bc_distance > -torch.log(
                torch.tensor(threshold, device=points.device)
            )

            cumulative_anomaly_mask |= anomaly_mask

            current_anomaly_count = anomaly_mask.sum().item()
            if (abs(current_anomaly_count - prev_anomaly_count) < 30) or (
                i == (iteration_threshold - 1)
            ):
                neighbors, distances = self.find_k_nearest_neighbors(points, 50)
                updated_weights = torch.where(
                    anomaly_mask.unsqueeze(2), estimated_weights, initial_weights
                )
                updated_weights = self.update_weights_with_neighbors(
                    points,
                    initial_weights,
                    updated_weights,
                    anomaly_mask,
                    neighbors,
                    distances,
                )
                break

            prev_anomaly_count = current_anomaly_count

        return updated_weights, cumulative_anomaly_mask, valid_mask, estimated_weights

    def find_k_nearest_neighbors_subset(self, points, subset_indices, k=10):
        """
        仅对指定子集的点计算最近邻。

        :param points: Tensor of shape (batch_size, n_points, dims)
        :param subset_indices: Tensor of indices，形状为 (num_subset_points,)
        :param k: int，最近邻的数量
        :return: neighbors, distances
        """
        subset_points = points[0, subset_indices]  # 假设 batch_size = 1
        all_points = points[0]  # 假设 batch_size = 1

        dist_matrix = torch.cdist(
            subset_points, all_points
        )  # Shape: (num_subset_points, n_points)

        # 对每个子集点，找到距离最近的k个邻居（排除自身）
        num_subset_points = subset_indices.shape[0]
        n_points = points.shape[1]

        distances, neighbors = torch.topk(
            dist_matrix, k, largest=False, dim=1
        )  # Shape: (num_subset_points, k)

        return neighbors, distances

    def update_weights_js(
        self,
        points,
        initial_weights,
        joints,
        js_threshold=0.8,
        cov_th=0.8,
        iteration_threshold=10,
        q=0.9,
    ):
        current_weights = initial_weights.clone()
        prev_anomaly_count = float("inf")
        cumulative_anomaly_mask = torch.zeros_like(
            initial_weights[:, :, 0], dtype=torch.bool
        )

        for i in range(iteration_threshold):
            masked_weights = current_weights.clone()
            masked_weights[
                cumulative_anomaly_mask.unsqueeze(-1).expand_as(masked_weights)
            ] = 0

            covariances, valid_mask = self.estimate_covariances(
                points, masked_weights, joints, cov_th, q=q
            )
            estimated_weights = self.compute_probabilities(points, covariances, joints)

            # Compute JS divergence
            m = 0.5 * (initial_weights + estimated_weights)
            js_divergence = 0.5 * (
                torch.sum(
                    initial_weights * torch.log(initial_weights / m + 1e-10), dim=2
                )
                + torch.sum(
                    estimated_weights * torch.log(estimated_weights / m + 1e-10), dim=2
                )
            )

            anomaly_mask = js_divergence > js_threshold

            cumulative_anomaly_mask |= anomaly_mask

            current_anomaly_count = anomaly_mask.sum().item()
            if (abs(current_anomaly_count - prev_anomaly_count) < 30) or (
                i == (iteration_threshold - 1)
            ):
                # 仅对异常点计算最近邻
                anomalous_indices = torch.nonzero(
                    anomaly_mask[0], as_tuple=False
                ).squeeze(1)
                if len(anomalous_indices) > 0:
                    neighbors, distances = self.find_k_nearest_neighbors_subset(
                        points, anomalous_indices, k=50
                    )
                    updated_weights = torch.where(
                        anomaly_mask.unsqueeze(2), estimated_weights, initial_weights
                    )
                    updated_weights = self.update_weights_with_neighbors(
                        points,
                        current_weights,
                        updated_weights,
                        anomaly_mask,
                        anomalous_indices,
                        neighbors,
                        distances,
                    )
                else:
                    updated_weights = (
                        initial_weights  # 如果没有异常点，保持初始权重不变
                    )
                break

            prev_anomaly_count = current_anomaly_count

        return updated_weights, cumulative_anomaly_mask, valid_mask, estimated_weights

    def update_weights_dbscan(
        self, points, initial_weights, joints, eps=0.5, min_samples=5
    ):
        updated_weights = initial_weights.clone()
        batch_size, num_points, _ = points.size()
        num_joints = initial_weights.size(2)
        anomaly_mask = torch.zeros(
            batch_size, num_points, dtype=torch.bool, device=points.device
        )
        valid_mask = torch.ones(
            batch_size, num_points, dtype=torch.bool, device=points.device
        )  # Assuming all points are valid
        estimated_weights = updated_weights.clone()

        for b in range(batch_size):
            points_b = points[b]  # Shape: [num_points, 3]
            initial_weights_b = initial_weights[b]  # Shape: [num_points, num_joints]

            points_np = points_b.cpu().numpy()

            from sklearn.cluster import DBSCAN

            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points_np)
            labels = clustering.labels_  # Shape: [num_points]

            # Anomalies are points labeled as -1
            anomalies = labels == -1
            normal_points = labels != -1

            anomaly_indices = np.where(anomalies)[0]
            normal_indices = np.where(normal_points)[0]

            anomaly_mask[b, anomaly_indices] = True

            if len(anomaly_indices) == 0:
                continue  # No anomalies detected in this batch

            # Find nearest normal points for each anomalous point
            from scipy.spatial import cKDTree

            normal_points_coords = points_np[normal_indices]
            anomalous_points_coords = points_np[anomaly_indices]

            tree = cKDTree(normal_points_coords)
            distances, nearest_indices = tree.query(anomalous_points_coords, k=1)

            nearest_normal_indices = normal_indices[nearest_indices]

            anomaly_indices_torch = torch.from_numpy(anomaly_indices).to(
                initial_weights_b.device
            )
            nearest_normal_indices_torch = torch.from_numpy(nearest_normal_indices).to(
                initial_weights_b.device
            )

            updated_weights[b, anomaly_indices_torch] = initial_weights_b[
                nearest_normal_indices_torch
            ]

        return updated_weights, anomaly_mask, valid_mask, estimated_weights

    def update_weights_kl(
        self,
        points,
        initial_weights,
        joints,
        kl_threshold=0.5,
        cov_th=0.8,
        angle_th=np.pi / 6,
        iteration_threshold=20,
    ):
        current_weights = initial_weights.clone()
        prev_anomaly_count = float("inf")

        for _ in range(1):  # Maximum 1 iteration as per your original code
            covariances, valid_mask = self.estimate_covariances(
                points, current_weights, joints, cov_th, angle_th
            )
            estimated_weights = self.compute_probabilities(points, covariances, joints)

            # Compute KL divergence
            kl_divergence = torch.sum(
                initial_weights
                * torch.log(initial_weights / (estimated_weights + 1e-10) + 1e-10),
                dim=2,
            )

            anomaly_mask = kl_divergence > kl_threshold

            current_anomaly_count = anomaly_mask.sum().item()
            if abs(current_anomaly_count - prev_anomaly_count) < iteration_threshold:
                neighbors, distances = self.find_k_nearest_neighbors(points, 30)
                updated_weights = torch.where(
                    anomaly_mask.unsqueeze(2), estimated_weights, initial_weights
                )
                updated_weights = self.update_weights_with_neighbors(
                    points,
                    initial_weights,
                    updated_weights,
                    anomaly_mask,
                    neighbors,
                    distances,
                )
                break

            neighbors, distances = self.find_k_nearest_neighbors(points, 10)
            updated_weights = torch.where(
                anomaly_mask.unsqueeze(2), estimated_weights, initial_weights
            )
            updated_weights = self.update_weights_with_neighbors(
                points,
                initial_weights,
                updated_weights,
                anomaly_mask,
                neighbors,
                distances,
            )

            current_weights = updated_weights
            prev_anomaly_count = current_anomaly_count

        return updated_weights, anomaly_mask, valid_mask

    def inverse_lbs(self, mu, frame, A, cluster_weight):
        mu = mu.detach().clone()
        frame = frame.detach().clone()
        T = torch.einsum(
            "bnj, bjrc -> bnrc", cluster_weight[..., : self.num_bones], A
        )  # [1,15550,4,4] ,lbs之后的每个gs的rotation矩阵
        R, t = T[:, :, :3, :3], T[:, :, :3, 3]
        R_inv = torch.inverse(R)

        mu_can = torch.einsum("bnij,bnj->bni", R_inv, mu - t)

        frame_can = torch.einsum("bnij,bnjk->bnik", R_inv, frame)

        return mu_can, frame_can

    def initialize_post_first_timestep(self, num_knn=20):
        xyz = self.prev_pts
        dist_sq, nn_ind, _ = knn_points(
            xyz[None], xyz[None], K=num_knn, return_nn=False
        )  # 1,n,K
        neighbor_weight = torch.exp(-200 * dist_sq)[0]
        self.neighbor_indices = nn_ind[0]
        self.neighbor_dist = torch.sqrt(dist_sq)
        self.neighbor_weight = neighbor_weight

    def initialize_per_timestep(self):
        pts = self.prev_pts
        rot = self.prev_rotation
        self.prev_inv_rot = matrix_to_quaternion(
            torch.inverse(quaternion_to_matrix(self.prev_rotation))
        ).detach()
        prev_offset = pts[self.neighbor_indices] - pts[:, None]
        self.prev_offset = prev_offset

    def compute_reg(self, K, INVERSE_REG=False):
        # !can cancel the knn, but the w reg is critical
        if K > 0:
            if INVERSE_REG:
                xyz = self.posed_mu
            else:
                xyz = self._xyz
            dist_sq, nn_ind, _ = knn_points(xyz[None], xyz[None], K=K, return_nn=False)
            nn_ind = nn_ind.squeeze(0)
            # reg the std inside knn

            if INVERSE_REG:
                q = self.posed_frame[nn_ind, :]
            else:
                q = self._rotation[nn_ind, :]  # N,K,4

            s = self.get_s[nn_ind, :]  # N,K,3
            o = self.get_o[nn_ind, :]  # N,K,1
            q_std = q.std(dim=1).mean()
            s_std = s.std(dim=1).mean()
            o_std = o.std(dim=1).mean()

            cd = self._features_dc[nn_ind, :]  # N,K,3
            ch = self._features_rest[nn_ind, :]  # N,K,C
            cd_std = cd.std(dim=1).mean()
            ch_std = ch.std(dim=1).mean()
            if ch.shape[-1] == 0:
                ch_std = torch.zeros_like(ch_std)
            w = self.posed_W[nn_ind, :]  # N,K,3
            w_rest = self._w_correction_rest[nn_ind, :]  # N,K,C
            f = self._features_localcode[nn_ind, :]  # N,K,C
            w_std = w.std(dim=1).mean()
            w_rest_std = w_rest.std(dim=1).mean()
            f_std = f.std(dim=1).mean()
            if w.shape[-1] == 0:
                w_std = torch.zeros_like(cd_std)
            if w_rest.shape[-1] == 0:
                w_rest_std = torch.zeros_like(cd_std)
            if f.shape[-1] == 0:
                f_std = torch.zeros_like(cd_std)
        else:
            dummy = torch.zeros(1).to(self._xyz).squeeze()
            q_std, s_std, o_std = dummy, dummy, dummy
            cd_std, ch_std = dummy, dummy
            w_std, w_rest_std, f_std = dummy, dummy, dummy
            dist_sq = dummy

        w_norm = self._w_correction_dc.norm(dim=-1).mean()  # N
        w_rest_norm = self._w_correction_rest.norm(dim=-1).mean()  # N

        max_s_square = torch.mean((self.get_s.max(dim=1).values) ** 2)

        return (
            q_std,
            s_std,
            o_std,
            cd_std,
            ch_std,
            w_std,
            w_rest_std,
            f_std,
            w_norm,
            w_rest_norm,
            dist_sq.mean(),
            max_s_square,
        )

    def compute_reg_rel(self):

        if self.start_reg:
            rel_rot = quat_mult(self.cur_rot, self.prev_inv_rot)
            rot = build_rotation(rel_rot)
            neighbor_pts = self.cur_pts[self.neighbor_indices]
            curr_offset = neighbor_pts - self.cur_pts[:, None]
            curr_offset_in_prev_coord = (
                rot.transpose(2, 1)[:, None] @ curr_offset[:, :, :, None]
            ).squeeze(-1)
            losses_rigid = weighted_l2_loss_v2(
                curr_offset_in_prev_coord, self.prev_offset, self.neighbor_weight
            )
            losses_rot = weighted_l2_loss_v2(
                rel_rot[self.neighbor_indices], rel_rot[:, None], self.neighbor_weight
            )
            curr_offset_mag = torch.sqrt((curr_offset**2).sum(-1) + 1e-20)
            losses_iso = weighted_l2_loss_v1(
                curr_offset_mag, self.neighbor_dist, self.neighbor_weight
            )
        return losses_rigid, losses_rot, losses_iso

    def get_optimizable_list(
        self,
        lr_p=0.00016,
        lr_q=0.001,
        lr_s=0.005,
        lr_o=0.05,
        lr_sph=0.0025,
        lr_sph_rest=None,
        lr_w=0.001,
        lr_w_rest=0.001,
        lr_f=0.0001,
        lr_mlp=0.005,
    ):
        lr_sph_rest = lr_sph / 20 if lr_sph_rest is None else lr_sph_rest
        l = [
            {"params": [self._xyz], "lr": lr_p, "name": "xyz"},
            {"params": [self._opacity], "lr": lr_o, "name": "opacity"},
            {"params": [self._scaling], "lr": lr_s, "name": "scaling"},
            {"params": [self._rotation], "lr": lr_q, "name": "rotation"},
            {"params": [self._features_dc], "lr": lr_sph, "name": "f_dc"},
            {"params": [self._features_rest], "lr": lr_sph_rest, "name": "f_rest"},
            {"params": [self._w_correction_dc], "lr": lr_w, "name": "w_dc"},
            {"params": [self._w_correction_rest], "lr": lr_w_rest, "name": "w_rest"},
            {"params": [self._features_localcode], "lr": lr_f, "name": "f_localcode"},
            {
                "params": list(self.voxel_deformer.parameters()),
                "lr": lr_mlp,
                "name": "mlp_params",
            },
        ]
        return l

    def record_xyz_grad_radii(self, viewspace_point_tensor, radii, update_filter):
        # Record the gradient norm, invariant across different poses
        assert len(viewspace_point_tensor) == self.N
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=False
        )
        self.xyz_gradient_denom[update_filter] += 1
        self.max_radii2D[update_filter] = torch.max(
            self.max_radii2D[update_filter], radii[update_filter]
        )
        return

    def _densification_postprocess(
        self,
        optimizer,
        new_xyz,
        new_r,
        new_s,
        new_o,
        new_sph_dc,
        new_sph_rest,
        new_w_dc,
        new_w_rest,
        new_localcode,
    ):
        d = {
            "xyz": new_xyz,
            "f_dc": new_sph_dc,
            "f_rest": new_sph_rest,
            "opacity": new_o,
            "scaling": new_s,
            "rotation": new_r,
            "w_dc": new_w_dc,
            "w_rest": new_w_rest,
            "f_localcode": new_localcode,
        }
        d = {k: v for k, v in d.items() if v is not None}

        optimizable_tensors = cat_tensors_to_optimizer(optimizer, d)

        self._xyz = optimizable_tensors["xyz"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._w_correction_dc = optimizable_tensors["w_dc"]
        self._w_correction_rest = optimizable_tensors["w_rest"]
        self._features_localcode = optimizable_tensors["f_localcode"]

        self.xyz_gradient_accum = torch.zeros(self._xyz.shape[0], device="cuda")
        self.xyz_gradient_denom = torch.zeros(self._xyz.shape[0], device="cuda")
        self.max_radii2D = torch.cat(
            [self.max_radii2D, torch.zeros_like(new_xyz[:, 0])], dim=0
        )
        return

    def _densify_and_clone(self, optimizer, grad_norm, grad_threshold, scale_th):
        # padding for enabling both call of clone and split
        padded_grad = torch.zeros((self.N), device="cuda")
        padded_grad[: grad_norm.shape[0]] = grad_norm.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_s, dim=1).values <= scale_th,
        )
        if selected_pts_mask.sum() == 0:
            return 0
        new_xyz = self._xyz[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_w_dc = self._w_correction_dc[selected_pts_mask]
        new_w_rest = self._w_correction_rest[selected_pts_mask]
        new_localcode = self._features_localcode[selected_pts_mask]

        self._densification_postprocess(
            optimizer,
            new_xyz=new_xyz,
            new_r=new_rotation,
            new_s=new_scaling,
            new_o=new_opacities,
            new_sph_dc=new_features_dc,
            new_sph_rest=new_features_rest,
            new_w_dc=new_w_dc,
            new_w_rest=new_w_rest,
            new_localcode=new_localcode,
        )

        return len(new_xyz)

    def _densify_and_split(
        self,
        optimizer,
        grad_norm,
        grad_threshold,
        scale_th,
        N=2,
    ):
        _scaling = self.get_s
        # padding for enabling both call of clone and split
        padded_grad = torch.zeros((self.N), device="cuda")
        padded_grad[: grad_norm.shape[0]] = grad_norm.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(_scaling, dim=1).values > scale_th,
        )
        if selected_pts_mask.sum() == 0:
            return 0

        stds = _scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = quaternion_to_matrix(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self._xyz[
            selected_pts_mask
        ].repeat(N, 1)
        new_scaling = _scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        new_scaling = torch.clamp(new_scaling, max=self.max_scale, min=self.min_scale)
        new_scaling = self.s_inv_act(new_scaling)
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1)
        new_opacities = self._opacity[selected_pts_mask].repeat(N, 1)
        new_w_dc = self._w_correction_dc[selected_pts_mask].repeat(N, 1)
        new_w_rest = self._w_correction_rest[selected_pts_mask].repeat(N, 1)
        new_localcode = self._features_localcode[selected_pts_mask].repeat(N, 1)

        self._densification_postprocess(
            optimizer,
            new_xyz=new_xyz,
            new_r=new_rotation,
            new_s=new_scaling,
            new_o=new_opacities,
            new_sph_dc=new_features_dc,
            new_sph_rest=new_features_rest,
            new_w_dc=new_w_dc,
            new_w_rest=new_w_rest,
            new_localcode=new_localcode,
        )

        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool),
            )
        )
        self._prune_points(optimizer, prune_filter)
        return len(new_xyz)

    def densify(self, optimizer, max_grad, percent_dense, extent, verbose=True):
        grads = self.xyz_gradient_accum / self.xyz_gradient_denom
        grads[grads.isnan()] = 0.0

        if (self._xyz.shape[0]) > 45000:
            return

        n_clone = self._densify_and_clone(
            optimizer, grads, max_grad, percent_dense * extent
        )
        n_split = self._densify_and_split(
            optimizer, grads, max_grad, percent_dense * extent, N=2
        )

        if verbose:
            logging.info(f"Densify: Clone[+] {n_clone}, Split[+] {n_split}")
        return

    def random_grow(self, optimizer, num_factor=0.05, std=0.1, init_opa_value=0.1):
        # * New operation, randomly add largely disturbed points to the geometry
        ind = torch.randperm(self.N)[: int(self.N * num_factor)]
        selected_pts_mask = torch.zeros(self.N, dtype=bool, device="cuda")
        selected_pts_mask[ind] = True

        new_xyz = self._xyz[selected_pts_mask]
        noise = torch.randn_like(new_xyz) * std
        new_xyz = new_xyz + noise
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]

        new_opacities = torch.ones_like(self._opacity[selected_pts_mask])
        new_opacities = new_opacities * self.o_inv_act(init_opa_value)

        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_w_dc = self._w_correction_dc[selected_pts_mask]
        new_w_rest = self._w_correction_rest[selected_pts_mask]
        new_localcode = self._features_localcode[selected_pts_mask]

        self._densification_postprocess(
            optimizer,
            new_xyz=new_xyz,
            new_r=new_rotation,
            new_s=new_scaling,
            new_o=new_opacities,
            new_sph_dc=new_features_dc,
            new_sph_rest=new_features_rest,
            new_w_dc=new_w_dc,
            new_w_rest=new_w_rest,
            new_localcode=new_localcode,
        )
        logging.info(f"Random grow: {len(new_xyz)}")
        return len(new_xyz)

    def statistical_outlier_removal_tensor(self, points, n_neighbors=20, std_ratio=2.0):
        """
        Perform Statistical Outlier Removal (SOR) using PyTorch tensors

        :param points: torch tensor of shape (N, 3) representing 3D points
        :param n_neighbors: number of neighbors to use
        :param std_ratio: standard deviation ratio
        :return: boolean mask, True for inliers, False for outliers
        """
        dist = torch.cdist(points, points)

        k_dist, _ = dist.topk(k=n_neighbors, dim=1, largest=False)

        mean_distances = k_dist.mean(dim=1)

        threshold = mean_distances.mean() + std_ratio * mean_distances.std()

        return mean_distances >= threshold

    def prune_points(
        self, optimizer, min_opacity, max_screen_size, verbose=True, step=-1
    ):
        opacity = self.o_act(self._opacity)
        prune_mask = (opacity < min_opacity).squeeze()

        if (
            (step > self.inverse_start_steps)
            and (step % 2 == 0)
            and (step < (self.inverse_start_steps + self.dbs_steps))
        ):

            dbscan = DBSCAN(eps=self.eps, min_samples=self.samples)
            dbscan_labels = dbscan.fit_predict(self._xyz.detach().clone().cpu())
            core_samples_mask = np.zeros_like(dbscan_labels, dtype=bool)
            core_samples_mask[dbscan.core_sample_indices_] = True
            cluster_mask = torch.from_numpy(dbscan_labels == -1).cuda()
            if cluster_mask.sum() < 300:
                prune_mask = torch.logical_or(prune_mask, cluster_mask)

        if max_screen_size:  # if a point is too large
            big_points_vs = self.max_radii2D > max_screen_size
            prune_mask = torch.logical_or(prune_mask, big_points_vs)
            self.max_radii2D = torch.zeros_like(self.max_radii2D)
        self._prune_points(optimizer, prune_mask)
        if verbose:
            logging.info(f"Prune: {prune_mask.sum()}")

    def _prune_points(self, optimizer, mask):
        valid_points_mask = ~mask
        optimizable_tensors = prune_optimizer(
            optimizer,
            valid_points_mask,
            exclude_names=self.op_update_exclude,
        )

        self._xyz = optimizable_tensors["xyz"]
        if getattr(self, "color_memory", None) is None:
            self._features_dc = optimizable_tensors["f_dc"]
            self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._w_correction_dc = optimizable_tensors["w_dc"]
        self._w_correction_rest = optimizable_tensors["w_rest"]
        self._features_localcode = optimizable_tensors["f_localcode"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.xyz_gradient_denom = self.xyz_gradient_denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        return

    @torch.no_grad()
    def regaussian(self, optimizer, max_scale=0.03):
        # * New operation, manually split the large gaussians with smaller ones to approximate

        # Extract points that satisfy the gradient condition
        _scaling = self.get_s
        selected_pts_mask = torch.max(_scaling, dim=1).values > max_scale

        step = 0
        before_num = self.N
        while selected_pts_mask.any():
            # This can be done more than 3 times, becuase there may be huge gaussians, which should be devided several times
            fg_xyz = self._xyz[selected_pts_mask]
            fg_scale = _scaling[selected_pts_mask]
            fg_frame = quaternion_to_matrix(self._rotation[selected_pts_mask])
            # each column is the direction of axis in global frame
            axis_ind = torch.argmax(fg_scale, dim=1)
            axis_scale = fg_scale.max(dim=1).values
            axis_dir = torch.gather(
                fg_frame, dim=2, index=axis_ind[:, None, None].expand(-1, 3, -1)
            ).squeeze(
                -1
            )  # N,3
            new_x1 = fg_xyz + axis_dir.squeeze() * axis_scale[:, None] / 2.0
            new_x2 = fg_xyz - axis_dir.squeeze() * axis_scale[:, None] / 2.0
            # Repeat will change [1,2,3...] to [1,2,3..., 1,2,3...]
            new_xyz = torch.cat([new_x1, new_x2], dim=0).reshape(-1, 3)
            new_scaling = _scaling[selected_pts_mask]
            new_scaling = torch.scatter(
                new_scaling,
                dim=1,
                index=axis_ind[:, None],
                src=axis_scale[:, None] / 2.0,
            ).repeat(2, 1)
            new_scaling = torch.clamp(
                new_scaling, max=self.max_scale, min=self.min_scale
            )
            new_scaling = self.s_inv_act(new_scaling)
            new_rotation = self._rotation[selected_pts_mask].repeat(2, 1)
            new_features_dc = self._features_dc[selected_pts_mask].repeat(2, 1)
            new_features_rest = self._features_rest[selected_pts_mask].repeat(2, 1)
            new_opacities = self._opacity[selected_pts_mask].repeat(2, 1)
            new_w_dc = self._w_correction_dc[selected_pts_mask].repeat(2, 1)
            new_w_rest = self._w_correction_rest[selected_pts_mask].repeat(2, 1)
            new_localcode = self._features_localcode[selected_pts_mask].repeat(2, 1)

            self._densification_postprocess(
                optimizer,
                new_xyz=new_xyz.float(),
                new_r=new_rotation.float(),
                new_s=new_scaling.float(),
                new_o=new_opacities.float(),
                new_sph_dc=new_features_dc.float(),
                new_sph_rest=new_features_rest.float(),
                new_w_dc=new_w_dc.float(),
                new_w_rest=new_w_rest.float(),
                new_localcode=new_localcode.float(),
            )

            prune_filter = torch.cat(
                (
                    selected_pts_mask,
                    torch.zeros(2 * selected_pts_mask.sum(), device="cuda", dtype=bool),
                )
            )
            self._prune_points(optimizer, prune_filter)

            step += 1
            logging.info(
                f"Regaussian-[{step}], {selected_pts_mask.sum()} ({selected_pts_mask.float().mean()*100}% pts-scale>{max_scale})"
            )

            _scaling = self.get_s
            selected_pts_mask = torch.max(_scaling, dim=1).values > max_scale
        logging.info(f"Re-gaussian: {before_num} -> {self.N}")
        return

    def reset_opacity(self, optimizer, value=0.01, verbose=True):
        opacities_new = self.o_inv_act(
            torch.min(self.o_act(self._opacity), torch.ones_like(self._opacity) * value)
        )
        optimizable_tensors = replace_tensor_to_optimizer(
            optimizer, opacities_new, "opacity"
        )
        if verbose:
            logging.info(f"Reset opacity to {value}")
        self._opacity = optimizable_tensors["opacity"]

    def get_voxel_args(self):
        return {"w_init": self.gt_weights, "v_init": self.gt_bone_points}

    def load_adjust(self, model_ckpt):
        # because N changed, have to re-init the buffers
        self._xyz = nn.Parameter(
            torch.as_tensor(model_ckpt["_xyz"], dtype=torch.float32)
        )

        self._features_dc = nn.Parameter(
            torch.as_tensor(model_ckpt["_features_dc"], dtype=torch.float32)
        )
        self._features_rest = nn.Parameter(
            torch.as_tensor(model_ckpt["_features_rest"], dtype=torch.float32)
        )
        self._opacity = nn.Parameter(
            torch.as_tensor(model_ckpt["_opacity"], dtype=torch.float32)
        )
        self._scaling = nn.Parameter(
            torch.as_tensor(model_ckpt["_scaling"], dtype=torch.float32)
        )

        self._rotation = nn.Parameter(
            torch.as_tensor(model_ckpt["_rotation"], dtype=torch.float32)
        )
        self._w_correction_dc = nn.Parameter(
            torch.as_tensor(model_ckpt["_w_correction_dc"], dtype=torch.float32)
        )
        self._w_correction_rest = nn.Parameter(
            torch.as_tensor(model_ckpt["_w_correction_rest"], dtype=torch.float32)
        )
        self._features_localcode = nn.Parameter(
            torch.as_tensor(model_ckpt["_features_localcode"], dtype=torch.float32)
        )
        self.xyz_gradient_accum = torch.as_tensor(
            model_ckpt["xyz_gradient_accum"], dtype=torch.float32
        )
        self.xyz_gradient_denom = torch.as_tensor(
            model_ckpt["xyz_gradient_denom"], dtype=torch.int64
        )
        self.max_radii2D = torch.as_tensor(
            model_ckpt["max_radii2D"], dtype=torch.float32
        )

        self.load_state_dict(model_ckpt, strict=False)

        scale_init_logit = self.s_inv_act(0.0005)
        s = torch.ones(len(self._opacity), 3) * scale_init_logit
        self._scaling = nn.Parameter(s, requires_grad=True)

        o = nn.Parameter(torch.ones(len(self._opacity), 1) * 0.99)
        self._opacity = nn.Parameter(o, requires_grad=True)

        # this is critical, reinit the funcs
        self._init_act(self.max_scale, self.min_scale)
        return

    def load(self, model_ckpt):
        # because N changed, have to re-init the buffers
        self._xyz = nn.Parameter(
            torch.as_tensor(model_ckpt["_xyz"], dtype=torch.float32)
        )

        self._features_dc = nn.Parameter(
            torch.as_tensor(model_ckpt["_features_dc"], dtype=torch.float32)
        )
        self._features_rest = nn.Parameter(
            torch.as_tensor(model_ckpt["_features_rest"], dtype=torch.float32)
        )
        self._opacity = nn.Parameter(
            torch.as_tensor(model_ckpt["_opacity"], dtype=torch.float32)
        )
        self._scaling = nn.Parameter(
            torch.as_tensor(model_ckpt["_scaling"], dtype=torch.float32)
        )

        self._rotation = nn.Parameter(
            torch.as_tensor(model_ckpt["_rotation"], dtype=torch.float32)
        )
        self._w_correction_dc = nn.Parameter(
            torch.as_tensor(model_ckpt["_w_correction_dc"], dtype=torch.float32)
        )
        self._w_correction_rest = nn.Parameter(
            torch.as_tensor(model_ckpt["_w_correction_rest"], dtype=torch.float32)
        )
        self._features_localcode = nn.Parameter(
            torch.as_tensor(model_ckpt["_features_localcode"], dtype=torch.float32)
        )
        self.xyz_gradient_accum = torch.as_tensor(
            model_ckpt["xyz_gradient_accum"], dtype=torch.float32
        )
        self.xyz_gradient_denom = torch.as_tensor(
            model_ckpt["xyz_gradient_denom"], dtype=torch.int64
        )
        self.max_radii2D = torch.as_tensor(
            model_ckpt["max_radii2D"], dtype=torch.float32
        )

        self.load_state_dict(model_ckpt, strict=False)
        # this is critical, reinit the funcs
        self._init_act(self.max_scale, self.min_scale)
        return
