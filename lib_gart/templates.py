import sys, os, os.path as osp

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from lib_gart.smplx.smplx import SMPLLayer
from lib_gart.smplx.smplx.lbs import (
    blend_shapes,
    vertices2joints,
    batch_rigid_transform,
)
from voxel_deformer import VoxelDeformer, SMPLVoxelDeformer

from pytorch3d.transforms import (
    matrix_to_axis_angle,
    axis_angle_to_matrix,
    quaternion_to_matrix,
    matrix_to_quaternion,
)
from model_utils import get_predefined_human_rest_pose, get_predefined_dog_rest_pose
from transforms3d.euler import euler2mat


def get_template(
    mode, init_beta, cano_pose_type, voxel_deformer_res, template_model_path=None
):
    if mode == "human":
        template = SMPLTemplate(
            smpl_model_path=template_model_path,
            init_beta=init_beta,
            cano_pose_type=cano_pose_type,
            voxel_deformer_res=voxel_deformer_res,
        )
    elif mode == "dog":
        template = SMALTemplate(
            init_beta=init_beta,
            cano_pose_type=cano_pose_type,
            voxel_deformer_res=voxel_deformer_res,
        )
    else:
        raise ValueError(f"Unknown mode {mode}")
    return template


class SMPLTemplate(nn.Module):
    def __init__(self, smpl_model_path, init_beta, cano_pose_type, voxel_deformer_res):
        super().__init__()
        self.dim = 24
        self._template_layer = SMPLLayer(model_path=smpl_model_path)

        if init_beta is None:
            init_beta = np.zeros(10)
        init_beta = torch.as_tensor(init_beta, dtype=torch.float32).cpu()
        self.register_buffer("init_beta", init_beta)
        self.cano_pose_type = cano_pose_type
        self.name = "smpl"

        can_pose = get_predefined_human_rest_pose(cano_pose_type)  # 23,3
        can_pose = axis_angle_to_matrix(
            torch.cat([torch.zeros(1, 3), can_pose], 0)
        )  # 24,3,3
        self.register_buffer("real_canonical_pose", can_pose)

        self.register_buffer("canonical_pose", can_pose)

        init_smpl_output = self._template_layer(
            betas=init_beta[None],
            body_pose=can_pose[None, 1:],
            global_orient=can_pose[None, 0],
            return_full_pose=True,
        )
        J_canonical, A0 = init_smpl_output.J, init_smpl_output.A  # 1,24,3 1,24,4,4
        A0_inv = torch.inverse(A0)
        self.register_buffer("A0_inv", A0_inv[0])
        self.register_buffer("J_canonical", J_canonical)

        posed_joints, A_pre = batch_rigid_transform(
            can_pose,
            J_canonical.expand(1, -1, -1),
            self._template_layer.parents,
        )  # _ , A = lbs处理后相对于canonical的变换rotation矩阵  joints是实际的joints位置，J_canonical不考虑shape时候的joints位置
        self.joints = posed_joints
        v_init = init_smpl_output.vertices  # 1,6890,3
        v_init = v_init[0]
        W_init = self._template_layer.lbs_weights  # 6890,24

        divisor = 8 if voxel_deformer_res == 128 else 4

        self.smpl_voxel_deformer = SMPLVoxelDeformer(
            vtx=v_init[None],
            vtx_features=W_init[None],
            resolution_dhw=[
                voxel_deformer_res // divisor,
                voxel_deformer_res,
                voxel_deformer_res,
            ],
        )

        head_joints = posed_joints[0][self._template_layer.parents[1:]]  # 23,3
        self.register_buffer("head_joints_canonical", head_joints)

        # * Important, record first joint position, because the global orientation is rotating using this joint position as center, so we can compute the action on later As
        j0_t = init_smpl_output.joints[0, 0]  # 根关节的平移向量
        self.register_buffer("j0_t", j0_t)

        return

    def _dot(self, a, b):
        return (a * b).sum(dim=-1, keepdim=True)

    def get_posed_joints(self, theta):
        joints, A = batch_rigid_transform(
            axis_angle_to_matrix(theta),
            self.J_canonical.expand(1, -1, -1),
            self._template_layer.parents,
        )
        return joints

    def closest_distance_to_points(self, points, individual=False):
        """Cartesian distance from points to bones (line segments).

        https://zalo.github.io/blog/closest-point-between-segments/

        :params bones: [n_bones,]
        :params points: If individual is true, the shape should be [..., n_bones, 3].
            else, the shape is [..., 3]
        :returns distances [..., n_bones]
        """
        points = points[..., None, :]
        heads = self.head_joints_canonical.to(points.device)
        tails = self.J_canonical[0][1:].to(points.device)
        t = self._dot(points - heads, tails - heads) / self._dot(
            tails - heads, tails - heads
        )
        points = heads + (tails - heads) * torch.clamp(t, 0, 1)
        dists = torch.linalg.norm(points - points, dim=-1)  # 10000 23
        return dists

    def assign_weights_to_nearest_joint(self, points, primary_weight=1):

        N, J = points.shape[0], self.J_canonical[0].shape[0]

        vertices_expanded = points[:, None, :]  # [N, 1, 3]
        joints_expanded = self.J_canonical[0][None, :, :].cuda()  # [1, J, 3]

        distances_squared = torch.sum(
            (vertices_expanded - joints_expanded) ** 2, dim=2
        )  # [N, J]

        _, nearest_joint_indices = torch.min(distances_squared, dim=1)  # [N]

        weights = torch.full(
            (N, J), (1 - primary_weight) / (J - 1), dtype=torch.float32
        ).to(points.device)

        weights[torch.arange(N), nearest_joint_indices] = primary_weight

        return weights

    def dist_batch(self, points):
        points = points[..., None, :]  # 10000,1,3
        a = self.J_canonical[0][1:].to(points.device)  # 23,3
        b = self.head_joints_canonical.to(points.device)  # 23,3

        dist_root = torch.norm(points - b[0], dim=-1)

        s = b - a
        w = points - a  # 10000,23,3
        w2 = points - b
        ps = (w * s).sum(-1)  # 10000,23
        res = torch.zeros((w.shape[0], w.shape[1]), dtype=points.dtype).to(
            points.device
        )  # 10000,23

        # ps <= 0
        ps_smaller_mask = ps <= 0  # 10000,23
        lower_mask = torch.where(ps_smaller_mask)
        res[lower_mask] += torch.norm(w[lower_mask], dim=-1)

        l2 = (s * s).sum(-1)  # 23
        # ps > 0 and ps >= l2
        ps_mask = ~ps_smaller_mask

        temp_mask_l2 = ps >= l2
        upper_mask = torch.where(ps_mask & temp_mask_l2)
        res[upper_mask] += torch.norm(w2[upper_mask], dim=-1)

        # ps > 0 and ps < l2
        within_mask = torch.where(ps_mask & ~temp_mask_l2)
        res[within_mask] += torch.norm(
            points[within_mask[0]].squeeze(dim=1)
            - (
                a[within_mask[1]]
                + (ps[within_mask] / l2[within_mask[1]]).unsqueeze(-1)
                * s[within_mask[1]]
            ),
            dim=-1,
        )

        res_new = torch.zeros(
            (res.shape[0], res.shape[1] + 1), dtype=res.dtype, device=res.device
        )
        res_new[:, 0] = dist_root.squeeze(-1)
        res_new[:, 1:] = res
        return res_new

    def closest_distance_to_joints(self, points, individual=False):
        """Cartesian distance from points to bones (line segments).

        https://zalo.github.io/blog/closest-point-between-segments/

        :params bones: [n_bones,]
        :params points: If individual is true, the shape should be [..., n_bones, 3].
            else, the shape is [..., 3]
        :returns distances [..., n_bones]
        """
        points = points[..., None, :]
        joints = self.J_canonical[0].to(points.device)
        dists = torch.linalg.norm(points - joints, dim=-1)  # 10000 23
        return dists

    def get_init_vf(self):
        init_smpl_output = self._template_layer(
            betas=self.init_beta[None],
            body_pose=self.canonical_pose[None, 1:],
            global_orient=self.canonical_pose[None, 0],
            return_full_pose=True,
        )
        v_init = init_smpl_output.vertices  # 1,6890,3
        v_init = v_init[0]
        faces = self._template_layer.faces_tensor
        return v_init, faces

    def get_rot_action(self, axis_angle):
        # apply this action to canonical additional bones
        # axis_angle: B,3
        assert axis_angle.ndim == 2 and axis_angle.shape[-1] == 3
        B = len(axis_angle)
        R = axis_angle_to_matrix(axis_angle)  # B,3,3
        I = torch.eye(3).to(R)[None].expand(B, -1, -1)  # B,3,3
        t0 = self.j0_t[None].expand(B, -1)  # B,3
        T = torch.eye(4).to(R)[None].expand(B, -1, -1)  # B,4,4
        T[:, :3, :3] = R
        T[:, :3, 3] = torch.einsum("bij, bj -> bi", I - R, t0)
        return T  # B,4,4

    def forward(self, theta=None, xyz_canonical=None, step=-1):
        if theta is None:
            A = None
        else:
            assert (
                theta.ndim == 3 and theta.shape[-1] == 3
            ), "pose should have shape Bx24x3, in axis-angle format"
            nB = len(theta)  # batch
            joints, A = batch_rigid_transform(
                axis_angle_to_matrix(theta),
                self.J_canonical.expand(nB, -1, -1),
                self._template_layer.parents,
            )  # _ , A = lbs处理后相对于canonical的变换rotation矩阵
            A = torch.einsum("bnij, njk->bnik", A, self.A0_inv)  # B,24,4,4

        if xyz_canonical is None:
            # forward theta only
            W = None
        else:
            W = None

        return W, A, joints


class SMALTemplate(nn.Module):
    def __init__(self, init_beta, cano_pose_type, voxel_deformer_res):
        super().__init__()
        self.dim = 35
        from lib_gart.smal.smal_tpg import SMAL

        self._template_layer = SMAL()

        if init_beta is None:
            init_beta = np.zeros(30)
        init_beta = torch.as_tensor(init_beta, dtype=torch.float32).cpu()
        self.register_buffer("init_beta", init_beta)

        self.cano_pose_type = cano_pose_type
        self.name = "smal"

        can_pose = get_predefined_dog_rest_pose(cano_pose_type)
        can_pose = torch.cat([torch.zeros(3), can_pose], 0)
        self.register_buffer("canonical_pose", can_pose)
        self.register_buffer("real_canonical_pose", can_pose)

        v_init, (A0, W0, T0, joint_xyz) = self._template_layer(
            beta=self.init_beta[None],
            betas_limbs=can_pose[None, -7:],
            pose=axis_angle_to_matrix(can_pose[None, :-7].reshape(1, 35, 3)),
        )

        A0_inv = torch.inverse(A0)
        self.register_buffer("A0_inv", A0_inv[0])

        self.register_buffer("J_canonical", joint_xyz[0])

        posed_joints, A_pre = batch_rigid_transform(
            axis_angle_to_matrix(can_pose[:-7].reshape(35, 3)),
            self.J_canonical.expand(1, -1, -1),
            self._template_layer.parents,
        )
        self.joints = posed_joints

        head_joints = posed_joints[0][self._template_layer.parents[1:]]
        self.register_buffer("head_joints_canonical", head_joints)

        self.voxel_deformer = VoxelDeformer(
            vtx=v_init,
            vtx_features=W0,
            resolution_dhw=[
                voxel_deformer_res,
                voxel_deformer_res // 2,
                voxel_deformer_res,
            ],
            short_dim_dhw=1,
            long_dim_dhw=2,
        )

        j0_t = joint_xyz[0, 0]
        self.register_buffer("j0_t", j0_t)
        return

    def _dot(self, a, b):
        return (a * b).sum(dim=-1, keepdim=True)

    def get_posed_joints(self, theta):
        joints, A = batch_rigid_transform(
            axis_angle_to_matrix(theta),
            self.J_canonical.expand(len(theta), -1, -1),
            self._template_layer.parents,
        )
        return joints

    def closest_distance_to_points(self, points, individual=False):
        points = points[..., None, :]
        heads = self.head_joints_canonical.to(points.device)
        tails = self.J_canonical[0][1:].to(points.device)
        t = self._dot(points - heads, tails - heads) / self._dot(
            tails - heads, tails - heads
        )
        points_proj = heads + (tails - heads) * torch.clamp(t, 0, 1)
        dists = torch.linalg.norm(points_proj - points, dim=-1)
        return dists

    def assign_weights_to_nearest_joint(self, points, primary_weight=1):
        N, J = points.shape[0], self.J_canonical.shape[0]

        vertices_expanded = points[:, None, :]
        joints_expanded = self.J_canonical[None, :, :].to(points.device)

        distances_squared = torch.sum((vertices_expanded - joints_expanded) ** 2, dim=2)
        _, nearest_joint_indices = torch.min(distances_squared, dim=1)

        weights = torch.full(
            (N, J), (1 - primary_weight) / (J - 1), dtype=torch.float32
        ).to(points.device)
        weights[torch.arange(N), nearest_joint_indices] = primary_weight

        return weights

    def dist_batch(self, points):
        points = points[..., None, :]
        a = self.J_canonical[1:].to(points.device)
        b = self.head_joints_canonical.to(points.device)

        dist_root = torch.norm(points - b[0], dim=-1)

        s = b - a
        w = points - a
        w2 = points - b
        ps = (w * s).sum(-1)
        res = torch.zeros((w.shape[0], w.shape[1]), dtype=points.dtype).to(
            points.device
        )

        ps_smaller_mask = ps <= 0
        lower_mask = torch.where(ps_smaller_mask)
        res[lower_mask] += torch.norm(w[lower_mask], dim=-1)

        l2 = (s * s).sum(-1)
        ps_mask = ~ps_smaller_mask

        temp_mask_l2 = ps >= l2
        upper_mask = torch.where(ps_mask & temp_mask_l2)
        res[upper_mask] += torch.norm(w2[upper_mask], dim=-1)

        within_mask = torch.where(ps_mask & ~temp_mask_l2)
        res[within_mask] += torch.norm(
            points[within_mask[0]].squeeze(dim=1)
            - (
                a[within_mask[1]]
                + (ps[within_mask] / l2[within_mask[1]]).unsqueeze(-1)
                * s[within_mask[1]]
            ),
            dim=-1,
        )

        res_new = torch.zeros(
            (res.shape[0], res.shape[1] + 1), dtype=res.dtype, device=res.device
        )
        res_new[:, 0] = dist_root.squeeze(-1)
        res_new[:, 1:] = res
        return res_new

    def closest_distance_to_joints(self, points, individual=False):
        points = points[..., None, :]
        joints = self.J_canonical.to(points.device)
        dists = torch.linalg.norm(points - joints, dim=-1)
        return dists

    @torch.no_grad()
    def get_init_vf(self):
        v_init, (A0, W0, T0, _) = self._template_layer(
            beta=self.init_beta[None],
            betas_limbs=self.canonical_pose[None, -7:],
            pose=axis_angle_to_matrix(self.canonical_pose[None, :-7].reshape(1, 35, 3)),
        )
        faces = self._template_layer.faces
        return v_init[0], torch.as_tensor(faces, dtype=torch.long)

    def get_rot_action(self, axis_angle):
        assert axis_angle.ndim == 2 and axis_angle.shape[-1] == 3
        B = len(axis_angle)
        R = axis_angle_to_matrix(axis_angle)
        I = torch.eye(3).to(R)[None].expand(B, -1, -1)
        t0 = self.j0_t[None].expand(B, -1)
        T = torch.eye(4).to(R)[None].expand(B, -1, -1)
        T[:, :3, :3] = R
        T[:, :3, 3] = torch.einsum("bij, bj -> bi", I - R, t0)
        return T

    def forward(self, theta=None, step=-1):
        if theta is None:
            A = None
            joints = None
        else:
            B = len(theta)
            pose = theta[:, :-7]
            betas_limbs = theta[:, -7:]
            _v, (A, W0, T0, joint_xyz) = self._template_layer(
                beta=self.init_beta[None].expand(B, -1),
                betas_limbs=betas_limbs,
                pose=axis_angle_to_matrix(pose.reshape(B, 35, 3)),
            )
            A = torch.einsum("bnij, njk->bnik", A, self.A0_inv)
            joints = joint_xyz

        W = None
        return W, A, joints
