import torch
import torch.nn as nn
import pytorch3d.transforms as tf

from network_utils import (
    HierarchicalPoseEncoder,
    VanillaCondMLP,
    HannwCondMLP,
    HashGrid,
)

from omegaconf import DictConfig, OmegaConf
from pytorch3d.transforms import (
    matrix_to_axis_angle,
    axis_angle_to_matrix,
    quaternion_to_matrix,
    matrix_to_quaternion,
)


def quaternion_multiply(r, s):
    r0, r1, r2, r3 = r.unbind(-1)
    s0, s1, s2, s3 = s.unbind(-1)
    t0 = r0 * s0 - r1 * s1 - r2 * s2 - r3 * s3
    t1 = r0 * s1 + r1 * s0 - r2 * s3 + r3 * s2
    t2 = r0 * s2 + r1 * s3 + r2 * s0 - r3 * s1
    t3 = r0 * s3 - r1 * s2 + r2 * s1 + r3 * s0
    t = torch.stack([t0, t1, t2, t3], dim=-1)
    return t


class NonRigidDeform(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def forward(self, gaussians, iteration, camera, compute_loss=True):
        raise NotImplementedError


class Identity(NonRigidDeform):
    def __init__(self, cfg, metadata):
        super().__init__(cfg)

    def forward(self, gaussians, iteration, camera, compute_loss=True):
        return gaussians, {}


class MLP(NonRigidDeform):
    def __init__(self, cfg, metadata):
        super().__init__(cfg)
        self.pose_encoder = HierarchicalPoseEncoder(**cfg.pose_encoder)
        d_cond = self.pose_encoder.n_output_dims

        # add latent code
        self.latent_dim = cfg.get("latent_dim", 0)
        if self.latent_dim > 0:
            d_cond += self.latent_dim
            self.frame_dict = metadata["frame_dict"]
            self.latent = nn.Embedding(len(self.frame_dict), self.latent_dim)

        d_in = 3
        d_out = 3 + 3 + 4
        self.feature_dim = cfg.get("feature_dim", 0)
        d_out += self.feature_dim

        # output dimension: position + scale + rotation
        self.mlp = VanillaCondMLP(d_in, d_cond, d_out, cfg.mlp)
        self.aabb = metadata["aabb"]

        self.delay = cfg.get("delay", 0)

    def forward(self, gaussians, iteration, camera, compute_loss=True):
        if iteration < self.delay:
            deformed_gaussians = gaussians.clone()
            if self.feature_dim > 0:
                setattr(
                    deformed_gaussians,
                    "non_rigid_feature",
                    torch.zeros(gaussians.get_xyz.shape[0], self.feature_dim).cuda(),
                )
            return deformed_gaussians, {}

        rots = camera.rots
        Jtrs = camera.Jtrs
        pose_feat = self.pose_encoder(rots, Jtrs)

        if self.latent_dim > 0:
            frame_idx = camera.frame_id
            if frame_idx not in self.frame_dict:
                latent_idx = len(self.frame_dict) - 1
            else:
                latent_idx = self.frame_dict[frame_idx]
            latent_idx = torch.Tensor([latent_idx]).long().to(pose_feat.device)
            latent_code = self.latent(latent_idx)
            latent_code = latent_code.expand(pose_feat.shape[0], -1)
            pose_feat = torch.cat([pose_feat, latent_code], dim=1)

        xyz = gaussians.get_xyz
        xyz_norm = self.aabb.normalize(xyz, sym=True)
        deformed_gaussians = gaussians.clone()
        deltas = self.mlp(xyz_norm, cond=pose_feat)

        delta_xyz = deltas[:, :3]
        delta_scale = deltas[:, 3:6]
        delta_rot = deltas[:, 6:10]

        deformed_gaussians._xyz = gaussians._xyz + delta_xyz

        scale_offset = self.cfg.get("scale_offset", "logit")
        if scale_offset == "logit":
            deformed_gaussians._scaling = gaussians._scaling + delta_scale
        elif scale_offset == "exp":
            deformed_gaussians._scaling = torch.log(
                torch.clamp_min(gaussians.get_scaling + delta_scale, 1e-6)
            )
        elif scale_offset == "zero":
            delta_scale = torch.zeros_like(delta_scale)
            deformed_gaussians._scaling = gaussians._scaling
        else:
            raise ValueError

        rot_offset = self.cfg.get("rot_offset", "add")
        if rot_offset == "add":
            deformed_gaussians._rotation = gaussians._rotation + delta_rot
        elif rot_offset == "mult":
            q1 = delta_rot
            q1[0] = 1.0  # [1,0,0,0] represents identity rotation
            delta_rot = delta_rot[1:]
            q2 = gaussians._rotation
            deformed_gaussians._rotation = tf.quaternion_multiply(q1, q2)
        else:
            raise ValueError

        if self.feature_dim > 0:
            setattr(deformed_gaussians, "non_rigid_feature", deltas[:, 10:])

        if compute_loss:
            # regularization
            loss_xyz = torch.norm(delta_xyz, p=2, dim=1).mean()
            loss_scale = torch.norm(delta_scale, p=1, dim=1).mean()
            loss_rot = torch.norm(delta_rot, p=1, dim=1).mean()
            loss_reg = {"nr_xyz": loss_xyz, "nr_scale": loss_scale, "nr_rot": loss_rot}
        else:
            loss_reg = {}
        return deformed_gaussians, loss_reg


class HannwMLP(NonRigidDeform):
    def __init__(self, cfg, metadata):
        super().__init__(cfg)
        self.pose_encoder = HierarchicalPoseEncoder(**cfg.pose_encoder)
        # output dimension: position + scale + rotation
        self.mlp = HannwCondMLP(
            3, self.pose_encoder.n_output_dims, 3 + 3 + 4, cfg.mlp, dim_coord=3
        )
        self.aabb = metadata["aabb"]

    def forward(self, gaussians, iteration, camera, compute_loss=True):
        rots = camera.rots
        Jtrs = camera.Jtrs
        pose_feat = self.pose_encoder(rots, Jtrs)

        xyz = gaussians.get_xyz
        xyz_norm = self.aabb.normalize(xyz, sym=True)
        deformed_gaussians = gaussians.clone()
        deltas = self.mlp(xyz_norm, iteration, cond=pose_feat)

        if iteration < self.cfg.mlp.embedder.kick_in_iter:
            deltas = deltas * torch.zeros_like(deltas)

        delta_xyz = deltas[:, :3]
        delta_scale = deltas[:, 3:6]
        delta_rot = deltas[:, -4:]

        deformed_gaussians._xyz = gaussians._xyz + delta_xyz

        scale_offset = self.cfg.get("scale_offset", "logit")
        if scale_offset == "logit":
            deformed_gaussians._scaling = gaussians._scaling + delta_scale
        elif scale_offset == "exp":
            deformed_gaussians._scaling = torch.log(
                torch.clamp_min(gaussians.get_scaling + delta_scale, 1e-6)
            )
        elif scale_offset == "zero":
            delta_scale = torch.zeros_like(delta_scale)
            deformed_gaussians._scaling = gaussians._scaling
        else:
            raise ValueError

        rot_offset = self.cfg.get("rot_offset", "add")
        if rot_offset == "add":
            deformed_gaussians._rotation = gaussians._rotation + delta_rot
        elif rot_offset == "mult":
            q1 = delta_rot
            q1[0] = 1.0  # [1,0,0,0] represents identity rotation
            delta_rot = delta_rot[1:]
            q2 = gaussians._rotation
            deformed_gaussians._rotation = quaternion_multiply(q1, q2)
        else:
            raise ValueError

        if compute_loss:
            # regularization
            loss_xyz = torch.norm(delta_xyz, p=2, dim=1).mean()
            loss_scale = torch.norm(delta_scale, p=1, dim=1).mean()
            loss_rot = torch.norm(delta_rot, p=1, dim=1).mean()
            loss_reg = {"nr_xyz": loss_xyz, "nr_scale": loss_scale, "nr_rot": loss_rot}
        else:
            loss_reg = {}
        return deformed_gaussians, loss_reg


class HashGridwithMLP(nn.Module):
    def __init__(self, aabb, delay):
        super().__init__()
        pose_encoder = {
            "dim_per_joint": 6,
            "num_joints": 24,
            "out_dim": -1,
            "rel_joints": False,
        }
        self.pose_encoder = HierarchicalPoseEncoder(**pose_encoder)
        d_cond = self.pose_encoder.n_output_dims

        d_out = 3 + 3 + 4
        self.feature_dim = 16
        d_out += self.feature_dim
        cfg_hashgrid = {
            "n_levels": 16,
            "base_resolution": 16,
            "log2_hashmap_size": 16,
            "max_resolution": 2048,
            "n_features_per_level": 2,
            "per_level_scale": 1.447269237440378,
        }
        cfg_mlp = {
            "n_neurons": 128,
            "n_hidden_layers": 3,
            "skip_in": [],
            "cond_in": [0],
            "multires": 0,
            "last_layer_init": False,
        }
        cfg_hashgrid = DictConfig(cfg_hashgrid)

        cfg_mlp = DictConfig(cfg_mlp)

        self.aabb = aabb
        self.hashgrid = HashGrid(cfg_hashgrid)
        self.mlp = VanillaCondMLP(self.hashgrid.n_output_dims, d_cond, d_out, cfg_mlp)

        self.delay = delay

    def canon_forward(self, xyz, rots, Jtrs):
        pose_feat = self.pose_encoder(rots.reshape([1, -1, 9]), Jtrs)

        xyz_norm = self.aabb.normalize(xyz, sym=True)
        feature = self.hashgrid(xyz_norm)
        deltas = self.mlp(feature, cond=pose_feat)

        delta_xyz = deltas[:, :3]
        delta_scale = deltas[:, 3:6]
        delta_rot = deltas[:, 6:10]  # 4 quaternion

        loss_xyz = torch.norm(delta_xyz, p=1, dim=1).mean()
        loss_scale = torch.norm(delta_scale, p=1, dim=1).mean()
        loss_rot = torch.norm(delta_rot, p=1, dim=1).mean()
        loss_reg = {"nr_xyz": loss_xyz, "nr_scale": loss_scale, "nr_rot": loss_rot}
        return loss_reg

    def forward(self, xyz, scales, rotation, step, rots, Jtrs):
        if step < self.delay:
            return xyz, scales, rotation, {}

        pose_feat = self.pose_encoder(rots.reshape([1, -1, 9]), Jtrs)
        xyz_norm = self.aabb.normalize(xyz, sym=True)
        feature = self.hashgrid(xyz_norm)
        deltas = self.mlp(feature, cond=pose_feat)

        delta_xyz = deltas[:, :3]
        delta_scale = deltas[:, 3:6]
        delta_rot = deltas[:, 6:10]  # 4 quaternion

        new_xyz = xyz + delta_xyz

        new_scaling = scales + delta_scale
        new_rotation = quaternion_to_matrix(
            torch.nn.functional.normalize(matrix_to_quaternion(rotation) + (delta_rot))
        )

        loss_xyz = torch.norm(delta_xyz, p=1, dim=1).mean()
        loss_scale = torch.norm(delta_scale, p=1, dim=1).mean()
        loss_rot = torch.norm(delta_rot, p=1, dim=1).mean()
        loss_reg = {"nr_xyz": loss_xyz, "nr_scale": loss_scale, "nr_rot": loss_rot}
        return new_xyz, new_scaling, new_rotation, loss_reg


def get_non_rigid_deform(cfg, metadata):
    name = cfg.name
    model_dict = {
        "identity": Identity,
        "mlp": MLP,
        "hannw_mlp": HannwMLP,
        "hashgrid": HashGridwithMLP,
    }
    return model_dict[name](cfg, metadata)
