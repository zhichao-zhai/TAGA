import torch
from torch import nn
import torch.nn.functional as F
from pytorch3d.ops import knn_points
from network import ImplicitNetwork


class VoxelMemory(nn.Module):
    # the voxel memory should be inited from voxel deformer

    def __init__(self, v_shape, offset, scale, ratio_dim, ratio, init_std=1e-4):
        super().__init__()
        voxel = torch.randn(*v_shape) * init_std
        self.voxel = nn.Parameter(voxel)
        self.register_buffer("offset", offset)
        self.register_buffer("scale", scale)
        self.register_buffer("ratio", ratio)
        self.ratio_dim = ratio_dim
        return

    def forward(self, xc, mode="bilinear"):
        shape = xc.shape  # ..., 3
        N = 1
        xc = xc.reshape(1, -1, 3)
        w = F.grid_sample(
            self.voxel.expand(N, -1, -1, -1, -1),
            self.normalize(xc)[:, :, None, None],
            align_corners=True,
            mode=mode,
            padding_mode="border",
        )
        w = w.squeeze(-1).squeeze(-1).permute(0, 2, 1)
        w = w.reshape(*shape[:-1], -1)
        return w

    def normalize(self, x):
        x_normalized = x.clone()
        x_normalized -= self.offset
        x_normalized /= self.scale
        x_normalized[..., self.ratio_dim] *= self.ratio
        return x_normalized

    def denormalize(self, x):
        x_denormalized = x.clone()
        x_denormalized[..., self.ratio_dim] /= self.ratio
        x_denormalized *= self.scale
        x_denormalized += self.offset
        return x_denormalized


class VoxelDeformer(nn.Module):
    def __init__(
        self,
        vtx,
        vtx_features,
        resolution_dhw=[8, 32, 32],
        short_dim_dhw=0,  # 0 is d, corresponding to z
        long_dim_dhw=1,
        voxel=True,
        iterations=10,
        coe=0.7,
    ) -> None:
        super().__init__()
        # vtx 1,N,3, vtx_features: 1,N,J
        # d-z h-y w-x; human is facing z; dog is facing x, z is upward, should compress on y

        self.resolution_dhw = resolution_dhw
        device = vtx.device
        d, h, w = self.resolution_dhw

        self.register_buffer(
            "ratio",
            torch.Tensor(
                [self.resolution_dhw[long_dim_dhw] / self.resolution_dhw[short_dim_dhw]]
            ).squeeze(),
        )
        self.ratio_dim = -1 - short_dim_dhw
        x_range = (
            torch.linspace(-1, 1, steps=w, device=device)
            .view(1, 1, 1, w)
            .expand(1, d, h, w)
        )
        y_range = (
            torch.linspace(-1, 1, steps=h, device=device)
            .view(1, 1, h, 1)
            .expand(1, d, h, w)
        )
        z_range = (
            torch.linspace(-1, 1, steps=d, device=device)
            .view(1, d, 1, 1)
            .expand(1, d, h, w)
        )
        grid = (
            torch.cat((x_range, y_range, z_range), dim=0)
            .reshape(1, 3, -1)
            .permute(0, 2, 1)
        )

        gt_bbox = torch.cat([vtx.min(dim=1).values, vtx.max(dim=1).values], dim=0).to(
            device
        )
        offset = (gt_bbox[0] + gt_bbox[1])[None, None, :] * 0.5
        self.register_buffer(
            "global_scale", torch.Tensor([1.2]).squeeze()
        )  # from Fast-SNARF
        scale = (gt_bbox[1] - gt_bbox[0]).max() / 2 * self.global_scale

        corner = torch.ones_like(offset[0]) * scale
        corner[0, self.ratio_dim] /= self.ratio
        min_vert = (offset - corner).reshape(1, 3)
        max_vert = (offset + corner).reshape(1, 3)
        self.bbox = torch.cat([min_vert, max_vert], dim=0)

        self.register_buffer("scale", scale)
        self.register_buffer("offset", offset)

        grid_denorm = self.denormalize(
            grid
        )  # grid_denorm is in the same scale as the canonical body

        weights = (
            self._query_weights_simple_init(
                grid_denorm,
                smpl_verts=vtx.detach().clone(),
                smpl_weights=vtx_features.detach().clone(),
            )
            .detach()
            .clone()
        )

        self.register_buffer("lbs_voxel_base", weights.detach())
        self.register_buffer("grid_denorm", grid_denorm)

        self.num_bones = vtx_features.shape[-1]
        self.voxel = voxel

        self.lbs_network = ImplicitNetwork(
            d_in=3,
            d_out=self.num_bones,
            depth=4,
            width=128,
            multires=0,
            dim_cond_embed=0,
            weight_norm=True,
            geometric_init=False,
            bias=1,
        ).to(device)

        self.cached_lbs_voxel_base = None
        self.iterations = iterations
        self.coe = coe

    def enable_use_voxelion(self):
        voxel_w_correction = torch.zeros_like(self.lbs_voxel_base)
        self.voxel_w_correction = nn.Parameter(voxel_w_correction)

    def enable_additional_correction(self, additional_channels, std=1e-4):
        additional_correction = (
            torch.ones(
                self.lbs_voxel_base.shape[0],
                additional_channels,
                *self.lbs_voxel_base.shape[2:]
            )
            * std
        )
        self.additional_correction = nn.Parameter(additional_correction)

    @property
    def get_voxel_weight(self):
        w = self.lbs_voxel_base
        if hasattr(self, "voxel_w_correction"):
            w = w + self.voxel_w_correction
        if hasattr(self, "additional_correction"):
            w = torch.cat([w, self.additional_correction], dim=1)
        return w

    def get_tv(self, name="dc"):
        if name == "dc":
            if not hasattr(self, "voxel_w_correction"):
                return torch.zeros(1).squeeze().to(self.lbs_voxel_base.device)
            d = self.voxel_w_correction
        elif name == "rest":
            if not hasattr(self, "additional_correction"):
                return torch.zeros(1).squeeze().to(self.lbs_voxel_base.device)
            d = self.additional_correction
        tv_x = torch.abs(d[:, :, 1:, :, :] - d[:, :, :-1, :, :]).mean()
        tv_y = torch.abs(d[:, :, :, 1:, :] - d[:, :, :, :-1, :]).mean()
        tv_z = torch.abs(d[:, :, :, :, 1:] - d[:, :, :, :, :-1]).mean()
        return (tv_x + tv_y + tv_z) / 3.0

    def get_mag(self, name="dc"):
        if name == "dc":
            if not hasattr(self, "voxel_w_correction"):
                return torch.zeros(1).squeeze().to(self.lbs_voxel_base.device)
            d = self.voxel_w_correction
        elif name == "rest":
            if not hasattr(self, "additional_correction"):
                return torch.zeros(1).squeeze().to(self.lbs_voxel_base.device)
            d = self.additional_correction
        return torch.norm(d, dim=1).mean()

    def forward(self, xc, mode="bilinear"):
        shape = xc.shape  # ..., 3
        N = 1  # Assuming batch size 1; adjust as needed
        xc = xc.reshape(1, -1, 3)

        if self.training:
            self.mlp_to_voxel()
            weights = self.get_voxel_weight
            # 清除缓存，因为在训练模式下缓存的权重可能已过时
            self.cached_lbs_voxel_base = None
        else:
            if self.cached_lbs_voxel_base is None:
                with torch.no_grad():
                    self.mlp_to_voxel()
                    self.cached_lbs_voxel_base = self.get_voxel_weight.detach()
            weights = self.cached_lbs_voxel_base

        w = F.grid_sample(
            weights.expand(N, -1, -1, -1, -1),
            self.normalize(xc)[:, :, None, None],
            align_corners=True,
            mode=mode,
            padding_mode="border",
        )
        w = w.squeeze(-1).squeeze(-1).permute(0, 2, 1)
        w = w.reshape(*shape[:-1], -1)
        # * the w may have more channels
        return w  # 1,6890,self.num_bones

    def normalize(self, x):
        x_normalized = x.clone()
        x_normalized -= self.offset
        x_normalized /= self.scale
        x_normalized[..., self.ratio_dim] *= self.ratio
        return x_normalized

    def denormalize(self, x):
        x_denormalized = x.clone()
        x_denormalized[..., self.ratio_dim] /= self.ratio
        x_denormalized *= self.scale
        x_denormalized += self.offset
        return x_denormalized

    def _query_weights_smpl(self, x, smpl_verts, smpl_weights):
        # adapted from https://github.com/jby1993/SelfReconCode/blob/main/model/Deformer.py
        dist, idx, _ = knn_points(x, smpl_verts.detach(), K=1)
        dist = dist.sqrt().clamp_(0.0001, 1.0)
        weights = smpl_weights[0, idx]

        ws = 1.0 / dist
        ws = ws / ws.sum(-1, keepdim=True)
        weights = (ws[..., None] * weights).sum(-2)

        c = smpl_weights.shape[-1]
        d, h, w = self.resolution_dhw
        weights = weights.permute(0, 2, 1).reshape(1, c, d, h, w)
        for _ in range(30):
            mean = (
                weights[:, :, 2:, 1:-1, 1:-1]
                + weights[:, :, :-2, 1:-1, 1:-1]
                + weights[:, :, 1:-1, 2:, 1:-1]
                + weights[:, :, 1:-1, :-2, 1:-1]
                + weights[:, :, 1:-1, 1:-1, 2:]
                + weights[:, :, 1:-1, 1:-1, :-2]
            ) / 6.0
            weights[:, :, 1:-1, 1:-1, 1:-1] = (
                weights[:, :, 1:-1, 1:-1, 1:-1] - mean
            ) * 0.7 + mean
            sums = weights.sum(1, keepdim=True)
            weights = weights / sums
        return weights.detach()

    def _query_weights_simple_init(self, x, smpl_verts, smpl_weights):
        # x.shape: (1, n, 3), smpl_verts.shape: (1, self.num_bones, 3), smpl_weights.shape: (1, self.num_bones, C)
        n, C = x.shape[1], smpl_weights.shape[-1]
        x_weights = torch.zeros((1, n, C), device=x.device)
        K = 500
        for i in range(smpl_verts.shape[1]):
            dist, idx, _ = knn_points(
                smpl_verts[:, i : i + 1, :], x, K=K
            )  # Assume knn_points can return distances and indices for the 30 nearest neighbors
            x_weights[0, idx, :] = smpl_weights[:, i, :]

        c = smpl_weights.shape[-1]
        d, h, w = self.resolution_dhw
        weights = x_weights.permute(0, 2, 1).reshape(1, c, d, h, w)

        return weights.detach()

    def mlp_to_voxel(self):
        d, h, w = self.resolution_dhw

        lbs_voxel_final = self.lbs_network(self.grid_denorm, {}, None)
        lbs_voxel_final = 20 * lbs_voxel_final

        lbs_voxel_final = F.softmax(lbs_voxel_final, dim=-1)

        weights = lbs_voxel_final.permute(0, 2, 1).reshape(1, self.num_bones, d, h, w)

        if self.voxel:
            for _ in range(self.iterations):
                mean = (
                    weights[:, :, 2:, 1:-1, 1:-1]
                    + weights[:, :, :-2, 1:-1, 1:-1]
                    + weights[:, :, 1:-1, 2:, 1:-1]
                    + weights[:, :, 1:-1, :-2, 1:-1]
                    + weights[:, :, 1:-1, 1:-1, 2:]
                    + weights[:, :, 1:-1, 1:-1, :-2]
                ) / 6.0
                updated_weights = (
                    weights[:, :, 1:-1, 1:-1, 1:-1] - mean
                ) * self.coe + mean

                temp_weights = weights.clone()
                temp_weights[:, :, 1:-1, 1:-1, 1:-1] = updated_weights
                weights = temp_weights

                sums = (
                    weights.sum(1, keepdim=True) + 1e-8
                )  # Adding epsilon for numerical stability
                weights = weights / sums

        if self.training:
            # 仅在训练模式下更新 lbs_voxel_base
            self.lbs_voxel_base = weights
        return weights  # 返回计算后的权重以便在推理模式下缓存


class SMPLVoxelDeformer(VoxelDeformer):
    def __init__(
        self,
        vtx,
        vtx_features,
        resolution_dhw=[8, 32, 32],
        short_dim_dhw=0,  # 0 is d, corresponding to z
        long_dim_dhw=1,
    ) -> None:
        super().__init__(vtx, vtx_features, resolution_dhw, short_dim_dhw, long_dim_dhw)

        return

    def forward(self, xc, mode="bilinear"):
        shape = xc.shape  # ..., 3
        N = 1  # 1,self.num_bones,16,64,64
        xc = xc.reshape(1, -1, 3)
        w = F.grid_sample(
            self.get_voxel_weight.expand(N, -1, -1, -1, -1),
            self.normalize(xc)[:, :, None, None],
            align_corners=True,
            mode=mode,
            padding_mode="border",
        )
        w = w.squeeze(-1).squeeze(-1).permute(0, 2, 1)
        w = w.reshape(*shape[:-1], -1)
        # * the w may have more channels
        return w  # 1,6890,self.num_bones


class Deformer(nn.Module):
    def __init__(
        self,
        vtx,
        vtx_features,
        resolution_dhw=[8, 32, 32],
        short_dim_dhw=0,  # 0 is d, corresponding to z
        long_dim_dhw=1,
    ) -> None:
        super().__init__()
        # vtx 1,N,3, vtx_features: 1,N,J
        # d-z h-y w-x; human is facing z; dog is facing x, z is upward, should compress on y

        self.resolution_dhw = resolution_dhw
        device = vtx.device
        d, h, w = self.resolution_dhw

        self.register_buffer(
            "ratio",
            torch.Tensor(
                [self.resolution_dhw[long_dim_dhw] / self.resolution_dhw[short_dim_dhw]]
            ).squeeze(),
        )
        self.ratio_dim = -1 - short_dim_dhw
        x_range = (
            (torch.linspace(-1, 1, steps=w, device=device))
            .view(1, 1, 1, w)
            .expand(1, d, h, w)
        )
        y_range = (
            (torch.linspace(-1, 1, steps=h, device=device))
            .view(1, 1, h, 1)
            .expand(1, d, h, w)
        )
        z_range = (
            (torch.linspace(-1, 1, steps=d, device=device))
            .view(1, d, 1, 1)
            .expand(1, d, h, w)
        )
        grid = (
            torch.cat((x_range, y_range, z_range), dim=0)
            .reshape(1, 3, -1)
            .permute(0, 2, 1)
        )

        gt_bbox = torch.cat([vtx.min(dim=1).values, vtx.max(dim=1).values], dim=0).to(
            device
        )
        offset = (gt_bbox[0] + gt_bbox[1])[None, None, :] * 0.5
        self.register_buffer(
            "global_scale", torch.Tensor([1.2]).squeeze()
        )  # from Fast-SNARF
        scale = (gt_bbox[1] - gt_bbox[0]).max() / 2 * self.global_scale

        corner = torch.ones_like(offset[0]) * scale
        corner[0, self.ratio_dim] /= self.ratio
        min_vert = (offset - corner).reshape(1, 3)
        max_vert = (offset + corner).reshape(1, 3)
        self.bbox = torch.cat([min_vert, max_vert], dim=0)

        self.register_buffer("scale", scale)
        self.register_buffer("offset", offset)

        grid_denorm = self.denormalize(
            grid
        )  # grid_denorm is in the same scale as the canonical body

        self.register_buffer("grid_denorm", grid_denorm)

        self.num_bones = vtx_features.shape[-1]

        self.lbs_network = ImplicitNetwork(
            d_in=3,
            d_out=self.num_bones,
            depth=4,
            width=128,
            multires=0,
            dim_cond_embed=0,
            weight_norm=True,
            geometric_init=False,
            bias=1,
        ).to("cuda")
        return

    def forward(self, xc, mode="bilinear"):
        shape = xc.shape  # ..., 3
        N = 1  # 1,self.num_bones,16,64,64
        xc = xc.reshape(1, -1, 3)
        # 1,N,1,1,3
        w = self.lbs_network(self.normalize(xc), {}, None)
        w = 20 * w

        w = F.softmax(w, dim=-1)
        w = w.squeeze(-1).squeeze(-1).permute(0, 2, 1)
        w = w.reshape(*shape[:-1], -1)
        # * the w may have more channels
        return w  # 1,6890,self.num_bones

    def normalize(self, x):
        x_normalized = x.clone()
        x_normalized -= self.offset
        x_normalized /= self.scale
        x_normalized[..., self.ratio_dim] *= self.ratio
        return x_normalized

    def denormalize(self, x):
        x_denormalized = x.clone()
        x_denormalized[..., self.ratio_dim] /= self.ratio
        x_denormalized *= self.scale
        x_denormalized += self.offset
        return x_denormalized
