"""TAGA CVPR video fitting and evaluation. See train.py for the public CLI."""

import logging
import os
import os.path as osp
import shutil
import time
import imageio
from matplotlib import pyplot as plt
import numpy as np
from omegaconf import OmegaConf
from pytorch3d.ops import knn_points
from pytorch3d.transforms import matrix_to_axis_angle
import torch
from torch import nn
from tqdm import tqdm
from transforms3d.euler import euler2mat
import yaml
from lib_data.get_data import prepare_real_seq
from lib_gart.model import AdditionalBones, GaussianTemplateModel
from lib_gart.network import ImplicitNetwork
from lib_gart.optim_utils import (
    get_expon_lr_func,
    get_expon_lr_func_interval,
    update_learning_rate,
)
from lib_gart.templates import get_template
from lib_guidance.camera_sampling import fov2K
from lib_render.gauspl_renderer import render_cam_pcl
from taga_runtime import tracking as wandb
from test_utils import test
from utils.misc import create_log, get_bbox, seed_everything
from utils.ssim import ssim
from utils.viz import viz_render
from viz_utils import viz_spinning


def psnr_metric(img_pred, img_gt):
    mse = np.mean((img_pred - img_gt) ** 2)
    psnr = -10 * np.log(mse) / np.log(10)
    return psnr


class TGFitter:
    """Shared CVPR video trainer for human and dog articulated Gaussian models."""

    def __init__(
        self,
        log_dir,
        profile_fn,
        mode,
        template_model_path="data/smpl_model/SMPL_NEUTRAL.pkl",
        device=torch.device("cuda:0"),
        cfg=None,
        **kwargs,
    ) -> None:
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.profile_fn = profile_fn
        if cfg == None:
            try:
                shutil.copy(
                    profile_fn, osp.join(self.log_dir, osp.basename(profile_fn))
                )
            except:
                pass
            cfg = OmegaConf.load(profile_fn)
        else:
            with open(osp.join(self.log_dir, osp.basename(profile_fn)), "w") as file:
                yaml.dump(cfg, file, default_flow_style=None)
        for k, v in cfg.items():
            setattr(self, k, v)
        self.OPACIT_PRUNE_TH = self.OPACIT_RESET_VALUE
        self.PRUNE_INTERVAL = self.DENSIFY_INTERVAL
        self.mode = mode
        assert self.mode in ["human", "dog"], "Only support human and dog for now"
        self.template_model_path = template_model_path
        self.device = device
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.FAST_TRAINING = getattr(self, "FAST_TRAINING", False)
        self.LAMBDA_SSIM = getattr(self, "LAMBDA_SSIM", 0.0)
        self.LAMBDA_LPIPS = getattr(self, "LAMBDA_LPIPS", 0.0)
        self.LAMBDA_INVERSE_ROTATION = getattr(self, "LAMBDA_INVERSE_ROTATION", 0.1)
        if self.LAMBDA_LPIPS > 0:
            from utils.lpips import LPIPS

            self.lpips = LPIPS(net="vgg").to(self.device)
            for param in self.lpips.parameters():
                param.requires_grad = False
        if isinstance(self.RESET_OPACITY_STEPS, int):
            self.RESET_OPACITY_STEPS = [
                i
                for i in range(1, self.TOTAL_steps)
                if i % self.RESET_OPACITY_STEPS == 0
            ]
        if isinstance(self.REGAUSSIAN_STEPS, int):
            self.REGAUSSIAN_STEPS = [
                i for i in range(1, self.TOTAL_steps) if i % self.REGAUSSIAN_STEPS == 0
            ]
        if self.mode == "human":
            viz_base_R_opencv = np.asarray(euler2mat(np.pi, 0, 0, "sxyz"))
        else:
            viz_base_R_opencv = np.asarray(euler2mat(np.pi / 2.0, 0, np.pi, "rxyz"))
        viz_base_R_opencv = torch.from_numpy(viz_base_R_opencv).float()
        self.viz_base_R = viz_base_R_opencv.to(self.device)
        if self.mode == "human":
            self.reg_base_R_global = (
                matrix_to_axis_angle(
                    torch.as_tensor(euler2mat(np.pi / 2.0, 0, np.pi / 2.0, "sxyz"))[
                        None
                    ]
                )[0]
                .float()
                .to(self.device)
            )
        else:
            pass
        self.writer = create_log(
            self.log_dir, name=osp.basename(self.profile_fn).split(".")[0], debug=False
        )
        return

    def prepare_real_seq(
        self,
        seq_name,
        dataset_mode,
        split,
        ins_avt_wild_start_end_skip=None,
        image_zoom_ratio=0.5,
        data_stay_gpu_flag=True,
    ):
        (provider, dataset) = prepare_real_seq(
            seq_name=seq_name,
            dataset_mode=dataset_mode,
            split=split,
            ins_avt_wild_start_end_skip=ins_avt_wild_start_end_skip,
            image_zoom_ratio=getattr(self, "IMAGE_ZOOM_RATIO", image_zoom_ratio),
            balance=getattr(self, "VIEW_BALANCE_FLAG", False),
        )
        provider.to(self.device)
        if getattr(self, "DATA_STAY_GPU_FLAG", data_stay_gpu_flag):
            provider.move_images_to_device(self.device)
        provider.viz_selection_prob(
            osp.join(self.log_dir, f"split_{split}_view_prob.png")
        )
        return (provider, dataset)

    def load_saved_model(self, ckpt_path=None):
        if ckpt_path is None:
            ckpt_path = osp.join(self.log_dir, "model.pth")
        ret = self._get_model_optimizer(betas=None)
        model = ret[0]
        model.load(torch.load(ckpt_path))
        model.to(self.device)
        model.eval()
        logging.info("After loading:")
        model.summary()
        return model

    def _get_model_optimizer(self, betas, add_bones_total_t=0):
        seed_everything(self.SEED)
        template = get_template(
            mode=self.mode,
            template_model_path=self.template_model_path,
            init_beta=betas,
            cano_pose_type=getattr(self, "CANO_POSE_TYPE", "t_pose"),
            voxel_deformer_res=getattr(self, "VOXEL_DEFORMER_RES", 64),
        )
        add_bones = AdditionalBones(
            num_bones=getattr(self, "W_REST_DIM", 0),
            mode=getattr(self, "W_REST_MODE", "pose-mlp"),
            total_t=add_bones_total_t,
            mlp_hidden_dims=getattr(
                self, "ADD_BONES_MLP_HIDDEN_DIMS", [256, 256, 256, 256]
            ),
            pose_dim=23 * 3 if self.mode == "human" else 34 * 3 + 7,
        )
        model = GaussianTemplateModel(
            template=template,
            add_bones=add_bones,
            log_dir=self.log_dir,
            w_correction_flag=getattr(self, "W_CORRECTION_FLAG", False),
            f_localcode_dim=getattr(self, "F_LOCALCODE_DIM", 0),
            max_sph_order=getattr(self, "MAX_SPH_ORDER", 0),
            w_memory_type=getattr(self, "W_MEMORY_TYPE", "point"),
            max_scale=getattr(self, "MAX_SCALE", 0.1),
            min_scale=getattr(self, "MIN_SCALE", 0.0),
            init_mode=getattr(self, "INIT_MODE", "random"),
            opacity_init_value=getattr(self, "OPACITY_INIT_VALUE", 0.9),
            onmesh_init_subdivide_num=getattr(self, "ONMESH_INIT_SUBDIVIDE_NUM", 0),
            onmesh_init_scale_factor=getattr(self, "ONMESH_INIT_SCALE_FACTOR", 1.0),
            onmesh_init_thickness_factor=getattr(
                self, "ONMESH_INIT_THICKNESS_FACTOR", 0.5
            ),
            nearmesh_init_num=getattr(self, "NEARMESH_INIT_NUM", 1000),
            nearmesh_init_std=getattr(self, "NEARMESH_INIT_STD", 0.5),
            scale_init_value=getattr(self, "SCALE_INIT_VALUE", 1.0),
            eps=getattr(self, "DBSCAN_EPS", 0.06),
            samples=getattr(self, "MIN_SAMPLES", 80),
            use_l1=getattr(self, "USE_L1", False),
            use_rotation=getattr(self, "USE_ROTATION", False),
            use_haus=getattr(self, "USE_HAUS", True),
            use_emd=getattr(self, "USE_EMD", False),
            use_dynamic=getattr(self, "USE_DYNAMIC", False),
            dynamic_start_steps=getattr(self, "DYNAMIC_START_STEPS", 20000),
            dynamic_end_steps=getattr(self, "DYNAMIC_END_STEPS", 20000),
            inverse_start_steps=getattr(self, "INVERSE_START_STEPS", 20000),
            inverse_end_steps=getattr(self, "INVERSE_END_STEPS", 20000),
            emd_gap=getattr(self, "EMD_GAP", 10),
            lambda_l1=getattr(self, "LAMBDA_L1", 0),
            lambda_emd=getattr(self, "LAMBDA_EMD", 1),
            lambda_haus=getattr(self, "LAMBDA_HAUS", 1),
            densify_interval=getattr(self, "DENSIFY_INTERVAL", 201),
            prune_interval=getattr(self, "PRUNE_INTERVAL", 201),
            lambda_assign=getattr(self, "LAMBDA_ASSIGN", 0.3),
            inverse_K=getattr(self, "INVERSE_K", 40),
            use_voxel=getattr(self, "USE_VOXEL", True),
            K=getattr(self, "BONE_SAMPLE_K", 5),
            inverse_interval=getattr(self, "INVERSE_INTERVAL", 50),
            threshold=getattr(self, "THRESHOLD", 0.5),
            delay=getattr(self, "DEFORM_DELAY", 1000),
            lambda_skin=getattr(self, "LAMBDA_SKIN", 1),
            resolution=getattr(self, "VOXEL_DEFORMER_RES", 64),
            noise_step=getattr(self, "NOISE_STEP", 2000),
            regaussian_steps=getattr(self, "REGAUSSIAN_STEPS", []),
            pose_noise=getattr(self, "POSE_NOISE", 0.1),
            voxel=getattr(self, "VOXEL", True),
            voxel_iterations=getattr(self, "VOXEL_ITER", 30),
            coe=getattr(self, "VOXEL_COE", 0.7),
            dbs_steps=getattr(self, "DBS_STEPS", 2000),
        ).to(self.device)
        model.debug_visualizations = getattr(self, "DEBUG_VIS", False)
        logging.info(f"Init with {model.N} Gaussians")
        LR_SPH_REST = getattr(self, "LR_SPH_REST", self.LR_SPH / 20.0)
        optimizer = torch.optim.Adam(
            model.get_optimizable_list(
                lr_p=self.LR_P,
                lr_o=self.LR_O,
                lr_s=self.LR_S,
                lr_q=self.LR_Q,
                lr_sph=self.LR_SPH,
                lr_sph_rest=LR_SPH_REST,
                lr_w=self.LR_W,
                lr_w_rest=self.LR_W_REST,
                lr_f=getattr(self, "LR_F_LOCAL", 0.0),
                lr_mlp=self.LR_MLP,
            )
        )
        xyz_scheduler_func = get_expon_lr_func(
            lr_init=self.LR_P,
            lr_final=self.LR_P_FINAL,
            lr_delay_mult=0.01,
            max_steps=self.TOTAL_steps,
        )
        w_dc_scheduler_func = get_expon_lr_func_interval(
            init_step=getattr(self, "W_START_STEP", 0),
            final_step=getattr(self, "W_END_STEP", self.TOTAL_steps),
            lr_init=self.LR_W,
            lr_final=getattr(self, "LR_W_FINAL", self.LR_W),
            lr_delay_mult=0.01,
        )
        LR_MLP_FINAL = self.LR_MLP / self.MLP_GAP
        mlp_scheduler_func = get_expon_lr_func_interval(
            init_step=self.MLP_START_STEPS,
            final_step=self.MLP_END_STEPS,
            lr_init=self.LR_MLP,
            lr_final=LR_MLP_FINAL,
            lr_delay_mult=0.01,
        )
        w_rest_scheduler_func = get_expon_lr_func_interval(
            init_step=getattr(self, "W_START_STEP", 0),
            final_step=getattr(self, "W_END_STEP", self.TOTAL_steps),
            lr_init=self.LR_W_REST,
            lr_final=getattr(self, "LR_W_REST_FINAL", self.LR_W_REST),
            lr_delay_mult=0.01,
        )
        sph_scheduler_func = get_expon_lr_func(
            lr_init=self.LR_SPH,
            lr_final=getattr(self, "LR_SPH_FINAL", self.LR_SPH),
            lr_delay_mult=0.01,
            max_steps=self.TOTAL_steps,
        )
        sph_rest_scheduler_func = get_expon_lr_func(
            lr_init=LR_SPH_REST,
            lr_final=getattr(self, "LR_SPH_REST_FINAL", LR_SPH_REST),
            lr_delay_mult=0.01,
            max_steps=self.TOTAL_steps,
        )
        return (
            model,
            optimizer,
            xyz_scheduler_func,
            w_dc_scheduler_func,
            w_rest_scheduler_func,
            sph_scheduler_func,
            sph_rest_scheduler_func,
            mlp_scheduler_func,
        )

    def _get_pose_optimizer(self, data_provider, add_bones):
        if data_provider is None and add_bones.num_bones == 0:
            return (torch.optim.Adam(params=[torch.zeros(1, requires_grad=True)]), {})
        (scheduler_dict, pose_optim_l) = ({}, [])
        if data_provider is not None:
            start_step = getattr(self, "POSE_OPTIMIZE_START_STEP", 0)
            end_step = getattr(self, "POSE_OPTIMIZE_END_STEP", self.TOTAL_steps)
            pose_optim_l.extend(
                [
                    {
                        "params": data_provider.pose_base_list,
                        "lr": self.POSE_R_BASE_LR,
                        "name": "pose_base",
                    },
                    {
                        "params": data_provider.pose_rest_list,
                        "lr": self.POSE_R_REST_LR,
                        "name": "pose_rest",
                    },
                    {
                        "params": data_provider.global_trans_list,
                        "lr": self.POSE_T_LR,
                        "name": "pose_trans",
                    },
                ]
            )
            scheduler_dict["pose_base"] = get_expon_lr_func_interval(
                init_step=start_step,
                final_step=end_step,
                lr_init=self.POSE_R_BASE_LR,
                lr_final=getattr(self, "POSE_R_BASE_LR_FINAL", self.POSE_R_BASE_LR),
                lr_delay_mult=0.01,
            )
            scheduler_dict["pose_rest"] = get_expon_lr_func_interval(
                init_step=start_step,
                final_step=end_step,
                lr_init=self.POSE_R_REST_LR,
                lr_final=getattr(self, "POSE_R_REST_LR_FINAL", self.POSE_R_REST_LR),
                lr_delay_mult=0.01,
            )
            scheduler_dict["pose_trans"] = get_expon_lr_func_interval(
                init_step=start_step,
                final_step=end_step,
                lr_init=self.POSE_T_LR,
                lr_final=getattr(self, "POSE_T_LR_FINAL", self.POSE_T_LR),
                lr_delay_mult=0.01,
            )
        if add_bones.num_bones > 0:
            pose_optim_l.append(
                {
                    "params": add_bones.parameters(),
                    "lr": getattr(self, "LR_W_REST_BONES", 0.0),
                    "name": "add_bones",
                }
            )
        pose_optim_mode = getattr(self, "POSE_OPTIM_MODE", "adam")
        if pose_optim_mode == "adam":
            optimizer_smpl = torch.optim.Adam(pose_optim_l)
        elif pose_optim_mode == "sgd":
            optimizer_smpl = torch.optim.SGD(pose_optim_l)
        else:
            raise NotImplementedError(f"Unknown pose optimizer mode {pose_optim_mode}")
        return (optimizer_smpl, scheduler_dict)

    def is_pose_step(self, step):
        flag = (
            step >= self.POSE_START_STEP
            and step <= self.POSE_END_STEP
            and (step % self.POSE_STEP_INTERVAL == 0)
        )
        return flag

    def _fit_step(
        self,
        model,
        data_pack,
        act_sph_ord,
        random_bg=True,
        scale_multiplier=1.0,
        opacity_multiplier=1.0,
        opa_th=-1,
        use_box_crop_pad=-1,
        default_bg=[1.0, 1.0, 1.0],
        add_bones_As=None,
        step=-1,
    ):
        """Render observed frames and assemble image, mask, bone and cycle losses."""
        (gt_rgb, gt_mask, K, pose_base, pose_rest, global_trans, time_index) = data_pack
        gt_rgb = gt_rgb.clone()
        pose = torch.cat([pose_base, pose_rest], dim=1)
        (H, W) = gt_rgb.shape[1:3]
        additional_dict = {}
        if add_bones_As is not None:
            additional_dict["As"] = add_bones_As
        (
            mu,
            fr,
            sc,
            op,
            sph,
            loss_convert,
            loss_3dgs_reg,
            loss_l1,
            loss_haus,
            loss_emd,
            loss_rotation,
            loss_bone_w,
        ) = model(
            pose,
            global_trans,
            additional_dict=additional_dict,
            active_sph_order=act_sph_ord,
            step=step,
        )
        sc = sc * scale_multiplier
        op = op * opacity_multiplier
        loss_recon = 0.0
        (loss_lpips, loss_ssim) = (
            torch.zeros(1).to(gt_rgb.device).squeeze(),
            torch.zeros(1).to(gt_rgb.device).squeeze(),
        )
        loss_mask = 0.0
        (render_pkg_list, rgb_target_list, mask_target_list) = ([], [], [])
        for i in range(len(gt_rgb)):
            if random_bg:
                bg = np.random.uniform(0.0, 1.0, size=3)
            else:
                bg = np.array(default_bg)
            bg_tensor = torch.from_numpy(bg).float().to(gt_rgb.device)
            gt_rgb[i][gt_mask[i] == 0] = bg_tensor[None, None, :]
            render_pkg = render_cam_pcl(
                mu[i], fr[i], sc[i], op[i], sph[i], H, W, K[i], False, act_sph_ord, bg
            )
            if opa_th > 0.0:
                bg_mask = render_pkg["alpha"][0] < opa_th
                render_pkg["rgb"][:, bg_mask] = (
                    render_pkg["rgb"][:, bg_mask] * 0.0 + default_bg[0]
                )
            if self.LAMBDA_LPIPS > 0:
                _loss_lpips = self.lpips(
                    render_pkg["rgb"][None], gt_rgb.permute(0, 3, 1, 2)
                )
                loss_lpips = loss_lpips + _loss_lpips
            if use_box_crop_pad > 0:
                (_yl, _yr, _xl, _xr) = get_bbox(gt_mask[i], use_box_crop_pad)
                for key in ["rgb", "alpha", "dep"]:
                    render_pkg[key] = render_pkg[key][:, _yl:_yr, _xl:_xr]
                rgb_target = gt_rgb[i][_yl:_yr, _xl:_xr]
                mask_target = gt_mask[i][_yl:_yr, _xl:_xr]
            else:
                rgb_target = gt_rgb[i]
                mask_target = gt_mask[i]
            _loss_recon = abs(render_pkg["rgb"].permute(1, 2, 0) - rgb_target).mean()
            _loss_mask = abs(render_pkg["alpha"].squeeze(0) - mask_target).mean()
            if self.LAMBDA_SSIM > 0:
                _loss_ssim = 1.0 - ssim(
                    render_pkg["rgb"][None], rgb_target.permute(2, 0, 1)[None]
                )
                loss_ssim = loss_ssim + _loss_ssim
            loss_recon = loss_recon + _loss_recon
            loss_mask = loss_mask + _loss_mask
            render_pkg_list.append(render_pkg)
            rgb_target_list.append(rgb_target)
            mask_target_list.append(mask_target)
        start_entropy_regularization_from = 2000
        end_entropy_regularization_at = 4000
        entropy_regularization_factor = 0.1
        loss_bce = torch.tensor(0).cuda()
        if (
            self.USE_BCE
            and step > start_entropy_regularization_from
            and (step < end_entropy_regularization_at)
        ):
            visibility_filter = render_pkg["radii"] > 0
            if visibility_filter is not None:
                vis_opacities = op[0][visibility_filter]
            else:
                vis_opacities = op[0]
            loss_bce = (
                entropy_regularization_factor
                * (
                    -vis_opacities * torch.log(vis_opacities + 1e-10)
                    - (1 - vis_opacities) * torch.log(1 - vis_opacities + 1e-10)
                ).mean()
            )
        loss_recon = loss_recon / len(gt_rgb)
        loss_mask = loss_mask / len(gt_rgb)
        loss_lpips = loss_lpips / len(gt_rgb)
        loss_ssim = loss_ssim / len(gt_rgb)
        loss_bce = loss_bce / len(gt_rgb)
        loss_bone_w = loss_bone_w / len(gt_rgb)
        loss_convert_reg = torch.tensor(0, dtype=torch.float32).cuda()
        self.LAMBDA_DEFORM = getattr(self, "LAMBDA_DEFORM", 0)
        if step < self.DEFORM_INIT_STEPS:
            for name, value in loss_convert.items():
                loss_convert_reg += self.LAMBDA_DEFORM * value
        else:
            for name, value in loss_convert.items():
                loss_convert_reg += self.LAMBDA_DEFORM_STAGE2 * value
        loss_convert_reg = loss_convert_reg / len(gt_rgb)
        loss_3dgs_reg = loss_3dgs_reg / len(gt_rgb)
        loss_l1 = loss_l1 / len(gt_rgb)
        loss_emd = loss_emd / len(gt_rgb)
        loss_rotation = loss_rotation / len(gt_rgb)
        loss_haus = loss_haus / len(gt_rgb)
        wandb.log(
            {
                "loss_recon": loss_recon.item(),
                "loss_mask": loss_mask.item(),
                "loss_lpips": loss_lpips.item(),
                "loss_ssim": loss_ssim.item(),
                "loss_bce": loss_bce.item(),
                "loss_convert": loss_convert_reg.item(),
                "loss_3dgs_reg": loss_3dgs_reg.item(),
                "loss_l1": loss_l1.item(),
                "loss_emd": loss_emd.item(),
                "loss_bone_weight": loss_bone_w.item(),
                "loss_haus": loss_haus.item(),
                "loss_rotation": loss_rotation.item(),
            }
        )
        loss = (
            loss_recon
            + self.LAMBDA_SSIM * loss_ssim
            + self.LAMBDA_LPIPS * loss_lpips
            + loss_bce
            + loss_convert_reg
            + loss_3dgs_reg
            + loss_l1
            + loss_haus
            + loss_emd
            + self.LAMBDA_INVERSE_ROTATION * loss_rotation
        )
        if step % 500 == 0:
            wandb.log(
                {
                    "train_loss_recon": _loss_recon.item(),
                    "train_loss_lpips": loss_lpips.item(),
                }
            )
        return (
            loss,
            loss_mask,
            loss_bone_w,
            render_pkg_list,
            rgb_target_list,
            mask_target_list,
            (mu, fr, sc, op, sph),
            {
                "loss_l1_recon": loss_recon,
                "loss_lpips": loss_lpips,
                "loss_ssim": loss_ssim,
            },
        )

    def compute_reg3D(self, model: GaussianTemplateModel):
        """Regularize Gaussian geometry and learned deformation parameters."""
        if model.start_reg:
            (losses_rigid, losses_rot, losses_iso) = model.compute_reg_rel()
            reg_loss = 1 * losses_rigid + 1 * losses_rot + 1 * losses_iso
            details = {
                "losses_rigid": losses_rigid.detach(),
                "losses_rot": losses_rot.detach(),
                "losses_iso": losses_iso.detach(),
            }
            return (reg_loss, details)
        else:
            K = getattr(self, "CANONICAL_SPACE_REG_K", 10)
            (
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
                dist_sq,
                max_s_sq,
            ) = model.compute_reg(K, getattr(self, "INVERSE_REG", False))
            lambda_std_q = getattr(self, "LAMBDA_STD_Q", 0.0)
            lambda_std_s = getattr(self, "LAMBDA_STD_S", 0.0)
            lambda_std_o = getattr(self, "LAMBDA_STD_O", 0.0)
            lambda_std_cd = getattr(self, "LAMBDA_STD_CD", 0.0)
            lambda_std_ch = getattr(self, "LAMBDA_STD_CH", lambda_std_cd)
            lambda_std_w = getattr(self, "LAMBDA_STD_W", 0.3)
            lambda_std_w_rest = getattr(self, "LAMBDA_STD_W_REST", lambda_std_w)
            lambda_std_f = getattr(self, "LAMBDA_STD_F", lambda_std_w)
            lambda_w_norm = getattr(self, "LAMBDA_W_NORM", 0.1)
            lambda_w_rest_norm = getattr(self, "LAMBDA_W_REST_NORM", lambda_w_norm)
            lambda_knn_dist = getattr(self, "LAMBDA_KNN_DIST", 0.0)
            lambda_small_scale = getattr(self, "LAMBDA_SMALL_SCALE", 0.0)
            reg_loss = (
                lambda_std_q * q_std
                + lambda_std_s * s_std
                + lambda_std_o * o_std
                + lambda_std_cd * cd_std
                + lambda_std_ch * ch_std
                + lambda_knn_dist * dist_sq
                + lambda_std_w * w_std
                + lambda_std_w_rest * w_rest_std
                + lambda_std_f * f_std
                + lambda_w_norm * w_norm
                + lambda_w_rest_norm * w_rest_norm
                + lambda_small_scale * max_s_sq
            )
            details = {
                "q_std": q_std.detach(),
                "s_std": s_std.detach(),
                "o_std": o_std.detach(),
                "cd_std": cd_std.detach(),
                "ch_std": ch_std.detach(),
                "w_std": w_std.detach(),
                "w_rest_std": w_rest_std.detach(),
                "f_std": f_std.detach(),
                "knn_dist_sq": dist_sq.detach(),
                "w_norm": w_norm.detach(),
                "w_rest_norm": w_rest_norm.detach(),
                "max_s_sq": max_s_sq.detach(),
            }
            return (reg_loss, details)

    def add_scalar(self, *args, **kwargs):
        if self.FAST_TRAINING:
            return
        if getattr(self, "NO_TB", False):
            return
        self.writer.add_scalar(*args, **kwargs)
        return

    def run(
        self, real_data_provider=None, pre_data_provider=None, init_data_provider=None
    ):
        """Fit canonical Gaussians and voxel skinning to one posed video sequence."""
        if real_data_provider is None:
            raise ValueError("A prepared video sequence is required")
        torch.cuda.empty_cache()
        init_beta = real_data_provider.betas
        total_t = real_data_provider.total_t
        (
            model,
            optimizer,
            xyz_scheduler_func,
            w_dc_scheduler_func,
            w_rest_scheduler_func,
            sph_scheduler_func,
            sph_rest_scheduler_func,
            mlp_scheduler_func,
        ) = self._get_model_optimizer(betas=init_beta, add_bones_total_t=total_t)
        (optimizer_pose, scheduler_pose) = self._get_pose_optimizer(
            real_data_provider, model.add_bones
        )
        vox_dc_scheduler_func = get_expon_lr_func_interval(
            init_step=self.VOXEL_START_STEPS,
            final_step=getattr(self, "VOXEL_END_STEP", self.TOTAL_steps),
            lr_init=self.LR_VOXEL,
            lr_final=getattr(self, "LR_VOXEL_FINAL", self.LR_P_FINAL),
            lr_delay_mult=0.01,
        )
        stat_n_list = []
        (active_sph_order, last_reset_step) = (0, -1)
        seed_everything(self.SEED)
        running_start_t = time.time()
        logging.info(f"Start training at {running_start_t}")
        voxel_mlp = ImplicitNetwork(
            d_in=3,
            d_out=24,
            depth=4,
            width=128,
            multires=0,
            dim_cond_embed=0,
            weight_norm=True,
            geometric_init=False,
            bias=1,
        ).to("cuda")
        gt_weights = model.gt_weights
        gt_bone_points = model.gt_bone_points
        criterion = nn.MSELoss()
        ce = nn.CrossEntropyLoss()
        if self.VOXEL_INIT:
            optimizer_mlp = torch.optim.Adam(voxel_mlp.parameters(), lr=0.0001)
        else:
            optimizer_mlp = torch.optim.Adam(
                model.voxel_deformer.parameters(), lr=0.0001
            )
        device = gt_bone_points.device
        resolution_dhw = [64 // 4, 64, 64]
        (d, h, w) = resolution_dhw
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
        gt_bbox = torch.cat(
            [gt_bone_points.min(dim=1).values, gt_bone_points.max(dim=1).values], dim=0
        ).to(device)
        offset = (gt_bbox[0] + gt_bbox[1])[None, None, :] * 0.5
        scale = (gt_bbox[1] - gt_bbox[0]).max() / 2 * torch.Tensor([1.2]).squeeze()
        grid_denorm = self.denormalize(grid, offset, scale)
        weights = (
            self._query_weights_simple_init(
                grid_denorm,
                smpl_verts=gt_bone_points.detach().clone(),
                smpl_weights=gt_weights.detach().clone(),
                K=self.VOXEL_INIT_K,
            )
            .detach()
            .clone()
        )
        mask = weights.abs().sum(dim=1) > 0
        voxel_mlp.train()
        optimizer_mlp.zero_grad()
        loss_deform = 0.0
        for wm_step in tqdm(range(self.WARM_UP_STEPS)):
            if not self.VOXEL_INIT:
                optimizer_mlp.zero_grad()
                W_joints = model.voxel_deformer(gt_bone_points)
                loss_bone_w = torch.nn.functional.mse_loss(W_joints, gt_weights)
                loss_bone_w.backward()
                optimizer_mlp.step()
        for step in tqdm(range(self.TOTAL_steps)):
            update_learning_rate(xyz_scheduler_func(step), "xyz", optimizer)
            update_learning_rate(sph_scheduler_func(step), "f_dc", optimizer)
            update_learning_rate(sph_rest_scheduler_func(step), "f_rest", optimizer)
            update_learning_rate(mlp_scheduler_func(step), "mlp_params", optimizer)
            update_learning_rate(w_dc_scheduler_func(step), ["w_dc"], optimizer)
            update_learning_rate(w_rest_scheduler_func(step), ["w_rest"], optimizer)
            if self.VOXEL_CORRECTION_FLAG and step > self.VOXEL_START_STEPS:
                update_learning_rate(
                    vox_dc_scheduler_func(step), ["w_dc_vox"], optimizer
                )
            for k, v in scheduler_pose.items():
                update_learning_rate(v(step), k, optimizer_pose)
            if step in self.INCREASE_SPH_STEP:
                if active_sph_order < self.MAX_SPH_ORDER:
                    active_sph_order += 1
                    logging.info(f"active_sph_order: {active_sph_order}")
            model.train()
            (optimizer.zero_grad(), optimizer_pose.zero_grad())
            loss = 0.0
            real_data_pack = real_data_provider(self.N_POSES_PER_STEP, continuous=False)
            (
                loss_recon,
                loss_mask,
                loss_bone_w,
                render_list,
                gt_rgb,
                gt_mask,
                model_ret,
                loss_dict,
            ) = self._fit_step(
                model,
                data_pack=real_data_pack,
                act_sph_ord=active_sph_order,
                random_bg=self.RAND_BG_FLAG,
                use_box_crop_pad=getattr(self, "BOX_CROP_PAD", -1),
                default_bg=getattr(self, "DEFAULT_BG", [1.0, 1.0, 1.0]),
                step=step,
            )
            if step == self.VOXEL_START_STEPS and self.VOXEL_CORRECTION_FLAG:
                optimizer.add_param_group(
                    {
                        "params": [model.voxel_deformer.voxel_w_correction],
                        "lr": 0.0001,
                        "name": "w_dc_vox",
                    }
                )
            wandb.log(
                {
                    "loss_fit_step_recon": loss_recon.item(),
                    "loss_fit_step_mask": loss_mask.item(),
                }
            )
            loss = loss + loss_recon
            for k, v in loss_dict.items():
                self.add_scalar(k, v.detach(), step)
            if step > self.MLP_START_STEPS and step < self.MLP_END_STEPS:
                if step < self.MLP_STAGE1:
                    loss = loss + loss_bone_w * self.LAMBDA_BONE_W_STAGE1
                elif step >= self.MLP_STAGE1:
                    loss = loss + loss_bone_w * self.LAMBDA_BONE_W_STAGE2
            if (
                last_reset_step < 0
                or (
                    step - last_reset_step > self.MASK_LOSS_PAUSE_AFTER_RESET
                    and step > getattr(self, "MASK_START_STEP", 0)
                )
            ) and True:
                loss = loss + self.LAMBDA_MASK * loss_mask
                self.add_scalar("loss_mask", loss_mask.detach(), step)
            self.add_scalar("loss", loss.detach(), step)
            if step > self.REG_START_STEP:
                (reg_loss, reg_details) = self.compute_reg3D(model)
                loss = reg_loss + loss
                for k, v in reg_details.items():
                    self.add_scalar(k, v.detach(), step)
                wandb.log({"loss_reg3D": reg_loss.item()})
            wandb.log({"loss_final": loss.item()})
            loss.backward()
            optimizer.step()
            if step > getattr(self, "POSE_OPTIMIZE_START_STEP", -1):
                optimizer_pose.step()
            self.add_scalar("N", model.N, step)
            if step > self.DENSIFY_START:
                for render_pkg in render_list:
                    model.record_xyz_grad_radii(
                        render_pkg["viewspace_points"],
                        render_pkg["radii"],
                        render_pkg["visibility_filter"],
                    )
            if (
                step > self.DENSIFY_START
                and step < getattr(self, "DENSIFY_END", 10000000)
                and (step % self.DENSIFY_INTERVAL == 0)
            ):
                N_old = model.N
                model.densify(
                    optimizer=optimizer,
                    max_grad=self.MAX_GRAD,
                    percent_dense=self.PERCENT_DENSE,
                    extent=0.5,
                    verbose=True,
                )
                logging.info(f"Densify: {N_old}->{model.N}")
            if step > self.PRUNE_START and step % self.PRUNE_INTERVAL == 0:
                N_old = model.N
                model.prune_points(
                    optimizer,
                    min_opacity=self.OPACIT_PRUNE_TH,
                    max_screen_size=10000000000.0,
                    step=step,
                )
                logging.info(f"Prune: {N_old}->{model.N}")
            if step in getattr(self, "RANDOM_GROW_STEPS", []):
                model.random_grow(
                    optimizer,
                    num_factor=getattr(self, "NUM_FACTOR", 0.1),
                    std=getattr(self, "RANDOM_GROW_STD", 0.1),
                    init_opa_value=getattr(self, "RANDOM_GROW_OPA", 0.01),
                )
            if step + 1 in self.RESET_OPACITY_STEPS:
                model.reset_opacity(optimizer, self.OPACIT_RESET_VALUE)
                last_reset_step = step
            if step + 1 in getattr(self, "REGAUSSIAN_STEPS", []):
                model.regaussian(optimizer, self.REGAUSSIAN_STD)
            stat_n_list.append(model.N)
            if self.FAST_TRAINING:
                continue
            if self._should_preview(step):
                (mu, fr, s, o, sph) = model_ret[:5]
                save_path = f"{self.log_dir}/viz_step/step_{step}.png"
                viz_render(gt_rgb[0], gt_mask[0], render_list[0], save_path=save_path)
                (_, _, K, pose_base, pose_rest, global_trans, time_index) = (
                    real_data_pack
                )
                viz_spinning(
                    model,
                    torch.cat([pose_base, pose_rest], 1)[:1],
                    global_trans[:1],
                    real_data_provider.H,
                    real_data_provider.W,
                    K[0],
                    save_path=f"{self.log_dir}/viz_step/spinning_{step}.gif",
                    weight_path=f"{self.log_dir}/pose_weight_step/weight_{step}.png",
                    pose_path=f"{self.log_dir}/pose_weight_step/pose_{step}.png",
                    smpl_path=f"{self.log_dir}/pose_weight_step/smpl_{step}.png",
                    time_index=time_index,
                    active_sph_order=active_sph_order,
                    bg_color=[1.0, 1.0, 1.0],
                    step=step,
                )
                can_pose = model.template.real_canonical_pose.detach()
                if self.mode == "human":
                    can_pose[0] = self.viz_base_R.to(can_pose.device)
                    can_pose = matrix_to_axis_angle(can_pose)[None]
                else:
                    can_pose[:3] = matrix_to_axis_angle(self.viz_base_R[None])[0]
                    can_pose = can_pose[None]
                can_trans = torch.zeros(len(can_pose), 3).to(can_pose)
                can_trans[:, -1] = 3.0
                (viz_H, viz_W) = (512, 512)
                viz_K = fov2K(60, viz_H, viz_W)
                viz_spinning(
                    model,
                    can_pose,
                    can_trans,
                    viz_H,
                    viz_W,
                    viz_K,
                    save_path=f"{self.log_dir}/viz_step/spinning_can_{step}.gif",
                    weight_path=f"{self.log_dir}/pose_weight_step/can_weight_{step}.png",
                    pose_path=f"{self.log_dir}/pose_weight_step/can_pose_{step}.png",
                    smpl_path=f"{self.log_dir}/pose_weight_step/can_smpl_{step}.png",
                    time_index=None,
                    active_sph_order=active_sph_order,
                    bg_color=[1.0, 1.0, 1.0],
                    step=step,
                )
                plt.figure(figsize=(15, 3))
                scale = model.get_s
                for i in range(3):
                    _s = scale[:, i].detach().cpu().numpy()
                    plt.subplot(1, 3, i + 1)
                    plt.hist(_s, bins=100)
                    plt.title(f"scale {i}")
                    (plt.grid(), plt.ylabel("count"), plt.xlabel("scale"))
                plt.tight_layout()
                plt.savefig(f"{self.log_dir}/viz_step/s_hist_step_{step}.png")
                plt.close()
                plt.figure(figsize=(12, 4))
                opacity = model.get_o.squeeze(-1)
                plt.hist(opacity.detach().cpu().numpy(), bins=100)
                plt.title(f"opacity")
                (plt.grid(), plt.ylabel("count"), plt.xlabel("opacity"))
                plt.tight_layout()
                plt.savefig(f"{self.log_dir}/viz_step/o_hist_step_{step}.png")
                plt.close()
        running_end_t = time.time()
        logging.info(
            f"Training time: {running_end_t - running_start_t:.3f} seconds i.e. {(running_end_t - running_start_t) / 60.0:.3f} minutes"
        )
        logging.info("Saving model...")
        model.eval()
        ckpt_path = f"{self.log_dir}/model.pth"
        torch.save(model.state_dict(), ckpt_path)
        pose_path = f"{self.log_dir}/training_poses.pth"
        torch.save(real_data_provider.state_dict(), pose_path)
        model.to("cpu")
        plt.figure(figsize=(5, 5))
        plt.plot(stat_n_list)
        (plt.title("N"), plt.xlabel("step"), plt.ylabel("N"))
        plt.savefig(f"{self.log_dir}/N.png")
        plt.close()
        model.summary()
        return (model, real_data_provider)

    def denormalize(self, x, offset, scale):
        x_denormalized = x.clone()
        x_denormalized[..., -1] /= torch.tensor(4.0, dtype=torch.float32)
        x_denormalized *= scale
        x_denormalized += offset
        return x_denormalized

    def _query_weights_simple_init(self, x, smpl_verts, smpl_weights, K=30):
        (n, C) = (x.shape[1], smpl_weights.shape[-1])
        x_weights = torch.zeros((1, n, C), device=x.device)
        for i in range(smpl_verts.shape[1]):
            (dist, idx, _) = knn_points(smpl_verts[:, i : i + 1, :], x, K=K)
            x_weights[0, idx, :] = smpl_weights[:, i, :]
        c = smpl_weights.shape[-1]
        (d, h, w) = [64 // 4, 64, 64]
        weights = x_weights.permute(0, 2, 1).reshape(1, c, d, h, w)
        return weights.detach()

    def testtime_pose_optimization(
        self,
        data_pack,
        model,
        evaluator,
        pose_base_lr=0.003,
        pose_rest_lr=0.003,
        trans_lr=0.003,
        steps=100,
        decay_steps=30,
        decay_factor=0.5,
        check_every_n_step=5,
        viz_fn=None,
        As=None,
        pose_As_lr=0.001,
    ):
        torch.cuda.empty_cache()
        seed_everything(self.SEED)
        model.eval()
        evaluator.eval()
        (gt_rgb, gt_mask, K, pose_b, pose_r, trans) = data_pack[:6]
        pose_b = pose_b.detach().clone()
        pose_r = pose_r.detach().clone()
        trans = trans.detach().clone()
        pose_b.requires_grad_(True)
        pose_r.requires_grad_(True)
        trans.requires_grad_(True)
        (gt_rgb, gt_mask) = (gt_rgb.to(self.device), gt_mask.to(self.device))
        optim_l = [
            {"params": [pose_b], "lr": pose_base_lr},
            {"params": [pose_r], "lr": pose_rest_lr},
            {"params": [trans], "lr": trans_lr},
        ]
        if As is not None:
            As = As.detach().clone()
            As.requires_grad_(True)
            optim_l.append({"params": [As], "lr": pose_As_lr})
        optimizer_smpl = torch.optim.SGD(optim_l)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer_smpl, step_size=decay_steps, gamma=decay_factor
        )
        (loss_list, psnr_list, ssim_list, lpips_list) = ([], [], [], [])
        viz_step_list = []
        for inner_step in range(steps):
            optimizer_smpl.zero_grad()
            (loss_recon, _, _, rendered_list, _, _, _, _) = self._fit_step(
                model,
                [gt_rgb, gt_mask, K, pose_b, pose_r, trans, None],
                act_sph_ord=model.max_sph_order,
                random_bg=False,
                default_bg=getattr(self, "DEFAULT_BG", [1.0, 1.0, 1.0]),
                add_bones_As=As,
                step=50000,
            )
            loss = loss_recon
            loss.backward()
            optimizer_smpl.step()
            scheduler.step()
            loss_list.append(float(loss))
            if inner_step % check_every_n_step == 0 or (
                inner_step == steps - 1 and viz_fn is not None
            ):
                viz_step_list.append(inner_step)
                with torch.no_grad():
                    _check_results = [
                        evaluator(
                            rendered_list[0]["rgb"].permute(1, 2, 0)[None],
                            gt_rgb[0][None],
                        )
                    ]
                psnr = torch.stack([r["psnr"] for r in _check_results]).mean().item()
                ssim = torch.stack([r["ssim"] for r in _check_results]).mean().item()
                lpips = torch.stack([r["lpips"] for r in _check_results]).mean().item()
                psnr_list.append(float(psnr))
                ssim_list.append(float(ssim))
                lpips_list.append(float(lpips))
        if viz_fn is not None:
            os.makedirs(osp.dirname(viz_fn), exist_ok=True)
            plt.figure(figsize=(16, 4))
            plt.subplot(1, 4, 1)
            plt.plot(viz_step_list, psnr_list)
            (plt.title(f"PSNR={psnr_list[-1]}"), plt.grid())
            plt.subplot(1, 4, 2)
            plt.plot(viz_step_list, ssim_list)
            (plt.title(f"SSIM={ssim_list[-1]}"), plt.grid())
            plt.subplot(1, 4, 3)
            plt.plot(viz_step_list, lpips_list)
            (plt.title(f"LPIPS={lpips_list[-1]}"), plt.grid())
            plt.subplot(1, 4, 4)
            plt.plot(loss_list)
            (plt.title(f"Loss={loss_list[-1]}"), plt.grid())
            plt.yscale("log")
            plt.tight_layout()
            plt.savefig(viz_fn)
            plt.close()
        if As is not None:
            As.detach().clone()
        return (
            pose_b.detach().clone(),
            pose_r.detach().clone(),
            trans.detach().clone(),
            As,
        )

    @torch.no_grad()
    def eval_fps(self, model, real_data_provider, rounds=1):
        model.eval()
        logging.info(f"Model has {model.N} points.")
        N_frames = len(real_data_provider.rgb_list)
        ret = real_data_provider(N_frames)
        (gt_rgb, gt_mask, K, pose_base, pose_rest, global_trans, time_index) = ret
        pose = torch.cat([pose_base, pose_rest], 1)
        (H, W) = gt_rgb.shape[1:3]
        sph_o = model.max_sph_order
        logging.info(f"FPS eval using active_sph_order: {sph_o}")
        (mu, fr, sc, op, sph, _, _, _, _, _, _, _) = model(
            pose[0:1],
            global_trans[0:1],
            additional_dict={},
            active_sph_order=sph_o,
            step=50000,
        )
        bg = [1.0, 1.0, 1.0]
        render_pkg = render_cam_pcl(
            mu[0], fr[0], sc[0], op[0], sph[0], H, W, K[0], False, sph_o, bg
        )
        pred = render_pkg["rgb"].permute(1, 2, 0).detach().cpu().numpy()
        imageio.imsave(osp.join(self.log_dir, "fps_eval_sample.png"), pred)
        logging.info("Start FPS test...")
        start_t = time.time()
        for j in tqdm(range(int(N_frames * rounds))):
            i = j % N_frames
            (mu, fr, sc, op, sph, _, _, _, _, _, _, _) = model(
                pose[i : i + 1],
                global_trans[i : i + 1],
                additional_dict={"t": time_index},
                active_sph_order=sph_o,
                step=50000,
            )
            bg = [1.0, 1.0, 1.0]
            render_pkg = render_cam_pcl(
                mu[0], fr[0], sc[0], op[0], sph[0], H, W, K[0], False, sph_o, bg
            )
        end_t = time.time()
        fps = rounds * N_frames / (end_t - start_t)
        logging.info(f"FPS: {fps}")
        wandb.log({"FPS": fps})
        with open(osp.join(self.log_dir, "fps.txt"), "w") as f:
            f.write(f"FPS: {fps}")
        return fps

    @torch.no_grad()
    def view_parts(
        self,
        model,
        real_data_provider,
        rounds=1,
        fake_data_provider=None,
        test_every_n_step=-1,
        test_dataset=None,
        guidance=None,
    ):
        model.eval()
        torch.cuda.empty_cache()
        model.initCluster()
        real_flag = real_data_provider is not None
        if real_flag:
            init_beta = real_data_provider.betas
            total_t = real_data_provider.total_t
        data_len = 100
        for step in tqdm(range(1)):
            loss = 0.0
            if real_flag:
                real_data_pack = real_data_provider(data_len, continuous=False)
                (gt_rgb, gt_mask, K, pose_base, pose_rest, global_trans, time_index) = (
                    real_data_pack
                )
                for i in range(100):
                    pose = torch.cat([pose_base[i], pose_rest[i]], dim=0)
                    model.poseCluster(time_index[i], pose[None], global_trans[i][None])
        return (model, real_data_provider)

    @torch.no_grad()
    def view_weights(self, model, path):
        if not os.path.exists(path):
            os.mkdir(path)
        model.showBoneWeight(os.path.join(path, "3d_weights.png"))

    def _should_preview(self, step):
        periodic = (step + 1) % getattr(self, "VIZ_INTERVAL", 100) == 0
        return periodic or step == 0


def tg_fitting_eval(solver, dataset_mode, seq_name, optimized_seq):
    if dataset_mode == "people_snapshot":
        test(
            solver,
            seq_name=seq_name,
            tto_flag=True,
            tto_step=300,
            tto_decay=70,
            dataset_mode=dataset_mode,
            pose_base_lr=0.003,
            pose_rest_lr=0.003,
            trans_lr=0.003,
            training_optimized_seq=optimized_seq,
        )
    elif dataset_mode == "zju":
        test(
            solver,
            seq_name=seq_name,
            tto_flag=False,
            tto_step=50,
            tto_decay=20,
            dataset_mode=dataset_mode,
            pose_base_lr=0.004,
            pose_rest_lr=0.004,
            trans_lr=0.004,
            training_optimized_seq=optimized_seq,
        )
    elif dataset_mode == "monohuman":
        test(
            solver,
            seq_name=seq_name,
            tto_flag=False,
            tto_step=50,
            tto_decay=20,
            dataset_mode=dataset_mode,
            pose_base_lr=0.004,
            pose_rest_lr=0.004,
            trans_lr=0.004,
            training_optimized_seq=optimized_seq,
        )
    elif dataset_mode == "instant_avatar_wild":
        test(
            solver,
            seq_name=seq_name,
            tto_flag=True,
            tto_step=50,
            tto_decay=20,
            dataset_mode=dataset_mode,
            pose_base_lr=0.004,
            pose_rest_lr=0.004,
            trans_lr=0.004,
            training_optimized_seq=optimized_seq,
        )
    elif dataset_mode == "dog_demo":
        test(
            solver,
            seq_name=seq_name,
            tto_flag=True,
            tto_step=160,
            tto_decay=50,
            dataset_mode=dataset_mode,
            pose_base_lr=0.02,
            pose_rest_lr=0.02,
            trans_lr=0.02,
        )
    else:
        pass
    return
