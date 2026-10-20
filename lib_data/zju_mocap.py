from torch.utils.data import Dataset
import os
import numpy as np
import os.path as osp
import imageio
import cv2
from transforms3d.axangles import mat2axangle, axangle2mat
from pytorch3d.transforms import axis_angle_to_matrix
import torch
import sys

sys.path.append(osp.dirname(osp.abspath(__file__)))

from lib_gart.smplx.smplx import SMPLLayer

META = {
    "my_377": {"begin_ith_frame": 0, "num_train_frame": 100, "frame_interval": 5},
    "my_386": {"begin_ith_frame": 0, "num_train_frame": 100, "frame_interval": 5},
    "my_387": {"begin_ith_frame": 0, "num_train_frame": 100, "frame_interval": 5},
    # Subject 390 is excluded from evaluation.
    "my_392": {"begin_ith_frame": 0, "num_train_frame": 100, "frame_interval": 5},
    "my_393": {"begin_ith_frame": 0, "num_train_frame": 100, "frame_interval": 5},
    "my_394": {"begin_ith_frame": 0, "num_train_frame": 100, "frame_interval": 5},
}

# Frame subsampling follows the Instant-NVR evaluation protocol.
from torch.utils.data.sampler import Sampler


class FrameSampler(Sampler):
    """Sampler certain frames for test"""

    def __init__(self, dataset, frame_sampler_interval):
        inds = np.arange(0, len(dataset.ims))
        ni = len(dataset.ims) // dataset.num_cams
        inds = inds.reshape(ni, -1)[::frame_sampler_interval]
        self.inds = inds.ravel()

    def __iter__(self):
        return iter(self.inds)

    def __len__(self):
        return len(self.inds)


def get_batch_sampler(dataset, frame_sampler_interval=6):
    # instant-nvr use 6
    sampler = FrameSampler(dataset, frame_sampler_interval=frame_sampler_interval)
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, 1, False)
    return batch_sampler


DEBUG = False


class Dataset(Dataset):
    # from instant avatar
    def __init__(
        self,
        data_root="data/zju-mocap",
        video_name="my_377",
        split="train",
        image_zoom_ratio=0.5,
        # for cfg input from instant-nvr
        # for zju mocap instant-nvr use test_view: []; training_view: [4]
        training_view=[4],
        num_eval_frame=-1,
        test_novel_pose=False,
        bg_color=0.0,
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.video_name = video_name
        self.image_zoom_ratio = image_zoom_ratio

        self.bg_color = bg_color

        root = osp.join(data_root, video_name)

        anno_fn = osp.join(root, "annots.npy")
        annots = np.load(anno_fn, allow_pickle=True).item()

        self.cams = annots["cams"]

        # Match the frame split used by Instant-NVR evaluation.

        num_cams = len(self.cams["K"])
        test_view = [i for i in range(num_cams) if i not in training_view]
        if len(test_view) == 0:
            test_view = [0]

        if split == "train" or split == "prune":
            self.view = training_view  # [1]  0,570,1
        elif split == "test":
            self.view = test_view
        elif split == "val":
            self.view = test_view[::4]

        i = META[self.video_name]["begin_ith_frame"]
        i_intv = META[self.video_name]["frame_interval"]
        self.f_intv = i_intv
        ni = META[self.video_name]["num_train_frame"]
        if split == "val":
            self.view = [5]
            self.tick = 0
            ni = 500
            i_intv = 1
        if test_novel_pose:
            i = (
                META[self.video_name]["begin_ith_frame"]
                + META[self.video_name]["num_train_frame"] * i_intv
            )
            ni = num_eval_frame

        self.ims = np.array(
            [
                np.array(ims_data["ims"])[self.view]
                for ims_data in annots["ims"][i : i + ni * i_intv][::i_intv]
            ]
        ).ravel()
        self.cam_inds = np.array(
            [
                np.arange(len(ims_data["ims"]))[self.view]
                for ims_data in annots["ims"][i : i + ni * i_intv][::i_intv]
            ]
        ).ravel()
        self.num_cams = len(self.view)

        # Transform the body model into each camera coordinate frame.

        # Camera extrinsics map world coordinates to camera coordinates:
        # p_c = T_cw @ p_w.

        # Project the low-precision camera rotation back to SO(3) using SVD.
        for cid in range(num_cams):
            _R = self.cams["R"][cid]
            u, s, vh = np.linalg.svd(_R)
            new_R = u @ vh
            self.cams["R"][cid] = new_R

        smpl_layer = SMPLLayer(
            os.environ.get(
                "TAGA_SMPL_MODEL_PATH",
                osp.join(
                    os.environ.get("TAGA_DATA_ROOT", "./data"),
                    "smpl-meta",
                    "SMPL_NEUTRAL.pkl",
                ),
            )
        )

        # Transform SMPL parameters into camera coordinates.
        self.smpl_theta_list, self.smpl_trans_list, smpl_beta_list = [], [], []
        self.meta = []
        for img_fn in self.ims:
            cam_ind = int(img_fn.split("/")[-2])
            frame_idx = int(img_fn.split("/")[-1].split(".")[0])
            self.meta.append({"cam_ind": cam_ind, "frame_idx": frame_idx})
            smpl_fn = osp.join(root, "smpl_params", f"{frame_idx}.npy")
            smpl_data = np.load(smpl_fn, allow_pickle=True).item()
            T_cw = np.eye(4)
            T_cw[:3, :3], T_cw[:3, 3] = (
                np.array(self.cams["R"][cam_ind]),
                np.array(self.cams["T"][cam_ind]).squeeze(-1) / 1000.0,
            )

            smpl_theta = smpl_data["poses"].reshape((24, 3))
            assert np.allclose(smpl_theta[0], 0)
            smpl_rot, smpl_trans = smpl_data["Rh"][0], smpl_data["Th"]
            smpl_R = axangle2mat(
                smpl_rot / (np.linalg.norm(smpl_rot) + 1e-6), np.linalg.norm(smpl_rot)
            )

            T_wh = np.eye(4)
            T_wh[:3, :3], T_wh[:3, 3] = smpl_R.copy(), smpl_trans.squeeze(0).copy()

            T_ch = T_cw.astype(np.float64) @ T_wh.astype(np.float64)

            smpl_global_rot_d, smpl_global_rot_a = mat2axangle(T_ch[:3, :3])
            smpl_global_rot = smpl_global_rot_d * smpl_global_rot_a

            smpl_trans = T_ch[:3, 3]  # 3

            smpl_theta[0] = smpl_global_rot
            beta = smpl_data["shapes"][0][:10]

            # Correct translation because SMPL global rotation is centered at joint 0.
            _pose = axis_angle_to_matrix(torch.from_numpy(smpl_theta)[None])
            so = smpl_layer(
                torch.from_numpy(beta)[None],
                body_pose=_pose[:, 1:],
            )
            j0 = (so.joints[0, 0]).numpy()
            t_correction = (_pose[0, 0].numpy() - np.eye(3)) @ j0
            smpl_trans = smpl_trans + t_correction

            self.smpl_theta_list.append(smpl_theta)
            smpl_beta_list.append(beta)
            self.smpl_trans_list.append(smpl_trans)

            if DEBUG:
                vtx_fn = osp.join(root, "vertices", f"{frame_idx}.npy")
                nb_vtx_world = np.load(vtx_fn)
                np.savetxt("../debug/nb_vtx_world.xyz", nb_vtx_world, fmt="%.6f")
                nb_vtx_cam = np.dot(nb_vtx_world.copy(), T_cw[:3, :3].T) + T_cw[:3, 3]
                np.savetxt("../debug/nb_vtx_cam.xyz", nb_vtx_cam, fmt="%.6f")
                T_hw = np.linalg.inv(T_wh)
                nb_vtx_human = np.dot(nb_vtx_world.copy(), T_hw[:3, :3].T) + T_hw[:3, 3]
                Rh = smpl_data["Rh"][0]
                R = cv2.Rodrigues(Rh)[0].astype(np.float32)
                Th = smpl_data["Th"][0]
                nb_vtx_human2 = np.dot(nb_vtx_world.copy() - Th, R)
                np.savetxt("../debug/nb_vtx_human2.xyz", nb_vtx_human2, fmt="%.6f")
                np.savetxt("../debug/nb_vtx_human.xyz", nb_vtx_human, fmt="%.6f")

                smpl_vtx_human2 = (
                    smpl_layer(
                        torch.from_numpy(beta)[None],
                        body_pose=_pose[:, 1:],
                        global_orient=_pose[:, 0],
                        transl=torch.from_numpy(smpl_trans)[None],
                    )
                    .vertices[0]
                    .numpy()
                )
                np.savetxt("../debug/smpl_vtx_cam2.xyz", smpl_vtx_human2, fmt="%.6f")

                smpl_vtx_human = smpl_layer(
                    torch.from_numpy(beta)[None], body_pose=_pose[:, 1:]
                )
                smpl_vtx_human = smpl_vtx_human.vertices[0].numpy()
                np.savetxt("../debug/smpl_vtx_human.xyz", smpl_vtx_human, fmt="%.6f")
                smpl_vtx_world = np.dot(smpl_vtx_human, T_wh[:3, :3].T) + T_wh[:3, 3]
                np.savetxt("../debug/smpl_vtx_world.xyz", smpl_vtx_world, fmt="%.6f")
                smpl_vtx_cam = np.dot(smpl_vtx_human, T_ch[:3, :3].T) + T_ch[:3, 3]
                np.savetxt("../debug/smpl_vtx_cam.xyz", smpl_vtx_cam, fmt="%.6f")

                # the smpl and nb are aligned

                img = imageio.imread(osp.join(root, img_fn)).astype(np.float32) / 255.0
                K = np.array(self.cams["K"][cam_ind])
                screen_smpl_vtx = np.dot(smpl_vtx_cam.copy(), K.T)
                screen_smpl_vtx = screen_smpl_vtx[:, :2] / screen_smpl_vtx[:, 2:]
                screen_smpl_vtx = screen_smpl_vtx.astype(np.int32)
                dbg = img.copy()
                for uv in screen_smpl_vtx:
                    dbg[uv[1], uv[0], :] = 1
                imageio.imsave("../debug/dbg.png", dbg)
                imageio.imsave("../debug/img.png", img)

                K = np.array(self.cams["K"][cam_ind])
                screen_smpl_vtx = np.dot(smpl_vtx_human2.copy(), K.T)
                screen_smpl_vtx = screen_smpl_vtx[:, :2] / screen_smpl_vtx[:, 2:]
                screen_smpl_vtx = screen_smpl_vtx.astype(np.int32)
                dbg = img.copy()
                for uv in screen_smpl_vtx:
                    dbg[uv[1], uv[0]] = 1
                imageio.imsave("../debug/dbg2.png", dbg)
                print()
        self.beta = np.array(smpl_beta_list).mean(0)

        return

    def __len__(self):
        return len(self.ims)

    def __getitem__(self, index):
        img_path = os.path.join(self.data_root, self.video_name, self.ims[index])
        img = imageio.imread(img_path).astype(np.float32) / 255.0
        mask_path = os.path.join(
            self.data_root,
            self.video_name,
            self.ims[index].replace("images", "mask").replace(".jpg", ".png"),
        )
        msk = imageio.imread(mask_path)

        H, W = img.shape[:2]
        msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
        cam_ind = self.cam_inds[index]
        K = np.array(self.cams["K"][cam_ind])
        D = np.array(self.cams["D"][cam_ind])
        img = cv2.undistort(img, K, D)
        msk = cv2.undistort(msk, K, D)

        H, W = int(img.shape[0] * self.image_zoom_ratio), int(
            img.shape[1] * self.image_zoom_ratio
        )
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
        K[:2] = K[:2] * self.image_zoom_ratio

        img[msk == 0] = self.bg_color

        ret = {
            "rgb": img.astype(np.float32),
            "mask": msk.astype(np.bool).astype(np.float32),
            "K": K.copy().astype(np.float32),
            "smpl_beta": self.beta.astype(np.float32),
            "smpl_pose": self.smpl_theta_list[index].astype(np.float32),
            "smpl_trans": self.smpl_trans_list[index].astype(np.float32),
            "idx": index,
        }

        assert cam_ind == self.meta[index]["cam_ind"]

        meta_info = {
            "video": self.video_name,
            "cam_ind": cam_ind,
            "frame_idx": self.meta[index]["frame_idx"],
        }
        viz_id = f"video{self.video_name}_dataidx{index}"
        meta_info["viz_id"] = viz_id
        return ret, meta_info
