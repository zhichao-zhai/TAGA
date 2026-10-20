# Data preparation

Download data and model assets from the original providers under their respective terms. Acquisition and license links are listed in the [README](../README.md#license). Use `--data-root` to select the prepared-data directory and `--model-path` to select a human body model stored elsewhere.

```text
data/
  smpl_model/SMPL_MALE.pkl
  smpl_model/SMPL_FEMALE.pkl
  smpl_model/SMPL_NEUTRAL.pkl
  smpl-meta/SMPL_NEUTRAL.pkl
  people_snapshot/<sequence>/
  zju_mocap/<sequence>/
  dog_data_official/<sequence>/
```

## Human body models

Obtain SMPL v1.1 from the [SMPL provider](https://smpl.is.tue.mpg.de/download.php). Select the appropriate gender model for PeopleSnapshot and name the files as shown above. The ZJU prepared-data convention uses `smpl-meta/SMPL_NEUTRAL.pkl`; `--model-path` selects the same model for its loader and fitting engine.

## PeopleSnapshot

Obtain the videos from [PeopleSnapshot](https://graphics.tu-bs.de/people-snapshot) and follow the [InstantAvatar preparation instructions](https://github.com/tijiang13/InstantAvatar) for camera, mask, pose and shape parameters. The loader expects `cameras.npz`, images, masks, and optimized pose/shape parameters in the InstantAvatar layout. Pose parameters include `betas`, `body_pose`/`thetas`, `global_orient`, and `transl`.

The sequence split definitions are in `lib_data/instant_avatar_people_snapshot.py`. New videos require corresponding tracking and split definitions. `utils/InstantAvatarPeopleProcessing.py` provides the included preprocessing CLI.

## ZJU-MoCap

Follow [Instant-NVR](https://github.com/zju3dv/instant-nvr) to obtain the prepared `my_*` sequence layout and matching `smpl-meta` assets. The loader uses `annots.npy`, images, masks, SMPL parameters, and vertices. Use that preparation convention, including its camera and body-parameter conventions, rather than substituting a raw-data layout.

## Dog videos

Follow the [GART data instructions](https://github.com/JiahuiLei/GART#real-videos) for the prepared dog videos, masks and BITE-estimated poses. Obtain D-SMAL assets through the [BITE model/data instructions](https://github.com/runa91/bite_release), and place the model assets under `lib_gart/smal/smal_data/` in your local dependency checkout.

## Animation motions

Human animation reads external motion `.npy` files under `novel_poses/`. Dog animation reads the expected pose `.npz` inputs under `novel_poses/husky/`. Obtain motions under the source dataset's terms and match the pose representation expected by `viz_utils.py`.
