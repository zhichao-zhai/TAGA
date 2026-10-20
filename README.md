# TAGA: Self-supervised Learning for Template-free Animatable Gaussian Articulated Model

**CVPR 2025**

[Paper](https://openaccess.thecvf.com/content/CVPR2025/html/Zhai_TAGA_Self-supervised_Learning_for_Template-free_Animatable_Gaussian_Articulated_Model_CVPR_2025_paper.html)

Official implementation of TAGA, which learns an animatable Gaussian articulated model from a monocular video.

## Installation

Use Linux with an NVIDIA GPU and the CUDA toolkit. The commands below follow [GART](https://github.com/JiahuiLei/GART)'s Python 3.9, PyTorch 2.0 and CUDA 11.8 environment:

```bash
conda create -n taga python=3.9 -y
conda activate taga
conda install pytorch==2.0.0 torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install pytorch3d -c pytorch3d
python -m pip install -r requirements.txt
```

Obtain the body-model interfaces, training LPIPS and alpha/depth rasterizer from [GART](https://github.com/JiahuiLei/GART/tree/16c11f8a5bb3ae249a9d04dc9d98c316e10f1126). From the TAGA repository root, link the directories and install the rasterizer:

```bash
GART_ROOT=/absolute/path/to/GART
ln -s "$GART_ROOT/lib_gart/smplx" lib_gart/smplx
ln -s "$GART_ROOT/lib_gart/smal" lib_gart/smal
ln -s "$GART_ROOT/utils/lpips" utils/lpips
python -m pip install "$GART_ROOT/lib_render/diff_gaussian_rasterization-alphadep"
```

See [dependency setup](docs/DEPENDENCIES.md) for the required interfaces and LPIPS weights, and [installation notes](docs/INSTALL.md) for setup checks and optional experiment tracking.

## Prepare data

Download the resources below from their original providers and accept the applicable licenses. Datasets, body-model assets and external pretrained weights are not redistributed in this repository.

<details>
<summary>SMPL body models</summary>

Download SMPL v1.1 (`SMPL_python_v.1.1.0.zip`) from the [SMPL website](https://smpl.is.tue.mpg.de/download.php) after registration and license acceptance. Rename the model files and place them under:

```text
data/smpl_model/
├── SMPL_MALE.pkl
├── SMPL_FEMALE.pkl
└── SMPL_NEUTRAL.pkl
```

</details>

<details>
<summary>PeopleSnapshot</summary>

Download the videos from [PeopleSnapshot](https://graphics.tu-bs.de/people-snapshot) and prepare images, masks, cameras and refined poses following [InstantAvatar](https://github.com/tijiang13/InstantAvatar). Place each prepared sequence under `data/people_snapshot/<sequence>/`, for example:

```text
data/people_snapshot/male-3-casual/
├── cameras.npz
├── images/
├── masks/
└── poses/
    ├── anim_nerf_train.npz
    └── anim_nerf_test.npz
```

</details>

<details>
<summary>ZJU-MoCap</summary>

Download the prepared ZJU-MoCap sequences and `smpl-meta.tar.gz` following [Instant-NVR](https://github.com/zju3dv/instant-nvr). Extract them into:

```text
data/
├── smpl-meta/SMPL_NEUTRAL.pkl
└── zju_mocap/
    ├── my_377/
    └── ...
```

</details>

<details>
<summary>Dog videos and D-SMAL</summary>

Download the prepared dog videos and BITE-estimated poses from [GART](https://github.com/JiahuiLei/GART#real-videos), and place each sequence under `data/dog_data_official/<sequence>/`, such as `alaskan/`.

Obtain the D-SMAL model assets following [BITE](https://github.com/runa91/bite_release) and place `smal_data/` under `lib_gart/smal/`. The [BITE license](https://github.com/runa91/bite_release/blob/master/LICENSE) and model terms apply.

</details>

See [data preparation](docs/DATA.md) for format details. To use an existing data directory, pass `--data-root /path/to/data`.

## Train and evaluate

Train on a PeopleSnapshot sequence:

```bash
python train.py --dataset people_snapshot --seq male-3-casual --output outputs/male-3
```

Evaluate the trained model:

```bash
python train.py --dataset people_snapshot --seq male-3-casual --operation eval --output outputs/male-3
```

For ZJU-MoCap or dog videos:

```bash
python train.py --dataset zju --seq my_377 --output outputs/zju-377
python train.py --dataset dog_demo --seq alaskan --output outputs/alaskan
```

See [usage](docs/USAGE.md) for motion-driven animation and configuration options.

## Citation

```bibtex
@InProceedings{Zhai_2025_CVPR,
    author    = {Zhai, Zhichao and Chen, Guikun and Wang, Wenguan and Zheng, Dong and Xiao, Jun},
    title     = {TAGA: Self-supervised Learning for Template-free Animatable Gaussian Articulated Model},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {21159-21169}
}
```

## Acknowledgements

Our implementation builds on [GART](https://github.com/JiahuiLei/GART), [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting), [InstantAvatar](https://github.com/tijiang13/InstantAvatar), [Instant-NVR](https://github.com/zju3dv/instant-nvr), [BITE](https://github.com/runa91/bite_release) and [LPIPS](https://github.com/richzhang/PerceptualSimilarity). We thank the authors for sharing their work.

## License

TAGA code is provided under the [MIT license](LICENSE.md), except for components with separate licenses. Gaussian-Splatting-derived files retain their research/evaluation terms. Downloaded dependencies, including the [SMPL implementation](https://github.com/JiahuiLei/GART/blob/16c11f8a5bb3ae249a9d04dc9d98c316e10f1126/lib_gart/smplx/LICENSE) and BITE, remain subject to their original licenses. See [third-party notices](THIRD_PARTY_NOTICES.md) for details.
