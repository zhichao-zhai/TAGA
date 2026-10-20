# Third-party notices

TAGA builds on [GART](https://github.com/JiahuiLei/GART) and other open research implementations. Copyright notices, source references and license exceptions in individual files remain in force.

| Component | Attribution and license |
| --- | --- |
| GART-derived implementation | Copyright Jiahui Lei, MIT; original notice in `licenses/GART-MIT.txt`. |
| `lib_gart/external.py`, `utils/general_utils.py`, `utils/system_utils.py`, `utils/ssim.py` | Inria/GRAPHDECO Gaussian-Splatting code. These files retain their separate non-commercial research/evaluation terms; see `licenses/Gaussian-Splatting.txt` and their source headers. |
| `lib_render/sh_utils.py` | PlenOctree authors; the full redistribution notice is retained in the file. |
| `utils/chamfer.py` | Facebook/PyTorch3D attribution is retained in the source. The BSD license is included in `licenses/PyTorch3D-BSD.txt`. |
| Neural/optimization utilities | Source references identify IDR, NeRF, SelfRecon and Plenoxels. IDR and NeRF MIT notices and the Plenoxels BSD notice are included under `licenses/`; source attributions remain in the corresponding files. |
| Data and evaluation utilities | Build on InstantAvatar, Instant-NVR and related prepared-data/evaluation conventions, as credited in source. |

The body-model implementations and assets, external LPIPS implementation/weights, datasets, motions and CUDA rasterizer are obtained separately from the providers linked in the [README](README.md#license). Each remains subject to its own terms; obtaining an upstream repository does not remove component-specific restrictions.
