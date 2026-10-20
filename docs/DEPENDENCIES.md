# Third-party setup

Obtain the components below from their upstream sources under their respective terms. The [README](../README.md#license) lists license and resource links.

## Body-model and LPIPS interfaces

The following GART paths provide the interfaces used by TAGA. The links refer to GART revision `16c11f8a5bb3ae249a9d04dc9d98c316e10f1126`.

| Component | Upstream path | Local path |
| --- | --- | --- |
| SMPL transform provider | [GART `lib_gart/smplx`](https://github.com/JiahuiLei/GART/tree/16c11f8a5bb3ae249a9d04dc9d98c316e10f1126/lib_gart/smplx) | `lib_gart/smplx/` |
| Dog transform provider | [GART `lib_gart/smal`](https://github.com/JiahuiLei/GART/tree/16c11f8a5bb3ae249a9d04dc9d98c316e10f1126/lib_gart/smal) | `lib_gart/smal/` |
| Training LPIPS | [GART `utils/lpips`](https://github.com/JiahuiLei/GART/tree/16c11f8a5bb3ae249a9d04dc9d98c316e10f1126/utils/lpips) | `utils/lpips/` |

After obtaining an upstream GART checkout, the directories can be linked into TAGA locally:

```bash
ln -s /path/to/GART/lib_gart/smplx lib_gart/smplx
ln -s /path/to/GART/lib_gart/smal lib_gart/smal
ln -s /path/to/GART/utils/lpips utils/lpips
```

SMPL model files are obtained separately from SMPL. Dog reconstruction also needs the D-SMAL model assets described in [DATA.md](DATA.md).

The SMPL implementation must return the joint transforms and canonical joints used by `templates.py` (`A` and `J`). The linked GART implementation provides these outputs. The ordinary pip SMPL-X interface is not interchangeable at these call sites.

Training imports `utils.lpips`, while held-out metrics use the `lpips` package. Keep the local GART training implementation, its `weights/v0.1` calibration files, and the upstream VGG feature weights available. The version `v0.1` identifies the LPIPS metric weights, not the Python package release number.

## Gaussian rasterizer

Install [GART's alpha/depth rasterizer](https://github.com/JiahuiLei/GART/tree/16c11f8a5bb3ae249a9d04dc9d98c316e10f1126/lib_render/diff_gaussian_rasterization-alphadep) in the active PyTorch/CUDA environment:

```bash
python -m pip install /path/to/GART/lib_render/diff_gaussian_rasterization-alphadep
```

The wrapper expects RGB, radii, depth and alpha from `diff_gaussian_rasterization`. Other implementations may use the same module name with a different return signature; use the linked implementation for this interface. Its [license](https://github.com/JiahuiLei/GART/blob/16c11f8a5bb3ae249a9d04dc9d98c316e10f1126/lib_render/diff_gaussian_rasterization-alphadep/LICENSE.md) applies independently of the TAGA license.

## Chamfer distance

The Python requirements include [chamferdist](https://github.com/krrish94/chamferdist). Install a build compatible with the active PyTorch/CUDA environment. TAGA's Chamfer wrapper retains its PyTorch3D source attribution.
