# Installation

TAGA uses Linux, an NVIDIA CUDA GPU, Python 3.9, PyTorch, torchvision, PyTorch3D, and CUDA rendering/Chamfer extensions. Install matching binary builds and compile extensions against the same PyTorch/CUDA environment. The [GART installation recipe](https://github.com/JiahuiLei/GART/blob/16c11f8a5bb3ae249a9d04dc9d98c316e10f1126/install.sh) provides a reference environment with PyTorch 2.0 and CUDA 11.8.

1. Create a dedicated environment and install compatible PyTorch, torchvision and PyTorch3D builds.
2. Install the Python requirements:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Obtain the body-model implementation, training LPIPS, and alpha/depth rasterizer described in [DEPENDENCIES.md](DEPENDENCIES.md). Obtain model assets and video data as described in [DATA.md](DATA.md). Their licenses and acquisition links are summarized in the [README](../README.md#license).
4. Check your setup with the selected sequence and paths:

   ```bash
   python train.py --dataset people_snapshot --seq male-3-casual \
     --data-root /path/to/data --model-path /path/to/SMPL_MALE.pkl --check
   ```

`--check` checks module availability, configured paths, and CUDA availability. `--dry-run` only prints the run configuration and missing paths, and needs Python and PyYAML.

For optional W&B tracking, install `requirements-tracking.txt` and choose `--wandb offline` or `--wandb online`. Tracking is disabled by default.
