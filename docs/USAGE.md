# Training, evaluation and animation

`train.py` operates on one sequence at a time. Select `people_snapshot`, `zju`, or `dog_demo` with `--dataset`, and pass the prepared sequence name with `--seq`.

## Train

```bash
python train.py --dataset people_snapshot --seq male-3-casual \
  --data-root /path/to/data --model-path /path/to/SMPL_MALE.pkl \
  --output outputs/male-3
```

The output directory must be empty when starting training. A run saves its configuration, model parameters, optimized poses, metrics and visualizations. Add `--evaluate-after-training` to run held-out evaluation after fitting.

Examples for the other datasets:

```bash
python train.py --dataset zju --seq my_377 --data-root /path/to/data --output outputs/zju-377
python train.py --dataset dog_demo --seq alaskan --data-root /path/to/data --output outputs/alaskan
```

## Configure a run

The default profiles are in `configs/`. Use `--config /path/to/profile.yaml` to select a profile, or override an existing key with `--set KEY=VALUE`:

```bash
python train.py --dataset people_snapshot --seq male-3-casual \
  --set TOTAL_steps=8000 --dry-run
```

Values use YAML syntax and must match the field type. `--dry-run` prints the selected configuration without importing GPU modules. `--check` checks paths, Python modules and CUDA before training.

W&B is disabled by default. Optional modes are `--wandb offline` and `--wandb online`. Use `--debug-visualizations` to export intermediate ambiguity-detection and skinning diagnostics.

## Evaluate

```bash
python train.py --dataset people_snapshot --seq male-3-casual \
  --operation eval --output outputs/male-3 --data-root /path/to/data \
  --model-path /path/to/SMPL_MALE.pkl
```

Evaluation loads the selected run's saved configuration, model parameters and optimized poses. The dataset and sequence must match the run metadata. The held-out evaluators compute PSNR, SSIM and LPIPS. For PeopleSnapshot, test poses are refined for 300 steps before computing the metrics. ZJU uses direct held-out evaluation, and dog evaluation also refines test poses.

## Animate

Prepare external motions as described in [DATA.md](DATA.md), then use `--operation visualize` with the trained run directory:

```bash
python train.py --dataset people_snapshot --seq male-3-casual \
  --operation visualize --output outputs/male-3 --data-root /path/to/data \
  --model-path /path/to/SMPL_MALE.pkl
```
