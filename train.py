#!/usr/bin/env python3
"""Train, evaluate or visualize a TAGA model from a posed monocular video."""
import argparse
import datetime
import importlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
DATASETS = {
    "people_snapshot": ("fitter", "people_snapshot", "people_snapshot"),
    "zju": ("fitter", "zju", "zju_mocap"),
    "dog_demo": ("fitter", "dog", "dog_data_official"),
}
MODULES = {
    "torch": "PyTorch",
    "numpy": "numpy",
    "scipy": "scipy",
    "pytorch3d": "PyTorch3D",
    "cv2": "opencv-python",
    "imageio": "imageio",
    "matplotlib": "matplotlib",
    "open3d": "open3d",
    "sklearn": "scikit-learn",
    "omegaconf": "omegaconf",
    "distinctipy": "distinctipy",
    "chamferdist": "chamferdist CUDA extension",
    "diff_gaussian_rasterization": "alpha/depth Gaussian rasterizer",
    "transforms3d": "transforms3d",
    "trimesh": "trimesh",
    "plyfile": "plyfile",
    "tensorboard": "tensorboard",
    "pandas": "pandas",
    "torchmetrics": "torchmetrics",
    "lpips": "LPIPS evaluation package",
    "termcolor": "termcolor",
    "kornia": "kornia",
    "PIL": "Pillow",
    "tqdm": "tqdm",
    "pycocotools": "pycocotools",
    "torchvision": "torchvision",
    "geomloss": "geomloss",
    "openpyxl": "openpyxl",
}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=DATASETS, required=True)
    p.add_argument(
        "--seq", required=True, help="Sequence identifier in the prepared dataset"
    )
    p.add_argument(
        "--config", type=Path, help="YAML profile; defaults to the dataset profile"
    )
    p.add_argument("--data-root", type=Path, default=ROOT / "data")
    p.add_argument(
        "--model-path", type=Path, help="User-provided SMPL pkl; not used for dog_demo"
    )
    p.add_argument(
        "--output",
        type=Path,
        help="New run directory, or an existing run for eval/visualize",
    )
    p.add_argument(
        "--operation", choices=("train", "eval", "visualize"), default="train"
    )
    p.add_argument("--evaluate-after-training", action="store_true")
    p.add_argument(
        "--debug-visualizations",
        action="store_true",
        help="Export intermediate GMM and skinning diagnostics",
    )
    p.add_argument(
        "--wandb", choices=("disabled", "offline", "online"), default="disabled"
    )
    p.add_argument("--project", default="taga")
    p.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override an existing YAML key; value uses YAML syntax",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configuration and missing assets; do not import GPU modules",
    )
    p.add_argument(
        "--check",
        action="store_true",
        help="Check dependencies, assets and CUDA; do not start a run",
    )
    return p


def prepare(args, p):
    try:
        import yaml
    except ImportError:
        p.error("PyYAML is required: python -m pip install PyYAML")
    saved_run = {}
    if args.operation != "train" and args.output is not None:
        metadata = args.output.expanduser().resolve() / "run_metadata.json"
        if metadata.is_file():
            saved_run = json.loads(metadata.read_text())
            if (
                saved_run.get("dataset") != args.dataset
                or saved_run.get("sequence") != args.seq
            ):
                p.error("Dataset/sequence do not match the selected training run")
    engine, profile_name, data_folder = DATASETS[args.dataset]
    if args.operation != "train" and args.output is None:
        p.error("--output must identify a locally trained run for eval/visualize")
    default_profile = (
        (args.output / "config.yaml")
        if args.operation != "train"
        else ROOT / f"configs/{profile_name}.yaml"
    )
    profile = (args.config or default_profile).expanduser().resolve()
    if not profile.is_file():
        p.error(f"Configuration does not exist: {profile}")
    with profile.open() as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        p.error("The YAML configuration must be a mapping")
    for item in args.overrides:
        if "=" not in item:
            p.error(f"Expected KEY=VALUE, received {item!r}")
        key, raw = item.split("=", 1)
        if key not in cfg:
            p.error(f"Unknown configuration key: {key}")
        value = yaml.safe_load(raw)
        old = cfg[key]
        valid = type(value) is type(old) or (
            type(old) is float and type(value) in {int, float}
        )
        if not valid:
            p.error(
                f"{key} requires {type(old).__name__}, received {type(value).__name__}"
            )
        cfg[key] = float(value) if type(old) is float else value
    if type(cfg.get("TOTAL_steps")) is not int or cfg["TOTAL_steps"] <= 0:
        p.error("TOTAL_steps must be a positive integer")
    data_root = args.data_root.expanduser().resolve()
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output = (
        (args.output or ROOT / "outputs" / f"{args.dataset}-{args.seq}-{timestamp}")
        .expanduser()
        .resolve()
    )
    if output.exists() and not output.is_dir():
        p.error("--output must be a directory")
    if args.operation == "train" and output.exists() and any(output.iterdir()):
        p.error("Training output must be empty; choose a new --output directory")
    if args.dataset == "dog_demo":
        model = None
    elif args.model_path:
        model = args.model_path.expanduser().resolve()
    elif args.dataset == "zju":
        model = data_root / "smpl-meta/SMPL_NEUTRAL.pkl"
    else:
        gender = (
            "FEMALE"
            if args.seq.startswith("female")
            else "MALE" if args.seq.startswith("male") else "NEUTRAL"
        )
        model = data_root / f"smpl_model/SMPL_{gender}.pkl"
    return engine, profile, cfg, data_root, data_folder, output, model


def missing_assets(args, cfg, data_root, data_folder, output, model):
    required = {
        ROOT
        / "lib_gart/smplx/smplx/body_models.py": "SMPL transform provider; see README third-party setup",
        data_root / data_folder / args.seq: "prepared video sequence",
    }
    if args.dataset == "dog_demo":
        required[ROOT / "lib_gart/smal/smal_tpg.py"] = (
            "SMAL implementation for dog reconstruction"
        )
        required[ROOT / "lib_gart/smal/smal_data"] = (
            "separately obtained dog template assets"
        )
    else:
        required[model] = "separately obtained body template"
    if cfg.get("LAMBDA_LPIPS", 0) > 0:
        required[ROOT / "utils/lpips/lpips.py"] = (
            "LPIPS training module; see docs/DEPENDENCIES.md"
        )
        required[ROOT / "utils/lpips/weights/v0.1/vgg.pth"] = (
            "external LPIPS calibration weights"
        )
    if args.operation != "train":
        required[output / "model.pth"] = "your locally trained model"
        required[output / "training_poses.pth"] = (
            "your locally optimized training poses"
        )
    if args.operation == "visualize":
        required[
            ROOT
            / ("novel_poses/husky" if args.dataset == "dog_demo" else "novel_poses")
        ] = "separately provided animation poses"
    return [
        {"path": str(path), "reason": reason}
        for path, reason in required.items()
        if not path.exists()
    ]


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    engine, profile, cfg, data_root, data_folder, output, model = prepare(args, p)
    missing = missing_assets(args, cfg, data_root, data_folder, output, model)
    plan = {
        "status": "configuration",
        "dataset": args.dataset,
        "sequence": args.seq,
        "operation": args.operation,
        "engine": engine,
        "profile": str(profile),
        "debug_visualizations": args.debug_visualizations,
        "output": str(output),
        "data_root": str(data_root),
        "body_model": str(model) if model else None,
        "tracking": args.wandb,
        "config": cfg,
        "missing_assets": missing,
    }
    notes = []
    if cfg.get("THRESHOLD") != 0.3:
        notes.append(
            "JSD detection starts at 0.3 with q=0.85 and follows the model schedule; YAML THRESHOLD does not change the initial detection cutoff."
        )
    plan["effective_initial_js_threshold"] = 0.3
    if cfg.get("PRUNE_INTERVAL") != cfg.get("DENSIFY_INTERVAL"):
        notes.append("Pruning follows DENSIFY_INTERVAL.")
    if cfg.get("OPACIT_PRUNE_TH") != cfg.get("OPACIT_RESET_VALUE"):
        notes.append("Opacity pruning uses OPACIT_RESET_VALUE.")
    plan["configuration_notes"] = notes
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    missing_modules = [
        label
        for name, label in MODULES.items()
        if importlib.util.find_spec(name) is None
    ]
    if args.wandb != "disabled" and importlib.util.find_spec("wandb") is None:
        missing_modules.append("wandb (requested tracking mode)")
    if missing or missing_modules:
        print(
            json.dumps(
                {
                    "missing_assets": missing,
                    "missing_modules": missing_modules,
                    "next_step": "See docs/DEPENDENCIES.md and docs/DATA.md",
                },
                indent=2,
            )
        )
        return 2
    import torch

    if not torch.cuda.is_available():
        p.error("A working CUDA GPU is required for TAGA")
    if args.check:
        print("Dependency and path checks passed. Training has not been started.")
        for note in notes:
            print(note)
        return 0
    for note in notes:
        print("Configuration note: " + note, file=sys.stderr)
    os.environ["TAGA_DATA_ROOT"] = str(data_root)
    if model is not None:
        os.environ["TAGA_SMPL_MODEL_PATH"] = str(model)
    # Existing relative template paths are rooted at the source directory.
    os.chdir(ROOT)
    module = importlib.import_module(f"taga_engines.{engine}")
    from taga_runtime import tracking

    run = tracking.init(
        mode=args.wandb,
        project=args.project,
        config={**cfg, "dataset": args.dataset, "seq": args.seq},
        save_code=False,
    )
    solver = None
    try:
        if args.operation == "train":
            output.mkdir(parents=True, exist_ok=True)
            (output / "run_metadata.json").write_text(
                json.dumps(
                    {
                        "dataset": args.dataset,
                        "sequence": args.seq,
                        "engine": engine,
                    },
                    indent=2,
                )
            )
        run_profile = output / (
            "config.yaml"
            if args.operation == "train"
            else f"{args.operation}-config.yaml"
        )
        solver = module.TGFitter(
            log_dir=str(output),
            profile_fn=str(run_profile),
            mode="dog" if args.dataset == "dog_demo" else "human",
            template_model_path=str(model) if model else None,
            device=torch.device("cuda:0"),
            cfg=cfg,
            FAST_TRAINING=False,
            DEBUG_VIS=args.debug_visualizations,
        )
        provider, _ = solver.prepare_real_seq(
            args.seq,
            args.dataset,
            split="train",
            ins_avt_wild_start_end_skip=getattr(solver, "START_END_SKIP", None),
        )
        if args.operation == "train":
            run_kwargs = {"pre_data_provider": provider}
            if "init_data_provider" in inspect.signature(solver.run).parameters:
                run_kwargs["init_data_provider"] = None
            _, optimized = solver.run(provider, **run_kwargs)
            if args.evaluate_after_training:
                module.tg_fitting_eval(solver, args.dataset, args.seq, optimized)
        else:
            provider.load_state_dict(
                torch.load(output / "training_poses.pth"), strict=True
            )
            if args.operation == "eval":
                module.tg_fitting_eval(solver, args.dataset, args.seq, provider)
            else:
                from viz_utils import viz_human_all, viz_dog_all

                visualize = viz_dog_all if args.dataset == "dog_demo" else viz_human_all
                visualize(solver, provider, viz_name="visualize")
    finally:
        if solver is not None and hasattr(solver, "writer"):
            solver.writer.close()
        run.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
