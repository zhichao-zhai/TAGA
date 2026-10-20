"""Optional experiment tracking. Disabled mode never imports the W&B SDK."""

config = {}
_backend = None


class _LocalRun:
    def finish(self):
        pass


def init(*, mode="disabled", config=None, **kwargs):
    global _backend
    globals()["config"] = dict(config or {})
    if mode == "disabled":
        _backend = None
        return _LocalRun()
    if mode not in {"offline", "online"}:
        raise ValueError(f"Unknown tracking mode: {mode}")
    import wandb

    _backend = wandb
    run = wandb.init(mode=mode, config=globals()["config"], **kwargs)
    globals()["config"] = dict(wandb.config)
    return run


def log(data, *args, **kwargs):
    if _backend is not None:
        return _backend.log(data, *args, **kwargs)


def Image(data, *args, **kwargs):
    if _backend is not None:
        return _backend.Image(data, *args, **kwargs)
    return None
