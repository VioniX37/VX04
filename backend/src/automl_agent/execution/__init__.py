"""Code execution: templates, model registry, renderer, supervised sandbox and inference."""

from .inference import InferenceError, build_schema, load_bundle, predict_dataframe, score_file
from .model_registry import SUPPORTED_MODELS, normalize_model, supported_models
from .renderer import build_config, render_template
from .sandbox import kill_run_processes, run_script, static_check

__all__ = [
    "SUPPORTED_MODELS",
    "InferenceError",
    "build_config",
    "build_schema",
    "load_bundle",
    "kill_run_processes",
    "normalize_model",
    "predict_dataframe",
    "render_template",
    "run_script",
    "score_file",
    "static_check",
    "supported_models",
]
