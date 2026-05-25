"""Differentiable tag rendering and camera response model (JAX)."""

from render_model.display import (
    IMSHOW_KWARGS,
    PIXELATED_STYLE,
    configure_matplotlib_image_display,
    centered_diff_limits,
    imshow_extent,
    pixelated_grid,
    pixelated_image,
)
from render_model.gamma import apply_gamma
from render_model.homography import (
    corner_rmse,
    corners_from_homography,
    homography_from_corners,
    homography_to_params,
    numpy_dlt_homography,
    numpy_sample_normal_offset_corners,
    numpy_sample_uniform_normal_offset_corners,
    params_to_homography,
)
from render_model.camera_params import (
    camera_params_physical_vector,
    decode_joint_params_per_step,
    model_params_to_vector,
    vector_to_model_params,
)
from render_model.optimizer import (
    JOINT_PARAM_NAMES,
    OptimizationParamTrace,
    OptimizeLMConfig,
    optimize_render_model_lm,
    optimize_render_model_lm_with_param_trace,
)
from render_model.pipeline import (
    SUPERSAMPLE,
    RenderModelParams,
    RenderModelStages,
    apply_render_model,
    bin_down,
    default_params,
    render_with_model,
    render_with_model_stages,
    scale_homography,
)
from render_model.psf import KERNEL_RADIUS, apply_gaussian_psf, gaussian_kernel_1d
from render_model.renderer import bbox_mask, render_tag_antialiased
from render_model.sharpen import apply_sharpening
from render_model.tag_data import TAG_CANONICAL_CORNERS, TAG_GRID_SIZE, build_tag_pattern

__all__ = [
    "IMSHOW_KWARGS",
    "KERNEL_RADIUS",
    "PIXELATED_STYLE",
    "SUPERSAMPLE",
    "TAG_CANONICAL_CORNERS",
    "TAG_GRID_SIZE",
    "JOINT_PARAM_NAMES",
    "OptimizationParamTrace",
    "OptimizeLMConfig",
    "RenderModelParams",
    "RenderModelStages",
    "apply_gamma",
    "apply_gaussian_psf",
    "apply_render_model",
    "apply_sharpening",
    "bbox_mask",
    "bin_down",
    "build_tag_pattern",
    "camera_params_physical_vector",
    "centered_diff_limits",
    "configure_matplotlib_image_display",
    "corner_rmse",
    "corners_from_homography",
    "decode_joint_params_per_step",
    "default_params",
    "gaussian_kernel_1d",
    "homography_from_corners",
    "homography_to_params",
    "imshow_extent",
    "model_params_to_vector",
    "numpy_dlt_homography",
    "numpy_sample_normal_offset_corners",
    "numpy_sample_uniform_normal_offset_corners",
    "optimize_render_model_lm",
    "optimize_render_model_lm_with_param_trace",
    "params_to_homography",
    "pixelated_grid",
    "pixelated_image",
    "render_tag_antialiased",
    "render_with_model",
    "render_with_model_stages",
    "scale_homography",
    "vector_to_model_params",
]
