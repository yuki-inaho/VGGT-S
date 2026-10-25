"""RGB-D preprocessing and exact inverse mapping for the semantic adapter.

The native VGGT encoder is square (default 518x518).  This module keeps the
original aspect ratio, pads the short side, and records enough metadata to
crop the padded prediction and resize it back to the original image size.

Depth is never bilinearly interpolated: missing depth (``depth == 0`` with
``valid == 0``) stays missing.  Depth and validity use nearest-neighbour
resizing, and the padded region is explicitly invalid.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PreprocessMeta:
    """Geometry of one RGB-D view mapped into the square encoder canvas."""

    orig_h: int
    orig_w: int
    resized_h: int
    resized_w: int
    pad_h: int
    pad_w: int
    canvas_size: int
    scale_x: float
    scale_y: float

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _as_rgb_float01(image: np.ndarray) -> torch.Tensor:
    """Validate RGB and return a ``3xHxW`` float tensor in ``[0, 1]``.

    Accepted contracts are deliberately narrow:

    * ``uint8`` in ``0..255`` (converted once by ``/255``),
    * floating in ``0..1`` (used as-is, never divided twice).

    Floating ``0..255`` is ambiguous and is rejected, as are negative or
    non-finite values and every other dtype.  There is no silent clipping or
    integer cast.
    """
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"RGB must be HxWx3, got {arr.shape}")
    if arr.dtype == np.uint8:
        tensor = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).float()
        return tensor / 255.0
    if np.issubdtype(arr.dtype, np.floating):
        if not np.isfinite(arr).all():
            raise ValueError("RGB contains non-finite float values")
        if float(arr.min()) < 0.0 or float(arr.max()) > 1.0:
            raise ValueError(
                "float RGB must be in [0,1]; float 0..255 is ambiguous and "
                "is rejected (pass uint8 or divide explicitly)")
        return torch.from_numpy(
            np.ascontiguousarray(arr, dtype=np.float32)).permute(2, 0, 1)
    raise TypeError(
        f"RGB must be uint8 [0,255] or float [0,1], got {arr.dtype}")


def _as_hw(value: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"{name} must be HxW, got {arr.shape}")
    return arr


def aspect_resize_pad_size(
    height: int,
    width: int,
    canvas_size: int = 518,
) -> Tuple[int, int, int, int]:
    """Return ``(resized_h, resized_w, pad_h, pad_w)`` for top-left padding."""
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    scale = float(canvas_size) / float(max(height, width))
    resized_h = min(canvas_size, max(1, int(round(height * scale))))
    resized_w = min(canvas_size, max(1, int(round(width * scale))))
    return resized_h, resized_w, canvas_size - resized_h, canvas_size - resized_w


def preprocess_view(
    rgb: np.ndarray,
    depth: np.ndarray,
    valid: np.ndarray,
    canvas_size: int = 518,
) -> Dict[str, Any]:
    """RGB-D view -> encoder canvas tensors + inverse metadata.

    Args:
        rgb: ``HxWx3`` uint8 in ``0..255`` or float in ``[0,1]``.  Float
            ``0..255`` is rejected rather than silently normalized twice.
        depth: ``HxW`` measured depth in metres.  ``0`` means missing.
        valid: ``HxW`` 0/1 depth validity mask.  ``0`` means missing.
        canvas_size: square native encoder input size.

    Returns:
        Dict with ``image`` (3,S,S float in [0,1]), ``depth`` (1,S,S metres),
        ``valid`` (1,S,S 0/1), and ``meta`` (``PreprocessMeta``).
    """
    orig_h, orig_w = np.asarray(rgb).shape[:2]
    rgb_t = _as_rgb_float01(rgb).unsqueeze(0)
    depth = _as_hw(depth, "depth").astype(np.float32)
    valid = _as_hw(valid, "valid").astype(np.float32)
    if depth.shape != (orig_h, orig_w) or valid.shape != (orig_h, orig_w):
        raise ValueError(
            f"RGB {(orig_h, orig_w)}, depth {depth.shape}, valid {valid.shape} "
            "must share the same original resolution")

    resized_h, resized_w, pad_h, pad_w = aspect_resize_pad_size(
        orig_h, orig_w, canvas_size)

    # RGB: antialiased bilinear is acceptable for appearance.
    rgb_t = F.interpolate(rgb_t,
                          size=(resized_h, resized_w),
                          mode="bilinear",
                          align_corners=False,
                          antialias=True)

    # Depth / validity: nearest only.  Never blend a missing 0 with a metre.
    depth_t = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)
    valid_t = torch.from_numpy(valid).unsqueeze(0).unsqueeze(0)
    depth_t = F.interpolate(depth_t,
                            size=(resized_h, resized_w),
                            mode="nearest")
    valid_t = F.interpolate(valid_t,
                            size=(resized_h, resized_w),
                            mode="nearest")
    valid_t = (valid_t > 0.5).float()
    depth_t = torch.clamp(depth_t, min=0.0) * valid_t

    image = F.pad(rgb_t, (0, pad_w, 0, pad_h), value=0.0)[0]
    depth_canvas = F.pad(depth_t, (0, pad_w, 0, pad_h), value=0.0)[0]
    valid_canvas = F.pad(valid_t, (0, pad_w, 0, pad_h), value=0.0)[0]

    meta = PreprocessMeta(
        orig_h=orig_h,
        orig_w=orig_w,
        resized_h=resized_h,
        resized_w=resized_w,
        pad_h=pad_h,
        pad_w=pad_w,
        canvas_size=canvas_size,
        scale_x=float(resized_w) / float(orig_w),
        scale_y=float(resized_h) / float(orig_h),
    )
    return {
        "image": image.contiguous(),
        "depth": depth_canvas.contiguous(),
        "valid": valid_canvas.contiguous(),
        "meta": meta,
    }


def align_depth_validity_tensor(
    depth: torch.Tensor,
    valid: torch.Tensor,
    meta: PreprocessMeta | Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply the exact RGB canvas aspect resize/pad to original-resolution depth.

    This function is for callers that hold original-resolution depth and the
    preprocessing metadata.  It uses nearest-neighbour resizing and top-left
    padding only; the padded region is invalid with zero depth.  Inputs already
    on the canvas must not be passed here.
    """
    if isinstance(meta, dict):
        meta = PreprocessMeta(**meta)
    if depth.shape != valid.shape:
        raise ValueError("depth and valid tensors must have the same shape")
    if depth.shape[-2:] != (meta.orig_h, meta.orig_w):
        raise ValueError(
            f"depth must be original {(meta.orig_h, meta.orig_w)}, got "
            f"{tuple(depth.shape[-2:])}; canvas-aligned depth should bypass "
            "alignment")
    lead = depth.shape[:-2]
    depth_flat = depth.reshape(-1, 1, meta.orig_h, meta.orig_w).float()
    valid_flat = valid.reshape(-1, 1, meta.orig_h, meta.orig_w).float()
    depth_flat = F.interpolate(depth_flat,
                               size=(meta.resized_h, meta.resized_w),
                               mode="nearest")
    valid_flat = F.interpolate(valid_flat,
                               size=(meta.resized_h, meta.resized_w),
                               mode="nearest")
    valid_flat = (valid_flat > 0.5).float()
    depth_flat = torch.clamp(depth_flat, min=0.0) * valid_flat
    depth_flat = F.pad(depth_flat,
                       (0, meta.pad_w, 0, meta.pad_h),
                       value=0.0)
    valid_flat = F.pad(valid_flat,
                       (0, meta.pad_w, 0, meta.pad_h),
                       value=0.0)
    depth_out = depth_flat.reshape(*lead, meta.canvas_size, meta.canvas_size)
    valid_out = valid_flat.reshape(*lead, meta.canvas_size, meta.canvas_size)
    return depth_out.to(depth.dtype), valid_out.to(valid.dtype)


def invert_logits(
    logits: torch.Tensor,
    meta: PreprocessMeta | Dict[str, Any],
    output_size: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    """Crop the padded region and resize semantic logits to native resolution.

    Args:
        logits: ``[..., h, w]`` logits on the encoder canvas (or a grid that
            maps to it by plain interpolation).
        meta: metadata from :func:`preprocess_view`.
        output_size: optional ``(H, W)``; defaults to the original image size.

    Returns:
        Logits with shape ``[..., H, W]``.  The padded right/bottom region is
        never resized back into the image.
    """
    if isinstance(meta, dict):
        meta = PreprocessMeta(**meta)
    grid_h, grid_w = logits.shape[-2:]
    # The native decoder can return a grid that is larger than the encoder
    # canvas (the decoder upsamples twice).  Crop in that grid directly so the
    # padded region is discarded before the single resize to native size.
    crop_h = max(1, int(round(meta.resized_h * grid_h / meta.canvas_size)))
    crop_w = max(1, int(round(meta.resized_w * grid_w / meta.canvas_size)))
    cropped = logits[..., :crop_h, :crop_w]
    if output_size is None:
        output_size = (meta.orig_h, meta.orig_w)
    out = F.interpolate(cropped,
                        size=(int(output_size[0]), int(output_size[1])),
                        mode="bilinear",
                        align_corners=False)
    return out


def invert_logits_batch(
    logits: torch.Tensor,
    metas: Sequence[PreprocessMeta | Dict[str, Any]],
    output_size: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    """Invert a batch, pairing each batch element with its own metadata."""
    if logits.shape[0] != len(metas):
        raise ValueError("logits batch and metadata length differ")
    return torch.stack([
        invert_logits(logits[i], metas[i], output_size=output_size)
        for i in range(logits.shape[0])
    ],
                     dim=0)
