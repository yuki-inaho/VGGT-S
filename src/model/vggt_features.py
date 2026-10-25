"""Frozen native VGGT feature extraction and RGB-D feature-cache contract.

This module never replaces the native VGGT aggregator or track head.  It calls
the released model's own ``aggregator`` and ``track_head.feature_extractor``.
The latter is the same feature-map path that the documented upstream
``feat=True`` interface returns; using it directly avoids an unrecorded edit to
the ignored upstream checkout.  A reproducible patch file is kept under
``docs/patches/`` for users who prefer the ``feat=True`` route.

Extraction must be done with actual ``N=1`` or ``N=2`` ordered views.  A
single-view cache is produced by a one-view encoder pass, never by deleting the
second view from an ``N=2`` pass.

The extraction device and compute dtype are explicit.  Storage dtype is kept
separate in the cache key and receipt, because a bf16 compute pass cached as
fp32 is not the same cache identity as an fp32 compute pass cached as fp32.
Depth validity is hashed into the key because it changes predictions even when
RGB and metric depth bytes are unchanged.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch

from .rgbd_preprocess import (PreprocessMeta, invert_logits_batch,
                              preprocess_view)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _dtype_from_name(name: str) -> torch.dtype:
    dtype = getattr(torch, str(name), None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"unknown torch dtype: {name}")
    return dtype


def _resolve_encoder_device(
    vggt: torch.nn.Module,
    device: Optional[Any],
) -> torch.device:
    """Use the requested device or one validated parameter device."""
    if device is not None:
        return torch.device(device)
    devices = {param.device for param in vggt.parameters()}
    if len(devices) != 1:
        raise ValueError(
            "cannot infer a single encoder device; pass device= explicitly "
            f"(found {sorted(str(d) for d in devices)})")
    return next(iter(devices))


def freeze_native_encoder(vggt: torch.nn.Module) -> Dict[str, int]:
    """Freeze the native aggregator and track head; return parameter counts."""
    groups = {}
    for name, module in (("aggregator", getattr(vggt, "aggregator", None)),
                         ("track_head", getattr(vggt, "track_head", None))):
        if module is None:
            groups[name] = 0
            continue
        count = 0
        for param in module.parameters():
            param.requires_grad_(False)
            count += param.numel()
        module.eval()
        groups[name] = count
    return groups


def native_encoder_grads_all_none(vggt: torch.nn.Module) -> bool:
    """True when every native encoder parameter currently has ``grad is None``."""
    for module in (getattr(vggt, "aggregator", None),
                   getattr(vggt, "track_head", None)):
        if module is None:
            continue
        for param in module.parameters():
            if param.grad is not None:
                return False
    return True


@torch.no_grad()
def extract_native_track_features(
    vggt: torch.nn.Module,
    images: torch.Tensor,
    *,
    device: Optional[Any] = None,
    compute_dtype: str = "float32",
) -> torch.Tensor:
    """Extract native track feature maps from actual ordered views.

    Args:
        vggt: released ``VGGT`` instance with ``aggregator`` and ``track_head``.
        images: ``[B, N, 3, H, W]`` in ``[0, 1]``.  ``N`` is the real number of
            extracted views (1 or 2).
        device: extraction device.  If omitted, the single validated device of
            the encoder parameters is used.
        compute_dtype: ``float32``, ``float16`` or ``bfloat16``.  CUDA
            autocast is enabled for the half types; it is not silently enabled
            on CPU.

    Returns:
        ``[B, N, C, H/2, W/2]`` native track feature maps (128 channels for the
        released VGGT-1B track head), detached and on the extraction device.
    """
    if images.ndim != 5:
        raise ValueError(f"images must be [B,N,3,H,W], got {tuple(images.shape)}")
    extract_device = _resolve_encoder_device(vggt, device)
    compute = _dtype_from_name(compute_dtype)
    images = images.to(device=extract_device)
    autocast_enabled = (extract_device != torch.device("cpu")
                        and compute in (torch.float16, torch.bfloat16))
    with torch.autocast(device_type=extract_device.type,
                        dtype=compute,
                        enabled=autocast_enabled):
        aggregated_tokens_list, patch_start_idx = vggt.aggregator(images)
        feature_maps = vggt.track_head.feature_extractor(
            aggregated_tokens_list, images, patch_start_idx)
    return feature_maps.detach()


def feature_cache_key(
    *,
    n_views: int,
    view_ids: Sequence[str],
    encoder_source_sha: str,
    encoder_weights_sha: str,
    rgb_shas: Sequence[str],
    depth_shas: Sequence[str],
    validity_shas: Sequence[str],
    preprocess: Sequence[PreprocessMeta | Dict[str, Any]],
    storage_dtype: str,
    compute_dtype: str,
    encoder_device_type: str,
) -> str:
    """Stable cache key covering view count/order and every prediction input."""
    if not (len(view_ids) == len(rgb_shas) == len(depth_shas)
            == len(validity_shas) == len(preprocess) == n_views):
        raise ValueError("cache-key metadata lengths must equal n_views")
    payload = {
        "n_views": int(n_views),
        "view_ids": list(view_ids),
        "encoder_source_sha": encoder_source_sha,
        "encoder_weights_sha": encoder_weights_sha,
        "rgb_shas": list(rgb_shas),
        "depth_shas": list(depth_shas),
        "validity_shas": list(validity_shas),
        "preprocess": [
            p.as_dict() if isinstance(p, PreprocessMeta) else dict(p)
            for p in preprocess
        ],
        "storage_dtype": str(storage_dtype),
        "compute_dtype": str(compute_dtype),
        "encoder_device_type": str(encoder_device_type),
    }
    encoded = json.dumps(payload, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(encoded)


def build_feature_cache(
    vggt: torch.nn.Module,
    rgb_views: Sequence[np.ndarray],
    depth_views: Sequence[np.ndarray],
    valid_views: Sequence[np.ndarray],
    *,
    encoder_source_sha: str,
    encoder_weights_sha: str,
    view_ids: Optional[Sequence[str]] = None,
    canvas_size: int = 518,
    device: Optional[Any] = None,
    compute_dtype: str = "float32",
    storage_dtype: str = "float32",
    storage_device: str = "cpu",
) -> Dict[str, Any]:
    """Run the frozen native encoder on the actual ordered views.

    The returned dict is the public feature-cache ABI consumed by
    ``SemanticAdapter.forward_cached``.  ``N=1`` runs a real one-view encoder
    pass.  If the native decoder needs two slots, the adapter duplicates the
    ``N=1`` self-feature explicitly and labels it as a fallback; the encoder
    result is not fabricated from an ``N=2`` pass.

    ``storage_dtype`` (what the cache keeps) is deliberately separate from
    ``compute_dtype`` (what the encoder executes in).  Both, the validity SHA,
    the device type, ordered view ids and extraction N are part of the key.
    """
    n_views = len(rgb_views)
    if n_views not in (1, 2):
        raise ValueError("this contract supports N=1 or N=2 actual views")
    if not (len(depth_views) == len(valid_views) == n_views):
        raise ValueError("rgb/depth/valid view counts differ")
    if view_ids is None:
        view_ids = [str(i) for i in range(n_views)]

    extract_device = _resolve_encoder_device(vggt, device)
    storage_torch_dtype = _dtype_from_name(storage_dtype)
    _dtype_from_name(compute_dtype)  # validate before any preprocessing

    prepared = [
        preprocess_view(rgb, depth, valid, canvas_size=canvas_size)
        for rgb, depth, valid in zip(rgb_views, depth_views, valid_views)
    ]
    images = torch.stack([p["image"] for p in prepared], dim=0).unsqueeze(0)
    aligned_depth = torch.stack([p["depth"] for p in prepared], dim=0).unsqueeze(0)
    depth_valid = torch.stack([p["valid"] for p in prepared], dim=0).unsqueeze(0)
    metas = [p["meta"] for p in prepared]

    feat_map = extract_native_track_features(
        vggt,
        images,
        device=extract_device,
        compute_dtype=compute_dtype,
    )
    feat_map = feat_map.detach().to(device=storage_device,
                                    dtype=storage_torch_dtype)
    aligned_depth = aligned_depth.to(device=storage_device)
    depth_valid = depth_valid.to(device=storage_device)

    rgb_shas = [_sha256_bytes(np.ascontiguousarray(rgb).tobytes())
                for rgb in rgb_views]
    depth_shas = [_sha256_bytes(np.ascontiguousarray(depth).tobytes())
                  for depth in depth_views]
    validity_shas = [_sha256_bytes(np.ascontiguousarray(valid).tobytes())
                     for valid in valid_views]
    key = feature_cache_key(n_views=n_views,
                            view_ids=view_ids,
                            encoder_source_sha=encoder_source_sha,
                            encoder_weights_sha=encoder_weights_sha,
                            rgb_shas=rgb_shas,
                            depth_shas=depth_shas,
                            validity_shas=validity_shas,
                            preprocess=metas,
                            storage_dtype=storage_dtype,
                            compute_dtype=compute_dtype,
                            encoder_device_type=extract_device.type)
    return {
        "format": "vggts-rgbd-feature-cache-v2",
        "cache_key": key,
        "n_views": n_views,
        "view_ids": list(view_ids),
        "feat_map": feat_map,
        "aligned_depth": aligned_depth,
        "depth_valid": depth_valid,
        "preprocess": [m.as_dict() for m in metas],
        "encoder_source_sha": encoder_source_sha,
        "encoder_weights_sha": encoder_weights_sha,
        "rgb_shas": rgb_shas,
        "depth_shas": depth_shas,
        "validity_shas": validity_shas,
        "storage_dtype": str(storage_dtype),
        "compute_dtype": str(compute_dtype),
        "storage_device": str(storage_device),
        "encoder_device": str(extract_device),
        "encoder_device_type": extract_device.type,
        "source": "native VGGT aggregator + track_head.feature_extractor",
        "singleview_fallback": n_views == 1,
    }


def invert_cached_logits(
    logits: torch.Tensor,
    cache: Dict[str, Any],
    output_size: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    """Invert semantic logits using the target-view cache metadata."""
    metas = cache["preprocess"]
    target_meta = metas[-1]
    size = (int(output_size[0]), int(output_size[1])) if output_size else None
    return invert_logits_batch(logits, [target_meta] * logits.shape[0], size)
