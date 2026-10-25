"""Automatic multi-class semantic adapter over frozen native VGGT-S features.

The adapter keeps the released VGGT-S native modules
(``PromptEncoder``, ``TwoWayTransformer``, ``MaskDecoder``) and the native
checkpoint, and adds only:

* learned class-query tokens (one per output class, including background),
* a learned logit calibration (scalar temperature + per-class bias),
* a small measured-depth/validity adapter that is added to each native
  per-view track feature map.

Inference is automatic: no GT points, masks, crops or class prompts are
accepted.  The output is a single competition logits tensor ``[B, K, H, W]``
containing every class.  Missing depth (``depth == 0`` and ``valid == 0``) is
never converted to a class and never forces the background class.

The native encoder and track head stay frozen and are not replaced by a
convolutional surrogate.  The trainable native decoder path receives gradients
from the automatic class-query objective.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder
from .rgbd_preprocess import (PreprocessMeta, align_depth_validity_tensor,
                              invert_logits)
from .transformer import TwoWayTransformer

NATIVE_PREFIXES = ("prompt_encoder.", "mask_decoder.")


class DepthValidityAdapter(nn.Module):
    """Small measured-depth + validity adapter for one native feature map.

    ``depth`` is metres.  ``valid == 0`` marks missing depth.  The network sees
    ``[clamp(depth, 0, 3) / 3 * valid, valid]``; it never receives a fabricated
    metric value for a missing pixel.
    """

    def __init__(self, embed_dim: int = 128, hidden_dim: int = 32) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.conv_in = nn.Conv2d(2, hidden_dim, kernel_size=3, padding=1)
        self.act = nn.GELU()
        self.conv_out = nn.Conv2d(hidden_dim, embed_dim, kernel_size=1)

    def forward(
        self,
        depth: torch.Tensor,
        valid: torch.Tensor,
        target_size: Tuple[int, int],
    ) -> torch.Tensor:
        """Return ``[B, N, C, Hf, Wf]`` depth embeddings."""
        if depth.ndim == 4:
            depth = depth.unsqueeze(2)
        if valid.ndim == 4:
            valid = valid.unsqueeze(2)
        if depth.shape != valid.shape:
            raise ValueError("depth and valid must have the same shape")
        b, n, _, h, w = depth.shape
        target_h, target_w = target_size
        if (h, w) != (target_h, target_w):
            depth = F.interpolate(depth.reshape(b * n, 1, h, w),
                                  size=(target_h, target_w),
                                  mode="nearest")
            valid = F.interpolate(valid.reshape(b * n, 1, h, w).float(),
                                  size=(target_h, target_w),
                                  mode="nearest")
        else:
            depth = depth.reshape(b * n, 1, h, w)
            valid = valid.reshape(b * n, 1, h, w).float()
        valid = (valid > 0.5).float()
        normalized = torch.clamp(depth, min=0.0, max=3.0) / 3.0 * valid
        x = torch.cat([normalized, valid], dim=1)
        x = self.conv_out(self.act(self.conv_in(x)))
        return x.reshape(b, n, self.embed_dim, target_h, target_w)


class SemanticAdapter(nn.Module):
    """Automatic ``K``-class semantic head on native VGGT-S decoder modules."""

    def __init__(
        self,
        num_classes: int = 5,
        img_size: int = 518,
        sa_tokens: int = 37,
        embed_dim: int = 128,
        blocks: int = 2,
        depth_hidden: int = 32,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError("num_classes must be positive")
        self.num_classes = int(num_classes)
        self.embed_dim = int(embed_dim)
        self.img_size = int(img_size)
        self.sa_tokens = int(sa_tokens)
        sa_img_attn_ratio = (img_size // 2 // sa_tokens)
        if sa_img_attn_ratio < 1:
            raise ValueError("img_size // 2 must be >= sa_tokens")
        self.sa_img_attn_ratio = sa_img_attn_ratio

        # Native modules, exactly as used by the released VGGT-S checkpoint.
        self.prompt_encoder = PromptEncoder(
            embed_dim=embed_dim,
            img_size=img_size,
            mask_in_chans=16,
            sa_img_attn_ratio=sa_img_attn_ratio,
            image_embedding_size=img_size // 2,
            sa_tokens=sa_tokens,
        )
        self.mask_decoder = MaskDecoder(
            transformer_dim=embed_dim,
            transformer=TwoWayTransformer(
                depth=blocks,
                embedding_dim=embed_dim,
                num_heads=8,
                sa_img_attn_ratio=sa_img_attn_ratio,
            ),
        )

        # New automatic semantic modules (not present in the native checkpoint).
        self.class_queries = nn.Embedding(self.num_classes, embed_dim)
        self.class_bias = nn.Parameter(torch.zeros(self.num_classes))
        self.logit_scale = nn.Parameter(torch.ones(()))
        self.depth_adapter = DepthValidityAdapter(embed_dim, depth_hidden)

        self.load_report: Optional[Dict[str, Any]] = None
        self._seed = int(seed)
        self._reset_new_parameters(self._seed)

    # ------------------------------------------------------------------ init
    def _reset_new_parameters(self, seed: int) -> None:
        """Deterministic initialization for the new, non-native modules."""
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        with torch.no_grad():
            self.class_queries.weight.normal_(mean=0.0, std=0.02, generator=gen)
            self.class_bias.zero_()
            self.logit_scale.fill_(1.0)
            for module in self.depth_adapter.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight,
                                            mode="fan_out",
                                            nonlinearity="relu",
                                            generator=gen)
                    if module.bias is not None:
                        module.bias.zero_()

    # -------------------------------------------------------------- checkpoint
    def load_native_checkpoint(
        self,
        checkpoint_path: str,
        *,
        strict_native: bool = True,
    ) -> Dict[str, Any]:
        """Load released native modules with an explicit new/missing-key report.

        Loads only ``ckpt['model']`` with the safe default
        ``weights_only=True``.  Every native ``prompt_encoder.*`` and
        ``mask_decoder.*`` tensor must be present; only the clearly new
        automatic modules may be reported as missing.  No native parameter is
        left randomly initialized when ``strict_native=True``.
        """
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        if not isinstance(state, dict):
            raise TypeError("checkpoint does not contain a state dictionary")

        stripped = OrderedDict()
        for key, value in state.items():
            clean = key[len("module."):] if key.startswith("module.") else key
            stripped[clean] = value

        ours = set(self.state_dict())
        # Report unexpected checkpoint keys against the ORIGINAL stripped
        # state, before filtering, so extras are never hidden.
        unexpected = sorted(set(stripped) - ours)
        cleaned = OrderedDict((k, v) for k, v in stripped.items() if k in ours)
        native_ours = {k for k in ours if k.startswith(NATIVE_PREFIXES)}
        native_loaded = {k for k in cleaned if k.startswith(NATIVE_PREFIXES)}
        missing = sorted(ours - set(cleaned))
        missing_native = sorted(native_ours - native_loaded)

        if strict_native and missing_native:
            raise RuntimeError(
                "native checkpoint is missing released keys; refusing random "
                f"native substitution: {missing_native[:8]}")

        self.load_state_dict(cleaned, strict=False)

        report = {
            "checkpoint": str(checkpoint_path),
            "torch_load_weights_only": True,
            "native_modules": ["prompt_encoder", "mask_decoder"],
            "native_loaded_count": len(native_loaded),
            "native_expected_count": len(native_ours),
            "new_keys": missing,
            "missing_native": missing_native,
            "unexpected_keys": unexpected,
            "replicated_native_parameters": [],
            "new_modules": [
                "class_queries", "class_bias", "logit_scale", "depth_adapter"
            ],
        }
        self.load_report = report
        return report

    # ----------------------------------------------------------------- helpers
    def _dense_ca_pe(self, height: int, width: int) -> torch.Tensor:
        base = self.prompt_encoder.ca_pe_layer((height, width)).unsqueeze(0)
        return torch.cat([
            base + self.prompt_encoder.ca_src_img_pe.weight.view(1, -1, 1, 1),
            base + self.prompt_encoder.ca_tar_img_pe.weight.view(1, -1, 1, 1),
        ],
                         dim=-1)

    def _dense_sa_pe(self, height: int, width: int) -> torch.Tensor:
        ratio = self.sa_img_attn_ratio
        if height % ratio or width % ratio:
            raise ValueError(
                f"feature grid {(height, width)} must be divisible by "
                f"sa_img_attn_ratio={ratio}")
        base = self.prompt_encoder.sa_pe_layer(
            (height // ratio, width // ratio)).unsqueeze(0)
        return torch.cat([
            base + self.prompt_encoder.sa_src_img_pe.weight.view(1, -1, 1, 1),
            base + self.prompt_encoder.sa_tar_img_pe.weight.view(1, -1, 1, 1),
        ],
                         dim=-1)

    def _align_depth_inputs(
        self,
        depth_b: torch.Tensor,
        depth_valid_b: torch.Tensor,
        preprocess_meta: Optional[Any],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Accept canvas-aligned depth, or align original depth with metadata.

        A shape that is neither the encoder canvas nor the original resolution
        is rejected rather than silently stretched to the feature grid.
        """
        if preprocess_meta is None:
            return depth_b, depth_valid_b
        if depth_b.shape != depth_valid_b.shape:
            raise ValueError("depth and valid must have the same shape")
        metas = (list(preprocess_meta)
                 if isinstance(preprocess_meta, (list, tuple))
                 else [preprocess_meta])
        metas = [PreprocessMeta(**m) if isinstance(m, dict) else m
                 for m in metas]
        n = depth_b.shape[1]
        if len(metas) not in (1, n):
            raise ValueError(
                f"expected one or {n} preprocessing metadata entries, got "
                f"{len(metas)}")

        aligned_depth: List[torch.Tensor] = []
        aligned_valid: List[torch.Tensor] = []
        for view in range(n):
            meta = metas[view] if len(metas) == n else metas[-1]
            depth_view = depth_b[:, view]
            valid_view = depth_valid_b[:, view]
            shape = tuple(depth_view.shape[-2:])
            canvas = (meta.canvas_size, meta.canvas_size)
            original = (meta.orig_h, meta.orig_w)
            if shape == canvas:
                aligned_depth.append(depth_view)
                aligned_valid.append(valid_view)
            elif shape == original:
                d, v = align_depth_validity_tensor(depth_view, valid_view, meta)
                aligned_depth.append(d)
                aligned_valid.append(v)
            else:
                raise ValueError(
                    f"depth view {view} is {shape}; expected canvas {canvas} "
                    f"or original {original}. Preprocess depth with the same "
                    "aspect resize/pad as RGB instead of stretching it")
        return (torch.stack(aligned_depth, dim=1),
                torch.stack(aligned_valid, dim=1))

    def parameter_groups(self) -> Dict[str, List[nn.Parameter]]:
        """Named trainable groups for native and new modules."""
        groups: Dict[str, List[nn.Parameter]] = {
            "native_prompt_encoder": [],
            "native_mask_decoder": [],
            "new_class_queries": [],
            "new_calibration": [],
            "new_depth_adapter": [],
        }
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("prompt_encoder."):
                groups["native_prompt_encoder"].append(param)
            elif name.startswith("mask_decoder."):
                groups["native_mask_decoder"].append(param)
            elif name.startswith("class_queries."):
                groups["new_class_queries"].append(param)
            elif name.startswith(("class_bias", "logit_scale")):
                groups["new_calibration"].append(param)
            elif name.startswith("depth_adapter."):
                groups["new_depth_adapter"].append(param)
            else:
                raise KeyError(f"unclassified parameter: {name}")
        return groups

    # ----------------------------------------------------------------- forward
    def forward(
        self,
        feat_map_b: torch.Tensor,
        depth_b: torch.Tensor,
        depth_valid_b: torch.Tensor,
        output_size: Optional[Sequence[int]] = None,
        preprocess_meta: Optional[Any] = None,
    ) -> torch.Tensor:
        """Predict ``[B, K, H, W]`` competition logits with no GT prompts.

        Args:
            feat_map_b: native track features ``[B, N, C, Hf, Wf]`` from an
                actual ``N=1`` or ``N=2`` frozen encoder pass.
            depth_b: measured depth in metres ``[B, N, h, w]`` (canvas or
                original resolution; resized nearest to ``Hf, Wf``).
            depth_valid_b: 0/1 validity ``[B, N, h, w]``.  ``0`` means missing.
            output_size: optional ``(H, W)`` output resolution.  When
                ``preprocess_meta`` is given, the padded canvas is cropped and
                resized exactly back to the original image.
            preprocess_meta: one ``PreprocessMeta`` for ``N=1``, or a sequence
                whose last element is the target-view metadata for ``N=2``.

        Returns:
            Competition logits over all ``K`` classes, including background.
        """
        if feat_map_b.ndim != 5:
            raise ValueError("feat_map_b must be [B,N,C,Hf,Wf]")
        b, n, c, hf, wf = feat_map_b.shape
        if n not in (1, 2):
            raise ValueError("N must be 1 or 2 actual extracted views")
        if c != self.embed_dim:
            raise ValueError(f"expected {self.embed_dim} channels, got {c}")
        if depth_b.shape[:2] != (b, n) or depth_valid_b.shape[:2] != (b, n):
            raise ValueError("depth/valid batch and view dimensions must match")

        depth_b, depth_valid_b = self._align_depth_inputs(
            depth_b, depth_valid_b, preprocess_meta)
        depth_embedding = self.depth_adapter(depth_b, depth_valid_b, (hf, wf))
        fused = feat_map_b + depth_embedding

        singleview_fallback = n == 1
        if singleview_fallback:
            # The native decoder has two ordered slots.  For a real N=1
            # encoder pass, duplicate the self-feature explicitly; this is not
            # a second view and no N=2 context was discarded.
            decoder_feat = torch.cat([fused, fused], dim=1)
        else:
            decoder_feat = fused

        ca_pe = self._dense_ca_pe(hf, wf)
        sa_pe = self._dense_sa_pe(hf, wf)

        # GT-free "no mask prompt": the native mask downscaling encoder sees an
        # all-zero mask.  It is a neutral prompt, not a class or background.
        zero_mask = torch.zeros(b, 1, hf * 2, wf * 2,
                                device=feat_map_b.device,
                                dtype=feat_map_b.dtype)
        src_dense = self.prompt_encoder.mask_downscaling(zero_mask)

        # Learned class queries are the only query tokens.  No point or mask
        # from GT is ever assembled into this tensor.
        class_tokens = self.class_queries.weight.unsqueeze(0).expand(b, -1, -1)
        empty_sparse = torch.zeros(b, 0, self.embed_dim,
                                   device=feat_map_b.device,
                                   dtype=feat_map_b.dtype)

        low_res_logits = self.mask_decoder(
            feat_map_b=decoder_feat,
            ca_img_pe=ca_pe,
            sa_img_pe=sa_pe,
            sparse_embeddings_b=empty_sparse,
            src_dense_embeddings_img_b=src_dense,
            tar_dense_embeddings_img_b=None,
            mask_token_b=class_tokens,
        )
        logits = low_res_logits * self.logit_scale
        logits = logits + self.class_bias.view(1, self.num_classes, 1, 1)

        if preprocess_meta is not None:
            meta = (preprocess_meta[-1] if isinstance(preprocess_meta,
                                                      (list, tuple))
                    else preprocess_meta)
            if isinstance(meta, dict):
                meta = PreprocessMeta(**meta)
            return invert_logits(logits, meta, output_size=output_size)
        if output_size is not None:
            logits = F.interpolate(logits,
                                   size=(int(output_size[0]), int(output_size[1])),
                                   mode="bilinear",
                                   align_corners=False)
        return logits

    def forward_cached(
        self,
        cache: Dict[str, Any],
        output_size: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        """Consume the native feature-cache ABI and return native-resolution logits."""
        if cache.get("n_views") != cache["feat_map"].shape[1]:
            raise ValueError("cache n_views does not match feat_map views")
        return self.forward(
            feat_map_b=cache["feat_map"],
            depth_b=cache["aligned_depth"],
            depth_valid_b=cache["depth_valid"],
            output_size=output_size,
            preprocess_meta=cache["preprocess"],
        )

    @torch.no_grad()
    def predict_proba(
        self,
        feat_map_b: torch.Tensor,
        depth_b: torch.Tensor,
        depth_valid_b: torch.Tensor,
        output_size: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        """Softmax competition probabilities ``[B, K, H, W]`` (inference helper)."""
        return torch.softmax(
            self.forward(feat_map_b, depth_b, depth_valid_b,
                         output_size=output_size), dim=1)
