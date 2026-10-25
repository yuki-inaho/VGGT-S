# Automatic semantic adapter for VGGT-S

This document describes the generic automatic multi-class semantic adapter.
It keeps the released VGGT-S native modules and checkpoint and does not replace
the VGGT-1B aggregator/track features with a convolutional surrogate.

## Contract

Inference inputs are:

1. ordered RGB views and their native track feature maps,
2. measured depth in metres,
3. a depth-validity mask.

There are **no** GT points, GT masks, target crops, class ids, or prompt points
in the inference API.  Semantic GT is used only by the separate loss/evaluation
pipeline.

The forward pass returns one competition logits tensor:

```text
[B, K, H, W]
```

`K` is configurable (`num_classes`) and includes every class, including
background.  The background class is an ordinary competing class; missing
depth does not force it.

## Native modules retained

`SemanticAdapter` instantiates the same native classes used by the released
predictor:

- `PromptEncoder`
- `TwoWayTransformer`
- `MaskDecoder`

`MaskDecoder` keeps its original default (`num_mask_tokens = 1`) and native
checkpoint keys.  The adapter passes `K` learned class queries through the
decoder's existing `mask_token_b` argument.  The one native binary
hypernetwork head is shared by those queries, so no native hypernetwork
parameter is replicated or randomized.  The native default path remains one
mask token and one head.

New, clearly identified modules:

| Module | Role |
|---|---|
| `class_queries` | learned query token per class; the only query source |
| `class_bias`, `logit_scale` | learned logit calibration |
| `depth_adapter` | measured-depth and validity embedding |

`load_native_checkpoint(..., strict_native=True)` loads the released state
dictionary with `torch.load(..., weights_only=True)`.  It reports:

- native keys expected and loaded,
- `new_keys` (only the new automatic modules),
- `missing_native` (must be empty),
- unexpected keys, computed from the original stripped checkpoint state before
  filtering into the adapter state (extras are never hidden).

If any `prompt_encoder.*` or `mask_decoder.*` key is missing, loading fails
instead of silently leaving a random native backbone.

Native parameters are exposed as groups by `parameter_groups()`:

- `native_prompt_encoder`
- `native_mask_decoder`
- `new_class_queries`
- `new_calibration`
- `new_depth_adapter`

## RGB-D geometry

The native encoder is square.  `rgbd_preprocess.preprocess_view` preserves the
input aspect ratio, resizes the long side to the canvas size, and pads the
short side.  Padding is top-left aligned, so the inverse operation is exact:
crop the padded region, then resize to the original image size.

Example: a `480 x 768` input with canvas `518` becomes `324 x 518` plus a
`194`-pixel bottom pad.  Prediction logits are cropped to `324 x 518` and
resized to `480 x 768`.  The padded rows are never resized back into the image.

RGB input contract:

- `uint8` in `0..255` is divided by 255 exactly once,
- floating RGB in `[0,1]` is used as-is and is never divided twice,
- float `0..255`, negative values, non-finite values, and non-uint8 integer
  arrays are rejected instead of being clipped or cast silently.

Depth rules:

- depth is measured metres,
- `depth == 0` with `valid == 0` is missing,
- depth and validity are resized with nearest-neighbour only,
- padded depth is zero and padded validity is zero,
- missing depth is never bilinearly blended into a fictitious metric value,
- the depth adapter sees `[clamp(depth, 0, 3) / 3 * valid, valid]`.

When `preprocess_meta` is passed to the adapter, depth/validity must be either
already on the encoder canvas or at the original image resolution.  Original
depth is aligned with the exact same aspect resize and top-left pad as RGB
(`align_depth_validity_tensor`).  Any other shape is rejected rather than
silently stretched to the feature grid.  `forward_cached` always uses the
cache's canvas-aligned depth, so RGB, depth and validity share one geometry.

The depth adapter is a small `Conv2d(2, hidden, 3) -> GELU -> Conv2d(hidden,
embed_dim, 1)` network.  Its output is added to each native per-view track
feature map before the native decoder.

Camera intrinsics are not required by this path.  If a caller uses K to align
depth, the transform must be documented by that caller; do not reproject
missing depth.

## Single-view and two-view

`N=1` means a real one-view frozen encoder extraction.  The native decoder
expects two ordered slots, so the adapter duplicates the `N=1` self-feature
explicitly and marks that as a decoder fallback.  It is not a second view and
does not reuse an `N=2` feature map with context removed.

`N=2` uses the actual ordered two-view extraction.  For the target-side
prediction, inverse preprocessing uses the last view's metadata.

## Frozen feature cache API

`model.vggt_features` provides:

```python
cache = build_feature_cache(
    vggt,
    rgb_views,          # ordered list of HxWx3 uint8 arrays, N = 1 or 2
    depth_views,        # ordered list of HxW metres
    valid_views,        # ordered list of HxW 0/1
    encoder_source_sha="<source commit or tree sha>",
    encoder_weights_sha="<native encoder weight sha>",
    view_ids=["view-0", "view-1"],
    canvas_size=518,
    device="cuda",              # or omit to infer one validated parameter device
    compute_dtype="bfloat16",  # encoder autocast compute dtype
    storage_dtype="float32",   # cache storage dtype
)
logits = adapter.forward_cached(cache, output_size=(H, W))
```

The cache contains:

- `feat_map`: `[1, N, 128, Hf, Wf]` native track features,
- `aligned_depth`, `depth_valid`: encoder-canvas tensors,
- `preprocess`: per-view `PreprocessMeta`,
- `cache_key`, `n_views`, `view_ids`, encoder/weight/RGB/depth/validity SHAs,
  storage dtype, compute dtype and encoder device type,
- `singleview_fallback`.

The extraction wrapper calls the native `vggt.aggregator` and
`vggt.track_head.feature_extractor` directly on the actual ordered views.  It
moves RGB to the explicit extraction device (or the single validated encoder
parameter device), applies CUDA autocast for half compute dtypes, then stores
the feature map in the separate storage dtype/device.  It does not need a
modification to the ignored upstream checkout.  For users who
prefer the documented `feat=True` interface, the reproducible patch is
`docs/patches/vggt-track-head-feat.patch` (SHA-256
`286dc3dd646aaadab192e98cc1c07e96629df31968115e4e689a1e741338fd83`), written
against VGGT source commit `a288dd0f14786c93483e45524328726ab7b1b4ce`.

The cache key covers:

```text
N / ordered view ids / encoder source SHA / encoder weight SHA /
RGB SHAs / depth SHAs / depth-validity SHAs / preprocess metadata /
storage dtype / compute dtype / encoder device type
```

So an `N=1` cache and an `N=2` cache never collide, reversing view order
changes the key, changing only validity changes the key, and a bf16-compute
cache is distinct from an fp32-compute cache even if both are stored as fp32.

## Minimal usage

```python
import torch
from model.semantic_adapter import SemanticAdapter
from model.vggt_features import build_feature_cache, freeze_native_encoder

freeze_native_encoder(vggt)

adapter = SemanticAdapter(num_classes=3)
report = adapter.load_native_checkpoint("official_ckpts/main_exp.pth")
assert report["missing_native"] == []

cache = build_feature_cache(
    vggt, rgb_views, depth_views, valid_views,
    encoder_source_sha=SOURCE_SHA,
    encoder_weights_sha=WEIGHTS_SHA,
    view_ids=view_ids,
)
logits = adapter.forward_cached(cache)          # [1, 3, H, W]
probs = torch.softmax(logits, dim=1)
```

Training uses the same forward pass.  Semantic GT may be resized for a
segmentation loss or evaluation, but it is never fed to the adapter.  The loss
must not select classes through depth validity.

## Tests

CPU-only synthetic tests cover:

- `K=3` and `K=5` competition logits, finiteness and softmax consistency,
- GT-free inference signature,
- depth-only perturbation changes logits,
- `depth=0, valid=0` does not force background and does not block prediction,
- gradients reach the depth adapter and native decoder parameters,
- an optimizer step changes an actual native tensor,
- deterministic initialization,
- explicit new/missing keys when loading a native checkpoint,
- frozen-encoder extraction has no grads and uses the actual `N` views,
- `N=1` decoder fallback equals explicit duplication,
- `N=1` vs `N=2` cache-key isolation,
- rectangular resize/pad/inverse alignment,
- the released public checkpoint loads with `weights_only=True`.

Synthetic feature-grid tests are architecture/shape tests.  They are not
measurements of real GPU throughput, native-resolution quality, or accuracy.

## Limitations

- The VGGT-1B encoder and track head remain fully sized and frozen; this scope
  did not run a GPU extraction or measure peak memory. Passing CPU tests with
  the cache helper is not proof of GPU N1/N2 extraction; that needs a separate
  actual-hardware run.
- The native decoder is trainable and receives the automatic class-query
  objective, but no full training or accuracy result is claimed here.
- The shared native binary hypernetwork head means class queries share one
  output projection; the class distinction comes from the learned query tokens
  and calibration.
- Real image quality, domain transfer, multi-view consistency, and the exact
  GPU memory/time envelope remain to be measured under a separate GPU gate.
- The native checkpoint license is not separately stated by its repository;
  the encoder checkpoint is CC-BY-NC-4.0.  Users must verify licensing for their
  use.
