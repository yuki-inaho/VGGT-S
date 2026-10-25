# Generic uv environment for VGGT-S

## Pinned environment

- Python 3.10 (uv-managed)
- `torch==2.8.0+cu128`
- `torchvision==0.23.0+cu128`
- `numpy==1.26.4`
- Remaining direct pins are in `pyproject.toml`; the complete resolution is in
  `uv.lock`.

The CUDA 12.8 wheels are an intentional override of the upstream VGGT pin so
that Blackwell-class GPUs have a compatible wheel. Validate the actual kernels
and memory use on the target hardware; dependency resolution alone is not a
runtime proof.

## Setup

```bash
uv python install 3.10
uv lock --python 3.10
uv sync --python 3.10
```

Verify the interpreter and wheel identity without using the GPU:

```bash
.venv/bin/python - <<'PY'
import torch, torchvision
print(torch.__version__, torch.version.cuda, torchvision.__version__)
PY
```

Before any import in a bounded process, set thread caps:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
```

## Tests

`pytest==9.1.1` is tracked in the `dev` dependency group, so the lock file is
reproducible with the test runner:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
uv run --frozen pytest tests
```

CUDA is not required for the CPU test suite; set `CUDA_VISIBLE_DEVICES=` when
validating only the synthetic paths.

## Native VGGT source

VGGT is not vendored by VGGT-S. Clone the public upstream repository at the
pinned commit into the ignored `third_party/vggt_main` directory:

```bash
git clone https://github.com/facebookresearch/vggt.git third_party/vggt_main
git -C third_party/vggt_main checkout a288dd0f14786c93483e45524328726ab7b1b4ce
```

`third_party/` is git-ignored. The VGGT-S README also requires an interface
patch to `vggt/heads/track_head.py::forward` so that `feat=True` returns the
feature maps. A reproducible patch against the pinned commit is tracked at
`docs/patches/vggt-track-head-feat.patch` (SHA-256
`286dc3dd646aaadab192e98cc1c07e96629df31968115e4e689a1e741338fd83`). It is not
applied automatically. The generic adapter in `src/model/vggt_features.py`
does not require it: it calls the native
`track_head.feature_extractor` directly for the same feature maps. See
`docs/AUTOMATIC_SEMANTIC_ADAPTER.md`.

## Checkpoints

Checkpoints are never committed. Obtain them from the public Hugging Face
repositories at the exact commits below and place them at the paths expected by
the native code:

| File | Public repository | Revision | Bytes | SHA-256 | License | Expected path |
|---|---|---|---|---|---|---|
| `model.pt` | `facebook/VGGT-1B` | `860abec7937da0a4c03c41d3c269c366e82abdf9` | 5026874952 | `d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0` | CC-BY-NC-4.0 | `third_party/vggt_main/model.pt` |
| `main_exp.pth` | `zbbhhh/VGGT-S` | `200a31d21e6cc878b4428f225d03854c98d25c20` | 455801176 | `eb6f84d0a57868d3265d2cf8ce0f142e3526471a5dc282b8d12f068e8a162a4e` | Repository code Apache-2.0; checkpoint-specific license not separately stated | `official_ckpts/main_exp.pth` |

Both artifacts are public and non-gated. Any local symlink or copy of a
checkpoint is ignored by git.

## Checkpoint loading compatibility

`torch>=2.6` defaults `torch.load` to `weights_only=True`. Both pinned
checkpoints were loaded on CPU with that default and the exact revisions above:

- `model.pt` loads as a plain tensor state dictionary.
- `main_exp.pth` loads as a dictionary with `model`, `optimizer`, `scheduler`,
  and `epoch`; its tensors and primitive containers are accepted by the
  restricted unpickler.

No `weights_only=False` override and no source compatibility patch is needed.
If a future checkpoint really requires allowlisted globals, document the exact
exception and use the narrowest trusted-checkpoint handling rather than a
blanket override.
