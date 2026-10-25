import numpy as np
import torch

from model.rgbd_preprocess import (aspect_resize_pad_size, invert_logits,
                                   preprocess_view)


def _rectangular_rgbd(h=480, w=768):
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[..., 0] = np.arange(w, dtype=np.uint8)[None, :]
    rgb[..., 1] = np.arange(h, dtype=np.uint8)[:, None]
    depth = np.full((h, w), 0.0, dtype=np.float32)
    valid = np.zeros((h, w), dtype=np.float32)
    depth[:240, :384] = 2.5
    depth[0, 0] = 1.25
    valid[:240, :384] = 1.0
    valid[0, 0] = 1.0
    return rgb, depth, valid


def test_aspect_resize_pad_hand_oracle():
    resized_h, resized_w, pad_h, pad_w = aspect_resize_pad_size(480, 768, 518)
    assert (resized_h, resized_w) == (324, 518)
    assert (pad_h, pad_w) == (194, 0)
    assert abs(resized_h / 480 - 518 / 768) < 1e-3


def test_rectangular_depth_nearest_validity_padding():
    rgb, depth, valid = _rectangular_rgbd()
    view = preprocess_view(rgb, depth, valid, canvas_size=518)
    meta = view["meta"]
    assert view["image"].shape == (3, 518, 518)
    assert view["depth"].shape == (1, 518, 518)
    assert view["valid"].shape == (1, 518, 518)
    assert meta.resized_h == 324 and meta.resized_w == 518
    # Original top-left metric depth survives nearest resize.
    assert torch.isclose(view["depth"][0, 0, 0], torch.tensor(1.25))
    assert torch.isclose(view["valid"][0, 0, 0], torch.tensor(1.0))
    # Right padding is invalid and has no fabricated metric depth.
    assert torch.count_nonzero(view["valid"][:, meta.resized_h:, :]) == 0
    assert torch.count_nonzero(view["depth"][:, meta.resized_h:, :]) == 0
    # Missing original depth stays exactly zero; it is not blended from 2.5 m.
    assert float(view["depth"][0, 500, 100]) == 0.0
    assert float(view["valid"][0, 500, 100]) == 0.0


def test_inverse_crops_padding_and_returns_native_resolution():
    rgb, depth, valid = _rectangular_rgbd()
    view = preprocess_view(rgb, depth, valid, canvas_size=518)
    meta = view["meta"]
    logits = torch.zeros(1, 3, 518, 518)
    logits[:, :, :meta.resized_h, :] = 2.0
    polluted = logits.clone()
    polluted[:, :, meta.resized_h:, :] = 1e6
    inv = invert_logits(logits, meta)
    inv_polluted = invert_logits(polluted, meta)
    assert inv.shape == (1, 3, 480, 768)
    assert torch.allclose(inv, torch.ones_like(inv) * 2.0)
    assert torch.allclose(inv, inv_polluted)


def test_inverse_can_return_requested_rectangular_size():
    rgb, depth, valid = _rectangular_rgbd(37, 53)
    view = preprocess_view(rgb, depth, valid, canvas_size=64)
    logits = torch.zeros(2, 5, 64, 64)
    out = invert_logits(logits, view["meta"], output_size=(23, 31))
    assert out.shape == (2, 5, 23, 31)


def test_inverse_crops_upscaled_decoder_grid_before_resize():
    rgb, depth, valid = _rectangular_rgbd()
    view = preprocess_view(rgb, depth, valid, canvas_size=518)
    meta = view["meta"]
    grid = meta.canvas_size * 4  # native mask decoder upsampling grid
    logits = torch.zeros(1, 3, grid, grid)
    crop_h = int(round(meta.resized_h * grid / meta.canvas_size))
    crop_w = int(round(meta.resized_w * grid / meta.canvas_size))
    logits[:, :, :crop_h, :crop_w] = 3.0
    polluted = logits.clone()
    polluted[:, :, crop_h:, :] = 1e6
    polluted[:, :, :, crop_w:] = 1e6
    inv = invert_logits(logits, meta)
    inv_polluted = invert_logits(polluted, meta)
    assert inv.shape == (1, 3, 480, 768)
    assert torch.allclose(inv, torch.full_like(inv, 3.0))
    assert torch.allclose(inv, inv_polluted)


def test_rgb_uint8_and_explicit_float01_hand_oracle_equivalent():
    import pytest
    rgb, depth, valid = _rectangular_rgbd(37, 53)
    float01 = rgb.astype(np.float32) / 255.0
    from_u8 = preprocess_view(rgb, depth, valid, canvas_size=64)
    from_float = preprocess_view(float01, depth, valid, canvas_size=64)
    assert torch.allclose(from_u8["image"], from_float["image"], atol=0, rtol=0)
    assert from_u8["image"].min() >= 0.0 and from_u8["image"].max() <= 1.0


def test_rgb_ambiguous_float255_and_invalid_inputs_fail_closed():
    import pytest
    rgb, depth, valid = _rectangular_rgbd(37, 53)
    bad_inputs = [
        rgb.astype(np.float32) * 2.0,          # ambiguous float 0..255-like
        rgb.astype(np.float32) / 255.0 * 300.0, # > 1
        -rgb.astype(np.float32) / 255.0,        # < 0
        np.full((37, 53, 3), np.nan, np.float32),
        rgb.astype(np.int16),                   # non-uint8 integer
    ]
    for bad in bad_inputs:
        with pytest.raises((ValueError, TypeError)):
            preprocess_view(bad, depth, valid, canvas_size=64)
