import inspect
from pathlib import Path

import numpy as np
import pytest
import torch

from model.mask_decoder import MaskDecoder
from model.predictor import Predictor
from model.rgbd_preprocess import (align_depth_validity_tensor, preprocess_view)
from model.semantic_adapter import DepthValidityAdapter, SemanticAdapter
from model.transformer import TwoWayTransformer
from model.vggt_features import (build_feature_cache, extract_native_track_features,
                                 feature_cache_key, freeze_native_encoder,
                                 native_encoder_grads_all_none)


def make_adapter(num_classes=5, seed=0):
    return SemanticAdapter(
        num_classes=num_classes,
        img_size=64,
        sa_tokens=16,
        embed_dim=128,
        blocks=1,
        depth_hidden=8,
        seed=seed,
    )


def make_inputs(b=1, n=1, k=5, seed=0):
    gen = torch.Generator().manual_seed(seed)
    feat = torch.randn(b, n, 128, 8, 12, generator=gen)
    depth = torch.full((b, n, 16, 24), 1.2)
    depth[..., 0, :] = 0.0
    valid = torch.ones(b, n, 16, 24)
    valid[..., 0, :] = 0.0
    return feat, depth, valid


@pytest.mark.parametrize("num_classes", [3, 5])
def test_outputs_one_competition_logits_tensor(num_classes):
    adapter = make_adapter(num_classes).eval()
    feat, depth, valid = make_inputs(k=num_classes)
    with torch.no_grad():
        logits = adapter(feat, depth, valid, output_size=(23, 31))
        probs = torch.softmax(logits, dim=1)
    assert logits.shape == (1, num_classes, 23, 31)
    assert torch.isfinite(logits).all()
    assert torch.allclose(probs.sum(dim=1), torch.ones(1, 23, 31), atol=1e-5)
    best = logits.argmax(dim=1, keepdim=True)
    assert torch.allclose(probs.gather(1, best), probs.max(dim=1, keepdim=True).values,
                           atol=1e-6)


def test_class_competition_calibration_can_change_winner():
    adapter = make_adapter(3).eval()
    feat, depth, valid = make_inputs(k=3)
    with torch.no_grad():
        before = adapter(feat, depth, valid, output_size=(11, 13))
        adapter.class_bias[2] = 20.0
        after = adapter(feat, depth, valid, output_size=(11, 13))
    assert torch.equal(after.argmax(dim=1), torch.full((1, 11, 13), 2))
    assert not torch.equal(before, after)


def test_inference_api_is_gt_free():
    params = set(inspect.signature(SemanticAdapter.forward).parameters)
    assert not params.intersection({
        "src_points", "tar_points", "src_mask", "tar_mask", "crop", "gt",
        "class_prompt", "point_prompt"
    })
    adapter = make_adapter(5).eval()
    feat, depth, valid = make_inputs()
    with torch.no_grad():
        out = adapter(feat, depth, valid)
    assert torch.isfinite(out).all()


def test_depth_only_perturbation_changes_logits():
    adapter = make_adapter(5).eval()
    feat, depth, valid = make_inputs(seed=7)
    depth2 = depth.clone()
    depth2[depth2 > 0] = 0.55
    with torch.no_grad():
        a = adapter(feat, depth, valid, output_size=(11, 13))
        b = adapter(feat, depth2, valid, output_size=(11, 13))
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    assert not torch.allclose(a, b, atol=1e-6)


def test_invalid_depth_does_not_force_background_or_block_prediction():
    adapter = make_adapter(5).eval()
    feat_a, _, _ = make_inputs(seed=11)
    feat_b, _, _ = make_inputs(seed=12)
    zero_depth = torch.zeros(1, 1, 16, 24)
    zero_valid = torch.zeros(1, 1, 16, 24)
    with torch.no_grad():
        a = adapter(feat_a, zero_depth, zero_valid, output_size=(11, 13))
        b = adapter(feat_b, zero_depth, zero_valid, output_size=(11, 13))
    probs = torch.softmax(a, dim=1)
    assert torch.isfinite(a).all()
    assert float(probs[:, 0].max()) < 0.999  # background is not forced
    assert float(a.std()) > 1e-3           # not a constant class tensor
    assert not torch.allclose(a, b, atol=1e-6)  # RGB still drives prediction


def test_n1_fallback_equals_explicit_duplicate_self_slot():
    adapter = make_adapter(5).eval()
    feat, depth, valid = make_inputs(n=1, seed=21)
    feat2 = torch.cat([feat, feat], dim=1)
    depth2 = torch.cat([depth, depth], dim=1)
    valid2 = torch.cat([valid, valid], dim=1)
    with torch.no_grad():
        single = adapter(feat, depth, valid, output_size=(7, 9))
        duplicated = adapter(feat2, depth2, valid2, output_size=(7, 9))
    assert torch.allclose(single, duplicated, atol=1e-6)


def test_gradients_flow_to_depth_and_native_decoder_and_step_updates():
    adapter = make_adapter(5)
    adapter.train()
    feat, depth, valid = make_inputs(seed=31)
    depth.requires_grad_(True)
    logits = adapter(feat, depth, valid, output_size=(11, 13))
    loss = logits.square().mean()
    loss.backward()
    assert depth.grad is not None and torch.isfinite(depth.grad).all()
    assert float(depth.grad.abs().sum()) > 0
    used = [
        adapter.depth_adapter.conv_in.weight,
        adapter.class_queries.weight,
        adapter.class_bias,
        adapter.prompt_encoder.mask_downscaling[0].weight,
        adapter.prompt_encoder.ca_src_img_pe.weight,
        adapter.mask_decoder.transformer.layers[0].self_attn_token.q_proj.weight,
        adapter.mask_decoder.transformer.layers[0].cross_attn_token_to_image.q_proj.weight,
        adapter.mask_decoder.output_upscaling[0].weight,
        adapter.mask_decoder.output_hypernetworks_mlps[0].layers[0].weight,
    ]
    for param in used:
        assert param.grad is not None and torch.isfinite(param.grad).all()

    before = adapter.mask_decoder.transformer.layers[0].self_attn_token.q_proj.weight.detach().clone()
    opt = torch.optim.SGD([p for g in adapter.parameter_groups().values() for p in g],
                          lr=0.01)
    opt.step()
    after = adapter.mask_decoder.transformer.layers[0].self_attn_token.q_proj.weight
    assert not torch.equal(before, after.detach())


def test_parameter_groups_classify_native_and_new_modules():
    adapter = make_adapter(5)
    groups = adapter.parameter_groups()
    assert groups["native_prompt_encoder"]
    assert groups["native_mask_decoder"]
    assert groups["new_class_queries"]
    assert groups["new_calibration"]
    assert groups["new_depth_adapter"]


def test_deterministic_new_module_init():
    a = make_adapter(5, seed=3)
    b = make_adapter(5, seed=3)
    c = make_adapter(5, seed=4)
    assert torch.equal(a.class_queries.weight, b.class_queries.weight)
    assert torch.equal(a.depth_adapter.conv_in.weight,
                       b.depth_adapter.conv_in.weight)
    assert not torch.equal(a.class_queries.weight, c.class_queries.weight)


def test_checkpoint_load_report_explicit_new_and_missing(tmp_path):
    source = make_adapter(3, seed=5)
    native = {
        k: v for k, v in source.state_dict().items()
        if k.startswith(("prompt_encoder.", "mask_decoder."))
    }
    ckpt = tmp_path / "native.pth"
    torch.save({"model": native}, ckpt)

    target = make_adapter(3, seed=6)
    report = target.load_native_checkpoint(str(ckpt), strict_native=True)
    assert report["torch_load_weights_only"] is True
    assert report["missing_native"] == []
    assert report["unexpected_keys"] == []
    assert set(report["new_keys"]) == {
        "class_queries.weight", "class_bias", "logit_scale",
        "depth_adapter.conv_in.weight", "depth_adapter.conv_in.bias",
        "depth_adapter.conv_out.weight", "depth_adapter.conv_out.bias",
    }
    assert "class_queries.weight" in report["new_keys"]
    for key, value in native.items():
        assert torch.equal(target.state_dict()[key], value)


def test_native_mask_decoder_default_path_unchanged():
    adapter = make_adapter(5).eval()
    feat, depth, valid = make_inputs(n=2, seed=41)
    with torch.no_grad():
        fused = feat + adapter.depth_adapter(depth, valid, (8, 12))
        ca_pe = adapter._dense_ca_pe(8, 12)
        sa_pe = adapter._dense_sa_pe(8, 12)
        zero_mask = torch.zeros(1, 1, 16, 24)
        src_dense = adapter.prompt_encoder.mask_downscaling(zero_mask)
        empty = torch.zeros(1, 0, 128)
        native_tokens = adapter.mask_decoder.mask_tokens.weight.unsqueeze(0)
        default_out = adapter.mask_decoder(fused, ca_pe, sa_pe, empty, src_dense,
                                           None)
        explicit_out = adapter.mask_decoder(fused, ca_pe, sa_pe, empty, src_dense,
                                            None, mask_token_b=native_tokens)
    assert default_out.shape[1] == 1
    assert torch.allclose(default_out, explicit_out)


def test_frozen_encoder_extraction_has_no_grads_and_actual_n():
    class FakeAggregator(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))

        def forward(self, images):
            return [self.anchor], 0

    class FakeFeatureExtractor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 4, 1)
            self.seen_n = None

        def forward(self, aggregated, images, patch_start_idx):
            b, n, c, h, w = images.shape
            self.seen_n = n
            out = self.conv(images.reshape(b * n, c, h, w))
            return out.reshape(b, n, 4, h, w)

    class FakeTrackHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.feature_extractor = FakeFeatureExtractor()

    class FakeVGGT(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.aggregator = FakeAggregator()
            self.track_head = FakeTrackHead()

    for n in (1, 2):
        vggt = FakeVGGT()
        counts = freeze_native_encoder(vggt)
        assert counts["aggregator"] == 1 and counts["track_head"] == 16
        images = torch.rand(1, n, 3, 8, 10)
        feats = extract_native_track_features(vggt, images)
        assert feats.shape == (1, n, 4, 8, 10)
        assert feats.requires_grad is False
        assert vggt.track_head.feature_extractor.seen_n == n
        assert native_encoder_grads_all_none(vggt)


def test_cache_key_n1_n2_and_view_order_isolation():
    base = dict(
        encoder_source_sha="source",
        encoder_weights_sha="weights",
        rgb_shas=["rgb0", "rgb1"],
        depth_shas=["d0", "d1"],
        validity_shas=["v0", "v1"],
        storage_dtype="float32",
        compute_dtype="float32",
        encoder_device_type="cpu",
        preprocess=[
            dict(orig_h=10, orig_w=20, resized_h=5, resized_w=10, pad_h=0,
                 pad_w=0, canvas_size=10, scale_x=0.5, scale_y=0.5),
            dict(orig_h=10, orig_w=20, resized_h=5, resized_w=10, pad_h=0,
                 pad_w=0, canvas_size=10, scale_x=0.5, scale_y=0.5),
        ],
    )
    k2 = feature_cache_key(n_views=2, view_ids=["a", "b"], **base)
    k1 = feature_cache_key(n_views=1, view_ids=["a"], rgb_shas=["rgb0"],
                           depth_shas=["d0"], validity_shas=["v0"],
                           preprocess=base["preprocess"][:1],
                           **{k: v for k, v in base.items()
                              if k not in ("rgb_shas", "depth_shas",
                                           "validity_shas", "preprocess")})
    k2_swapped = feature_cache_key(n_views=2, view_ids=["b", "a"], **base)
    k2_repeat = feature_cache_key(n_views=2, view_ids=["a", "b"], **base)
    assert len({k1, k2, k2_swapped}) == 3
    assert k2 == k2_repeat

    k_validity = feature_cache_key(
        n_views=2, view_ids=["a", "b"],
        **{**base, "validity_shas": ["different", "v1"]})
    k_compute = feature_cache_key(
        n_views=2, view_ids=["a", "b"],
        **{**base, "compute_dtype": "bfloat16"})
    k_storage = feature_cache_key(
        n_views=2, view_ids=["a", "b"],
        **{**base, "storage_dtype": "bfloat16"})
    assert len({k2, k_validity, k_compute, k_storage}) == 4


def test_build_feature_cache_real_n_and_depth_padding():
    class FakeAggregator(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))

        def forward(self, images):
            return [self.anchor], 0

    class FakeFeatureExtractor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 4, 1)
            self.calls = []
            self.devices = []

        def forward(self, aggregated, images, patch_start_idx):
            b, n, c, h, w = images.shape
            self.calls.append(n)
            self.devices.append(images.device)
            return self.conv(images.reshape(b * n, c, h, w)).reshape(
                b, n, 4, h, w)

    class FakeVGGT(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.aggregator = FakeAggregator()
            self.track_head = torch.nn.Module()
            self.track_head.feature_extractor = FakeFeatureExtractor()

    rgb = np.zeros((37, 53, 3), dtype=np.uint8)
    depth = np.full((37, 53), 1.5, dtype=np.float32)
    valid = np.ones((37, 53), dtype=np.float32)
    vggt = FakeVGGT()
    cache1 = build_feature_cache(
        vggt, [rgb], [depth], [valid], encoder_source_sha="s",
        encoder_weights_sha="w", canvas_size=32)
    cache2 = build_feature_cache(
        vggt, [rgb, rgb], [depth, depth], [valid, valid],
        encoder_source_sha="s", encoder_weights_sha="w", canvas_size=32,
        storage_dtype="float32", compute_dtype="bfloat16")
    assert cache1["n_views"] == 1 and cache1["singleview_fallback"] is True
    assert cache2["n_views"] == 2 and cache2["singleview_fallback"] is False
    assert vggt.track_head.feature_extractor.calls == [1, 2]
    assert all(d.type == "cpu" for d in
               vggt.track_head.feature_extractor.devices)
    assert cache1["cache_key"] != cache2["cache_key"]
    assert cache2["compute_dtype"] == "bfloat16"
    assert cache2["storage_dtype"] == "float32"
    assert cache1["encoder_device_type"] == "cpu"
    assert len(cache1["validity_shas"]) == 1
    assert cache2["feat_map"].dtype == torch.float32
    assert cache1["aligned_depth"].shape[:2] == (1, 1)
    assert cache1["feat_map"].shape[:2] == (1, 1)


def test_actual_public_checkpoint_predictor_load_cpu():
    path = Path(__file__).resolve().parents[1] / "official_ckpts" / "main_exp.pth"
    if not path.exists():
        pytest.skip("public VGGT-S checkpoint is not present")
    raw = torch.load(str(path), map_location="cpu", weights_only=True)
    assert isinstance(raw, dict) and "model" in raw
    predictor = Predictor(resume_ckpt=str(path))
    assert isinstance(predictor, Predictor)

    adapter = SemanticAdapter(num_classes=5, seed=0)
    report = adapter.load_native_checkpoint(str(path), strict_native=True)
    assert report["missing_native"] == []
    assert report["unexpected_keys"] == []
    assert report["native_loaded_count"] == report["native_expected_count"]
    native_state = {k[len("module."):] if k.startswith("module.") else k: v
                    for k, v in raw["model"].items()}
    for key in (
            "prompt_encoder.ca_src_img_pe.weight",
            "prompt_encoder.mask_downscaling.0.weight",
            "mask_decoder.mask_tokens.weight",
            "mask_decoder.transformer.layers.0.self_attn_token.q_proj.weight",
            "mask_decoder.output_hypernetworks_mlps.0.layers.0.weight",
    ):
        assert torch.equal(adapter.state_dict()[key], native_state[key])
    assert report["new_keys"]
    assert all(k.startswith(("class_queries.", "class_bias", "logit_scale",
                             "depth_adapter.")) for k in report["new_keys"])


def test_checkpoint_unexpected_key_reported_before_filtering(tmp_path):
    source = make_adapter(3, seed=5)
    state = {k: v.clone() for k, v in source.state_dict().items()}
    state["retired.old_head.weight"] = torch.zeros(1)
    ckpt = tmp_path / "native-with-extra.pth"
    torch.save({"model": state}, ckpt)
    target = make_adapter(3, seed=6)
    report = target.load_native_checkpoint(str(ckpt), strict_native=True)
    assert "retired.old_head.weight" in report["unexpected_keys"]
    assert report["missing_native"] == []


def test_raw_depth_metadata_alignment_matches_preprocessed_canvas():
    rgb = np.zeros((37, 53, 3), dtype=np.uint8)
    depth = np.zeros((37, 53), dtype=np.float32)
    valid = np.zeros((37, 53), dtype=np.float32)
    depth[5, 7] = 2.0
    valid[5, 7] = 1.0
    depth[30, 40] = 1.0  # invalid nonzero must not become metric depth
    view = preprocess_view(rgb, depth, valid, canvas_size=32)
    meta = view["meta"]
    assert meta.resized_h == 22 and meta.resized_w == 32 and meta.pad_h == 10

    canvas_depth, canvas_valid = align_depth_validity_tensor(
        torch.from_numpy(depth), torch.from_numpy(valid), meta)
    assert canvas_depth.shape == (32, 32) and canvas_valid.shape == (32, 32)
    assert float(canvas_valid[meta.resized_h:, :].sum()) == 0.0
    assert float(canvas_depth[meta.resized_h:, :].abs().sum()) == 0.0
    mapped_invalid = (int(30 * meta.resized_h / 37), int(40 * meta.resized_w / 53))
    assert float(canvas_depth[mapped_invalid]) == 0.0
    valid_count = float(canvas_valid.sum())
    assert float(canvas_depth.sum()) == pytest.approx(2.0 * valid_count)

    adapter = make_adapter(5).eval()
    feat = torch.randn(1, 1, 128, 8, 12, generator=torch.Generator().manual_seed(51))
    raw_depth = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)
    raw_valid = torch.from_numpy(valid).unsqueeze(0).unsqueeze(0)
    aligned_depth = view["depth"].unsqueeze(0)
    aligned_valid = view["valid"].unsqueeze(0)
    with torch.no_grad():
        from_raw = adapter(feat, raw_depth, raw_valid, output_size=(23, 31),
                           preprocess_meta=meta)
        from_canvas = adapter(feat, aligned_depth, aligned_valid,
                              output_size=(23, 31), preprocess_meta=meta)
    assert from_raw.shape == (1, 5, 23, 31)
    assert torch.allclose(from_raw, from_canvas, atol=1e-6)
    with pytest.raises(ValueError):
        adapter(feat, torch.zeros(1, 1, 10, 10), torch.zeros(1, 1, 10, 10),
                preprocess_meta=meta)
