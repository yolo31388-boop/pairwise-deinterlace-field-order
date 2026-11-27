"""六条修复目标的行为回归测试（原契约测试之外的补充检查）。"""

from fractions import Fraction

import numpy as np

import media.deinterlace_field_order as mod


def test_field_order_aliases_and_metadata():
    assert mod.parse_field_order("bff") == "even"
    assert mod.parse_field_order("Lower Field First") == "bottom"
    assert mod.parse_field_order({"tff": 1}) == "odd"
    info = mod.parse_field_order({"field_order": "bb"})
    assert info["top_field_first"] is False
    assert info["array_known_row_parity"] == 1


def test_missing_marker_detects_actual_odd():
    clip = mod.make_demo()
    info = mod.parse_field_order({}, frames=clip["frames"])
    assert info["order"] == "odd"


def test_pair_places_odd_first_on_zero_even_rows():
    first = np.full((3, 2), 10.0, dtype=np.float32)
    second = np.full((3, 2), 20.0, dtype=np.float32)
    pair = mod.field_pair(first, second, "odd")
    assert pair["woven"][0, 0] == 10.0
    assert pair["woven"][1, 0] == 20.0
    result = mod.interpolate_field_pair(first, second, "odd")
    assert result["frame"].shape == (6, 2)


def test_ela_preserves_known_rows():
    frame = np.random.default_rng(0).random((16, 12)).astype(np.float32)
    for order in ("odd", "even"):
        parity = mod._known_row_parity(order)
        out = mod.edge_directed_interpolate(frame, order)
        np.testing.assert_array_equal(out[parity::2], frame[parity::2])
        assert np.isfinite(out).all()


def test_occlusion_threshold_and_array_blocks():
    assert mod.occlusion_fallback(0.9, 0.1)["fallback"] is True
    assert mod.occlusion_fallback(0.0, 0.2)["fallback"] is False
    demo = mod.make_demo()
    bad = mod.occlusion_fallback(
        demo["moving_block"], demo["occluded_candidate"], threshold=0.05
    )
    assert bad["fallback"] is True
    assert bad["block"] is not None


def test_clear_vectors_mutation_and_cut_score():
    vectors = [(1, 2), (3, 4)]
    result = mod.clear_vectors_on_cut(True, vectors)
    assert result["cleared"] is True
    assert vectors == []
    assert result == []
    kept = mod.clear_vectors_on_cut(False, [(1, 2)])
    assert len(kept["vectors"]) == 1
    assert mod.clear_vectors_on_cut(0.9, threshold=0.3)["cleared"] is True


def test_source_rate_preserved():
    assert mod.keep_source_rate(24, 60) == 24
    ntsc = mod.keep_source_rate(field_rate=59.94)
    assert ntsc["source_rate"] == Fraction(24000, 1001)
    assert mod.keep_source_rate(1)["source_rate"] == 24


def test_pipeline_review_export_determinism(tmp_path):
    report = mod.deinterlace(1)
    assert report["validation"]["ok"]
    assert report["cuts"] == [2]
    assert report["cut_resets"]
    assert report["source_rate"] == 24
    assert report["determinism"]["deterministic"]

    review_path = tmp_path / "review.json"
    review = mod.review_output(report, review_path)
    assert review["ok"] is True
    assert review_path.exists()

    export_path = tmp_path / "out.json"
    exported = mod.export(report, export_path)
    assert exported == str(export_path)
    assert export_path.exists() and export_path.stat().st_size > 0
