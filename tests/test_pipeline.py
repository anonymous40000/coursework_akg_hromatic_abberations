import numpy as np

from achrom.pipeline import process_frame
from achrom.synthetic import make_dot_frame


def test_pipeline_reports_geometry_before_and_after_cleanup_separately():
    source, _ = make_dot_frame(
        shape=(500, 500),
        k_blue=0.025,
        k_red=-0.02,
    )
    final, report, stages = process_frame(source, return_stages=True)

    assert final.shape == source.shape
    assert stages["source"].shape == source.shape
    assert stages["geometry"].shape == source.shape
    assert report["residual_after_geometry"]["b"] < report["residual_before_geometry"]["b"]
    assert report["residual_after_geometry"]["r"] < report["residual_before_geometry"]["r"]
    assert "models" not in report["geometry"]
    assert report["defringe"]["method"] == "directed-fg-bg-evid"


def test_pipeline_can_disable_colour_cleanup():
    source, _ = make_dot_frame(
        shape=(500, 500),
        k_blue=0.02,
        k_red=-0.02,
    )
    final, report, stages = process_frame(source, enable_defringe=False, return_stages=True)

    assert report["defringe"]["method"] == "disabled"
    assert np.array_equal(final, stages["geometry"])
    assert report["residual_after_geometry"]["b"] < report["residual_before_geometry"]["b"]
