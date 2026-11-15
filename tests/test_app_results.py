import cv2
import numpy as np

import app


def test_results_keep_dots_in_file_names(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "RESULTS_DIR", tmp_path / "results")
    image = np.full((96, 128, 3), 128, np.uint8)
    first = tmp_path / "photo.v1.png"
    second = tmp_path / "photo.v2.png"
    cv2.imwrite(str(first), image)
    cv2.imwrite(str(second), image)

    code = app._run_cli([str(first), str(second)])

    assert code == 0
    assert (tmp_path / "results" / "photo.v1.png").exists()
    assert (tmp_path / "results" / "photo.v2.png").exists()
