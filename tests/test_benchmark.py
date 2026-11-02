from achrom.benchmark import benchmark


def test_benchmark_returns_row_per_size():
    coeffs = [0.02, 0.0, 0.0]
    rows = benchmark(coeffs, coeffs, sizes=[(100, 100), (200, 200), (400, 300)])
    assert len(rows) == 3
    for w, h, ms in rows:
        assert ms > 0
        assert (w, h) in [(100, 100), (200, 200), (400, 300)]
