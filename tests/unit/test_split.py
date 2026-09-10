from fraud_detection.data.split import split_by_time


def test_temporal_split_has_no_overlapping_steps(paysim_frame):
    result = split_by_time(paysim_frame)
    assert result.train["step"].max() < result.validation["step"].min()
    assert result.validation["step"].max() < result.test["step"].min()
