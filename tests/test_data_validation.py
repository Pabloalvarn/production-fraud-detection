from fraud_detection.data.validation import EXPECTED_COLUMNS, validate_dataframe


def test_valid_paysim_frame_passes(paysim_frame):
    report = validate_dataframe(paysim_frame)
    assert report["valid"] is True
    assert report["row_count"] == len(paysim_frame)
    assert 0 < report["fraud_prevalence"] < 1


def test_missing_column_is_reported(paysim_frame):
    frame = paysim_frame.drop(columns="amount")
    report = validate_dataframe(frame, raise_on_error=False)
    assert report["valid"] is False
    assert "amount" in report["missing_columns"]
    assert EXPECTED_COLUMNS - set(frame.columns) == {"amount"}
