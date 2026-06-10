import pandas as pd

from antarctic_precip_pmb.grace import decimal_year_to_month_start, forward_delta_s


def test_decimal_year_to_month_start_nearest():
    assert decimal_year_to_month_start(2013.0) == pd.Timestamp("2013-01-01")


def test_forward_delta_s_assigned_to_starting_month():
    storage = pd.DataFrame(
        {"A-Ap": [10.0, 13.0, 12.0]},
        index=pd.to_datetime(["2013-01-01", "2013-02-01", "2013-03-01"]),
    )
    out = forward_delta_s(storage)
    assert list(out.index) == [pd.Timestamp("2013-01-01"), pd.Timestamp("2013-02-01")]
    assert out.loc["2013-01-01", "A-Ap"] == 3.0
    assert out.loc["2013-02-01", "A-Ap"] == -1.0
