import pandas as pd

from antarctic_precip_pmb.aggregation import annual_totals, month_to_season, seasonal_means


def test_month_to_season():
    assert month_to_season(1) == "DJF"
    assert month_to_season(4) == "MAM"
    assert month_to_season(7) == "JJA"
    assert month_to_season(10) == "SON"


def test_annual_totals_require_complete_years():
    df = pd.DataFrame(
        {"basin_2": [1.0] * 13},
        index=pd.date_range("2013-01-01", periods=13, freq="MS"),
    )
    out = annual_totals(df, require_complete_years=True)
    assert list(out.index) == [2013]
    assert out.loc[2013, "basin_2"] == 12.0


def test_seasonal_means_are_ordered():
    df = pd.DataFrame(
        {"basin_2": [1.0, 2.0, 3.0, 4.0]},
        index=pd.to_datetime(["2013-01-01", "2013-04-01", "2013-07-01", "2013-10-01"]),
    )
    out = seasonal_means(df)
    assert list(out.index.astype(str)) == ["DJF", "MAM", "JJA", "SON"]
