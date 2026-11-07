import pandas as pd

from ridership import data


def test_load_handles_commas_dates_and_missing(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("Date,Mode,Count\n11/05/2026,Subway,\"3,100,000\"\n11/05/2026,Bus,\n11/04/2026,Subway,\"3,000,000\"\n11/04/2026,Subway,\"3,000,000\"\n")
    d = data.load_raw(p)
    assert len(d) == 3                                        # exact duplicate row dropped
    s = d[(d["mode"] == "Subway") & (d["date"] == "2026-11-05")]["count"].iloc[0]
    assert s == 3_100_000 and d[d["mode"] == "Bus"]["count"].isna().all()


def test_wide_is_a_complete_daily_grid_and_tail_is_trimmed():
    long = pd.DataFrame({"date": pd.to_datetime(["2026-01-01", "2026-01-03", "2026-01-04", "2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"]),
                         "mode": ["A", "A", "A", "B", "B", "B", "B"], "count": [1.0, 3.0, None, 10.0, 20.0, 30.0, 40.0]})
    wide = data.to_wide(long)
    assert len(wide) == 4 and wide["A"].isna().sum() == 2    # Jan 2 inside the range, Jan 4 NaN
    a = data.get_series(wide, "A")
    assert a.index[-1] == pd.Timestamp("2026-01-03")           # trailing NaN removed, interior NaN kept
    assert data.get_series(wide, "B").index[-1] == pd.Timestamp("2026-01-04")


def test_api_records_parse_like_the_csv():
    recs = [{"date": "2026-11-05T00:00:00.000", "mode": "Subway", "count": "3100000"}, {"date": "2026-11-05T00:00:00.000", "mode": "Bus", "count": "1,300,000"}]
    d = data.parse_api_records(recs)
    assert d["date"].iloc[0] == pd.Timestamp("2026-11-05") and set(d["mode"]) == {"Subway", "Bus"}
    assert d[d["mode"] == "Bus"]["count"].iloc[0] == 1_300_000


def test_merge_update_prefers_new_values_and_adds_new_days():
    old = pd.DataFrame({"date": pd.to_datetime(["2026-01-01", "2026-01-02"]), "mode": ["A", "A"], "count": [1.0, None]})
    new = pd.DataFrame({"date": pd.to_datetime(["2026-01-02", "2026-01-03"]), "mode": ["A", "A"], "count": [2.0, 3.0]})
    m = data.merge_update(old, new)
    assert m["count"].tolist() == [1.0, 2.0, 3.0]


def test_save_and_reload_roundtrip(tmp_path):
    d = pd.DataFrame({"date": pd.to_datetime(["2026-01-01", "2026-01-02"]), "mode": ["A", "A"], "count": [1234567.0, float("nan")]})
    data.save_raw(d, tmp_path / "r.csv")
    back = data.load_raw(tmp_path / "r.csv")
    assert back["count"].iloc[0] == 1234567 and back["count"].isna().iloc[1]
