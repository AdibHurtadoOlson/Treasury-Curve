import numpy as np
import pandas as pd
import pytest
from fiworkbench.curve import nelson_siegel, fit_nelson_siegel
from fiworkbench.data import TENORS, validate_curve_data


def test_nelson_siegel_returns_finite_values():
    y = nelson_siegel(np.array([0.25, 2, 10, 30]), 0.04, -0.02, 0.01, 2.5)
    assert np.isfinite(y).all()
    with pytest.raises(ValueError): nelson_siegel(np.array([1]), 0.04, 0, 0, 0)


def test_fit_recovers_smooth_synthetic_curve_parameters_approximately():
    names = list(TENORS)
    t = np.array([TENORS[n] for n in names])
    y = nelson_siegel(t, 0.04, -0.02, 0.01, 2.5)
    row = pd.Series(y, index=names)
    params = fit_nelson_siegel(row)
    fitted = nelson_siegel(t, params["beta0"], params["beta1"], params["beta2"], params["tau"])
    assert np.sqrt(np.mean((fitted - y) ** 2)) < 0.002
    assert params["tau"] > 0


def test_fred_values_are_converted_from_percent_to_decimal():
    raw = pd.DataFrame({"observation_date": ["2024-01-02"] * 2,
                        "DGS1MO": [5.2, 5.2], "DGS3MO": [5.1, 5.1], "DGS6MO": [5.0, 5.0],
                        "DGS1": [4.8, 4.8], "DGS2": [4.3, 4.3], "DGS5": [4.0, 4.0],
                        "DGS10": [4.1, 4.1]})
    # duplicate date is a quality failure instead of a silent average.
    with pytest.raises(ValueError): validate_curve_data(raw)
    raw = raw.iloc[:1].copy()
    raw = pd.concat([raw.assign(observation_date=(pd.Timestamp("2024-01-02") + pd.Timedelta(days=i)).strftime("%Y-%m-%d")) for i in range(10)], ignore_index=True)
    result = validate_curve_data(raw)
    assert result.iloc[0]["DGS10"] == pytest.approx(0.041)
