"""
Unit Tests for 2SLS Estimation
"""
import pandas as pd
import numpy as np
from src.econometrics.iv_2sls import SimultaneousEquationEstimator

def test_2sls_estimation():
    np.random.seed(42)
    n = 100
    z = np.random.normal(0, 1, n)
    u = np.random.normal(0, 1, n)
    x = 0.8 * z + u
    y = 2.5 * x + np.random.normal(0, 1, n)
    
    df = pd.DataFrame({"y": y, "x": x, "z": z})
    estimator = SimultaneousEquationEstimator(df)
    res = estimator.estimate_2sls(dep_var="y", endogenous_vars=["x"], exogenous_vars=[], instruments=["z"])
    assert res is not None
    assert len(res.params) == 2
