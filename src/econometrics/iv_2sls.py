"""
Production IV / 2SLS Econometric Engine
"""
import pandas as pd
import statsmodels.api as sm
from statsmodels.sandbox.regression.gmm import IV2SLS

class SimultaneousEquationEstimator:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def estimate_ols(self, dep_var: str, exog_vars: list[str]):
        Y = self.data[dep_var]
        X = sm.add_constant(self.data[exog_vars])
        return sm.OLS(Y, X).fit()

    def estimate_2sls(self, dep_var: str, endogenous_vars: list[str], exogenous_vars: list[str], instruments: list[str]):
        Y = self.data[dep_var]
        regressors = endogenous_vars + exogenous_vars
        all_instruments = exogenous_vars + instruments
        X = sm.add_constant(self.data[regressors])
        Z = sm.add_constant(self.data[all_instruments])
        return IV2SLS(Y, X, instrument=Z).fit()

    def first_stage_diagnostics(self, endogenous_var: str, exogenous_vars: list[str], instruments: list[str]) -> dict:
        Z = sm.add_constant(self.data[exogenous_vars + instruments])
        first_stage = sm.OLS(self.data[endogenous_var], Z).fit()
        excl_str = " = 0, ".join(instruments) + " = 0"
        f_test = first_stage.f_test(excl_str)
        return {
            "r_squared": first_stage.rsquared,
            "f_statistic": float(f_test.fvalue),
            "p_value": float(f_test.pvalue)
        }
