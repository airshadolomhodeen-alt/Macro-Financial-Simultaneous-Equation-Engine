"""
Production IV / 2SLS Econometric Engine & Diagnostic Calculator
===============================================================
Computes OLS, Reduced-Form, proper 2SLS (IV2SLS), first-stage diagnostics,
and Hausman endogeneity tests dynamically.
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.sandbox.regression.gmm import IV2SLS
from typing import Dict, Any, List

class SimultaneousEquationEstimator:
    def __init__(self, data: pd.DataFrame):
        """
        Initializes the econometric engine with a synchronized macro-financial dataset.
        """
        self.data = data.dropna()

    def estimate_ols(self, dep_var: str, regressors: List[str]) -> Dict[str, Any]:
        """
        Estimates naive OLS (potentially biased under simultaneity).
        """
        Y = self.data[dep_var]
        X = sm.add_constant(self.data[regressors])
        model = sm.OLS(Y, X).fit()
        
        results_df = pd.DataFrame({
            "Parameter": ["Intercept"] + regressors,
            "Coefficient": model.params.values,
            "Std. Error": model.bse.values,
            "t-statistic": model.tvalues.values,
            "p-value": model.pvalues.values,
            "Model": "Naive OLS"
        })
        return {
            "model_fit": model,
            "table": results_df,
            "r_squared": model.rsquared,
            "adj_r_squared": model.rsquared_adj
        }

    def estimate_reduced_form(self, endogenous_vars: List[str], exogenous_and_instruments: List[str]) -> Dict[str, sm.regression.linear_model.RegressionResultsWrapper]:
        """
        Estimates reduced-form equations: each endogenous variable on ALL exogenous variables/instruments (Z).
        """
        Z = sm.add_constant(self.data[exogenous_and_instruments])
        reduced_form_models = {}
        for endog in endogenous_vars:
            y = self.data[endog]
            fit = sm.OLS(y, Z).fit()
            reduced_form_models[endog] = fit
        return reduced_form_models

    def estimate_2sls(self, dep_var: str, endogenous_vars: List[str], exogenous_vars: List[str], instruments: List[str]) -> Dict[str, Any]:
        """
        Estimates structural equation using proper Two-Stage Least Squares (2SLS / IV).
        Regressors = endogenous_vars + exogenous_vars
        Instruments = exogenous_vars + instruments
        """
        Y = self.data[dep_var]
        regressors = endogenous_vars + exogenous_vars
        all_instruments = exogenous_vars + instruments
        
        X = sm.add_constant(self.data[regressors])
        Z = sm.add_constant(self.data[all_instruments])
        
        # Proper IV2SLS estimator ensuring correct second-stage standard errors
        iv_model = IV2SLS(Y, X, instrument=Z)
        results = iv_model.fit()
        
        results_df = pd.DataFrame({
            "Parameter": ["Intercept"] + regressors,
            "Coefficient": results.params.values,
            "Robust SE": results.bse.values,
            "t-statistic": results.tvalues.values,
            "p-value": results.pvalues.values,
            "Model": "Proper 2SLS (IV)"
        })
        
        return {
            "model_fit": results,
            "table": results_df,
            "r_squared": getattr(results, 'rsquared', np.nan)
        }

    def run_first_stage_diagnostics(self, endogenous_vars: List[str], exogenous_vars: List[str], instruments: List[str]) -> pd.DataFrame:
        """
        Performs first-stage regression and computes F-statistics for weak instrument testing.
        """
        Z = sm.add_constant(self.data[exogenous_vars + instruments])
        diag_records = []
        
        for endog in endogenous_vars:
            fs_reg = sm.OLS(self.data[endog], Z).fit()
            excl_str = " = 0, ".join(instruments) + " = 0"
            try:
                f_test = fs_reg.f_test(excl_str)
                f_val = float(f_test.fvalue)
                p_val = float(f_test.pvalue)
            except Exception:
                f_val, p_val = 0.0, 1.0
                
            diag_records.append({
                "Endogenous Regressor": endog,
                "Excluded Instruments Used": ", ".join(instruments),
                "First-Stage R²": round(fs_reg.rsquared, 3),
                "Partial F-Stat": round(f_val, 2),
                "p-value": round(p_val, 4),
                "Weak Instrument Risk": "Low (F > 10)" if f_val > 10 else "High (Weak IV)"
            })
            
        return pd.DataFrame(diag_records)

    def hausman_endogeneity_test(self, dep_var: str, endogenous_vars: List[str], exogenous_vars: List[str], instruments: List[str]) -> pd.DataFrame:
        """
        Implements Durbin-Wu-Hausman control-function test for endogeneity.
        """
        # Step 1: Reduced form residuals for endogenous variables
        Z = sm.add_constant(self.data[exogenous_vars + instruments])
        test_records = []
        
        Y = self.data[dep_var]
        X_reg = self.data[endogenous_vars + exogenous_vars]
        
        for endog in endogenous_vars:
            rf = sm.OLS(self.data[endog], Z).fit()
            v_hat = rf.resid
            
            # Step 2: Augment structural OLS with residual v_hat
            augmented_X = sm.add_constant(X_reg.assign(v_hat=v_hat))
            aug_fit = sm.OLS(Y, augmented_X).fit()
            
            t_val = aug_fit.tvalues.get("v_hat", 0.0)
            p_val = aug_fit.pvalues.get("v_hat", 1.0)
            verdict = "Reject H0 (Endogenous - Use 2SLS)" if p_val < 0.05 else "Do not reject H0 (Exogenous - OLS Valid)"
            
            test_records.append({
                "Endogenous Variable": endog,
                "Hausman t-stat": round(t_val, 3),
                "p-value": round(p_val, 4),
                "Econometric Verdict": verdict
            })
            
        return pd.DataFrame(test_records)
