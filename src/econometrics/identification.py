"""
Structural Identification Matrix & Order Condition Verification
"""
import pandas as pd

class IdentificationEngine:
    def __init__(self, equations_spec: dict):
        self.spec = equations_spec

    def verify_order_condition(self, equation_name: str, total_endogenous_count: int, total_excluded_instruments: int) -> dict:
        """
        Order condition for identification (necessary condition):
        K - k >= M - 1
        Where K - k is the number of excluded exogenous instruments,
        and M - 1 is the number of included endogenous variables on the RHS minus 1.
        """
        eq = self.spec.get(equation_name, {})
        endog_rhs = len(eq.get("endogenous", []))
        excluded_inst = len(eq.get("instruments", []))
        
        is_identified = excluded_inst >= endog_rhs
        status = "Just-identified" if excluded_inst == endog_rhs else ("Over-identified" if excluded_inst > endog_rhs else "Under-identified")
        
        return {
            "equation": equation_name,
            "excluded_instruments": excluded_inst,
            "endogenous_rhs": endog_rhs,
            "identification_status": status,
            "satisfies_order_condition": is_identified
        }
