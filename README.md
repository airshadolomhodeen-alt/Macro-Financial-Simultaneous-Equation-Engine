# Macro-Financial Simultaneous Equation Engine

An institutional-grade econometric research platform combining structural simultaneous-equation modeling (SEM), instrumental variables (2SLS/IV), weak-instrument diagnostics, HAC robust standard errors, and walk-forward out-of-sample directional forecasting for XAUUSD.

---

## 1. Project Overview
This repository implements a rigorous econometric framework to untangle the simultaneous interactions between gold spot prices (XAUUSD), the US Dollar Index (DXY), macroeconomic liquidity (USM2), monetary policy (FEDFUNDS), and inflation/output indicators.

Unlike standard black-box trading algorithms, this engine enforces structural identification, tests for endogeneity via the Durbin-Wu-Hausman test, validates instrument relevance with first-stage F-statistics, and strictly avoids look-ahead bias during time-series forecasting.

## 2. Architecture
