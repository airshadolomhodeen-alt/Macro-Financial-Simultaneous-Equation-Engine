"""
Variable Mapping and Transformation Registry
"""

VARIABLE_REGISTRY = {
    "XAUUSD": {
        "name": "Gold Spot / US Dollar",
        "category": "Market Data",
        "default_transform": "LOG_DIFF",
        "description": "Spot price of gold per troy ounce in USD.",
        "role": "Endogenous (Gold Market)"
    },
    "DXY": {
        "name": "US Dollar Index",
        "category": "Market Data",
        "default_transform": "LOG_DIFF",
        "description": "Geometric weighted average of the foreign exchange value of the USD.",
        "role": "Endogenous / Regressor"
    },
    "USM2": {
        "name": "M2 Money Supply",
        "category": "Macroeconomic",
        "default_transform": "PERCENT_CHANGE",
        "description": "Total money supply including cash, checking deposits, and easily convertible money.",
        "role": "Exogenous / Control"
    },
    "FEDFUNDS": {
        "name": "Federal Funds Effective Rate",
        "category": "Macroeconomic",
        "default_transform": "LEVEL",
        "description": "Interest rate at which depository institutions trade federal funds.",
        "role": "Endogenous / Policy Rate"
    },
    "CPIAUCSL": {
        "name": "Consumer Price Index for All Urban Consumers",
        "category": "Macroeconomic",
        "default_transform": "PERCENT_CHANGE",
        "description": "Headline inflation measure tracking price changes in a basket of goods.",
        "role": "Exogenous / Control"
    },
    "GDPC1": {
        "name": "Real Gross Domestic Product",
        "category": "Macroeconomic",
        "default_transform": "LOG_DIFF",
        "description": "Inflation-adjusted value of goods and services produced.",
        "role": "Exogenous / Macro Control"
    },
    "RBUSBIS": {
        "name": "BIS Real Effective Exchange Rate / BIS Indices",
        "category": "Macroeconomic",
        "default_transform": "PERCENT_CHANGE",
        "description": "Bank for International Settlements effective exchange rate metrics.",
        "role": "Excluded Instrument"
    },
    "UNRATE": {
        "name": "Unemployment Rate",
        "category": "Macroeconomic",
        "default_transform": "LEVEL",
        "description": "Percentage of the total labor force that is unemployed.",
        "role": "Exogenous / Labor Market"
    },
    "PCEC96": {
        "name": "Real Personal Consumption Expenditures",
        "category": "Macroeconomic",
        "default_transform": "LOG_DIFF",
        "description": "Real spending by households on goods and services.",
        "role": "Excluded Instrument / Control"
    },
    "GCEC1": {
        "name": "Real Government Consumption Expenditures",
        "category": "Macroeconomic",
        "default_transform": "LOG_DIFF",
        "description": "Government consumption expenditures and gross investment.",
        "role": "Excluded Instrument"
    },
    "NETEXC": {
        "name": "Net Exports of Goods and Services",
        "category": "Macroeconomic",
        "default_transform": "LEVEL",
        "description": "Exports minus imports of goods and services.",
        "role": "Exogenous / Trade Balance"
    },
    "USINTR": {
        "name": "US Interest Rate / Treasury Yield Proxy",
        "category": "Market/Macro",
        "default_transform": "LEVEL",
        "description": "Short- to medium-term benchmark yield proxy.",
        "role": "Exogenous / Control"
    }
}
