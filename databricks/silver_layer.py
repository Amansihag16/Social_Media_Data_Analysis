"""
Silver Layer - Medallion Architecture (Alias for 02_silver_layer.py)
Cleans, standardizes, enriches with VADER NLP, and writes Silver tables.
"""

import importlib
_module = importlib.import_module("databricks.02_silver_layer")
SilverLayer = _module.SilverLayer

if __name__ == "__main__":
    silver = SilverLayer()
    silver.run_all()
