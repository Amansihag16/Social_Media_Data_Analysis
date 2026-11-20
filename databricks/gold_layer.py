"""
Gold Layer - Medallion Architecture (Alias for 03_gold_layer.py)
Aggregates business metrics and generates analytical tables and CSV exports for Power BI.
"""

import importlib
_module = importlib.import_module("databricks.03_gold_layer")
GoldLayer = _module.GoldLayer

if __name__ == "__main__":
    gold = GoldLayer()
    gold.run_all()
