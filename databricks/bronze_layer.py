"""
Bronze Layer - Medallion Architecture (Alias for 01_bronze_layer.py)
Ingests raw multi-platform feeds into Bronze Delta storage.
"""

import importlib
_module = importlib.import_module("databricks.01_bronze_layer")
BronzeLayer = _module.BronzeLayer

if __name__ == "__main__":
    bronze = BronzeLayer()
    bronze.run_all()
