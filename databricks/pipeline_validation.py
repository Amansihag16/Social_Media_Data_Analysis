"""
Pipeline Validation - Medallion Architecture (Alias for 04_pipeline_validation.py)
Performs integrity checks and schema validations across all layers.
"""

import importlib
_module = importlib.import_module("databricks.04_pipeline_validation")
PipelineValidator = _module.PipelineValidator

if __name__ == "__main__":
    validator = PipelineValidator()
    validator.run_audit()
