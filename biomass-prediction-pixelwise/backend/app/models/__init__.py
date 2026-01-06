"""
Models module initialization.
"""

# Make models importable
try:
    from .gee_data_fetcher import GEEDataFetcher
    from .ml_integration import predict_with_ml_model
    __all__ = ['GEEDataFetcher', 'predict_with_ml_model']
except ImportError as e:
    # Graceful fallback if dependencies aren't available
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"Could not import models: {e}")
    __all__ = []