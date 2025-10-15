"""
Configuration settings for the biomass prediction application.
"""
import os
from typing import Dict, Any, List
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings."""
    
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 1
    
    # Model Configuration
    MODEL_PATH: str = "./models"
    DATA_PATH: str = "./data"
    OUTPUT_PATH: str = "./outputs"
    PROCESSED_DATA_PATH: str = "./data/processed"
    
    # External APIs
    EARTHDATA_USERNAME: str = ""
    EARTHDATA_PASSWORD: str = ""
    GLM_API_KEY: str = ""
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    # CORS Configuration
    CORS_ORIGINS: List[str] = ["*"]
    
    # Supported Cities with bounding boxes [min_lon, min_lat, max_lon, max_lat]
    SUPPORTED_CITIES: Dict[str, List[float]] = {
        "Mumbai": [72.7, 18.9, 73.0, 19.2],
        "Delhi": [77.0, 28.4, 77.3, 28.7],
        "Bangalore": [77.4, 12.8, 77.8, 13.1],
        "Chennai": [80.1, 12.8, 80.3, 13.2],
        "Kolkata": [88.2, 22.4, 88.4, 22.7],
        "Hyderabad": [78.3, 17.2, 78.6, 17.5]
    }
    
    # Model Information
    MODEL_INFO: Dict[str, Any] = {
        "version": "1.0.0",
        "type": "CNN-LSTM",
        "input_shape": [12, 64, 64, 13],  # [time_steps, height, width, features]
        "output_shape": [64, 64, 1],  # [height, width, biomass]
        "metrics": {
            "r2": 0.85,
            "rmse": 12.5,
            "mae": 8.3
        },
        "trained_on": "2023-01-01 to 2023-12-31",
        "description": "CNN-LSTM model for predicting above-ground biomass using satellite data"
    }
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Create a global settings instance
settings = Settings()

def get_settings() -> Settings:
    """Get the current settings instance."""
    return settings

def get_model_path() -> str:
    """Get the path to the model file with fallbacks."""
    model_dir = settings.MODEL_PATH
    
    # Try different model paths in order of preference
    model_paths = [
        os.path.join(model_dir, "biomass_cnn_lstm_quantized.tflite"),
        os.path.join(model_dir, "biomass_cnn_lstm_best.h5"),
        os.path.join(model_dir, "biomass_cnn_lstm.h5")
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            return path
    
    # If no model is found, return the default path
    return model_paths[0]

def get_scalers_path() -> str:
    """Get the path to the scalers file."""
    return os.path.join(settings.PROCESSED_DATA_PATH, "scalers.pkl")

def get_coordinates_paths() -> List[str]:
    """Get the paths to the coordinates files."""
    return [
        os.path.join(settings.PROCESSED_DATA_PATH, "common_lons.npy"),
        os.path.join(settings.PROCESSED_DATA_PATH, "common_lats.npy")
    ]