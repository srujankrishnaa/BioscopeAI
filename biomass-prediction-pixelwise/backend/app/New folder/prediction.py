"""
Prediction API endpoints.
"""
import os
import logging
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File
from pydantic import BaseModel

# Import model components
from app.models.predictor import BiomassPredictor

logger = logging.getLogger(__name__)

router = APIRouter()

# Pydantic models for request/response
class PredictionRequest(BaseModel):
    """Model for prediction request."""
    city_name: str
    city_bbox: List[float] = None  # [min_lon, min_lat, max_lon, max_lat]
    start_date: str = "2023-01-01"
    end_date: str = "2023-12-31"

class PredictionResponse(BaseModel):
    """Model for prediction response."""
    city_name: str
    city_bbox: List[float]
    heatmap_path: str
    interactive_map_path: str
    report_text: str
    pdf_report_path: str
    statistics: Dict[str, Any]

# Global predictor instance
predictor = None

def get_predictor():
    """Get or initialize the predictor instance."""
    global predictor
    if predictor is None:
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                 "models", "biomass_cnn_lstm_quantized.tflite")
        scalers_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                   "data", "processed", "scalers.pkl")
        common_coords_path = [
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                       "data", "processed", "common_lons.npy"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                       "data", "processed", "common_lats.npy")
        ]
        
        # Check if model exists
        if not os.path.exists(model_path):
            # Try fallback paths
            fallback_paths = [
                model_path.replace('.tflite', '.h5'),
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                            "models", "biomass_cnn_lstm_best.h5"),
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                            "models", "biomass_cnn_lstm.h5")
            ]
            
            for path in fallback_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            else:
                raise HTTPException(status_code=500, detail="Model not found")
        
        # Initialize predictor
        try:
            predictor = BiomassPredictor(
                model_path=model_path,
                scalers_path=scalers_path,
                common_coords_path=common_coords_path
            )
            logger.info("Predictor initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing predictor: {e}")
            raise HTTPException(status_code=500, detail=f"Error initializing predictor: {str(e)}")
    
    return predictor

@router.post("/predict", response_model=PredictionResponse)
async def predict_biomass(request: PredictionRequest, background_tasks: BackgroundTasks):
    """
    Predict biomass for a given city.
    
    Args:
        request: Prediction request containing city name and optional parameters
        background_tasks: FastAPI background tasks
        
    Returns:
        PredictionResponse: Prediction results including paths to generated outputs
    """
    try:
        predictor = get_predictor()
        
        # Make prediction
        result = predictor.predict_for_city(
            city_name=request.city_name,
            city_bbox=request.city_bbox,
            start_date=request.start_date,
            end_date=request.end_date
        )
        
        if 'error' in result:
            raise HTTPException(status_code=500, detail=result['error'])
        
        return PredictionResponse(
            city_name=result['city_name'],
            city_bbox=result['city_bbox'],
            heatmap_path=result['heatmap_path'],
            interactive_map_path=result['interactive_map_path'],
            report_text=result['report_text'],
            pdf_report_path=result['pdf_report_path'],
            statistics=result['statistics']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Error in prediction: {str(e)}")

@router.get("/cities")
async def get_supported_cities():
    """
    Get list of supported cities with their bounding boxes.
    
    Returns:
        Dict: Dictionary of city names and their bounding boxes
    """
    try:
        # Default cities with bounding boxes [min_lon, min_lat, max_lon, max_lat]
        cities = {
            "Mumbai": [72.7, 18.9, 73.0, 19.2],
            "Delhi": [77.0, 28.4, 77.3, 28.7],
            "Bangalore": [77.4, 12.8, 77.8, 13.1],
            "Chennai": [80.1, 12.8, 80.3, 13.2],
            "Kolkata": [88.2, 22.4, 88.4, 22.7],
            "Hyderabad": [78.3, 17.2, 78.6, 17.5]
        }
        
        return cities
        
    except Exception as e:
        logger.error(f"Error getting supported cities: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting supported cities: {str(e)}")

@router.get("/model/info")
async def get_model_info():
    """
    Get information about the current model.
    
    Returns:
        Dict: Model information including version and metrics
    """
    try:
        # This would typically load model metadata from a file
        model_info = {
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
        
        return model_info
        
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")