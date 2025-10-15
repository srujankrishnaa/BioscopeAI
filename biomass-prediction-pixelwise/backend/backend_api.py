"""
Backend API for Urban Biomass Prediction System
Academic Research Implementation

This module demonstrates the core backend functionality for:
1. Processing city-based biomass prediction requests
2. Integrating satellite data with machine learning models
3. Generating visualization outputs
4. Providing structured API responses

Author: Research Team
Purpose: Academic demonstration of biomass prediction API
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np
import logging

# Import research modules
from app.models.gee_data_fetcher import GEEDataFetcher
from app.api.satellite_image_generator import generate_satellite_heatmap

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize data fetcher
gee_fetcher = GEEDataFetcher()

# Request/Response Models
class PredictionRequest(BaseModel):
    """Model for biomass prediction request"""
    city: str

class SatelliteData(BaseModel):
    """Satellite-derived vegetation indices"""
    ndvi: float
    evi: float
    lai: float
    lst: float
    data_source: str

class BiomassResults(BaseModel):
    """Above Ground Biomass analysis results"""
    total_agb: float
    tree_biomass: float
    shrub_biomass: float
    herbaceous_biomass: float
    canopy_cover: float
    cooling_potential: float
    carbon_sequestration: float

class ForecastData(BaseModel):
    """Biomass growth forecasting"""
    year_1: float
    year_3: float
    year_5: float
    growth_rate: float

class UrbanMetrics(BaseModel):
    """Urban environmental performance metrics"""
    epi_score: int
    tree_cities_score: int
    green_space_ratio: float

class Location(BaseModel):
    """Geographic location information"""
    coordinates: str
    bbox: List[float]

class HeatMap(BaseModel):
    """Generated heatmap information"""
    image_url: str
    description: str

class PredictionResponse(BaseModel):
    """Complete prediction response"""
    city: str
    location: Location
    timestamp: str
    satellite_data: SatelliteData
    current_agb: BiomassResults
    forecasting: ForecastData
    urban_metrics: UrbanMetrics
    planning_recommendations: List[str]
    heat_map: HeatMap

def get_city_coordinates(city_name: str) -> Optional[Tuple[float, float, float, float]]:
    """
    Get bounding box coordinates for a city using geocoding
    
    Args:
        city_name: Name of the city to geocode
        
    Returns:
        Tuple of (min_lon, min_lat, max_lon, max_lat) or None if not found
    """
    try:
        import requests
        
        # Use Nominatim API for geocoding
        url = "https://nominatim.openstreetmap.org/search"
        params = {
            'q': city_name,
            'format': 'json',
            'limit': 1
        }
        headers = {'User-Agent': 'BiomassResearch/1.0'}
        
        response = requests.get(url, params=params, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data:
                bbox = data[0]['boundingbox']
                return (
                    float(bbox[2]),  # min_lon
                    float(bbox[0]),  # min_lat
                    float(bbox[3]),  # max_lon
                    float(bbox[1])   # max_lat
                )
        
        logger.warning(f"Could not geocode city: {city_name}")
        return None
        
    except Exception as e:
        logger.error(f"Geocoding error for {city_name}: {e}")
        return None

def calculate_urban_metrics(agb: float, canopy_cover: float) -> UrbanMetrics:
    """
    Calculate urban environmental performance metrics
    
    Args:
        agb: Above Ground Biomass value
        canopy_cover: Percentage of canopy coverage
        
    Returns:
        UrbanMetrics object with calculated scores
    """
    # Environmental Performance Index (simplified calculation)
    # Based on vegetation health indicators
    epi_score = int(min(100, (agb / 100.0) * 60 + (canopy_cover / 100) * 40))
    
    # Tree Cities Score (based on canopy coverage standards)
    tree_cities_score = int(min(100, (canopy_cover / 25.0) * 100))
    
    # Green space ratio
    green_space_ratio = round(canopy_cover / 100.0, 3)
    
    return UrbanMetrics(
        epi_score=epi_score,
        tree_cities_score=tree_cities_score,
        green_space_ratio=green_space_ratio
    )

def generate_recommendations(agb: float, canopy_cover: float, epi_score: int) -> List[str]:
    """
    Generate urban planning recommendations based on analysis results
    
    Args:
        agb: Above Ground Biomass value
        canopy_cover: Canopy coverage percentage
        epi_score: Environmental Performance Index score
        
    Returns:
        List of actionable recommendations
    """
    recommendations = []
    
    # Canopy cover recommendations
    if canopy_cover < 25:
        recommendations.append(
            f"Current canopy cover is {canopy_cover:.1f}%. Consider establishing "
            "urban forests and green corridors to increase carbon sequestration capacity."
        )
    
    # Biomass density recommendations
    if agb < 40:
        recommendations.append(
            "Current biomass density is below optimal levels. Focus on: "
            "native species plantations, green roof programs, and park expansion."
        )
    
    # Environmental performance recommendations
    if epi_score < 60:
        recommendations.append(
            f"EPI score ({epi_score}/100) indicates room for improvement. "
            "Implement smart irrigation systems to maintain vegetation health "
            "during dry seasons and maximize biomass growth potential."
        )
    
    # General recommendations
    recommendations.append(
        "Create neighborhood-level green infrastructure plans to distribute "
        "biomass equitably across all districts, especially in high-density areas."
    )
    
    if agb > 70:
        recommendations.append(
            f"Excellent biomass density ({agb:.1f} Mg/ha)! Maintain current "
            "green spaces and consider implementing a monitoring program."
        )
    
    return recommendations[:5]  # Limit to 5 recommendations

@router.post("/predict", response_model=PredictionResponse)
async def predict_biomass(request: PredictionRequest):
    """
    Main endpoint for urban biomass prediction
    
    This endpoint processes a city name and returns comprehensive biomass analysis
    including satellite data, current biomass estimates, forecasting, and recommendations.
    """
    try:
        logger.info(f"Processing prediction request for: {request.city}")
        
        # Step 1: Get city coordinates
        bbox = get_city_coordinates(request.city)
        if not bbox:
            raise HTTPException(
                status_code=404, 
                detail=f"City not found: {request.city}"
            )
        
        # Step 2: Fetch satellite data
        satellite_data = gee_fetcher.fetch_satellite_data(bbox)
        if not satellite_data:
            raise HTTPException(
                status_code=500,
                detail="Failed to fetch satellite data"
            )
        
        # Step 3: Calculate biomass from satellite indices
        biomass_results = gee_fetcher.calculate_biomass_from_indices(
            satellite_data['ndvi'],
            satellite_data['evi'],
            satellite_data['lai']
        )
        
        # Step 4: Generate biomass forecasting
        forecast_data = gee_fetcher.forecast_biomass(
            biomass_results['total_agb'],
            satellite_data['ndvi'],
            satellite_data['lai']
        )
        
        # Step 5: Calculate urban performance metrics
        urban_metrics = calculate_urban_metrics(
            biomass_results['total_agb'],
            biomass_results['canopy_cover']
        )
        
        # Step 6: Generate planning recommendations
        recommendations = generate_recommendations(
            biomass_results['total_agb'],
            biomass_results['canopy_cover'],
            urban_metrics.epi_score
        )
        
        # Step 7: Generate visualization heatmap
        heatmap_path = generate_satellite_heatmap(
            request.city,
            "Urban Analysis",
            satellite_data['ndvi_array'],
            biomass_results,
            bbox,
            use_real_satellite=True
        )
        
        # Step 8: Prepare response
        center_lon = (bbox[0] + bbox[2]) / 2
        center_lat = (bbox[1] + bbox[3]) / 2
        
        response = PredictionResponse(
            city=request.city,
            location=Location(
                coordinates=f"{center_lat:.4f}, {center_lon:.4f}",
                bbox=list(bbox)
            ),
            timestamp=datetime.now().isoformat(),
            satellite_data=SatelliteData(
                ndvi=satellite_data['ndvi'],
                evi=satellite_data['evi'],
                lai=satellite_data['lai'],
                lst=satellite_data['lst'],
                data_source=satellite_data['data_source']
            ),
            current_agb=BiomassResults(**biomass_results),
            forecasting=ForecastData(**forecast_data),
            urban_metrics=urban_metrics,
            planning_recommendations=recommendations,
            heat_map=HeatMap(
                image_url=f"http://localhost:8000{heatmap_path}",
                description=f"Biomass analysis for {request.city}"
            )
        )
        
        logger.info(f"Prediction completed successfully for {request.city}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during prediction"
        )

@router.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "gee_data_fetcher": "operational",
            "satellite_generator": "operational",
            "api_server": "operational"
        }
    }

@router.get("/supported-cities")
async def get_supported_cities():
    """Get list of cities commonly used for testing"""
    return {
        "cities": [
            "Bangalore", "Mumbai", "Delhi", "Chennai", "Hyderabad",
            "Kolkata", "Pune", "Ahmedabad", "Jaipur", "Lucknow"
        ],
        "note": "API supports any city with available satellite coverage"
    }

# Example usage for testing
if __name__ == "__main__":
    """
    Simple test to verify API functionality
    """
    import asyncio
    
    async def test_prediction():
        request = PredictionRequest(city="Bangalore")
        try:
            result = await predict_biomass(request)
            print("Test successful!")
            print(f"City: {result.city}")
            print(f"Total AGB: {result.current_agb.total_agb}")
            print(f"Canopy Cover: {result.current_agb.canopy_cover}")
        except Exception as e:
            print(f"Test failed: {e}")
    
    # Run test
    asyncio.run(test_prediction())