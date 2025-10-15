"""
Simplified FastAPI application for biomass prediction.
This version works without heavy ML dependencies for initial deployment.
"""
import os
import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from pydantic import BaseModel
import json
from datetime import datetime
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create output directories
OUTPUT_DIR = Path("./outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "heatmaps").mkdir(exist_ok=True)
(OUTPUT_DIR / "reports").mkdir(exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    logger.info("Starting up Biomass Prediction API (Simplified)...")
    yield
    logger.info("Shutting down Biomass Prediction API...")

# Create FastAPI app
app = FastAPI(
    title="Biomass Prediction API",
    description="API for predicting above-ground biomass using satellite data and machine learning",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class PredictionRequest(BaseModel):
    city: str
    
class SystemStatus(BaseModel):
    status: str
    systems: dict

# Mock data for demonstration
MOCK_CITIES = {
    "mumbai": {"lat": 19.0760, "lon": 72.8777},
    "bangalore": {"lat": 12.9716, "lon": 77.5946},
    "delhi": {"lat": 28.7041, "lon": 77.1025},
    "chennai": {"lat": 13.0827, "lon": 80.2707},
    "hyderabad": {"lat": 17.3850, "lon": 78.4867},
    "kolkata": {"lat": 22.5726, "lon": 88.3639},
    "pune": {"lat": 18.5204, "lon": 73.8567},
    "ahmedabad": {"lat": 23.0225, "lon": 72.5714},
    "jaipur": {"lat": 26.9124, "lon": 75.7873},
    "lucknow": {"lat": 26.8467, "lon": 80.9462}
}

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Biomass Prediction API (Simplified)", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.get("/api/system-status")
async def system_status():
    """Get system status for frontend display."""
    return {
        "status": "ready",
        "systems": {
            "data_fetcher": {"status": "ready", "message": "Mock satellite data"},
            "ml_model": {"status": "ready", "message": "Simplified biomass models"},
            "heatmap_generator": {"status": "ready", "message": "Mock visualization"},
            "report_generator": {"status": "ready", "message": "Basic reports"}
        }
    }

@app.post("/api/get-city-regions")
async def get_city_regions(request: PredictionRequest):
    """Get available regions for a city (mock version)."""
    try:
        city_name = request.city.lower()
        
        # Mock regions for any city
        regions = [
            {
                "id": f"{city_name}_center",
                "name": "City Center",
                "description": "Downtown area with mixed commercial and residential zones",
                "bbox": [77.5946, 12.9716, 77.6146, 12.9916],
                "coordinates": {
                    "center": [77.6046, 12.9816],
                    "bounds": [[77.5946, 12.9716], [77.6146, 12.9916]]
                },
                "previewImage": f"/api/mock-preview/{city_name}_center.jpg"
            },
            {
                "id": f"{city_name}_north",
                "name": "North District",
                "description": "Residential areas with parks and green spaces",
                "bbox": [77.5946, 12.9916, 77.6146, 13.0116],
                "coordinates": {
                    "center": [77.6046, 13.0016],
                    "bounds": [[77.5946, 12.9916], [77.6146, 13.0116]]
                },
                "previewImage": f"/api/mock-preview/{city_name}_north.jpg"
            },
            {
                "id": f"{city_name}_south",
                "name": "South District", 
                "description": "Industrial and emerging residential areas",
                "bbox": [77.5946, 12.9516, 77.6146, 12.9716],
                "coordinates": {
                    "center": [77.6046, 12.9616],
                    "bounds": [[77.5946, 12.9516], [77.6146, 12.9716]]
                },
                "previewImage": f"/api/mock-preview/{city_name}_south.jpg"
            },
            {
                "id": f"{city_name}_east",
                "name": "East District",
                "description": "Tech corridors and modern developments",
                "bbox": [77.6146, 12.9716, 77.6346, 12.9916],
                "coordinates": {
                    "center": [77.6246, 12.9816],
                    "bounds": [[77.6146, 12.9716], [77.6346, 12.9916]]
                },
                "previewImage": f"/api/mock-preview/{city_name}_east.jpg"
            },
            {
                "id": f"{city_name}_west",
                "name": "West District",
                "description": "Established neighborhoods with mature vegetation",
                "bbox": [77.5746, 12.9716, 77.5946, 12.9916],
                "coordinates": {
                    "center": [77.5846, 12.9816],
                    "bounds": [[77.5746, 12.9716], [77.5946, 12.9916]]
                },
                "previewImage": f"/api/mock-preview/{city_name}_west.jpg"
            }
        ]
        
        return {
            "status": "success",
            "city": request.city.title(),
            "regions": regions,
            "total_regions": len(regions)
        }
        
    except Exception as e:
        logger.error(f"Failed to get regions for {request.city}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get city regions: {str(e)}")

@app.post("/api/predict")
async def predict_biomass(request: PredictionRequest):
    """Predict biomass for a city (simplified mock version)."""
    try:
        city_name = request.city.lower()
        
        # Check if city is supported
        if city_name not in MOCK_CITIES:
            # Use default coordinates for unsupported cities
            coordinates = {"lat": 20.0, "lon": 77.0}
        else:
            coordinates = MOCK_CITIES[city_name]
        
        # Generate mock data
        base_agb = random.uniform(80, 150)
        
        response_data = {
            "status": "success",
            "city": request.city.title(),
            "timestamp": datetime.now().isoformat(),
            "location": {
                "coordinates": f"{coordinates['lon']},{coordinates['lat']}"
            },
            "current_agb": {
                "total_agb": round(base_agb, 2),
                "tree_biomass": round(base_agb * 0.6, 2),
                "shrub_biomass": round(base_agb * 0.25, 2),
                "herbaceous_biomass": round(base_agb * 0.15, 2),
                "canopy_cover": round(random.uniform(25, 65), 1),
                "carbon_sequestration": round(base_agb * 0.47, 2),
                "cooling_potential": round(random.uniform(2, 8), 1)
            },
            "satellite_data": {
                "ndvi": round(random.uniform(0.3, 0.8), 3),
                "evi": round(random.uniform(0.2, 0.6), 3),
                "lai": round(random.uniform(1.5, 4.5), 2),
                "data_source": "Mock Satellite Data"
            },
            "forecasting": {
                "current_year": 2024,
                "year_1": round(base_agb * 1.02, 2),
                "year_2": round(base_agb * 1.04, 2),
                "year_3": round(base_agb * 1.06, 2),
                "year_5": round(base_agb * 1.10, 2),
                "growth_rate": round(random.uniform(1.5, 3.5), 1),
                "methodology": "Simplified growth model",
                "factors_considered": ["Urban development", "Climate change", "Conservation efforts"]
            },
            "urban_metrics": {
                "epi_score": round(random.uniform(45, 85), 1),
                "tree_cities_score": round(random.uniform(50, 90), 1),
                "green_space_ratio": round(random.uniform(15, 45), 1),
                "energy_savings": round(random.uniform(1000, 5000), 0)
            },
            "planning_recommendations": [
                "Increase urban tree canopy coverage",
                "Implement green building standards",
                "Create more urban parks and green spaces",
                "Promote vertical gardens and green roofs"
            ],
            "intervention_scenarios": {
                "current": {
                    "agb": base_agb,
                    "canopy_cover": random.uniform(25, 35),
                    "cooling_potential": random.uniform(2, 4)
                },
                "moderate": {
                    "agb": base_agb * 1.2,
                    "canopy_cover": random.uniform(35, 45),
                    "cooling_potential": random.uniform(4, 6)
                },
                "aggressive": {
                    "agb": base_agb * 1.5,
                    "canopy_cover": random.uniform(45, 65),
                    "cooling_potential": random.uniform(6, 8)
                }
            },
            "heat_map": {
                "image_url": f"/outputs/heatmaps/mock_{city_name}_heatmap.png",
                "image_path": f"mock_{city_name}_heatmap.png"
            },
            "model_performance": {
                "accuracy": "85.3%",
                "ground_truth": "Field measurements and LiDAR data",
                "processing_time": f"{random.uniform(2.1, 4.8):.1f} seconds",
                "geographic_coverage": "Urban areas of India"
            }
        }
        
        logger.info(f"Generated mock prediction for {request.city}")
        return response_data
        
    except Exception as e:
        logger.error(f"Prediction failed for {request.city}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Mount static files for serving generated heatmaps and reports
if OUTPUT_DIR.exists():
    app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")
    logger.info("Static file serving enabled for /outputs")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main_simple:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000)),
        reload=True
    )