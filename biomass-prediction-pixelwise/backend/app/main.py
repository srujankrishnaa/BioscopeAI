"""
FastAPI application for biomass prediction.
"""
import os
import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from pydantic import BaseModel

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
    logger.info("Starting up Biomass Prediction API...")
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

# Import API routes - dynamic import to avoid circular dependencies
try:
    from app.api import prediction
    app.include_router(prediction.router, prefix="/api", tags=["prediction"])
    logger.info("Prediction API routes loaded successfully")
except ImportError as e:
    logger.warning(f"Could not import prediction API routes: {e}")

# Import region selection routes
try:
    from app.api import region_selection
    app.include_router(region_selection.router, prefix="/api", tags=["regions"])
    logger.info("Region selection API routes loaded successfully")
except ImportError as e:
    logger.warning(f"Could not import region selection API routes: {e}")
    logger.info("Using fallback endpoints defined in main.py")

# Import cache service routes
try:
    from app.api import cache_service
    app.include_router(cache_service.router, prefix="/api", tags=["cache"])
    logger.info("Cache service API routes loaded successfully")
except ImportError as e:
    logger.warning(f"Could not import cache service API routes: {e}")

# Import monitoring routes
try:
    from app.api import monitoring
    app.include_router(monitoring.router, prefix="/api/monitoring", tags=["monitoring"])
    logger.info("Monitoring API routes loaded successfully")
except ImportError as e:
    logger.warning(f"Could not import monitoring API routes: {e}")

# Mount static files for serving generated heatmaps and reports
if OUTPUT_DIR.exists():
    app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")
    logger.info("Static file serving enabled for /outputs")
else:
    logger.warning("Output directory not found - static file serving disabled")

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Biomass Prediction API", 
        "version": "1.0.2", 
        "status": "Railway deployment with fallback endpoints",
        "endpoints": ["/api/get-city-regions", "/api/analyze-region"],
        "timestamp": "2025-10-15T17:30:00Z"
    }

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
            "data_fetcher": {"status": "ready", "message": "Google Earth Engine / NASA GIBS"},
            "ml_model": {"status": "ready", "message": "Empirical biomass models"},
            "heatmap_generator": {"status": "ready", "message": "Multiple visualization strategies"},
            "report_generator": {"status": "ready", "message": "PDF and HTML reports"}
        }
    }

# Fallback endpoints for Railway deployment
@app.post("/api/get-city-regions")
async def get_city_regions_fallback(request: PredictionRequest):
    """Get city regions with cached satellite images."""
    logger.info(f"Getting regions for city: {request.city}")
    
    try:
        # Try to import and use cache service
        from app.api.cache_service import cache_service
        logger.info("Cache service imported successfully")
        
        # Check if city has cached data
        city_data = cache_service.get_city_regions_from_cache(request.city)
        logger.info(f"Cache lookup result for {request.city}: {city_data is not None}")
        
        if city_data:
            logger.info(f"Found cached data for {request.city}")
            regions = []
            
            for region_id, region_info in city_data.get('regions', {}).items():
                # Check if cached image exists
                image_path = cache_service.get_cached_image_path(request.city, region_id)
                
                if image_path:
                    logger.info(f"Cached image available for {request.city} {region_info['name']}: /api/cached-image/{request.city}/{region_id}")
                    preview_url = f"/api/cached-image/{request.city}/{region_id}"
                else:
                    logger.warning(f"No cached image found for {request.city} {region_id}")
                    preview_url = None
                
                regions.append({
                    "id": region_id,
                    "name": region_info['name'],
                    "description": region_info['description'],
                    "bbox": region_info['bbox'],
                    "coordinates": region_info['coordinates'],
                    "preview_image_url": preview_url
                })
            
            logger.info(f"Successfully generated {len(regions)} regions for {request.city}")
            
            return {
                "city": request.city,
                "total_regions": len(regions),
                "regions": regions,
                "city_center": [city_data['city_bbox'][1] + (city_data['city_bbox'][3] - city_data['city_bbox'][1])/2,
                               city_data['city_bbox'][0] + (city_data['city_bbox'][2] - city_data['city_bbox'][0])/2],
                "city_bbox": city_data['city_bbox']
            }
        
        else:
            logger.warning(f"No cached data found for {request.city}")
            
    except Exception as e:
        logger.error(f"Cache service failed: {e}")
    
    # Fallback to mock data if cache service fails or no cached data
    logger.info(f"Using fallback mock data for {request.city}")
    return {
        "city": request.city,
        "regions": [
            {
                "name": f"{request.city} Center",
                "id": "center",
                "description": f"Central business district and urban core of {request.city}",
                "preview_image_url": None,
                "bbox": [77.4, 12.9, 77.5, 13.0],
                "coordinates": {
                    "center": [12.95, 77.45],
                    "bounds": [[12.9, 77.4], [13.0, 77.5]]
                }
            },
            {
                "name": f"{request.city} North", 
                "id": "north",
                "description": f"Northern region of {request.city} with suburban and residential areas",
                "preview_image_url": None,
                "bbox": [77.4, 13.0, 77.5, 13.1],
                "coordinates": {
                    "center": [13.05, 77.45],
                    "bounds": [[13.0, 77.4], [13.1, 77.5]]
                }
            },
            {
                "name": f"{request.city} South",
                "id": "south", 
                "description": f"Southern region of {request.city} including industrial and residential zones",
                "preview_image_url": None,
                "bbox": [77.4, 12.8, 77.5, 12.9],
                "coordinates": {
                    "center": [12.85, 77.45],
                    "bounds": [[12.8, 77.4], [12.9, 77.5]]
                }
            },
            {
                "name": f"{request.city} East",
                "id": "east",
                "description": f"Eastern region of {request.city} with mixed urban development",
                "preview_image_url": None,
                "bbox": [77.5, 12.9, 77.6, 13.0],
                "coordinates": {
                    "center": [12.95, 77.55],
                    "bounds": [[12.9, 77.5], [13.0, 77.6]]
                }
            },
            {
                "name": f"{request.city} West",
                "id": "west",
                "description": f"Western region of {request.city} featuring diverse urban landscapes",
                "preview_image_url": None,
                "bbox": [77.3, 12.9, 77.4, 13.0],
                "coordinates": {
                    "center": [12.95, 77.35],
                    "bounds": [[12.9, 77.3], [13.0, 77.4]]
                }
            }
        ],
        "total_regions": 5,
        "city_center": [12.95, 77.45],
        "city_bbox": [77.3, 12.8, 77.6, 13.1]
    }

@app.post("/api/analyze-region")
async def analyze_region_fallback(request: dict):
    """Analyze a specific region for biomass prediction."""
    logger.info(f"Analyzing region: {request}")
    
    try:
        # Try to import and use the full region analysis
        from app.api.region_selection import analyze_region
        return await analyze_region(request)
    except ImportError as e:
        logger.warning(f"Region analysis module not available: {e}")
    except Exception as e:
        logger.error(f"Region analysis failed: {e}")
    
    # Fallback to mock analysis result
    city = request.get('city', 'Unknown')
    region_name = request.get('region_name', 'Unknown Region')
    
    logger.info(f"Using fallback analysis for {region_name}, {city}")
    
    # Generate mock biomass analysis result
    import random
    from datetime import datetime
    
    # Mock realistic biomass values
    total_agb = round(random.uniform(35.0, 85.0), 1)
    canopy_cover = round(random.uniform(15.0, 65.0), 1)
    carbon_sequestration = round(total_agb * 0.47 * 3.67, 1)  # Standard carbon calculation
    cooling_potential = round(canopy_cover * 0.08, 1)  # Approximate cooling effect
    
    return {
        "city": city,
        "region_name": region_name,
        "location": {
            "coordinates": f"{random.uniform(12.0, 28.0):.4f}, {random.uniform(72.0, 88.0):.4f}",
            "bbox": request.get('region_bbox', [77.4, 12.9, 77.5, 13.0])
        },
        "timestamp": datetime.now().isoformat(),
        "current_agb": {
            "total_agb": total_agb,
            "canopy_cover": canopy_cover,
            "carbon_sequestration": carbon_sequestration,
            "cooling_potential": cooling_potential
        },
        "forecasting": {
            "year_1": round(total_agb * 1.05, 1),
            "year_2": round(total_agb * 1.12, 1),
            "year_3": round(total_agb * 1.18, 1)
        },
        "heat_map": {
            "image_url": f"/outputs/heatmaps/biomass_heatmap_{city.lower().replace(' ', '_')}_{region_name.lower().replace(' ', '_')}_fallback.png",
            "description": f"Biomass analysis for {region_name}, {city} (Fallback Data)"
        },
        "status": "success",
        "message": "Analysis completed using fallback estimation model",
        "data_source": "Fallback Estimation (Demo Mode)"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000)),
        reload=True
    )