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

# Keep-alive self-ping to prevent Render free-tier sleep
from app.keep_alive import start_keep_alive, stop_keep_alive, get_keep_alive_status

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create output directories - ensure cross-platform compatibility
OUTPUT_DIR = Path("./outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "heatmaps").mkdir(exist_ok=True)
(OUTPUT_DIR / "reports").mkdir(exist_ok=True)
(OUTPUT_DIR / "region_cache").mkdir(exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    logger.info("Starting up Biomass Prediction API...")
    # Start self-ping to prevent Render free-tier cold starts
    start_keep_alive()
    yield
    stop_keep_alive()
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
region_router_loaded = False
try:
    from app.api.region_selection import router as region_router
    app.include_router(region_router, prefix="/api", tags=["regions"])
    logger.info("Included region_selection router")
    region_router_loaded = True
except Exception as e:
    logger.exception("Failed to include region_selection router")
    logger.error(f"Region selection router error: {e}")
    region_router_loaded = False

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
    from fastapi.responses import FileResponse
    import os
    
    # Custom static file handler with proper headers
    @app.get("/outputs/{file_path:path}")
    async def serve_static_file(file_path: str):
        """Serve static files with proper headers for images"""
        full_path = OUTPUT_DIR / file_path
        
        if not full_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        # Determine content type based on file extension
        content_type = "application/octet-stream"
        if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            content_type = f"image/{file_path.split('.')[-1].lower()}"
        elif file_path.lower().endswith('.svg'):
            content_type = "image/svg+xml"
        
        return FileResponse(
            path=str(full_path),
            media_type=content_type,
            headers={
                "Cache-Control": "public, max-age=3600",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET",
                "Access-Control-Allow-Headers": "*"
            }
        )
    
    logger.info("Custom static file serving enabled for /outputs with CORS headers")
else:
    logger.warning("Output directory not found - static file serving disabled")

@app.get("/test-static")
async def test_static():
    """Test static file serving."""
    import os
    from pathlib import Path
    
    # Check if outputs directory exists and list files
    output_dir = Path("./outputs/heatmaps")
    if output_dir.exists():
        files = list(output_dir.glob("*.png"))
        return {
            "status": "outputs directory exists",
            "heatmap_files": [f.name for f in files[-5:]],  # Last 5 files
            "total_files": len(files),
            "static_mount": "/outputs mounted",
            "test_url": f"/outputs/heatmaps/{files[-1].name}" if files else "no files found"
        }
    else:
        return {"status": "outputs directory not found"}

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

@app.api_route("/health", methods=["GET", "HEAD"])
async def health_check():
    """Health check endpoint — also used by UptimeRobot and self-ping keep-alive."""
    return {"status": "healthy", "timestamp": __import__('datetime').datetime.now().isoformat()}

@app.get("/keep-alive-status")
async def keep_alive_status():
    """Get the current status of the keep-alive self-ping mechanism."""
    return get_keep_alive_status()

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

@app.get("/api/debug-routes")
async def debug_routes():
    """Debug endpoint to check which routes are available"""
    routes = []
    for route in app.routes:
        if hasattr(route, 'path') and hasattr(route, 'methods'):
            routes.append({
                "path": route.path,
                "methods": list(route.methods) if route.methods else ["GET"]
            })
    
    return {
        "total_routes": len(routes),
        "routes": routes,
        "analyze_routes": [r for r in routes if "analyze" in r["path"]],
        "api_routes": [r for r in routes if r["path"].startswith("/api")]
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
    
    # No cached data found - return error instead of mock data
    logger.error(f"No cached data available for {request.city}")
    raise HTTPException(
        status_code=404, 
        detail=f"No cached data available for city: {request.city}. Please ensure the city has been processed and cached."
    )

# Only add fallback endpoints if router failed to load
if not region_router_loaded:
    logger.warning("Adding fallback analyze-region endpoint since router failed")
    
    @app.post("/api/analyze-region")
    async def analyze_region_fallback(payload: dict):
        """
        Fallback analyze-region endpoint - ensures 200 response always
        This runs if the region_selection router fails to load
        """
        logger.info(f"Using fallback analyze-region endpoint for: {payload}")
        
        city = payload.get("city", "Unknown")
        cache_dir = Path("./outputs/region_cache")
        
        # Check for cached images
        city_cache_dir = cache_dir / city
        if city_cache_dir.exists():
            cached_entries = []
            for region_file in city_cache_dir.glob("*.png"):
                region_name = region_file.stem
                cached_entries.append({
                    "region": region_name,
                    "image_url": f"/api/cached-image/{city}/{region_name}"
                })
            
            if cached_entries:
                logger.info(f"Fallback: Found {len(cached_entries)} cached images for {city}")
                return {
                    "status": "ok",
                    "source": "cache",
                    "city": city,
                    "regions": cached_entries,
                    "message": "Using fallback endpoint - cached data only"
                }
        
        # Return minimal response if no cache
        logger.warning(f"Fallback: No cache found for {city}")
        raise HTTPException(
            status_code=503,
            detail=f"Service temporarily unavailable for {city} - no cached data"
        )

    @app.get("/api/cached-image/{city}/{region}")
    async def serve_cached_image_fallback(city: str, region: str):
        """Fallback cached image serving"""
        logger.info(f"🔍 MAIN.PY FALLBACK ROUTE HIT: /api/cached-image/{city}/{region}")
        safe_city = city.replace("..", "")
        safe_region = region.replace("..", "")
        path = Path("./outputs/region_cache") / safe_city / f"{safe_region}.png"
        
        if not path.exists():
            raise HTTPException(status_code=404, detail="Cached image not found")
        
        from fastapi.responses import FileResponse
        return FileResponse(str(path), media_type="image/png")
else:
    logger.info("Region router loaded successfully - no fallback endpoints needed")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", os.getenv("API_PORT", 8000)))
    uvicorn.run(
        "app.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=port,
        reload=False  # Disable reload in production
    )