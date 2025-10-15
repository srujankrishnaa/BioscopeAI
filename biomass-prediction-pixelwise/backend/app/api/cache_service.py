"""
Cache Service for Pre-generated Satellite Images
Serves cached satellite images for fast region preview loading
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter()

class CacheService:
    """Service for managing cached satellite images"""
    
    def __init__(self):
        self.cache_dir = Path("./outputs/region_cache")
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.cache_metadata = {}
        self.load_metadata()
    
    def load_metadata(self):
        """Load cache metadata"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    self.cache_metadata = json.load(f)
                logger.info(f"Loaded cache metadata for {len(self.cache_metadata)} cities")
            except Exception as e:
                logger.error(f"Failed to load cache metadata: {e}")
                self.cache_metadata = {}
        else:
            logger.warning("No cache metadata found. Run generate_city_cache.py first.")
    
    def get_city_regions_from_cache(self, city_name: str) -> Optional[Dict]:
        """Get cached region data for a city"""
        # Try exact match first
        if city_name in self.cache_metadata:
            return self.cache_metadata[city_name]
        
        # Try case-insensitive match
        city_lower = city_name.lower()
        for cached_city, data in self.cache_metadata.items():
            if cached_city.lower() == city_lower:
                return data
        
        # Try partial match
        for cached_city, data in self.cache_metadata.items():
            if city_lower in cached_city.lower() or cached_city.lower() in city_lower:
                return data
        
        return None
    
    def get_cached_image_path(self, city_name: str, region_id: str) -> Optional[Path]:
        """Get path to cached satellite image"""
        city_data = self.get_city_regions_from_cache(city_name)
        if not city_data:
            return None
        
        region_data = city_data.get('regions', {}).get(region_id)
        if not region_data or not region_data.get('image_path'):
            return None
        
        image_path = self.cache_dir / region_data['image_path']
        if image_path.exists():
            return image_path
        
        return None
    
    def is_city_cached(self, city_name: str) -> bool:
        """Check if city regions are cached"""
        return self.get_city_regions_from_cache(city_name) is not None
    
    def get_cached_cities(self) -> List[str]:
        """Get list of all cached cities"""
        return list(self.cache_metadata.keys())

# Global cache service instance
cache_service = CacheService()

@router.get("/cached-cities")
async def get_cached_cities():
    """Get list of cities with cached satellite images"""
    try:
        cached_cities = cache_service.get_cached_cities()
        return {
            "cached_cities": cached_cities,
            "total_count": len(cached_cities),
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error getting cached cities: {e}")
        raise HTTPException(status_code=500, detail="Failed to get cached cities")

@router.get("/cache-status/{city_name}")
async def get_cache_status(city_name: str):
    """Check if a city has cached satellite images"""
    try:
        is_cached = cache_service.is_city_cached(city_name)
        city_data = cache_service.get_city_regions_from_cache(city_name)
        
        if is_cached and city_data:
            return {
                "city": city_name,
                "is_cached": True,
                "regions_count": len(city_data.get('regions', {})),
                "generated_at": city_data.get('generated_at'),
                "regions": list(city_data.get('regions', {}).keys())
            }
        else:
            return {
                "city": city_name,
                "is_cached": False,
                "message": "City not found in cache. Run cache generation first."
            }
    
    except Exception as e:
        logger.error(f"Error checking cache status for {city_name}: {e}")
        raise HTTPException(status_code=500, detail="Failed to check cache status")

@router.get("/cached-image/{city_name}/{region_id}")
async def get_cached_satellite_image(city_name: str, region_id: str):
    """Serve cached satellite image for a city region"""
    try:
        logger.info(f"Serving cached image for {city_name} - {region_id}")
        
        # Get cached image path
        image_path = cache_service.get_cached_image_path(city_name, region_id)
        
        if not image_path:
            raise HTTPException(
                status_code=404,
                detail=f"Cached image not found for {city_name} {region_id}"
            )
        
        # Serve the image file
        return FileResponse(
            path=str(image_path),
            media_type="image/png",
            headers={
                "Cache-Control": "public, max-age=86400",  # Cache for 24 hours
                "Content-Disposition": f"inline; filename={city_name}_{region_id}_satellite.png"
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving cached image: {e}")
        raise HTTPException(status_code=500, detail="Failed to serve cached image")

@router.get("/cache-info")
async def get_cache_info():
    """Get comprehensive cache information"""
    try:
        cached_cities = cache_service.get_cached_cities()
        total_regions = 0
        city_details = {}
        
        for city in cached_cities:
            city_data = cache_service.get_city_regions_from_cache(city)
            if city_data:
                regions = city_data.get('regions', {})
                total_regions += len(regions)
                city_details[city] = {
                    "regions_count": len(regions),
                    "generated_at": city_data.get('generated_at'),
                    "regions": list(regions.keys())
                }
        
        return {
            "cache_summary": {
                "total_cities": len(cached_cities),
                "total_regions": total_regions,
                "cache_directory": str(cache_service.cache_dir),
                "metadata_file": str(cache_service.metadata_file)
            },
            "cities": city_details,
            "status": "success"
        }
    
    except Exception as e:
        logger.error(f"Error getting cache info: {e}")
        raise HTTPException(status_code=500, detail="Failed to get cache information")