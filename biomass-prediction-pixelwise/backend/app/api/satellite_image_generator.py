"""
Real Satellite Imagery with Biomass Overlay Generator
Fetches actual satellite images from Google Earth Engine and overlays biomass analysis
"""

import ee
import numpy as np
import requests
from PIL import Image
import io
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib.colors import LinearSegmentedColormap
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from app.utils.city_limits import get_max_pixels_for_city, cap_dimensions, MAX_REQUEST_BYTES, GEE_HARD_LIMIT_BYTES
from app.utils.indian_states import STATE_LIMITS, DEFAULT_STATE_LIMIT, INDIAN_STATES
from mpl_toolkits.axes_grid1 import make_axes_locatable

logger = logging.getLogger(__name__)

def get_quality_settings(quality: str = 'balanced') -> dict:
    """
    Get quality settings for satellite image processing
    
    Args:
        quality: Quality level ('high', 'balanced', 'fast')
        
    Returns:
        Dictionary with quality parameters
    """
    quality_configs = {
        'high': {
            'dimensions': 2048,
            'collection_limit': 30,
            'rgb_stats_scale': 100,
            'save_dpi': 300,
            'interpolation': 'lanczos'
        },
        'balanced': {
            'dimensions': 1792,
            'collection_limit': 12,
            'rgb_stats_scale': 300,
            'save_dpi': 200,
            'interpolation': 'nearest'
        },
        'fast': {
            'dimensions': 1024,
            'collection_limit': 6,
            'rgb_stats_scale': 500,
            'save_dpi': 150,
            'interpolation': 'nearest'
        }
    }
    
    return quality_configs.get(quality, quality_configs['balanced'])

def make_requests_session(retries=3, backoff_factor=0.5, status_forcelist=(500,502,503,504)):
    """Create a requests session with retry logic"""
    session = requests.Session()
    retry = Retry(total=retries, read=retries, connect=retries,
                  backoff_factor=backoff_factor, status_forcelist=status_forcelist)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('https://', adapter)
    session.mount('http://', adapter)
    return session

def mask_s2_clouds(img):
    """Apply simple cloud mask to Sentinel-2 image using QA60 band"""
    try:
        # Using QA60 band for cloud masking (simple but effective)
        qa = img.select('QA60')
        # Bits 10 and 11 are clouds and cirrus respectively
        cloud_mask = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
        return img.updateMask(cloud_mask)
    except:
        # If QA60 is not available, return the original image
        logger.warning("Could not apply cloud mask, using original image")
        return img


def get_smart_region(bbox: tuple, city_name: str) -> tuple:
    """
    Create a smart region that adapts to city size to avoid GEE export failures
    Uses intelligent urban core detection and adaptive clipping for high-quality exports
    
    Args:
        bbox: (min_lon, min_lat, max_lon, max_lat)
        city_name: Name of the city
        
    Returns:
        Tuple of (optimized_bbox, region_geometry, strategy_used)
    """
    # Calculate bounding box dimensions
    bbox_width = bbox[2] - bbox[0]   # longitude span
    bbox_height = bbox[3] - bbox[1]  # latitude span
    bbox_area = bbox_width * bbox_height
    
    # Convert to approximate kilometers (1 degree ≈ 111 km)
    width_km = bbox_width * 111
    height_km = bbox_height * 111
    
    logger.info(f"📏 {city_name} bbox: {width_km:.1f}km × {height_km:.1f}km (area: {bbox_area:.4f}°²)")
    
    # Strategy 1: No clipping needed for small/medium cities
    if bbox_area <= 0.08:  # Up to ~9km × 9km (reduced threshold)
        region = ee.Geometry.Rectangle([bbox[0], bbox[1], bbox[2], bbox[3]])
        logger.info(f"📐 Using full bbox for {city_name} (small city)")
        return bbox, region, "full_bbox"
    
    # Strategy 2: Smart clipping for medium cities - slight focus on center
    elif bbox_area <= 0.2:  # Up to ~14km × 14km
        # Calculate center point
        center_lon = (bbox[0] + bbox[2]) / 2
        center_lat = (bbox[1] + bbox[3]) / 2
        
        # Slight clipping to focus on urban core
        focus_factor = 0.85  # Keep 85% of original area
        focused_width = bbox_width * focus_factor
        focused_height = bbox_height * focus_factor
        
        clipped_bbox = (
            center_lon - focused_width / 2,
            center_lat - focused_height / 2,
            center_lon + focused_width / 2,
            center_lat + focused_height / 2
        )
        
        region = ee.Geometry.Rectangle([clipped_bbox[0], clipped_bbox[1], clipped_bbox[2], clipped_bbox[3]])
        
        clipped_width_km = focused_width * 111
        clipped_height_km = focused_height * 111
        
        logger.info(f"✂️ Light clipping {city_name} from {width_km:.1f}×{height_km:.1f}km to {clipped_width_km:.1f}×{clipped_height_km:.1f}km")
        logger.info(f"📐 Using focused region for {city_name}")
        
        return clipped_bbox, region, "light_clip"
    
    # Strategy 3: Aggressive clipping for large cities - focus on urban core
    else:
        # Calculate center point
        center_lon = (bbox[0] + bbox[2]) / 2
        center_lat = (bbox[1] + bbox[3]) / 2
        
        # Define optimal size for high-quality exports (15-20km × 15-20km)
        optimal_size_km = 18
        optimal_size_deg = optimal_size_km / 111  # Convert to degrees
        
        # Calculate focused region based on city size
        if bbox_area > 1.0:  # Very large area (>100km × 100km)
            # Use much smaller region for very large cities/states
            focused_width = min(optimal_size_deg * 0.9, bbox_width * 0.25)
            focused_height = min(optimal_size_deg * 0.9, bbox_height * 0.25)
        elif bbox_area > 0.5:  # Large area (70km × 70km)
            focused_width = min(optimal_size_deg, bbox_width * 0.4)
            focused_height = min(optimal_size_deg, bbox_height * 0.4)
        else:  # Moderately large area (30km × 30km)
            focused_width = min(optimal_size_deg * 1.2, bbox_width * 0.6)
            focused_height = min(optimal_size_deg * 1.2, bbox_height * 0.6)
        
        # Ensure minimum size for quality
        min_size_deg = 0.1  # ~11km
        focused_width = max(focused_width, min_size_deg)
        focused_height = max(focused_height, min_size_deg)
        
        # Create clipped bbox
        clipped_bbox = (
            center_lon - focused_width / 2,
            center_lat - focused_height / 2,
            center_lon + focused_width / 2,
            center_lat + focused_height / 2
        )
        
        region = ee.Geometry.Rectangle([clipped_bbox[0], clipped_bbox[1], clipped_bbox[2], clipped_bbox[3]])
        
        clipped_width_km = focused_width * 111
        clipped_height_km = focused_height * 111
        
        logger.info(f"✂️ Aggressive clipping {city_name} from {width_km:.1f}×{height_km:.1f}km to {clipped_width_km:.1f}×{clipped_height_km:.1f}km")
        logger.info(f"📐 Using urban core region for {city_name}")
        
        return clipped_bbox, region, "urban_core_clip"


def estimate_request_size(width_km: float, height_km: float, scale_m: int, dimensions: int = None, use_uint8: bool = True) -> int:
    """
    Estimate the request size in bytes to avoid GEE payload limit errors
    
    Args:
        width_km: Width of the region in kilometers
        height_km: Height of the region in kilometers
        scale_m: Scale in meters per pixel (if using scale-based export)
        dimensions: Dimensions in pixels (if using dimensions-based export)
        use_uint8: Whether using uint8 visualization (1 byte/channel) or float32 (4 bytes/channel)
        
    Returns:
        Estimated request size in bytes
    """
    # Calculate pixel dimensions
    if dimensions:
        # dimensions represents the long side; preserve aspect ratio using bbox ratio
        aspect = height_km / width_km if width_km > 0 else 1
        pixels_x = dimensions
        pixels_y = max(1, int(round(dimensions * aspect)))
    else:
        # Using scale-based export
        pixels_x = int(round(width_km * 1000 / scale_m))
        pixels_y = int(round(height_km * 1000 / scale_m))
    
    # Estimate bytes per pixel based on data type
    if use_uint8:
        # uint8: 1 byte per channel
        bytes_per_pixel = 3  # RGB = 3 channels
    else:
        # float32: 4 bytes per channel
        bytes_per_pixel = 12  # RGB = 3 channels × 4 bytes
    
    # Calculate total bytes for one image (we request RGB and NDVI separately)
    total_bytes = pixels_x * pixels_y * bytes_per_pixel
    
    # Add overhead for headers, metadata, etc. (10% buffer)
    total_bytes = int(total_bytes * 1.1)
    
    return total_bytes


def get_max_safe_dimensions(use_uint8: bool = True) -> int:
    """
    Calculate the maximum safe dimensions for GEE requests
    
    Args:
        use_uint8: Whether using uint8 visualization
        
    Returns:
        Maximum safe dimension in pixels
    """
    import math
    # GEE size limit (50,331,648 bytes)
    MAX_SIZE_BYTES = 50331648
    
    if use_uint8:
        # uint8: 1 byte per channel, 3 channels for RGB
        bytes_per_pixel = 3
    else:
        # float32: 4 bytes per channel, 3 channels for RGB
        bytes_per_pixel = 12
    
    # Calculate maximum square dimension
    max_dim = int(math.sqrt(MAX_SIZE_BYTES / bytes_per_pixel))
    
    # Add safety margin (5% reduction)
    max_dim = int(max_dim * 0.95)
    
    return max_dim


def adjust_parameters_for_size_limit(width_km: float, height_km: float, initial_scale: int = 10,
                                   initial_dimensions: int = 2048, target_min_pixels: int = 1726,
                                   use_uint8: bool = True) -> dict:
    """
    Adjust export parameters to stay within GEE size limits while maximizing quality
    
    Args:
        width_km: Width of the region in kilometers
        height_km: Height of the region in kilometers
        initial_scale: Initial scale in meters per pixel
        initial_dimensions: Initial dimensions in pixels
        target_min_pixels: Target minimum pixels for quality
        use_uint8: Whether using uint8 visualization
        
    Returns:
        Dictionary with adjusted parameters
    """
    # Get maximum safe dimensions for the data type
    max_safe_dim = get_max_safe_dimensions(use_uint8)
    
    # Start with the requested dimensions but cap at safe limit
    dimensions = min(initial_dimensions, max_safe_dim)
    
    # GEE size limit (50MB with safety margin)
    MAX_SIZE_BYTES = 48 * 1024 * 1024  # 48MB with safety margin
    
    # Estimate size with uint8 visualization
    estimated_size = estimate_request_size(width_km, height_km, None, dimensions, use_uint8)
    
    # If still too large, reduce dimensions further
    while estimated_size > MAX_SIZE_BYTES and dimensions > target_min_pixels:
        dimensions -= 128  # Reduce in steps of 128px
        estimated_size = estimate_request_size(width_km, height_km, None, dimensions, use_uint8)
        logger.info(f"🔧 Reducing dimensions to {dimensions}px (estimated size: {estimated_size/1024/1024:.1f}MB)")
    
    # If dimensions-based export is still too large or below target, try scale-based
    if dimensions < target_min_pixels:
        logger.info(f"🔧 Switching to scale-based export for better quality")
        scale = initial_scale
        estimated_size = estimate_request_size(width_km, height_km, scale, None, use_uint8)
        
        # If too large, increase scale (lower resolution)
        while estimated_size > MAX_SIZE_BYTES and scale < 30:
            scale += 2  # Increase in steps of 2m
            estimated_size = estimate_request_size(width_km, height_km, scale, None, use_uint8)
            logger.info(f"🔧 Increasing scale to {scale}m/pixel (estimated size: {estimated_size/1024/1024:.1f}MB)")
        
        # Calculate estimated pixels
        estimated_pixels = int(width_km * 1000 / scale)
        logger.info(f"📐 Final scale-based export: {scale}m/pixel (~{estimated_pixels}×{estimated_pixels}px)")
        
        return {
            'use_scale': True,
            'scale': scale,
            'dimensions': None,
            'estimated_size': estimated_size,
            'estimated_pixels': estimated_pixels
        }
    else:
        logger.info(f"📐 Final dimensions-based export: {dimensions}×{dimensions}px (estimated size: {estimated_size/1024/1024:.1f}MB)")
        return {
            'use_scale': False,
            'scale': None,
            'dimensions': dimensions,
            'estimated_size': estimated_size,
            'estimated_pixels': dimensions
        }


def fetch_high_res_satellite_and_ndvi(bbox: tuple, region_name: str, max_pixels: int = None, quality: str = 'balanced') -> tuple:
    """
    Fetch HIGH RESOLUTION Sentinel-2 satellite image AND calculate NDVI for biomass
    ULTRA-ROBUST version with smart region handling and request size optimization
    
    Args:
        bbox: (min_lon, min_lat, max_lon, max_lat)
        region_name: Name of the region (city or state)
        max_pixels: Maximum pixels for the region (optional, will use limits if not provided)
        
    Returns:
        Dictionary with 'rgb', 'ndvi', and 'metadata' or None
    """
    # Calculate bounding box dimensions
    bbox_width = bbox[2] - bbox[0]   # longitude span
    bbox_height = bbox[3] - bbox[1]  # latitude span
    bbox_area = bbox_width * bbox_height
    
    # Convert to kilometers
    width_km = bbox_width * 111
    height_km = bbox_height * 111
    
    logger.info(f"📏 {region_name} bbox: {width_km:.1f}km × {height_km:.1f}km (area: {bbox_area:.4f}°²)")
    
    # Determine if this is a state or city and get appropriate max pixels
    if max_pixels is None:
        # Check if this is a state (from the states list)
        from app.utils.indian_states import INDIAN_STATES
        if region_name in INDIAN_STATES:
            max_pixels = STATE_LIMITS.get(region_name, DEFAULT_STATE_LIMIT)
            logger.info(f"📍 Detected state: {region_name}, using max pixels: {max_pixels}")
        else:
            # It's a city, use city limits
            max_pixels = get_max_pixels_for_city(region_name)
            logger.info(f"🏙️ Detected city: {region_name}, using max pixels: {max_pixels}")
    
    # Define optimized export strategies based on region size
    if bbox_area <= 0.1:  # Small regions
        export_strategies = [
            {"name": "high_quality_dimensions", "use_scale": False, "use_dimensions": True, "target_pixels": min(2048, max_pixels)},
            {"name": "optimal_scale", "use_scale": True, "use_dimensions": False, "target_meters": 8},
            {"name": "medium_quality_dimensions", "use_scale": False, "use_dimensions": True, "target_pixels": min(1792, max_pixels)},
            {"name": "balanced_scale", "use_scale": True, "use_dimensions": False, "target_meters": 10},
            {"name": "emergency_clip", "use_scale": True, "use_dimensions": False, "emergency_clip": True, "target_meters": 6}
        ]
    elif bbox_area <= 0.3:  # Medium regions
        export_strategies = [
            {"name": "high_quality_dimensions", "use_scale": False, "use_dimensions": True, "target_pixels": min(2048, max_pixels)},
            {"name": "optimal_scale", "use_scale": True, "use_dimensions": False, "target_meters": 10},
            {"name": "medium_quality_dimensions", "use_scale": False, "use_dimensions": True, "target_pixels": min(1792, max_pixels)},
            {"name": "balanced_scale", "use_scale": True, "use_dimensions": False, "target_meters": 12},
            {"name": "emergency_clip", "use_scale": True, "use_dimensions": False, "emergency_clip": True, "target_meters": 8}
        ]
    else:  # Large regions (states)
        export_strategies = [
            {"name": "high_quality_dimensions", "use_scale": False, "use_dimensions": True, "target_pixels": min(2048, max_pixels)},
            {"name": "optimal_scale", "use_scale": True, "use_dimensions": False, "target_meters": 12},
            {"name": "medium_quality_dimensions", "use_scale": False, "use_dimensions": True, "target_pixels": min(1792, max_pixels)},
            {"name": "balanced_scale", "use_scale": True, "use_dimensions": False, "target_meters": 15},
            {"name": "emergency_clip", "use_scale": True, "use_dimensions": False, "emergency_clip": True, "target_meters": 10}
        ]
    
    for strategy_idx, export_strategy in enumerate(export_strategies):
        try:
            logger.info(f"🔄 Trying export strategy {strategy_idx + 1}/{len(export_strategies)}: {export_strategy['name']}")
            
            # Get smart region based on region size (may be modified for emergency strategies)
            if export_strategy.get('emergency_clip'):
                # Emergency strategy: create a high-quality small region around center
                center_lon = (bbox[0] + bbox[2]) / 2
                center_lat = (bbox[1] + bbox[3]) / 2
                
                # Calculate optimal emergency size based on region type
                bbox_width = bbox[2] - bbox[0]
                bbox_height = bbox[3] - bbox[1]
                bbox_area = bbox_width * bbox_height
                
                # Special handling for major Indian cities that need higher resolution
                major_cities = ['hyderabad', 'bangalore', 'bengaluru', 'chennai', 'mumbai', 'delhi', 'kolkata', 'pune']
                region_lower = region_name.lower()
                
                if region_lower in major_cities:
                    # Use a larger, higher-quality region for major cities
                    emergency_size = 0.18  # ~20km × 20km for better quality
                    logger.warning(f"🏙️ Using enhanced emergency clip for major city {region_name}")
                elif bbox_area > 1.0:  # Very large area (states)
                    emergency_size = 0.12  # ~13km × 13km (increased from 0.09)
                elif bbox_area > 0.5:  # Large area
                    emergency_size = 0.15  # ~16km × 16km (increased from 0.12)
                else:  # Medium area
                    emergency_size = 0.18  # ~20km × 20km (increased from 0.15)
                
                emergency_bbox = (
                    center_lon - emergency_size/2,
                    center_lat - emergency_size/2,
                    center_lon + emergency_size/2,
                    center_lat + emergency_size/2
                )
                optimized_bbox = emergency_bbox
                region = ee.Geometry.Rectangle([emergency_bbox[0], emergency_bbox[1], emergency_bbox[2], emergency_bbox[3]])
                strategy = "emergency_clip"
                emergency_width_km = emergency_size * 111
                emergency_height_km = emergency_size * 111
                logger.warning(f"🚨 Using enhanced emergency clip: {emergency_width_km:.1f}km × {emergency_height_km:.1f}km for {region_name}")
                
                # Use optimized parameters for emergency clip
                initial_scale = export_strategy.get('target_meters', 8)
                if region_lower in major_cities:
                    initial_scale = 6  # Ultra-high for major cities
                
                # Calculate optimal parameters for emergency region
                adjusted_params = adjust_parameters_for_size_limit(
                    emergency_width_km, emergency_height_km,
                    initial_scale=initial_scale,
                    initial_dimensions=2048,
                    target_min_pixels=1726,
                    use_uint8=True  # Using uint8 visualization
                )
                
                # Apply region-specific limits
                if adjusted_params.get('dimensions'):
                    dimensions = adjusted_params['dimensions']
                    # Check if it's a state or city for appropriate capping
                    from app.utils.indian_states import INDIAN_STATES
                    if region_name in INDIAN_STATES:
                        # It's a state, use state limits
                        capped_dimensions = min(dimensions, STATE_LIMITS.get(region_name, DEFAULT_STATE_LIMIT))
                    else:
                        # It's a city, use city limits
                        capped_dimensions = cap_dimensions(dimensions, dimensions, region_name)
                        if capped_dimensions != (dimensions, dimensions):
                            capped_dimensions = capped_dimensions[0]
                    
                    if capped_dimensions != dimensions:
                        dimensions = capped_dimensions
                        logger.warning(f"🏙️ Region-specific limit: Capped {region_name} dimensions to {dimensions}px")
                        adjusted_params['dimensions'] = dimensions
                        adjusted_params['estimated_pixels'] = dimensions
                
                if adjusted_params['use_scale']:
                    scale = adjusted_params['scale']
                    logger.info(f"🎯 Emergency clip using scale: {scale}m/pixel (~{adjusted_params['estimated_pixels']}×{adjusted_params['estimated_pixels']}px)")
                else:
                    dimensions = adjusted_params['dimensions']
                    logger.info(f"🎯 Emergency clip using dimensions: {dimensions}×{dimensions}px")
            else:
                optimized_bbox, region, strategy = get_smart_region(bbox, region_name)
                
                # Calculate region dimensions for this strategy
                opt_width = optimized_bbox[2] - optimized_bbox[0]
                opt_height = optimized_bbox[3] - optimized_bbox[1]
                opt_width_km = opt_width * 111
                opt_height_km = opt_height * 111
                
                # Use optimized parameters for regular strategies
                if export_strategy.get('target_meters'):
                    initial_scale = export_strategy['target_meters']
                    target_min_pixels = 1726
                    
                    # Calculate optimal parameters
                    adjusted_params = adjust_parameters_for_size_limit(
                        opt_width_km, opt_height_km,
                        initial_scale=initial_scale,
                        initial_dimensions=2048,
                        target_min_pixels=target_min_pixels,
                        use_uint8=True  # Using uint8 visualization
                    )
                    
                    # Apply city-specific limits
                    if adjusted_params.get('dimensions'):
                        dimensions = adjusted_params['dimensions']
                        cw, ch = cap_dimensions(dimensions, dimensions, region_name)
                        if (cw, ch) != (dimensions, dimensions):
                            logger.warning(f"🏙️ City-specific limit: Capped {region_name} dimensions to {cw}×{ch}px")
                        # choose smaller to keep square
                        dimensions = min(cw, ch)
                        adjusted_params['dimensions'] = dimensions
                        adjusted_params['estimated_pixels'] = dimensions
                    
                    if adjusted_params['use_scale']:
                        scale = adjusted_params['scale']
                        logger.info(f"🎯 Strategy using scale: {scale}m/pixel (~{adjusted_params['estimated_pixels']}×{adjusted_params['estimated_pixels']}px)")
                    else:
                        dimensions = adjusted_params['dimensions']
                        scale = None
                        logger.info(f"🎯 Strategy using dimensions: {dimensions}×{dimensions}px")
                elif export_strategy.get('target_pixels'):
                    dimensions = export_strategy['target_pixels']
                    scale = None
                    
                    # Check if dimensions are too large for this region
                    estimated_size = estimate_request_size(opt_width_km, opt_height_km, None, dimensions)
                    if estimated_size > 48 * 1024 * 1024:  # 48MB limit
                        # Adjust dimensions to fit within limit
                        adjusted_params = adjust_parameters_for_size_limit(
                            opt_width_km, opt_height_km,
                            initial_dimensions=dimensions,
                            target_min_pixels=1726,
                            use_uint8=True  # Using uint8 visualization
                        )
                        dimensions = adjusted_params['dimensions']
                        
                        # Apply city-specific limits
                        cw, ch = cap_dimensions(dimensions, dimensions, region_name)
                        if (cw, ch) != (dimensions, dimensions):
                            logger.warning(f"🏙️ City-specific limit: Capped {region_name} dimensions to {cw}×{ch}px")
                        # choose smaller to keep square
                        dimensions = min(cw, ch)
                        adjusted_params['dimensions'] = dimensions
                        adjusted_params['estimated_pixels'] = dimensions
                        
                        logger.info(f"🎯 Adjusted dimensions to: {dimensions}×{dimensions}px (estimated size: {adjusted_params['estimated_size']/1024/1024:.1f}MB)")
            
            # Get quality settings for this request
            quality_settings = get_quality_settings(quality)
            collection_limit = quality_settings['collection_limit']
            rgb_stats_scale = quality_settings['rgb_stats_scale']
            
            # STRATEGY 1: Try recent data with low cloud cover (using quality-based limit)
            logger.info(f"🔍 Strategy 1: Fetching recent cloud-free Sentinel-2 imagery for {region_name}...")
            collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                         .filterBounds(region)
                         .filterDate('2023-01-01', '2025-12-31')
                         .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
                         .sort('CLOUDY_PIXEL_PERCENTAGE')
                         .limit(collection_limit))  # Use quality-based limit
            
            # Check if we have images
            count = collection.size().getInfo()
            logger.info(f"Using top {count} images with <20% cloud cover (limited for performance)")
            
            if count == 0:
                # STRATEGY 2: Relax cloud cover threshold (limited to 30 images)
                logger.warning(f"⚠️ Strategy 2: No clear images, trying with <50% cloud cover...")
                collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                             .filterBounds(region)
                             .filterDate('2023-01-01', '2025-12-31')
                             .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 50))
                             .sort('CLOUDY_PIXEL_PERCENTAGE')
                             .limit(collection_limit))  # Use quality-based limit
                count = collection.size().getInfo()
                logger.info(f"Using top {count} images with <50% cloud cover (limited for performance)")
            
            if count == 0:
                # STRATEGY 3: Use ALL available data (any cloud cover) and create median composite (limited to 50)
                logger.warning(f"⚠️ Strategy 3: Using median composite from all available images...")
                collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                             .filterBounds(region)
                             .filterDate('2022-01-01', '2025-12-31')
                             .sort('system:time_start', False)  # Most recent first
                             .limit(max(collection_limit, 6)))  # Use quality-based limit with minimum
                count = collection.size().getInfo()
                logger.info(f"Using {count} most recent images (limited for performance)")
            
            if count == 0:
                logger.error(f"❌ NO SENTINEL-2 DATA AVAILABLE for {region_name}!")
                if strategy_idx < len(export_strategies) - 1:
                    logger.info(f"Trying next export strategy...")
                    continue
                else:
                    return None
            
            # Apply cloud mask to all images in collection
            logger.info(f"✅ Applying cloud mask to {count} images...")
            collection = collection.map(mask_s2_clouds)
            
            # Use MEDIAN composite to remove clouds automatically
            logger.info(f"✅ Creating cloud-free median composite from {count} images...")
            sentinel = collection.median()
            
            # Get RGB bands for true color
            rgb = sentinel.select(['B4', 'B3', 'B2'])
            
            # Calculate high-res NDVI from Sentinel-2
            nir = sentinel.select('B8')  # Near infrared
            red = sentinel.select('B4')  # Red
            ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI')
            
            # IMPROVED VISUALIZATION: Auto-stretch for optimal brightness
            # Get percentile values for better contrast using quality-based scale
            rgb_stats = rgb.reduceRegion(
                reducer=ee.Reducer.percentile([2, 98]),
                geometry=region,
                scale=rgb_stats_scale,  # Use quality-based scale for computation
                maxPixels=1e7  # Reduced maxPixels for faster computation
            ).getInfo()
            
            # Use percentile values or defaults for visualization
            min_vals = [
                rgb_stats.get('B4_p2', 0),
                rgb_stats.get('B3_p2', 0),
                rgb_stats.get('B2_p2', 0)
            ]
            max_vals = [
                rgb_stats.get('B4_p98', 3000),
                rgb_stats.get('B3_p98', 3000),
                rgb_stats.get('B2_p98', 3000)
            ]
            
            logger.info(f"📊 RGB stats - Min: {min_vals}, Max: {max_vals}")
            
            # Use the adjusted parameters calculated above
            # scale and dimensions are already set in the previous section
            
            # Visualize RGB with optimized brightness
            # Use uint8 visualization to reduce payload size (4 bytes -> 1 byte per channel)
            rgb_viz = rgb.visualize(
                min=min_vals,
                max=max_vals,
                gamma=[1.4, 1.3, 1.2]  # Slightly boost brightness, especially in red channel
            ).uint8()  # Convert to uint8 to reduce payload size by 4x
            
            # Also convert NDVI to uint8 for consistent payload size
            ndvi_viz = ndvi.visualize(
                palette=['000000', 'ffffff'],
                min=-0.2,
                max=0.9
            ).uint8()  # Convert to uint8 to reduce payload size by 4x
            
            # Prepare export parameters based on strategy
            if 'scale' in locals() and scale is not None:
                # Use scale-based export
                rgb_params = {
                    'region': region,
                    'scale': scale,
                    'format': 'png'
                }
                # NDVI is already visualized with palette, no need to specify again
                ndvi_params = {
                    'region': region,
                    'scale': scale,
                    'format': 'png'
                }
                logger.info(f"📤 Exporting with scale: {scale}m/pixel (uint8 visualization)")
            else:  # dimensions-based
                rgb_params = {
                    'region': region,
                    'dimensions': dimensions,
                    'format': 'png'
                }
                # NDVI is already visualized with palette, no need to specify again
                ndvi_params = {
                    'region': region,
                    'dimensions': dimensions,
                    'format': 'png'
                }
                logger.info(f"📤 Exporting with dimensions: {dimensions}×{dimensions}px (uint8 visualization)")
            
            # Get the image URLs
            rgb_url = rgb_viz.getThumbURL(rgb_params)
            ndvi_url = ndvi_viz.getThumbURL(ndvi_params)
            
            # Calculate dimensions for the optimized bbox
            opt_width = optimized_bbox[2] - optimized_bbox[0]
            opt_height = optimized_bbox[3] - optimized_bbox[1]
            opt_width_km = opt_width * 111
            opt_height_km = opt_height * 111
            
            # Estimate output dimensions for logging
            if 'scale' in locals() and scale is not None:
                estimated_width = int(opt_width_km * 1000 / scale)
                estimated_height = int(opt_height_km * 1000 / scale)
                logger.info(f"🎯 Export strategy '{export_strategy['name']}': {scale}m/pixel (~{estimated_width}×{estimated_height}px)")
            else:
                estimated_width = estimated_height = dimensions
                logger.info(f"🎯 Export strategy '{export_strategy['name']}': {dimensions}×{dimensions}px")
            
            logger.info(f"✅ Generated Sentinel-2 URLs for {region_name}")
            
            # Create session with retry logic
            session = make_requests_session()
            
            # Download RGB using improved streaming method
            logger.info(f"📥 Downloading RGB satellite image (strategy: {export_strategy['name']})...")
            rgb_array = download_image_stream(rgb_url, session)
            
            if rgb_array is None:
                logger.warning(f"⚠️ RGB download failed with strategy {export_strategy['name']}")
                
                # For HTTP 400 errors (likely size limit), try server-side export as last resort
                if strategy_idx == len(export_strategies) - 1:
                    logger.info(f"🚀 Attempting server-side export as last resort for {region_name}...")
                    
                    # Use the last processed image for export
                    if 'rgb_viz' in locals():
                        # Try Drive export for RGB
                        drive_result = export_to_drive_and_download(
                            rgb_viz, region, f"RGB_{export_strategy['name']}",
                            scale if 'scale' in locals() else 10, region_name
                        )
                        
                        if drive_result is not None:
                            rgb_array = drive_result
                            logger.info(f"✅ Server-side export succeeded for {region_name}")
                        else:
                            logger.error("❌ Server-side export also failed")
                            return None
                
                if strategy_idx < len(export_strategies) - 1:
                    logger.info(f"Trying next export strategy...")
                    continue
                else:
                    logger.error("❌ All export strategies failed for RGB")
                    return None
            
            # Download NDVI using improved streaming method
            logger.info(f"📥 Downloading NDVI data (strategy: {export_strategy['name']})...")
            ndvi_response_array = download_image_stream(ndvi_url, session)
            
            if ndvi_response_array is not None:
                # Convert to grayscale and normalize
                if len(ndvi_response_array.shape) == 3:
                    ndvi_gray = np.array(Image.fromarray(ndvi_response_array).convert('L'))
                else:
                    ndvi_gray = ndvi_response_array
                
                # Normalize to 0-1 range
                ndvi_array = ndvi_gray.astype(float) / 255.0
                logger.info(f"✅ NDVI processed: {ndvi_array.shape}, min={ndvi_array.min():.3f}, max={ndvi_array.max():.3f}")
            else:
                logger.warning(f"⚠️ Failed to download NDVI, using estimated values from RGB")
                # Fallback: estimate NDVI from RGB green channel
                if rgb_array is not None and len(rgb_array.shape) == 3:
                    green = rgb_array[:,:,1].astype(float) / 255.0
                    ndvi_array = np.clip(green * 0.8, 0, 1)  # Rough approximation
                else:
                    ndvi_array = None
            
            if ndvi_array is None:
                # For HTTP 400 errors, try server-side export for NDVI as last resort
                if strategy_idx == len(export_strategies) - 1 and 'ndvi_viz' in locals():
                    logger.info(f"🚀 Attempting server-side NDVI export as last resort for {region_name}...")
                    
                    drive_result = export_to_drive_and_download(
                        ndvi_viz, region, f"NDVI_{export_strategy['name']}",
                        scale if 'scale' in locals() else 10, region_name
                    )
                    
                    if drive_result is not None:
                        # Convert to grayscale if needed
                        if len(drive_result.shape) == 3:
                            ndvi_gray = np.array(Image.fromarray(drive_result).convert('L'))
                            ndvi_array = ndvi_gray.astype(float) / 255.0
                        else:
                            ndvi_array = drive_result.astype(float) / 255.0
                        logger.info(f"✅ Server-side NDVI export succeeded for {region_name}")
                    else:
                        logger.warning("⚠️ Server-side NDVI export failed, using estimation")
                
                # Final fallback: estimate NDVI from RGB
                if ndvi_array is None:
                    logger.warning("⚠️ Using fallback NDVI estimation")
                    if len(rgb_array.shape) == 3:
                        green = rgb_array[:,:,1].astype(float) / 255.0
                        ndvi_array = np.clip(green * 0.8, 0, 1)
                    else:
                        logger.error("❌ Cannot estimate NDVI from grayscale image")
                        return None
            
            # If we got here, the strategy worked!
            logger.info(f"✅✅ Strategy '{export_strategy['name']}' succeeded for {region_name}!")
            logger.info(f"   RGB: {rgb_array.shape}, NDVI: {ndvi_array.shape}")
            
            # Prepare metadata
            height = rgb_array.shape[0]
            width = rgb_array.shape[1]
            metadata = {
                'width': width,
                'height': height,
                'visualization_used': 'uint8',
                'strategy_used': export_strategy['name'],
                'region_type': 'state' if region_name in INDIAN_STATES else 'city'
            }
            
            # Return the result in the expected format
            return {
                'rgb': rgb_array,
                'ndvi': ndvi_array,
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"❌ Strategy '{export_strategy['name']}' failed for {region_name}: {e}")
            if strategy_idx < len(export_strategies) - 1:
                logger.info(f"Trying next export strategy...")
                time.sleep(2)  # Brief pause before trying next strategy
                continue
            else:
                logger.error(f"❌ All export strategies failed for {region_name}")
                return None
    
    # This should never be reached, but just in case
    logger.error(f"❌ Unexpected error in fetch_high_res_satellite_and_ndvi for {region_name}")
    return None


def export_to_drive_and_download(image: ee.Image, region: ee.Geometry, filename: str,
                                scale: int, city_name: str) -> Optional[np.ndarray]:
    """
    Export image to Google Drive and download it as a fallback for large requests
    
    Args:
        image: Earth Engine image to export
        region: Region geometry
        filename: Filename for the export
        scale: Scale in meters per pixel
        city_name: Name of the city (for logging)
        
    Returns:
        Downloaded image as numpy array or None
    """
    try:
        logger.info(f"🚀 Starting server-side export to Drive for {city_name}...")
        
        # Create export task
        task = ee.batch.Export.image.toDrive(
            image=image,
            description=f'{city_name}_{filename}',
            folder='biomass_exports',
            fileNamePrefix=f'{city_name}_{filename}',
            region=region,
            scale=scale,
            maxPixels=1e13,
            fileFormat='GeoTIFF'
        )
        
        # Start the task
        task.start()
        logger.info(f"✅ Export task started for {city_name}")
        
        # Poll for completion (simplified - in production, you'd want async handling)
        import time
        max_wait_time = 300  # 5 minutes max wait
        wait_interval = 10   # Check every 10 seconds
        elapsed_time = 0
        
        while elapsed_time < max_wait_time:
            status = task.status()
            state = status.get('state')
            
            if state == 'COMPLETED':
                logger.info(f"✅ Export completed for {city_name}")
                # In a real implementation, you would download from Drive here
                # For now, return None as this is a fallback mechanism
                logger.warning(f"⚠️ Drive export completed but download not implemented - using fallback")
                return None
            elif state == 'FAILED':
                error_message = status.get('error_message', 'Unknown error')
                logger.error(f"❌ Export failed for {city_name}: {error_message}")
                return None
            elif state in ['RUNNING', 'READY']:
                logger.info(f"⏳ Export in progress for {city_name} ({elapsed_time}s elapsed)...")
                time.sleep(wait_interval)
                elapsed_time += wait_interval
            else:
                logger.warning(f"⚠️ Unknown export state for {city_name}: {state}")
                time.sleep(wait_interval)
                elapsed_time += wait_interval
        
        logger.error(f"❌ Export timed out for {city_name} after {max_wait_time}s")
        return None
        
    except Exception as e:
        logger.error(f"❌ Server-side export failed for {city_name}: {e}")
        return None


def download_image_stream(url: str, session: requests.Session, max_retries: int = 3, base_timeout: int = 120) -> Optional[np.ndarray]:
    """
    Download satellite image from URL using streaming with retry logic
    
    Args:
        url: Image URL to download
        session: Requests session with retry configuration
        max_retries: Maximum number of retry attempts
        base_timeout: Base timeout in seconds
        
    Returns:
        numpy array of the image or None if failed
    """
    for attempt in range(max_retries):
        try:
            timeout = base_timeout + (attempt * 30)
            logger.info(f"Attempt {attempt+1}/{max_retries} downloading with timeout {timeout}s")
            
            resp = session.get(url, timeout=timeout, stream=True)
            if resp.status_code == 200:
                buf = io.BytesIO()
                # Stream chunks to buffer to avoid memory spikes
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        buf.write(chunk)
                buf.seek(0)
                
                img = Image.open(buf)
                arr = np.array(img)
                
                # Basic validation
                if arr.size == 0 or (arr.max() == 0 and arr.min() == 0):
                    raise ValueError("Downloaded image is empty or all black")
                
                logger.info(f"✅ Downloaded: {arr.shape}, min={arr.min()}, max={arr.max()}")
                return arr
            else:
                logger.warning(f"Download status {resp.status_code}")
                
        except requests.exceptions.Timeout:
            logger.warning(f"⏰ Timeout on attempt {attempt+1}")
            if attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))
        except Exception as e:
            logger.warning(f"Error on attempt {attempt+1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))
    
    return None


def download_satellite_image(url: str) -> Optional[np.ndarray]:
    """
    Download satellite image from URL (legacy function for compatibility)
    
    Returns:
        numpy array of the image
    """
    try:
        session = make_requests_session()
        return download_image_stream(url, session)
    except Exception as e:
        logger.error(f"Error downloading satellite image: {e}")
        return None


def generate_satellite_heatmap(
    city_name: str,
    region_name: str,
    ndvi_array: np.ndarray,
    agb_data: Dict,
    bbox: tuple,
    use_real_satellite: bool = True,
    quality: str = 'balanced'
) -> str:
    """
    Generate HIGH-QUALITY professional heatmap using REAL Sentinel-2 satellite imagery
    
    Args:
        city_name: Name of the city
        region_name: Specific region being analyzed
        ndvi_array: NDVI data array (IGNORED - we fetch high-res from GEE)
        agb_data: Biomass data dictionary
        bbox: Bounding box (min_lon, min_lat, max_lon, max_lat)
        use_real_satellite: If True, fetch real satellite image as background
        
    Returns:
        Path to saved heatmap image
    """
    from datetime import datetime
    
    # Get quality settings early for consistent usage throughout function
    quality_settings = get_quality_settings(quality)
    interp = quality_settings.get('interpolation', 'nearest')
    
    # Safety fallback for interpolation
    if not interp or interp not in ['nearest', 'bilinear', 'lanczos', 'bicubic']:
        interp = 'nearest'  # Safe default
    
    # Create output directory
    output_dir = Path("./outputs/heatmaps")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"biomass_heatmap_{city_name.lower().replace(' ', '_')}_{timestamp}.png"
    filepath = output_dir / filename
    
    # Create HIGH-RESOLUTION figure with TWO PANELS (side-by-side)
    fig = plt.figure(figsize=(32, 14), dpi=200)  # Wide format for side-by-side
    
    # Create 1 row, 2 columns of subplots
    ax1 = plt.subplot(121)  # Left panel - Raw satellite
    ax2 = plt.subplot(122)  # Right panel - Biomass analysis
    
    # Try to fetch HIGH-RES satellite imagery AND NDVI
    satellite_rgb = None
    ndvi_data = None
    
    if use_real_satellite:
        logger.info(f"🛰️ Fetching HIGH-RES Sentinel-2 imagery for {city_name}...")
        try:
            # Initialize GEE with project ID
            try:
                ee.Initialize(project='ee-lanbprojectclassification')
                logger.info("✅ GEE initialized successfully")
            except Exception as e:
                logger.warning(f"GEE init failed: {e}")
                use_real_satellite = False
            
            if use_real_satellite:
                # Fetch RGB satellite image and NDVI data array
                result = fetch_high_res_satellite_and_ndvi(bbox, city_name, quality=quality)
                
                if result is not None:
                    satellite_rgb = result['rgb']
                    ndvi_data = result['ndvi']
                    logger.info(f"✅ Successfully fetched satellite imagery: {satellite_rgb.shape}")
                else:
                    logger.warning("Could not get satellite image")
                    use_real_satellite = False
        except Exception as e:
            logger.warning(f"Could not fetch satellite images: {e}")
            use_real_satellite = False
    
    # Use HIGH-RES imagery if available
    if satellite_rgb is not None and ndvi_data is not None:
        logger.info("🎨 Creating side-by-side comparison: Raw Satellite vs Biomass Analysis...")
        
        # Compute extent once: (left, right, bottom, top) for geospatial alignment
        extent = [bbox[0], bbox[2], bbox[1], bbox[3]]
        
        # Tighten figure layout — slightly less vertical size to avoid blank space
        fig.set_size_inches(28, 12)
        
        # ===== LEFT PANEL: RAW SATELLITE IMAGE =====
        ax1.imshow(satellite_rgb, extent=extent, origin='lower',
                  aspect='auto', interpolation=interp, zorder=1)
        ax1.set_xlim(extent[0], extent[1])
        ax1.set_ylim(extent[2], extent[3])
        ax1.set_xlabel('Longitude (°E)', fontsize=13, weight='bold')
        ax1.set_ylabel('Latitude (°N)', fontsize=13, weight='bold')
        ax1.set_title('Raw Sentinel-2 Satellite Image\n(True Color RGB)', 
                     fontsize=16, weight='bold', pad=15)
        ax1.grid(False)  # Remove grid to reduce clutter
        
        # Light watermark (less intrusive)
        ax1.text(0.98, 0.02, 'RAW', transform=ax1.transAxes,
                fontsize=14, color='white', ha='right', va='bottom', 
                alpha=0.12, weight='bold')
        
        # ===== RIGHT PANEL: BIOMASS ANALYSIS =====
        # 1. Display satellite as base
        ax2.imshow(satellite_rgb, extent=extent, origin='lower',
                  aspect='auto', interpolation=interp, zorder=1)
        ax2.set_xlim(extent[0], extent[1])
        ax2.set_ylim(extent[2], extent[3])
        
        # 2. Create biomass map from NDVI
        biomass_map = ndvi_data * agb_data['total_agb'] * 2.0
        biomass_map = np.power(biomass_map, 1.2)  # Contrast enhancement
        
        # 3. Create VIBRANT color-coded biomass overlay with improved transparency
        colors = [
            (0.4, 0.2, 0.1, 0.8),   # Dark Brown - Buildings/Urban
            (0.7, 0.5, 0.3, 0.8),   # Tan - Bare soil/Roads
            (0.9, 0.8, 0.3, 0.8),   # Yellow - Sparse vegetation/Grass
            (0.6, 0.9, 0.2, 0.85),  # Yellow-Green - Shrubs/Gardens  
            (0.3, 0.9, 0.3, 0.9),   # Bright Green - Trees/Parks
            (0.1, 0.7, 0.1, 0.95),  # Forest Green - Dense vegetation
            (0.0, 0.5, 0.0, 0.98),  # Dark Green - Very dense forest
        ]
        cmap = LinearSegmentedColormap.from_list('agb_overlay', colors, N=256)
        
        # 4. Overlay biomass colors with transparency to show satellite details
        im = ax2.imshow(biomass_map, extent=extent, origin='lower',
                       cmap=cmap, alpha=0.85, vmin=0, vmax=120,
                       aspect='auto', interpolation=interp, zorder=2)
        
        ax2.set_xlabel('Longitude (°E)', fontsize=13, weight='bold')
        ax2.set_ylabel('Latitude (°N)', fontsize=13, weight='bold')
        ax2.set_title('Above Ground Biomass Analysis\n(Biomass Distribution Overlay)', 
                     fontsize=16, weight='bold', pad=15)
        ax2.grid(False)  # Remove grid to reduce clutter
        
        # Nicer colorbar placement using divider
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes("right", size="3%", pad=0.06)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label('Above Ground Biomass (Mg/ha)', fontsize=12, weight='bold')
        cbar.ax.tick_params(labelsize=10)
        
    else:
        # Fallback: Use low-res synthetic map (single panel)
        logger.warning("Using fallback low-res heatmap")
        ax1.remove()  # Remove left panel
        ax2.remove()  # Remove right panel
        ax = plt.subplot(111)  # Use full figure
        
        agb_map = ndvi_array * agb_data['total_agb']
        
        # Add spatial variation
        if agb_map.size > 1:
            y, x = np.ogrid[:agb_map.shape[0], :agb_map.shape[1]]
            center_y, center_x = agb_map.shape[0] // 2, agb_map.shape[1] // 2
            distance_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            max_distance = np.sqrt(center_x**2 + center_y**2)
            urban_gradient = 0.6 + (distance_from_center / max_distance) * 0.8
            agb_map *= urban_gradient
        
        colors = [
            '#8B4513',  '#DEB887',  '#F0E68C',  '#9ACD32',
            '#90EE90',  '#32CD32',  '#228B22',  '#006400',
        ]
        cmap = LinearSegmentedColormap.from_list('agb_fallback', colors, N=256)
        im = ax.imshow(agb_map, extent=[bbox[0], bbox[2], bbox[1], bbox[3]], 
                      cmap=cmap, vmin=0, vmax=150, aspect='auto', interpolation='bilinear')
        
        ax.set_xlabel('Longitude (°E)', fontsize=13, weight='bold')
        ax.set_ylabel('Latitude (°N)', fontsize=13, weight='bold')
        ax.set_title(f'{city_name} - {region_name}\nAbove Ground Biomass Distribution', 
                    fontsize=16, weight='bold', pad=15)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Determine which axis to use for overlays
    main_ax = ax2 if satellite_rgb is not None else ax
    
    # Add city boundary overlay on BOTH panels if side-by-side
    try:
        from app.api.prediction import get_city_boundary
        boundary = get_city_boundary(city_name, bbox)
        if boundary is not None and boundary.size > 0:
            # The boundary is a numpy array with (lon, lat) pairs
            lons = boundary[:, 0]
            lats = boundary[:, 1]
            
            # Add to right panel (or single panel)
            main_ax.plot(lons, lats, color='white', linewidth=2.5, alpha=0.9, zorder=10)
            main_ax.plot(lons, lats, color='black', linewidth=1, alpha=0.6, zorder=9)
            
            # Add to left panel if side-by-side
            if satellite_rgb is not None:
                ax1.plot(lons, lats, color='white', linewidth=2.5, alpha=0.9, zorder=10)
                ax1.plot(lons, lats, color='black', linewidth=1, alpha=0.6, zorder=9)
    except Exception as e:
        logger.warning(f"Could not add city boundary: {e}")
    

    
    # Add statistics box on right panel
    stats_text = (
        f"Region Statistics:\n"
        f"Total AGB: {agb_data['total_agb']:.1f} Mg/ha\n"
        f"Canopy Cover: {agb_data['canopy_cover']:.1f}%\n"
        f"Tree Biomass: {agb_data.get('tree_biomass', 0):.1f} Mg/ha\n"
        f"Resolution: {'Sentinel-2 (10m)' if satellite_rgb is not None else 'MODIS (1km)'}"
    )
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    main_ax.text(0.02, 0.98, stats_text, transform=main_ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props, family='monospace')
    
    # Add legend for land cover types on right panel
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#008000', label='🌲 Dense Forest/Trees (>80 Mg/ha)'),
        Patch(facecolor='#4DE94D', label='🌳 Parks/Urban Forest (50-80 Mg/ha)'),
        Patch(facecolor='#9AE832', label='🌿 Shrubs/Gardens (30-50 Mg/ha)'),
        Patch(facecolor='#E6CC4D', label='🌾 Grasslands (10-30 Mg/ha)'),
        Patch(facecolor='#8B4513', label='🏢 Buildings/Roads (<10 Mg/ha)'),
    ]
    main_ax.legend(handles=legend_elements, loc='lower right', fontsize=10, 
                  framealpha=0.95, edgecolor='gray', title='Biomass Classification',
                  title_fontsize=11)
    
    # Add scale bar on right panel
    from matplotlib_scalebar.scalebar import ScaleBar
    try:
        scalebar = ScaleBar(111320, location='lower left', frameon=True, 
                           color='black', box_color='white', box_alpha=0.8)
        main_ax.add_artist(scalebar)
    except:
        pass
    
    # Add north arrow on right panel
    main_ax.annotate('N', xy=(0.95, 0.95), xytext=(0.95, 0.90),
                    xycoords='axes fraction', fontsize=20, weight='bold',
                    ha='center', va='center',
                    arrowprops=dict(arrowstyle='->', lw=2.5, color='black'))
    
    # Add north arrow on left panel too if side-by-side
    if satellite_rgb is not None:
        ax1.annotate('N', xy=(0.95, 0.95), xytext=(0.95, 0.90),
                    xycoords='axes fraction', fontsize=20, weight='bold',
                    ha='center', va='center',
                    arrowprops=dict(arrowstyle='->', lw=2.5, color='black'))
    
    # Add overall figure title for side-by-side view
    if satellite_rgb is not None:
        fig.suptitle(
            f'🛰️ {city_name} - {region_name}\nSatellite Imagery vs Biomass Analysis Comparison',
            fontsize=20, weight='bold', y=0.98
        )
    
    # Add data source footer
    footer_text = (
        f"Data Sources: Sentinel-2 (ESA Copernicus) | GEDI L4A (NASA) | "
        f"Methodology: Kumar et al. (2021) | "
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )
    fig.text(0.5, 0.01, footer_text, ha='center', fontsize=10, 
            style='italic', color='gray', weight='bold')
    
    # Tight layout (adjust for suptitle if side-by-side)
    plt.tight_layout(rect=[0, 0.02, 1, 0.96] if satellite_rgb is not None else [0, 0.02, 1, 1])
    
    # Save with enhanced quality settings
    plt.savefig(filepath, 
                dpi=quality_settings['save_dpi'], 
                bbox_inches='tight', 
                facecolor='white',
                pad_inches=0.1)
    plt.close(fig)
    
    logger.info(f"✅ {'Side-by-side' if satellite_rgb is not None else 'Single'} heatmap saved: {filepath}")
    
    # Return the URL path for the frontend
    return f"/outputs/heatmaps/{filename}"


# Add missing import
from datetime import datetime


# Add missing import
from datetime import datetime

# Simple test function for development
if __name__ == "__main__":
    """
    Simple test harness for development and debugging
    Run with: python -m app.api.satellite_image_generator
    """
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Test bbox for Hyderabad (small area for quick testing)
    test_bbox = (78.45, 17.35, 78.55, 17.45)  # ~11km x 11km
    test_city = "Hyderabad"
    
    # Mock AGB data for testing
    test_agb_data = {
        'total_agb': 85.3,
        'canopy_cover': 42.7,
        'tree_biomass': 55.4,
        'cooling_potential': 2.1
    }
    
    # Mock NDVI array (small for testing)
    test_ndvi = np.random.rand(100, 100) * 0.8 + 0.1  # Random NDVI 0.1-0.9
    
    print("🧪 Testing satellite image generator...")
    print(f"📍 Test area: {test_city}")
    print(f"📏 Bbox: {test_bbox}")
    print(f"🎯 Quality: balanced")
    
    try:
        # Test the main function
        result_path = generate_satellite_heatmap(
            city_name=test_city,
            region_name="Test Region",
            ndvi_array=test_ndvi,
            agb_data=test_agb_data,
            bbox=test_bbox,
            use_real_satellite=True,
            quality='balanced'
        )
        
        print(f"✅ Test completed successfully!")
        print(f"📁 Generated heatmap: {result_path}")
        print(f"🔍 Check the file at: ./outputs/heatmaps/")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()