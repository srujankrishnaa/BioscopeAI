#!/usr/bin/env python3
"""
City Region Satellite Image Cache Generator
Pre-generates satellite images for all major Indian cities and their 5 regions

This script runs offline to create a cache of satellite images that can be
served instantly to users, eliminating timeout issues.
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple
import json

# Import our existing modules
from app.models.gee_data_fetcher import GEEDataFetcher
from app.api.satellite_image_generator import fetch_high_res_satellite_and_ndvi
from app.api.region_selection import calculate_city_regions, get_fallback_city_bbox

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cache_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CityRegionCacheGenerator:
    """Generates and manages cached satellite images for city regions"""
    
    def __init__(self):
        self.gee_fetcher = GEEDataFetcher()
        self.cache_dir = Path("./outputs/region_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 28 Indian State Capitals + 8 Union Territory Capitals (ONLY)
        self.state_capitals_to_cache = [
            # 28 State Capitals
            'Mumbai',           # Maharashtra
            'Bangalore',        # Karnataka  
            'Chennai',          # Tamil Nadu
            'Hyderabad',        # Telangana
            'Kolkata',          # West Bengal
            'Ahmedabad',        # Gujarat (commercial capital)
            'Gandhinagar',      # Gujarat (official capital)
            'Jaipur',           # Rajasthan
            'Lucknow',          # Uttar Pradesh
            'Bhopal',           # Madhya Pradesh
            'Patna',            # Bihar
            'Thiruvananthapuram', # Kerala
            'Bhubaneswar',      # Odisha
            'Ranchi',           # Jharkhand
            'Raipur',           # Chhattisgarh
            'Panaji',           # Goa
            'Shimla',           # Himachal Pradesh
            'Srinagar',         # Jammu & Kashmir (summer capital)
            'Jammu',            # Jammu & Kashmir (winter capital)
            'Guwahati',         # Assam (largest city)
            'Dispur',           # Assam (official capital)
            'Agartala',         # Tripura
            'Aizawl',           # Mizoram
            'Imphal',           # Manipur
            'Kohima',           # Nagaland
            'Itanagar',         # Arunachal Pradesh
            'Gangtok',          # Sikkim
            'Shillong',         # Meghalaya
            
            # 8 Union Territory Capitals
            'Delhi',            # Delhi
            'Chandigarh',       # Chandigarh & Punjab & Haryana
            'Puducherry',       # Puducherry
            'Port Blair',       # Andaman & Nicobar
            'Kavaratti',        # Lakshadweep
            'Daman',            # Daman & Diu
            'Silvassa',         # Dadra & Nagar Haveli
            'Ladakh'            # Ladakh (Leh)
        ]
        
        # Cache metadata
        self.cache_metadata = {}
        self.metadata_file = self.cache_dir / "cache_metadata.json"
    
    def load_existing_metadata(self):
        """Load existing cache metadata"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    self.cache_metadata = json.load(f)
                logger.info(f"Loaded existing metadata for {len(self.cache_metadata)} cities")
            except Exception as e:
                logger.error(f"Failed to load metadata: {e}")
                self.cache_metadata = {}
    
    def save_metadata(self):
        """Save cache metadata"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.cache_metadata, f, indent=2)
            logger.info("Cache metadata saved successfully")
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    def get_city_bbox(self, city_name: str) -> Tuple[float, float, float, float]:
        """Get city bounding box with corrected coordinates for problematic cities"""
        
        # VALIDATED corrected bounding boxes for cities with ocean/water issues
        corrected_bboxes = {
            'Daman': (72.8200, 20.3900, 72.8600, 20.4300),      # ✅ Correct - mainland focus
            'Kavaratti': (72.6300, 10.5500, 72.6500, 10.5700),  # ✅ Improved - includes reef boundary
            'Panaji': (73.8150, 15.4850, 73.8450, 15.5250),     # ✅ Improved - better positioning
            'Port Blair': (92.7200, 11.6500, 92.7800, 11.7100), # ✅ Correct - balanced coverage
            'Puducherry': (79.7900, 11.8900, 79.8300, 11.9300), # ✅ Improved - shifted westward
            'Chennai': (80.1600, 13.0000, 80.2400, 13.0800),    # ✅ Improved - less Bay of Bengal
            'Thiruvananthapuram': (76.9050, 8.4850, 76.9950, 8.5750), # ✅ Improved - recentered, reduced southern periphery
            'Gandhinagar': (72.6000, 23.1500, 72.7000, 23.3000), # ✅ Improved - full region coverage
            'Visakhapatnam': (83.2000, 17.6500, 83.3500, 17.7500) # ✅ Added - coastal city with proper land focus
        }
        
        # Use corrected bbox if available
        if city_name in corrected_bboxes:
            logger.info(f"Using corrected bbox for {city_name} (ocean/water issues fixed)")
            return corrected_bboxes[city_name]
        
        # Try GEE fetcher first
        bbox = self.gee_fetcher.get_city_bbox(city_name)
        if bbox:
            return bbox
        
        # Use fallback coordinates
        bbox = get_fallback_city_bbox(city_name)
        if bbox:
            return bbox
        
        raise ValueError(f"Could not find coordinates for {city_name}")
    
    async def generate_region_satellite_image(self, region_bbox: Tuple[float, float, float, float], 
                                            region_name: str, city_name: str) -> str:
        """Generate high-quality satellite image for a region"""
        try:
            logger.info(f"Generating satellite image for {region_name}")
            
            # Create city-specific cache directory
            city_cache_dir = self.cache_dir / city_name.replace(' ', '_')
            city_cache_dir.mkdir(exist_ok=True)
            
            # Generate safe filename
            safe_region_name = region_name.replace(' ', '_').replace('-', '_')
            filename = f"{safe_region_name}_satellite.png"
            filepath = city_cache_dir / filename
            
            # Skip if already exists
            if filepath.exists():
                logger.info(f"Satellite image already exists: {filename}")
                return str(filepath.relative_to(self.cache_dir))
            
            # Use the existing high-quality satellite image generator
            satellite_data = fetch_high_res_satellite_and_ndvi(
                region_bbox,
                region_name,
                quality='high'  # Use high quality for cache
            )
            
            if satellite_data and satellite_data.get('rgb') is not None:
                import matplotlib.pyplot as plt
                
                # Save high-quality RGB satellite image
                plt.figure(figsize=(12, 10), dpi=200)  # High resolution
                plt.imshow(satellite_data['rgb'])
                plt.axis('off')
                plt.title(f'{region_name} - Satellite View', 
                         fontsize=16, fontweight='bold', pad=20)
                
                # Add metadata
                plt.figtext(0.02, 0.02, 
                           f'Source: Sentinel-2 via Google Earth Engine | Generated: {time.strftime("%Y-%m-%d %H:%M")}',
                           fontsize=10, style='italic', alpha=0.7)
                
                plt.tight_layout()
                plt.savefig(filepath, bbox_inches='tight', dpi=200, 
                           facecolor='white', edgecolor='none')
                plt.close()
                
                logger.info(f"High-quality satellite image saved: {filename}")
                return str(filepath.relative_to(self.cache_dir))
            
            else:
                logger.warning(f"No satellite data available for {region_name}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to generate satellite image for {region_name}: {e}")
            return None
    
    async def cache_city_regions(self, city_name: str):
        """Cache all 5 regions for a city"""
        try:
            logger.info(f"Starting cache generation for {city_name}")
            
            # Get city bounding box
            city_bbox = self.get_city_bbox(city_name)
            logger.info(f"City bbox for {city_name}: {city_bbox}")
            
            # Calculate 5 regions
            regions_data = calculate_city_regions(city_bbox, city_name)
            
            # Initialize city metadata
            city_metadata = {
                'city_name': city_name,
                'city_bbox': list(city_bbox),
                'regions': {},
                'generated_at': time.strftime("%Y-%m-%d %H:%M:%S"),
                'total_regions': len(regions_data)
            }
            
            # Generate satellite image for each region
            for region_data in regions_data:
                region_id = region_data['id']
                region_name = region_data['name']
                region_bbox = tuple(region_data['bbox'])
                
                logger.info(f"Processing region: {region_name}")
                
                # Generate satellite image
                image_path = await self.generate_region_satellite_image(
                    region_bbox, region_name, city_name
                )
                
                # Store region metadata
                city_metadata['regions'][region_id] = {
                    'name': region_name,
                    'description': region_data['description'],
                    'bbox': region_data['bbox'],
                    'coordinates': region_data['coordinates'],
                    'image_path': image_path,
                    'generated_at': time.strftime("%Y-%m-%d %H:%M:%S")
                }
                
                # Small delay to avoid overwhelming GEE
                await asyncio.sleep(2)
            
            # Save city metadata
            self.cache_metadata[city_name] = city_metadata
            self.save_metadata()
            
            logger.info(f"✅ Successfully cached {len(regions_data)} regions for {city_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to cache regions for {city_name}: {e}")
    
    async def generate_all_caches(self):
        """Generate cache for remaining state capitals only"""
        # Load existing metadata first
        self.load_existing_metadata()
        
        # Find which state capitals are already cached
        already_cached = []
        remaining_cities = []
        
        for city in self.state_capitals_to_cache:
            if city in self.cache_metadata:
                already_cached.append(city)
            else:
                remaining_cities.append(city)
        
        logger.info("=" * 80)
        logger.info("🏛️ STATE CAPITALS CACHE STATUS")
        logger.info("=" * 80)
        logger.info(f"Total State Capitals: {len(self.state_capitals_to_cache)}")
        logger.info(f"Already Cached: {len(already_cached)}")
        logger.info(f"Remaining to Download: {len(remaining_cities)}")
        
        if already_cached:
            logger.info(f"\n✅ Already Cached ({len(already_cached)}):")
            for city in already_cached:
                logger.info(f"  - {city}")
        
        if remaining_cities:
            logger.info(f"\n📥 Remaining to Download ({len(remaining_cities)}):")
            for city in remaining_cities:
                logger.info(f"  - {city}")
        else:
            logger.info("\n🎉 All state capitals are already cached!")
            self.print_cache_summary()
            return
        
        logger.info("=" * 80)
        
        # Process only remaining cities
        total_remaining = len(remaining_cities)
        
        for i, city in enumerate(remaining_cities, 1):
            logger.info(f"📍 Processing city {i}/{total_remaining}: {city}")
            
            try:
                await self.cache_city_regions(city)
                logger.info(f"✅ Completed {city} ({i}/{total_remaining})")
                
                # Longer delay between cities to be respectful to GEE
                if i < total_remaining:
                    logger.info("⏳ Waiting 10 seconds before next city...")
                    await asyncio.sleep(10)
                    
            except Exception as e:
                logger.error(f"❌ Failed to process {city}: {e}")
                continue
        
        logger.info("🎉 State capitals cache generation completed!")
        self.print_cache_summary()
    
    def print_cache_summary(self):
        """Print summary of cached data"""
        logger.info("=" * 60)
        logger.info("CACHE GENERATION SUMMARY")
        logger.info("=" * 60)
        
        total_cities = len(self.cache_metadata)
        total_regions = sum(len(city_data['regions']) for city_data in self.cache_metadata.values())
        
        logger.info(f"Total Cities Cached: {total_cities}")
        logger.info(f"Total Regions Cached: {total_regions}")
        logger.info(f"Cache Directory: {self.cache_dir}")
        
        # List cached cities
        logger.info("\nCached Cities:")
        for city_name, city_data in self.cache_metadata.items():
            region_count = len(city_data['regions'])
            logger.info(f"  - {city_name}: {region_count} regions")
        
        logger.info("=" * 60)

async def main():
    """Main function to run cache generation"""
    generator = CityRegionCacheGenerator()
    
    print("🏛️ State Capitals Satellite Image Cache Generator")
    print("=" * 60)
    print("This will generate satellite images for remaining Indian state capitals.")
    print("Only downloads cities that haven't been cached yet.")
    print("=" * 60)
    
    # Ask for confirmation
    response = input("Do you want to start cache generation? (y/N): ")
    if response.lower() != 'y':
        print("Cache generation cancelled.")
        return
    
    # Start generation
    start_time = time.time()
    await generator.generate_all_caches()
    end_time = time.time()
    
    duration = end_time - start_time
    logger.info(f"⏱️ Total time taken: {duration/3600:.2f} hours")

if __name__ == "__main__":
    asyncio.run(main())