#!/usr/bin/env python3
"""
Delete and Regenerate Problematic Cities
Deletes cities with ocean/water issues and regenerates them with validated corrected bounding boxes
"""

import asyncio
import logging
import time
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import json

# Import our existing modules
from app.models.gee_data_fetcher import GEEDataFetcher
from app.api.satellite_image_generator import fetch_high_res_satellite_and_ndvi
from app.api.region_selection import calculate_city_regions

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('problematic_cities_fix.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ProblematicCitiesRegenerator:
    """Deletes and regenerates problematic cities with corrected bounding boxes"""
    
    def __init__(self):
        self.gee_fetcher = GEEDataFetcher()
        self.cache_dir = Path("./outputs/region_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cities with problematic bounding boxes that need regeneration
        self.problematic_cities = [
            'Daman',
            'Kavaratti', 
            'Panaji',
            'Port Blair',
            'Puducherry',
            'Chennai',
            'Thiruvananthapuram',
            'Gandhinagar',
            'Visakhapatnam'  # Added - missing from previous list
        ]
        
        # VALIDATED corrected bounding boxes
        self.corrected_bboxes = {
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
    
    def validate_bbox(self, city_name: str, bbox: Tuple[float, float, float, float]) -> bool:
        """Validate bounding box according to checklist"""
        min_lon, min_lat, max_lon, max_lat = bbox
        
        # Check basic validity
        if min_lon >= max_lon or min_lat >= max_lat:
            logger.error(f"❌ {city_name}: Invalid bbox - min >= max")
            return False
        
        # Check minimum size (not too tight)
        width = max_lon - min_lon
        height = max_lat - min_lat
        
        if width < 0.01 or height < 0.01:
            logger.warning(f"⚠️ {city_name}: Very tight bbox (w={width:.4f}°, h={height:.4f}°)")
        
        # Log dimensions
        area_km2 = width * height * 111 * 111  # Approximate
        logger.info(f"📏 {city_name}: {width:.4f}° x {height:.4f}° ({area_km2:.0f} km²)")
        
        return True
    
    def delete_city_cache(self, city_name: str):
        """Delete existing cache for a city"""
        try:
            # Delete directory
            city_cache_dir = self.cache_dir / city_name.replace(' ', '_')
            if city_cache_dir.exists():
                shutil.rmtree(city_cache_dir)
                logger.info(f"🗑️ Deleted cache directory: {city_cache_dir}")
            
            # Remove from metadata
            if city_name in self.cache_metadata:
                del self.cache_metadata[city_name]
                logger.info(f"🗑️ Removed {city_name} from metadata")
            
        except Exception as e:
            logger.error(f"Failed to delete cache for {city_name}: {e}")
    
    async def generate_region_satellite_image(self, region_bbox: Tuple[float, float, float, float], 
                                            region_name: str, city_name: str) -> str:
        """Generate high-quality satellite image for a region"""
        try:
            logger.info(f"Generating CORRECTED satellite image for {region_name}")
            
            # Create city-specific cache directory
            city_cache_dir = self.cache_dir / city_name.replace(' ', '_')
            city_cache_dir.mkdir(exist_ok=True)
            
            # Generate safe filename
            safe_region_name = region_name.replace(' ', '_').replace('-', '_')
            filename = f"{safe_region_name}_satellite.png"
            filepath = city_cache_dir / filename
            
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
                plt.title(f'{region_name} - Satellite View (CORRECTED BBOX)', 
                         fontsize=16, fontweight='bold', pad=20)
                
                # Add metadata
                plt.figtext(0.02, 0.02, 
                           f'Source: Sentinel-2 via Google Earth Engine | CORRECTED BBOX | Generated: {time.strftime("%Y-%m-%d %H:%M")}',
                           fontsize=10, style='italic', alpha=0.7)
                
                plt.tight_layout()
                plt.savefig(filepath, bbox_inches='tight', dpi=200, 
                           facecolor='white', edgecolor='none')
                plt.close()
                
                logger.info(f"✅ CORRECTED satellite image saved: {filename}")
                return str(filepath.relative_to(self.cache_dir))
            
            else:
                logger.warning(f"No satellite data available for {region_name}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to generate satellite image for {region_name}: {e}")
            return None
    
    async def regenerate_city_regions(self, city_name: str):
        """Regenerate all 5 regions for a city with corrected bbox"""
        try:
            logger.info(f"🔧 REGENERATING: {city_name}")
            
            # Get corrected bounding box
            city_bbox = self.corrected_bboxes[city_name]
            logger.info(f"Using CORRECTED bbox for {city_name}: {city_bbox}")
            
            # Validate bbox
            if not self.validate_bbox(city_name, city_bbox):
                logger.error(f"❌ Invalid bbox for {city_name}, skipping")
                return
            
            # Calculate 5 regions with corrected bbox
            regions_data = calculate_city_regions(city_bbox, city_name)
            
            # Initialize city metadata
            city_metadata = {
                'city_name': city_name,
                'city_bbox': list(city_bbox),
                'regions': {},
                'generated_at': time.strftime("%Y-%m-%d %H:%M:%S"),
                'total_regions': len(regions_data),
                'status': 'CORRECTED_BBOX',
                'bbox_version': 'VALIDATED_V2'
            }
            
            # Generate satellite image for each region
            for region_data in regions_data:
                region_id = region_data['id']
                region_name = region_data['name']
                region_bbox = tuple(region_data['bbox'])
                
                logger.info(f"Processing CORRECTED region: {region_name}")
                
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
                    'generated_at': time.strftime("%Y-%m-%d %H:%M:%S"),
                    'status': 'CORRECTED'
                }
                
                # Small delay to avoid overwhelming GEE
                await asyncio.sleep(3)
            
            # Save city metadata
            self.cache_metadata[city_name] = city_metadata
            self.save_metadata()
            
            logger.info(f"✅ Successfully regenerated {len(regions_data)} regions for {city_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to regenerate regions for {city_name}: {e}")
    
    async def delete_and_regenerate_all(self):
        """Delete and regenerate all problematic cities"""
        # Load existing metadata first
        self.load_existing_metadata()
        
        logger.info("=" * 80)
        logger.info("🔧 DELETE & REGENERATE PROBLEMATIC CITIES")
        logger.info("=" * 80)
        logger.info(f"Cities to fix: {len(self.problematic_cities)}")
        
        # Show what will be fixed
        logger.info("\n🚨 Cities with ocean/water issues:")
        for city_name in self.problematic_cities:
            bbox = self.corrected_bboxes[city_name]
            logger.info(f"  - {city_name}: {bbox}")
        
        logger.info("=" * 80)
        
        # Process each problematic city
        total_cities = len(self.problematic_cities)
        
        for i, city_name in enumerate(self.problematic_cities, 1):
            logger.info(f"\n🔧 Processing city {i}/{total_cities}: {city_name}")
            
            try:
                # Step 1: Delete existing cache
                logger.info(f"Step 1: Deleting existing cache for {city_name}")
                self.delete_city_cache(city_name)
                
                # Step 2: Regenerate with corrected bbox
                logger.info(f"Step 2: Regenerating {city_name} with corrected bbox")
                await self.regenerate_city_regions(city_name)
                
                logger.info(f"✅ Completed {city_name} ({i}/{total_cities})")
                
                # Longer delay between cities to be respectful to GEE
                if i < total_cities:
                    logger.info("⏳ Waiting 15 seconds before next city...")
                    await asyncio.sleep(15)
                    
            except Exception as e:
                logger.error(f"❌ Failed to process {city_name}: {e}")
                continue
        
        # Save final metadata
        self.save_metadata()
        
        logger.info("\n🎉 Problematic cities regeneration completed!")
        self.print_regeneration_summary()
    
    def print_regeneration_summary(self):
        """Print summary of regenerated cities"""
        logger.info("=" * 60)
        logger.info("REGENERATION SUMMARY")
        logger.info("=" * 60)
        
        regenerated_cities = [city for city in self.cache_metadata.keys() 
                             if city in self.problematic_cities]
        
        logger.info(f"Cities Regenerated: {len(regenerated_cities)}")
        logger.info(f"Cache Directory: {self.cache_dir}")
        
        # List regenerated cities
        logger.info("\nRegenerated Cities:")
        for city_name in regenerated_cities:
            city_data = self.cache_metadata[city_name]
            region_count = len(city_data['regions'])
            status = city_data.get('status', 'UNKNOWN')
            bbox_version = city_data.get('bbox_version', 'UNKNOWN')
            logger.info(f"  - {city_name}: {region_count} regions ({status}, {bbox_version})")
        
        logger.info("=" * 60)

async def main():
    """Main function to delete and regenerate problematic cities"""
    regenerator = ProblematicCitiesRegenerator()
    
    print("🔧 Delete & Regenerate Problematic Cities")
    print("=" * 60)
    print("This will DELETE and regenerate cities with ocean/water issues.")
    print("Uses VALIDATED corrected bounding boxes.")
    print("=" * 60)
    
    # Show what will be processed
    print("\n🚨 Cities to be deleted and regenerated:")
    for city_name in regenerator.problematic_cities:
        bbox = regenerator.corrected_bboxes[city_name]
        print(f"  - {city_name}: {bbox}")
    
    print("=" * 60)
    
    # Ask for confirmation
    response = input("Do you want to DELETE and regenerate these cities? (y/N): ")
    if response.lower() != 'y':
        print("Operation cancelled.")
        return
    
    # Start regeneration
    start_time = time.time()
    await regenerator.delete_and_regenerate_all()
    end_time = time.time()
    
    duration = end_time - start_time
    logger.info(f"⏱️ Total time taken: {duration/60:.2f} minutes")

if __name__ == "__main__":
    asyncio.run(main())