#!/usr/bin/env python3
"""
Fix Problematic Cities with Incorrect Bounding Boxes
Downloads only cities that have ocean/water issues in their regions
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
        logging.FileHandler('problematic_cities_fix.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ProblematicCitiesFixer:
    """Fixes cities with incorrect bounding boxes that show too much ocean/water"""
    
    def __init__(self):
        self.gee_fetcher = GEEDataFetcher()
        self.cache_dir = Path("./outputs/region_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cities with problematic bounding boxes (ocean/water issues)
        self.problematic_cities = {
            'Daman': {
                'issues': ['North: ocean-only', 'South: ocean-only'],
                'fix': 'Recalibrate bbox to focus on mainland Daman, reduce coastal regions'
            },
            'Kavaratti': {
                'issues': ['East: ocean-only', 'West: ocean-only', 'North: ocean-only', 'South: ocean-only'],
                'fix': 'Tight island-centered bbox (~0.02° radius around island)'
            },
            'Panaji': {
                'issues': ['South: dominated by water and airport', 'West: mostly Arabian Sea'],
                'fix': 'Reduce southern latitude, shift west regions eastward (~0.03-0.04°)'
            },
            'Port Blair': {
                'issues': ['East: ocean-only'],
                'fix': 'Adjust eastern bbox to include more land area'
            },
            'Puducherry': {
                'issues': ['East: ocean-only'],
                'fix': 'Shift eastern region westward to include more urban area'
            },
            'Chennai': {
                'issues': ['East: mostly ocean, missing urban area'],
                'fix': 'Adjust eastern bbox to include urban areas instead of Bay of Bengal'
            },
            'Thiruvananthapuram': {
                'issues': ['South: ocean-heavy', 'West: ocean-heavy'],
                'fix': 'Reduce coastal extension, focus on mainland city area'
            },
            'Gandhinagar': {
                'issues': ['East: extremely narrow strips', 'West: extremely narrow strips'],
                'fix': 'Fix bounding box distortion, ensure proper region proportions'
            }
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
    
    def get_corrected_city_bbox(self, city_name: str) -> Tuple[float, float, float, float]:
        """Get corrected city bounding box for problematic cities"""
        
        # Corrected bounding boxes for problematic cities
        corrected_bboxes = {
            'Daman': (72.8200, 20.3900, 72.8600, 20.4300),  # Tighter mainland focus
            'Kavaratti': (72.6350, 10.5550, 72.6450, 10.5650),  # Very tight island focus (~0.01° radius)
            'Panaji': (73.8000, 15.4800, 73.8500, 15.5200),  # Reduced southern water, more mainland
            'Port Blair': (92.7200, 11.6500, 92.7800, 11.7100),  # Adjusted eastern boundary
            'Puducherry': (79.8000, 11.9000, 79.8400, 11.9400),  # Shifted away from ocean
            'Chennai': (80.1800, 13.0000, 80.2600, 13.0800),  # Reduced eastern ocean exposure
            'Thiruvananthapuram': (76.9000, 8.4800, 77.0000, 8.5800),  # Less coastal extension
            'Gandhinagar': (72.6200, 23.2000, 72.6800, 23.2600)  # Fixed aspect ratio distortion
        }
        
        if city_name in corrected_bboxes:
            logger.info(f"Using corrected bbox for {city_name}")
            return corrected_bboxes[city_name]
        
        # Fallback to original method
        bbox = self.gee_fetcher.get_city_bbox(city_name)
        if bbox:
            return bbox
        
        bbox = get_fallback_city_bbox(city_name)
        if bbox:
            return bbox
        
        raise ValueError(f"Could not find coordinates for {city_name}")
    
    def delete_existing_city_cache(self, city_name: str):
        """Delete existing cache for a problematic city"""
        try:
            city_cache_dir = self.cache_dir / city_name.replace(' ', '_')
            if city_cache_dir.exists():
                import shutil
                shutil.rmtree(city_cache_dir)
                logger.info(f"🗑️ Deleted existing cache directory for {city_name}")
            
            # Remove from metadata
            if city_name in self.cache_metadata:
                del self.cache_metadata[city_name]
                self.save_metadata()
                logger.info(f"🗑️ Removed {city_name} from cache metadata")
                
        except Exception as e:
            logger.error(f"Failed to delete cache for {city_name}: {e}")
    
    async def generate_region_satellite_image(self, region_bbox: Tuple[float, float, float, float], 
                                            region_name: str, city_name: str) -> str:
        """Generate high-quality satellite image for a region"""
        try:
            logger.info(f"Generating corrected satellite image for {region_name}")
            
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
                plt.title(f'{region_name} - Satellite View (CORRECTED)', 
                         fontsize=16, fontweight='bold', pad=20)
                
                # Add metadata
                plt.figtext(0.02, 0.02, 
                           f'Source: Sentinel-2 via Google Earth Engine | CORRECTED BBOX | Generated: {time.strftime("%Y-%m-%d %H:%M")}',
                           fontsize=10, style='italic', alpha=0.7)
                
                plt.tight_layout()
                plt.savefig(filepath, bbox_inches='tight', dpi=200, 
                           facecolor='white', edgecolor='none')
                plt.close()
                
                logger.info(f"✅ Corrected satellite image saved: {filename}")
                return str(filepath.relative_to(self.cache_dir))
            
            else:
                logger.warning(f"No satellite data available for {region_name}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to generate satellite image for {region_name}: {e}")
            return None
    
    async def fix_city_regions(self, city_name: str):
        """Fix all 5 regions for a problematic city"""
        try:
            logger.info(f"🔧 FIXING PROBLEMATIC CITY: {city_name}")
            
            # Show what issues we're fixing
            if city_name in self.problematic_cities:
                issues = self.problematic_cities[city_name]['issues']
                fix_description = self.problematic_cities[city_name]['fix']
                logger.info(f"Issues to fix: {', '.join(issues)}")
                logger.info(f"Fix strategy: {fix_description}")
            
            # Delete existing problematic cache
            self.delete_existing_city_cache(city_name)
            
            # Get corrected city bounding box
            city_bbox = self.get_corrected_city_bbox(city_name)
            logger.info(f"Corrected bbox for {city_name}: {city_bbox}")
            
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
                'issues_fixed': self.problematic_cities.get(city_name, {}).get('issues', [])
            }
            
            # Generate satellite image for each region
            for region_data in regions_data:
                region_id = region_data['id']
                region_name = region_data['name']
                region_bbox = tuple(region_data['bbox'])
                
                logger.info(f"Processing corrected region: {region_name}")
                
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
                await asyncio.sleep(2)
            
            # Save city metadata
            self.cache_metadata[city_name] = city_metadata
            self.save_metadata()
            
            logger.info(f"✅ Successfully fixed and cached {len(regions_data)} regions for {city_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to fix regions for {city_name}: {e}")
    
    async def fix_all_problematic_cities(self):
        """Fix all cities with problematic bounding boxes"""
        # Load existing metadata first
        self.load_existing_metadata()
        
        problematic_city_names = list(self.problematic_cities.keys())
        
        logger.info("=" * 80)
        logger.info("🔧 FIXING PROBLEMATIC CITIES WITH BBOX ISSUES")
        logger.info("=" * 80)
        logger.info(f"Total Problematic Cities: {len(problematic_city_names)}")
        
        logger.info(f"\n🚨 Cities to Fix:")
        for city_name, details in self.problematic_cities.items():
            logger.info(f"  - {city_name}: {', '.join(details['issues'])}")
        
        logger.info("=" * 80)
        
        # Process each problematic city
        total_cities = len(problematic_city_names)
        
        for i, city_name in enumerate(problematic_city_names, 1):
            logger.info(f"🔧 Fixing city {i}/{total_cities}: {city_name}")
            
            try:
                await self.fix_city_regions(city_name)
                logger.info(f"✅ Fixed {city_name} ({i}/{total_cities})")
                
                # Longer delay between cities to be respectful to GEE
                if i < total_cities:
                    logger.info("⏳ Waiting 15 seconds before next city...")
                    await asyncio.sleep(15)
                    
            except Exception as e:
                logger.error(f"❌ Failed to fix {city_name}: {e}")
                continue
        
        logger.info("🎉 Problematic cities fix completed!")
        self.print_fix_summary()
    
    def print_fix_summary(self):
        """Print summary of fixed cities"""
        logger.info("=" * 60)
        logger.info("PROBLEMATIC CITIES FIX SUMMARY")
        logger.info("=" * 60)
        
        fixed_cities = [city for city in self.cache_metadata.keys() 
                       if city in self.problematic_cities]
        
        logger.info(f"Cities Fixed: {len(fixed_cities)}")
        logger.info(f"Cache Directory: {self.cache_dir}")
        
        # List fixed cities
        logger.info("\nFixed Cities:")
        for city_name in fixed_cities:
            city_data = self.cache_metadata[city_name]
            region_count = len(city_data['regions'])
            status = city_data.get('status', 'UNKNOWN')
            logger.info(f"  - {city_name}: {region_count} regions ({status})")
        
        logger.info("=" * 60)

async def main():
    """Main function to fix problematic cities"""
    fixer = ProblematicCitiesFixer()
    
    print("🔧 Problematic Cities Bbox Fixer")
    print("=" * 60)
    print("This will fix cities with ocean/water issues in their regions.")
    print("Deletes old cache and regenerates with corrected bounding boxes.")
    print("=" * 60)
    
    # Show what will be fixed
    print("\n🚨 Cities to be fixed:")
    for city_name, details in fixer.problematic_cities.items():
        print(f"  - {city_name}: {', '.join(details['issues'])}")
    
    print("=" * 60)
    
    # Ask for confirmation
    response = input("Do you want to fix these problematic cities? (y/N): ")
    if response.lower() != 'y':
        print("Fix operation cancelled.")
        return
    
    # Start fixing
    start_time = time.time()
    await fixer.fix_all_problematic_cities()
    end_time = time.time()
    
    duration = end_time - start_time
    logger.info(f"⏱️ Total time taken: {duration/60:.2f} minutes")

if __name__ == "__main__":
    asyncio.run(main())