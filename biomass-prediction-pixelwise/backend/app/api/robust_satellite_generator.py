"""
Robust Satellite Image Generator
Uses the robust GEE client for reliable satellite image generation
"""

import logging
import numpy as np
from typing import Dict, Tuple, Optional, Any
from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path
import asyncio

from app.utils.gee_robust import get_robust_gee_client, GEEResponse

logger = logging.getLogger(__name__)

class RobustSatelliteGenerator:
    """Robust satellite image generator with comprehensive error handling"""
    
    def __init__(self):
        self.client = get_robust_gee_client()
        self.output_dir = Path("./outputs/satellite_images")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    async def generate_satellite_image(self, 
                                     bbox: Tuple[float, float, float, float],
                                     region_name: str,
                                     quality: str = 'high',
                                     include_indices: bool = True) -> Dict[str, Any]:
        """
        Generate satellite image with vegetation indices
        
        Args:
            bbox: Bounding box (min_lon, min_lat, max_lon, max_lat)
            region_name: Name of the region for file naming
            quality: Quality level ('low', 'medium', 'high')
            include_indices: Whether to calculate vegetation indices
        
        Returns:
            Dictionary with image data and metadata
        """
        try:
            logger.info(f"Generating satellite image for {region_name} with {quality} quality")
            
            # Set parameters based on quality
            quality_params = self._get_quality_parameters(quality)
            
            # Fetch RGB satellite image
            rgb_response = await self.client.fetch_satellite_image(
                bbox=bbox,
                scale=quality_params['scale'],
                dimensions=quality_params['dimensions'],
                collection="COPERNICUS/S2_SR",
                bands=['B4', 'B3', 'B2'],  # Red, Green, Blue
                date_range=quality_params['date_range']
            )
            
            if not rgb_response.success:
                logger.error(f"RGB image fetch failed: {rgb_response.error_message}")
                return {
                    'success': False,
                    'error': f"RGB fetch failed: {rgb_response.error_message}",
                    'error_type': rgb_response.error_type.value if rgb_response.error_type else 'unknown'
                }
            
            result = {
                'success': True,
                'region_name': region_name,
                'bbox': bbox,
                'quality': quality,
                'rgb_image': rgb_response.data['image_array'],
                'metadata': {
                    'response_time_ms': rgb_response.response_time_ms,
                    'cache_hit': rgb_response.cache_hit,
                    'strategy_used': rgb_response.strategy_used.value,
                    'scale': quality_params['scale'],
                    'dimensions': quality_params['dimensions']
                }
            }
            
            # Fetch vegetation indices if requested
            if include_indices:
                indices_data = await self._fetch_vegetation_indices(bbox, quality_params)
                if indices_data['success']:
                    result.update(indices_data)
                else:
                    logger.warning(f"Vegetation indices fetch failed: {indices_data.get('error')}")
                    # Continue without indices rather than failing completely
                    result['indices_warning'] = indices_data.get('error')
            
            # Generate and save visualization
            image_path = await self._save_satellite_visualization(result, region_name)
            result['image_path'] = image_path
            
            logger.info(f"Successfully generated satellite image for {region_name}")
            return result
            
        except Exception as e:
            logger.error(f"Satellite image generation failed for {region_name}: {e}")
            return {
                'success': False,
                'error': str(e),
                'error_type': 'generation_error'
            }
    
    def _get_quality_parameters(self, quality: str) -> Dict:
        """Get parameters based on quality level"""
        quality_configs = {
            'low': {
                'scale': 60,
                'dimensions': 512,
                'date_range': ('2023-01-01', '2024-12-31')
            },
            'medium': {
                'scale': 30,
                'dimensions': 1024,
                'date_range': ('2023-01-01', '2024-12-31')
            },
            'high': {
                'scale': 10,
                'dimensions': 2048,
                'date_range': ('2023-01-01', '2024-12-31')
            }
        }
        
        return quality_configs.get(quality, quality_configs['medium'])
    
    async def _fetch_vegetation_indices(self, bbox: Tuple[float, float, float, float], 
                                      quality_params: Dict) -> Dict:
        """Fetch vegetation indices (NDVI, EVI, LAI)"""
        try:
            # Fetch NIR and Red bands for NDVI calculation
            nir_red_response = await self.client.fetch_satellite_image(
                bbox=bbox,
                scale=quality_params['scale'],
                dimensions=quality_params['dimensions'],
                collection="COPERNICUS/S2_SR",
                bands=['B8', 'B4'],  # NIR, Red
                date_range=quality_params['date_range']
            )
            
            if not nir_red_response.success:
                return {
                    'success': False,
                    'error': f"NIR/Red bands fetch failed: {nir_red_response.error_message}"
                }
            
            # Calculate vegetation indices
            nir_red_data = nir_red_response.data['image_array']
            
            if len(nir_red_data.shape) == 3 and nir_red_data.shape[2] >= 2:
                nir = nir_red_data[:, :, 0].astype(np.float32)
                red = nir_red_data[:, :, 1].astype(np.float32)
                
                # Calculate NDVI
                ndvi = np.where(
                    (nir + red) != 0,
                    (nir - red) / (nir + red),
                    0
                )
                
                # Clip NDVI to valid range
                ndvi = np.clip(ndvi, -1, 1)
                
                # Calculate EVI (simplified)
                evi = np.where(
                    (nir + 6 * red + 1) != 0,
                    2.5 * (nir - red) / (nir + 6 * red + 1),
                    0
                )
                evi = np.clip(evi, -1, 1)
                
                # Estimate LAI from NDVI
                lai = np.where(
                    ndvi > 0,
                    3.618 * ndvi - 0.118,  # Empirical relationship
                    0
                )
                lai = np.clip(lai, 0, 8)
                
                return {
                    'success': True,
                    'ndvi': float(np.mean(ndvi)),
                    'evi': float(np.mean(evi)),
                    'lai': float(np.mean(lai)),
                    'ndvi_array': ndvi,
                    'evi_array': evi,
                    'lai_array': lai,
                    'vegetation_stats': {
                        'ndvi_min': float(np.min(ndvi)),
                        'ndvi_max': float(np.max(ndvi)),
                        'ndvi_std': float(np.std(ndvi)),
                        'vegetation_coverage': float(np.sum(ndvi > 0.3) / ndvi.size * 100)
                    }
                }
            else:
                return {
                    'success': False,
                    'error': 'Invalid NIR/Red data shape for vegetation index calculation'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f"Vegetation indices calculation failed: {str(e)}"
            }
    
    async def _save_satellite_visualization(self, result: Dict, region_name: str) -> str:
        """Save satellite image visualization"""
        try:
            # Generate safe filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = region_name.replace(' ', '_').replace('-', '_')
            filename = f"satellite_{safe_name}_{timestamp}.png"
            filepath = self.output_dir / filename
            
            # Create visualization
            if 'ndvi_array' in result:
                # Create multi-panel visualization
                fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=150)
                
                # RGB image
                rgb_image = result['rgb_image']
                if rgb_image.max() > 1:
                    rgb_image = rgb_image / 255.0  # Normalize if needed
                
                axes[0, 0].imshow(np.clip(rgb_image, 0, 1))
                axes[0, 0].set_title(f'{region_name} - RGB Satellite Image', fontsize=14, fontweight='bold')
                axes[0, 0].axis('off')
                
                # NDVI
                ndvi_plot = axes[0, 1].imshow(result['ndvi_array'], cmap='RdYlGn', vmin=-1, vmax=1)
                axes[0, 1].set_title(f'NDVI (Mean: {result["ndvi"]:.3f})', fontsize=14, fontweight='bold')
                axes[0, 1].axis('off')
                plt.colorbar(ndvi_plot, ax=axes[0, 1], shrink=0.8)
                
                # EVI
                evi_plot = axes[1, 0].imshow(result['evi_array'], cmap='Greens', vmin=-1, vmax=1)
                axes[1, 0].set_title(f'EVI (Mean: {result["evi"]:.3f})', fontsize=14, fontweight='bold')
                axes[1, 0].axis('off')
                plt.colorbar(evi_plot, ax=axes[1, 0], shrink=0.8)
                
                # LAI
                lai_plot = axes[1, 1].imshow(result['lai_array'], cmap='YlGn', vmin=0, vmax=8)
                axes[1, 1].set_title(f'LAI (Mean: {result["lai"]:.3f})', fontsize=14, fontweight='bold')
                axes[1, 1].axis('off')
                plt.colorbar(lai_plot, ax=axes[1, 1], shrink=0.8)
                
            else:
                # Simple RGB visualization
                fig, ax = plt.subplots(figsize=(12, 10), dpi=150)
                
                rgb_image = result['rgb_image']
                if rgb_image.max() > 1:
                    rgb_image = rgb_image / 255.0
                
                ax.imshow(np.clip(rgb_image, 0, 1))
                ax.set_title(f'{region_name} - Satellite Image', fontsize=16, fontweight='bold')
                ax.axis('off')
            
            # Add metadata
            metadata_text = (
                f"Quality: {result['quality']} | "
                f"Scale: {result['metadata']['scale']}m | "
                f"Strategy: {result['metadata']['strategy_used']} | "
                f"Cache: {'Hit' if result['metadata']['cache_hit'] else 'Miss'}"
            )
            
            plt.figtext(0.02, 0.02, metadata_text, fontsize=10, style='italic', alpha=0.7)
            plt.figtext(0.98, 0.02, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                       fontsize=10, style='italic', alpha=0.7, ha='right')
            
            plt.tight_layout()
            plt.savefig(filepath, bbox_inches='tight', dpi=150, facecolor='white')
            plt.close()
            
            logger.info(f"Satellite visualization saved: {filename}")
            return f"/outputs/satellite_images/{filename}"
            
        except Exception as e:
            logger.error(f"Failed to save satellite visualization: {e}")
            return None
    
    async def generate_biomass_heatmap(self, 
                                     bbox: Tuple[float, float, float, float],
                                     region_name: str,
                                     quality: str = 'high') -> Dict[str, Any]:
        """Generate biomass heatmap with satellite imagery"""
        try:
            logger.info(f"Generating biomass heatmap for {region_name}")
            
            # Get satellite data with vegetation indices
            satellite_data = await self.generate_satellite_image(
                bbox=bbox,
                region_name=region_name,
                quality=quality,
                include_indices=True
            )
            
            if not satellite_data['success']:
                return satellite_data
            
            # Calculate biomass from vegetation indices
            biomass_data = self._calculate_biomass_from_indices(
                satellite_data.get('ndvi', 0),
                satellite_data.get('evi', 0),
                satellite_data.get('lai', 0)
            )
            
            # Generate biomass heatmap visualization
            heatmap_path = await self._save_biomass_heatmap(
                satellite_data, biomass_data, region_name
            )
            
            result = {
                'success': True,
                'region_name': region_name,
                'bbox': bbox,
                'satellite_data': {
                    'ndvi': satellite_data.get('ndvi'),
                    'evi': satellite_data.get('evi'),
                    'lai': satellite_data.get('lai'),
                    'vegetation_stats': satellite_data.get('vegetation_stats')
                },
                'biomass_data': biomass_data,
                'heatmap_path': heatmap_path,
                'metadata': satellite_data.get('metadata')
            }
            
            logger.info(f"Successfully generated biomass heatmap for {region_name}")
            return result
            
        except Exception as e:
            logger.error(f"Biomass heatmap generation failed for {region_name}: {e}")
            return {
                'success': False,
                'error': str(e),
                'error_type': 'heatmap_generation_error'
            }
    
    def _calculate_biomass_from_indices(self, ndvi: float, evi: float, lai: float) -> Dict:
        """Calculate biomass metrics from vegetation indices"""
        try:
            # Biomass estimation using empirical relationships
            # These are simplified models - in practice, you'd use more sophisticated algorithms
            
            # Above Ground Biomass (AGB) estimation
            agb_from_ndvi = max(0, 150 * ndvi - 20)  # Simplified linear relationship
            agb_from_lai = max(0, 25 * lai)  # LAI-based estimation
            
            # Weighted average of different estimates
            total_agb = (agb_from_ndvi * 0.6 + agb_from_lai * 0.4)
            
            # Canopy cover estimation
            canopy_cover = min(100, max(0, (ndvi + 1) * 50))  # Convert NDVI to percentage
            
            # Carbon stock estimation (AGB * carbon fraction)
            carbon_stock = total_agb * 0.47  # Typical carbon fraction
            
            # Biomass density classification
            if total_agb < 20:
                density_class = "Low"
            elif total_agb < 60:
                density_class = "Medium"
            elif total_agb < 100:
                density_class = "High"
            else:
                density_class = "Very High"
            
            return {
                'total_agb': round(total_agb, 2),
                'carbon_stock': round(carbon_stock, 2),
                'canopy_cover': round(canopy_cover, 1),
                'density_class': density_class,
                'biomass_components': {
                    'agb_from_ndvi': round(agb_from_ndvi, 2),
                    'agb_from_lai': round(agb_from_lai, 2)
                },
                'vegetation_health': {
                    'ndvi_score': min(100, max(0, (ndvi + 1) * 50)),
                    'evi_score': min(100, max(0, (evi + 1) * 50)),
                    'lai_score': min(100, lai * 12.5)
                }
            }
            
        except Exception as e:
            logger.error(f"Biomass calculation failed: {e}")
            return {
                'total_agb': 0,
                'carbon_stock': 0,
                'canopy_cover': 0,
                'density_class': "Unknown",
                'error': str(e)
            }
    
    async def _save_biomass_heatmap(self, satellite_data: Dict, 
                                  biomass_data: Dict, region_name: str) -> str:
        """Save biomass heatmap visualization"""
        try:
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = region_name.replace(' ', '_').replace('-', '_')
            filename = f"biomass_heatmap_{safe_name}_{timestamp}.png"
            filepath = self.output_dir / filename
            
            # Create biomass heatmap
            fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=150)
            
            # RGB satellite image
            rgb_image = satellite_data['rgb_image']
            if rgb_image.max() > 1:
                rgb_image = rgb_image / 255.0
            
            axes[0, 0].imshow(np.clip(rgb_image, 0, 1))
            axes[0, 0].set_title(f'{region_name} - Satellite Image', fontsize=14, fontweight='bold')
            axes[0, 0].axis('off')
            
            # NDVI with biomass overlay
            ndvi_array = satellite_data.get('ndvi_array')
            if ndvi_array is not None:
                ndvi_plot = axes[0, 1].imshow(ndvi_array, cmap='RdYlGn', vmin=-1, vmax=1)
                axes[0, 1].set_title(f'NDVI - Vegetation Index', fontsize=14, fontweight='bold')
                axes[0, 1].axis('off')
                plt.colorbar(ndvi_plot, ax=axes[0, 1], shrink=0.8)
                
                # Biomass density map (derived from NDVI)
                biomass_array = np.maximum(0, 150 * ndvi_array - 20)
                biomass_plot = axes[1, 0].imshow(biomass_array, cmap='YlOrRd', vmin=0, vmax=150)
                axes[1, 0].set_title(f'Biomass Density (Mg/ha)', fontsize=14, fontweight='bold')
                axes[1, 0].axis('off')
                plt.colorbar(biomass_plot, ax=axes[1, 0], shrink=0.8)
            
            # Biomass statistics
            axes[1, 1].axis('off')
            stats_text = (
                f"BIOMASS ANALYSIS\n\n"
                f"Total AGB: {biomass_data['total_agb']} Mg/ha\n"
                f"Carbon Stock: {biomass_data['carbon_stock']} Mg C/ha\n"
                f"Canopy Cover: {biomass_data['canopy_cover']}%\n"
                f"Density Class: {biomass_data['density_class']}\n\n"
                f"VEGETATION INDICES\n\n"
                f"NDVI: {satellite_data.get('ndvi', 0):.3f}\n"
                f"EVI: {satellite_data.get('evi', 0):.3f}\n"
                f"LAI: {satellite_data.get('lai', 0):.3f}\n\n"
                f"DATA QUALITY\n\n"
                f"Resolution: {satellite_data['metadata']['scale']}m\n"
                f"Cache Hit: {'Yes' if satellite_data['metadata']['cache_hit'] else 'No'}\n"
                f"Response Time: {satellite_data['metadata']['response_time_ms']}ms"
            )
            
            axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes,
                           fontsize=12, verticalalignment='top', fontfamily='monospace',
                           bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
            
            plt.suptitle(f'Biomass Analysis - {region_name}', fontsize=18, fontweight='bold')
            plt.tight_layout()
            plt.savefig(filepath, bbox_inches='tight', dpi=150, facecolor='white')
            plt.close()
            
            logger.info(f"Biomass heatmap saved: {filename}")
            return f"/outputs/satellite_images/{filename}"
            
        except Exception as e:
            logger.error(f"Failed to save biomass heatmap: {e}")
            return None

# Global instance
robust_satellite_generator = None

def get_robust_satellite_generator() -> RobustSatelliteGenerator:
    """Get global robust satellite generator instance"""
    global robust_satellite_generator
    
    if robust_satellite_generator is None:
        robust_satellite_generator = RobustSatelliteGenerator()
    
    return robust_satellite_generator