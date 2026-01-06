import axios from 'axios';

export interface RegionData {
  id: string;
  name: string;
  description: string;
  bbox: number[];
  coordinates: {
    center: [number, number];
    bounds: [[number, number], [number, number]];
  };
  preview_image_url?: string;
}

export interface CityRegionsResponse {
  city: string;
  total_regions: number;
  regions: RegionData[];
  city_center: [number, number];
  city_bbox: number[];
}

export interface RegionRequest {
  city: string;
}

export interface RegionAnalysisRequest {
  region_bbox: number[];
  region_name: string;
  city: string;
}

class RegionService {
  private baseURL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

  /**
   * Get available regions for a city
   */
  async getCityRegions(cityName: string): Promise<CityRegionsResponse> {
    try {
      const response = await axios.post<CityRegionsResponse>(
        `${this.baseURL}/api/get-city-regions`,
        { city: cityName },
        {
          timeout: 120000, // 2 minutes timeout for satellite image generation
          headers: {
            'Content-Type': 'application/json',
          },
        }
      );

      return response.data;
    } catch (error) {
      if (axios.isAxiosError(error)) {
        if (error.response?.status === 404) {
          throw new Error(`City "${cityName}" not found. Please check the spelling and try again.`);
        } else if (error.response?.status === 500) {
          throw new Error('Server error while fetching city regions. Please try again later.');
        } else if (error.code === 'ECONNABORTED') {
          throw new Error('Request timeout. The server is taking too long to respond.');
        }
      }
      throw new Error('Failed to fetch city regions. Please check your internet connection.');
    }
  }

  /**
   * Analyze a specific region for biomass prediction
   */
  async analyzeRegion(request: RegionAnalysisRequest): Promise<any> {
    try {
      // Step 1: Start job
      const startResponse = await axios.post(
        `${this.baseURL}/api/analyze-region`,
        {
          region_bbox: request.region_bbox,
          region_name: request.region_name,
          city: request.city,
        },
        {
          timeout: 10000, // Quick response
          headers: {
            'Content-Type': 'application/json',
          },
        }
      );

      const { job_id } = startResponse.data;
      console.log('🚀 Job started:', job_id);

      // Step 2: Poll for completion
      return await this.pollJobStatus(job_id);
    } catch (error) {
      if (axios.isAxiosError(error)) {
        if (error.response?.status === 404) {
          throw new Error('Region not found or invalid coordinates.');
        } else if (error.response?.status === 500) {
          throw new Error('Server error during region analysis. Please try again.');
        } else if (error.code === 'ECONNABORTED') {
          throw new Error('Request timeout. Please try again.');
        }
      }
      throw new Error('Failed to analyze region. Please try again.');
    }
  }

  private async pollJobStatus(jobId: string, maxAttempts: number = 60): Promise<any> {
    for (let attempt = 0; attempt < maxAttempts; attempt++) {
      try {
        const response = await axios.get(
          `${this.baseURL}/api/job-status/${jobId}`,
          { timeout: 5000 }
        );

        const job = response.data;

        if (job.status === 'completed') {
          console.log('✅ Job completed:', job.heatmap_url);
          return {
            status: 'ok',
            source: 'live',
            city: job.city,
            region_name: job.region_name,
            heat_map: {
              image_url: job.heatmap_url,
              description: `Biomass analysis for ${job.region_name}, ${job.city}`
            },
            satellite_data: {
              ndvi: job.stats?.ndvi || 0,
              evi: job.stats?.evi || 0,
              lai: job.stats?.lai || 0,
              data_source: job.stats?.data_source || 'Google Earth Engine'
            },
            current_agb: {
              total_agb: job.stats?.total_agb || 0,
              canopy_cover: job.stats?.canopy_cover || 0,
              tree_biomass: job.stats?.tree_biomass || 0
            },
            forecasting: job.stats?.forecasting || {},
            timestamp: job.completed_at
          };
        }

        if (job.status === 'failed') {
          throw new Error(job.error_message || 'Job failed');
        }

        // Still processing, wait and retry
        console.log(`⏳ Job ${jobId} still processing... (${attempt + 1}/${maxAttempts})`);
        await new Promise(resolve => setTimeout(resolve, 2000)); // Wait 2 seconds

      } catch (error) {
        if (attempt === maxAttempts - 1) {
          throw new Error('Analysis took too long to complete. Please try again.');
        }
        // Continue polling on error
        await new Promise(resolve => setTimeout(resolve, 2000));
      }
    }

    throw new Error('Analysis timeout - took too long to complete');
  }

  /**
   * Get region preview image URL
   */
  getPreviewImageUrl(imagePath: string): string {
    if (!imagePath) return '/placeholder-satellite.png';
    
    // Handle both absolute and relative paths
    if (imagePath.startsWith('http')) {
      return imagePath;
    }
    
    // Handle API preview endpoints
    if (imagePath.startsWith('/api/region-preview')) {
      return `${this.baseURL}${imagePath}`;
    }
    
    const fullUrl = `${this.baseURL}${imagePath}`;
    console.log('🔍 RegionService.getPreviewImageUrl:', {
      input: imagePath,
      baseURL: this.baseURL,
      output: fullUrl
    });
    
    return fullUrl;
  }

  /**
   * Validate region data
   */
  validateRegionData(region: RegionData): boolean {
    return !!(
      region.id &&
      region.name &&
      region.bbox &&
      region.bbox.length === 4 &&
      region.coordinates &&
      region.coordinates.center &&
      region.coordinates.center.length === 2
    );
  }

  /**
   * Calculate region area in square kilometers (approximate)
   */
  calculateRegionArea(bbox: number[]): number {
    const [minLon, minLat, maxLon, maxLat] = bbox;
    
    // Approximate conversion: 1 degree ≈ 111 km
    const widthKm = (maxLon - minLon) * 111;
    const heightKm = (maxLat - minLat) * 111;
    
    return widthKm * heightKm;
  }

  /**
   * Format region coordinates for display
   */
  formatCoordinates(coordinates: [number, number]): string {
    const [lat, lon] = coordinates;
    return `${lat.toFixed(4)}°N, ${lon.toFixed(4)}°E`;
  }

  /**
   * Get region color based on ID (for UI consistency)
   */
  getRegionColor(regionId: string): string {
    const colors = {
      center: '#22c55e',    // Green
      north: '#3b82f6',     // Blue
      south: '#f59e0b',     // Amber
      east: '#ef4444',      // Red
      west: '#8b5cf6',      // Purple
    };
    
    return colors[regionId as keyof typeof colors] || '#6b7280';
  }

  /**
   * Get region icon based on ID
   */
  getRegionIcon(regionId: string): string {
    const icons = {
      center: '🏙️',
      north: '⬆️',
      south: '⬇️',
      east: '➡️',
      west: '⬅️',
    };
    
    return icons[regionId as keyof typeof icons] || '📍';
  }
}

export default new RegionService();