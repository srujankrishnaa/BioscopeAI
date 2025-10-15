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
      const response = await axios.post(
        `${this.baseURL}/api/analyze-region`,
        {
          region_bbox: request.region_bbox,
          region_name: request.region_name,
          city: request.city,
        },
        {
          timeout: 120000, // 2 minutes timeout for full analysis
          headers: {
            'Content-Type': 'application/json',
          },
        }
      );

      return response.data;
    } catch (error) {
      if (axios.isAxiosError(error)) {
        if (error.response?.status === 404) {
          throw new Error('Region not found or invalid coordinates.');
        } else if (error.response?.status === 500) {
          throw new Error('Server error during region analysis. Please try again.');
        } else if (error.code === 'ECONNABORTED') {
          throw new Error('Analysis timeout. Please try again with a smaller region.');
        }
      }
      throw new Error('Failed to analyze region. Please try again.');
    }
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
    
    return `${this.baseURL}${imagePath}`;
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