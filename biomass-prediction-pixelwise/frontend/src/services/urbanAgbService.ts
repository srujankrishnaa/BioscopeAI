/**
 * Urban AGB Prediction Service
 * Integrates with our production Urban AGB system
 */

export interface UrbanAGBRequest {
  city: string;
  coordinates?: {
    latitude: number;
    longitude: number;
  };
}

export interface UrbanAGBResponse {
  status: 'success' | 'error';
  city: string;
  timestamp: string;
  location: {
    latitude: number;
    longitude: number;
    coordinates: string;
  };
  current_agb: {
    total_agb: number;
    tree_biomass: number;
    shrub_biomass: number;
    herbaceous_biomass: number;
    canopy_cover: number;
    carbon_sequestration: number;
    cooling_potential: number;
  };
  satellite_data: {
    ndvi: number;
    evi: number;
    lai: number;
    data_source: string;
  };
  forecasting: {
    current_year?: number;
    year_1: number;
    year_2?: number;
    year_3: number;
    year_5: number;
    growth_rate: number;
    methodology?: string;
    factors_considered?: string[];
  };
  urban_metrics: {
    epi_score: number;
    tree_cities_score: number;
    green_space_ratio: number;
    energy_savings: number;
  };
  planning_recommendations: string[];
  intervention_scenarios: {
    [key: string]: {
      agb: number;
      canopy_cover: number;
      cooling_potential: number;
    };
  };
  heat_map: {
    image_path: string;
    image_url: string | null;
  };
  model_performance: {
    accuracy: string;
    ground_truth: string;
    processing_time: string;
    geographic_coverage: string;
  };
  error?: string;
}

export interface SystemStatus {
  status: string;
  timestamp: string;
  systems: {
    [key: string]: {
      status: 'ready' | 'error';
      description: string;
      error?: string;
    };
  };
  version?: string;
}

class UrbanAGBService {
  private baseUrl: string;

  constructor() {
    // Connect to backend API - default to localhost:8000
    this.baseUrl = process.env.REACT_APP_API_URL || 'http://localhost:8000';
  }

  /**
   * Get system status
   */
  async getSystemStatus(): Promise<SystemStatus> {
    try {
      const response = await fetch(`${this.baseUrl}/api/system-status`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      
      // Transform backend response to match frontend interface
      return {
        status: data.status,
        timestamp: new Date().toISOString(),
        systems: Object.entries(data.systems).reduce((acc, [key, value]: [string, any]) => {
          acc[key] = {
            status: value.status,
            description: value.message,
          };
          return acc;
        }, {} as any)
      };
    } catch (error) {
      console.error('Failed to get system status:', error);
      throw new Error('Failed to connect to Urban AGB system');
    }
  }

  /**
   * Predict Urban AGB for a city
   */
  async predictUrbanAGB(request: UrbanAGBRequest): Promise<UrbanAGBResponse> {
    try {
      console.log('🚀 Sending Urban AGB prediction request:', request);
      
      const response = await fetch(`${this.baseUrl}/api/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ city: request.city }),
      });

      console.log('📡 Response status:', response.status);
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      console.log('📦 Urban AGB response:', data);
      
      // Transform backend response to match frontend interface
      const transformedData: UrbanAGBResponse = {
        status: 'success',
        city: data.city,
        timestamp: data.timestamp,
        location: {
          latitude: parseFloat(data.location.coordinates.split(',')[1]),
          longitude: parseFloat(data.location.coordinates.split(',')[0]),
          coordinates: data.location.coordinates
        },
        current_agb: {
          total_agb: data.current_agb.total_agb,
          tree_biomass: data.current_agb.tree_biomass,
          shrub_biomass: data.current_agb.shrub_biomass,
          herbaceous_biomass: data.current_agb.herbaceous_biomass,
          canopy_cover: data.current_agb.canopy_cover,
          carbon_sequestration: data.current_agb.carbon_sequestration,
          cooling_potential: data.current_agb.cooling_potential
        },
        satellite_data: {
          ndvi: data.satellite_data.ndvi,
          evi: data.satellite_data.evi,
          lai: data.satellite_data.lai,
          data_source: data.satellite_data.data_source
        },
        forecasting: {
          current_year: data.forecasting.current_year,
          year_1: data.forecasting.year_1,
          year_2: data.forecasting.year_2,
          year_3: data.forecasting.year_3,
          year_5: data.forecasting.year_5,
          growth_rate: data.forecasting.growth_rate,
          methodology: data.forecasting.methodology,
          factors_considered: data.forecasting.factors_considered
        },
        urban_metrics: {
          epi_score: data.urban_metrics.epi_score,
          tree_cities_score: data.urban_metrics.tree_cities_score,
          green_space_ratio: data.urban_metrics.green_space_ratio,
          energy_savings: 0 // Not provided by backend yet
        },
        planning_recommendations: data.planning_recommendations,
        intervention_scenarios: {},  // Not provided by backend yet
        heat_map: {
          image_path: data.heat_map.image_url,
          image_url: data.heat_map.image_url
        },
        model_performance: {
          accuracy: data.model_performance.accuracy,
          ground_truth: data.model_performance.ground_truth,
          processing_time: data.model_performance.processing_time,
          geographic_coverage: data.model_performance.geographic_coverage
        }
      };
      
      return transformedData;
    } catch (error) {
      console.error('Urban AGB prediction failed:', error);
      throw error;
    }
  }

  /**
   * Get predefined cities/states for quick selection
   * ALL CACHED CITIES - 44 cities with satellite imagery available
   */
  getPredefinedCities(): Array<{id: string, name: string, country: string}> {
    return [
      // 🏛️ STATE CAPITALS (28 States)
      { id: 'mumbai', name: 'Mumbai', country: 'Maharashtra' },
      { id: 'bangalore', name: 'Bangalore', country: 'Karnataka' },
      { id: 'chennai', name: 'Chennai', country: 'Tamil Nadu' },
      { id: 'hyderabad', name: 'Hyderabad', country: 'Telangana' },
      { id: 'kolkata', name: 'Kolkata', country: 'West Bengal' },
      { id: 'ahmedabad', name: 'Ahmedabad', country: 'Gujarat' },
      { id: 'gandhinagar', name: 'Gandhinagar', country: 'Gujarat (Capital)' },
      { id: 'jaipur', name: 'Jaipur', country: 'Rajasthan' },
      { id: 'lucknow', name: 'Lucknow', country: 'Uttar Pradesh' },
      { id: 'bhopal', name: 'Bhopal', country: 'Madhya Pradesh' },
      { id: 'patna', name: 'Patna', country: 'Bihar' },
      { id: 'thiruvananthapuram', name: 'Thiruvananthapuram', country: 'Kerala' },
      { id: 'bhubaneswar', name: 'Bhubaneswar', country: 'Odisha' },
      { id: 'ranchi', name: 'Ranchi', country: 'Jharkhand' },
      { id: 'raipur', name: 'Raipur', country: 'Chhattisgarh' },
      { id: 'panaji', name: 'Panaji', country: 'Goa' },
      { id: 'shimla', name: 'Shimla', country: 'Himachal Pradesh' },
      { id: 'srinagar', name: 'Srinagar', country: 'Jammu & Kashmir' },
      { id: 'jammu', name: 'Jammu', country: 'Jammu & Kashmir (Winter)' },
      { id: 'guwahati', name: 'Guwahati', country: 'Assam' },
      { id: 'agartala', name: 'Agartala', country: 'Tripura' },
      { id: 'aizawl', name: 'Aizawl', country: 'Mizoram' },
      { id: 'imphal', name: 'Imphal', country: 'Manipur' },
      { id: 'kohima', name: 'Kohima', country: 'Nagaland' },
      { id: 'itanagar', name: 'Itanagar', country: 'Arunachal Pradesh' },
      { id: 'gangtok', name: 'Gangtok', country: 'Sikkim' },
      { id: 'shillong', name: 'Shillong', country: 'Meghalaya' },
      { id: 'visakhapatnam', name: 'Visakhapatnam', country: 'Andhra Pradesh' },
      
      // 🏛️ UNION TERRITORY CAPITALS (8 UTs)
      { id: 'delhi', name: 'Delhi', country: 'Delhi (NCT)' },
      { id: 'chandigarh', name: 'Chandigarh', country: 'Chandigarh (UT)' },
      { id: 'puducherry', name: 'Puducherry', country: 'Puducherry' },
      { id: 'port-blair', name: 'Port Blair', country: 'Andaman & Nicobar' },
      { id: 'kavaratti', name: 'Kavaratti', country: 'Lakshadweep' },
      { id: 'daman', name: 'Daman', country: 'Daman & Diu' },
      { id: 'silvassa', name: 'Silvassa', country: 'Dadra & Nagar Haveli' },
      { id: 'ladakh', name: 'Ladakh', country: 'Ladakh (Leh)' },
      
      // 🏙️ MAJOR CITIES (8 Additional)
      { id: 'pune', name: 'Pune', country: 'Maharashtra' },
      { id: 'nagpur', name: 'Nagpur', country: 'Maharashtra' },
      { id: 'indore', name: 'Indore', country: 'Madhya Pradesh' },
      { id: 'kanpur', name: 'Kanpur', country: 'Uttar Pradesh' },
      { id: 'thane', name: 'Thane', country: 'Maharashtra' },
      { id: 'ludhiana', name: 'Ludhiana', country: 'Punjab' },
      { id: 'agra', name: 'Agra', country: 'Uttar Pradesh' },
      { id: 'ghaziabad', name: 'Ghaziabad', country: 'Uttar Pradesh' },
      { id: 'vadodara', name: 'Vadodara', country: 'Gujarat' }
    ];
  }

  /**
   * Format biomass value for display
   */
  formatBiomass(value: number): string {
    return `${value.toFixed(1)} Mg/ha`;
  }

  /**
   * Format percentage for display
   */
  formatPercentage(value: number): string {
    return `${value.toFixed(1)}%`;
  }

  /**
   * Format temperature for display
   */
  formatTemperature(value: number): string {
    return `${value.toFixed(1)}°C`;
  }

  /**
   * Get health status color based on AGB value
   */
  getHealthStatusColor(agb: number): string {
    if (agb >= 120) return 'text-green-500';
    if (agb >= 100) return 'text-blue-500';
    if (agb >= 80) return 'text-yellow-500';
    if (agb >= 60) return 'text-orange-500';
    return 'text-red-500';
  }

  /**
   * Get health status text based on AGB value
   */
  getHealthStatusText(agb: number): string {
    if (agb >= 120) return 'Excellent';
    if (agb >= 100) return 'Very Good';
    if (agb >= 80) return 'Good';
    if (agb >= 60) return 'Moderate';
    return 'Poor';
  }

  /**
   * Calculate carbon credits potential
   */
  calculateCarbonCredits(carbonSequestration: number, area: number = 1): number {
    // Assuming 1 hectare area and $15 per ton CO2
    return carbonSequestration * area * 15;
  }

  /**
   * Get intervention priority based on metrics
   */
  getInterventionPriority(epiScore: number, treeCitiesScore: number): 'High' | 'Medium' | 'Low' {
    const avgScore = (epiScore + treeCitiesScore) / 2;
    if (avgScore < 60) return 'High';
    if (avgScore < 80) return 'Medium';
    return 'Low';
  }
}

export const urbanAGBService = new UrbanAGBService();
export default urbanAGBService;