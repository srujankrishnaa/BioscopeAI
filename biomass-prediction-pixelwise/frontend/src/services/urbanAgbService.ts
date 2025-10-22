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
    // Connect to backend API - use Railway URL in production
    this.baseUrl = process.env.REACT_APP_API_URL || 
      (typeof window !== 'undefined' ? window.location.origin : 'http://localhost:8000');
    console.debug("Urban AGB Service initialized with base URL:", this.baseUrl);
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
      
      // Use analyze-region endpoint which generates heatmaps
      const apiUrl = `${this.baseUrl}/api/analyze-region`;
      console.debug("Calling analyze:", apiUrl);
      
      const response = await fetch(apiUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          city: request.city,
          region_name: "Center", // Default to center region
          region_bbox: this.getCityBoundingBox(request.city)
        }),
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

  /**
   * Get bounding box for a city (approximate coordinates)
   */
  private getCityBoundingBox(cityName: string): [number, number, number, number] {
    const city = cityName.toLowerCase().trim();
    
    // City coordinates: [min_lon, min_lat, max_lon, max_lat]
    const cityCoordinates: { [key: string]: [number, number, number, number] } = {
      'mumbai': [72.7757, 18.8896, 72.9781, 19.2183],
      'delhi': [76.8388, 28.4089, 77.3462, 28.8842],
      'bangalore': [77.4601, 12.8339, 77.7840, 13.1746],
      'hyderabad': [78.2479, 17.2473, 78.6677, 17.5618],
      'chennai': [80.0255, 12.8342, 80.3242, 13.2277],
      'kolkata': [88.2636, 22.4697, 88.4304, 22.6405],
      'pune': [73.6816, 18.4088, 73.9857, 18.6298],
      'ahmedabad': [72.4194, 22.9734, 72.6947, 23.1636],
      'jaipur': [75.6499, 26.8105, 76.0399, 27.0238],
      'surat': [72.6369, 21.0702, 72.9489, 21.2787],
      'lucknow': [80.7718, 26.6307, 81.0861, 27.0047],
      'kanpur': [80.2319, 26.3598, 80.5562, 26.5499],
      'nagpur': [78.9629, 21.0514, 79.2423, 21.2514],
      'indore': [75.6876, 22.6273, 76.0013, 22.8171],
      'thane': [72.9375, 19.1136, 73.0297, 19.2183],
      'bhopal': [77.2497, 23.1585, 77.5370, 23.3441],
      'visakhapatnam': [83.1777, 17.6599, 83.3532, 17.7731],
      'pimpri': [73.7672, 18.6186, 73.8390, 18.6745],
      'patna': [85.0002, 25.5020, 85.2401, 25.6751],
      'vadodara': [73.0169, 22.2587, 73.2815, 22.3894],
      'ludhiana': [75.7849, 30.8320, 75.9349, 30.9320],
      'agra': [77.9126, 27.1303, 78.0845, 27.2479],
      'nashik': [73.6816, 19.9975, 73.8370, 20.0110],
      'faridabad': [77.2674, 28.3670, 77.3674, 28.4670],
      'meerut': [77.6687, 28.9685, 77.7687, 29.0685],
      'rajkot': [70.7429, 22.2587, 70.8429, 22.3587],
      'kalyan': [73.1340, 19.2183, 73.2340, 19.3183],
      'vasai': [72.7757, 19.3919, 72.8757, 19.4919],
      'varanasi': [82.9739, 25.2677, 83.0739, 25.3677],
      'srinagar': [74.7973, 34.0837, 74.8973, 34.1837],
      'aurangabad': [75.2933, 19.8762, 75.3933, 19.9762],
      'dhanbad': [86.4304, 23.7957, 86.5304, 23.8957],
      'amritsar': [74.8723, 31.6340, 74.9723, 31.7340],
      'navi mumbai': [73.0297, 19.0330, 73.1297, 19.1330],
      'allahabad': [81.8463, 25.4358, 81.9463, 25.5358],
      'ranchi': [85.2672, 23.3441, 85.3672, 23.4441],
      'howrah': [88.2636, 22.5958, 88.3636, 22.6958],
      'coimbatore': [76.9366, 11.0168, 77.0366, 11.1168],
      'jabalpur': [79.9864, 23.1815, 80.0864, 23.2815],
      'gwalior': [78.1828, 26.2124, 78.2828, 26.3124],
      'vijayawada': [80.5562, 16.5062, 80.6562, 16.6062],
      'jodhpur': [73.0169, 26.2389, 73.1169, 26.3389],
      'madurai': [78.0747, 9.9252, 78.1747, 10.0252],
      'raipur': [81.5404, 21.2514, 81.6404, 21.3514],
      'kota': [75.8648, 25.2138, 75.9648, 25.3138],
      'chandigarh': [76.7635, 30.7333, 76.8635, 30.8333],
      'guwahati': [91.7362, 26.1445, 91.8362, 26.2445],
      'solapur': [75.8648, 17.6599, 75.9648, 17.7599],
      'hubli': [75.1240, 15.3647, 75.2240, 15.4647],
      'bareilly': [79.4304, 28.3670, 79.5304, 28.4670],
      'moradabad': [78.7733, 28.8386, 78.8733, 28.9386],
      'mysore': [76.6394, 12.2958, 76.7394, 12.3958],
      'gurgaon': [77.0266, 28.4595, 77.1266, 28.5595],
      'aligarh': [78.0747, 27.8974, 78.1747, 27.9974],
      'jalandhar': [75.5762, 31.3260, 75.6762, 31.4260],
      'tiruchirappalli': [78.7047, 10.7905, 78.8047, 10.8905],
      'bhubaneswar': [85.7982, 20.2961, 85.8982, 20.3961],
      'salem': [78.1460, 11.6643, 78.2460, 11.7643],
      'warangal': [79.5881, 17.9689, 79.6881, 18.0689],
      'mira': [72.8757, 19.2919, 72.9757, 19.3919],
      'thiruvananthapuram': [76.9366, 8.5241, 77.0366, 8.6241],
      'bhiwandi': [73.0297, 19.2919, 73.1297, 19.3919],
      'saharanpur': [77.5463, 29.9680, 77.6463, 30.0680],
      'guntur': [80.4365, 16.2970, 80.5365, 16.3970],
      'amravati': [77.7499, 20.9374, 77.8499, 21.0374],
      'bikaner': [73.3119, 28.0229, 73.4119, 28.1229],
      'noida': [77.3910, 28.5355, 77.4910, 28.6355],
      'jamshedpur': [86.1844, 22.8046, 86.2844, 22.9046],
      'bhilai nagar': [81.3509, 21.1938, 81.4509, 21.2938],
      'cuttack': [85.8245, 20.4625, 85.9245, 20.5625],
      'firozabad': [78.3941, 27.1592, 78.4941, 27.2592],
      'kochi': [76.2144, 9.9312, 76.3144, 10.0312],
      'bhavnagar': [72.1019, 21.7645, 72.2019, 21.8645],
      'dehradun': [78.0322, 30.3165, 78.1322, 30.4165],
      'durgapur': [87.3119, 23.4833, 87.4119, 23.5833],
      'asansol': [86.9842, 23.6739, 87.0842, 23.7739],
      'nanded': [77.2663, 19.1383, 77.3663, 19.2383],
      'kolhapur': [74.2433, 16.7050, 74.3433, 16.8050],
      'ajmer': [74.6399, 26.4499, 74.7399, 26.5499],
      'akola': [77.0082, 20.7002, 77.1082, 20.8002],
      'gulbarga': [76.8343, 17.3297, 76.9343, 17.4297],
      'jamnagar': [70.0692, 22.4697, 70.1692, 22.5697],
      'ujjain': [75.7849, 23.1765, 75.8849, 23.2765],
      'loni': [77.2863, 28.7594, 77.3863, 28.8594],
      'siliguri': [88.3953, 26.7271, 88.4953, 26.8271],
      'jhansi': [78.5685, 25.4484, 78.6685, 25.5484],
      'ulhasnagar': [73.1340, 19.2183, 73.2340, 19.3183],
      'jammu': [74.8723, 32.7266, 74.9723, 32.8266],
      'sangli': [74.5815, 16.8524, 74.6815, 16.9524],
      'mangalore': [74.7972, 12.9141, 74.8972, 13.0141],
      'erode': [77.7172, 11.3410, 77.8172, 11.4410],
      'belgaum': [74.4977, 15.8497, 74.5977, 15.9497],
      'ambattur': [80.1623, 13.0983, 80.2623, 13.1983],
      'tirunelveli': [77.6869, 8.7139, 77.7869, 8.8139],
      'malegaon': [74.5815, 20.5579, 74.6815, 20.6579],
      'gaya': [84.9994, 24.7914, 85.0994, 24.8914],
      'jalgaon': [75.5648, 21.0077, 75.6648, 21.1077],
      'udaipur': [73.6816, 24.5854, 73.7816, 24.6854],
      'maheshtala': [88.2477, 22.4697, 88.3477, 22.5697],
      'silvassa': [73.0169, 20.2737, 73.1169, 20.3737]
    };

    // Return coordinates for the city, or default to Delhi if not found
    return cityCoordinates[city] || cityCoordinates['delhi'];
  }
}

export const urbanAGBService = new UrbanAGBService();
export default urbanAGBService;