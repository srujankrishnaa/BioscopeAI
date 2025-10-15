import React, { useState } from 'react';
import Navbar from '../components/Navbar';
import Footer from '../components/Footer';
import { Globe } from '../components/ui/globe';

const MapExplorerPage: React.FC = () => {
  // State for selected location and data
  const [selectedLocation, setSelectedLocation] = useState<string | null>(null);
  const [isDataPanelOpen, setIsDataPanelOpen] = useState(false);
  
  // Mock locations data
  const locations = [
    { id: 'hyd', name: 'Hyderabad', coordinates: [17.4065, 78.4772] as [number, number], biomassIndex: 87, trend: '+2.3%', health: 'Healthy' },
    { id: 'mum', name: 'Mumbai', coordinates: [19.076, 72.8777] as [number, number], biomassIndex: 72, trend: '-1.5%', health: 'Warning' },
    { id: 'blr', name: 'Bangalore', coordinates: [12.9716, 77.5946] as [number, number], biomassIndex: 91, trend: '+3.7%', health: 'Healthy' },
    { id: 'del', name: 'Delhi', coordinates: [28.7041, 77.1025] as [number, number], biomassIndex: 64, trend: '-3.2%', health: 'Critical' },
    { id: 'che', name: 'Chennai', coordinates: [13.0827, 80.2707] as [number, number], biomassIndex: 79, trend: '+0.8%', health: 'Moderate' },
    { id: 'kol', name: 'Kolkata', coordinates: [22.5726, 88.3639] as [number, number], biomassIndex: 76, trend: '+1.1%', health: 'Moderate' },
    { id: 'pun', name: 'Pune', coordinates: [18.5204, 73.8567] as [number, number], biomassIndex: 85, trend: '+2.8%', health: 'Healthy' },
    { id: 'ahm', name: 'Ahmedabad', coordinates: [23.0225, 72.5714] as [number, number], biomassIndex: 68, trend: '-2.1%', health: 'Warning' },
  ];
  
  // Get selected location data
  const getSelectedLocationData = () => {
    return locations.find(loc => loc.id === selectedLocation);
  };
  
  // Handle location selection
  const handleLocationSelect = (locationId: string) => {
    setSelectedLocation(locationId);
    setIsDataPanelOpen(true);
  };
  
  // Mock time series data for biomass trends
  const getTimeSeriesData = () => {
    return [
      { month: 'Jan', value: 65 },
      { month: 'Feb', value: 68 },
      { month: 'Mar', value: 72 },
      { month: 'Apr', value: 75 },
      { month: 'May', value: 79 },
      { month: 'Jun', value: 82 },
      { month: 'Jul', value: 85 },
      { month: 'Aug', value: 87 },
    ];
  };

  const selectedLocationData = getSelectedLocationData();

  return (
    <div className="min-h-screen bg-green text-off-white font-graphik">
      <Navbar />
      
      <div className="pt-20 px-6">
        <div className="max-w-7xl mx-auto">
          {/* Page Header */}
          <div className="text-center mb-8">
            <h1 className="text-4xl font-bold font-deacon mb-4">GLOBAL BIOMASS EXPLORER</h1>
            <p className="text-off-white/70 max-w-2xl mx-auto">
              Explore real-time biomass data from around the world. Click on any location marker to view detailed analysis and trends.
            </p>
          </div>
          
          {/* Main Content */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-12">
            {/* Globe Section */}
            <div className="lg:col-span-2">
              <div className="bg-off-white/5 backdrop-blur-sm rounded-2xl p-8 border border-off-white/10">
                <div className="flex justify-between items-center mb-6">
                  <h2 className="text-xl font-bold">Interactive World Map</h2>
                  <div className="flex items-center space-x-4">
                    <div className="flex items-center space-x-2">
                      <div className="w-3 h-3 bg-neon-100 rounded-full"></div>
                      <span className="text-sm text-off-white/70">Monitoring Stations</span>
                    </div>
                  </div>
                </div>
                
                <div className="relative h-96 flex items-center justify-center">
                  <Globe 
                    className="opacity-90"
                    config={{
                      width: 600,
                      height: 600,
                      onRender: () => {},
                      devicePixelRatio: 2,
                      phi: 0,
                      theta: 0.3,
                      dark: 0.3,
                      diffuse: 0.4,
                      mapSamples: 16000,
                      mapBrightness: 0.8,
                      baseColor: [0.2, 0.4, 0.2],
                      markerColor: [85 / 255, 221 / 255, 74 / 255],
                      glowColor: [0.2, 0.4, 0.2],
                      markers: locations.map(loc => ({
                        location: loc.coordinates,
                        size: 0.08
                      })),
                    }}
                  />
                </div>
                
                <div className="mt-6 text-center">
                  <p className="text-off-white/60 text-sm">
                    Drag to rotate • Scroll to zoom • Click markers for details
                  </p>
                </div>
              </div>
            </div>
            
            {/* Location List & Data Panel */}
            <div className="space-y-6">
              {/* Location List */}
              <div className="bg-off-white/5 backdrop-blur-sm rounded-2xl p-6 border border-off-white/10">
                <h3 className="text-lg font-bold mb-4">Monitoring Locations</h3>
                <div className="space-y-3 max-h-64 overflow-y-auto">
                  {locations.map((location) => (
                    <button
                      key={location.id}
                      type="button"
                      onClick={() => handleLocationSelect(location.id)}
                      className={`w-full text-left p-3 rounded-lg transition-all ${
                        selectedLocation === location.id 
                          ? 'bg-neon-100/20 border border-neon-100/30' 
                          : 'hover:bg-off-white/10 border border-transparent'
                      }`}
                    >
                      <div className="flex justify-between items-center">
                        <div>
                          <h4 className="font-medium">{location.name}</h4>
                          <p className="text-xs text-off-white/70">
                            {location.coordinates[0].toFixed(2)}°, {location.coordinates[1].toFixed(2)}°
                          </p>
                        </div>
                        <div className="text-right">
                          <div className={`text-sm font-bold ${
                            location.health === 'Healthy' ? 'text-neon-100' :
                            location.health === 'Warning' ? 'text-yellow-500' :
                            location.health === 'Critical' ? 'text-red-500' :
                            'text-blue-400'
                          }`}>
                            {location.biomassIndex}%
                          </div>
                          <div className={`text-xs ${
                            location.trend.startsWith('+') ? 'text-neon-100' : 'text-red-400'
                          }`}>
                            {location.trend}
                          </div>
                        </div>
                      </div>
                    </button>
                  ))}
                </div>
              </div>
              
              {/* Data Panel */}
              {selectedLocationData && (
                <div className="bg-off-white/5 backdrop-blur-sm rounded-2xl p-6 border border-off-white/10">
                  <div className="flex justify-between items-center mb-4">
                    <h3 className="text-lg font-bold">{selectedLocationData.name}</h3>
                    <button 
                      type="button"
                      onClick={() => setIsDataPanelOpen(false)}
                      className="text-off-white/50 hover:text-off-white"
                      aria-label="Close data panel"
                      title="Close data panel"
                    >
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                      </svg>
                    </button>
                  </div>
                  
                  {/* Key Metrics */}
                  <div className="grid grid-cols-2 gap-4 mb-6">
                    <div className="text-center p-3 bg-off-white/10 rounded-lg">
                      <div className="text-2xl font-bold text-neon-100">{selectedLocationData.biomassIndex}%</div>
                      <div className="text-xs text-off-white/70">Biomass Index</div>
                    </div>
                    <div className="text-center p-3 bg-off-white/10 rounded-lg">
                      <div className={`text-2xl font-bold ${
                        selectedLocationData.trend.startsWith('+') ? 'text-neon-100' : 'text-red-400'
                      }`}>
                        {selectedLocationData.trend}
                      </div>
                      <div className="text-xs text-off-white/70">Monthly Change</div>
                    </div>
                  </div>
                  
                  {/* Health Status */}
                  <div className="mb-6">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm text-off-white/70">Health Status</span>
                      <span className={`text-sm font-medium ${
                        selectedLocationData.health === 'Healthy' ? 'text-neon-100' :
                        selectedLocationData.health === 'Warning' ? 'text-yellow-500' :
                        selectedLocationData.health === 'Critical' ? 'text-red-500' :
                        'text-blue-400'
                      }`}>
                        {selectedLocationData.health}
                      </span>
                    </div>
                    <div className="w-full bg-off-white/20 rounded-full h-2">
                      <div 
                        className={`h-2 rounded-full ${
                          selectedLocationData.health === 'Healthy' ? 'bg-neon-100' :
                          selectedLocationData.health === 'Warning' ? 'bg-yellow-500' :
                          selectedLocationData.health === 'Critical' ? 'bg-red-500' :
                          'bg-blue-400'
                        }`}
                        style={{ width: `${selectedLocationData.biomassIndex}%` }}
                      ></div>
                    </div>
                  </div>
                  
                  {/* Mini Chart */}
                  <div className="mb-4">
                    <h4 className="text-sm font-medium mb-3">8-Month Trend</h4>
                    <div className="flex items-end justify-between h-16 space-x-1">
                      {getTimeSeriesData().map((data, index) => (
                        <div key={index} className="flex flex-col items-center flex-1">
                          <div 
                            className="w-full bg-neon-100/30 rounded-t"
                            style={{ height: `${(data.value / 100) * 100}%` }}
                          ></div>
                          <span className="text-xs text-off-white/50 mt-1">{data.month}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                  
                  {/* Action Buttons */}
                  <div className="space-y-2">
                    <button 
                      type="button"
                      className="w-full bg-neon-100 text-green px-4 py-2 rounded-lg font-medium text-sm hover:bg-neon-80 transition-all"
                    >
                      View Detailed Report
                    </button>
                    <button 
                      type="button"
                      className="w-full border border-off-white/20 text-off-white px-4 py-2 rounded-lg text-sm hover:bg-off-white/10 transition-all"
                    >
                      Download Data
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>
          
          {/* Quick Stats */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-12">
            <div className="bg-off-white/5 backdrop-blur-sm rounded-xl p-6 border border-off-white/10 text-center">
              <div className="text-3xl font-bold text-neon-100 mb-2">{locations.length}</div>
              <div className="text-off-white/70 text-sm">Active Stations</div>
            </div>
            <div className="bg-off-white/5 backdrop-blur-sm rounded-xl p-6 border border-off-white/10 text-center">
              <div className="text-3xl font-bold text-neon-100 mb-2">
                {Math.round(locations.reduce((acc, loc) => acc + loc.biomassIndex, 0) / locations.length)}%
              </div>
              <div className="text-off-white/70 text-sm">Avg Biomass Index</div>
            </div>
            <div className="bg-off-white/5 backdrop-blur-sm rounded-xl p-6 border border-off-white/10 text-center">
              <div className="text-3xl font-bold text-neon-100 mb-2">
                {locations.filter(loc => loc.health === 'Healthy').length}
              </div>
              <div className="text-off-white/70 text-sm">Healthy Locations</div>
            </div>
            <div className="bg-off-white/5 backdrop-blur-sm rounded-xl p-6 border border-off-white/10 text-center">
              <div className="text-3xl font-bold text-yellow-500 mb-2">
                {locations.filter(loc => loc.health === 'Warning' || loc.health === 'Critical').length}
              </div>
              <div className="text-off-white/70 text-sm">Need Attention</div>
            </div>
          </div>
        </div>
      </div>
      
      <Footer />
    </div>
  );
};

export default MapExplorerPage;