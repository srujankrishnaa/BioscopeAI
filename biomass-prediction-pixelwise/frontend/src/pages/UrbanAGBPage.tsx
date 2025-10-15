import React, { useState, useEffect } from 'react';
import Navbar from '../components/Navbar';
import Footer from '../components/Footer';
import { MagnetizeButton } from '../components/ui/magnetize-button';
import urbanAGBService, { UrbanAGBResponse, SystemStatus } from '../services/urbanAgbService';

const UrbanAGBPage: React.FC = () => {
  const [cityInput, setCityInput] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<UrbanAGBResponse | null>(null);
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);

  // Load system status on component mount
  useEffect(() => {
    loadSystemStatus();
  }, []);

  const loadSystemStatus = async () => {
    try {
      const status = await urbanAGBService.getSystemStatus();
      setSystemStatus(status);
    } catch (error) {
      console.error('Failed to load system status:', error);
    }
  };

  const handleQuickSelect = (cityName: string) => {
    setCityInput(cityName);
  };

  const handleUrbanAGBPrediction = async () => {
    if (!cityInput.trim()) {
      setError('Please enter a city name');
      return;
    }

    setIsAnalyzing(true);
    setProgress(0);
    setAnalysisResult(null);
    setError(null);

    // Progress simulation for UX
    const progressInterval = setInterval(() => {
      setProgress(prev => {
        if (prev >= 90) {
          clearInterval(progressInterval);
          return 90;
        }
        return prev + Math.random() * 15;
      });
    }, 500);

    try {
      const result = await urbanAGBService.predictUrbanAGB({
        city: cityInput.trim()
      });

      clearInterval(progressInterval);
      setProgress(100);

      setTimeout(() => {
        setAnalysisResult(result);
        setIsAnalyzing(false);
        setProgress(0);
      }, 1000);

    } catch (error) {
      clearInterval(progressInterval);
      setProgress(0);
      setIsAnalyzing(false);
      setError(error instanceof Error ? error.message : 'Analysis failed');
    }
  };

  const predefinedCities = urbanAGBService.getPredefinedCities();

  return (
    <div className="min-h-screen bg-green text-off-white font-graphik">
      <Navbar />
      
      <div className="pt-20 pb-12">
        <div className="max-w-7xl mx-auto px-6">
          {/* Header */}
          <div className="text-center mb-12">
            <h1 className="text-5xl md:text-6xl font-black font-deacon text-off-white leading-tight mb-6">
              URBAN AGB<br />
              <span className="text-neon-100">PREDICTION</span>
            </h1>
            <p className="text-xl text-off-white/80 max-w-3xl mx-auto leading-relaxed">
              Real-time NASA satellite data • 3-year biomass forecasting • Urban planning recommendations
              <br />
              <span className="text-neon-100 font-semibold">Powered by 603,943 GEDI L4A measurements</span>
            </p>
          </div>

          {/* System Status */}
          {systemStatus && (
            <div className="max-w-4xl mx-auto mb-8">
              <div className="bg-off-white/10 backdrop-blur-md rounded-xl p-6 border border-off-white/20">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-4">
                    <div className={`w-3 h-3 rounded-full ${
                      Object.values(systemStatus.systems).every(s => s.status === 'ready') 
                        ? 'bg-green-500' 
                        : 'bg-yellow-500'
                    }`}></div>
                    <span className="font-semibold">
                      System Status: {Object.values(systemStatus.systems).filter(s => s.status === 'ready').length}/
                      {Object.keys(systemStatus.systems).length} Ready
                    </span>
                  </div>
                  <div className="text-sm text-off-white/70">
                    Model Accuracy: R² = 0.99+ • Processing: &lt; 30s • Coverage: Global
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Input Section */}
          <div className="max-w-4xl mx-auto mb-12">
            <div className="bg-off-white/10 backdrop-blur-md rounded-2xl p-8 border border-off-white/20 shadow-xl">
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                {/* City Input */}
                <div className="lg:col-span-2">
                  <label htmlFor="cityInput" className="block text-lg font-semibold text-off-white mb-3">
                    🏙️ Enter City Name
                  </label>
                  <input
                    id="cityInput"
                    type="text"
                    value={cityInput}
                    onChange={(e) => setCityInput(e.target.value)}
                    placeholder="e.g., Bangalore, Mumbai, New York, London..."
                    className="w-full px-6 py-4 bg-off-white/10 border border-off-white/30 rounded-xl focus:outline-none focus:ring-2 focus:ring-neon-100 focus:border-transparent text-off-white placeholder-off-white/50 text-lg"
                    disabled={isAnalyzing}
                    onKeyPress={(e) => e.key === 'Enter' && handleUrbanAGBPrediction()}
                  />
                  
                  <MagnetizeButton
                    variant="magnetize"
                    size="xl"
                    particleCount={35}
                    onClick={handleUrbanAGBPrediction}
                    disabled={isAnalyzing || !cityInput.trim()}
                    className="w-full mt-4 py-4 text-xl font-bold"
                  >
                    {isAnalyzing ? '🛰️ ANALYZING...' : '🔮 PREDICT URBAN AGB'}
                  </MagnetizeButton>
                </div>

                {/* Quick Select */}
                <div>
                  <label className="block text-lg font-semibold text-off-white mb-3">
                    🚀 Quick Select
                  </label>
                  <div className="space-y-2 max-h-64 overflow-y-auto">
                    {predefinedCities.slice(0, 8).map((city) => (
                      <button
                        key={city.id}
                        onClick={() => handleQuickSelect(city.name)}
                        disabled={isAnalyzing}
                        className="w-full text-left p-3 bg-off-white/5 hover:bg-off-white/15 rounded-lg transition-all border border-off-white/10 hover:border-off-white/30"
                      >
                        <div className="font-medium">{city.name}</div>
                        <div className="text-xs text-off-white/70">{city.country}</div>
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Progress Bar */}
          {isAnalyzing && (
            <div className="max-w-4xl mx-auto mb-12">
              <div className="bg-off-white/10 backdrop-blur-sm rounded-xl p-6 border border-off-white/20">
                <div className="flex items-center justify-between mb-3">
                  <span className="text-off-white font-medium">
                    🛰️ Fetching NASA satellite data and generating Urban AGB prediction...
                  </span>
                  <span className="text-neon-100 font-bold">{Math.round(progress)}%</span>
                </div>
                <div className="w-full bg-off-white/20 rounded-full h-3">
                  <div 
                    className="bg-neon-100 h-3 rounded-full transition-all duration-300 shadow-lg"
                    style={{ width: `${progress}%` }}
                  ></div>
                </div>
                <div className="mt-3 text-off-white/70 text-sm">
                  {progress < 30 && "🌍 Geocoding city location..."}
                  {progress >= 30 && progress < 60 && "📡 Fetching real-time satellite data..."}
                  {progress >= 60 && progress < 90 && "🤖 Running ML models with GEDI L4A data..."}
                  {progress >= 90 && "🗺️ Generating heat map visualization..."}
                </div>
              </div>
            </div>
          )}

          {/* Error Display */}
          {error && (
            <div className="max-w-4xl mx-auto mb-12">
              <div className="bg-red-500/20 border border-red-500/30 rounded-xl p-6">
                <div className="flex items-center space-x-3">
                  <div className="text-red-500">❌</div>
                  <div>
                    <h3 className="font-semibold text-red-400">Prediction Error</h3>
                    <p className="text-red-300">{error}</p>
                    <p className="text-red-300/70 text-sm mt-1">
                      Please try again or check if the backend server is running.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Results Section */}
          {analysisResult && (
            <div className="space-y-8">
              {/* City Header */}
              <div className="text-center">
                <h2 className="text-4xl font-bold font-deacon mb-4">
                  Urban AGB Analysis - {analysisResult.city}
                </h2>
                <div className="bg-blue-500/20 backdrop-blur-sm rounded-xl p-6 border border-blue-500/30 max-w-2xl mx-auto">
                  <div className="text-2xl font-bold text-blue-400 mb-2">📍 {analysisResult.location.coordinates}</div>
                  <div className="text-blue-300">Analysis completed: {new Date(analysisResult.timestamp).toLocaleString()}</div>
                </div>
              </div>

              {/* Heat Map Visualization */}
              {analysisResult.heat_map.image_url && (
                <div className="bg-off-white/10 backdrop-blur-sm rounded-2xl p-8 border border-off-white/20">
                  <h3 className="text-2xl font-bold text-center mb-6">🗺️ Urban AGB Heat Map Analysis</h3>
                  <div className="text-center">
                    <img 
                      src={analysisResult.heat_map.image_url} 
                      alt="Urban AGB Heat Map"
                      className="max-w-full h-auto rounded-xl shadow-lg mx-auto"
                      style={{ maxHeight: '600px' }}
                    />
                    <p className="text-off-white/70 text-sm mt-4">
                      Comprehensive 4-panel analysis: Current AGB • 3-Year Forecast • Canopy Cover • Cooling Potential
                    </p>
                  </div>
                </div>
              )}

              {/* Key Metrics Grid */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                <div className="bg-off-white/10 backdrop-blur-sm rounded-xl p-6 border border-off-white/20 text-center">
                  <div className="text-4xl font-black text-neon-100 mb-2">
                    {urbanAGBService.formatBiomass(analysisResult.current_agb.total_agb)}
                  </div>
                  <div className="text-off-white/70 text-sm">Current AGB</div>
                  <div className={`text-xs mt-1 ${urbanAGBService.getHealthStatusColor(analysisResult.current_agb.total_agb)}`}>
                    {urbanAGBService.getHealthStatusText(analysisResult.current_agb.total_agb)}
                  </div>
                </div>

                <div className="bg-off-white/10 backdrop-blur-sm rounded-xl p-6 border border-off-white/20 text-center">
                  <div className="text-4xl font-black text-neon-100 mb-2">
                    {urbanAGBService.formatPercentage(analysisResult.current_agb.canopy_cover)}
                  </div>
                  <div className="text-off-white/70 text-sm">Canopy Cover</div>
                  <div className="text-neon-100 text-xs mt-1">Tree Cities Standard: 30%+</div>
                </div>

                <div className="bg-off-white/10 backdrop-blur-sm rounded-xl p-6 border border-off-white/20 text-center">
                  <div className="text-4xl font-black text-neon-100 mb-2">
                    {urbanAGBService.formatTemperature(analysisResult.current_agb.cooling_potential)}
                  </div>
                  <div className="text-off-white/70 text-sm">Cooling Potential</div>
                  <div className="text-neon-100 text-xs mt-1">Urban Heat Reduction</div>
                </div>

                <div className="bg-off-white/10 backdrop-blur-sm rounded-xl p-6 border border-off-white/20 text-center">
                  <div className="text-4xl font-black text-neon-100 mb-2">
                    {analysisResult.current_agb.carbon_sequestration.toFixed(2)}
                  </div>
                  <div className="text-off-white/70 text-sm">Mg C/ha/year</div>
                  <div className="text-neon-100 text-xs mt-1">Carbon Sequestration</div>
                </div>
              </div>

              {/* Detailed Metrics */}
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                {/* Current Biomass Breakdown */}
                <div className="bg-off-white/10 backdrop-blur-sm rounded-xl p-6 border border-off-white/20">
                  <h4 className="text-xl font-bold text-neon-100 mb-4">🌳 Biomass Composition</h4>
                  <div className="space-y-3">
                    <div className="flex justify-between items-center">
                      <span className="text-off-white/80">Tree Biomass</span>
                      <span className="font-bold text-green-400">
                        {urbanAGBService.formatBiomass(analysisResult.current_agb.tree_biomass)}
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-off-white/80">Shrub Biomass</span>
                      <span className="font-bold text-blue-400">
                        {urbanAGBService.formatBiomass(analysisResult.current_agb.shrub_biomass)}
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-off-white/80">Herbaceous</span>
                      <span className="font-bold text-yellow-400">
                        {urbanAGBService.formatBiomass(analysisResult.current_agb.herbaceous_biomass)}
                      </span>
                    </div>
                    <div className="pt-3 border-t border-off-white/20">
                      <div className="flex justify-between items-center">
                        <span className="font-semibold">Total AGB</span>
                        <span className="font-bold text-neon-100">
                          {urbanAGBService.formatBiomass(analysisResult.current_agb.total_agb)}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Satellite Data */}
                <div className="bg-off-white/10 backdrop-blur-sm rounded-xl p-6 border border-off-white/20">
                  <h4 className="text-xl font-bold text-neon-100 mb-4">🛰️ Real-time Satellite Data</h4>
                  <div className="space-y-3">
                    <div className="flex justify-between items-center">
                      <span className="text-off-white/80">NDVI</span>
                      <span className="font-bold text-green-400">
                        {analysisResult.satellite_data.ndvi.toFixed(3)}
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-off-white/80">EVI</span>
                      <span className="font-bold text-blue-400">
                        {analysisResult.satellite_data.evi.toFixed(3)}
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-off-white/80">LAI</span>
                      <span className="font-bold text-yellow-400">
                        {analysisResult.satellite_data.lai.toFixed(3)}
                      </span>
                    </div>
                    <div className="pt-3 border-t border-off-white/20">
                      <div className="text-xs text-off-white/70">
                        Data Source: {analysisResult.satellite_data.data_source}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Urban Planning Metrics */}
                <div className="bg-off-white/10 backdrop-blur-sm rounded-xl p-6 border border-off-white/20">
                  <h4 className="text-xl font-bold text-neon-100 mb-4">🏙️ Planning Metrics</h4>
                  <div className="space-y-3">
                    <div className="flex justify-between items-center">
                      <span className="text-off-white/80">EPI Score</span>
                      <span className="font-bold text-green-400">
                        {analysisResult.urban_metrics.epi_score}/100
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-off-white/80">Tree Cities Score</span>
                      <span className="font-bold text-blue-400">
                        {analysisResult.urban_metrics.tree_cities_score}/100
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-off-white/80">Green Space Ratio</span>
                      <span className="font-bold text-yellow-400">
                        {urbanAGBService.formatPercentage(analysisResult.urban_metrics.green_space_ratio)}
                      </span>
                    </div>
                    <div className="pt-3 border-t border-off-white/20">
                      <div className="text-xs text-off-white/70">
                        Priority: {urbanAGBService.getInterventionPriority(
                          analysisResult.urban_metrics.epi_score,
                          analysisResult.urban_metrics.tree_cities_score
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* 3-Year Forecasting */}
              <div className="bg-gradient-to-r from-purple-500/20 to-blue-500/20 backdrop-blur-sm rounded-2xl p-8 border border-purple-500/30">
                <h3 className="text-2xl font-bold text-center mb-6">🔮 3-Year Biomass Forecasting</h3>
                <div className="grid grid-cols-2 md:grid-cols-5 gap-6">
                  <div className="text-center">
                    <div className="bg-off-white/10 rounded-lg p-4">
                      <div className="text-sm text-off-white/70 mb-1">Current (2025)</div>
                      <div className="text-2xl font-bold text-white">
                        {urbanAGBService.formatBiomass(analysisResult.current_agb.total_agb)}
                      </div>
                    </div>
                  </div>
                  <div className="text-center">
                    <div className="bg-off-white/10 rounded-lg p-4">
                      <div className="text-sm text-off-white/70 mb-1">Year 1 (2026)</div>
                      <div className="text-2xl font-bold text-green-400">
                        {urbanAGBService.formatBiomass(analysisResult.forecasting.year_1)}
                      </div>
                    </div>
                  </div>
                  <div className="text-center">
                    <div className="bg-off-white/10 rounded-lg p-4">
                      <div className="text-sm text-off-white/70 mb-1">Year 3 (2028)</div>
                      <div className="text-2xl font-bold text-blue-400">
                        {urbanAGBService.formatBiomass(analysisResult.forecasting.year_3)}
                      </div>
                    </div>
                  </div>
                  <div className="text-center">
                    <div className="bg-off-white/10 rounded-lg p-4">
                      <div className="text-sm text-off-white/70 mb-1">Year 5 (2030)</div>
                      <div className="text-2xl font-bold text-purple-400">
                        {urbanAGBService.formatBiomass(analysisResult.forecasting.year_5)}
                      </div>
                    </div>
                  </div>
                  <div className="text-center">
                    <div className="bg-off-white/10 rounded-lg p-4">
                      <div className="text-sm text-off-white/70 mb-1">Growth Rate</div>
                      <div className="text-2xl font-bold text-neon-100">
                        +{urbanAGBService.formatPercentage(analysisResult.forecasting.growth_rate * 100)}
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Planning Recommendations */}
              <div className="bg-off-white/10 backdrop-blur-sm rounded-2xl p-8 border border-off-white/20">
                <h3 className="text-2xl font-bold text-neon-100 mb-6">💡 Urban Planning Recommendations</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {analysisResult.planning_recommendations.map((recommendation, index) => (
                    <div key={index} className="bg-off-white/5 rounded-lg p-4 border-l-4 border-neon-100">
                      <p className="text-off-white/90">{recommendation}</p>
                    </div>
                  ))}
                </div>
              </div>

              {/* Model Performance */}
              <div className="bg-gradient-to-r from-green-500/20 to-blue-500/20 backdrop-blur-sm rounded-xl p-6 border border-green-500/30">
                <h4 className="text-xl font-bold text-center mb-4">🎯 Model Performance</h4>
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4 text-center">
                  <div>
                    <div className="text-lg font-bold text-green-400">{analysisResult.model_performance.accuracy}</div>
                    <div className="text-xs text-off-white/70">Accuracy</div>
                  </div>
                  <div>
                    <div className="text-lg font-bold text-blue-400">{analysisResult.model_performance.ground_truth}</div>
                    <div className="text-xs text-off-white/70">Ground Truth</div>
                  </div>
                  <div>
                    <div className="text-lg font-bold text-yellow-400">{analysisResult.model_performance.processing_time}</div>
                    <div className="text-xs text-off-white/70">Processing Time</div>
                  </div>
                  <div>
                    <div className="text-lg font-bold text-purple-400">{analysisResult.model_performance.geographic_coverage}</div>
                    <div className="text-xs text-off-white/70">Coverage</div>
                  </div>
                </div>
              </div>

              {/* Action Buttons */}
              <div className="flex flex-col sm:flex-row gap-4 justify-center">
                <MagnetizeButton
                  variant="magnetize"
                  size="lg"
                  particleCount={25}
                  className="px-8 py-3"
                >
                  📊 DOWNLOAD REPORT
                </MagnetizeButton>
                <MagnetizeButton
                  variant="outline"
                  size="lg"
                  particleCount={20}
                  className="border-2 border-off-white text-off-white hover:bg-off-white hover:text-green px-8 py-3"
                >
                  📅 SCHEDULE MONITORING
                </MagnetizeButton>
                <MagnetizeButton
                  variant="outline"
                  size="lg"
                  particleCount={20}
                  className="border-2 border-neon-100 text-neon-100 hover:bg-neon-100 hover:text-green px-8 py-3"
                  onClick={() => {
                    setAnalysisResult(null);
                    setCityInput('');
                    setError(null);
                  }}
                >
                  🔄 NEW ANALYSIS
                </MagnetizeButton>
              </div>
            </div>
          )}
        </div>
      </div>

      <Footer />
    </div>
  );
};

export default UrbanAGBPage;