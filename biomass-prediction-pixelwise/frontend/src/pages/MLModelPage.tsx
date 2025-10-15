import React, { useState, useEffect } from 'react';
import Navbar from '../components/Navbar';
import Footer from '../components/Footer';
import { MagnetizeButton } from '../components/ui/magnetize-button';
import RegionSelector from '../components/RegionSelector';
import urbanAGBService, { UrbanAGBResponse, SystemStatus } from '../services/urbanAgbService';
import regionService, { RegionData, CityRegionsResponse } from '../services/regionService';

const MLModelPage: React.FC = () => {
  const [cityInput, setCityInput] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<UrbanAGBResponse | null>(null);
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [scrollY, setScrollY] = useState(0);

  // Regional selection state
  const [showRegionSelection, setShowRegionSelection] = useState(false);
  const [cityRegions, setCityRegions] = useState<CityRegionsResponse | null>(null);
  const [isLoadingRegions, setIsLoadingRegions] = useState(false);
  const [selectedRegion, setSelectedRegion] = useState<RegionData | null>(null);

  // Track scroll position for animations
  useEffect(() => {
    const handleScroll = () => setScrollY(window.scrollY);
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

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
    // Smooth scroll to results section
    const inputSection = document.getElementById('input-section');
    if (inputSection) {
      inputSection.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  };

  const handleAnalysis = async () => {
    if (!cityInput.trim()) {
      setError('Please enter a city name');
      return;
    }

    setIsLoadingRegions(true);
    setError(null);
    setAnalysisResult(null);

    try {
      // First, get city regions for selection
      const regions = await regionService.getCityRegions(cityInput.trim());
      setCityRegions(regions);
      setShowRegionSelection(true);

      // Scroll to region selection
      setTimeout(() => {
        const regionSection = document.getElementById('region-selection-section');
        if (regionSection) {
          regionSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
      }, 100);

    } catch (error) {
      setError(error instanceof Error ? error.message : 'Failed to load city regions');
    } finally {
      setIsLoadingRegions(false);
    }
  };

  const handleRegionSelect = async (region: RegionData) => {
    setSelectedRegion(region);
    setIsAnalyzing(true);
    setProgress(0);
    setError(null);
    setShowRegionSelection(false);

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
      // Analyze the selected region
      const result = await regionService.analyzeRegion({
        region_bbox: region.bbox,
        region_name: region.name,
        city: cityInput.trim()
      });

      clearInterval(progressInterval);
      setProgress(100);

      setTimeout(() => {
        setAnalysisResult(result);
        setIsAnalyzing(false);
        setProgress(0);

        // Scroll to results
        setTimeout(() => {
          const resultsSection = document.getElementById('results-section');
          if (resultsSection) {
            resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
          }
        }, 100);
      }, 1000);

    } catch (error) {
      clearInterval(progressInterval);
      setProgress(0);
      setIsAnalyzing(false);
      setError(error instanceof Error ? error.message : 'Regional analysis failed');
    }
  };

  const handleBackToCity = () => {
    setShowRegionSelection(false);
    setCityRegions(null);
    setSelectedRegion(null);
    setError(null);
  };

  const predefinedCities = urbanAGBService.getPredefinedCities();

  return (
    <div className="min-h-screen bg-green text-off-white font-graphik">
      {/* Hero Section with Earth Animation Background */}
      <section className="relative h-screen overflow-hidden">
        {/* Video Background */}
        <div className="absolute inset-0 z-0">
          <video
            autoPlay
            loop
            muted
            playsInline
            preload="auto"
            onLoadedData={(e) => (e.currentTarget.style.opacity = '1')}
            className="w-full h-full object-cover opacity-0 transition-opacity duration-1000 ease-in-out"
            poster="/earth-poster.svg"
          >
            {/* Use MP4 as primary source */}
            <source src="/earth animation.mp4" type="video/mp4" />
            <source src="/earth animation.webm" type="video/webm" />
            Your browser does not support the video tag.
          </video>

          {/* Fallback gradient background for when video is loading */}
          <div className="absolute inset-0 bg-gradient-to-br from-green via-green-800 to-green-900 -z-10"></div>

          {/* Dark overlay for contrast and readability */}
          <div className="absolute inset-0 bg-gradient-to-b from-green/80 via-green/60 to-green/90"></div>
        </div>

        {/* Navbar */}
        <div className="relative z-50">
          <Navbar />
        </div>

        {/* Hero Content */}
        <div className="relative z-40 h-full flex flex-col justify-center items-center px-6 pt-24 pb-24">
          <div
            className="text-center max-w-6xl mx-auto mt-8"
            style={{
              transform: `translateY(${scrollY * 0.3}px)`,
              opacity: 1 - scrollY / 500
            }}
          >
            {/* Main Heading */}
            <h1 className="text-4xl sm:text-5xl md:text-6xl lg:text-7xl xl:text-8xl font-black leading-tight mb-8 tracking-tight font-deacon text-center">
              <div className="text-off-white drop-shadow-2xl mb-2">
                ABOVE GROUND
              </div>
              <div className="text-neon-100 drop-shadow-2xl animate-pulse">
                BIOMASS PREDICTION
              </div>
            </h1>

            {/* Subtitle */}
            <p className="text-lg sm:text-xl md:text-2xl text-off-white/90 mb-3 leading-relaxed max-w-4xl mx-auto drop-shadow-lg px-4 text-center">
              satellite-powered biomass analysis for any location on Earth.
            </p>
            <p className="text-base sm:text-lg md:text-xl text-neon-100 font-semibold mb-14 drop-shadow-lg text-center">
              Real-time • Accurate • Actionable
            </p>

            {/* CTA Button */}
            <div className="mb-0">
              <MagnetizeButton
                variant="magnetize"
                size="xl"
                particleCount={40}
                className="px-16 py-6 text-xl md:text-2xl font-bold shadow-2xl"
                onClick={() => {
                  const inputSection = document.getElementById('input-section');
                  if (inputSection) {
                    inputSection.scrollIntoView({ behavior: 'smooth' });
                  }
                }}
              >
                START ANALYSIS
              </MagnetizeButton>
            </div>
          </div>

          {/* Scroll Indicator */}
          <div className="absolute bottom-12 right-12 animate-bounce">
            <div className="flex flex-col items-center space-y-3">
              <span className="text-off-white/70 text-xs md:text-sm font-semibold tracking-widest uppercase">
                Scroll to Explore
              </span>
              <svg
                className="w-7 h-7 text-neon-100"
                fill="none"
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2.5"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path d="M19 14l-7 7m0 0l-7-7m7 7V3"></path>
              </svg>
            </div>
          </div>
        </div>
      </section>


      {/* Main Content */}
      <div className="relative z-10 bg-green">
        {/* Input Section */}
        <section id="input-section" className="min-h-screen flex items-center justify-center py-24 lg:py-32 px-6">
          <div className="max-w-5xl mx-auto w-full">
            {/* Section Header */}
            <div className="text-center mb-20">
              <h2 className="text-4xl sm:text-5xl md:text-6xl lg:text-7xl font-black font-deacon text-off-white leading-tight mb-8">
                ANALYZE YOUR
                <span className="text-neon-100 block mt-2">LOCATION</span>
              </h2>
              <p className="text-lg sm:text-xl md:text-2xl text-off-white/80 max-w-3xl mx-auto leading-relaxed px-4">
                Enter any city in India to unlock comprehensive biomass insights
              </p>
            </div>

            {/* System Status Badge */}
            {systemStatus && (
              <div className="max-w-3xl mx-auto mb-10">
                <div className="bg-off-white/5 backdrop-blur-md rounded-2xl p-5 border border-off-white/10">
                  <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
                    <div className="flex items-center space-x-4">
                      <div className={`w-3 h-3 rounded-full ${systemStatus.status === 'operational' ? 'bg-neon-100' : 'bg-red-500'
                        } animate-pulse`}></div>
                      <span className="text-off-white font-semibold text-base">
                        System Operational
                      </span>
                      {systemStatus.version && (
                        <span className="text-off-white/50 text-sm bg-off-white/5 px-2 py-1 rounded">v{systemStatus.version}</span>
                      )}
                    </div>
                    <div className="text-off-white/70 text-sm font-medium">
                      🌍 All cities supported
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Main Input Card */}
            <div className="max-w-4xl mx-auto">
              <div className="bg-gradient-to-br from-off-white/15 to-off-white/5 backdrop-blur-xl rounded-3xl p-8 sm:p-10 lg:p-14 border border-off-white/20 shadow-2xl">
                <div className="space-y-10">
                  {/* Text Input */}
                  <div>
                    <label htmlFor="city" className="block text-lg font-bold text-off-white mb-4">
                      🌆 City Name
                    </label>
                    <input
                      id="city"
                      type="text"
                      value={cityInput}
                      onChange={(e) => setCityInput(e.target.value)}
                      onKeyPress={(e) => e.key === 'Enter' && handleAnalysis()}
                      placeholder="Type any city name..."
                      className="w-full px-6 py-5 bg-off-white/10 border-2 border-off-white/20 focus:border-neon-100 rounded-2xl focus:outline-none focus:ring-4 focus:ring-neon-100/20 text-off-white placeholder-off-white/40 text-lg transition-all"
                      disabled={isAnalyzing}
                    />
                  </div>

                  {/* Quick Select Pills - All Indian States & UTs */}
                  <div>
                    <label className="block text-lg font-semibold text-off-white mb-4">
                      ⚡ Quick Select: All Indian States & Union Territories
                    </label>
                    <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-3 max-h-96 overflow-y-auto p-4 bg-off-white/5 rounded-2xl border border-off-white/10">
                      {predefinedCities.map((city) => (
                        <button
                          key={city.id}
                          type="button"
                          onClick={() => handleQuickSelect(city.name)}
                          className="group relative px-4 py-3 bg-off-white/10 hover:bg-neon-100/20 border border-off-white/20 hover:border-neon-100 rounded-xl text-off-white text-sm font-medium transition-all transform hover:scale-105 hover:shadow-lg disabled:opacity-50 disabled:cursor-not-allowed"
                          disabled={isAnalyzing}
                          title={city.country}
                        >
                          <span className="relative z-10 block text-center">{city.name}</span>
                          <span className="relative z-10 block text-xs text-off-white/50 mt-1 text-center">{city.country}</span>
                          <div className="absolute inset-0 bg-gradient-to-r from-neon-100/0 via-neon-100/10 to-neon-100/0 rounded-xl opacity-0 group-hover:opacity-100 transition-opacity"></div>
                        </button>
                      ))}
                    </div>
                    <p className="text-xs text-off-white/50 mt-3 text-center">
                      📍 {predefinedCities.length} locations available • All 28 States + 8 UTs covered
                    </p>
                  </div>

                  {/* Action Button */}
                  <MagnetizeButton
                    variant="magnetize"
                    size="xl"
                    particleCount={35}
                    onClick={handleAnalysis}
                    disabled={isAnalyzing || !cityInput.trim()}
                    className="w-full py-6 text-xl font-bold shadow-xl"
                  >
                    {isAnalyzing ? (
                      <span className="flex items-center justify-center space-x-3">
                        <svg className="animate-spin h-6 w-6" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                        <span>ANALYZING...</span>
                      </span>
                    ) : (
                      <span className="flex items-center justify-center space-x-2">
                        <span>🛰️ EXPLORE REGIONS</span>
                      </span>
                    )}
                  </MagnetizeButton>

                  {/* Error Message */}
                  {error && (
                    <div className="p-5 bg-red-500/20 border-2 border-red-500 rounded-2xl text-red-200 text-center animate-shake">
                      <span className="font-semibold">⚠️ {error}</span>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Progress Bar */}
            {isAnalyzing && (
              <div className="max-w-4xl mx-auto mt-12 animate-fadeIn">
                <div className="bg-off-white/10 backdrop-blur-md rounded-3xl p-10 border border-off-white/20 shadow-xl">
                  <div className="flex items-center justify-between mb-6">
                    <span className="text-off-white font-bold text-lg">🔄 Processing Satellite Data</span>
                    <span className="text-neon-100 font-black text-3xl">{Math.round(progress)}%</span>
                  </div>
                  <div className="w-full bg-off-white/20 rounded-full h-5 overflow-hidden shadow-inner">
                    <div
                      className="h-full bg-gradient-to-r from-neon-100 via-green-400 to-neon-100 rounded-full transition-all duration-300 shadow-lg animate-pulse"
                      style={{ width: `${progress}%` }}
                    ></div>
                  </div>
                  <div className="mt-6 text-off-white/70 text-base text-center font-medium">
                    {progress < 30 && "📍 Geocoding city location..."}
                    {progress >= 30 && progress < 60 && "🛰️ Fetching satellite imagery from Google Earth Engine..."}
                    {progress >= 60 && progress < 90 && "🧮 Calculating biomass using empirical models..."}
                    {progress >= 90 && "🎨 Generating high-quality heatmap..."}
                  </div>
                </div>
              </div>
            )}
          </div>
        </section>

        {/* Region Selection Section */}
        {showRegionSelection && cityRegions && (
          <section id="region-selection-section" className="py-24 lg:py-32 px-6">
            <RegionSelector
              cityName={cityInput}
              regions={cityRegions.regions.map(region => ({
                ...region,
                previewImage: region.preview_image_url ? regionService.getPreviewImageUrl(region.preview_image_url) : undefined
              }))}
              onRegionSelect={handleRegionSelect}
              onBack={handleBackToCity}
              isLoading={isAnalyzing}
            />
          </section>
        )}

        {/* Loading Regions Section */}
        {isLoadingRegions && (
          <section className="py-24 lg:py-32 px-6">
            <div className="max-w-4xl mx-auto text-center">
              <div className="bg-gradient-to-br from-off-white/15 to-off-white/5 backdrop-blur-xl rounded-3xl p-12 border border-off-white/20 shadow-2xl">
                <div className="mb-8">
                  <div className="inline-block bg-neon-100/20 backdrop-blur-md border border-neon-100/30 rounded-full px-8 py-3 mb-6">
                    <span className="text-neon-100 font-bold text-sm md:text-base uppercase tracking-wider">
                      🛰️ Loading Regions
                    </span>
                  </div>
                  <h2 className="text-4xl sm:text-5xl md:text-6xl font-black font-deacon text-off-white mb-6">
                    Preparing {cityInput}
                    <span className="text-neon-100 block mt-2">SATELLITE DATA</span>
                  </h2>
                </div>

                <div className="space-y-6">
                  <div className="flex items-center justify-center space-x-4">
                    <svg className="animate-spin h-8 w-8 text-neon-100" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    <span className="text-off-white font-bold text-xl">Fetching satellite imagery...</span>
                  </div>

                  <div className="text-off-white/70 text-base max-w-2xl mx-auto leading-relaxed">
                    We're downloading real satellite images from Google Earth Engine for North, South, East, West, and Center regions of {cityInput}.
                    This may take a moment as we process high-resolution imagery.
                  </div>

                  <div className="grid grid-cols-5 gap-4 mt-8">
                    {['North', 'South', 'East', 'West', 'Center'].map((region, index) => (
                      <div key={region} className="bg-off-white/5 rounded-xl p-4 text-center">
                        <div className="w-12 h-12 bg-gradient-to-br from-neon-100/20 to-neon-100/5 rounded-lg mx-auto mb-2 flex items-center justify-center">
                          <span className="text-neon-100 text-lg">
                            {region === 'North' && '⬆️'}
                            {region === 'South' && '⬇️'}
                            {region === 'East' && '➡️'}
                            {region === 'West' && '⬅️'}
                            {region === 'Center' && '🏙️'}
                          </span>
                        </div>
                        <span className="text-off-white/70 text-sm font-medium">{region}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </section>
        )}

        {/* Results Section */}
        {analysisResult && (
          <section id="results-section" className="py-24 lg:py-32 px-6">
            <div className="max-w-7xl mx-auto">
              {/* Results Header */}
              <div className="text-center mb-20 animate-fadeIn">
                <div className="inline-block bg-neon-100/20 backdrop-blur-md border border-neon-100/30 rounded-full px-8 py-3 mb-8">
                  <span className="text-neon-100 font-bold text-sm md:text-base uppercase tracking-wider">✅ Analysis Complete</span>
                </div>
                <h2 className="text-4xl sm:text-5xl md:text-6xl lg:text-7xl font-black font-deacon text-off-white mb-6">
                  {selectedRegion ? selectedRegion.name : analysisResult.city}
                </h2>
                <p className="text-off-white/70 text-base md:text-lg font-medium mb-4">
                  📍 {analysisResult.location.coordinates}
                </p>
                {selectedRegion && (
                  <div className="max-w-4xl mx-auto">
                    <p className="text-off-white/80 text-base md:text-lg leading-relaxed bg-off-white/5 backdrop-blur-sm rounded-2xl p-6 border border-off-white/10">
                      {selectedRegion.description}
                    </p>
                  </div>
                )}
              </div>

              {/* Key Metrics Grid */}
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 lg:gap-8 mb-16 animate-slideUp">
                <div className="group bg-gradient-to-br from-neon-100/20 to-neon-100/5 backdrop-blur-sm rounded-2xl p-8 border border-neon-100/30 text-center hover:scale-105 transition-transform shadow-xl">
                  <div className="text-5xl font-black text-neon-100 mb-3 group-hover:scale-110 transition-transform">
                    {analysisResult.current_agb.total_agb.toFixed(1)}
                  </div>
                  <div className="text-off-white/70 text-sm font-medium mb-1">Mg/ha</div>
                  <div className="text-neon-100 text-xs font-bold uppercase tracking-wide">Total Biomass</div>
                </div>

                <div className="group bg-gradient-to-br from-blue-500/20 to-blue-500/5 backdrop-blur-sm rounded-2xl p-8 border border-blue-500/30 text-center hover:scale-105 transition-transform shadow-xl">
                  <div className="text-5xl font-black text-blue-400 mb-3 group-hover:scale-110 transition-transform">
                    {analysisResult.current_agb.canopy_cover.toFixed(1)}%
                  </div>
                  <div className="text-off-white/70 text-sm font-medium mb-1">Coverage</div>
                  <div className="text-blue-400 text-xs font-bold uppercase tracking-wide">Canopy Cover</div>
                </div>

                <div className="group bg-gradient-to-br from-cyan-500/20 to-cyan-500/5 backdrop-blur-sm rounded-2xl p-8 border border-cyan-500/30 text-center hover:scale-105 transition-transform shadow-xl">
                  <div className="text-5xl font-black text-cyan-400 mb-3 group-hover:scale-110 transition-transform">
                    {analysisResult.current_agb.cooling_potential.toFixed(1)}°C
                  </div>
                  <div className="text-off-white/70 text-sm font-medium mb-1">Reduction</div>
                  <div className="text-cyan-400 text-xs font-bold uppercase tracking-wide">Cooling Potential</div>
                </div>

                <div className="group bg-gradient-to-br from-emerald-500/20 to-emerald-500/5 backdrop-blur-sm rounded-2xl p-8 border border-emerald-500/30 text-center hover:scale-105 transition-transform shadow-xl">
                  <div className="text-5xl font-black text-emerald-400 mb-3 group-hover:scale-110 transition-transform">
                    {analysisResult.current_agb.carbon_sequestration.toFixed(0)}
                  </div>
                  <div className="text-off-white/70 text-sm font-medium mb-1">Tons CO₂/yr</div>
                  <div className="text-emerald-400 text-xs font-bold uppercase tracking-wide">Carbon Storage</div>
                </div>
              </div>

              {/* Above Ground Biomass Analysis */}
              <div className="mb-16 animate-fadeIn">
                <div className="bg-gradient-to-br from-off-white/15 to-off-white/5 backdrop-blur-xl rounded-3xl p-8 lg:p-12 border border-off-white/20 shadow-2xl">
                  <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between mb-8 gap-4">
                    <div className="flex-1">
                      <h3 className="text-2xl md:text-3xl lg:text-4xl font-black text-off-white font-deacon leading-tight mb-2">
                        🗺️ ABOVE GROUND BIOMASS HEATMAP
                      </h3>
                      <p className="text-neon-100 text-sm md:text-base font-semibold uppercase tracking-wider">
                        Generated using Hybrid CNN+LSTM Model
                      </p>
                    </div>
                    <div className="flex flex-col items-start lg:items-end gap-2">
                      <span className="text-neon-100 text-sm font-semibold bg-neon-100/20 px-3 py-1 rounded-full">
                        HIGH RESOLUTION
                      </span>
                      <span className="text-off-white/70 text-xs font-medium">
                        Deep Learning Analysis
                      </span>
                    </div>
                  </div>
                  <div className="bg-off-white/5 rounded-2xl overflow-hidden border-2 border-off-white/10 shadow-inner">
                    <img
                      src={analysisResult.heat_map.image_url || '/placeholder-heatmap.png'}
                      alt={`Above Ground Biomass analysis of ${selectedRegion ? selectedRegion.name : analysisResult.city}`}
                      className="w-full h-auto"
                      onError={(e) => {
                        const target = e.target as HTMLImageElement;
                        target.src = '/placeholder-heatmap.png';
                      }}
                    />
                  </div>
                  <div className="mt-8 space-y-4">
                    <p className="text-off-white/80 text-sm md:text-base text-center leading-relaxed">
                      Advanced biomass distribution analysis using hybrid CNN+LSTM deep learning model, showing vegetation density,
                      urban forests, and green spaces across <span className="text-neon-100 font-semibold">
                        {selectedRegion ? selectedRegion.name : analysisResult.city}
                      </span>
                    </p>
                    <div className="flex flex-wrap justify-center gap-4 text-xs text-off-white/60">
                      <span className="bg-off-white/5 px-3 py-1 rounded-full">🛰️ Sentinel-2 Imagery</span>
                      <span className="bg-off-white/5 px-3 py-1 rounded-full">🧠 Deep Learning</span>
                      <span className="bg-off-white/5 px-3 py-1 rounded-full">📊 10m Resolution</span>
                      <span className="bg-off-white/5 px-3 py-1 rounded-full">⚡ Real-time Analysis</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Data Grid */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 lg:gap-10 mb-16">
                {/* Forecasting */}
                <div className="bg-gradient-to-br from-off-white/15 to-off-white/5 backdrop-blur-xl rounded-3xl p-8 border border-off-white/20 shadow-xl animate-slideRight">
                  <h3 className="text-2xl font-black text-off-white mb-6 flex items-center">
                    <span className="mr-3">📈</span>
                    3-YEAR FORECAST
                  </h3>
                  <div className="space-y-5">
                    <div className="flex justify-between items-center p-4 bg-off-white/5 rounded-xl">
                      <span className="text-off-white/70 font-medium">Year 1 (2026)</span>
                      <span className="text-neon-100 font-black text-xl">
                        {analysisResult.forecasting.year_1.toFixed(1)} Mg/ha
                      </span>
                    </div>
                    <div className="flex justify-between items-center p-4 bg-off-white/5 rounded-xl">
                      <span className="text-off-white/70 font-medium">Year 3 (2028)</span>
                      <span className="text-neon-100 font-black text-xl">
                        {analysisResult.forecasting.year_3.toFixed(1)} Mg/ha
                      </span>
                    </div>
                    <div className="flex justify-between items-center p-4 bg-off-white/5 rounded-xl">
                      <span className="text-off-white/70 font-medium">Year 5 (2030)</span>
                      <span className="text-neon-100 font-black text-xl">
                        {analysisResult.forecasting.year_5.toFixed(1)} Mg/ha
                      </span>
                    </div>
                    <div className="pt-5 border-t border-off-white/20">
                      <div className="flex justify-between items-center">
                        <span className="text-off-white font-semibold">📊 Growth Rate</span>
                        <span className="text-neon-100 font-black text-2xl">
                          {(analysisResult.forecasting.growth_rate * 100).toFixed(1)}%
                        </span>
                      </div>
                      <p className="text-off-white/50 text-xs mt-2">Annual biomass increase</p>
                    </div>
                  </div>
                </div>

                {/* Urban Metrics */}
                <div className="bg-gradient-to-br from-off-white/15 to-off-white/5 backdrop-blur-xl rounded-3xl p-8 border border-off-white/20 shadow-xl animate-slideLeft">
                  <h3 className="text-2xl font-black text-off-white mb-6 flex items-center">
                    <span className="mr-3">🏙️</span>
                    URBAN METRICS
                  </h3>
                  <div className="space-y-5">
                    <div className="flex justify-between items-center p-4 bg-off-white/5 rounded-xl">
                      <span className="text-off-white/70 font-medium">EPI Score</span>
                      <span className="text-neon-100 font-black text-xl">
                        {analysisResult.urban_metrics.epi_score.toFixed(1)}/100
                      </span>
                    </div>
                    <div className="flex justify-between items-center p-4 bg-off-white/5 rounded-xl">
                      <span className="text-off-white/70 font-medium">Green Space Ratio</span>
                      <span className="text-neon-100 font-black text-xl">
                        {(analysisResult.urban_metrics.green_space_ratio * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="flex justify-between items-center p-4 bg-off-white/5 rounded-xl">
                      <span className="text-off-white/70 font-medium">Tree Cities Score</span>
                      <span className="text-neon-100 font-black text-xl">
                        {analysisResult.urban_metrics.tree_cities_score}/5
                      </span>
                    </div>
                    <div className="pt-5 border-t border-off-white/20">
                      <div className="flex justify-between items-center">
                        <span className="text-off-white font-semibold">⚡ Energy Savings</span>
                        <span className="text-neon-100 font-black text-xl">
                          0 kWh
                        </span>
                      </div>
                      <p className="text-off-white/50 text-xs mt-2">Annual energy reduction</p>
                    </div>
                  </div>
                </div>
              </div>

              {/* Satellite Data */}
              <div className="bg-gradient-to-br from-off-white/15 to-off-white/5 backdrop-blur-xl rounded-3xl p-8 lg:p-10 border border-off-white/20 shadow-xl mb-16 animate-fadeIn">
                <h3 className="text-2xl font-black text-off-white mb-6 flex items-center">
                  <span className="mr-3">🛰️</span>
                  SATELLITE DATA
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <div className="bg-gradient-to-br from-neon-100/10 to-transparent rounded-2xl p-6 text-center border border-neon-100/20">
                    <div className="text-4xl font-black text-neon-100 mb-2">
                      {analysisResult.satellite_data.ndvi.toFixed(3)}
                    </div>
                    <div className="text-off-white/70 text-sm font-bold mb-1">NDVI</div>
                    <div className="text-off-white/50 text-xs">Normalized Difference Vegetation Index</div>
                  </div>
                  <div className="bg-gradient-to-br from-blue-500/10 to-transparent rounded-2xl p-6 text-center border border-blue-500/20">
                    <div className="text-4xl font-black text-blue-400 mb-2">
                      {analysisResult.satellite_data.evi.toFixed(3)}
                    </div>
                    <div className="text-off-white/70 text-sm font-bold mb-1">EVI</div>
                    <div className="text-off-white/50 text-xs">Enhanced Vegetation Index</div>
                  </div>
                  <div className="bg-gradient-to-br from-emerald-500/10 to-transparent rounded-2xl p-6 text-center border border-emerald-500/20">
                    <div className="text-4xl font-black text-emerald-400 mb-2">
                      {analysisResult.satellite_data.lai.toFixed(2)}
                    </div>
                    <div className="text-off-white/70 text-sm font-bold mb-1">LAI</div>
                    <div className="text-off-white/50 text-xs">Leaf Area Index</div>
                  </div>
                </div>
                <p className="text-off-white/50 text-xs mt-6 text-center">
                  📡 Data Source: <span className="text-neon-100 font-semibold">{analysisResult.satellite_data.data_source}</span>
                </p>
              </div>

              {/* Recommendations */}
              <div className="bg-gradient-to-br from-off-white/15 to-off-white/5 backdrop-blur-xl rounded-3xl p-10 lg:p-12 border border-off-white/20 shadow-xl mb-16 animate-slideUp">
                <h3 className="text-2xl md:text-3xl lg:text-4xl font-black text-off-white mb-10 flex items-center">
                  <span className="mr-4">💡</span>
                  PLANNING RECOMMENDATIONS
                </h3>
                <div className="grid gap-6">
                  {analysisResult.planning_recommendations.map((rec: string, index: number) => (
                    <div key={index} className="group flex items-start space-x-5 p-5 bg-off-white/5 hover:bg-off-white/10 rounded-2xl transition-all border border-off-white/10 hover:border-neon-100/30">
                      <div className="flex-shrink-0 w-12 h-12 bg-gradient-to-br from-neon-100 to-green-400 rounded-xl flex items-center justify-center text-green font-black text-xl shadow-lg group-hover:scale-110 transition-transform">
                        {index + 1}
                      </div>
                      <div className="flex-1 pt-2">
                        <p className="text-off-white leading-relaxed">{rec}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Action Buttons */}
              <div className="flex flex-col sm:flex-row gap-6 justify-center items-center animate-fadeIn pt-8">
                <MagnetizeButton
                  variant="magnetize"
                  size="lg"
                  particleCount={25}
                  className="px-12 py-5 text-lg font-bold shadow-xl"
                  onClick={() => {
                    setAnalysisResult(null);
                    setCityInput('');
                    window.scrollTo({ top: 0, behavior: 'smooth' });
                  }}
                >
                  🔄 NEW ANALYSIS
                </MagnetizeButton>
                <MagnetizeButton
                  variant="outline"
                  size="lg"
                  particleCount={20}
                  className="border-2 border-neon-100 text-neon-100 hover:bg-neon-100 hover:text-green px-12 py-5 text-lg font-bold shadow-xl"
                  onClick={() => window.location.href = '/model'}
                >
                  🔄 ANALYZE ANOTHER REGION
                </MagnetizeButton>
              </div>
            </div>
          </section>
        )}
      </div>

      <Footer />
    </div>
  );
};

export default MLModelPage;
