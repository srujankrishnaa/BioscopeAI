import React, { useState, useEffect } from 'react';
import Navbar from '../components/Navbar';
import Footer from '../components/Footer';
import { MagnetizeButton } from '../components/ui/magnetize-button';
import urbanAGBService, { UrbanAGBResponse, SystemStatus } from '../services/urbanAgbService';

const MLModelPage: React.FC = () => {
  const [cityInput, setCityInput] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<UrbanAGBResponse | null>(null);
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [scrollY, setScrollY] = useState(0);

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
      setError(error instanceof Error ? error.message : 'Analysis failed');
    }
  };

  const predefinedCities = urbanAGBService.getPredefinedCities();

  return (
    <div className="min-h-screen bg-green text-off-white font-graphik">
      {/* Hero Section with Animated Background */}
      <section className="relative h-screen overflow-hidden">
        {/* Animated Background */}
        <div className="absolute inset-0 z-0">
          {/* Fallback gradient background */}
          <div className="absolute inset-0 bg-gradient-to-br from-green via-green-800 to-green-900"></div>
          
          {/* Animated particles/stars effect */}
          <div className="absolute inset-0 opacity-30">
            <div className="absolute top-1/4 left-1/4 w-2 h-2 bg-neon-100 rounded-full animate-pulse"></div>
            <div className="absolute top-1/3 right-1/3 w-1 h-1 bg-white rounded-full animate-ping"></div>
            <div className="absolute bottom-1/4 left-1/3 w-1.5 h-1.5 bg-neon-100 rounded-full animate-pulse delay-1000"></div>
            <div className="absolute top-1/2 right-1/4 w-1 h-1 bg-white rounded-full animate-ping delay-500"></div>
            <div className="absolute bottom-1/3 right-1/2 w-2 h-2 bg-neon-100 rounded-full animate-pulse delay-2000"></div>
            <div className="absolute top-3/4 left-1/2 w-1 h-1 bg-white rounded-full animate-ping delay-1500"></div>
            <div className="absolute top-1/6 right-1/6 w-1.5 h-1.5 bg-neon-100 rounded-full animate-pulse delay-3000"></div>
          </div>

          {/* Moving gradient overlay */}
          <div className="absolute inset-0 bg-gradient-to-r from-transparent via-neon-100/5 to-transparent animate-pulse"></div>
          
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
                      <div className={`w-3 h-3 rounded-full ${
                        systemStatus.status === 'ready' ? 'bg-neon-100' : 'bg-red-500'
                      } animate-pulse`}></div>
                      <span className="text-off-white font-semibold text-base">
                        System Operational
                      </span>
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
                  
                  {/* Quick Select Pills */}
                  <div>
                    <label className="block text-lg font-semibold text-off-white mb-4">
                      ⚡ Quick Select: Popular Cities
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
                      📍 {predefinedCities.length} locations available
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
                        <span>🛰️ PREDICT BIOMASS</span>
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
                  {analysisResult.city}
                </h2>
                <p className="text-off-white/70 text-base md:text-lg font-medium">
                  📍 {analysisResult.location.coordinates}
                </p>
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

              {/* Heatmap Showcase */}
              <div className="mb-16 animate-fadeIn">
                <div className="bg-gradient-to-br from-off-white/15 to-off-white/5 backdrop-blur-xl rounded-3xl p-8 lg:p-10 border border-off-white/20 shadow-2xl">
                  <div className="flex items-center justify-between mb-6">
                    <h3 className="text-3xl font-black text-off-white font-deacon">🗺️ BIOMASS HEATMAP</h3>
                    <span className="text-neon-100 text-sm font-semibold">HIGH RESOLUTION</span>
                  </div>
                  <div className="bg-off-white/5 rounded-2xl overflow-hidden border-2 border-off-white/10 shadow-inner">
                    <img 
                      src={analysisResult.heat_map.image_url || '/placeholder-heatmap.png'} 
                      alt={`Biomass heatmap of ${analysisResult.city}`}
                      className="w-full h-auto"
                      onError={(e) => {
                        const target = e.target as HTMLImageElement;
                        target.src = '/placeholder-heatmap.png';
                      }}
                    />
                  </div>
                  <p className="text-off-white/60 text-sm mt-6 text-center leading-relaxed">
                    High-resolution biomass distribution showing vegetation density, urban forests, and green spaces across <span className="text-neon-100 font-semibold">{analysisResult.city}</span>
                  </p>
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
                  onClick={() => window.location.href = '/dashboard'}
                >
                  📊 VIEW DASHBOARD
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