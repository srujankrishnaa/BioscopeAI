import React, { useState, useEffect } from 'react';
import Navbar from '../components/Navbar';
import Footer from '../components/Footer';
import { MagnetizeButton } from '../components/ui/magnetize-button';
import RegionSelector from '../components/RegionSelector';
import urbanAGBService, { UrbanAGBResponse, SystemStatus } from '../services/urbanAgbService';
import regionService, { RegionData } from '../services/regionService';

type AppStep = 'city' | 'region' | 'analysis' | 'results';

const MLModelPageWithRegions: React.FC = () => {
  // Existing state
  const [cityInput, setCityInput] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<UrbanAGBResponse | null>(null);
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  // New state for region selection
  const [currentStep, setCurrentStep] = useState<AppStep>('city');
  const [availableRegions, setAvailableRegions] = useState<RegionData[]>([]);
  const [selectedRegion, setSelectedRegion] = useState<RegionData | null>(null);
  const [isLoadingRegions, setIsLoadingRegions] = useState(false);

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
    const inputSection = document.getElementById('input-section');
    if (inputSection) {
      inputSection.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  };

  const handleCitySubmit = async () => {
    if (!cityInput.trim()) {
      setError('Please enter a city name');
      return;
    }
    
    setIsLoadingRegions(true);
    setError(null);

    try {
      // Fetch available regions for the city
      const regionsResponse = await regionService.getCityRegions(cityInput.trim());
      
      // Map the regions to include previewImage property
      const mappedRegions = regionsResponse.regions.map(region => ({
        ...region,
        previewImage: region.preview_image_url ? regionService.getPreviewImageUrl(region.preview_image_url) : undefined
      }));
      
      setAvailableRegions(mappedRegions);
      setCurrentStep('region');
      
      // Scroll to region selection
      setTimeout(() => {
        const regionSection = document.getElementById('region-section');
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
    setCurrentStep('analysis');
    setIsAnalyzing(true);
    setProgress(0);
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
      // For now, use existing prediction API with region-specific data
      // In future, this would call regionService.analyzeRegion()
      const result = await urbanAGBService.predictUrbanAGB({
        city: `${cityInput.trim()} - ${region.name}`
      });
      
      clearInterval(progressInterval);
      setProgress(100);

      setTimeout(() => {
        setAnalysisResult(result);
        setIsAnalyzing(false);
        setCurrentStep('results');
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
      setCurrentStep('region');
    }
  };

  const handleBackToCity = () => {
    setCurrentStep('city');
    setAvailableRegions([]);
    setSelectedRegion(null);
    setError(null);
  };

  const handleBackToRegions = () => {
    setCurrentStep('region');
    setSelectedRegion(null);
    setAnalysisResult(null);
    setError(null);
  };

  const handleNewAnalysis = () => {
    setCurrentStep('city');
    setAvailableRegions([]);
    setSelectedRegion(null);
    setAnalysisResult(null);
    setCityInput('');
    setError(null);
    window.scrollTo({ top: 0, behavior: 'smooth' });
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
            poster={`${process.env.PUBLIC_URL}/earth-poster.svg`}
          >
            <source src="/earth animation.mp4" type="video/mp4" />
            <source src="/earth animation.webm" type="video/webm" />
            Your browser does not support the video tag.
          </video>

          <div className="absolute inset-0 bg-gradient-to-br from-green via-green-800 to-green-900 -z-10"></div>
          <div className="absolute inset-0 bg-gradient-to-b from-green/80 via-green/60 to-green/90"></div>
        </div>

        {/* Navbar */}
        <div className="relative z-50">
          <Navbar />
        </div>

        {/* Hero Content */}
        <div className="relative z-40 h-full flex flex-col justify-center items-center px-6 pt-24 pb-24">
          <div className="text-center max-w-6xl mx-auto mt-8 transform transition-transform">
            <h1 className="text-4xl sm:text-5xl md:text-6xl lg:text-7xl xl:text-8xl font-black leading-tight mb-8 tracking-tight font-deacon text-center">
              <div className="text-off-white drop-shadow-2xl mb-2">
                ABOVE GROUND
              </div>
              <div className="text-neon-100 drop-shadow-2xl animate-pulse">
                BIOMASS PREDICTION
              </div>
            </h1>

            <p className="text-lg sm:text-xl md:text-2xl text-off-white/90 mb-3 leading-relaxed max-w-4xl mx-auto drop-shadow-lg px-4 text-center">
              satellite-powered biomass analysis for any location on Earth.
            </p>
            <p className="text-base sm:text-lg md:text-xl text-neon-100 font-semibold mb-14 drop-shadow-lg text-center">
              Real-time • Regional • Actionable
            </p>

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
        {/* Step 1: City Input Section */}
        {currentStep === 'city' && (
          <section id="input-section" className="min-h-screen flex items-center justify-center py-24 lg:py-32 px-6">
            <div className="max-w-5xl mx-auto w-full">
              <div className="text-center mb-20">
                <h2 className="text-4xl sm:text-5xl md:text-6xl lg:text-7xl font-black font-deacon text-off-white leading-tight mb-8">
                  ANALYZE YOUR
                  <span className="text-neon-100 block mt-2">LOCATION</span>
                </h2>
                <p className="text-lg sm:text-xl md:text-2xl text-off-white/80 max-w-3xl mx-auto leading-relaxed px-4">
                  Enter any city to explore regional biomass insights
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
                        🌍 Regional Analysis Available
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Main Input Card */}
              <div className="max-w-4xl mx-auto">
                <div className="bg-gradient-to-br from-off-white/15 to-off-white/5 backdrop-blur-xl rounded-3xl p-8 sm:p-10 lg:p-14 border border-off-white/20 shadow-2xl">
                  <div className="space-y-10">
                    <div>
                      <label htmlFor="city" className="block text-lg font-bold text-off-white mb-4">
                        🌆 City Name
                      </label>
                      <input
                        id="city"
                        type="text"
                        value={cityInput}
                        onChange={(e) => setCityInput(e.target.value)}
                        onKeyDown={(e) => e.key === 'Enter' && handleCitySubmit()}
                        placeholder="Type any city name..."
                        className="w-full px-6 py-5 bg-off-white/10 border-2 border-off-white/20 focus:border-neon-100 rounded-2xl focus:outline-none focus:ring-4 focus:ring-neon-100/20 text-off-white placeholder-off-white/40 text-lg transition-all"
                        disabled={isLoadingRegions}
                      />
                    </div>
                    
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
                            disabled={isLoadingRegions}
                            title={city.country}
                          >
                            <span className="relative z-10 block text-center">{city.name}</span>
                            <span className="relative z-10 block text-xs text-off-white/50 mt-1 text-center">{city.country}</span>
                          </button>
                        ))}
                      </div>
                    </div>

                    <MagnetizeButton
                      variant="magnetize"
                      size="xl"
                      particleCount={35}
                      onClick={handleCitySubmit}
                      disabled={isLoadingRegions || !cityInput.trim()}
                      className="w-full py-6 text-xl font-bold shadow-xl"
                    >
                      {isLoadingRegions ? (
                        <span className="flex items-center justify-center space-x-3">
                          <svg className="animate-spin h-6 w-6" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                          </svg>
                          <span>LOADING REGIONS...</span>
                        </span>
                      ) : (
                        <span>🗺️ EXPLORE REGIONS</span>
                      )}
                    </MagnetizeButton>

                    {error && (
                      <div className="p-5 bg-red-500/20 border-2 border-red-500 rounded-2xl text-red-200 text-center">
                        <span className="font-semibold">⚠️ {error}</span>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          </section>
        )}

        {/* Step 2: Region Selection Section */}
        {currentStep === 'region' && (
          <section id="region-section" className="min-h-screen flex items-center justify-center py-24 lg:py-32 px-6">
            <RegionSelector
              cityName={cityInput}
              regions={availableRegions}
              onRegionSelect={handleRegionSelect}
              onBack={handleBackToCity}
              isLoading={isAnalyzing}
            />
          </section>
        )}

        {/* Step 3: Analysis Progress */}
        {currentStep === 'analysis' && isAnalyzing && (
          <section className="min-h-screen flex items-center justify-center py-24 lg:py-32 px-6">
            <div className="max-w-4xl mx-auto w-full">
              <div className="bg-off-white/10 backdrop-blur-md rounded-3xl p-10 border border-off-white/20 shadow-xl">
                <div className="text-center mb-8">
                  <h3 className="text-3xl font-black text-off-white mb-4">
                    🛰️ Analyzing {selectedRegion?.name}
                  </h3>
                  <p className="text-off-white/70">
                    Processing satellite data for detailed biomass analysis...
                  </p>
                </div>
                
                <div className="flex items-center justify-between mb-6">
                  <span className="text-off-white font-bold text-lg">Processing Satellite Data</span>
                  <span className="text-neon-100 font-black text-3xl">{Math.round(progress)}%</span>
                </div>
                
                <div className="w-full bg-off-white/20 rounded-full h-5 overflow-hidden shadow-inner">
                  <div 
                    className="h-full bg-gradient-to-r from-neon-100 via-green-400 to-neon-100 rounded-full transition-all duration-300 shadow-lg animate-pulse"
                    style={{ width: `${progress}%` }}
                  ></div>
                </div>
                
                <div className="mt-6 text-off-white/70 text-base text-center font-medium">
                  {progress < 30 && "📍 Processing region coordinates..."}
                  {progress >= 30 && progress < 60 && "🛰️ Fetching high-resolution satellite imagery..."}
                  {progress >= 60 && progress < 90 && "🧮 Calculating regional biomass distribution..."}
                  {progress >= 90 && "🎨 Generating detailed heatmap..."}
                </div>
              </div>
            </div>
          </section>
        )}

        {/* Step 4: Results Section */}
        {currentStep === 'results' && analysisResult && (
          <section id="results-section" className="py-24 lg:py-32 px-6">
            <div className="max-w-7xl mx-auto">
              <div className="text-center mb-20">
                <div className="inline-block bg-neon-100/20 backdrop-blur-md border border-neon-100/30 rounded-full px-8 py-3 mb-8">
                  <span className="text-neon-100 font-bold text-sm md:text-base uppercase tracking-wider">✅ Regional Analysis Complete</span>
                </div>
                <h2 className="text-4xl sm:text-5xl md:text-6xl lg:text-7xl font-black font-deacon text-off-white mb-6">
                  {analysisResult.city}
                </h2>
                <p className="text-off-white/70 text-base md:text-lg font-medium">
                  📍 {analysisResult.location.coordinates}
                </p>
              </div>

              {/* Results content would go here - same as your existing results */}
              
              <div className="flex flex-col sm:flex-row gap-6 justify-center items-center pt-8">
                <MagnetizeButton
                  variant="outline"
                  size="lg"
                  particleCount={20}
                  className="border-2 border-off-white/30 text-off-white hover:bg-off-white/10 px-12 py-5 text-lg font-bold"
                  onClick={handleBackToRegions}
                >
                  ← Back to Regions
                </MagnetizeButton>
                
                <MagnetizeButton
                  variant="magnetize"
                  size="lg"
                  particleCount={25}
                  className="px-12 py-5 text-lg font-bold shadow-xl"
                  onClick={handleNewAnalysis}
                >
                  🔄 NEW ANALYSIS
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

export default MLModelPageWithRegions;