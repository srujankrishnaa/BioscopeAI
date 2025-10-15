import React, { useState, useEffect } from 'react';
import { MagnetizeButton } from './ui/magnetize-button';

interface RegionData {
  id: string;
  name: string;
  description: string;
  bbox: number[];
  previewImage?: string;
  coordinates: {
    center: [number, number];
    bounds: [[number, number], [number, number]];
  };
}

interface RegionSelectorProps {
  cityName: string;
  regions: RegionData[];
  onRegionSelect: (region: RegionData) => void;
  onBack: () => void;
  isLoading?: boolean;
}

const RegionSelector: React.FC<RegionSelectorProps> = ({
  cityName,
  regions,
  onRegionSelect,
  onBack,
  isLoading = false
}) => {
  const [selectedRegion, setSelectedRegion] = useState<RegionData | null>(null);
  const [previewsLoaded, setPreviewsLoaded] = useState<{ [key: string]: boolean }>({});
  const [showFakeLoading, setShowFakeLoading] = useState(true);
  const [hasStartedLoading, setHasStartedLoading] = useState(false);
  const [loadingProgress, setLoadingProgress] = useState(0);
  const [currentLoadingRegion, setCurrentLoadingRegion] = useState(0);

  // Reset fake loading when cityName changes (new city selected)
  useEffect(() => {
    setShowFakeLoading(true);
    setLoadingProgress(0);
    setCurrentLoadingRegion(0);
    setSelectedRegion(null);
    setPreviewsLoaded({});
  }, [cityName]);

  // Fake loading simulation - makes users think images are being downloaded
  useEffect(() => {
    if (!showFakeLoading) return;

    console.log('🚀 Starting fake loading for:', cityName);
    console.log('📊 showFakeLoading state:', showFakeLoading);

    // Variable timing for each region to make it feel more realistic
    // TESTING: Longer durations to make sure it's visible
    const regionDurations = [
      { name: 'Center', duration: 3000, progressEnd: 20 },   // 3 seconds -> 20%
      { name: 'North', duration: 4000, progressEnd: 40 },    // 4 seconds -> 40%
      { name: 'East', duration: 3500, progressEnd: 60 },     // 3.5 seconds -> 60%
      { name: 'West', duration: 3000, progressEnd: 80 },     // 3 seconds -> 80%
      { name: 'South', duration: 3500, progressEnd: 100 }    // 3.5 seconds -> 100%
    ];

    const updateInterval = 100; // Update every 100ms for smooth animation
    const startTime = Date.now();

    // Reset progress states
    setLoadingProgress(0);
    setCurrentLoadingRegion(0);

    console.log('⏰ Progress interval starting...');

    const progressInterval = setInterval(() => {
      const totalElapsed = Date.now() - startTime;

      // Calculate cumulative durations to find current region and progress
      let cumulativeDuration = 0;
      let currentRegionIndex = 0;
      let progress = 0;

      // Find which region we should be in based on total elapsed time
      for (let i = 0; i < regionDurations.length; i++) {
        if (totalElapsed <= cumulativeDuration + regionDurations[i].duration) {
          currentRegionIndex = i;
          break;
        }
        cumulativeDuration += regionDurations[i].duration;
        currentRegionIndex = i + 1; // Move to next region
      }

      // Update current region if it changed
      setCurrentLoadingRegion(Math.min(currentRegionIndex, regionDurations.length - 1));

      // Calculate progress based on total elapsed time
      if (currentRegionIndex >= regionDurations.length) {
        // All regions completed
        progress = 100;
      } else {
        // Calculate progress within current region
        const regionElapsed = totalElapsed - cumulativeDuration;
        const regionDuration = regionDurations[currentRegionIndex].duration;
        const regionProgressPercent = Math.min(regionElapsed / regionDuration, 1);

        if (currentRegionIndex === 0) {
          // First region: 0% to 20%
          progress = regionProgressPercent * 20;
        } else {
          // Subsequent regions: previous end + current progress
          const previousEnd = regionDurations[currentRegionIndex - 1].progressEnd;
          const currentRange = regionDurations[currentRegionIndex].progressEnd - previousEnd;
          progress = previousEnd + (regionProgressPercent * currentRange);
        }
      }

      // Ensure progress doesn't exceed 100%
      progress = Math.min(progress, 100);
      setLoadingProgress(progress);

      // Complete loading when we reach 100%
      if (progress >= 100) {
        clearInterval(progressInterval);

        // Small delay before showing regions for better UX
        setTimeout(() => {
          setShowFakeLoading(false);
          // Mark all images as loaded after fake loading completes
          const loadedState: { [key: string]: boolean } = {};
          regions.forEach(region => {
            loadedState[region.id] = true;
          });
          setPreviewsLoaded(loadedState);
        }, 500);
      }
    }, updateInterval);

    return () => clearInterval(progressInterval);
  }, [cityName, regions, showFakeLoading]);

  const handleImageLoad = (regionId: string) => {
    if (!showFakeLoading) {
      setPreviewsLoaded(prev => ({ ...prev, [regionId]: true }));
    }
  };

  const handleRegionClick = (region: RegionData) => {
    if (!showFakeLoading) {
      setSelectedRegion(region);
    }
  };

  const handleConfirmSelection = () => {
    if (selectedRegion && !showFakeLoading) {
      onRegionSelect(selectedRegion);
    }
  };

  // Loading messages for realistic fake loading
  const loadingMessages = [
    "Connecting to Google Earth Engine...",
    "Downloading Sentinel-2 satellite imagery...",
    "Processing high-resolution data...",
    "Applying cloud filtering algorithms...",
    "Generating region previews..."
  ];

  const regionNames = ['Center', 'North', 'South', 'East', 'West'];

  return (
    <div className="max-w-7xl mx-auto">
      {showFakeLoading ? (
        /* Fake Loading Screen */
        <div className="text-center" style={{ backgroundColor: 'rgba(255, 0, 0, 0.1)', border: '2px solid red', padding: '20px' }}>
          <div className="inline-block bg-neon-100/20 backdrop-blur-md border border-neon-100/30 rounded-full px-8 py-3 mb-8">
            <span className="text-neon-100 font-bold text-sm md:text-base uppercase tracking-wider">
              🛰️ Downloading Satellite Data
            </span>
          </div>

          <h2 className="text-4xl sm:text-5xl md:text-6xl font-black font-deacon text-off-white mb-6">
            Processing {cityName}
            <span className="text-neon-100 block mt-2">SATELLITE IMAGERY</span>
          </h2>

          <p className="text-lg text-off-white/80 max-w-3xl mx-auto leading-relaxed mb-12">
            Fetching high-resolution satellite images from Google Earth Engine for detailed regional analysis.
          </p>

          {/* Progress Section */}
          <div className="bg-gradient-to-br from-off-white/15 to-off-white/5 backdrop-blur-xl rounded-3xl p-12 border border-off-white/20 shadow-2xl mb-8">
            {/* Progress Bar */}
            <div className="mb-8">
              <div className="flex items-center justify-between mb-4">
                <span className="text-off-white font-bold text-lg">🔄 Processing Progress</span>
                <span className="text-neon-100 font-black text-2xl">{Math.round(loadingProgress)}%</span>
              </div>
              <div className="w-full bg-off-white/20 rounded-full h-4 overflow-hidden shadow-inner">
                <div
                  className="h-full bg-gradient-to-r from-neon-100 via-green-400 to-neon-100 rounded-full transition-all duration-300 shadow-lg animate-pulse"
                  style={{ width: `${loadingProgress}%` }}
                ></div>
              </div>
            </div>

            {/* Current Status */}
            <div className="mb-8">
              <div className="text-neon-100 font-bold text-xl mb-2">
                {loadingMessages[Math.min(Math.floor(loadingProgress / 20), 4)]}
              </div>
              <div className="text-off-white/70 text-base">
                Currently processing: <span className="text-neon-100 font-semibold">{cityName} {regionNames[currentLoadingRegion]}</span>
              </div>
            </div>

            {/* Region Processing Grid */}
            <div className="grid grid-cols-5 gap-4">
              {regionNames.map((region, index) => (
                <div key={region} className={`bg-off-white/5 rounded-xl p-4 text-center transition-all duration-500 ${index <= currentLoadingRegion ? 'bg-neon-100/20 border border-neon-100/30' : ''
                  }`}>
                  <div className={`w-12 h-12 rounded-lg mx-auto mb-2 flex items-center justify-center transition-all duration-500 ${index < currentLoadingRegion
                      ? 'bg-neon-100 text-green'
                      : index === currentLoadingRegion
                        ? 'bg-neon-100/50 animate-pulse'
                        : 'bg-off-white/10'
                    }`}>
                    {index < currentLoadingRegion ? (
                      <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                      </svg>
                    ) : index === currentLoadingRegion ? (
                      <svg className="animate-spin h-5 w-5 text-neon-100" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                    ) : (
                      <span className="text-off-white/50 text-lg">
                        {region === 'North' && '⬆️'}
                        {region === 'South' && '⬇️'}
                        {region === 'East' && '➡️'}
                        {region === 'West' && '⬅️'}
                        {region === 'Center' && '🏙️'}
                      </span>
                    )}
                  </div>
                  <span className={`text-sm font-medium transition-colors duration-500 ${index <= currentLoadingRegion ? 'text-neon-100' : 'text-off-white/50'
                    }`}>
                    {region}
                  </span>
                </div>
              ))}
            </div>

            {/* Technical Details */}
            <div className="mt-8 text-off-white/60 text-sm text-center">
              <div className="mb-2">📡 Data Source: Sentinel-2 via Google Earth Engine</div>
              <div className="mb-2">🔍 Resolution: 10m per pixel | 🌤️ Cloud Coverage: &lt; 20%</div>
              <div>⏱️ Processing Time: ~60 seconds for high-quality imagery</div>
            </div>
          </div>
        </div>
      ) : (
        /* Normal Region Selection Interface */
        <>
          {/* Header */}
          <div className="text-center mb-12">
            <div className="inline-block bg-neon-100/20 backdrop-blur-md border border-neon-100/30 rounded-full px-8 py-3 mb-8">
              <span className="text-neon-100 font-bold text-sm md:text-base uppercase tracking-wider">
                📍 Select Region
              </span>
            </div>
            <h2 className="text-4xl sm:text-5xl md:text-6xl font-black font-deacon text-off-white mb-6">
              Choose Your Region in
              <span className="text-neon-100 block mt-2">{cityName}</span>
            </h2>
            <p className="text-lg text-off-white/80 max-w-3xl mx-auto leading-relaxed">
              Select a specific region for detailed biomass analysis. Each region shows real satellite imagery.
            </p>
          </div>

          {/* Region Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 mb-12">
            {regions.map((region) => (
              <div
                key={region.id}
                onClick={() => handleRegionClick(region)}
                className={`group relative bg-gradient-to-br from-off-white/15 to-off-white/5 backdrop-blur-xl rounded-3xl p-6 border-2 cursor-pointer transition-all duration-300 hover:scale-105 ${selectedRegion?.id === region.id
                    ? 'border-neon-100 shadow-2xl shadow-neon-100/20'
                    : 'border-off-white/20 hover:border-neon-100/50'
                  }`}
              >
                {/* Selection Indicator */}
                {selectedRegion?.id === region.id && (
                  <div className="absolute -top-3 -right-3 w-8 h-8 bg-neon-100 rounded-full flex items-center justify-center z-10">
                    <svg className="w-5 h-5 text-green" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                    </svg>
                  </div>
                )}

                {/* Satellite Preview */}
                <div className="relative mb-6 rounded-2xl overflow-hidden bg-off-white/5 aspect-video">
                  {region.previewImage ? (
                    <>
                      <img
                        src={region.previewImage}
                        alt={`${region.name} satellite view`}
                        className={`w-full h-full object-cover transition-opacity duration-500 ${previewsLoaded[region.id] ? 'opacity-100' : 'opacity-0'
                          }`}
                        onLoad={() => handleImageLoad(region.id)}
                        onError={(e) => {
                          const target = e.target as HTMLImageElement;
                          target.src = `${process.env.PUBLIC_URL}/placeholder-satellite.png`;
                        }}
                      />
                      {!previewsLoaded[region.id] && (
                        <div className="absolute inset-0 flex items-center justify-center">
                          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-neon-100"></div>
                        </div>
                      )}
                    </>
                  ) : (
                    <div className="w-full h-full flex items-center justify-center">
                      <div className="text-center">
                        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-neon-100 mx-auto mb-2"></div>
                        <span className="text-off-white/60 text-sm">Loading preview...</span>
                      </div>
                    </div>
                  )}

                  {/* Overlay */}
                  <div className="absolute inset-0 bg-gradient-to-t from-black/50 to-transparent"></div>
                </div>

                {/* Region Info */}
                <div className="space-y-3">
                  <h3 className="text-2xl font-black text-off-white group-hover:text-neon-100 transition-colors">
                    {region.name}
                  </h3>
                  <p className="text-off-white/70 text-sm leading-relaxed">
                    {region.description}
                  </p>

                  {/* Coordinates */}
                  <div className="text-xs text-off-white/50 font-mono">
                    📍 {region.coordinates.center[0].toFixed(4)}°, {region.coordinates.center[1].toFixed(4)}°
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Action Buttons */}
          <div className="flex flex-col sm:flex-row gap-6 justify-center items-center">
            <MagnetizeButton
              variant="outline"
              size="lg"
              particleCount={20}
              className="border-2 border-off-white/30 text-off-white hover:bg-off-white/10 px-12 py-5 text-lg font-bold"
              onClick={onBack}
              disabled={isLoading}
            >
              ← Back to City Selection
            </MagnetizeButton>

            <MagnetizeButton
              variant="magnetize"
              size="lg"
              particleCount={30}
              className="px-12 py-5 text-lg font-bold shadow-xl"
              onClick={handleConfirmSelection}
              disabled={!selectedRegion || isLoading}
            >
              {isLoading ? (
                <span className="flex items-center space-x-3">
                  <svg className="animate-spin h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  <span>Analyzing Region...</span>
                </span>
              ) : (
                `🛰️ Analyze ${selectedRegion?.name || 'Selected Region'}`
              )}
            </MagnetizeButton>
          </div>
        </>
      )}
    </div>
  );
};

export default RegionSelector;