import React, { useEffect, useState } from 'react';
import { Globe } from '../components/ui/globe';
import { MagnetizeButton } from '../components/ui/magnetize-button';

const LandingPage: React.FC = () => {
  const [scrollY, setScrollY] = useState(0);

  useEffect(() => {
    const handleScroll = () => setScrollY(window.scrollY);
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <div className="bg-green text-off-white font-graphik">
      {/* Hero Section - Full Screen */}
      <section className="h-screen relative overflow-hidden">
        {/* Bright Background Image */}
        <div className="absolute inset-0 z-0">
          <img
            src="/kikin-landscape.png"
            alt="Beautiful landscape illustration with mountains, trees, river, and outdoor elements"
            className="w-full h-full object-cover brightness-125 contrast-110 saturate-110"
          />
          {/* Very light overlay to maintain brightness */}
          <div className="absolute inset-0 bg-black/10"></div>
        </div>

        {/* Navigation */}
        <nav className="relative z-50 flex justify-between items-center px-8 py-6">
          <div className="text-xl font-bold tracking-wider text-white font-deacon"
            style={{ textShadow: '2px 2px 8px rgba(0,0,0,0.7)' }}>
            BIOSCOPE
          </div>
          <div className="flex items-center space-x-6">
            <button
              type="button"
              onClick={() => window.location.href = '/login'}
              className="text-white hover:text-neon-100 transition-colors text-sm font-medium"
              style={{ textShadow: '1px 1px 4px rgba(0,0,0,0.7)' }}
            >
              LOG IN
            </button>
            <button
              type="button"
              onClick={() => window.location.href = '/model'}
              className="bg-neon-100 text-green px-6 py-2 rounded-full font-semibold text-sm hover:bg-neon-80 transition-all shadow-lg hover:scale-105 transform"
            >
              GET STARTED
            </button>
          </div>
        </nav>

        {/* Hero Content - Text as part of image */}
        <div className="relative z-40 h-full flex flex-col justify-center items-center px-8 -mt-20">
          <div className="text-center">
            <h1 className="text-6xl md:text-8xl font-black leading-none mb-6 tracking-tight font-deacon">
              <div className="text-white mb-2"
                style={{ textShadow: '3px 3px 12px rgba(0,0,0,0.8), 1px 1px 3px rgba(0,0,0,0.9)' }}>
                MAPPING
              </div>
              <div className="text-neon-100"
                style={{ textShadow: '3px 3px 12px rgba(0,0,0,0.8), 1px 1px 3px rgba(0,0,0,0.9)' }}>
                THE FUTURE
              </div>
            </h1>

            {/* Clean text without backdrop */}
            <p className="text-lg text-white max-w-2xl mx-auto leading-relaxed mb-2"
              style={{ textShadow: '2px 2px 8px rgba(0,0,0,0.8)' }}>
              Advanced satellite biomass analysis powered by NASA GIBS data.
            </p>
            <p className="text-lg text-white max-w-2xl mx-auto leading-relaxed mb-8"
              style={{ textShadow: '2px 2px 8px rgba(0,0,0,0.8)' }}>
              Get access to real-time vegetation mapping for sustainable decision making.
            </p>

            <MagnetizeButton
              variant="magnetize"
              size="xl"
              particleCount={25}
              onClick={() => window.location.href = '/model'}
            >
              GET STARTED
            </MagnetizeButton>
          </div>
        </div>
      </section>

      {/* Page 1 - What Our ML Model Can Do */}
      <section className="h-screen bg-off-white text-green relative overflow-hidden">
        {/* Floating Badge Images - Positioned at screen corners */}
        <div className="absolute inset-0 pointer-events-none">
          {/* Top Left Corner */}
          <div
            className={`absolute top-8 left-8 w-28 h-28 transform rotate-12 hover:scale-110 transition-all duration-700 pointer-events-auto ${scrollY > window.innerHeight * 0.8 ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-10'
              }`}
          >
            <img src="/badge-realtime.png" alt="Real-time Analysis Badge" className="w-full h-full object-contain drop-shadow-xl" />
          </div>

          {/* Top Right Corner */}
          <div
            className={`absolute top-9 right-9 w-30 h-28 transform -rotate-6 hover:scale-110 transition-all duration-700 delay-200 pointer-events-auto ${scrollY > window.innerHeight * 0.8 ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-10'
              }`}
          >
            <img src="/badge-global.png" alt="Global Coverage Badge" className="w-full h-full object-contain drop-shadow-xl" />
          </div>

          {/* Bottom Left Corner */}
          <div
            className={`absolute bottom-8 left-8 w-28 h-28 transform -rotate-12 hover:scale-110 transition-all duration-700 delay-400 pointer-events-auto ${scrollY > window.innerHeight * 1.2 ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-10'
              }`}
          >
            <img src="/badge-accurate.png" alt="Accurate Analysis Badge" className="w-full h-full object-contain drop-shadow-xl" />
          </div>

          {/* Bottom Right Corner */}
          <div
            className={`absolute bottom-8 right-8 w-28 h-28 transform rotate-15 hover:scale-110 transition-all duration-700 delay-600 pointer-events-auto ${scrollY > window.innerHeight * 1.2 ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-10'
              }`}
          >
            <img src="/badge-sustainable.png" alt="Sustainable Solutions Badge" className="w-full h-full object-contain drop-shadow-xl" />
          </div>
        </div>

        <div className="h-full flex items-center justify-center relative z-10">
          <div className="max-w-7xl mx-auto px-8 grid md:grid-cols-2 gap-16 items-center">
            {/* Left Side - Text Content */}
            <div className="text-center md:text-left order-2 md:order-1">
              <h2 className="text-5xl md:text-6xl font-black font-deacon text-green leading-tight mb-6">
                FOR THE<br />
                <span className="text-neon-130 drop-shadow-lg">PLANET</span>
              </h2>
              <h3 className="text-2xl md:text-3xl font-bold text-green mb-6 leading-tight">
                Our Powerful AI Model Delivers Insights for a Better World
              </h3>
              <p className="text-lg text-green leading-relaxed mb-8 max-w-xl font-medium">
                BioScope ML harnesses advanced satellite imagery and machine learning to provide
                unprecedented insights into global vegetation health and biomass distribution.
              </p>
              <MagnetizeButton
                variant="magnetize"
                size="xl"
                particleCount={20}
                onClick={() => window.location.href = '/model'}
              >
                EXPLORE OUR MODEL
              </MagnetizeButton>
            </div>

            {/* Right Side - Centered Illustration */}
            <div className="flex justify-center items-center order-1 md:order-2">
              <div className="relative w-full max-w-lg">
                <img
                  src="/planet-care-illustration.png"
                  alt="People caring for the planet - environmental sustainability illustration"
                  className="w-full h-auto object-contain drop-shadow-2xl"
                />
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Page 2 - Statistics & Performance with Interactive Globe */}
      <section className="h-screen bg-green text-off-white flex items-center relative overflow-hidden">
        <div className="max-w-6xl mx-auto px-8 grid md:grid-cols-2 gap-12 items-center z-10 relative">
          <div>
            <div className="w-12 h-12 bg-neon-100 rounded-full flex items-center justify-center mb-6">
              <span className="text-green font-bold text-xl">🌍</span>
            </div>
            <h2 className="text-4xl md:text-5xl font-black font-deacon text-off-white leading-tight mb-6">
              GLOBAL<br />COVERAGE
            </h2>
            <p className="text-base text-off-white/80 leading-relaxed mb-8">
              Our platform processes millions of satellite data points daily across 195 countries,
              delivering unprecedented accuracy in biomass analysis and environmental monitoring worldwide.
            </p>

            {/* Key Statistics */}
            <div className="grid grid-cols-2 gap-6 mb-8">
              <div className="bg-off-white/10 rounded-lg p-4 backdrop-blur-sm">
                <div className="text-3xl font-black text-neon-100 mb-1">2.5M+</div>
                <div className="text-xs text-off-white/70">Data Points Daily</div>
              </div>
              <div className="bg-off-white/10 rounded-lg p-4 backdrop-blur-sm">
                <div className="text-3xl font-black text-neon-100 mb-1">99.9%</div>
                <div className="text-xs text-off-white/70">Accuracy Rate</div>
              </div>
              <div className="bg-off-white/10 rounded-lg p-4 backdrop-blur-sm">
                <div className="text-3xl font-black text-neon-100 mb-1">24/7</div>
                <div className="text-xs text-off-white/70">Processing</div>
              </div>
              <div className="bg-off-white/10 rounded-lg p-4 backdrop-blur-sm">
                <div className="text-3xl font-black text-neon-100 mb-1">195</div>
                <div className="text-xs text-off-white/70">Countries</div>
              </div>
            </div>
          </div>

          {/* Interactive Globe */}
          <div className="relative h-96 w-96 mx-auto">
            <Globe
              className="opacity-80"
              config={{
                width: 600,
                height: 600,
                onRender: () => { },
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
                markers: [
                  // Major biomass monitoring locations
                  { location: [17.4065, 78.4772], size: 0.08 }, // Hyderabad, India
                  { location: [19.076, 72.8777], size: 0.1 },   // Mumbai, India
                  { location: [12.9716, 77.5946], size: 0.08 }, // Bangalore, India
                  { location: [28.7041, 77.1025], size: 0.1 },  // Delhi, India
                  { location: [-23.5505, -46.6333], size: 0.1 }, // São Paulo, Brazil
                  { location: [40.7128, -74.006], size: 0.1 },   // New York, USA
                  { location: [51.5074, -0.1278], size: 0.08 },  // London, UK
                  { location: [35.6762, 139.6503], size: 0.08 }, // Tokyo, Japan
                  { location: [-33.8688, 151.2093], size: 0.07 }, // Sydney, Australia
                  { location: [55.7558, 37.6176], size: 0.07 },  // Moscow, Russia
                ],
              }}
            />

            {/* Globe overlay text */}
            <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
              <div className="text-center">
                <div className="text-2xl font-black text-neon-100 mb-1">LIVE</div>
                <div className="text-xs text-off-white/70">Monitoring</div>
              </div>
            </div>
          </div>
        </div>

        {/* Background gradient overlay */}
        <div className="absolute inset-0 bg-gradient-to-r from-green via-transparent to-green opacity-50"></div>
      </section>

      {/* Page 3 - Our Process */}
      <section className="h-screen bg-off-white text-green relative overflow-hidden flex items-center">
        {/* Background Image - Full Opacity */}
        <div className="absolute inset-0 z-0">
          <img
            src="/partnership-landscape.png"
            alt="Partnership and collaboration in natural landscape"
            className="w-full h-full object-cover"
          />
        </div>

        {/* Content Layout */}
        <div className="relative z-10 w-full h-full flex flex-col">
          {/* Top Right - Heading */}
          <div className="flex justify-end pt-16 pr-16">
            <div className="max-w-md text-right">
              <h2 className="text-5xl md:text-6xl font-black leading-tight font-sans text-white"
                style={{ textShadow: '3px 3px 12px rgba(0,0,0,0.8), 1px 1px 3px rgba(0,0,0,0.9)' }}>
                OUR<br />
                <span className="text-neon-100"
                  style={{ textShadow: '3px 3px 12px rgba(0,0,0,0.8), 1px 1px 3px rgba(0,0,0,0.9)' }}>
                  PROCESS
                </span>
              </h2>
            </div>
          </div>

          {/* Bottom Left - Description */}
          <div className="flex-grow flex items-end pb-16 pl-16">
            <div className="max-w-lg">
              <p className="text-xl leading-relaxed font-sans font-medium text-white"
                style={{ textShadow: '2px 2px 8px rgba(0,0,0,0.8), 1px 1px 2px rgba(0,0,0,0.9)' }}>
                We capture multi-spectral satellite imagery, process it through advanced
                machine learning models, and deliver precise biomass predictions with 99.9% accuracy.
              </p>
              <p className="text-lg mt-4 font-sans text-white/90"
                style={{ textShadow: '2px 2px 8px rgba(0,0,0,0.8), 1px 1px 2px rgba(0,0,0,0.9)' }}>
                From space to insights, our AI transforms raw satellite data into actionable
                environmental intelligence for sustainable decision making.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Page 4 - Call to Action with Environmental Landscape */}
      <section className="h-screen bg-green text-off-white relative overflow-hidden">
        {/* Environmental Landscape Background */}
        <div className="absolute inset-0 z-0">
          <img
            src="/environmental-landscape.png"
            alt="Environmental landscape with mountains, forest, and wildlife"
            className="w-full h-full object-cover"
          />
          <div className="absolute inset-0 bg-green/30"></div>
        </div>

        {/* Main Content - Organized Layout */}
        <div className="relative z-10 h-full flex flex-col">
          {/* Top Section - Main Call to Action */}
          <div className="flex-grow flex items-center justify-center">
            <div className="max-w-5xl mx-auto px-8 text-center">
              {/* Main Heading */}
              <h1 className="text-6xl md:text-8xl font-black font-deacon leading-tight mb-8"
                style={{ textShadow: '4px 4px 16px rgba(0,0,0,0.9), 2px 2px 6px rgba(0,0,0,1)' }}>
                <span className="text-off-white">START YOUR</span><br />
                <span className="text-neon-100">IMPACT TODAY</span>
              </h1>

              {/* Subtitle */}
              <p className="text-xl md:text-2xl text-off-white mb-12 max-w-4xl mx-auto font-medium leading-relaxed"
                style={{ textShadow: '3px 3px 12px rgba(0,0,0,0.9), 1px 1px 4px rgba(0,0,0,1)' }}>
                Join organizations worldwide using BioScope ML to make data-driven decisions
                for environmental sustainability and positive climate impact.
              </p>

              {/* Call to Action Buttons */}
              <div className="flex flex-col sm:flex-row gap-6 justify-center items-center">
                <MagnetizeButton
                  variant="magnetize"
                  size="xl"
                  particleCount={30}
                  className="px-12 py-5 text-xl font-bold"
                  onClick={() => window.location.href = '/model'}
                >
                  GET STARTED FREE
                </MagnetizeButton>
                <MagnetizeButton
                  variant="outline"
                  size="xl"
                  particleCount={25}
                  className="border-2 border-off-white text-off-white px-12 py-5 text-xl font-bold hover:bg-off-white hover:text-green shadow-2xl hover:scale-105 transform transition-all"
                >
                  SCHEDULE DEMO
                </MagnetizeButton>
              </div>
            </div>
          </div>

          {/* Bottom Section - Contact Information */}
          <div className="pb-8 px-8">
            <div className="max-w-6xl mx-auto">
              <div className="grid md:grid-cols-3 gap-6">
                {/* Contact Info */}
                <div className="bg-off-white/15 backdrop-blur-md rounded-xl p-6 border border-off-white/30 shadow-xl">
                  <div className="flex items-center mb-4">
                    <div className="w-8 h-8 bg-neon-100 rounded-full flex items-center justify-center mr-4">
                      <span className="text-green font-bold text-sm">✉</span>
                    </div>
                    <span className="font-bold text-off-white text-lg">Get in touch</span>
                  </div>
                  <p className="text-off-white/90 font-medium">contact@bioscope-ml.com</p>
                  <p className="text-off-white/70 text-sm mt-1">Ready to help 24/7</p>
                </div>

                {/* Company Info */}
                <div className="bg-off-white/15 backdrop-blur-md rounded-xl p-6 border border-off-white/30 shadow-xl">
                  <div className="flex items-center mb-4">
                    <div className="w-8 h-8 bg-neon-100 rounded-full flex items-center justify-center mr-4">
                      <span className="text-green font-bold text-sm">🏢</span>
                    </div>
                    <span className="font-bold text-off-white text-lg">BioScope ML</span>
                  </div>
                  <p className="text-off-white/90 font-medium">Environmental Intelligence Platform</p>
                  <p className="text-off-white/70 text-sm mt-1">Powered by NASA GIBS data</p>
                </div>

                {/* Social Links */}
                <div className="bg-off-white/15 backdrop-blur-md rounded-xl p-6 border border-off-white/30 shadow-xl">
                  <div className="flex items-center mb-4">
                    <div className="w-8 h-8 bg-neon-100 rounded-full flex items-center justify-center mr-4">
                      <span className="text-green font-bold text-sm">🔗</span>
                    </div>
                    <span className="font-bold text-off-white text-lg">Follow us</span>
                  </div>
                  <div className="flex space-x-4">
                    <button className="text-off-white/80 hover:text-neon-100 transition-colors font-medium">LinkedIn</button>
                    <button className="text-off-white/80 hover:text-neon-100 transition-colors font-medium">Twitter</button>
                    <button className="text-off-white/80 hover:text-neon-100 transition-colors font-medium">GitHub</button>
                  </div>
                  <p className="text-off-white/70 text-sm mt-2">Join our community</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};

export default LandingPage;