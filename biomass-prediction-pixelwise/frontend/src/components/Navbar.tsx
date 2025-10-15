import React, { useState, useEffect } from 'react';

const Navbar: React.FC = () => {
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      const isScrolled = window.scrollY > 10;
      if (isScrolled !== scrolled) {
        setScrolled(isScrolled);
      }
    };

    window.addEventListener('scroll', handleScroll);
    return () => {
      window.removeEventListener('scroll', handleScroll);
    };
  }, [scrolled]);

  return (
    <nav 
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${scrolled ? 'bg-green/90 backdrop-blur-md py-3' : 'bg-transparent py-5'}`}
    >
      <div className="max-w-7xl mx-auto px-6 flex justify-between items-center">
        {/* Logo */}
        <div className="flex items-center space-x-2">
          <div className="w-8 h-8 bg-neon-100 rounded-full flex items-center justify-center">
            <span className="text-green font-bold text-sm">ML</span>
          </div>
          <span className="text-xl font-bold tracking-wider text-off-white font-deacon">BIOSCOPE</span>
        </div>

        {/* Navigation Links - Desktop */}
        <div className="hidden md:flex items-center space-x-8">
          <a href="/" className="text-off-white hover:text-neon-100 transition-colors text-sm font-medium">Home</a>
          <a href="/login" className="text-off-white hover:text-neon-100 transition-colors text-sm font-medium">
            Log In
          </a>
          <a href="/dashboard" className="bg-neon-100 text-green px-6 py-2 rounded-full font-semibold text-sm hover:bg-neon-80 transition-all">
            Get Started
          </a>
        </div>

        {/* Mobile Menu Button */}
        <button className="md:hidden text-off-white">
          <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
          </svg>
        </button>
      </div>
    </nav>
  );
};

export default Navbar;