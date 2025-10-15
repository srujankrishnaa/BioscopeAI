import React, { useState } from 'react';
import Navbar from '../components/Navbar';
import Footer from '../components/Footer';

const LoginPage: React.FC = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [rememberMe, setRememberMe] = useState(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    // Handle login logic here
    console.log('Login attempt with:', { email, password, rememberMe });
  };

  return (
    <div className="min-h-screen bg-green text-off-white font-graphik flex flex-col">
      <Navbar />
      
      <div className="flex-grow flex items-center justify-center px-4 py-24">
        <div className="w-full max-w-md">
          {/* Login Card */}
          <div className="bg-off-white/5 backdrop-blur-md rounded-2xl p-8 shadow-xl border border-off-white/10">
            {/* Header */}
            <div className="text-center mb-8">
              <div className="w-12 h-12 bg-neon-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <span className="text-green font-bold text-xl">ML</span>
              </div>
              <h2 className="text-3xl font-bold font-deacon text-off-white">WELCOME BACK</h2>
              <p className="text-off-white/70 mt-2">Sign in to access your BioScope dashboard</p>
            </div>
            
            {/* Login Form */}
            <form onSubmit={handleSubmit} className="space-y-6">
              {/* Email Field */}
              <div>
                <label htmlFor="email" className="block text-sm font-medium text-off-white mb-2">
                  Email Address
                </label>
                <input
                  id="email"
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="w-full px-4 py-3 bg-off-white/10 border border-off-white/20 rounded-lg focus:outline-none focus:ring-2 focus:ring-neon-100 focus:border-transparent text-off-white placeholder-off-white/50"
                  placeholder="you@example.com"
                  required
                />
              </div>
              
              {/* Password Field */}
              <div>
                <div className="flex justify-between items-center mb-2">
                  <label htmlFor="password" className="block text-sm font-medium text-off-white">
                    Password
                  </label>
                  <a href="#" className="text-xs text-neon-100 hover:text-neon-80 transition-colors">
                    Forgot password?
                  </a>
                </div>
                <input
                  id="password"
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="w-full px-4 py-3 bg-off-white/10 border border-off-white/20 rounded-lg focus:outline-none focus:ring-2 focus:ring-neon-100 focus:border-transparent text-off-white placeholder-off-white/50"
                  placeholder="••••••••"
                  required
                />
              </div>
              
              {/* Remember Me */}
              <div className="flex items-center">
                <input
                  id="remember-me"
                  type="checkbox"
                  checked={rememberMe}
                  onChange={(e) => setRememberMe(e.target.checked)}
                  className="h-4 w-4 bg-off-white/10 border-off-white/20 rounded focus:ring-neon-100 text-neon-100"
                />
                <label htmlFor="remember-me" className="ml-2 block text-sm text-off-white/70">
                  Remember me
                </label>
              </div>
              
              {/* Submit Button */}
              <button
                type="submit"
                className="w-full bg-neon-100 text-green px-6 py-3 rounded-lg font-bold text-base hover:bg-neon-80 transition-all shadow-lg hover:scale-105 transform"
              >
                Sign In
              </button>
            </form>
            
            {/* Sign Up Link */}
            <div className="mt-8 text-center">
              <p className="text-off-white/70">
                Don't have an account?{' '}
                <a href="#" className="text-neon-100 hover:text-neon-80 transition-colors font-medium">
                  Sign up
                </a>
              </p>
            </div>
          </div>
        </div>
      </div>
      
      <Footer />
    </div>
  );
};

export default LoginPage;