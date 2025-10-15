/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Your Bowwe site colors
        'green': '#122315',
        'off-white': '#f3ede4',
        'neon-100': '#55dd4a',
        'neon-80': '#77e46e',
        'neon-130': '#3b9b34',
        'neon-50': '#abe59e',
        'turquoise-100': '#73d3eb',
        'white': '#ffffff',
        'black': '#000000',
        // Additional colors for illustration
        'orange': '#E9A26A',
        'coral': '#FF6B6B',
        'blue': '#4A90E2',
      },
      fontFamily: {
        'deacon': ['Deacon', 'sans-serif'],
        'graphik': ['Graphik', 'sans-serif'],
        'sans': ['Graphik', 'Inter', 'sans-serif'],
        'display': ['Deacon', 'sans-serif'],
      },
      animation: {
        'fade-in-up': 'fadeInUp 0.6s ease-out',
        'parallax': 'parallax 20s ease-in-out infinite',
        'fadeIn': 'fadeIn 0.8s ease-out',
        'slideUp': 'slideUp 0.6s ease-out',
        'slideRight': 'slideRight 0.6s ease-out',
        'slideLeft': 'slideLeft 0.6s ease-out',
        'shake': 'shake 0.5s ease-in-out',
      },
      keyframes: {
        fadeInUp: {
          '0%': {
            opacity: '0',
            transform: 'translateY(30px)',
          },
          '100%': {
            opacity: '1',
            transform: 'translateY(0)',
          },
        },
        parallax: {
          '0%, 100%': {
            transform: 'translateY(0px)',
          },
          '50%': {
            transform: 'translateY(-10px)',
          },
        },
        fadeIn: {
          '0%': {
            opacity: '0',
          },
          '100%': {
            opacity: '1',
          },
        },
        slideUp: {
          '0%': {
            opacity: '0',
            transform: 'translateY(40px)',
          },
          '100%': {
            opacity: '1',
            transform: 'translateY(0)',
          },
        },
        slideRight: {
          '0%': {
            opacity: '0',
            transform: 'translateX(-40px)',
          },
          '100%': {
            opacity: '1',
            transform: 'translateX(0)',
          },
        },
        slideLeft: {
          '0%': {
            opacity: '0',
            transform: 'translateX(40px)',
          },
          '100%': {
            opacity: '1',
            transform: 'translateX(0)',
          },
        },
        shake: {
          '0%, 100%': {
            transform: 'translateX(0)',
          },
          '10%, 30%, 50%, 70%, 90%': {
            transform: 'translateX(-5px)',
          },
          '20%, 40%, 60%, 80%': {
            transform: 'translateX(5px)',
          },
        },
      },
    },
  },
  plugins: [],
}