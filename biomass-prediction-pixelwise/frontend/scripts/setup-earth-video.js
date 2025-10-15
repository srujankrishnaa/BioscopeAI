#!/usr/bin/env node

/**
 * Script to help set up the earth animation video
 * This script provides instructions and options for getting the earth animation
 */

const fs = require('fs');
const path = require('path');

console.log('🌍 Earth Animation Video Setup');
console.log('================================');

const publicDir = path.join(__dirname, '..', 'public');
const videoFiles = ['earth-animation.mp4', 'earth-animation.webm'];

console.log('\n📁 Checking public directory:', publicDir);

// Check if video files exist
const missingFiles = videoFiles.filter(file => {
  const filePath = path.join(publicDir, file);
  return !fs.existsSync(filePath);
});

if (missingFiles.length === 0) {
  console.log('✅ All video files are present!');
  process.exit(0);
}

console.log('\n❌ Missing video files:', missingFiles.join(', '));
console.log('\n🔧 Options to fix this:');
console.log('\n1. Download from free sources:');
console.log('   - Pixabay: https://pixabay.com/videos/search/earth%20rotation/');
console.log('   - Pexels: https://www.pexels.com/search/videos/earth/');
console.log('   - Unsplash: https://unsplash.com/s/videos/earth');

console.log('\n2. Create with AI tools:');
console.log('   - RunwayML: https://runwayml.com/');
console.log('   - Stable Video Diffusion');
console.log('   - Luma AI Dream Machine');

console.log('\n3. Use NASA resources:');
console.log('   - NASA Goddard: https://svs.gsfc.nasa.gov/');
console.log('   - NASA Earth Observatory');

console.log('\n4. Simple CSS animation alternative:');
console.log('   - Use CSS keyframes for rotating earth effect');
console.log('   - Combine with particle effects');

console.log('\n📋 Requirements:');
console.log('   - Duration: 10-30 seconds (will loop)');
console.log('   - Resolution: 1920x1080 or higher');
console.log('   - Format: MP4 (H.264) and WebM (VP9) for browser compatibility');
console.log('   - File size: < 10MB for good loading performance');

console.log('\n🎬 Recommended search terms:');
console.log('   - "earth rotation animation"');
console.log('   - "planet earth spinning"');
console.log('   - "earth from space loop"');
console.log('   - "rotating globe animation"');

console.log('\n💡 Quick fix - Create placeholder videos:');
console.log('   Run: npm run create-placeholder-videos');

console.log('\n📝 After downloading:');
console.log('   1. Place files in: ' + publicDir);
console.log('   2. Name them: earth-animation.mp4 and earth-animation.webm');
console.log('   3. Restart your development server');

console.log('\n🚀 Your app will work with just the gradient background until videos are added!');