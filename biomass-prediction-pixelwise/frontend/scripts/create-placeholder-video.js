#!/usr/bin/env node

/**
 * Creates a simple placeholder video using HTML5 Canvas and MediaRecorder
 * This creates a basic rotating earth-like animation as a temporary solution
 */

const fs = require('fs');
const path = require('path');

console.log('🎬 Creating placeholder earth animation...');

// Create an HTML file that generates the video
const htmlContent = `
<!DOCTYPE html>
<html>
<head>
    <title>Earth Animation Generator</title>
    <style>
        body { margin: 0; padding: 20px; background: #000; color: white; font-family: Arial; }
        canvas { border: 1px solid #333; display: block; margin: 20px auto; }
        .controls { text-align: center; margin: 20px; }
        button { padding: 10px 20px; margin: 5px; background: #4CAF50; color: white; border: none; cursor: pointer; }
        button:hover { background: #45a049; }
        .status { text-align: center; margin: 20px; }
    </style>
</head>
<body>
    <h1>🌍 Earth Animation Generator</h1>
    <canvas id="canvas" width="1920" height="1080"></canvas>
    
    <div class="controls">
        <button onclick="startRecording()">🎥 Start Recording</button>
        <button onclick="stopRecording()">⏹️ Stop Recording</button>
        <button onclick="downloadVideo()">💾 Download Video</button>
    </div>
    
    <div class="status" id="status">Ready to record</div>
    
    <script>
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        const status = document.getElementById('status');
        
        let mediaRecorder;
        let recordedChunks = [];
        let animationId;
        let rotation = 0;
        
        // Animation function
        function animate() {
            // Clear canvas
            ctx.fillStyle = '#000011';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            // Draw stars
            ctx.fillStyle = 'white';
            for (let i = 0; i < 200; i++) {
                const x = (i * 123) % canvas.width;
                const y = (i * 456) % canvas.height;
                const size = Math.sin(i) * 2 + 2;
                ctx.fillRect(x, y, size, size);
            }
            
            // Draw earth
            const centerX = canvas.width / 2;
            const centerY = canvas.height / 2;
            const radius = 300;
            
            // Earth base
            const gradient = ctx.createRadialGradient(centerX - 100, centerY - 100, 0, centerX, centerY, radius);
            gradient.addColorStop(0, '#4CAF50');
            gradient.addColorStop(0.3, '#2E7D32');
            gradient.addColorStop(0.7, '#1B5E20');
            gradient.addColorStop(1, '#0D2818');
            
            ctx.fillStyle = gradient;
            ctx.beginPath();
            ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
            ctx.fill();
            
            // Continents (simplified)
            ctx.fillStyle = '#8BC34A';
            ctx.save();
            ctx.translate(centerX, centerY);
            ctx.rotate(rotation);
            
            // Draw some continent shapes
            for (let i = 0; i < 8; i++) {
                const angle = (i / 8) * Math.PI * 2;
                const x = Math.cos(angle) * (radius * 0.7);
                const y = Math.sin(angle) * (radius * 0.4);
                const size = 30 + Math.sin(rotation + i) * 10;
                
                ctx.beginPath();
                ctx.arc(x, y, size, 0, Math.PI * 2);
                ctx.fill();
            }
            
            ctx.restore();
            
            // Atmosphere glow
            const glowGradient = ctx.createRadialGradient(centerX, centerY, radius, centerX, centerY, radius + 50);
            glowGradient.addColorStop(0, 'rgba(100, 200, 255, 0.3)');
            glowGradient.addColorStop(1, 'rgba(100, 200, 255, 0)');
            
            ctx.fillStyle = glowGradient;
            ctx.beginPath();
            ctx.arc(centerX, centerY, radius + 50, 0, Math.PI * 2);
            ctx.fill();
            
            // Update rotation
            rotation += 0.01;
            
            animationId = requestAnimationFrame(animate);
        }
        
        function startRecording() {
            recordedChunks = [];
            const stream = canvas.captureStream(30); // 30 FPS
            
            mediaRecorder = new MediaRecorder(stream, {
                mimeType: 'video/webm;codecs=vp9'
            });
            
            mediaRecorder.ondataavailable = function(event) {
                if (event.data.size > 0) {
                    recordedChunks.push(event.data);
                }
            };
            
            mediaRecorder.onstop = function() {
                status.textContent = 'Recording stopped. Click download to save.';
            };
            
            mediaRecorder.start();
            status.textContent = 'Recording... (record for 10-15 seconds)';
            
            // Start animation
            animate();
        }
        
        function stopRecording() {
            if (mediaRecorder && mediaRecorder.state !== 'inactive') {
                mediaRecorder.stop();
                cancelAnimationFrame(animationId);
            }
        }
        
        function downloadVideo() {
            if (recordedChunks.length === 0) {
                alert('No recording available. Please record first.');
                return;
            }
            
            const blob = new Blob(recordedChunks, { type: 'video/webm' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'earth-animation.webm';
            a.click();
            URL.revokeObjectURL(url);
            
            status.textContent = 'Video downloaded! Rename to earth-animation.webm and place in public folder.';
        }
        
        // Start animation immediately for preview
        animate();
    </script>
    
    <div style="margin: 20px; padding: 20px; background: #333; border-radius: 5px;">
        <h3>📋 Instructions:</h3>
        <ol>
            <li>Click "Start Recording" to begin capturing the animation</li>
            <li>Let it record for 10-15 seconds (it will loop)</li>
            <li>Click "Stop Recording" to finish</li>
            <li>Click "Download Video" to save the file</li>
            <li>Rename the downloaded file to "earth-animation.webm"</li>
            <li>Place it in your public folder: <code>biomass-prediction-pixelwise/frontend/public/</code></li>
            <li>For MP4 version, use online converter or FFmpeg</li>
        </ol>
        
        <h3>🔧 For MP4 conversion (optional):</h3>
        <p>Use FFmpeg: <code>ffmpeg -i earth-animation.webm earth-animation.mp4</code></p>
        <p>Or use online converters like CloudConvert or Convertio</p>
    </div>
</body>
</html>
`;

// Write the HTML file
const outputPath = path.join(__dirname, '..', 'public', 'earth-video-generator.html');
fs.writeFileSync(outputPath, htmlContent);

console.log('✅ Created earth video generator!');
console.log('📂 Open this file in your browser:', outputPath);
console.log('🌐 Or visit: http://localhost:3000/earth-video-generator.html');
console.log('\n📋 Steps:');
console.log('1. Open the HTML file in your browser');
console.log('2. Click "Start Recording"');
console.log('3. Wait 10-15 seconds');
console.log('4. Click "Stop Recording"');
console.log('5. Click "Download Video"');
console.log('6. Rename to earth-animation.webm and place in public folder');
console.log('\n🎬 This creates a simple rotating earth animation as a placeholder!');