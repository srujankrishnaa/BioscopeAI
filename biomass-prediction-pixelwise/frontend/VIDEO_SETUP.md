# 🎬 Video Setup Guide

## Video File Placement

Your Earth animation video needs to be placed in the `public` folder for the ML Model page to work.

### Steps:

1. **Place your video file:**
   ```
   biomass-prediction-pixelwise/frontend/public/earth-animation.mp4
   ```

2. **Optional: Add a fallback image** (if video doesn't load):
   ```
   biomass-prediction-pixelwise/frontend/public/earth-fallback.jpg
   ```

### File Structure:
```
frontend/
├── public/
│   ├── earth-animation.mp4  ← Your 1-minute Earth video goes here
│   ├── earth-fallback.jpg   ← Optional fallback image
│   ├── index.html
│   └── ...
├── src/
└── ...
```

### Video Requirements:

✅ **Format:** MP4 (H.264 codec recommended)  
✅ **Duration:** ~1 minute (loops automatically)  
✅ **Resolution:** 1920x1080 or higher for best quality  
✅ **File Size:** Keep under 50MB for faster loading  
✅ **Aspect Ratio:** 16:9 recommended  

### Optimization Tips:

If your video is too large, compress it:

**Using FFmpeg (command line):**
```bash
ffmpeg -i earth-animation.mp4 -vcodec h264 -crf 28 -preset fast earth-animation-compressed.mp4
```

**Using Online Tools:**
- https://www.freeconvert.com/video-compressor
- https://www.videosmaller.com/

### Alternative: Use a YouTube/Vimeo Video

If you prefer to host the video externally, replace the video tag in `MLModelPage.tsx` with:

```tsx
{/* For YouTube */}
<iframe
  src="https://www.youtube.com/embed/YOUR_VIDEO_ID?autoplay=1&mute=1&loop=1&playlist=YOUR_VIDEO_ID"
  className="w-full h-full"
  allow="autoplay; loop"
  frameBorder="0"
/>

{/* For Vimeo */}
<iframe
  src="https://player.vimeo.com/video/YOUR_VIDEO_ID?autoplay=1&loop=1&muted=1&background=1"
  className="w-full h-full"
  frameBorder="0"
  allow="autoplay; fullscreen"
/>
```

## Testing

After placing the video:

1. Start your frontend: `npm start`
2. Navigate to: `http://localhost:3000/model`
3. You should see the Earth video playing in the hero section! 🌍

## Troubleshooting

**Video not showing?**
- Check the file path is exactly: `public/earth-animation.mp4`
- Check browser console for errors (F12 → Console)
- Try a different browser (Chrome recommended)
- Check video codec compatibility

**Video loads slowly?**
- Compress the video file (see optimization tips above)
- Use a CDN for hosting (AWS S3, Cloudflare)
- Add video preload attribute: `preload="metadata"`

---

✨ **Need help?** The video will auto-play, loop, and be muted by default. The fallback image will show if the video fails to load.

