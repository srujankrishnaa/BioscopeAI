#!/usr/bin/env python3
"""
Create sample cached images for testing cache fallback functionality
"""
import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def create_sample_heatmap(city, region, width=400, height=300):
    """Create a sample heatmap image for testing"""
    # Create a gradient background
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    
    # Create a simple gradient effect
    for y in range(height):
        for x in range(width):
            # Create a radial gradient from center
            center_x, center_y = width // 2, height // 2
            distance = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
            max_distance = (width ** 2 + height ** 2) ** 0.5 / 2
            
            # Normalize distance to 0-1
            normalized_distance = min(distance / max_distance, 1.0)
            
            # Create color based on distance (green in center, red at edges)
            red = int(255 * normalized_distance)
            green = int(255 * (1 - normalized_distance))
            blue = 50
            
            img.putpixel((x, y), (red, green, blue))
    
    # Add text overlay
    try:
        # Try to use a default font
        font = ImageFont.load_default()
    except:
        font = None
    
    # Add title
    title = f"{city.title()} - {region.title()}"
    draw.text((10, 10), title, fill='white', font=font)
    draw.text((10, 30), "Sample Biomass Heatmap", fill='white', font=font)
    draw.text((10, height - 30), "Cache Preview", fill='yellow', font=font)
    
    return img

def main():
    """Create sample cache for testing"""
    cache_dir = Path("./outputs/region_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Cities to create cache for
    cities = ["Ranchi", "Delhi", "Mumbai", "Bangalore", "Chennai"]
    regions = ["center", "north", "south", "east", "west"]
    
    print("🎨 Creating sample cached images...")
    
    for city in cities:
        city_dir = cache_dir / city
        city_dir.mkdir(exist_ok=True)
        
        for region in regions:
            img = create_sample_heatmap(city, region)
            img_path = city_dir / f"{region}.png"
            img.save(img_path)
            print(f"   Created: {img_path}")
    
    print(f"\n✅ Sample cache created in {cache_dir}")
    print(f"   Total images: {len(cities) * len(regions)}")
    
    # List created files
    print("\n📁 Cache structure:")
    for city_dir in cache_dir.iterdir():
        if city_dir.is_dir():
            images = list(city_dir.glob("*.png"))
            print(f"   {city_dir.name}/: {len(images)} images")

if __name__ == "__main__":
    main()