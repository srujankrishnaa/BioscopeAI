"""
City-specific configuration for satellite image export limits
Defines maximum pixel dimensions for each city to avoid GEE payload limit errors
"""

# City-specific maximum pixel dimensions
# These values are calibrated to avoid the 50MB request limit while maintaining quality
CITY_MAX_PX = {
    # Major metros - can handle higher resolution but with conservative limits
    "Hyderabad": 1536,      # Problematic city - reduced from 2048
    "Bangalore": 1792,      # Slightly reduced for safety
    "Bengaluru": 1792,      # Alternate spelling
    "Chennai": 1792,
    "Mumbai": 1792,
    "Delhi": 1792,
    "Kolkata": 1792,
    "Pune": 1792,
    
    # State/region names - often have large bounding boxes
    "Punjab": 1536,         # Large area - reduced resolution
    "Kerala": 1536,         # Large area - reduced resolution
    
    # Smaller cities - conservative limits
    "Ludhiana": 1536,       # Problematic city - reduced from 2048
    "Guwahati": 1536,
    "Jaipur": 1536,
    "Lucknow": 1536,
    "Indore": 1536,
    "Ahmedabad": 1536,
    "Surat": 1536,
    "Nagpur": 1536,
    "Coimbatore": 1536,
    "Kochi": 1536,
    "Bhopal": 1536,
    "Visakhapatnam": 1536,
    "Patna": 1536,
    "Vadodara": 1536,
    "Agra": 1536,
    "Nashik": 1536,
    "Faridabad": 1536,
    "Meerut": 1536,
    "Rajkot": 1536,
    "Kalyan": 1536,
    "Vasai": 1536,
    "Dhanbad": 1536,
    "Amritsar": 1536,
    "Navi Mumbai": 1536,
    "Allahabad": 1536,
    "Ranchi": 1536,
    "Howrah": 1536,
    "Coimbatore": 1536,
    "Jabalpur": 1536,
    "Gwalior": 1536,
    "Vijayawada": 1536,
    "Jodhpur": 1536,
    "Madurai": 1536,
    "Raipur": 1536,
    "Kota": 1536,
    "Chandigarh": 1536,
    "Guwahati": 1536,
    "Hubli": 1536,
    "Dharwad": 1536,
    "Mysore": 1536,
    "Tiruchirappalli": 1536,
    "Bareilly": 1536,
    "Gurgaon": 1536,
    "Aligarh": 1536,
    "Jalandhar": 1536,
    "Tiruppur": 1536,
    "Bhubaneswar": 1536,
    "Salem": 1536,
    "Mira": 1536,
    "Thane": 1536,
    "Bhiwandi": 1536,
    "Saharanpur": 1536,
    "Gorakhpur": 1536,
    "Bikaner": 1536,
    "Amravati": 1536,
    "Noida": 1536,
    "Jamshedpur": 1536,
    "Bhilai": 1536,
    "Cuttack": 1536,
    "Firozabad": 1536,
    "Kochi": 1536,
    "Nellore": 1536,
    "Bhavnagar": 1536,
    "Dehradun": 1536,
    "Durgapur": 1536,
    "Asansol": 1536,
    "Rourkela": 1536,
    "Nanded": 1536,
    "Kolhapur": 1536,
    "Ajmer": 1536,
    "Akola": 1536,
    "Gulbarga": 1536,
    "Jamnagar": 1536,
    "Ujjain": 1536,
    "Loni": 1536,
    "Siliguri": 1536,
    "Jhansi": 1536,
    "Ulhasnagar": 1536,
    "Nellore": 1536,
    "Jammu": 1536,
    "Sangli": 1536,
    "Mangalore": 1536,
    "Erode": 1536,
    "Belgaum": 1536,
    "Ambattur": 1536,
    "Tirunelveli": 1536,
    "Malegaon": 1536,
    "Gaya": 1536,
    "Jalgaon": 1536,
    "Udaipur": 1536,
    "Maheshtala": 1536,
}

# Default maximum for cities not in the list
DEFAULT_MAX_PX = 1536

# Maximum request size in bytes (50MB with safety margin)
MAX_REQUEST_BYTES = 48 * 1024 * 1024  # 48MB

# GEE hard limit
GEE_HARD_LIMIT_BYTES = 50331648  # 50.33MB


def get_max_pixels_for_city(city_name: str) -> int:
    """
    Get the maximum pixel dimensions for a city
    
    Args:
        city_name: Name of the city
        
    Returns:
        Maximum pixel dimension
    """
    # Try exact match first
    if city_name in CITY_MAX_PX:
        return CITY_MAX_PX[city_name]
    
    # Try case-insensitive match
    city_lower = city_name.lower()
    for name, max_px in CITY_MAX_PX.items():
        if name.lower() == city_lower:
            return max_px
    
    # Try partial match (for cities with suffixes like "Hyderabad, India")
    for name, max_px in CITY_MAX_PX.items():
        if name.lower() in city_lower or city_lower in name.lower():
            return max_px
    
    # Return default
    return DEFAULT_MAX_PX


def cap_dimensions(width: int, height: int, city_name: str) -> tuple:
    """
    Cap dimensions based on city-specific limits
    
    Args:
        width: Requested width in pixels
        height: Requested height in pixels
        city_name: Name of the city
        
    Returns:
        Tuple of (capped_width, capped_height)
    """
    max_px = get_max_pixels_for_city(city_name)
    
    # If both dimensions are already within limits, return as-is
    if width <= max_px and height <= max_px:
        return width, height
    
    # Scale down to fit within limits while preserving aspect ratio
    scale = min(max_px / width, max_px / height, 1.0)
    capped_width = int(width * scale)
    capped_height = int(height * scale)
    
    return capped_width, capped_height