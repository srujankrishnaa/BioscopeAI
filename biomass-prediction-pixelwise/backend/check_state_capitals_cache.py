#!/usr/bin/env python3

"""
Check State Capitals Cache Status
Shows which state capitals are already cached and which need to be downloaded
"""

import json
from pathlib import Path

def check_cache_status():
    """Check which state capitals are cached vs remaining"""
    
    # 28 Indian State Capitals + 8 Union Territory Capitals
    state_capitals = [
        # 28 State Capitals
        'Mumbai',           # Maharashtra
        'Bangalore',        # Karnataka  
        'Chennai',          # Tamil Nadu
        'Hyderabad',        # Telangana
        'Kolkata',          # West Bengal
        'Ahmedabad',        # Gujarat (commercial capital)
        'Gandhinagar',      # Gujarat (official capital)
        'Jaipur',           # Rajasthan
        'Lucknow',          # Uttar Pradesh
        'Bhopal',           # Madhya Pradesh
        'Patna',            # Bihar
        'Thiruvananthapuram', # Kerala
        'Bhubaneswar',      # Odisha
        'Ranchi',           # Jharkhand
        'Raipur',           # Chhattisgarh
        'Panaji',           # Goa
        'Shimla',           # Himachal Pradesh
        'Srinagar',         # Jammu & Kashmir (summer capital)
        'Jammu',            # Jammu & Kashmir (winter capital)
        'Guwahati',         # Assam (largest city)
        'Dispur',           # Assam (official capital)
        'Agartala',         # Tripura
        'Aizawl',           # Mizoram
        'Imphal',           # Manipur
        'Kohima',           # Nagaland
        'Itanagar',         # Arunachal Pradesh
        'Gangtok',          # Sikkim
        'Shillong',         # Meghalaya
        
        # 8 Union Territory Capitals
        'Delhi',            # Delhi
        'Chandigarh',       # Chandigarh & Punjab & Haryana
        'Puducherry',       # Puducherry
        'Port Blair',       # Andaman & Nicobar
        'Kavaratti',        # Lakshadweep
        'Daman',            # Daman & Diu
        'Silvassa',         # Dadra & Nagar Haveli
        'Ladakh'            # Ladakh (Leh)
    ]
    
    # Load existing cache metadata
    cache_metadata_file = Path("./outputs/region_cache/cache_metadata.json")
    cached_cities = []
    
    if cache_metadata_file.exists():
        try:
            with open(cache_metadata_file, 'r') as f:
                cache_data = json.load(f)
                cached_cities = list(cache_data.keys())
        except Exception as e:
            print(f"Error reading cache metadata: {e}")
    
    # Find already cached vs remaining
    already_cached = []
    remaining_cities = []
    
    for city in state_capitals:
        if city in cached_cities:
            already_cached.append(city)
        else:
            remaining_cities.append(city)
    
    # Print status
    print("=" * 80)
    print("🏛️ INDIAN STATE CAPITALS CACHE STATUS")
    print("=" * 80)
    print(f"Total State Capitals: {len(state_capitals)}")
    print(f"Already Cached: {len(already_cached)}")
    print(f"Remaining to Download: {len(remaining_cities)}")
    print("=" * 80)
    
    if already_cached:
        print(f"\n✅ ALREADY CACHED ({len(already_cached)} cities):")
        for i, city in enumerate(already_cached, 1):
            print(f"  {i:2d}. {city}")
    
    if remaining_cities:
        print(f"\n📥 REMAINING TO DOWNLOAD ({len(remaining_cities)} cities):")
        for i, city in enumerate(remaining_cities, 1):
            print(f"  {i:2d}. {city}")
        
        print(f"\n🎯 NEXT STEPS:")
        print(f"Run: python generate_city_cache.py")
        print(f"This will download satellite images for the {len(remaining_cities)} remaining state capitals.")
    else:
        print(f"\n🎉 ALL STATE CAPITALS ARE CACHED!")
        print(f"No additional downloads needed.")
    
    print("=" * 80)
    
    return already_cached, remaining_cities

if __name__ == "__main__":
    check_cache_status()