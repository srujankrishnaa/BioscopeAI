
#!/usr/bin/env python3
"""
Indian States and Union Territories configuration
Contains a comprehensive list of all Indian states and union territories for testing
"""

# List of all Indian states and union territories
INDIAN_STATES = [
    # States
    "Andhra Pradesh",
    "Arunachal Pradesh",
    "Assam",
    "Bihar",
   

    "Chhattisgarh",
    "Goa",
    "Gujarat",
    "Haryana",
    "Himachal Pradesh",
    "Jharkhand",
    "Karnataka",
    "Kerala",
    "Madhya Pradesh",
    "Maharashtra",
    "Manipur",
    "Meghalaya",
    "Mizoram",
    "Nagaland",
    "Odisha",
    "Punjab",
    "Rajasthan",
    "Sikkim",
    "Tamil Nadu",
    "Telangana",
    "Tripura",
    "Uttar Pradesh",
    "Uttarakhand",
    "West Bengal",
    
    # Union Territories
    "Andaman and Nicobar Islands",
    "Chandigarh",
    "Dadra and Nagar Haveli and Daman and Diu",
    "Delhi",
    "Jammu and Kashmir",
    "Ladakh",
    "Lakshadweep",
    "Puducherry"
]

# State-wise maximum pixel dimensions for satellite imagery
# States are generally larger than cities, so we need more conservative limits
STATE_LIMITS = {
    # Large states - most conservative limits
    "Rajasthan": 1024,
    "Madhya Pradesh": 1024,
    "Maharashtra": 1024,
    "Uttar Pradesh": 1024,
    "Gujarat": 1024,
    "Karnataka": 1024,
    "Andhra Pradesh": 1024,
    "Odisha": 1024,
    "Chhattisgarh": 1024,
    "Tamil Nadu": 1024,
    "Telangana": 1024,
    "Bihar": 1024,
    "West Bengal": 1024,
    "Jammu and Kashmir": 1024,
    
    # Medium-sized states
    "Arunachal Pradesh": 1280,
    "Assam": 1280,
    "Haryana": 1280,
    "Himachal Pradesh": 1280,
    "Jharkhand": 1280,
    "Kerala": 1280,
    "Punjab": 1280,
    "Uttarakhand": 1280,
    
    # Smaller states
    "Goa": 1536,
    "Manipur": 1536,
    "Meghalaya": 1536,
    "Mizoram": 1536,
    "Nagaland": 1536,
    "Sikkim": 1536,
    "Tripura": 1536,
    
    # Union Territories
    "Delhi": 1280,
    "Puducherry": 1536,
    "Chandigarh": 1536,
    "Andaman and Nicobar Islands": 1280,
    "Dadra and Nagar Haveli and Daman and Diu": 1536,
    "Lakshadweep": 1536,
    "Ladakh": 1280
}

# Default limit for states not in the above dictionary
DEFAULT_STATE_LIMIT = 1024