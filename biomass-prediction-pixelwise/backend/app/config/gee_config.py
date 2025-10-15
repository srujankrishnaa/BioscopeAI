"""
Google Earth Engine Configuration
Centralized configuration for robust GEE integration
"""

import os
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class GEEConfig:
    """Configuration for Google Earth Engine integration"""
    
    # Authentication
    service_account_key: str = os.getenv('GEE_SERVICE_ACCOUNT_KEY', '')
    project_id: str = os.getenv('GEE_PROJECT_ID', 'ee-lanbprojectclassification')
    
    # Retry and backoff settings
    max_retries: int = int(os.getenv('GEE_MAX_RETRIES', '3'))
    base_delay: float = float(os.getenv('GEE_BASE_DELAY', '1.0'))
    max_delay: float = float(os.getenv('GEE_MAX_DELAY', '60.0'))
    
    # Rate limiting
    rate_limit_rps: int = int(os.getenv('GEE_RATE_LIMIT_RPS', '10'))
    
    # Circuit breaker
    circuit_breaker_failure_threshold: int = int(os.getenv('GEE_CB_FAILURE_THRESHOLD', '5'))
    circuit_breaker_recovery_timeout: int = int(os.getenv('GEE_CB_RECOVERY_TIMEOUT', '60'))
    
    # Caching
    redis_url: str = os.getenv('REDIS_URL', '')
    cache_ttl_hours: int = int(os.getenv('GEE_CACHE_TTL_HOURS', '24'))
    
    # Request limits
    thumbnail_pixel_limit: int = int(os.getenv('GEE_THUMBNAIL_PIXEL_LIMIT', '10000000'))  # 10M pixels
    thumbnail_byte_limit: int = int(os.getenv('GEE_THUMBNAIL_BYTE_LIMIT', '33554432'))   # 32MB
    
    # Quality presets
    quality_presets: Dict[str, Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.quality_presets is None:
            self.quality_presets = {
                'low': {
                    'scale': 60,
                    'dimensions': 512,
                    'timeout': 30,
                    'description': 'Low quality for quick previews'
                },
                'medium': {
                    'scale': 30,
                    'dimensions': 1024,
                    'timeout': 60,
                    'description': 'Medium quality for general use'
                },
                'high': {
                    'scale': 10,
                    'dimensions': 2048,
                    'timeout': 120,
                    'description': 'High quality for detailed analysis'
                },
                'ultra': {
                    'scale': 5,
                    'dimensions': 4096,
                    'timeout': 300,
                    'description': 'Ultra high quality for research'
                }
            }
    
    def get_quality_config(self, quality: str) -> Dict[str, Any]:
        """Get configuration for specified quality level"""
        return self.quality_presets.get(quality, self.quality_presets['medium'])
    
    def validate(self) -> Dict[str, str]:
        """Validate configuration and return any issues"""
        issues = {}
        
        # Check authentication
        if self.service_account_key and not os.path.exists(self.service_account_key):
            issues['service_account_key'] = f"Service account key file not found: {self.service_account_key}"
        
        # Check numeric ranges
        if self.max_retries < 0 or self.max_retries > 10:
            issues['max_retries'] = "max_retries should be between 0 and 10"
        
        if self.base_delay < 0.1 or self.base_delay > 10:
            issues['base_delay'] = "base_delay should be between 0.1 and 10 seconds"
        
        if self.rate_limit_rps < 1 or self.rate_limit_rps > 100:
            issues['rate_limit_rps'] = "rate_limit_rps should be between 1 and 100"
        
        if self.cache_ttl_hours < 1 or self.cache_ttl_hours > 168:  # 1 week max
            issues['cache_ttl_hours'] = "cache_ttl_hours should be between 1 and 168 (1 week)"
        
        return issues
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'service_account_key': '***' if self.service_account_key else None,
            'project_id': self.project_id,
            'max_retries': self.max_retries,
            'base_delay': self.base_delay,
            'max_delay': self.max_delay,
            'rate_limit_rps': self.rate_limit_rps,
            'circuit_breaker_failure_threshold': self.circuit_breaker_failure_threshold,
            'circuit_breaker_recovery_timeout': self.circuit_breaker_recovery_timeout,
            'redis_url': '***' if self.redis_url else None,
            'cache_ttl_hours': self.cache_ttl_hours,
            'thumbnail_pixel_limit': self.thumbnail_pixel_limit,
            'thumbnail_byte_limit': self.thumbnail_byte_limit,
            'quality_presets': list(self.quality_presets.keys())
        }

# Global configuration instance
_gee_config = None

def get_gee_config() -> GEEConfig:
    """Get global GEE configuration instance"""
    global _gee_config
    
    if _gee_config is None:
        _gee_config = GEEConfig()
    
    return _gee_config

def reload_gee_config() -> GEEConfig:
    """Reload GEE configuration from environment"""
    global _gee_config
    _gee_config = GEEConfig()
    return _gee_config