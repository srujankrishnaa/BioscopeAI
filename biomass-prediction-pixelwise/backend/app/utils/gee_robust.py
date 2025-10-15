"""
Robust Google Earth Engine Integration
Implements comprehensive error handling, retry logic, caching, and monitoring
"""

import ee
import json
import time
import hashlib
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional, Any, List
from dataclasses import dataclass, asdict
from enum import Enum
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import numpy as np
from pathlib import Path
import io
from PIL import Image
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

import os
from contextlib import asynccontextmanager
try:
    import aiofiles
    AIOFILES_AVAILABLE = True
except ImportError:
    AIOFILES_AVAILABLE = False
    aiofiles = None
import tempfile

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GEEErrorType(Enum):
    """Classification of GEE errors for different handling strategies"""
    QUOTA_EXCEEDED = "quota_exceeded"
    TIMEOUT = "timeout"
    INVALID_REQUEST = "invalid_request"
    AUTHENTICATION = "authentication"
    SERVER_ERROR = "server_error"
    NETWORK_ERROR = "network_error"
    UNKNOWN = "unknown"

class ExportStrategy(Enum):
    """Different export strategies based on data size and requirements"""
    THUMBNAIL = "thumbnail"
    DRIVE_EXPORT = "drive_export"
    GCS_EXPORT = "gcs_export"
    TILED_EXPORT = "tiled_export"

@dataclass
class GEERequest:
    """Structured GEE request with metadata"""
    bbox: Tuple[float, float, float, float]
    scale: int
    dimensions: Optional[int]
    collection: str
    date_range: Tuple[str, str]
    bands: List[str]
    estimated_pixels: int
    estimated_bytes: int
    strategy: ExportStrategy
    request_id: str
    timestamp: datetime

@dataclass
class GEEResponse:
    """Structured GEE response with metadata"""
    success: bool
    data: Optional[Any]
    error_type: Optional[GEEErrorType]
    error_message: Optional[str]
    response_time_ms: int
    bytes_transferred: int
    cache_hit: bool
    retry_count: int
    strategy_used: ExportStrategy

class CircuitBreaker:
    """Circuit breaker for GEE API calls"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def can_execute(self) -> bool:
        """Check if request can be executed"""
        if self.state == "CLOSED":
            return True
        elif self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
                return True
            return False
        else:  # HALF_OPEN
            return True
    
    def record_success(self):
        """Record successful request"""
        self.failure_count = 0
        self.state = "CLOSED"
    
    def record_failure(self):
        """Record failed request"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"

class RateLimiter:
    """Rate limiter for GEE API calls"""
    
    def __init__(self, max_requests_per_second: int = 10):
        self.max_requests_per_second = max_requests_per_second
        self.requests = []
    
    async def acquire(self):
        """Acquire rate limit token"""
        now = time.time()
        # Remove old requests
        self.requests = [req_time for req_time in self.requests if now - req_time < 1.0]
        
        if len(self.requests) >= self.max_requests_per_second:
            sleep_time = 1.0 - (now - self.requests[0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        
        self.requests.append(now)

class GEECache:
    """Caching layer for GEE responses"""
    
    def __init__(self, redis_url: Optional[str] = None, ttl_hours: int = 24):
        self.ttl_seconds = ttl_hours * 3600
        self.redis_client = None
        
        if redis_url:
            try:
                import redis
                self.redis_client = redis.from_url(redis_url)
                logger.info("Redis cache initialized")
            except Exception as e:
                logger.warning(f"Redis cache initialization failed: {e}")
        
        # Fallback to file-based cache
        self.cache_dir = Path("./cache/gee")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _generate_cache_key(self, request: GEERequest) -> str:
        """Generate cache key from request parameters"""
        key_data = {
            'bbox': request.bbox,
            'scale': request.scale,
            'dimensions': request.dimensions,
            'collection': request.collection,
            'date_range': request.date_range,
            'bands': request.bands
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    async def get(self, request: GEERequest) -> Optional[Any]:
        """Get cached response"""
        cache_key = self._generate_cache_key(request)
        
        if self.redis_client and REDIS_AVAILABLE:
            try:
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    return json.loads(cached_data)
            except Exception as e:
                logger.warning(f"Redis cache get failed: {e}")
        
        # File-based cache fallback
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                cache_time = cache_file.stat().st_mtime
                if time.time() - cache_time < self.ttl_seconds:
                    if AIOFILES_AVAILABLE:
                        async with aiofiles.open(cache_file, 'r') as f:
                            content = await f.read()
                            return json.loads(content)
                    else:
                        # Synchronous fallback
                        with open(cache_file, 'r') as f:
                            content = f.read()
                            return json.loads(content)
                else:
                    cache_file.unlink()  # Remove expired cache
            except Exception as e:
                logger.warning(f"File cache get failed: {e}")
        
        return None
    
    async def set(self, request: GEERequest, data: Any):
        """Set cached response"""
        cache_key = self._generate_cache_key(request)
        serialized_data = json.dumps(data, default=str)
        
        if self.redis_client and REDIS_AVAILABLE:
            try:
                self.redis_client.setex(cache_key, self.ttl_seconds, serialized_data)
                return
            except Exception as e:
                logger.warning(f"Redis cache set failed: {e}")
        
        # File-based cache fallback
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            if AIOFILES_AVAILABLE:
                async with aiofiles.open(cache_file, 'w') as f:
                    await f.write(serialized_data)
            else:
                # Synchronous fallback
                with open(cache_file, 'w') as f:
                    f.write(serialized_data)
        except Exception as e:
            logger.warning(f"File cache set failed: {e}")

class RobustGEEClient:
    """Robust Google Earth Engine client with comprehensive error handling"""
    
    def __init__(self, 
                 service_account_key: Optional[str] = None,
                 redis_url: Optional[str] = None,
                 max_retries: int = 3,
                 base_delay: float = 1.0,
                 max_delay: float = 60.0,
                 rate_limit_rps: int = 10):
        
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        
        # Initialize components
        self.circuit_breaker = CircuitBreaker()
        self.rate_limiter = RateLimiter(rate_limit_rps)
        self.cache = GEECache(redis_url)
        
        # Initialize GEE
        self.initialized = False
        self._initialize_gee(service_account_key)
        
        # Request session with retry logic
        self.session = self._create_session()
        
        # Metrics
        self.metrics = {
            'requests_total': 0,
            'requests_successful': 0,
            'requests_failed': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'circuit_breaker_trips': 0,
            'rate_limit_delays': 0
        }
    
    def _initialize_gee(self, service_account_key: Optional[str]):
        """Initialize Google Earth Engine with proper error handling"""
        try:
            if service_account_key and os.path.exists(service_account_key):
                credentials = ee.ServiceAccountCredentials(
                    email=None,
                    key_file=service_account_key
                )
                ee.Initialize(credentials, project='ee-lanbprojectclassification')
                logger.info("✅ GEE initialized with service account")
            else:
                ee.Initialize(project='ee-lanbprojectclassification')
                logger.info("✅ GEE initialized with user authentication")
            
            self.initialized = True
            
        except Exception as e:
            logger.error(f"❌ GEE initialization failed: {e}")
            self.initialized = False
    
    def _create_session(self) -> requests.Session:
        """Create requests session with retry logic"""
        session = requests.Session()
        
        retry_strategy = Retry(
            total=self.max_retries,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST"],  # Updated from method_whitelist
            backoff_factor=1,
            raise_on_status=False
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    def _classify_error(self, error: Exception) -> GEEErrorType:
        """Classify error type for appropriate handling"""
        error_str = str(error).lower()
        
        if "quota" in error_str or "limit" in error_str:
            return GEEErrorType.QUOTA_EXCEEDED
        elif "timeout" in error_str or "timed out" in error_str:
            return GEEErrorType.TIMEOUT
        elif "invalid" in error_str or "bad request" in error_str:
            return GEEErrorType.INVALID_REQUEST
        elif "auth" in error_str or "permission" in error_str:
            return GEEErrorType.AUTHENTICATION
        elif "server error" in error_str or "internal error" in error_str:
            return GEEErrorType.SERVER_ERROR
        elif "network" in error_str or "connection" in error_str:
            return GEEErrorType.NETWORK_ERROR
        else:
            return GEEErrorType.UNKNOWN
    
    def _calculate_backoff_delay(self, attempt: int, error_type: GEEErrorType) -> float:
        """Calculate backoff delay with jitter"""
        base_delay = self.base_delay
        
        # Different base delays for different error types
        if error_type == GEEErrorType.QUOTA_EXCEEDED:
            base_delay = 30.0  # Longer delay for quota issues
        elif error_type == GEEErrorType.TIMEOUT:
            base_delay = 5.0   # Medium delay for timeouts
        
        # Exponential backoff with jitter
        delay = min(base_delay * (2 ** attempt), self.max_delay)
        jitter = np.random.uniform(0.1, 0.3) * delay
        
        return delay + jitter
    
    def _estimate_request_size(self, bbox: Tuple[float, float, float, float], 
                              scale: int, dimensions: Optional[int] = None) -> Tuple[int, int]:
        """Estimate pixels and bytes for a request"""
        min_lon, min_lat, max_lon, max_lat = bbox
        
        if dimensions:
            pixels = dimensions * dimensions
        else:
            # Calculate based on scale and bbox
            width_m = (max_lon - min_lon) * 111320  # Approximate meters per degree
            height_m = (max_lat - min_lat) * 111320
            
            width_pixels = int(width_m / scale)
            height_pixels = int(height_m / scale)
            pixels = width_pixels * height_pixels
        
        # Estimate bytes (3 bands * 1 byte per pixel + overhead)
        estimated_bytes = pixels * 3 * 1.2  # 20% overhead
        
        return pixels, int(estimated_bytes)
    
    def _select_strategy(self, pixels: int, bytes_estimate: int) -> ExportStrategy:
        """Select appropriate export strategy based on size"""
        # GEE limits
        THUMBNAIL_PIXEL_LIMIT = 1e7  # 10M pixels
        THUMBNAIL_BYTE_LIMIT = 32 * 1024 * 1024  # 32MB
        
        if pixels <= THUMBNAIL_PIXEL_LIMIT and bytes_estimate <= THUMBNAIL_BYTE_LIMIT:
            return ExportStrategy.THUMBNAIL
        elif pixels <= 1e8:  # 100M pixels
            return ExportStrategy.DRIVE_EXPORT
        else:
            return ExportStrategy.TILED_EXPORT
    
    async def _execute_with_retry(self, request: GEERequest) -> GEEResponse:
        """Execute request with comprehensive retry logic"""
        start_time = time.time()
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                # Check circuit breaker
                if not self.circuit_breaker.can_execute():
                    self.metrics['circuit_breaker_trips'] += 1
                    raise Exception("Circuit breaker is OPEN")
                
                # Apply rate limiting
                await self.rate_limiter.acquire()
                
                # Execute the actual request
                if request.strategy == ExportStrategy.THUMBNAIL:
                    result = await self._execute_thumbnail_request(request)
                elif request.strategy == ExportStrategy.DRIVE_EXPORT:
                    result = await self._execute_drive_export(request)
                elif request.strategy == ExportStrategy.TILED_EXPORT:
                    result = await self._execute_tiled_export(request)
                else:
                    raise ValueError(f"Unsupported strategy: {request.strategy}")
                
                # Success
                self.circuit_breaker.record_success()
                self.metrics['requests_successful'] += 1
                
                response_time = int((time.time() - start_time) * 1000)
                
                return GEEResponse(
                    success=True,
                    data=result,
                    error_type=None,
                    error_message=None,
                    response_time_ms=response_time,
                    bytes_transferred=len(str(result)) if result else 0,
                    cache_hit=False,
                    retry_count=attempt,
                    strategy_used=request.strategy
                )
                
            except Exception as e:
                last_error = e
                error_type = self._classify_error(e)
                
                logger.warning(f"Request attempt {attempt + 1} failed: {error_type.value} - {e}")
                
                # Record failure
                self.circuit_breaker.record_failure()
                
                # Don't retry certain error types
                if error_type in [GEEErrorType.INVALID_REQUEST, GEEErrorType.AUTHENTICATION]:
                    break
                
                # Calculate backoff delay
                if attempt < self.max_retries:
                    delay = self._calculate_backoff_delay(attempt, error_type)
                    logger.info(f"Retrying in {delay:.2f} seconds...")
                    await asyncio.sleep(delay)
        
        # All retries failed
        self.metrics['requests_failed'] += 1
        response_time = int((time.time() - start_time) * 1000)
        error_type = self._classify_error(last_error) if last_error else GEEErrorType.UNKNOWN
        
        return GEEResponse(
            success=False,
            data=None,
            error_type=error_type,
            error_message=str(last_error),
            response_time_ms=response_time,
            bytes_transferred=0,
            cache_hit=False,
            retry_count=self.max_retries,
            strategy_used=request.strategy
        )
    
    async def _execute_thumbnail_request(self, request: GEERequest) -> Dict:
        """Execute thumbnail request"""
        if not self.initialized:
            raise Exception("GEE not initialized")
        
        # Create geometry
        geometry = ee.Geometry.Rectangle(list(request.bbox))
        
        # Get image collection
        if request.collection == "COPERNICUS/S2_SR":
            collection = ee.ImageCollection(request.collection) \
                .filterBounds(geometry) \
                .filterDate(request.date_range[0], request.date_range[1]) \
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
                .select(request.bands)
            
            image = collection.median().clip(geometry)
        else:
            raise ValueError(f"Unsupported collection: {request.collection}")
        
        # Get thumbnail URL
        thumbnail_params = {
            'region': geometry,
            'dimensions': request.dimensions or 1024,
            'format': 'png'
        }
        
        if request.bands == ['B4', 'B3', 'B2']:  # RGB
            thumbnail_params.update({
                'min': 0,
                'max': 3000,
                'bands': request.bands
            })
        
        thumbnail_url = image.getThumbURL(thumbnail_params)
        
        # Download thumbnail
        response = self.session.get(thumbnail_url, timeout=30)
        response.raise_for_status()
        
        # Convert to numpy array
        image_data = Image.open(io.BytesIO(response.content))
        image_array = np.array(image_data)
        
        return {
            'image_array': image_array,
            'thumbnail_url': thumbnail_url,
            'method': 'thumbnail'
        }
    
    async def _execute_drive_export(self, request: GEERequest) -> Dict:
        """Execute Drive export request"""
        # TODO: Implement Drive export with polling and download
        raise NotImplementedError("Drive export not yet implemented")
    
    async def _execute_tiled_export(self, request: GEERequest) -> Dict:
        """Execute tiled export request"""
        # TODO: Implement tiled export with stitching
        raise NotImplementedError("Tiled export not yet implemented")
    
    async def fetch_satellite_image(self, 
                                  bbox: Tuple[float, float, float, float],
                                  scale: int = 10,
                                  dimensions: Optional[int] = None,
                                  collection: str = "COPERNICUS/S2_SR",
                                  bands: List[str] = ['B4', 'B3', 'B2'],
                                  date_range: Optional[Tuple[str, str]] = None) -> GEEResponse:
        """
        Fetch satellite image with comprehensive error handling and caching
        
        Args:
            bbox: Bounding box (min_lon, min_lat, max_lon, max_lat)
            scale: Scale in meters per pixel
            dimensions: Fixed dimensions (optional)
            collection: GEE collection name
            bands: List of bands to fetch
            date_range: Date range tuple (start, end)
        
        Returns:
            GEEResponse with image data or error information
        """
        self.metrics['requests_total'] += 1
        
        # Set default date range
        if not date_range:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            date_range = (start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        
        # Estimate request size
        pixels, bytes_estimate = self._estimate_request_size(bbox, scale, dimensions)
        
        # Select strategy
        strategy = self._select_strategy(pixels, bytes_estimate)
        
        # Create request object
        request = GEERequest(
            bbox=bbox,
            scale=scale,
            dimensions=dimensions,
            collection=collection,
            date_range=date_range,
            bands=bands,
            estimated_pixels=pixels,
            estimated_bytes=bytes_estimate,
            strategy=strategy,
            request_id=hashlib.md5(f"{bbox}{scale}{dimensions}{collection}{bands}{date_range}".encode()).hexdigest()[:8],
            timestamp=datetime.now()
        )
        
        # Check cache first
        cached_result = await self.cache.get(request)
        if cached_result:
            self.metrics['cache_hits'] += 1
            return GEEResponse(
                success=True,
                data=cached_result,
                error_type=None,
                error_message=None,
                response_time_ms=0,
                bytes_transferred=0,
                cache_hit=True,
                retry_count=0,
                strategy_used=strategy
            )
        
        self.metrics['cache_misses'] += 1
        
        # Execute request with retry logic
        response = await self._execute_with_retry(request)
        
        # Cache successful responses
        if response.success and response.data:
            await self.cache.set(request, response.data)
        
        # Log structured request/response
        self._log_request_response(request, response)
        
        return response
    
    def _log_request_response(self, request: GEERequest, response: GEEResponse):
        """Log structured request/response data"""
        log_data = {
            'request_id': request.request_id,
            'bbox': request.bbox,
            'strategy': request.strategy.value,
            'estimated_pixels': request.estimated_pixels,
            'estimated_bytes': request.estimated_bytes,
            'success': response.success,
            'response_time_ms': response.response_time_ms,
            'cache_hit': response.cache_hit,
            'retry_count': response.retry_count,
            'error_type': response.error_type.value if response.error_type else None,
            'timestamp': request.timestamp.isoformat()
        }
        
        if response.success:
            logger.info(f"GEE_REQUEST_SUCCESS: {json.dumps(log_data)}")
        else:
            logger.error(f"GEE_REQUEST_FAILED: {json.dumps(log_data)}")
    
    def get_metrics(self) -> Dict:
        """Get current metrics"""
        return {
            **self.metrics,
            'circuit_breaker_state': self.circuit_breaker.state,
            'cache_size': len(list(self.cache.cache_dir.glob('*.json'))) if self.cache.cache_dir.exists() else 0
        }
    
    async def health_check(self) -> Dict:
        """Perform health check"""
        health_status = {
            'gee_initialized': self.initialized,
            'circuit_breaker_state': self.circuit_breaker.state,
            'metrics': self.get_metrics(),
            'timestamp': datetime.now().isoformat()
        }
        
        # Test basic GEE functionality
        if self.initialized:
            try:
                # Simple test request
                test_bbox = (77.5, 12.9, 77.6, 13.0)  # Small Bangalore area
                test_response = await self.fetch_satellite_image(
                    bbox=test_bbox,
                    dimensions=64,  # Very small for quick test
                    date_range=('2023-01-01', '2023-12-31')
                )
                health_status['test_request_success'] = test_response.success
                health_status['test_response_time_ms'] = test_response.response_time_ms
            except Exception as e:
                health_status['test_request_success'] = False
                health_status['test_error'] = str(e)
        
        return health_status

# Global instance
robust_gee_client = None

def get_robust_gee_client() -> RobustGEEClient:
    """Get global robust GEE client instance"""
    global robust_gee_client
    
    if robust_gee_client is None:
        robust_gee_client = RobustGEEClient(
            service_account_key=os.getenv('GEE_SERVICE_ACCOUNT_KEY'),
            redis_url=os.getenv('REDIS_URL'),
            max_retries=int(os.getenv('GEE_MAX_RETRIES', '3')),
            rate_limit_rps=int(os.getenv('GEE_RATE_LIMIT_RPS', '10'))
        )
    
    return robust_gee_client