"""
Monitoring and Metrics API for GEE Integration
Provides health checks, metrics, and operational insights
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import logging
from datetime import datetime
import json
try:
    from app.utils.gee_robust import get_robust_gee_client
    GEE_ROBUST_AVAILABLE = True
except ImportError:
    GEE_ROBUST_AVAILABLE = False
    get_robust_gee_client = None

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/health")
async def health_check():
    """Comprehensive health check for GEE integration"""
    try:
        if not GEE_ROBUST_AVAILABLE:
            return {
                "status": "degraded",
                "timestamp": datetime.now().isoformat(),
                "error": "Robust GEE client not available - missing dependencies",
                "details": {
                    "gee_initialized": False,
                    "robust_client_available": False
                }
            }
        
        client = get_robust_gee_client()
        health_data = await client.health_check()
        
        # Determine overall health status
        overall_status = "healthy"
        if not health_data.get('gee_initialized', False):
            overall_status = "degraded"
        elif health_data.get('circuit_breaker_state') == "OPEN":
            overall_status = "degraded"
        elif not health_data.get('test_request_success', True):
            overall_status = "unhealthy"
        
        return {
            "status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "details": health_data
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

@router.get("/metrics")
async def get_metrics():
    """Get detailed metrics for GEE operations"""
    try:
        if not GEE_ROBUST_AVAILABLE:
            return {
                "timestamp": datetime.now().isoformat(),
                "error": "Robust GEE client not available - missing dependencies",
                "raw_metrics": {},
                "derived_metrics": {
                    "success_rate_percent": 0,
                    "cache_hit_rate_percent": 0,
                    "failure_rate_percent": 0
                }
            }
        
        client = get_robust_gee_client()
        metrics = client.get_metrics()
        
        # Calculate derived metrics
        total_requests = metrics.get('requests_total', 0)
        successful_requests = metrics.get('requests_successful', 0)
        failed_requests = metrics.get('requests_failed', 0)
        
        success_rate = (successful_requests / total_requests * 100) if total_requests > 0 else 0
        cache_hit_rate = (metrics.get('cache_hits', 0) / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "timestamp": datetime.now().isoformat(),
            "raw_metrics": metrics,
            "derived_metrics": {
                "success_rate_percent": round(success_rate, 2),
                "cache_hit_rate_percent": round(cache_hit_rate, 2),
                "failure_rate_percent": round(100 - success_rate, 2)
            },
            "status": {
                "circuit_breaker_state": metrics.get('circuit_breaker_state', 'UNKNOWN'),
                "cache_size": metrics.get('cache_size', 0)
            }
        }
        
    except Exception as e:
        logger.error(f"Metrics collection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to collect metrics: {str(e)}")

@router.get("/metrics/prometheus")
async def get_prometheus_metrics():
    """Get metrics in Prometheus format"""
    try:
        client = get_robust_gee_client()
        metrics = client.get_metrics()
        
        # Generate Prometheus format
        prometheus_metrics = []
        
        # Counter metrics
        prometheus_metrics.append(f"gee_requests_total {metrics.get('requests_total', 0)}")
        prometheus_metrics.append(f"gee_requests_successful_total {metrics.get('requests_successful', 0)}")
        prometheus_metrics.append(f"gee_requests_failed_total {metrics.get('requests_failed', 0)}")
        prometheus_metrics.append(f"gee_cache_hits_total {metrics.get('cache_hits', 0)}")
        prometheus_metrics.append(f"gee_cache_misses_total {metrics.get('cache_misses', 0)}")
        prometheus_metrics.append(f"gee_circuit_breaker_trips_total {metrics.get('circuit_breaker_trips', 0)}")
        prometheus_metrics.append(f"gee_rate_limit_delays_total {metrics.get('rate_limit_delays', 0)}")
        
        # Gauge metrics
        circuit_breaker_state = metrics.get('circuit_breaker_state', 'CLOSED')
        cb_state_value = {'CLOSED': 0, 'HALF_OPEN': 1, 'OPEN': 2}.get(circuit_breaker_state, -1)
        prometheus_metrics.append(f"gee_circuit_breaker_state {cb_state_value}")
        prometheus_metrics.append(f"gee_cache_size {metrics.get('cache_size', 0)}")
        
        return "\n".join(prometheus_metrics)
        
    except Exception as e:
        logger.error(f"Prometheus metrics collection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to collect Prometheus metrics: {str(e)}")

@router.post("/circuit-breaker/reset")
async def reset_circuit_breaker():
    """Reset circuit breaker (admin operation)"""
    try:
        client = get_robust_gee_client()
        client.circuit_breaker.failure_count = 0
        client.circuit_breaker.state = "CLOSED"
        client.circuit_breaker.last_failure_time = None
        
        logger.info("Circuit breaker manually reset")
        
        return {
            "status": "success",
            "message": "Circuit breaker reset to CLOSED state",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Circuit breaker reset failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reset circuit breaker: {str(e)}")

@router.get("/cache/stats")
async def get_cache_stats():
    """Get detailed cache statistics"""
    try:
        client = get_robust_gee_client()
        cache = client.cache
        
        stats = {
            "cache_type": "redis" if cache.redis_client else "file",
            "cache_directory": str(cache.cache_dir) if cache.cache_dir else None,
            "ttl_hours": cache.ttl_seconds / 3600,
            "timestamp": datetime.now().isoformat()
        }
        
        # File-based cache stats
        if cache.cache_dir and cache.cache_dir.exists():
            cache_files = list(cache.cache_dir.glob('*.json'))
            total_size = sum(f.stat().st_size for f in cache_files)
            
            stats.update({
                "file_count": len(cache_files),
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2)
            })
        
        # Redis cache stats
        if cache.redis_client:
            try:
                redis_info = cache.redis_client.info()
                stats.update({
                    "redis_connected": True,
                    "redis_memory_used": redis_info.get('used_memory_human'),
                    "redis_keys": cache.redis_client.dbsize()
                })
            except Exception as e:
                stats.update({
                    "redis_connected": False,
                    "redis_error": str(e)
                })
        
        return stats
        
    except Exception as e:
        logger.error(f"Cache stats collection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to collect cache stats: {str(e)}")

@router.delete("/cache/clear")
async def clear_cache():
    """Clear all cached data (admin operation)"""
    try:
        client = get_robust_gee_client()
        cache = client.cache
        
        cleared_items = 0
        
        # Clear Redis cache
        if cache.redis_client:
            try:
                cleared_items += cache.redis_client.dbsize()
                cache.redis_client.flushdb()
                logger.info("Redis cache cleared")
            except Exception as e:
                logger.warning(f"Redis cache clear failed: {e}")
        
        # Clear file cache
        if cache.cache_dir and cache.cache_dir.exists():
            cache_files = list(cache.cache_dir.glob('*.json'))
            for cache_file in cache_files:
                try:
                    cache_file.unlink()
                    cleared_items += 1
                except Exception as e:
                    logger.warning(f"Failed to delete cache file {cache_file}: {e}")
            
            logger.info(f"File cache cleared: {len(cache_files)} files")
        
        return {
            "status": "success",
            "message": f"Cache cleared: {cleared_items} items removed",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Cache clear failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")

@router.get("/config")
async def get_configuration():
    """Get current GEE client configuration"""
    try:
        client = get_robust_gee_client()
        
        config = {
            "gee_initialized": client.initialized,
            "max_retries": client.max_retries,
            "base_delay": client.base_delay,
            "max_delay": client.max_delay,
            "rate_limit_rps": client.rate_limiter.max_requests_per_second,
            "circuit_breaker": {
                "failure_threshold": client.circuit_breaker.failure_threshold,
                "recovery_timeout": client.circuit_breaker.recovery_timeout,
                "current_state": client.circuit_breaker.state
            },
            "cache": {
                "ttl_hours": client.cache.ttl_seconds / 3600,
                "redis_enabled": client.cache.redis_client is not None,
                "cache_directory": str(client.cache.cache_dir)
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return config
        
    except Exception as e:
        logger.error(f"Configuration retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get configuration: {str(e)}")

@router.get("/logs/recent")
async def get_recent_logs(limit: int = 100):
    """Get recent GEE operation logs"""
    try:
        # This would typically read from a log aggregation system
        # For now, return a placeholder response
        return {
            "message": "Log aggregation not implemented yet",
            "suggestion": "Check application logs or implement log aggregation service",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Log retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get logs: {str(e)}")

@router.post("/test/request")
async def test_gee_request(
    bbox: str = "77.5,12.9,77.6,13.0",  # Default: Small Bangalore area
    dimensions: int = 256
):
    """Test GEE request with specified parameters"""
    try:
        # Parse bbox
        bbox_parts = [float(x.strip()) for x in bbox.split(',')]
        if len(bbox_parts) != 4:
            raise ValueError("bbox must have 4 comma-separated values: min_lon,min_lat,max_lon,max_lat")
        
        bbox_tuple = tuple(bbox_parts)
        
        client = get_robust_gee_client()
        
        # Execute test request
        response = await client.fetch_satellite_image(
            bbox=bbox_tuple,
            dimensions=dimensions,
            date_range=('2023-01-01', '2023-12-31')
        )
        
        # Return response metadata (not the actual image data)
        return {
            "test_successful": response.success,
            "response_time_ms": response.response_time_ms,
            "strategy_used": response.strategy_used.value,
            "cache_hit": response.cache_hit,
            "retry_count": response.retry_count,
            "error_type": response.error_type.value if response.error_type else None,
            "error_message": response.error_message,
            "bbox_tested": bbox_tuple,
            "dimensions_tested": dimensions,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Test request failed: {e}")
        raise HTTPException(status_code=500, detail=f"Test request failed: {str(e)}")