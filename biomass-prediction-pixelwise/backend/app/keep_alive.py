"""
Keep-Alive Self-Ping Module for Render Free Tier

Render's free tier spins down services after 15 minutes of inactivity.
This module provides a background self-ping mechanism that sends an HTTP 
request to the service's own /health endpoint every 14 minutes to prevent 
the service from sleeping.

This is a complementary solution to UptimeRobot — use BOTH for maximum reliability:
  - UptimeRobot: External monitoring (pings from outside)
  - Self-ping: Internal fallback (pings from within the service itself)

Usage:
    In your FastAPI app's lifespan:
    
        from app.keep_alive import start_keep_alive, stop_keep_alive

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            start_keep_alive()
            yield
            stop_keep_alive()
"""

import os
import asyncio
import logging
import httpx
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Configuration
KEEP_ALIVE_INTERVAL = int(os.getenv("KEEP_ALIVE_INTERVAL_SECONDS", "840"))  # 14 minutes
KEEP_ALIVE_ENABLED = os.getenv("KEEP_ALIVE_ENABLED", "true").lower() == "true"

# Module-level state
_keep_alive_task: asyncio.Task | None = None
_ping_count: int = 0
_last_ping_time: str | None = None
_last_ping_status: str = "never"


async def _self_ping_loop():
    """
    Continuously pings the service's own /health endpoint to prevent
    Render free-tier cold starts.
    """
    global _ping_count, _last_ping_time, _last_ping_status

    # Determine the service URL to ping
    # On Render, RENDER_EXTERNAL_URL is auto-set. Fallback to manual config.
    service_url = (
        os.getenv("RENDER_EXTERNAL_URL")
        or os.getenv("BACKEND_EXTERNAL_URL")
        or f"http://localhost:{os.getenv('PORT', '8000')}"
    )
    ping_url = f"{service_url}/health"

    logger.info(
        f"🏓 Keep-alive self-ping started | URL: {ping_url} | "
        f"Interval: {KEEP_ALIVE_INTERVAL}s ({KEEP_ALIVE_INTERVAL // 60}min)"
    )

    # Wait a bit before the first ping to let the server fully start
    await asyncio.sleep(30)

    async with httpx.AsyncClient(timeout=30.0) as client:
        while True:
            try:
                response = await client.get(ping_url)
                _ping_count += 1
                _last_ping_time = datetime.now(timezone.utc).isoformat()
                _last_ping_status = f"{response.status_code} OK" if response.status_code == 200 else f"{response.status_code} ERROR"

                logger.info(
                    f"🏓 Keep-alive ping #{_ping_count} → {response.status_code} | "
                    f"Next ping in {KEEP_ALIVE_INTERVAL}s"
                )
            except Exception as e:
                _last_ping_status = f"error: {str(e)[:100]}"
                logger.warning(f"🏓 Keep-alive ping failed: {e}")

            await asyncio.sleep(KEEP_ALIVE_INTERVAL)


def start_keep_alive():
    """Start the background keep-alive self-ping task."""
    global _keep_alive_task

    if not KEEP_ALIVE_ENABLED:
        logger.info("🏓 Keep-alive is DISABLED (set KEEP_ALIVE_ENABLED=true to enable)")
        return

    if _keep_alive_task is not None and not _keep_alive_task.done():
        logger.warning("🏓 Keep-alive task already running, skipping restart")
        return

    loop = asyncio.get_event_loop()
    _keep_alive_task = loop.create_task(_self_ping_loop())
    logger.info("🏓 Keep-alive background task scheduled")


def stop_keep_alive():
    """Cancel the background keep-alive task."""
    global _keep_alive_task

    if _keep_alive_task is not None and not _keep_alive_task.done():
        _keep_alive_task.cancel()
        logger.info("🏓 Keep-alive task cancelled")
    _keep_alive_task = None


def get_keep_alive_status() -> dict:
    """Get current keep-alive status for monitoring endpoints."""
    return {
        "enabled": KEEP_ALIVE_ENABLED,
        "interval_seconds": KEEP_ALIVE_INTERVAL,
        "total_pings": _ping_count,
        "last_ping_time": _last_ping_time,
        "last_ping_status": _last_ping_status,
        "task_running": _keep_alive_task is not None and not _keep_alive_task.done() if _keep_alive_task else False,
    }
