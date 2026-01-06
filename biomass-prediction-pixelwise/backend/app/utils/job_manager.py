"""
Job Management System
Tracks async heatmap generation jobs
"""

import uuid
from datetime import datetime
from typing import Dict, Optional
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class Job:
    job_id: str
    status: str  # 'processing', 'completed', 'failed'
    city: str
    region_name: str
    created_at: str
    completed_at: Optional[str] = None
    heatmap_url: Optional[str] = None
    error_message: Optional[str] = None
    stats: Optional[Dict] = None

class JobManager:
    """In-memory job tracking (use Redis for production)"""
    
    def __init__(self):
        self.jobs: Dict[str, Job] = {}
    
    def create_job(self, city: str, region_name: str) -> str:
        """Create new job and return job ID"""
        job_id = f"{city.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        
        job = Job(
            job_id=job_id,
            status='processing',
            city=city,
            region_name=region_name,
            created_at=datetime.now().isoformat()
        )
        
        self.jobs[job_id] = job
        logger.info(f"📝 Created job: {job_id}")
        return job_id
    
    def update_job(self, job_id: str, **kwargs):
        """Update job with new data"""
        if job_id in self.jobs:
            job = self.jobs[job_id]
            for key, value in kwargs.items():
                if hasattr(job, key):
                    setattr(job, key, value)
            logger.info(f"📝 Updated job {job_id}: {kwargs.get('status', 'updated')}")
    
    def get_job(self, job_id: str) -> Optional[Dict]:
        """Get job status"""
        job = self.jobs.get(job_id)
        if job:
            return asdict(job)
        return None
    
    def mark_completed(self, job_id: str, heatmap_url: str, stats: Dict):
        """Mark job as completed"""
        self.update_job(
            job_id,
            status='completed',
            completed_at=datetime.now().isoformat(),
            heatmap_url=heatmap_url,
            stats=stats
        )
    
    def mark_failed(self, job_id: str, error_message: str):
        """Mark job as failed"""
        self.update_job(
            job_id,
            status='failed',
            completed_at=datetime.now().isoformat(),
            error_message=error_message
        )

# Global instance
_job_manager = None

def get_job_manager() -> JobManager:
    """Get global job manager instance"""
    global _job_manager
    if _job_manager is None:
        _job_manager = JobManager()
    return _job_manager

