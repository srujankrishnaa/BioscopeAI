"""
AWS S3 Storage Integration
Handles uploading and managing heatmap images
"""

import boto3
import os
import logging
from pathlib import Path
from typing import Optional
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

class S3Storage:
    """Handles all S3 operations for heatmap storage"""
    
    def __init__(self):
        self.bucket_name = os.getenv('AWS_S3_BUCKET', 'biomass-heatmap')
        self.region = os.getenv('AWS_REGION', 'ap-south-1')
        
        # Initialize S3 client
        try:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                region_name=self.region
            )
            logger.info(f"✅ S3 client initialized for bucket: {self.bucket_name}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize S3 client: {e}")
            self.s3_client = None
    
    def upload_heatmap(self, local_file_path: str, s3_key: str) -> Optional[str]:
        """
        Upload heatmap image to S3
        
        Args:
            local_file_path: Path to local file
            s3_key: S3 object key (e.g., 'heatmaps/pune_20260106.png')
        
        Returns:
            Public URL of uploaded file, or None if failed
        """
        if not self.s3_client:
            logger.error("S3 client not initialized")
            return None
        
        try:
            # Upload file with public-read ACL
            self.s3_client.upload_file(
                local_file_path,
                self.bucket_name,
                s3_key,
                ExtraArgs={
                    'ContentType': 'image/png',
                    'CacheControl': 'public, max-age=31536000',  # 1 year cache
                }
            )
            
            # Construct public URL
            public_url = f"https://{self.bucket_name}.s3.{self.region}.amazonaws.com/{s3_key}"
            
            logger.info(f"✅ Uploaded to S3: {public_url}")
            return public_url
            
        except ClientError as e:
            logger.error(f"❌ S3 upload failed: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Unexpected error uploading to S3: {e}")
            return None
    
    def delete_heatmap(self, s3_key: str) -> bool:
        """Delete heatmap from S3"""
        if not self.s3_client:
            return False
        
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_key)
            logger.info(f"🗑️ Deleted from S3: {s3_key}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to delete from S3: {e}")
            return False

# Global instance
_s3_storage = None

def get_s3_storage() -> S3Storage:
    """Get global S3 storage instance"""
    global _s3_storage
    if _s3_storage is None:
        _s3_storage = S3Storage()
    return _s3_storage

