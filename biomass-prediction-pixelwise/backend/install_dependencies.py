#!/usr/bin/env python3

"""
Install Missing Dependencies for Robust GEE Integration
Installs required packages for production deployment
"""

import subprocess
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def install_package(package_name):
    """Install a package using pip"""
    try:
        logger.info(f"Installing {package_name}...")
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", package_name
        ], capture_output=True, text=True, check=True)
        
        logger.info(f"✅ {package_name} installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to install {package_name}: {e}")
        logger.error(f"Error output: {e.stderr}")
        return False

def main():
    """Install all missing dependencies"""
    logger.info("🚀 Installing Missing Dependencies for Robust GEE Integration")
    logger.info("=" * 80)
    
    # Required packages
    packages = [
        "earthengine-api",
        "redis", 
        "Pillow",
        "aiofiles"
    ]
    
    success_count = 0
    failed_packages = []
    
    for package in packages:
        if install_package(package):
            success_count += 1
        else:
            failed_packages.append(package)
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("📊 INSTALLATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total packages: {len(packages)}")
    logger.info(f"Successfully installed: {success_count}")
    logger.info(f"Failed: {len(failed_packages)}")
    
    if failed_packages:
        logger.error(f"\n❌ Failed packages: {', '.join(failed_packages)}")
        logger.info("\n🔧 Manual installation commands:")
        for package in failed_packages:
            logger.info(f"pip install {package}")
    else:
        logger.info("\n✅ All dependencies installed successfully!")
        logger.info("\n🎯 Next steps:")
        logger.info("1. Run integration test: python test_backend_integration.py")
        logger.info("2. Start the server: python -m uvicorn app.main:app --reload")
        logger.info("3. Test API endpoints: http://localhost:8000/docs")

if __name__ == "__main__":
    main()