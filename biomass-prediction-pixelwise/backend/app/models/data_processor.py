# Data Preprocessing Pipeline for Above Ground Biomass Prediction
# Using Rasterio instead of GDAL for better compatibility

# 1.1 Setup and Dependencies
import earthaccess
import xarray as xr
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from datetime import datetime
import h5py
import pickle
import logging
from tqdm import tqdm
import dask
from dask.diagnostics import ProgressBar
import re
import json
import shutil
from datetime import datetime, timedelta
import subprocess

# Try to import rasterio, but don't fail if it's not available
try:
    import rasterio
    from rasterio.transform import from_origin
    RASTERIO_AVAILABLE = True
    print("Rasterio is available for geospatial data processing")
except ImportError:
    RASTERIO_AVAILABLE = False
    print("Rasterio not available, using alternative methods for geospatial data")

# Try to import rioxarray for geospatial xarray operations
try:
    import rioxarray
    RIOXARRAY_AVAILABLE = True
    print("Rioxarray is available for geospatial xarray operations")
except ImportError:
    RIOXARRAY_AVAILABLE = False
    print("Rioxarray not available, using alternative methods for geospatial operations")

# Try to import PIL for image processing
try:
    from PIL import Image
    PIL_AVAILABLE = True
    print("PIL is available for image processing")
except ImportError:
    PIL_AVAILABLE = False
    print("PIL not available, image processing capabilities limited")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('preprocessing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 1.2 Data Loading and Alignment
class DataProcessor:
    def __init__(self, data_dir=None, target_resolution=0.05, config=None):
        # Create data directory if it doesn't exist
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        
        # Normalize the path to use the correct separator for the current OS
        self.data_dir = os.path.normpath(data_dir)
        self.target_resolution = target_resolution  # degrees (approx. 5km)
        self.scalers = {}
        self.config = config or {}
        
        # Set up Dask for parallel processing
        dask.config.set(scheduler='threads', num_workers=4)
        
        # Create necessary subdirectories
        os.makedirs(os.path.join(self.data_dir, 'nasa'), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, 'processed'), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, 'visualizations'), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, 'validation_reports'), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, 'geotiff'), exist_ok=True)
        
        logger.info(f"Using data directory: {self.data_dir}")
        logger.info(f"Rasterio available: {RASTERIO_AVAILABLE}")
        logger.info(f"Rioxarray available: {RIOXARRAY_AVAILABLE}")
        logger.info(f"PIL available: {PIL_AVAILABLE}")
        
    def load_all_data(self):
        """Load all data from NASA Earth Access"""
        logger.info("Loading all data from NASA Earth Access...")
        
        # Focus on specific important regions in India
        regions = {
            'western_ghats': [74.0, 8.0, 78.0, 14.0],  # Biodiversity hotspot
            'central_india': [76.0, 20.0, 84.0, 26.0]  # Forest region
        }
        
        # Define datasets to fetch for 2024
        datasets_to_fetch = {
            'MODIS_NDVI': {
                'short_name': 'MOD13Q1',
                'version': '061',
                'description': 'Vegetation Index (NDVI)'
            },
            'MODIS_EVI': {
                'short_name': 'MOD13Q1',
                'version': '061',
                'description': 'Vegetation Index (EVI)'
            },
            'MODIS_LAI': {
                'short_name': 'MCD15A3H',
                'version': '061',
                'description': 'Leaf Area Index'
            },
            'MODIS_LST': {
                'short_name': 'MOD11A2',
                'version': '061',
                'description': 'Land Surface Temperature'
            },
            'MODIS_NPP': {
                'short_name': 'MOD17A3H',
                'version': '061',
                'description': 'Net Primary Productivity'
            },
            'SRTM': {
                'short_name': 'SRTMGL1',
                'version': '003',
                'description': 'Elevation data'
            },
            'GEDI': {
                'short_name': 'GEDI_L4A_AGB_Density_V2_1',
                'version': '2.1',
                'description': 'Biomass data'
            },
            'SMAP': {
                'short_name': 'SPL4SMAU',
                'version': '006',
                'description': 'Soil Moisture'
            },
            'CHIRPS': {
                'short_name': 'CHIRPS_DAILY',
                'version': '2.0',
                'description': 'Rainfall data'
            }
        }
        
        # Authenticate with NASA Earthdata
        try:
            auth = earthaccess.login()
            logger.info("Successfully authenticated with NASA Earthdata")
        except Exception as e:
            logger.error(f"Failed to authenticate with NASA Earthdata: {e}")
            return {}
        
        # Clean up old data before downloading new data
        self.cleanup_old_data(max_age_days=30)
        
        # Fetch data for each region
        all_data = {}
        for region_name, bbox in regions.items():
            logger.info(f"Fetching data for region: {region_name}")
            
            # Fetch data for 2024
            region_data = self.fetch_nasa_data_for_region(
                bbox, "2024-01-01", "2024-12-31", datasets_to_fetch, region_name
            )
            
            if region_data:
                all_data[region_name] = region_data
        
        return all_data
    
    def estimate_download_size(self, results):
        """Estimate download size for search results"""
        total_size = 0
        for result in results:
            try:
                size = result.get('size', 0)
                total_size += size
            except:
                pass
        return total_size
    
    def cleanup_old_data(self, max_age_days=30):
        """Clean up data older than specified number of days"""
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        
        for root, dirs, files in os.walk(os.path.join(self.data_dir, 'nasa')):
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                try:
                    dir_time = datetime.fromtimestamp(os.path.getmtime(dir_path))
                    if dir_time < cutoff_date:
                        logger.info(f"Removing old data: {dir_path}")
                        shutil.rmtree(dir_path)
                except Exception as e:
                    logger.error(f"Error cleaning up {dir_path}: {e}")
    
    def fetch_nasa_data_for_region(self, bbox, start_date, end_date, datasets_to_fetch, region_name):
        """Fetch NASA data for a specific region"""
        logger.info(f"Fetching NASA data for {region_name} from {start_date} to {end_date}")
        
        region_data = {}
        
        for name, config in datasets_to_fetch.items():
            try:
                logger.info(f"Fetching {name} data for {region_name}...")
                
                # Search for data
                results = earthaccess.search_data(
                    short_name=config['short_name'],
                    version=config['version'],
                    temporal=(start_date, end_date),
                    bounding_box=(bbox[0], bbox[1], bbox[2], bbox[3]),  # Correct format: (min_lon, min_lat, max_lon, max_lat)
                    count=20  # Limit to 20 granules to avoid excessive downloads
                )
                
                if results and len(results) > 0:
                    # Estimate download size
                    estimated_size = self.estimate_download_size(results)
                    logger.info(f"Estimated download size for {name} in {region_name}: {estimated_size / (1024*1024):.2f} MB")
                    
                    # Download data
                    local_path = os.path.join(self.data_dir, 'nasa', region_name, name)
                    os.makedirs(local_path, exist_ok=True)
                    
                    logger.info(f"Found {len(results)} granules for {name} in {region_name}")
                    logger.info(f"Downloading to {local_path}")
                    
                    files = earthaccess.download(results, local_path=local_path)
                    
                    # Convert HDF files to GeoTIFF for better compatibility with rasterio
                    if name.startswith('MODIS_'):
                        self.convert_hdf_to_geotiff(local_path)
                    
                    # Load the data
                    if name == 'MODIS_NDVI':
                        region_data['ndvi'] = self.load_modis_vi_data(local_path, 'NDVI')
                    elif name == 'MODIS_EVI':
                        region_data['evi'] = self.load_modis_vi_data(local_path, 'EVI')
                    elif name == 'MODIS_LAI':
                        region_data['lai'] = self.load_modis_lai_data(local_path)
                    elif name == 'MODIS_LST':
                        region_data['temperature'] = self.load_modis_lst_data(local_path)
                    elif name == 'MODIS_NPP':
                        region_data['npp'] = self.load_modis_npp_data(local_path)
                    elif name == 'SRTM':
                        region_data['elevation'] = self.load_srtm_data(local_path)
                    elif name == 'GEDI':
                        region_data['gedi'] = self.load_gedi_data(local_path)
                    elif name == 'SMAP':
                        region_data['soil_moisture'] = self.load_smap_data(local_path)
                    elif name == 'CHIRPS':
                        region_data['rainfall'] = self.load_chirps_data(local_path)
                    
                    logger.info(f"Successfully fetched and loaded {name} data for {region_name}")
                else:
                    logger.warning(f"No results found for {name} in {region_name}")
                    
            except Exception as e:
                logger.error(f"Error fetching {name} for {region_name}: {e}")
        
        return region_data
    
    def convert_hdf_to_geotiff(self, hdf_dir):
        """Convert all HDF files in a directory to GeoTIFF using gdal_translate"""
        logger.info(f"Converting HDF files to GeoTIFF in {hdf_dir}")
        
        # Find all HDF files
        hdf_files = glob.glob(os.path.join(hdf_dir, '*.hdf'))
        
        if not hdf_files:
            logger.warning(f"No HDF files found in {hdf_dir}")
            return
        
        # Create output directory for GeoTIFF files
        geotiff_dir = os.path.join(self.data_dir, 'geotiff', os.path.basename(hdf_dir))
        os.makedirs(geotiff_dir, exist_ok=True)
        
        # Convert each HDF file
        for hdf_file in hdf_files:
            try:
                # Get the base filename without extension
                base_name = os.path.splitext(os.path.basename(hdf_file))[0]
                
                # Use gdal_translate to convert HDF to GeoTIFF
                # First, list subdatasets
                cmd = f"gdalinfo {hdf_file}"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                
                if result.returncode != 0:
                    logger.error(f"Error listing subdatasets in {hdf_file}: {result.stderr}")
                    continue
                
                # Extract subdataset names
                subdatasets = []
                for line in result.stdout.split('\n'):
                    if 'SUBDATASET_' in line and '_NAME=' in line:
                        subdataset = line.split('=')[1].strip('"')
                        subdatasets.append(subdataset)
                
                # Convert each subdataset to GeoTIFF
                for i, subdataset in enumerate(subdatasets):
                    # Create output filename
                    output_file = os.path.join(geotiff_dir, f"{base_name}_{i}.tif")
                    
                    # Convert using gdal_translate
                    cmd = f"gdal_translate -of GTiff {subdataset} {output_file}"
                    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                    
                    if result.returncode != 0:
                        logger.error(f"Error converting {subdataset} to GeoTIFF: {result.stderr}")
                        continue
                    
                    logger.info(f"Converted {subdataset} to {output_file}")
                
                logger.info(f"Successfully converted {hdf_file} to GeoTIFF")
                
            except Exception as e:
                logger.error(f"Error converting {hdf_file} to GeoTIFF: {e}")
    
    def load_modis_vi_data(self, path, vi_type):
        """Load MODIS Vegetation Index data (NDVI or EVI) using Rasterio"""
        # First try to load from GeoTIFF files (and attempt auto-conversion if missing)
        geotiff_dir = os.path.join(self.data_dir, 'geotiff', os.path.basename(path))
        if not os.path.exists(geotiff_dir):
            # Try converting any HDFs present into GeoTIFFs
            self.convert_hdf_to_geotiff(path)
        if os.path.exists(geotiff_dir):
            tif_files = glob.glob(os.path.join(geotiff_dir, '*.tif'))
            if tif_files:
                return self.load_modis_vi_from_geotiff(tif_files, vi_type)
        
        # Fall back to HDF files
        files = glob.glob(os.path.join(path, '*.hdf'))
        if not files:
            return None
        
        datasets = []
        for file in files:
            try:
                # Extract date from filename
                date_str = self.extract_date_from_filename(os.path.basename(file))
                
                # Try to load with rasterio
                if RASTERIO_AVAILABLE:
                    try:
                        vi_data = self.load_modis_vi_with_rasterio(file, vi_type)
                        if vi_data is not None:
                            # Create xarray
                            da = xr.DataArray(
                                vi_data,
                                dims=['lat', 'lon'],
                                coords={
                                    'lat': np.linspace(vi_data.shape[0], 0, vi_data.shape[0]),
                                    'lon': np.linspace(0, vi_data.shape[1], vi_data.shape[1])
                                }
                            )
                            
                            if date_str:
                                da = da.expand_dims('time')
                                da = da.assign_coords(time=[pd.to_datetime(date_str)])
                            
                            datasets.append(da)
                            continue
                    except Exception as e:
                        logger.debug(f"Failed to load {file} with rasterio: {e}")
                
                # Fall back to h5py if rasterio fails or is not available
                logger.warning(f"Rasterio failed for {file}, trying h5py")
                with h5py.File(file, 'r') as f:
                    # Try different possible paths for VI data
                    possible_paths = [
                        f'MODIS_Grid_16DAY_250m_5000_VI/250m 16 days {vi_type}',
                        f'MODIS_Grid_16DAY_250m_5000_VI/{vi_type}',
                        f'MODIS_Grid_16DAY_250m_5000_VI/250m_16-days_{vi_type}',
                    ]
                    
                    vi_data = None
                    qc_data = None
                    
                    for path in possible_paths:
                        if path in f:
                            vi_data = f[path][:]
                            break
                    
                    # Try different paths for QC data
                    qc_paths = [
                        'MODIS_Grid_16DAY_250m_5000_VI/250m 16 days VI Quality',
                        'MODIS_Grid_16DAY_250m_5000_VI/VI Quality',
                        'MODIS_Grid_16DAY_250m_5000_VI/250m_16-days_VI_Quality',
                    ]
                    
                    for path in qc_paths:
                        if path in f:
                            qc_data = f[path][:]
                            break
                    
                    if vi_data is None:
                        logger.error(f"Could not find VI data in {file}")
                        continue
                    
                    # Apply quality control if QC data is available
                    if qc_data is not None:
                        good_quality = (qc_data & 0b00000011) == 0
                        vi_data = np.where(good_quality, vi_data, np.nan)
                    
                    # Scale the data
                    vi_data = vi_data * 0.0001  # Scale factor for MODIS VI
                    
                    # Create xarray
                    da = xr.DataArray(
                        vi_data,
                        dims=['lat', 'lon'],
                        coords={
                            'lat': np.linspace(vi_data.shape[0], 0, vi_data.shape[0]),
                            'lon': np.linspace(0, vi_data.shape[1], vi_data.shape[1])
                        }
                    )
                    
                    if date_str:
                        da = da.expand_dims('time')
                        da = da.assign_coords(time=[pd.to_datetime(date_str)])
                    
                    datasets.append(da)
                    
            except Exception as e:
                logger.error(f"Error loading MODIS {vi_type} file {file}: {e}")
        
        if datasets:
            # Concatenate along time dimension
            combined = xr.concat(datasets, dim='time')
            return combined
        else:
            return None
    
    def load_modis_vi_from_geotiff(self, tif_files, vi_type):
        """Load MODIS VI data from GeoTIFF files"""
        datasets = []
        
        for file in tif_files:
            try:
                # Extract date from filename
                date_str = self.extract_date_from_filename(os.path.basename(file))
                
                # Load with rasterio
                if RASTERIO_AVAILABLE:
                    with rasterio.open(file) as src:
                        # Read the data
                        vi_data = src.read(1)
                        
                        # Create xarray
                        da = xr.DataArray(
                            vi_data,
                            dims=['lat', 'lon'],
                            coords={
                                'lat': np.linspace(src.bounds.bottom, src.bounds.top, vi_data.shape[0]),
                                'lon': np.linspace(src.bounds.left, src.bounds.right, vi_data.shape[1])
                            }
                        )
                        
                        if date_str:
                            da = da.expand_dims('time')
                            da = da.assign_coords(time=[pd.to_datetime(date_str)])
                        
                        datasets.append(da)
                
            except Exception as e:
                logger.error(f"Error loading MODIS {vi_type} GeoTIFF file {file}: {e}")
        
        if datasets:
            # Concatenate along time dimension
            combined = xr.concat(datasets, dim='time')
            return combined
        else:
            return None
    
    def load_modis_vi_with_rasterio(self, file_path, vi_type):
        """Load MODIS VI data using Rasterio for HDF4 files"""
        try:
            # Open the HDF file with rasterio
            with rasterio.open(file_path) as src:
                # List subdatasets
                subdatasets = src.subdatasets
                
                # Find the subdataset that contains the VI data
                vi_data = None
                qc_data = None
                
                for subdataset in subdatasets:
                    if vi_type.upper() in subdataset:
                        with rasterio.open(subdataset) as sds:
                            vi_data = sds.read(1)
                    elif 'VI Quality' in subdataset:
                        with rasterio.open(subdataset) as sds:
                            qc_data = sds.read(1)
                
                if vi_data is None:
                    logger.error(f"Could not find {vi_type} data in {file_path}")
                    return None
                
                # Apply quality control if QC data is available
                if qc_data is not None:
                    good_quality = (qc_data & 0b00000011) == 0
                    vi_data = np.where(good_quality, vi_data, np.nan)
                
                # Scale the data
                vi_data = vi_data * 0.0001  # Scale factor for MODIS VI
                
                return vi_data
                
        except Exception as e:
            logger.error(f"Error loading MODIS VI data with rasterio from {file_path}: {e}")
            return None
    
    def load_modis_lai_data(self, path):
        """Load MODIS Leaf Area Index data using Rasterio"""
        # First try to load from GeoTIFF files (and attempt auto-conversion if missing)
        geotiff_dir = os.path.join(self.data_dir, 'geotiff', os.path.basename(path))
        if not os.path.exists(geotiff_dir):
            self.convert_hdf_to_geotiff(path)
        if os.path.exists(geotiff_dir):
            tif_files = glob.glob(os.path.join(geotiff_dir, '*.tif'))
            if tif_files:
                return self.load_modis_lai_from_geotiff(tif_files)
        
        # Fall back to HDF files
        files = glob.glob(os.path.join(path, '*.hdf'))
        if not files:
            return None
        
        datasets = []
        for file in files:
            try:
                # Extract date from filename
                date_str = self.extract_date_from_filename(os.path.basename(file))
                
                # Try to load with rasterio
                if RASTERIO_AVAILABLE:
                    try:
                        lai_data = self.load_modis_lai_with_rasterio(file)
                        if lai_data is not None:
                            # Create xarray
                            da = xr.DataArray(
                                lai_data,
                                dims=['lat', 'lon'],
                                coords={
                                    'lat': np.linspace(lai_data.shape[0], 0, lai_data.shape[0]),
                                    'lon': np.linspace(0, lai_data.shape[1], lai_data.shape[1])
                                }
                            )
                            
                            if date_str:
                                da = da.expand_dims('time')
                                da = da.assign_coords(time=[pd.to_datetime(date_str)])
                            
                            datasets.append(da)
                            continue
                    except Exception as e:
                        logger.debug(f"Failed to load {file} with rasterio: {e}")
                
                # Fall back to h5py if rasterio fails or is not available
                logger.warning(f"Rasterio failed for {file}, trying h5py")
                with h5py.File(file, 'r') as f:
                    # Try different possible paths for LAI data
                    possible_paths = [
                        'MOD_Grid_MCD15A3H/FparLai/Lai',
                        'MOD_Grid_MCD15A3H/Lai',
                        'MOD_Grid_MCD15A3H/FparLai/LAI',
                    ]
                    
                    lai_data = None
                    qc_data = None
                    
                    for path in possible_paths:
                        if path in f:
                            lai_data = f[path][:]
                            break
                    
                    # Try different paths for QC data
                    qc_paths = [
                        'MOD_Grid_MCD15A3H/FparLai_QC/QC',
                        'MOD_Grid_MCD15A3H/QC',
                        'MOD_Grid_MCD15A3H/FparLai_QC/FparLai_QC',
                    ]
                    
                    for path in qc_paths:
                        if path in f:
                            qc_data = f[path][:]
                            break
                    
                    if lai_data is None:
                        logger.error(f"Could not find LAI data in {file}")
                        continue
                    
                    # Apply quality control if QC data is available
                    if qc_data is not None:
                        good_quality = (qc_data & 0b00000011) == 0
                        lai_data = np.where(good_quality, lai_data, np.nan)
                    
                    # Scale the data
                    lai_data = lai_data * 0.1  # Scale factor for MODIS LAI
                    
                    # Create xarray
                    da = xr.DataArray(
                        lai_data,
                        dims=['lat', 'lon'],
                        coords={
                            'lat': np.linspace(lai_data.shape[0], 0, lai_data.shape[0]),
                            'lon': np.linspace(0, lai_data.shape[1], lai_data.shape[1])
                        }
                    )
                    
                    if date_str:
                        da = da.expand_dims('time')
                        da = da.assign_coords(time=[pd.to_datetime(date_str)])
                    
                    datasets.append(da)
                    
            except Exception as e:
                logger.error(f"Error loading MODIS LAI file {file}: {e}")
        
        if datasets:
            # Concatenate along time dimension
            combined = xr.concat(datasets, dim='time')
            return combined
        else:
            return None
    
    def load_modis_lai_from_geotiff(self, tif_files):
        """Load MODIS LAI data from GeoTIFF files"""
        datasets = []
        
        for file in tif_files:
            try:
                # Extract date from filename
                date_str = self.extract_date_from_filename(os.path.basename(file))
                
                # Load with rasterio
                if RASTERIO_AVAILABLE:
                    with rasterio.open(file) as src:
                        # Read the data
                        lai_data = src.read(1)
                        
                        # Create xarray
                        da = xr.DataArray(
                            lai_data,
                            dims=['lat', 'lon'],
                            coords={
                                'lat': np.linspace(src.bounds.bottom, src.bounds.top, lai_data.shape[0]),
                                'lon': np.linspace(src.bounds.left, src.bounds.right, lai_data.shape[1])
                            }
                        )
                        
                        if date_str:
                            da = da.expand_dims('time')
                            da = da.assign_coords(time=[pd.to_datetime(date_str)])
                        
                        datasets.append(da)
                
            except Exception as e:
                logger.error(f"Error loading MODIS LAI GeoTIFF file {file}: {e}")
        
        if datasets:
            # Concatenate along time dimension
            combined = xr.concat(datasets, dim='time')
            return combined
        else:
            return None
    
    def load_modis_lai_with_rasterio(self, file_path):
        """Load MODIS LAI data using Rasterio for HDF4 files"""
        try:
            # Open the HDF file with rasterio
            with rasterio.open(file_path) as src:
                # List subdatasets
                subdatasets = src.subdatasets
                
                # Find the subdataset that contains the LAI data
                lai_data = None
                qc_data = None
                
                for subdataset in subdatasets:
                    if 'Lai' in subdataset or 'LAI' in subdataset:
                        with rasterio.open(subdataset) as sds:
                            lai_data = sds.read(1)
                    elif 'QC' in subdataset and ('FparLai' in subdataset or 'Lai' in subdataset):
                        with rasterio.open(subdataset) as sds:
                            qc_data = sds.read(1)
                
                if lai_data is None:
                    logger.error(f"Could not find LAI data in {file_path}")
                    return None
                
                # Apply quality control if QC data is available
                if qc_data is not None:
                    good_quality = (qc_data & 0b00000011) == 0
                    lai_data = np.where(good_quality, lai_data, np.nan)
                
                # Scale the data
                lai_data = lai_data * 0.1  # Scale factor for MODIS LAI
                
                return lai_data
                
        except Exception as e:
            logger.error(f"Error loading MODIS LAI data with rasterio from {file_path}: {e}")
            return None
    
    def load_modis_lst_data(self, path):
        """Load MODIS Land Surface Temperature data using Rasterio"""
        # First try to load from GeoTIFF files (and attempt auto-conversion if missing)
        geotiff_dir = os.path.join(self.data_dir, 'geotiff', os.path.basename(path))
        if not os.path.exists(geotiff_dir):
            self.convert_hdf_to_geotiff(path)
        if os.path.exists(geotiff_dir):
            tif_files = glob.glob(os.path.join(geotiff_dir, '*.tif'))
            if tif_files:
                return self.load_modis_lst_from_geotiff(tif_files)
        
        # Fall back to HDF files
        files = glob.glob(os.path.join(path, '*.hdf'))
        if not files:
            return None
        
        datasets = []
        for file in files:
            try:
                # Extract date from filename
                date_str = self.extract_date_from_filename(os.path.basename(file))
                
                # Try to load with rasterio
                if RASTERIO_AVAILABLE:
                    try:
                        lst_data = self.load_modis_lst_with_rasterio(file)
                        if lst_data is not None:
                            # Create xarray
                            da = xr.DataArray(
                                lst_data,
                                dims=['lat', 'lon'],
                                coords={
                                    'lat': np.linspace(lst_data.shape[0], 0, lst_data.shape[0]),
                                    'lon': np.linspace(0, lst_data.shape[1], lst_data.shape[1])
                                }
                            )
                            
                            if date_str:
                                da = da.expand_dims('time')
                                da = da.assign_coords(time=[pd.to_datetime(date_str)])
                            
                            datasets.append(da)
                            continue
                    except Exception as e:
                        logger.debug(f"Failed to load {file} with rasterio: {e}")
                
                # Fall back to h5py if rasterio fails or is not available
                logger.warning(f"Rasterio failed for {file}, trying h5py")
                with h5py.File(file, 'r') as f:
                    # Try different possible paths for LST data
                    possible_paths = [
                        'MODIS_Grid_Daily_1km_LST/LST_Day_1km',
                        'MODIS_Grid_Daily_1km_LST/LST',
                        'MODIS_Grid_Daily_1km_LST/LST_Day',
                    ]
                    
                    lst_data = None
                    qc_data = None
                    
                    for path in possible_paths:
                        if path in f:
                            lst_data = f[path][:]
                            break
                    
                    # Try different paths for QC data
                    qc_paths = [
                        'MODIS_Grid_Daily_1km_LST/QC_Day',
                        'MODIS_Grid_Daily_1km_LST/QC',
                        'MODIS_Grid_Daily_1km_LST/QC_Day',
                    ]
                    
                    for path in qc_paths:
                        if path in f:
                            qc_data = f[path][:]
                            break
                    
                    if lst_data is None:
                        logger.error(f"Could not find LST data in {file}")
                        continue
                    
                    # Apply quality control if QC data is available
                    if qc_data is not None:
                        good_quality = (qc_data & 0b00000011) == 0
                        lst_data = np.where(good_quality, lst_data, np.nan)
                    
                    # Convert Kelvin to Celsius
                    lst_data = lst_data * 0.02 - 273.15
                    
                    # Create xarray
                    da = xr.DataArray(
                        lst_data,
                        dims=['lat', 'lon'],
                        coords={
                            'lat': np.linspace(lst_data.shape[0], 0, lst_data.shape[0]),
                            'lon': np.linspace(0, lst_data.shape[1], lst_data.shape[1])
                        }
                    )
                    
                    if date_str:
                        da = da.expand_dims('time')
                        da = da.assign_coords(time=[pd.to_datetime(date_str)])
                    
                    datasets.append(da)
                    
            except Exception as e:
                logger.error(f"Error loading MODIS LST file {file}: {e}")
        
        if datasets:
            # Concatenate along time dimension
            combined = xr.concat(datasets, dim='time')
            return combined
        else:
            return None
    
    def load_modis_lst_from_geotiff(self, tif_files):
        """Load MODIS LST data from GeoTIFF files"""
        datasets = []
        
        for file in tif_files:
            try:
                # Extract date from filename
                date_str = self.extract_date_from_filename(os.path.basename(file))
                
                # Load with rasterio
                if RASTERIO_AVAILABLE:
                    with rasterio.open(file) as src:
                        # Read the data
                        lst_data = src.read(1)
                        
                        # Create xarray
                        da = xr.DataArray(
                            lst_data,
                            dims=['lat', 'lon'],
                            coords={
                                'lat': np.linspace(src.bounds.bottom, src.bounds.top, lst_data.shape[0]),
                                'lon': np.linspace(src.bounds.left, src.bounds.right, lst_data.shape[1])
                            }
                        )
                        
                        if date_str:
                            da = da.expand_dims('time')
                            da = da.assign_coords(time=[pd.to_datetime(date_str)])
                        
                        datasets.append(da)
                
            except Exception as e:
                logger.error(f"Error loading MODIS LST GeoTIFF file {file}: {e}")
        
        if datasets:
            # Concatenate along time dimension
            combined = xr.concat(datasets, dim='time')
            return combined
        else:
            return None
    
    def load_modis_lst_with_rasterio(self, file_path):
        """Load MODIS LST data using Rasterio for HDF4 files"""
        try:
            # Open the HDF file with rasterio
            with rasterio.open(file_path) as src:
                # List subdatasets
                subdatasets = src.subdatasets
                
                # Find the subdataset that contains the LST data
                lst_data = None
                qc_data = None
                
                for subdataset in subdatasets:
                    if 'LST' in subdataset and ('Day' in subdataset or 'day' in subdataset):
                        with rasterio.open(subdataset) as sds:
                            lst_data = sds.read(1)
                    elif 'QC' in subdataset and ('Day' in subdataset or 'day' in subdataset):
                        with rasterio.open(subdataset) as sds:
                            qc_data = sds.read(1)
                
                if lst_data is None:
                    logger.error(f"Could not find LST data in {file_path}")
                    return None
                
                # Apply quality control if QC data is available
                if qc_data is not None:
                    good_quality = (qc_data & 0b00000011) == 0
                    lst_data = np.where(good_quality, lst_data, np.nan)
                
                # Convert Kelvin to Celsius
                lst_data = lst_data * 0.02 - 273.15
                
                return lst_data
                
        except Exception as e:
            logger.error(f"Error loading MODIS LST data with rasterio from {file_path}: {e}")
            return None
    
    def load_modis_npp_data(self, path):
        """Load MODIS Net Primary Productivity data using Rasterio"""
        # First try to load from GeoTIFF files (and attempt auto-conversion if missing)
        geotiff_dir = os.path.join(self.data_dir, 'geotiff', os.path.basename(path))
        if not os.path.exists(geotiff_dir):
            self.convert_hdf_to_geotiff(path)
        if os.path.exists(geotiff_dir):
            tif_files = glob.glob(os.path.join(geotiff_dir, '*.tif'))
            if tif_files:
                return self.load_modis_npp_from_geotiff(tif_files)
        
        # Fall back to HDF files
        files = glob.glob(os.path.join(path, '*.hdf'))
        if not files:
            return None
        
        datasets = []
        for file in files:
            try:
                # Extract date from filename
                date_str = self.extract_date_from_filename(os.path.basename(file))
                
                # Try to load with rasterio
                if RASTERIO_AVAILABLE:
                    try:
                        npp_data = self.load_modis_npp_with_rasterio(file)
                        if npp_data is not None:
                            # Create xarray
                            da = xr.DataArray(
                                npp_data,
                                dims=['lat', 'lon'],
                                coords={
                                    'lat': np.linspace(npp_data.shape[0], 0, npp_data.shape[0]),
                                    'lon': np.linspace(0, npp_data.shape[1], npp_data.shape[1])
                                }
                            )
                            
                            if date_str:
                                da = da.expand_dims('time')
                                da = da.assign_coords(time=[pd.to_datetime(date_str)])
                            
                            datasets.append(da)
                            continue
                    except Exception as e:
                        logger.debug(f"Failed to load {file} with rasterio: {e}")
                
                # Fall back to h5py if rasterio fails or is not available
                logger.warning(f"Rasterio failed for {file}, trying h5py")
                with h5py.File(file, 'r') as f:
                    # Try different possible paths for NPP data
                    possible_paths = [
                        'MOD_Grid_MOD17A3H/Npp',
                        'MOD_Grid_MOD17A3H/NPP',
                        'MOD_Grid_MOD17A3H/Npp',
                    ]
                    
                    npp_data = None
                    qc_data = None
                    
                    for path in possible_paths:
                        if path in f:
                            npp_data = f[path][:]
                            break
                    
                    # Try different paths for QC data
                    qc_paths = [
                        'MOD_Grid_MOD17A3H/Npp_QC/QC',
                        'MOD_Grid_MOD17A3H/QC',
                        'MOD_Grid_MOD17A3H/Npp_QC',
                    ]
                    
                    for path in qc_paths:
                        if path in f:
                            qc_data = f[path][:]
                            break
                    
                    if npp_data is None:
                        logger.error(f"Could not find NPP data in {file}")
                        continue
                    
                    # Apply quality control if QC data is available
                    if qc_data is not None:
                        good_quality = (qc_data & 0b00000011) == 0
                        npp_data = np.where(good_quality, npp_data, np.nan)
                    
                    # Scale the data (kg C/m2/year)
                    npp_data = npp_data * 0.0001  # Scale factor for MODIS NPP
                    
                    # Create xarray
                    da = xr.DataArray(
                        npp_data,
                        dims=['lat', 'lon'],
                        coords={
                            'lat': np.linspace(npp_data.shape[0], 0, npp_data.shape[0]),
                            'lon': np.linspace(0, npp_data.shape[1], npp_data.shape[1])
                        }
                    )
                    
                    if date_str:
                        da = da.expand_dims('time')
                        da = da.assign_coords(time=[pd.to_datetime(date_str)])
                    
                    datasets.append(da)
                    
            except Exception as e:
                logger.error(f"Error loading MODIS NPP file {file}: {e}")
        
        if datasets:
            # Concatenate along time dimension
            combined = xr.concat(datasets, dim='time')
            return combined
        else:
            return None
    
    def load_modis_npp_from_geotiff(self, tif_files):
        """Load MODIS NPP data from GeoTIFF files"""
        datasets = []
        
        for file in tif_files:
            try:
                # Extract date from filename
                date_str = self.extract_date_from_filename(os.path.basename(file))
                
                # Load with rasterio
                if RASTERIO_AVAILABLE:
                    with rasterio.open(file) as src:
                        # Read the data
                        npp_data = src.read(1)
                        
                        # Create xarray
                        da = xr.DataArray(
                            npp_data,
                            dims=['lat', 'lon'],
                            coords={
                                'lat': np.linspace(src.bounds.bottom, src.bounds.top, npp_data.shape[0]),
                                'lon': np.linspace(src.bounds.left, src.bounds.right, npp_data.shape[1])
                            }
                        )
                        
                        if date_str:
                            da = da.expand_dims('time')
                            da = da.assign_coords(time=[pd.to_datetime(date_str)])
                        
                        datasets.append(da)
                
            except Exception as e:
                logger.error(f"Error loading MODIS NPP GeoTIFF file {file}: {e}")
        
        if datasets:
            # Concatenate along time dimension
            combined = xr.concat(datasets, dim='time')
            return combined
        else:
            return None
    
    def load_modis_npp_with_rasterio(self, file_path):
        """Load MODIS NPP data using Rasterio for HDF4 files"""
        try:
            # Open the HDF file with rasterio
            with rasterio.open(file_path) as src:
                # List subdatasets
                subdatasets = src.subdatasets
                
                # Find the subdataset that contains the NPP data
                npp_data = None
                qc_data = None
                
                for subdataset in subdatasets:
                    if 'Npp' in subdataset or 'NPP' in subdataset:
                        with rasterio.open(subdataset) as sds:
                            npp_data = sds.read(1)
                    elif 'QC' in subdataset and ('Npp' in subdataset or 'NPP' in subdataset):
                        with rasterio.open(subdataset) as sds:
                            qc_data = sds.read(1)
                
                if npp_data is None:
                    logger.error(f"Could not find NPP data in {file_path}")
                    return None
                
                # Apply quality control if QC data is available
                if qc_data is not None:
                    good_quality = (qc_data & 0b00000011) == 0
                    npp_data = np.where(good_quality, npp_data, np.nan)
                
                # Scale the data (kg C/m2/year)
                npp_data = npp_data * 0.0001  # Scale factor for MODIS NPP
                
                return npp_data
                
        except Exception as e:
            logger.error(f"Error loading MODIS NPP data with rasterio from {file_path}: {e}")
            return None
    
    def load_srtm_data(self, path):
        """Load SRTM elevation data using rasterio or alternative methods"""
        files = glob.glob(os.path.join(path, '*.hgt'))
        if not files:
            logger.warning(f"No SRTM files found in {path}")
            return None
        
        if RASTERIO_AVAILABLE:
            return self.load_srtm_with_rasterio(files)
        else:
            return self.load_srtm_with_numpy(files)

    def load_srtm_with_rasterio(self, files):
        """Load SRTM elevation data using rasterio"""
        datasets = []
        
        for file in files:
            try:
                with rasterio.open(file) as src:
                    # Read the elevation data
                    data = src.read(1)
                    
                    # Convert to xarray
                    da = xr.DataArray(
                        data,
                        dims=['lat', 'lon'],
                        coords={
                            'lat': np.linspace(src.bounds.bottom, src.bounds.top, data.shape[0]),
                            'lon': np.linspace(src.bounds.left, src.bounds.right, data.shape[1])
                        }
                    )
                    
                    # Add CRS information if available
                    if RIOXARRAY_AVAILABLE:
                        da = da.rio.write_crs(src.crs, inplace=True)
                    
                    datasets.append(da)
                    
            except Exception as e:
                logger.error(f"Error loading SRTM file {file} with rasterio: {e}")
        
        if datasets:
            # Merge datasets
            combined = xr.concat(datasets, dim='lat')
            return combined
        else:
            return None

    def load_srtm_with_numpy(self, files):
        """Load SRTM elevation data using numpy (alternative to rasterio)"""
        datasets = []
        
        for file in files:
            try:
                # SRTM HGT files are 16-bit signed integers in big-endian format
                with open(file, 'rb') as f:
                    # Read the file header to get dimensions
                    f.seek(0)
                    # First 6 bytes: record length (unused)
                    # Next 4 bytes: width
                    width = int.from_bytes(f.read(4), byteorder='big')
                    # Next 4 bytes: height
                    height = int.from_bytes(f.read(4), byteorder='big')
                    
                    # Skip to data (header is 256 bytes)
                    f.seek(256)
                    
                    # Read elevation data
                    data = np.fromfile(f, dtype=np.int16, count=width * height)
                    
                    # Convert to float and handle no-data values (-32768)
                    data = data.astype(np.float32)
                    data[data == -32768] = np.nan
                    
                    # Reshape to 2D
                    data = data.reshape((height, width))
                    
                    # Create xarray
                    da = xr.DataArray(
                        data,
                        dims=['lat', 'lon'],
                        coords={
                            'lat': np.linspace(90 - (height * 0.0008333), 90, height),
                            'lon': np.linspace(-180 + (width * 0.0008333), -180 + (width * 0.0008333), width)
                        }
                    )
                    
                    datasets.append(da)
                    
            except Exception as e:
                logger.error(f"Error loading SRTM file {file} with numpy: {e}")
        
        if datasets:
            # Merge datasets
            combined = xr.concat(datasets, dim='lat')
            return combined
        else:
            return None
    
    def load_gedi_data(self, path):
        """Load GEDI biomass data"""
        gedi_files = glob.glob(os.path.join(path, '*.h5'))
        if not gedi_files:
            return None
        
        all_data = []
        
        for file in gedi_files:
            try:
                # Process each GEDI file
                df = self.process_gedi_file(file)
                if df is not None and not df.empty:
                    all_data.append(df)
            except Exception as e:
                logger.error(f"Error processing GEDI file {file}: {e}")
                continue
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        else:
            logger.warning("No GEDI data could be loaded")
            return pd.DataFrame()
    
    def process_gedi_file(self, file_path):
        """Process a single GEDI HDF5 file"""
        try:
            with h5py.File(file_path, 'r') as f:
                # Use a more robust approach to find datasets
                def find_dataset(group, prefix=''):
                    datasets = {}
                    for name, item in group.items():
                        full_path = f"{prefix}/{name}" if prefix else name
                        if isinstance(item, h5py.Dataset):
                            datasets[full_path] = item
                        elif isinstance(item, h5py.Group):
                            datasets.update(find_dataset(item, full_path))
                    return datasets
                
                all_datasets = find_dataset(f)
                
                # Find the required datasets with more flexible matching
                lat_paths = [p for p in all_datasets if 'lat' in p.lower() and 'lowest' in p.lower()]
                lon_paths = [p for p in all_datasets if 'lon' in p.lower() and 'lowest' in p.lower()]
                agbd_paths = [p for p in all_datasets if 'agbd' in p.lower() and 'quality' not in p.lower()]
                quality_paths = [p for p in all_datasets if 'quality' in p.lower() and 'agbd' in p.lower()]
                
                # Use the first found path for each required variable
                data = {}
                if lat_paths:
                    data['lat'] = all_datasets[lat_paths[0]][:]
                if lon_paths:
                    data['lon'] = all_datasets[lon_paths[0]][:]
                if agbd_paths:
                    data['agbd'] = all_datasets[agbd_paths[0]][:]
                if quality_paths:
                    data['quality'] = all_datasets[quality_paths[0]][:]
                
                # If we couldn't find the required data, return empty DataFrame
                if not all(key in data for key in ['lat', 'lon', 'agbd']):
                    logger.warning(f"Missing required datasets in {file_path}")
                    return pd.DataFrame()
                
                # Create DataFrame with additional metadata
                df = pd.DataFrame({
                    'latitude': data['lat'],
                    'longitude': data['lon'],
                    'biomass': data['agbd'],
                    'sensitivity': data.get('sensitivity', np.nan),  # Add if available
                    'elev_low': data.get('elev_lowestmode', np.nan),  # Add if available
                    'file': os.path.basename(file_path)
                })
                
                # Add quality if available
                if 'quality' in data:
                    df['quality'] = data['quality']
                    # Filter by quality (1 = good quality)
                    df = df[df['quality'] == 1]
                
                # Remove invalid values with more comprehensive filtering
                df = df.dropna()
                df = df[(df['biomass'] >= 0) & (df['biomass'] < 1000)]  # Reasonable biomass range
                df = df[(df['latitude'] >= -90) & (df['latitude'] <= 90)]
                df = df[(df['longitude'] >= -180) & (df['longitude'] <= 180)]
                
                return df
                
        except Exception as e:
            logger.error(f"Error processing GEDI file {file_path}: {e}")
            return pd.DataFrame()
    
    def load_smap_data(self, path):
        """Load SMAP soil moisture data"""
        files = glob.glob(os.path.join(path, '*.nc4'))
        if not files:
            return None
        
        datasets = []
        for file in files:
            try:
                # Open NetCDF file
                ds = xr.open_dataset(file)
                
                # Extract soil moisture data
                sm_data = ds['sm_surface'][:]
                sm_quality = ds['sm_quality'][:]
                
                # Apply quality control
                good_quality = (sm_quality & 0b00000000) == 0
                sm_data = np.where(good_quality, sm_data, np.nan)
                
                # Create xarray
                da = xr.DataArray(
                    sm_data,
                    dims=['lat', 'lon'],
                    coords={
                        'lat': ds['lat'][:],
                        'lon': ds['lon'][:]
                    }
                )
                
                # Add time dimension if available
                if 'time' in ds.dims:
                    da = da.expand_dims('time')
                    da = da.assign_coords(time=[ds['time'].values])
                
                datasets.append(da)
                ds.close()
                
            except Exception as e:
                logger.error(f"Error loading SMAP file {file}: {e}")
        
        if datasets:
            # Concatenate along time dimension
            combined = xr.concat(datasets, dim='time')
            return combined
        else:
            return None
    
    def load_chirps_data(self, path):
        """Load CHIRPS rainfall data"""
        files = glob.glob(os.path.join(path, '*.tif'))
        if not files:
            return None
        
        # Sort files to ensure consistent time ordering
        files.sort()
        
        # Load all TIFF files with memory mapping and progress bar
        datasets = []
        for file in tqdm(files, desc="Loading CHIRPS data"):
            try:
                # Extract date from filename
                date_str = self.extract_date_from_filename(os.path.basename(file))
                
                # Load TIFF with rioxarray if available
                if RIOXARRAY_AVAILABLE:
                    da = rioxarray.open_rasterio(file, chunks='auto', lock=False)
                elif RASTERIO_AVAILABLE:
                    # Fall back to rasterio
                    with rasterio.open(file) as src:
                        da = xr.DataArray(
                            src.read(1),
                            dims=['lat', 'lon'],
                            coords={
                                'lat': np.linspace(src.bounds.bottom, src.bounds.top, src.shape[0]),
                                'lon': np.linspace(src.bounds.left, src.bounds.right, src.shape[1])
                            }
                        )
                else:
                    # Fall back to numpy
                    logger.warning(f"Loading CHIRPS file {file} with numpy - limited georeferencing")
                    with open(file, 'rb') as f:
                        # Read TIFF header
                        header = f.read(8)
                        
                        # Check if it's a valid TIFF file
                        if header[:2] != b'II' and header[:2] != b'MM':
                            logger.warning(f"File {file} does not appear to be a valid TIFF file")
                            continue
                        
                        # Determine byte order
                        byte_order = '>' if header[:2] == b'MM' else '<'
                        
                        # Read offset to first IFD
                        offset = int.from_bytes(f.read(4), byte_order=byte_order)
                        
                        # Go to first IFD
                        f.seek(offset)
                        
                        # Read number of directory entries
                        num_entries = int.from_bytes(f.read(2), byte_order=byte_order)
                        
                        # Read directory entries to find image data
                        image_offset = None
                        width = None
                        height = None
                        bits_per_sample = None
                        
                        for _ in range(num_entries):
                            tag = int.from_bytes(f.read(2), byte_order=byte_order)
                            type_ = int.from_bytes(f.read(2), byte_order=byte_order)
                            length = int.from_bytes(f.read(4), byte_order=byte_order)
                            value_offset = int.from_bytes(f.read(4), byte_order=byte_order)
                            
                            if tag == 256:  # ImageWidth
                                f.seek(value_offset)
                                width = int.from_bytes(f.read(length), byte_order=byte_order)
                            elif tag == 257:  # ImageLength
                                f.seek(value_offset)
                                height = int.from_bytes(f.read(length), byte_order=byte_order)
                            elif tag == 258:  # BitsPerSample
                                f.seek(value_offset)
                                bits_per_sample = int.from_bytes(f.read(length), byte_order=byte_order)
                            elif tag == 273:  # StripOffsets
                                f.seek(value_offset)
                                image_offset = int.from_bytes(f.read(4), byte_order=byte_order)
                        
                        if None in [image_offset, width, height, bits_per_sample]:
                            logger.warning(f"Could not find required TIFF tags in {file}")
                            continue
                        
                        # Read image data
                        f.seek(image_offset)
                        
                        # Calculate bytes per pixel
                        bytes_per_pixel = bits_per_sample // 8
                        
                        # Read image data
                        if bits_per_sample == 8:
                            data = np.fromfile(f, dtype=np.uint8, count=width * height)
                        elif bits_per_sample == 16:
                            data = np.fromfile(f, dtype=np.uint16, count=width * height)
                        else:
                            logger.warning(f"Unsupported bits per sample: {bits_per_sample}")
                            continue
                        
                        # Reshape to 2D
                        data = data.reshape((height, width))
                        
                        # Convert to float
                        data = data.astype(np.float32)
                        
                        # Create xarray
                        # Note: Without proper georeferencing, we'll use generic coordinates
                        da = xr.DataArray(
                            data,
                            dims=['lat', 'lon'],
                            coords={
                                'lat': np.linspace(90, -90, height),
                                'lon': np.linspace(-180, 180, width)
                            }
                        )
                
                # Handle different band structures
                if 'band' in da.dims and da.band.size > 1:
                    # For multi-band TIFFs, select the first band or average
                    da = da.sel(band=1)
                
                # Add time dimension if extracted
                if date_str:
                    da = da.expand_dims('time')
                    da = da.assign_coords(time=[pd.to_datetime(date_str)])
                
                # Rename spatial dimensions to standard names
                if 'x' in da.dims:
                    da = da.rename({'x': 'lon', 'y': 'lat'})
                
                # Set spatial reference if missing
                if RIOXARRAY_AVAILABLE and da.rio.crs is None:
                    da = da.rio.set_crs("EPSG:4326")
                
                datasets.append(da)
            except Exception as e:
                logger.error(f"Error loading {file}: {e}")
                continue
        
        if not datasets:
            return None
        
        # Concatenate along time dimension if multiple files
        if len(datasets) > 1:
            try:
                with ProgressBar():
                    combined = xr.concat(datasets, dim='time')
            except Exception as e:
                logger.error(f"Error concatenating CHIRPS datasets: {e}")
                # If concatenation fails, return the first dataset
                combined = datasets[0]
        else:
            combined = datasets[0]
        
        return combined
    
    def extract_date_from_filename(self, filename):
        """Extract date from filename with multiple format support"""
        try:
            # Try MODIS format: MOD13Q1...doyYYYYDDD...
            m = re.search(r'doy(\d{7})', filename)
            if m:
                year = int(m.group(1)[:4])
                doy = int(m.group(1)[4:7])
                return pd.to_datetime(f'{year}-01-01') + pd.Timedelta(days=doy - 1)

            # Try MODIS AYYYYDDD token: MOD11A2.A2019001.h24v06.061....
            m = re.search(r'\.A(\d{7})\.', filename)
            if m:
                token = m.group(1)
                year = int(token[:4])
                doy = int(token[4:7])
                return pd.to_datetime(f'{year}-01-01') + pd.Timedelta(days=doy - 1)
            
            # Try YYYYMMDD format
            date_match = re.search(r'(?<!\d)(\d{8})(?!\d)', filename)
            if date_match:
                date_str = date_match.group(1)
                return pd.to_datetime(date_str, format='%Y%m%d')
            
            # Try YYYY-MM-DD format
            date_match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
            if date_match:
                return pd.to_datetime(date_match.group(1))
            
            # Try YYYY format
            year_match = re.search(r'(?<!\d)(\d{4})(?!\d)', filename)
            if year_match:
                return pd.to_datetime(f"{year_match.group(1)}-01-01")
            
            # Default to None if no date pattern found
            return None
        except Exception as e:
            logger.error(f"Could not extract date from {filename}: {e}")
            return None
    
    def align_datasets(self, datasets):
        """Align all datasets to common grid and time with improved error handling"""
        logger.info("Aligning datasets to common grid")
        
        # Filter out None values
        valid_datasets = {k: v for k, v in datasets.items() if v is not None}
        
        if not valid_datasets:
            raise ValueError("No valid datasets found for alignment")
        
        # Ensure all datasets have the same CRS
        for name, ds in valid_datasets.items():
            if hasattr(ds, 'rio') and ds.rio.crs is not None:
                if ds.rio.crs != "EPSG:4326":
                    if RIOXARRAY_AVAILABLE:
                        valid_datasets[name] = ds.rio.reproject("EPSG:4326")
                    else:
                        logger.warning(f"Cannot reproject {name} without rioxarray")
        
        # Find common spatial grid with more robust handling
        lons = []
        lats = []
        
        for name, ds in valid_datasets.items():
            if hasattr(ds, 'lon') and hasattr(ds, 'lat'):
                lons.extend([ds.lon.min().values, ds.lon.max().values])
                lats.extend([ds.lat.min().values, ds.lat.max().values])
            elif name == 'gedi' and not ds.empty:
                # For GEDI data, use the min/max of the coordinates
                lons.extend([ds.longitude.min(), ds.longitude.max()])
                lats.extend([ds.latitude.min(), ds.latitude.max()])
        
        if not lons or not lats:
            raise ValueError("No valid spatial data found in datasets")
        
        min_lon, max_lon = min(lons), max(lons)
        min_lat, max_lat = min(lats), max(lats)
        
        # Create common grid
        common_lons = np.arange(min_lon, max_lon, self.target_resolution)
        common_lats = np.arange(max_lat, min_lat, -self.target_resolution)
        
        logger.info(f"Common grid: {len(common_lons)} x {len(common_lats)} points")
        
        # Align all datasets with better error handling
        aligned = {}
        for name, ds in tqdm(valid_datasets.items(), desc="Aligning datasets"):
            try:
                if name == 'gedi':
                    # Process GEDI point data
                    aligned[name] = self.align_gedi_data(ds, common_lons, common_lats)
                elif hasattr(ds, 'lon'):
                    # Process gridded data with better interpolation
                    aligned[name] = ds.interp(
                        lon=common_lons,
                        lat=common_lats,
                        method='linear',
                        kwargs={'fill_value': np.nan}  # Handle missing data
                    )
            except Exception as e:
                logger.error(f"Error aligning dataset {name}: {e}")
                # Create a NaN-filled array as fallback
                aligned[name] = xr.DataArray(
                    np.full((len(common_lats), len(common_lons)), np.nan),
                    dims=['lat', 'lon'],
                    coords={'lat': common_lats, 'lon': common_lons}
                )
        
        return aligned, common_lons, common_lats
    
    def align_gedi_data(self, gedi_df, common_lons, common_lats):
        """Align GEDI point data to common grid"""
        # Create grid
        grid = np.zeros((len(common_lats), len(common_lons)))
        count = np.zeros((len(common_lats), len(common_lons)))
        
        # Assign GEDI points to grid cells
        for _, row in gedi_df.iterrows():
            lon_idx = np.argmin(np.abs(common_lons - row['longitude']))
            lat_idx = np.argmin(np.abs(common_lats - row['latitude']))
            
            if 0 <= lat_idx < len(common_lats) and 0 <= lon_idx < len(common_lons):
                grid[lat_idx, lon_idx] += row['biomass']
                count[lat_idx, lon_idx] += 1
        
        # Calculate mean biomass per cell
        mean_biomass = np.divide(grid, count, out=np.zeros_like(grid), where=count!=0)
        
        # Convert to xarray
        return xr.DataArray(
            mean_biomass,
            dims=['lat', 'lon'],
            coords={'lat': common_lats, 'lon': common_lons}
        )
    
    def apply_quality_control(self, datasets):
        """Apply quality control flags with improved error handling"""
        logger.info("Applying quality control flags")
        
        # Apply VI quality control (guard against missing/invalid)
        if datasets.get('vi_quality') is not None:
            try:
                if 'ndvi' in datasets and datasets['ndvi'] is not None:
                    good_quality = datasets['vi_quality'] < 2
                    datasets['ndvi'] = datasets['ndvi'].where(good_quality)
                if 'evi' in datasets and datasets['evi'] is not None:
                    good_quality = datasets['vi_quality'] < 2
                    datasets['evi'] = datasets['evi'].where(good_quality)
                logger.info("Applied VI quality control")
            except Exception as e:
                logger.warning(f"Skipping VI QC due to error: {e}")
        
        # Apply FPAR/LAI quality control (guard against missing/invalid)
        if datasets.get('fpar_lai_qc') is not None:
            try:
                good_quality = datasets['fpar_lai_qc'] < 2
                if 'fpar' in datasets and datasets['fpar'] is not None:
                    datasets['fpar'] = datasets['fpar'].where(good_quality)
                if 'lai' in datasets and datasets['lai'] is not None:
                    datasets['lai'] = datasets['lai'].where(good_quality)
                logger.info("Applied FPAR/LAI quality control")
            except Exception as e:
                logger.warning(f"Skipping FPAR/LAI QC due to error: {e}")
        
        return datasets
    
    def normalize_data(self, datasets):
        """Normalize all datasets"""
        logger.info("Normalizing datasets")
        normalized = {}
        
        for name, ds in tqdm(datasets.items(), desc="Normalizing data"):
            if name == 'gedi':
                # Normalize GEDI data
                scaler = StandardScaler()
                data = ds.values.flatten()
                data = data[~np.isnan(data)]  # Remove NaN
                if len(data) > 0:
                    data = data.reshape(-1, 1)
                    scaler.fit(data)
                    
                    # Apply normalization
                    normalized_data = scaler.transform(ds.values.flatten().reshape(-1, 1))
                    normalized[name] = normalized_data.reshape(ds.shape)
                    self.scalers[name] = scaler
                else:
                    # If no valid data, use original
                    normalized[name] = ds.values
                    self.scalers[name] = StandardScaler()
                
            elif hasattr(ds, 'values'):
                # Normalize gridded data
                scaler = StandardScaler()
                data = ds.values.flatten()
                data = data[~np.isnan(data)]  # Remove NaN
                if len(data) > 0:
                    data = data.reshape(-1, 1)
                    scaler.fit(data)
                    
                    # Apply normalization
                    normalized_data = scaler.transform(ds.values.flatten().reshape(-1, 1))
                    normalized[name] = normalized_data.reshape(ds.shape)
                    self.scalers[name] = scaler
                else:
                    # If no valid data, use original
                    normalized[name] = ds.values
                    self.scalers[name] = StandardScaler()
        
        return normalized
    
    def create_sequences(self, datasets, seq_length=12, max_missing_ratio=0.2):
        """Create time series sequences for LSTM with missing data handling"""
        logger.info(f"Creating sequences with length {seq_length}")
        
        # Find all time steps
        all_times = []
        for ds in datasets.values():
            if hasattr(ds, 'time'):
                all_times.extend(ds.time.values)
        
        unique_times = sorted(list(set(all_times)))
        logger.info(f"Found {len(unique_times)} unique time steps")
        
        # Create sequences
        sequences = []
        targets = []
        
        for i in tqdm(range(len(unique_times) - seq_length), desc="Creating sequences"):
            # Get sequence of time steps
            seq_times = unique_times[i:i+seq_length]
            target_time = unique_times[i+seq_length]
            
            # Extract data for each variable
            seq_data = []
            missing_count = 0
            total_count = 0
            
            for name, ds in datasets.items():
                if name == 'gedi':
                    # GEDI is static, so we repeat it
                    seq_data.append(np.tile(ds.values, (seq_length, 1, 1)))
                    # Count missing values in GEDI
                    missing_count += np.sum(np.isnan(ds.values))
                    total_count += ds.values.size
                elif hasattr(ds, 'time'):
                    # Time-varying data
                    try:
                        time_data = ds.sel(time=seq_times).values
                        seq_data.append(time_data)
                        # Count missing values
                        missing_count += np.sum(np.isnan(time_data))
                        total_count += time_data.size
                    except Exception as e:
                        logger.error(f"Error extracting time data for {name}: {e}")
                        # Create NaN-filled array as fallback
                        nan_array = np.full((seq_length, ds.shape[1], ds.shape[2]), np.nan)
                        seq_data.append(nan_array)
                        missing_count += nan_array.size
                        total_count += nan_array.size
                else:
                    # Static data (e.g., landcover)
                    seq_data.append(np.tile(ds.values, (seq_length, 1, 1)))
                    # Count missing values
                    missing_count += np.sum(np.isnan(ds.values))
                    total_count += ds.values.size
            
            # Check if sequence has too much missing data
            if total_count > 0 and (missing_count / total_count) > max_missing_ratio:
                continue  # Skip this sequence
            
            # Stack all variables
            seq_data = np.stack(seq_data, axis=-1)  # (seq_length, height, width, features)
            sequences.append(seq_data)
            
            # Target is GEDI biomass if available, otherwise use LAI
            if 'gedi' in datasets:
                targets.append(datasets['gedi'].values)
            elif 'lai' in datasets and hasattr(datasets['lai'], 'time'):
                try:
                    targets.append(datasets['lai'].sel(time=target_time).values)
                except:
                    # If target time not available, use last available time step
                    targets.append(datasets['lai'].sel(time=seq_times[-1]).values)
            else:
                # Fallback to a default variable
                targets.append(datasets['ndvi'].sel(time=target_time).values)
        
        # Convert to numpy arrays
        sequences = np.array(sequences)
        targets = np.array(targets)
        
        # Log sequence statistics
        logger.info(f"Created {len(sequences)} valid sequences out of {len(unique_times) - seq_length} possible")
        logger.info(f"Sequence shape: {sequences.shape}")
        logger.info(f"Target shape: {targets.shape}")
        
        return sequences, targets
    
    def visualize_data(self, datasets):
        """Visualize the preprocessed data"""
        logger.info("Creating data visualizations")
        
        # Create output directory for visualizations
        os.makedirs(os.path.join(self.data_dir, 'visualizations'), exist_ok=True)
        
        # Plot biomass distribution
        if 'gedi' in datasets:
            plt.figure(figsize=(10, 6))
            sns.histplot(datasets['gedi'].values.flatten(), bins=50, kde=True)
            plt.title('Biomass Distribution')
            plt.xlabel('Biomass (Mg/ha)')
            plt.ylabel('Frequency')
            plt.savefig(os.path.join(self.data_dir, 'visualizations', 'biomass_distribution.png'))
            plt.close()
        
        # Plot time series for a sample location
        if 'ndvi' in datasets and hasattr(datasets['ndvi'], 'time'):
            plt.figure(figsize=(12, 6))
            # Select a sample point in the middle of the domain
            mid_lat = len(datasets['ndvi'].lat) // 2
            mid_lon = len(datasets['ndvi'].lon) // 2
            datasets['ndvi'].isel(lat=mid_lat, lon=mid_lon).plot.line('o-')
            plt.title('NDVI Time Series at Sample Location')
            plt.ylabel('NDVI')
            plt.savefig(os.path.join(self.data_dir, 'visualizations', 'ndvi_timeseries.png'))
            plt.close()
        
        # Plot spatial coverage
        if 'ndvi' in datasets:
            plt.figure(figsize=(10, 8))
            # Use the first time step
            datasets['ndvi'].isel(time=0).plot(cmap='viridis')
            plt.title('Spatial Coverage of NDVI')
            plt.savefig(os.path.join(self.data_dir, 'visualizations', 'spatial_coverage.png'))
            plt.close()
        
        logger.info(f"Visualizations saved to {os.path.join(self.data_dir, 'visualizations')}")
    
    def validate_preprocessed_data(self, sequences, targets, region_name):
        """Comprehensive validation of preprocessed data"""
        logger.info(f"Validating preprocessed data for {region_name}")
        
        validation_results = {
            'region': region_name,
            'sequences_shape': sequences.shape,
            'targets_shape': targets.shape,
            'issues': []
        }
        
        # Check for NaN values
        nan_sequences = np.sum(np.isnan(sequences))
        nan_targets = np.sum(np.isnan(targets))
        nan_seq_pct = nan_sequences / sequences.size * 100
        nan_tgt_pct = nan_targets / targets.size * 100
        
        if nan_seq_pct > 5:  # More than 5% NaN values
            validation_results['issues'].append(f"High NaN values in sequences: {nan_seq_pct:.2f}%")
        
        if nan_tgt_pct > 5:  # More than 5% NaN values
            validation_results['issues'].append(f"High NaN values in targets: {nan_tgt_pct:.2f}%")
        
        # Check for infinite values
        inf_sequences = np.sum(np.isinf(sequences))
        inf_targets = np.sum(np.isinf(targets))
        
        if inf_sequences > 0:
            validation_results['issues'].append(f"Infinite values in sequences: {inf_sequences}")
        
        if inf_targets > 0:
            validation_results['issues'].append(f"Infinite values in targets: {inf_targets}")
        
        # Check data ranges
        seq_min, seq_max = np.nanmin(sequences), np.nanmax(sequences)
        tgt_min, tgt_max = np.nanmin(targets), np.nanmax(targets)
        
        validation_results['sequence_range'] = [seq_min, seq_max]
        validation_results['target_range'] = [tgt_min, tgt_max]
        
        # Check for reasonable biomass values
        if tgt_max > 1000:  # Unrealistically high biomass
            validation_results['issues'].append(f"Unrealistically high biomass values: {tgt_max}")
        
        if tgt_min < 0:  # Negative biomass
            validation_results['issues'].append(f"Negative biomass values: {tgt_min}")
        
        # Save validation report
        report_path = os.path.join(self.data_dir, 'validation_reports', f"{region_name}_validation.json")
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(validation_results, f, indent=2)
        
        logger.info(f"Validation report saved to {report_path}")
        
        return validation_results
    
    def process_all_regions(self, all_data):
        """Process data from all regions"""
        logger.info("Processing data from all regions...")
        
        # Combine data from all regions
        combined_data = {}
        
        for region_name, region_data in all_data.items():
            logger.info(f"Processing data for region: {region_name}")
            
            # Apply quality control
            qc_data = self.apply_quality_control(region_data)
            
            # Align datasets
            aligned_data, common_lons, common_lats = self.align_datasets(qc_data)
            
            # Normalize data
            normalized_data = self.normalize_data(aligned_data)
            
            # Create sequences
            sequences, targets = self.create_sequences(normalized_data, seq_length=12)
            
            # Store processed data
            combined_data[region_name] = {
                'sequences': sequences,
                'targets': targets,
                'common_lons': common_lons,
                'common_lats': common_lats
            }
            
            # Validate the preprocessed data
            logger.info(f"Validating preprocessed data for {region_name}...")
            validation_results = self.validate_preprocessed_data(sequences, targets, region_name)
            
            # Create visualizations
            logger.info(f"Creating visualizations for {region_name}...")
            self.visualize_data(aligned_data)
            
            # Save processed data
            region_dir = os.path.join(self.data_dir, 'processed', region_name)
            os.makedirs(region_dir, exist_ok=True)
            
            np.save(os.path.join(region_dir, 'sequences.npy'), sequences)
            np.save(os.path.join(region_dir, 'targets.npy'), targets)
            np.save(os.path.join(region_dir, 'common_lons.npy'), common_lons)
            np.save(os.path.join(region_dir, 'common_lats.npy'), common_lats)
            
            # Save scalers for inference
            with open(os.path.join(region_dir, 'scalers.pkl'), 'wb') as f:
                pickle.dump(self.scalers, f)
            
            logger.info(f"Processing complete for {region_name}!")
            logger.info(f"Sequences shape: {sequences.shape}")
            logger.info(f"Targets shape: {targets.shape}")
            logger.info(f"Data saved to {region_dir}")
            
            # Log validation results
            if validation_results['issues']:
                logger.warning(f"Validation issues for {region_name}: {validation_results['issues']}")
            else:
                logger.info(f"No validation issues found for {region_name}")
        
        return combined_data


# 1.3 Preprocessing Execution
def main():
    try:
        # Initialize processor
        processor = DataProcessor()
        
        # Load all data from NASA Earth Access
        logger.info("Loading all data from NASA Earth Access...")
        all_data = processor.load_all_data()
        
        if not all_data:
            logger.error("No data could be loaded. Exiting.")
            return
        
        # Process data from all regions
        logger.info("Processing data from all regions...")
        processed_data = processor.process_all_regions(all_data)
        
        logger.info("All preprocessing complete!")
        
    except Exception as e:
        logger.error(f"Error in preprocessing pipeline: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()