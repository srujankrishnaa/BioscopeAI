# Data Preprocessing Pipeline for Above Ground Biomass Prediction
# Combines satellite data, climate data, and GEDI biomass measurements

# 1.1 Setup and Dependencies
import earthaccess
import xarray as xr
import rioxarray
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
    def __init__(self, data_dir, target_resolution=0.05, config=None):
        self.data_dir = data_dir
        self.target_resolution = target_resolution  # degrees (approx. 5km)
        self.scalers = {}
        self.config = config or {}
        
        # Set up Dask for parallel processing
        dask.config.set(scheduler='threads', num_workers=4)
        
    def load_local_datasets(self):
        """Load all local datasets with improved error handling"""
        logger.info("Loading local datasets...")
        datasets = {}
        
        # Define datasets to load
        dataset_configs = {
            'evi': 'evi',
            'fpar': 'fpar',
            'gpp': 'gpp',
            'lai': 'lai',
            'ndvi': 'ndvi',
            'npp': 'npp',
            'rainfall': 'rainfall',
            'temperature': 'temperature',
            'landcover': 'landcover',
            'fpar_lai_qc': 'fpar_lai_qc',
            'vi_quality': 'vi quality'
        }
        
        # Load each dataset with progress bar
        for name, folder in tqdm(dataset_configs.items(), desc="Loading datasets"):
            try:
                datasets[name] = self.load_tiff_folder(folder)
                if datasets[name] is not None:
                    logger.info(f"Successfully loaded {name}")
                else:
                    logger.warning(f"Failed to load {name}")
            except Exception as e:
                logger.error(f"Error loading {name}: {e}")
                datasets[name] = None
        
        # Load GEDI biomass data
        try:
            datasets['gedi'] = self.load_gedi_data()
            if datasets['gedi'] is not None and not datasets['gedi'].empty:
                logger.info(f"Successfully loaded GEDI data with {len(datasets['gedi'])} records")
            else:
                logger.warning("No GEDI data loaded")
        except Exception as e:
            logger.error(f"Error loading GEDI data: {e}")
            datasets['gedi'] = pd.DataFrame()
        
        return datasets

    def load_tiff_folder(self, folder_name):
        """Load TIFF files from a specific folder with improved handling"""
        folder_path = os.path.join(self.data_dir, folder_name)
        if not os.path.exists(folder_path):
            logger.warning(f"Folder not found: {folder_path}")
            return None
        
        # Find all TIFF files in the folder
        tiff_files = glob.glob(os.path.join(folder_path, '*.tif'))
        if not tiff_files:
            logger.warning(f"No TIFF files found in: {folder_path}")
            return None
        
        # Sort files to ensure consistent time ordering
        tiff_files.sort()
        
        # Load all TIFF files with memory mapping and progress bar
        datasets = []
        for file in tqdm(tiff_files, desc=f"Loading {folder_name}"):
            try:
                # Extract date from filename with multiple format support
                time_str = self.extract_date_from_filename(os.path.basename(file))
                
                # Load TIFF with rioxarray with memory mapping
                da = rioxarray.open_rasterio(file, chunks='auto', lock=False)
                
                # Handle different band structures
                if da.band.size > 1:
                    # For multi-band TIFFs, select the first band or average
                    da = da.sel(band=1)
                
                # Add time dimension if extracted
                if time_str:
                    da = da.expand_dims('time')
                    da = da.assign_coords(time=[pd.to_datetime(time_str)])
                
                # Rename spatial dimensions to standard names
                if 'x' in da.dims:
                    da = da.rename({'x': 'lon', 'y': 'lat'})
                
                # Set spatial reference if missing
                if da.rio.crs is None:
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
                logger.error(f"Error concatenating datasets for {folder_name}: {e}")
                # If concatenation fails, return the first dataset
                combined = datasets[0]
        else:
            combined = datasets[0]
        
        return combined

    def extract_date_from_filename(self, filename):
        """Extract date from filename with multiple format support"""
        try:
            # Try MODIS format: MOD13Q1.061__250m_16_days_EVI_doy2023033000000_aid0001.tif
            if 'doy' in filename:
                doy_part = filename.split('doy')[1][:7]  # e.g., "2023033"
                year = int(doy_part[:4])
                doy = int(doy_part[4:7])
                
                # Convert DOY to date
                date = pd.to_datetime(f'{year}-01-01') + pd.Timedelta(days=doy-1)
                return date
            
            # Try YYYYMMDD format
            date_match = re.search(r'(\d{8})', filename)
            if date_match:
                date_str = date_match.group(1)
                return pd.to_datetime(date_str, format='%Y%m%d')
            
            # Try YYYY-MM-DD format
            date_match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
            if date_match:
                return pd.to_datetime(date_match.group(1))
            
            # Try YYYY format
            year_match = re.search(r'(\d{4})', filename)
            if year_match:
                return pd.to_datetime(f"{year_match.group(1)}-01-01")
            
            # Default to None if no date pattern found
            return None
        except Exception as e:
            logger.error(f"Could not extract date from {filename}: {e}")
            return None

    def load_netcdf(self, pattern):
        """Load NetCDF files matching pattern"""
        files = glob.glob(os.path.join(self.data_dir, pattern))
        if not files:
            logger.warning(f"No files found for pattern: {pattern}")
            return None
        
        # Open and concatenate all files
        datasets = [xr.open_dataset(f) for f in files]
        combined = xr.concat(datasets, dim='time')
        return combined

    def load_gedi_data(self):
        """Load GEDI biomass data from multiple folders with robust path handling"""
        gedi_folders = glob.glob(os.path.join(self.data_dir, 'GEDI_L4A_AGB_Density*'))
        all_data = []
        
        for folder in tqdm(gedi_folders, desc="Processing GEDI folders"):
            files = glob.glob(os.path.join(folder, '*.h5'))
            for file in files:
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
        """Process a single GEDI HDF5 file with robust path handling"""
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

    def fetch_nasa_data(self, bbox, start_date, end_date):
        """Fetch NASA data using earthaccess"""
        logger.info(f"Fetching NASA data for bbox {bbox} from {start_date} to {end_date}")
        
        # Authenticate
        try:
            auth = earthaccess.login()
        except Exception as e:
            logger.error(f"Failed to authenticate with earthaccess: {e}")
            return {}
        
        # Define datasets to fetch
        datasets_to_fetch = {
            'SRTM': 'SRTMGL1',  # Elevation
            'SMAP': 'SPL4SMAU',  # Soil moisture
            'MODIS_FIRE': 'MCD64A1'  # Burned area
        }
        
        fetched_data = {}
        for name, short_name in tqdm(datasets_to_fetch.items(), desc="Fetching NASA data"):
            try:
                # Search for data
                results = earthaccess.search_data(
                    short_name=short_name,
                    temporal=(start_date, end_date),
                    bounding_box=bbox,
                    count=10
                )
                
                if results:
                    # Download data
                    local_path = os.path.join(self.data_dir, 'nasa', name)
                    os.makedirs(local_path, exist_ok=True)
                    files = earthaccess.download(results, local_path=local_path)
                    
                    # Load the data
                    if name == 'SRTM':
                        fetched_data['elevation'] = self.load_netcdf(os.path.join(local_path, '*.hgt'))
                    elif name == 'SMAP':
                        fetched_data['soil_moisture'] = self.load_netcdf(os.path.join(local_path, '*.nc4'))
                    elif name == 'MODIS_FIRE':
                        fetched_data['burned_area'] = self.load_netcdf(os.path.join(local_path, '*.hdf'))
                    
                    logger.info(f"Successfully fetched and loaded {name}")
                else:
                    logger.warning(f"No results found for {name}")
            except Exception as e:
                logger.error(f"Error fetching {name}: {e}")
        
        return fetched_data

    def align_datasets(self, datasets):
        """Align all datasets to common grid and time with improved error handling"""
        logger.info("Aligning datasets to common grid")
        
        # Ensure all datasets have the same CRS
        for name, ds in datasets.items():
            if hasattr(ds, 'rio') and ds.rio.crs is not None:
                if ds.rio.crs != "EPSG:4326":
                    datasets[name] = ds.rio.reproject("EPSG:4326")
        
        # Find common spatial grid with more robust handling
        lons = []
        lats = []
        
        for name, ds in datasets.items():
            if hasattr(ds, 'lon') and hasattr(ds, 'lat'):
                lons.extend([ds.lon.min().values, ds.lon.max().values])
                lats.extend([ds.lat.min().values, ds.lat.max().values])
        
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
        for name, ds in tqdm(datasets.items(), desc="Aligning datasets"):
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
        """Apply quality control flags"""
        logger.info("Applying quality control flags")
        
        # Apply VI quality control
        if 'vi_quality' in datasets and 'ndvi' in datasets:
            # Create mask for good quality data
            good_quality = datasets['vi_quality'] < 2  # Example threshold
            datasets['ndvi'] = datasets['ndvi'].where(good_quality)
            datasets['evi'] = datasets['evi'].where(good_quality)
            logger.info("Applied VI quality control")
        
        # Apply FPAR/LAI quality control
        if 'fpar_lai_qc' in datasets:
            good_quality = datasets['fpar_lai_qc'] < 2  # Example threshold
            datasets['fpar'] = datasets['fpar'].where(good_quality)
            datasets['lai'] = datasets['lai'].where(good_quality)
            logger.info("Applied FPAR/LAI quality control")
        
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
        os.makedirs('./data/visualizations', exist_ok=True)
        
        # Plot biomass distribution
        if 'gedi' in datasets:
            plt.figure(figsize=(10, 6))
            sns.histplot(datasets['gedi'].values.flatten(), bins=50, kde=True)
            plt.title('Biomass Distribution')
            plt.xlabel('Biomass (Mg/ha)')
            plt.ylabel('Frequency')
            plt.savefig('./data/visualizations/biomass_distribution.png')
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
            plt.savefig('./data/visualizations/ndvi_timeseries.png')
            plt.close()
        
        # Plot spatial coverage
        if 'ndvi' in datasets:
            plt.figure(figsize=(10, 8))
            # Use the first time step
            datasets['ndvi'].isel(time=0).plot(cmap='viridis')
            plt.title('Spatial Coverage of NDVI')
            plt.savefig('./data/visualizations/spatial_coverage.png')
            plt.close()
        
        logger.info("Visualizations saved to ./data/visualizations/")


# 1.3 Preprocessing Execution
def validate_preprocessed_data(sequences, targets):
    """Validate the preprocessed data"""
    logger.info("Validating preprocessed data")
    
    # Check for NaN values
    nan_sequences = np.sum(np.isnan(sequences))
    nan_targets = np.sum(np.isnan(targets))
    
    logger.info(f"NaN values in sequences: {nan_sequences} ({nan_sequences/sequences.size*100:.2f}%)")
    logger.info(f"NaN values in targets: {nan_targets} ({nan_targets/targets.size*100:.2f}%)")
    
    # Check data ranges
    logger.info(f"Sequence range: [{np.nanmin(sequences):.4f}, {np.nanmax(sequences):.4f}]")
    logger.info(f"Target range: [{np.nanmin(targets):.4f}, {np.nanmax(targets):.4f}]")
    
    # Check for infinite values
    inf_sequences = np.sum(np.isinf(sequences))
    inf_targets = np.sum(np.isinf(targets))
    
    if inf_sequences > 0 or inf_targets > 0:
        logger.warning(f"Infinite values found - sequences: {inf_sequences}, targets: {inf_targets}")
    
    logger.info("Data validation complete.")

def main():
    try:
        # Initialize processor
        processor = DataProcessor(data_dir='./dataset')
        
        # Load local datasets
        logger.info("Loading local datasets...")
        local_datasets = processor.load_local_datasets()
        
        if not local_datasets:
            logger.error("No local datasets loaded. Exiting.")
            return
        
        # Fetch NASA data for a region (example: India)
        logger.info("Fetching NASA data...")
        india_bbox = [68.1, 6.7, 97.4, 37.1]  # [min_lon, min_lat, max_lon, max_lat]
        nasa_datasets = processor.fetch_nasa_data(india_bbox, "2019-01-01", "2023-12-31")
        
        # Combine all datasets
        all_datasets = {**local_datasets, **nasa_datasets}
        
        # Apply quality control
        logger.info("Applying quality control...")
        qc_datasets = processor.apply_quality_control(all_datasets)
        
        # Align datasets
        logger.info("Aligning datasets...")
        aligned_datasets, common_lons, common_lats = processor.align_datasets(qc_datasets)
        
        # Normalize data
        logger.info("Normalizing data...")
        normalized_datasets = processor.normalize_data(aligned_datasets)
        
        # Create sequences
        logger.info("Creating sequences...")
        sequences, targets = processor.create_sequences(normalized_datasets, seq_length=12)
        
        # Validate the preprocessed data
        logger.info("Validating preprocessed data...")
        validate_preprocessed_data(sequences, targets)
        
        # Create visualizations
        logger.info("Creating visualizations...")
        processor.visualize_data(aligned_datasets)
        
        # Create output directory
        os.makedirs('./data/processed', exist_ok=True)
        
        # Save preprocessed data
        np.save('./data/processed/sequences.npy', sequences)
        np.save('./data/processed/targets.npy', targets)
        np.save('./data/processed/common_lons.npy', common_lons)
        np.save('./data/processed/common_lats.npy', common_lats)
        
        # Save scalers for inference
        with open('./data/processed/scalers.pkl', 'wb') as f:
            pickle.dump(processor.scalers, f)
        
        logger.info(f"Preprocessing complete!")
        logger.info(f"Sequences shape: {sequences.shape}")
        logger.info(f"Targets shape: {targets.shape}")
        logger.info(f"Data saved to ./data/processed/")
        
    except Exception as e:
        logger.error(f"Error in preprocessing pipeline: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()