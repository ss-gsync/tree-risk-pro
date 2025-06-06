"""
Tree Risk Assessment Data Collection Module
Handles collecting and validating input data from various sensors
"""

import logging
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
import json
import time
from datetime import datetime
import pandas as pd
import concurrent.futures
from functools import partial
import urllib.request
import io

# Shared configuration class
@dataclass
class PipelineConfig:
    # Input/Output paths
    input_path: str = "/ttt/data/input"
    output_path: str = "/ttt/data/output"
    store_path: str = "/ttt/data/zarr"
    ml_path: str = "/ttt/data/ml"
    
    # Processing parameters
    tile_size: Tuple[int, int] = (1024, 1024)
    overlap_pixels: int = 128  # Increased overlap for better tree detection at boundaries
    target_resolution: Tuple[int, int] = (2048, 1536)  # Higher resolution for better detail
    batch_size: int = 16
    
    # Camera parameters
    camera_matrix: np.ndarray = field(default_factory=lambda: np.array([
        [1000.0, 0.0, 960.0],
        [0.0, 1000.0, 540.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64))
    
    distortion_coeffs: np.ndarray = field(default_factory=lambda: np.array([
        0.0, 0.0, 0.0, 0.0, 0.0
    ], dtype=np.float64))
    
    # Zarr configuration
    zarr_chunks: Tuple[int, int, int] = (1, 512, 512)  # Optimized for spatial access patterns

    # Redis configuration
    redis_host: str = "localhost"
    redis_ports: Dict[str, int] = field(default_factory=lambda: {
        'metadata': 6380,
        'queue': 6381,
        'results': 6382
    })
    
    # Tree detection enhancement parameters
    vegetation_enhancement: bool = True
    
    # ML pipeline configuration
    use_ml: bool = True
    run_ml_directly: bool = True
    yolo_confidence: float = 0.2 
    deepforest_score_threshold: float = 0.4
    contrast_enhancement: bool = True
    shadow_correction: bool = True
    
    # Performance parameters
    max_workers: int = 8  # Maximum number of worker threads for parallel processing
    retry_count: int = 3  # Number of retries for failed operations
    retry_delay: float = 1.0  # Delay between retries in seconds
    
    def __post_init__(self):
        """Validate configuration and create directories"""
        # Create directories if they don't exist
        for path_attr in ['input_path', 'output_path', 'store_path', 'ml_path']:
            path = getattr(self, path_attr)
            Path(path).mkdir(parents=True, exist_ok=True)
        
        # Validate configurations
        self._validate_config()
            
    def _validate_config(self):
        """Validate configuration parameters"""
        # Check that tile size makes sense
        if self.tile_size[0] <= 0 or self.tile_size[1] <= 0:
            raise ValueError(f"Invalid tile size: {self.tile_size}")
            
        # Check that overlap is reasonable
        if self.overlap_pixels < 0 or self.overlap_pixels >= min(self.tile_size):
            raise ValueError(f"Invalid overlap: {self.overlap_pixels}")
            
        # Check camera matrix is valid
        if self.camera_matrix.shape != (3, 3):
            raise ValueError(f"Camera matrix must be 3x3, got {self.camera_matrix.shape}")
            
        # Check distortion coefficients
        if len(self.distortion_coeffs) not in [4, 5, 8]:
            raise ValueError(f"Distortion coefficients must have 4, 5, or 8 elements")
            
        # Check worker count
        if self.max_workers <= 0:
            raise ValueError(f"Max workers must be positive, got {self.max_workers}")

class DataCollector:
    """Collects and validates input data from sensors"""
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    async def collect_data(self, mission_id: str, map_coords: Optional[Dict] = None) -> Dict:
        """Collect all sensor data for a mission with parallel processing"""
        self.logger.info(f"Starting data collection for mission {mission_id}")
        
        try:
            # If map_coords is provided, we'll use it regardless of mission directory
            if map_coords:
                # Process with map coordinates
                self.logger.info(f"Using map coordinates for {mission_id}")
                
                # Use the fetch_map_imagery method directly
                rgb_data = self.fetch_map_imagery(mission_id, map_coords)
                
                # Return with minimal data structure
                return {
                    'rgb': rgb_data,
                    'lidar': {'points': [], 'timestamps': [], 'metadata': {}},
                    'imu': {'timestamps': [], 'position': np.array([]), 'orientation': np.array([]), 'metadata': {}},
                    'metadata': {'mission_id': mission_id, 'map_coordinates': map_coords}
                }
            
            # For regular missions, check directory
            mission_path = Path(self.config.input_path) / mission_id
            if not mission_path.exists():
                self.logger.warning(f"Mission directory not found: {mission_path}")
                return self._create_empty_data()
                
            # Standard data collection
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                # Start all collection tasks
                rgb_future = executor.submit(self.collect_rgb_data, mission_id)
                lidar_future = executor.submit(self.collect_lidar_data, mission_id)
                imu_future = executor.submit(self.collect_imu_data, mission_id)
                metadata_future = executor.submit(self.collect_metadata, mission_id)
                
                # Wait for all tasks to complete
                rgb_data = rgb_future.result()
                lidar_data = lidar_future.result()
                imu_data = imu_future.result()
                metadata = metadata_future.result()
            
            # Validate data consistency
            self._validate_data_consistency(rgb_data, lidar_data, imu_data)
            
            return {
                'rgb': rgb_data,
                'lidar': lidar_data,
                'imu': imu_data,
                'metadata': metadata
            }
            
        except Exception as e:
            self.logger.error(f"Error collecting data for mission {mission_id}: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return self._create_empty_data()
            
    def _create_empty_data(self) -> Dict:
        """Create empty data structure for error cases"""
        return {
            'rgb': {'images': [], 'timestamps': [], 'metadata': []},
            'lidar': {'point_clouds': [], 'timestamps': [], 'metadata': []},
            'imu': {
                'timestamps': [],
                'position': np.array([]),
                'orientation': np.array([]),
                'metadata': {'sample_rate': 0, 'record_count': 0}
            },
            'metadata': {
                'mission_id': 'unknown',
                'timestamp': datetime.now(),
                'data_source': 'unknown',
                'error': 'Failed to collect data'
            }
        }
        
    def _validate_data_consistency(self, rgb_data: Dict, lidar_data: Dict, imu_data: Dict):
        """Validate that collected data is consistent"""
        # Check timestamps alignment
        if not rgb_data['timestamps'] or not lidar_data['timestamps'] or not imu_data['timestamps']:
            return  # Skip if any data is missing
            
        # Log summary of collected data
        self.logger.info(f"Collected {len(rgb_data['images'])} RGB images")
        self.logger.info(f"Collected {len(lidar_data['point_clouds'])} LiDAR point clouds")
        self.logger.info(f"Collected {len(imu_data['position'])} IMU records")
        
        # Check time spans
        if len(rgb_data['timestamps']) > 1 and len(lidar_data['timestamps']) > 1:
            rgb_timespan = (rgb_data['timestamps'][-1] - rgb_data['timestamps'][0]).total_seconds()
            lidar_timespan = (lidar_data['timestamps'][-1] - lidar_data['timestamps'][0]).total_seconds()
            
            # Log a warning if time spans are very different
            if abs(rgb_timespan - lidar_timespan) > max(rgb_timespan, lidar_timespan) * 0.5:
                self.logger.warning(f"RGB and LiDAR time spans differ significantly: "
                                  f"RGB: {rgb_timespan:.1f}s, LiDAR: {lidar_timespan:.1f}s")
        
    def fetch_map_imagery(self, mission_id: str, coords: Dict) -> Dict:
        """
        Fetch map imagery from the provided coordinates
        
        Args:
            mission_id: Identifier for the mission
            coords: Dict containing map coordinates with bounds, center, zoom
            
        Returns:
            Dict with RGB imagery data
        """
        self.logger.info(f"Fetching map imagery for coordinates: {coords}")
        
        # Get coordinates and view mode information
        bounds = coords.get('bounds')
        center = coords.get('center', [0, 0])
        zoom = coords.get('zoom', 18)
        is_3d_mode = coords.get('is3DMode', False)
        view_mode = coords.get('viewMode', '2D')
        
        if not bounds or len(bounds) != 2:
            self.logger.error("Invalid bounds in coordinates")
            return {'images': [], 'timestamps': [], 'metadata': []}
        
        # Coordinates for Google Static Map API
        sw = bounds[0]  # Southwest corner [lng, lat]
        ne = bounds[1]  # Northeast corner [lng, lat]
        
        # Try to find a captured map image from the frontend
        try:
            # Check for map image in the temp directory
            import os
            import glob
            ml_dir = self.config.ml_path
            
            # Look for any map image for this area
            map_files = glob.glob(os.path.join(ml_dir, f"{mission_id}_map*"))
            
            if map_files:
                # Use the first map file found
                capture_path = map_files[0]
                self.logger.info(f"Found map image at {capture_path}")
                image = cv2.imread(capture_path)
                if image is not None:
                    self.logger.info(f"Successfully loaded map image: {image.shape}")
                else:
                    raise ValueError("Failed to read map image")
            else:
                self.logger.error(f"No map image found in {ml_dir} for mission {mission_id}")
                self.logger.error(f"Available files: {os.listdir(ml_dir)}")
                raise ValueError("No map image available for processing")
            
            # Return with timestamp
            timestamp = datetime.now()
            return {
                'images': [image],
                'timestamps': [timestamp],
                'metadata': [{
                    'coords': coords,
                    'timestamp': timestamp,
                    'is_map_image': True,
                    'view_mode': view_mode,
                    'is_3d_mode': is_3d_mode
                }]
            }
            
        except Exception as e:
            self.logger.error(f"Failed to fetch map imagery: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {'images': [], 'timestamps': [], 'metadata': []}
    
    def collect_rgb_data(self, mission_id: str, map_coords: Optional[Dict] = None) -> Dict:
        """Collect RGB imagery data with retry mechanism"""
        # If map coordinates are provided, fetch map imagery instead
        if map_coords:
            return self.fetch_map_imagery(mission_id, map_coords)
            
        # Otherwise continue with normal file-based collection
        rgb_path = Path(self.config.input_path) / mission_id / 'rgb'
        
        images = []
        timestamps = []
        metadata = []
        
        if not rgb_path.exists():
            self.logger.warning(f"RGB directory not found: {rgb_path}")
            return {'images': [], 'timestamps': [], 'metadata': []}
            
        # Find all image files with different extensions
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
            image_files.extend(list(rgb_path.glob(f"*{ext}")))
            
        if not image_files:
            self.logger.warning(f"No image files found in {rgb_path}")
            return {'images': [], 'timestamps': [], 'metadata': []}
            
        # Sort image files
        image_files = sorted(image_files)
        
        # Process each image with retries
        for img_path in image_files:
            for retry in range(self.config.retry_count + 1):
                try:
                    # Read the image
                    img = cv2.imread(str(img_path))
                    if img is None:
                        raise ValueError(f"Failed to read image: {img_path}")
                        
                    # Convert BGR to RGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        
                    # Extract timestamp from filename or metadata
                    timestamp = datetime.fromtimestamp(img_path.stat().st_mtime)
                    
                    # Try to get more accurate timestamp from EXIF if available
                    try:
                        exif_timestamp = self._extract_exif_timestamp(img_path)
                        if exif_timestamp:
                            timestamp = exif_timestamp
                    except Exception as e:
                        # Continue with file timestamp if exif extraction fails
                        pass
                    
                    images.append(img)
                    timestamps.append(timestamp)
                    metadata.append({
                        'filename': img_path.name,
                        'size_bytes': img_path.stat().st_size,
                        'resolution': img.shape[:2],
                        'path': str(img_path)
                    })
                    
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    if retry < self.config.retry_count:
                        self.logger.warning(f"Error processing image {img_path}, retrying ({retry+1}/{self.config.retry_count}): {e}")
                        time.sleep(self.config.retry_delay)
                    else:
                        self.logger.error(f"Failed to process image {img_path} after {self.config.retry_count} retries: {e}")
            
        self.logger.info(f"Collected {len(images)} RGB images for mission {mission_id}")
            
        return {
            'images': images,
            'timestamps': timestamps,
            'metadata': metadata
        }
        
    def _extract_exif_timestamp(self, img_path: Path) -> Optional[datetime]:
        """Extract timestamp from EXIF metadata if available"""
        try:
            # Try to use PIL for EXIF extraction
            from PIL import Image
            from PIL.ExifTags import TAGS
            
            with Image.open(img_path) as img:
                if hasattr(img, '_getexif') and img._getexif():
                    exif = {TAGS.get(tag, tag): value for tag, value in img._getexif().items()}
                    
                    # Look for DateTime or DateTimeOriginal tags
                    for tag in ['DateTime', 'DateTimeOriginal']:
                        if tag in exif:
                            # Parse datetime string (format: 'YYYY:MM:DD HH:MM:SS')
                            dt_str = exif[tag]
                            dt = datetime.strptime(dt_str, '%Y:%m:%d %H:%M:%S')
                            return dt
                            
            return None
            
        except ImportError:
            # PIL not available
            return None
        except Exception:
            # Any other error in EXIF extraction
            return None
        
    def collect_lidar_data(self, mission_id: str) -> Dict:
        """Collect LiDAR point cloud data with retry mechanism"""
        lidar_path = Path(self.config.input_path) / mission_id / 'lidar'
        
        point_clouds = []
        timestamps = []
        metadata = []
        
        if not lidar_path.exists():
            self.logger.warning(f"LiDAR directory not found: {lidar_path}")
            return {'point_clouds': [], 'timestamps': [], 'metadata': []}
        
        # Look for different LiDAR file formats
        lidar_files = []
        for ext in ['*.npy', '*.pcd', '*.las', '*.laz', '*.ply', '*.xyz']:
            lidar_files.extend(list(lidar_path.glob(ext)))
            
        if not lidar_files:
            self.logger.warning(f"No LiDAR files found in {lidar_path}")
            return {'point_clouds': [], 'timestamps': [], 'metadata': []}
        
        # Sort lidar files
        lidar_files = sorted(lidar_files)
        
        # Process each file with retries
        for pc_file in lidar_files:
            for retry in range(self.config.retry_count + 1):
                try:
                    # Get corresponding metadata file
                    meta_file = pc_file.with_suffix('.json')
                    
                    # Load points based on file format
                    if pc_file.suffix == '.npy':
                        points = np.load(pc_file)
                    elif pc_file.suffix == '.pcd':
                        # Use Open3D for PCD files if available
                        try:
                            import open3d as o3d
                            pcd = o3d.io.read_point_cloud(str(pc_file))
                            points = np.asarray(pcd.points)
                        except ImportError:
                            self.logger.warning("open3d not available for reading PCD files")
                            raise ImportError("open3d not available for reading PCD files")
                    elif pc_file.suffix in ['.las', '.laz']:
                        # Use laspy for LAS/LAZ files if available
                        try:
                            import laspy
                            las = laspy.read(pc_file)
                            points = np.vstack((las.x, las.y, las.z)).transpose()
                        except ImportError:
                            self.logger.warning("laspy not available for reading LAS/LAZ files")
                            raise ImportError("laspy not available for reading LAS/LAZ files")
                    elif pc_file.suffix == '.ply':
                        # Use Open3D for PLY files if available
                        try:
                            import open3d as o3d
                            pcd = o3d.io.read_point_cloud(str(pc_file))
                            points = np.asarray(pcd.points)
                        except ImportError:
                            self.logger.warning("open3d not available for reading PLY files")
                            raise ImportError("open3d not available for reading PLY files")
                    elif pc_file.suffix == '.xyz':
                        # Simple XYZ format (space or comma separated)
                        points = np.loadtxt(pc_file, delimiter=None)
                    else:
                        self.logger.warning(f"Unsupported point cloud format: {pc_file}")
                        continue
                    
                    # Use metadata file if exists, otherwise create basic metadata
                    scan_meta = {}
                    if meta_file.exists():
                        with open(meta_file) as f:
                            scan_meta = json.load(f)
                            scan_timestamp = datetime.fromtimestamp(scan_meta.get('timestamp', pc_file.stat().st_mtime))
                    else:
                        scan_timestamp = datetime.fromtimestamp(pc_file.stat().st_mtime)
                    
                    # Ensure points are in the right shape
                    if len(points.shape) == 1:
                        # Single point, reshape
                        points = points.reshape(1, -1)
                    
                    if points.shape[1] < 3:
                        # Add zero z-coordinate if needed
                        points = np.column_stack([points, np.zeros(points.shape[0])])
                    
                    point_clouds.append(points)
                    timestamps.append(scan_timestamp)
                    metadata.append({
                        'filename': pc_file.name,
                        'size_bytes': pc_file.stat().st_size,
                        'point_count': len(points),
                        'path': str(pc_file),
                        **scan_meta  # Include any additional metadata from JSON
                    })
                    
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    if retry < self.config.retry_count:
                        self.logger.warning(f"Error processing point cloud {pc_file}, retrying ({retry+1}/{self.config.retry_count}): {e}")
                        time.sleep(self.config.retry_delay)
                    else:
                        self.logger.error(f"Failed to process point cloud {pc_file} after {self.config.retry_count} retries: {e}")
        
        self.logger.info(f"Collected {len(point_clouds)} LiDAR point clouds for mission {mission_id}")
            
        return {
            'point_clouds': point_clouds,
            'timestamps': timestamps,
            'metadata': metadata
        }
                
    def collect_imu_data(self, mission_id: str) -> Dict:
        """Collect IMU sensor data with retry mechanism"""
        imu_path = Path(self.config.input_path) / mission_id / 'imu'
        
        if not imu_path.exists():
            self.logger.warning(f"IMU directory not found: {imu_path}")
            return {
                'timestamps': [],
                'position': np.array([]),
                'orientation': np.array([]),
                'metadata': {'sample_rate': 0, 'record_count': 0}
            }
        
        # Look for IMU data in different formats
        imu_file = None
        for format in ['imu_data.csv', 'pose.csv', 'imu.json', 'trajectory.csv', 'position.csv', 'gps.csv']:
            if (imu_path / format).exists():
                imu_file = imu_path / format
                break
                
        if imu_file is None:
            self.logger.warning(f"No IMU data files found in {imu_path}")
            return {
                'timestamps': [],
                'position': np.array([]),
                'orientation': np.array([]),
                'metadata': {'sample_rate': 0, 'record_count': 0}
            }
            
        # Process with retries
        for retry in range(self.config.retry_count + 1):
            try:
                # Process based on file format
                if imu_file.suffix == '.csv':
                    # Read IMU data from CSV format
                    imu_data = pd.read_csv(imu_file)
                    
                    # Check for required columns
                    required_cols = [
                        ('timestamp', ['timestamp', 'time', 'date_time']),
                        ('latitude', ['latitude', 'lat', 'y']),
                        ('longitude', ['longitude', 'lon', 'long', 'x']),
                        ('altitude', ['altitude', 'alt', 'height', 'z']),
                        ('roll', ['roll', 'r']),
                        ('pitch', ['pitch', 'p']),
                        ('yaw', ['yaw', 'y', 'heading'])
                    ]
                    
                    # Map available columns
                    column_map = {}
                    for req_col, alternatives in required_cols:
                        for alt in alternatives:
                            if alt in imu_data.columns:
                                column_map[req_col] = alt
                                break
                        if req_col not in column_map:
                            self.logger.warning(f"Required column {req_col} not found in IMU data")
                            
                    # Convert timestamps to datetime
                    if 'timestamp' in column_map:
                        timestamps = pd.to_datetime(imu_data[column_map['timestamp']])
                    else:
                        # Generate synthetic timestamps if not available
                        timestamps = pd.date_range(
                            start=datetime.now(),
                            periods=len(imu_data),
                            freq='10ms'
                        )
                    
                    # Extract position and orientation
                    position_cols = [column_map.get(c) for c in ['latitude', 'longitude', 'altitude'] if c in column_map]
                    orientation_cols = [column_map.get(c) for c in ['roll', 'pitch', 'yaw'] if c in column_map]
                    
                    if position_cols:
                        position = imu_data[position_cols].values
                    else:
                        position = np.zeros((len(imu_data), 3))
                        
                    if orientation_cols:
                        orientation = imu_data[orientation_cols].values
                    else:
                        orientation = np.zeros((len(imu_data), 3))
                    
                elif imu_file.suffix == '.json':
                    # Read from JSON format
                    with open(imu_file) as f:
                        imu_json = json.load(f)
                        
                    if isinstance(imu_json, list):
                        # List of IMU records
                        timestamps = [datetime.fromisoformat(record.get('timestamp', '')) 
                                    for record in imu_json if 'timestamp' in record]
                        
                        position = np.array([
                            [record.get('latitude', 0), record.get('longitude', 0), record.get('altitude', 0)]
                            for record in imu_json
                        ])
                        
                        orientation = np.array([
                            [record.get('roll', 0), record.get('pitch', 0), record.get('yaw', 0)]
                            for record in imu_json
                        ])
                    else:
                        # Single IMU record or different structure
                        self.logger.warning(f"Unexpected JSON format in {imu_file}")
                        return {
                            'timestamps': [],
                            'position': np.array([]),
                            'orientation': np.array([]),
                            'metadata': {'sample_rate': 0, 'record_count': 0}
                        }
                else:
                    self.logger.warning(f"Unsupported IMU data format: {imu_file}")
                    return {
                        'timestamps': [],
                        'position': np.array([]),
                        'orientation': np.array([]),
                        'metadata': {'sample_rate': 0, 'record_count': 0}
                    }
                
                # Calculate sample rate safely
                time_diff = (timestamps.max() - timestamps.min()).total_seconds()
                if time_diff <= 0:
                    # Handle case where timestamps are identical or invalid
                    self.logger.warning(f"Invalid time difference in IMU data for mission {mission_id}")
                    sample_rate = 100.0  # Default to 100Hz
                else:
                    sample_rate = (len(timestamps) - 1) / time_diff
                
                self.logger.info(f"Collected IMU data with {len(timestamps)} records for mission {mission_id}")
                
                return {
                    'timestamps': timestamps,
                    'position': position,
                    'orientation': orientation,
                    'metadata': {
                        'sample_rate': sample_rate,
                        'record_count': len(timestamps),
                        'file_path': str(imu_file)
                    }
                }
                
            except Exception as e:
                if retry < self.config.retry_count:
                    self.logger.warning(f"Error processing IMU data, retrying ({retry+1}/{self.config.retry_count}): {e}")
                    time.sleep(self.config.retry_delay)
                else:
                    self.logger.error(f"Failed to process IMU data after {self.config.retry_count} retries: {e}")
                    import traceback
                    self.logger.error(traceback.format_exc())
                
        # Return empty data if all retries fail
        return {
            'timestamps': [],
            'position': np.array([]),
            'orientation': np.array([]),
            'metadata': {'sample_rate': 0, 'record_count': 0, 'error': str(e)}
        }
            
    def collect_metadata(self, mission_id: str) -> Dict:
        """Collect mission metadata with retry mechanism"""
        metadata_path = Path(self.config.input_path) / mission_id / 'metadata.json'
        mission_info_path = Path(self.config.input_path) / mission_id / 'mission_info.json'
        config_path = Path(self.config.input_path) / mission_id / 'config.json'
        
        # Try multiple metadata file locations
        metadata_files = [
            metadata_path,
            mission_info_path,
            config_path
        ]
        
        metadata = {
            'mission_id': mission_id,
            'timestamp': datetime.now().isoformat(),
            'data_source': 'unknown'
        }
        
        # Try each possible metadata file
        for meta_file in metadata_files:
            if meta_file.exists():
                for retry in range(self.config.retry_count + 1):
                    try:
                        with open(meta_file) as f:
                            file_metadata = json.load(f)
                            
                        # Merge with existing metadata
                        metadata.update(file_metadata)
                        
                        # Ensure required fields
                        metadata['mission_id'] = mission_id
                        if 'timestamp' not in metadata:
                            metadata['timestamp'] = datetime.now().isoformat()
                            
                        self.logger.info(f"Loaded metadata from {meta_file}")
                        break  # Success, exit retry loop
                        
                    except Exception as e:
                        if retry < self.config.retry_count:
                            self.logger.warning(f"Error reading metadata file {meta_file}, retrying ({retry+1}/{self.config.retry_count}): {e}")
                            time.sleep(self.config.retry_delay)
                        else:
                            self.logger.error(f"Failed to read metadata file {meta_file} after {self.config.retry_count} retries: {e}")
        
        # Try to get weather data if not in metadata
        if 'weather' not in metadata:
            weather_path = Path(self.config.input_path) / mission_id / 'weather.json'
            if weather_path.exists():
                try:
                    with open(weather_path) as f:
                        weather_data = json.load(f)
                    metadata['weather'] = weather_data
                except Exception as e:
                    self.logger.warning(f"Error reading weather data: {e}")
        
        return metadata


# For testing the module independently
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    import sys
    import asyncio
    
    # Create configuration
    config = PipelineConfig()
    
    # Create data collector
    collector = DataCollector(config)
    
    # Process mission
    async def main():
        mission_id = "test_mission"
        if len(sys.argv) > 1:
            mission_id = sys.argv[1]
            
        data = await collector.collect_data(mission_id)
        print(f"Collected data summary:")
        print(f"RGB images: {len(data['rgb']['images'])}")
        print(f"LiDAR point clouds: {len(data['lidar']['point_clouds'])}")
        print(f"IMU records: {len(data['imu']['position'])}")
        
    # Run with asyncio
    asyncio.run(main())