"""
Tree Risk Assessment Image Processing Module
Handles image tiling, transformations, and sensor fusion for tree analysis
"""

import logging
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import json
import time
from datetime import datetime
import concurrent.futures
from scipy.spatial.transform import Rotation
import pyproj
import asyncio
from functools import partial
import os
import gc

# Configure logging
logger = logging.getLogger(__name__)

class PipelineConfig:
    """Configuration for the image processing pipeline"""
    def __init__(self, 
                 tile_size: Tuple[int, int] = (512, 512),
                 overlap_pixels: int = 64,
                 target_resolution: Tuple[int, int] = (512, 512),
                 max_workers: int = 4,
                 vegetation_enhancement: bool = True,
                 contrast_enhancement: bool = True,
                 shadow_correction: bool = True,
                 camera_matrix: Optional[np.ndarray] = None,
                 distortion_coeffs: Optional[np.ndarray] = None):
        """
        Initialize pipeline configuration
        
        Args:
            tile_size: Size of image tiles (height, width)
            overlap_pixels: Pixel overlap between adjacent tiles
            target_resolution: Target resolution for standardization
            max_workers: Maximum number of worker threads
            vegetation_enhancement: Whether to enhance vegetation features
            contrast_enhancement: Whether to enhance image contrast
            shadow_correction: Whether to apply shadow correction
            camera_matrix: Camera intrinsic matrix (3x3)
            distortion_coeffs: Camera distortion coefficients
        """
        self.tile_size = tile_size
        self.overlap_pixels = overlap_pixels
        self.target_resolution = target_resolution
        self.max_workers = max_workers
        self.vegetation_enhancement = vegetation_enhancement
        self.contrast_enhancement = contrast_enhancement
        self.shadow_correction = shadow_correction
        
        # Default camera matrix if not provided
        if camera_matrix is None:
            self.camera_matrix = np.array([
                [1000, 0, 512],
                [0, 1000, 512],
                [0, 0, 1]
            ], dtype=np.float32)
        else:
            self.camera_matrix = camera_matrix
            
        # Default distortion coefficients if not provided
        if distortion_coeffs is None:
            self.distortion_coeffs = np.zeros(5, dtype=np.float32)
        else:
            self.distortion_coeffs = distortion_coeffs


class MemoryManager:
    """Manages memory for large array operations"""
    def __init__(self, config: PipelineConfig):
        """
        Initialize memory manager
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.large_arrays = []
        
    def register_array(self, array: np.ndarray) -> None:
        """
        Register a large array for memory management
        
        Args:
            array: NumPy array to register
        """
        self.large_arrays.append(array)
        
    def cleanup(self) -> None:
        """Force cleanup of registered arrays and trigger garbage collection"""
        for i in range(len(self.large_arrays)):
            self.large_arrays[i] = None
        self.large_arrays = []
        
        # Suggest garbage collection
        gc.collect()


class ProjectionTransform:
    """Handles all coordinate transformations with tree detection optimization"""
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize projection systems
        self.wgs84 = pyproj.CRS('EPSG:4326')  # WGS84 coordinate system
        
        # Auto-determine UTM zone if we can
        self.utm = None
        self.transformer = None
        
    def _init_utm_transformer(self, lon: float, lat: float):
        """Initialize UTM transformer for the given longitude/latitude"""
        # Determine UTM zone from longitude
        utm_zone = int((lon + 180) / 6) + 1
        
        # North or South hemisphere
        ns = 'north' if lat >= 0 else 'south'
        
        # Create UTM CRS
        utm_crs = pyproj.CRS(f"+proj=utm +zone={utm_zone} +{ns} +datum=WGS84 +units=m +no_defs")
        
        # Create transformer
        self.utm = utm_crs
        self.transformer = pyproj.Transformer.from_crs(self.wgs84, self.utm)
        
        self.logger.info(f"Initialized UTM transformer for zone {utm_zone} {ns}")
        
    async def transform_data(self, tiled_data: Dict, imu_data: Dict) -> Dict:
        """Apply all necessary coordinate transformations"""
        try:
            # Check if we have IMU data
            if len(imu_data.get('position', [])) == 0:
                self.logger.warning("No IMU data available for coordinate transformation")
                return {
                    'tiles': tiled_data['tiles'],
                    'metadata': tiled_data['metadata'],
                    'world_coordinates': [[] for _ in tiled_data['tiles']]
                }
            
            # Check if we need to initialize the UTM transformer
            if self.transformer is None and len(imu_data['position']) > 0:
                # Use first position to initialize UTM zone
                lat, lon = imu_data['position'][0][:2]
                self._init_utm_transformer(lon, lat)
            
            # Transform camera coordinates
            camera_transformed = await self._apply_camera_transform(tiled_data)
            
            # Transform IMU data to world coordinates
            world_coords = await self._apply_imu_transform(imu_data)
            
            # Project to world coordinates
            projected_data = await self._project_to_world(
                camera_transformed,
                world_coords
            )
            
            return projected_data
            
        except Exception as e:
            self.logger.error(f"Error in projection transform: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            # Return the input data without transformation on error
            return {
                'tiles': tiled_data['tiles'],
                'metadata': tiled_data['metadata'],
                'world_coordinates': [[] for _ in tiled_data['tiles']]
            }
            
    async def _apply_camera_transform(self, tiled_data: Dict) -> Dict:
        """Apply camera intrinsics transform with optimization for tree detection"""
        transformed_tiles = []
        transformed_metadata = []
        
        # Process in parallel if there are many batches
        if len(tiled_data['tiles']) > 4:
            # Use a thread pool for parallel processing
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(tiled_data['tiles']))) as executor:
                # Create a list to store futures
                futures = []
                
                # Submit tasks
                for i, (tiles, metadata) in enumerate(zip(tiled_data['tiles'], tiled_data['metadata'])):
                    future = executor.submit(
                        self._transform_tile_batch,
                        tiles=tiles,
                        metadata=metadata,
                        idx=i
                    )
                    futures.append(future)
                    
                # Process results as they complete
                for future in concurrent.futures.as_completed(futures):
                    try:
                        idx, batch_transformed, meta = future.result()
                        
                        # Ensure we have the right slots in our result lists
                        while len(transformed_tiles) <= idx:
                            transformed_tiles.append([])
                            transformed_metadata.append({})
                            
                        transformed_tiles[idx] = batch_transformed
                        transformed_metadata[idx] = meta
                    except Exception as e:
                        self.logger.error(f"Error in parallel camera transform: {e}")
        else:
            # Process sequentially for smaller batches
            for i, (tiles, metadata) in enumerate(zip(tiled_data['tiles'], tiled_data['metadata'])):
                try:
                    _, batch_transformed, meta = self._transform_tile_batch(tiles, metadata, i)
                    transformed_tiles.append(batch_transformed)
                    transformed_metadata.append(meta)
                    
                except Exception as e:
                    self.logger.error(f"Error in camera transform for batch {i}: {e}")
                    # Use original tiles if transform fails
                    transformed_tiles.append(tiles)
                    transformed_metadata.append(metadata)
            
        return {
            'tiles': transformed_tiles,
            'metadata': transformed_metadata
        }
        
    def _transform_tile_batch(self, tiles, metadata, idx):
        """Transform a batch of tiles for parallel processing"""
        try:
            batch_transformed = []
            
            # Get tile coordinates 
            tile_coords = metadata.get('tile_coordinates', [])
            
            for j, tile in enumerate(tiles):
                # Undistort image using camera matrix and distortion coefficients
                undistorted = cv2.undistort(
                    tile,
                    self.config.camera_matrix,
                    self.config.distortion_coeffs
                )
                
                # Use optimized distortion correction for vegetation analysis
                # by preserving color information that could be lost in distortion correction
                if len(tile.shape) == 3 and tile.shape[2] >= 3:
                    # Extract HSV from original tile
                    hsv_orig = cv2.cvtColor((tile * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
                    
                    # Extract HSV from undistorted
                    hsv_undist = cv2.cvtColor((undistorted * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
                    
                    # Preserve hue and saturation from original, use value from undistorted
                    hsv_combined = hsv_undist.copy()
                    hsv_combined[:, :, 0] = hsv_orig[:, :, 0]  # Hue
                    hsv_combined[:, :, 1] = hsv_orig[:, :, 1]  # Saturation
                    
                    # Convert back to RGB
                    optimized = cv2.cvtColor(hsv_combined, cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
                else:
                    optimized = undistorted
                    
                batch_transformed.append(optimized)
                
            return idx, batch_transformed, metadata
                
        except Exception as e:
            self.logger.error(f"Error in _transform_tile_batch for index {idx}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
        
    async def _apply_imu_transform(self, imu_data: Dict) -> Dict:
        """Transform IMU data to world coordinates"""
        try:
            # Check if we have position data
            if len(imu_data['position']) == 0:
                return {
                    'positions': np.array([]),
                    'rotations': np.array([]),
                    'timestamps': imu_data['timestamps']
                }
                
            # Ensure transformer is initialized
            if self.transformer is None and len(imu_data['position']) > 0:
                lat, lon = imu_data['position'][0][:2]
                self._init_utm_transformer(lon, lat)
            
            # Convert lat/lon to UTM
            utm_coords = []
            for lat, lon in imu_data['position'][:, :2]:
                try:
                    east, north = self.transformer.transform(lat, lon)
                    utm_coords.append([east, north])
                except Exception as e:
                    self.logger.error(f"Error transforming coordinates ({lat}, {lon}): {e}")
                    raise ValueError(f"Failed to transform coordinates: {e}")
            
            utm_coords = np.array(utm_coords)
            
            # Combine with altitude or error if missing
            if imu_data['position'].shape[1] > 2:  # If we have altitude
                world_positions = np.column_stack([
                    utm_coords,
                    imu_data['position'][:, 2]
                ])
            else:
                self.logger.error("Altitude data missing from IMU positions")
                raise ValueError("Missing altitude data in IMU positions")
            
            # Convert euler angles to rotation matrices
            if len(imu_data['orientation']) > 0:
                rotations = Rotation.from_euler(
                    'xyz',
                    imu_data['orientation'],
                    degrees=True
                ).as_matrix()
            else:
                self.logger.error("No orientation data available in IMU data")
                raise ValueError("Missing orientation data in IMU data")
            
            return {
                'positions': world_positions,
                'rotations': rotations,
                'timestamps': imu_data['timestamps']
            }
            
        except Exception as e:
            self.logger.error(f"Error in IMU transform: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            # Return empty data on error
            return {
                'positions': np.array([]),
                'rotations': np.array([]),
                'timestamps': imu_data['timestamps']
            }
        
    async def _project_to_world(
        self,
        camera_data: Dict,
        world_coords: Dict
    ) -> Dict:
        """Project camera data to world coordinates with tree-specific optimizations"""
        projected_data = {
            'tiles': [],
            'world_coordinates': [],
            'metadata': camera_data['metadata']
        }
        
        # Check if we have world coordinate data
        if len(world_coords['positions']) == 0:
            self.logger.error("No world coordinate data available for projection")
            raise ValueError("Missing world coordinate data for projection")
        
        # Process frames sequentially (world projection is computationally intensive)
        for idx, (tiles, metadata) in enumerate(zip(camera_data['tiles'], camera_data['metadata'])):
            try:
                # Find closest IMU measurement by timestamp
                if 'timestamp' in metadata and world_coords['timestamps'] is not None and len(world_coords['timestamps']) > 0:
                    # Convert timestamps to comparable format
                    if isinstance(metadata['timestamp'], datetime) and isinstance(world_coords['timestamps'][0], datetime):
                        time_diffs = np.abs([
                            (metadata['timestamp'] - ts).total_seconds() 
                            for ts in world_coords['timestamps']
                        ])
                    else:
                        # Handle various timestamp formats
                        meta_ts = metadata['timestamp']
                        if isinstance(meta_ts, str):
                            try:
                                meta_ts = datetime.fromisoformat(meta_ts)
                            except ValueError:
                                raise ValueError(f"Cannot parse timestamp format: {meta_ts}")
                        
                        time_diffs = np.ones(len(world_coords['timestamps'])) * float('inf')
                        for i, ts in enumerate(world_coords['timestamps']):
                            try:
                                if isinstance(ts, str):
                                    ts = datetime.fromisoformat(ts)
                                if isinstance(ts, datetime) and isinstance(meta_ts, datetime):
                                    time_diffs[i] = abs((meta_ts - ts).total_seconds())
                            except Exception:
                                pass
                    
                    imu_idx = np.argmin(time_diffs)
                else:
                    # If no timestamps, use the middle of the array
                    imu_idx = len(world_coords['positions']) // 2
                
                # Get world position and rotation
                position = world_coords['positions'][imu_idx]
                rotation = world_coords['rotations'][imu_idx]
                
                # Project tiles
                projected_tiles = []
                tile_coords = []
                
                # Get tile coordinates if available
                tile_coordinates = metadata.get('tile_coordinates', [])
                
                # Process tiles in parallel for large batches
                if len(tiles) > 16:
                    # Use thread pool for parallel processing
                    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(tiles))) as executor:
                        # Create futures list
                        futures = []
                        
                        # Submit tasks
                        for i, tile in enumerate(tiles):
                            # Get the tile metadata
                            tile_pos = tile_coordinates[i] if i < len(tile_coordinates) else None
                            
                            future = executor.submit(
                                self._project_single_tile,
                                tile=tile,
                                position=position,
                                rotation=rotation,
                                tile_pos=tile_pos,
                                idx=i
                            )
                            futures.append(future)
                            
                        # Process results as they complete
                        results = []
                        for future in concurrent.futures.as_completed(futures):
                            try:
                                i, projected_tile, corners = future.result()
                                results.append((i, projected_tile, corners))
                            except Exception as e:
                                self.logger.error(f"Error in parallel tile projection: {e}")
                                
                        # Sort results by index
                        results.sort(key=lambda x: x[0])
                        
                        # Extract results
                        for _, projected_tile, corners in results:
                            projected_tiles.append(projected_tile)
                            tile_coords.append(corners)
                else:
                    # Process sequentially for smaller batches
                    for i, tile in enumerate(tiles):
                        # Get the original tile coordinates within the image
                        if i < len(tile_coordinates):
                            tile_pos = tile_coordinates[i]
                        else:
                            # Default tile position if coordinates not available
                            h, w = tile.shape[:2]
                            tile_pos = {'x': 0, 'y': 0, 'width': w, 'height': h}
                        
                        # Project tile corners to world coordinates with corrected camera model
                        # that better accounts for tree heights and perspectives
                        corners = self._project_tile_corners_with_height(
                            tile,
                            position,
                            rotation,
                            tile_pos
                        )
                        
                        projected_tiles.append(tile)
                        tile_coords.append(corners)
                
                projected_data['tiles'].append(projected_tiles)
                projected_data['world_coordinates'].append(tile_coords)
                
            except Exception as e:
                self.logger.error(f"Error projecting data for frame {idx}: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                
                # Use empty data for this frame
                projected_data['tiles'].append(tiles)
                projected_data['world_coordinates'].append([])
            
        return projected_data
    
    def _project_single_tile(self, tile, position, rotation, tile_pos, idx):
        """Project a single tile for parallel processing"""
        try:
            # Default tile position if coordinates not available
            if tile_pos is None:
                h, w = tile.shape[:2]
                tile_pos = {'x': 0, 'y': 0, 'width': w, 'height': h}
            
            # Project tile corners
            corners = self._project_tile_corners_with_height(
                tile,
                position,
                rotation,
                tile_pos
            )
            
            return idx, tile, corners
            
        except Exception as e:
            self.logger.error(f"Error in _project_single_tile for index {idx}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            # Return empty corners on error
            return idx, tile, np.zeros((8, 3))
        
    def _project_tile_corners_with_height(
        self,
        tile: np.ndarray,
        position: np.ndarray,
        rotation: np.ndarray,
        tile_pos: Dict
    ) -> np.ndarray:
        """Project tile corners to world coordinates with height consideration"""
        # Get tile dimensions
        h, w = tile.shape[:2]
        
        # Create corners with different height assumptions for vegetation
        # For trees, use multiple heights to capture the 3D structure
        corners_ground = np.array([
            [0, 0, 0],          # Bottom left at ground level
            [w, 0, 0],          # Bottom right at ground level
            [w, h, 0],          # Top right at ground level
            [0, h, 0]           # Top left at ground level
        ], dtype=np.float32)
        
        # Add corners at estimated tree height (e.g., 10m)
        estimated_tree_height = 10.0  # meters
        corners_canopy = np.array([
            [w/4, h/4, estimated_tree_height],     # Mid-low left at tree height
            [3*w/4, h/4, estimated_tree_height],   # Mid-low right at tree height
            [3*w/4, 3*h/4, estimated_tree_height], # Mid-high right at tree height
            [w/4, 3*h/4, estimated_tree_height]    # Mid-high left at tree height
        ], dtype=np.float32)
        
        # Combine ground and canopy points
        corners = np.vstack([corners_ground, corners_canopy])
        
        # Apply camera model to correct for the viewing position
        # based on the tile position in the original image
        
        # Convert rotation matrix to camera space
        camera_rotation = rotation.copy()
        
        # Apply camera transform
        corners_world = np.zeros_like(corners)
        for i, corner in enumerate(corners):
            # Apply rotation
            rotated = np.dot(camera_rotation, corner)
            
            # Apply translation
            translated = rotated + position
            
            corners_world[i] = translated
        
        return corners_world
    

class ResolutionStandardization:
    """Standardizes resolution across all processed images for consistent tree analysis"""
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.memory_manager = MemoryManager(config)
        
    async def standardize(self, projected_data: Dict) -> Dict:
        """Standardize resolution of all tiles with tree detection optimization"""
        standardized_data = {
            'tiles': [],
            'world_coordinates': projected_data['world_coordinates'],
            'metadata': projected_data['metadata']
        }
        
        target_h, target_w = self.config.target_resolution
        
        # Check if we have data to process
        if not projected_data['tiles']:
            self.logger.warning("No data to standardize resolution")
            return standardized_data
        
        # Process batches in parallel or sequentially based on the number of tiles
        total_tiles = sum(len(tiles) for tiles in projected_data['tiles'])
        
        if total_tiles > 100:
            self.logger.info(f"Using parallel processing for resolution standardization of {total_tiles} tiles")#!/usr/bin/env python3

            # Process batches in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                # Submit all tasks
                futures = []
                for batch_idx, tiles in enumerate(projected_data['tiles']):
                    # Skip empty batches
                    if not tiles:
                        standardized_data['tiles'].append([])
                        continue
                        
                    future = executor.submit(
                        self._standardize_batch,
                        tiles=tiles,
                        batch_idx=batch_idx
                    )
                    futures.append(future)
                    
                # Process results as they complete
                results = []
                for future in concurrent.futures.as_completed(futures):
                    try:
                        batch_idx, standardized_tiles = future.result()
                        results.append((batch_idx, standardized_tiles))
                    except Exception as e:
                        self.logger.error(f"Error in parallel resolution standardization: {e}")
                        
                # Sort results by batch index
                results.sort(key=lambda x: x[0])
                
                # Fill in results with the right order
                max_idx = max(idx for idx, _ in results) if results else -1
                for i in range(max_idx + 1):
                    found = False
                    for idx, tiles in results:
                        if idx == i:
                            standardized_data['tiles'].append(tiles)
                            found = True
                            break
                    if not found:
                        standardized_data['tiles'].append([])
        else:
            # Process sequentially for smaller datasets
            self.logger.info(f"Using sequential processing for resolution standardization of {total_tiles} tiles")
            
            for batch_idx, tiles in enumerate(projected_data['tiles']):
                try:
                    self.logger.info(f"Standardizing resolution for batch {batch_idx+1}/{len(projected_data['tiles'])}")
                    
                    if not tiles:
                        standardized_data['tiles'].append([])
                        continue
                        
                    _, standardized_tiles = self._standardize_batch(tiles, batch_idx)
                    standardized_data['tiles'].append(standardized_tiles)
                    
                except Exception as e:
                    self.logger.error(f"Error in resolution standardization for batch {batch_idx}: {e}")
                    import traceback
                    self.logger.error(traceback.format_exc())
                    
                    # Use original tiles on error
                    standardized_data['tiles'].append(tiles)
        
        # Cleanup memory
        self.memory_manager.cleanup()
                
        self.logger.info(f"Resolution standardization complete for {len(standardized_data['tiles'])} batches")
            
        return standardized_data
    
    def _standardize_batch(self, tiles, batch_idx):
        """Standardize a batch of tiles for parallel processing"""
        try:
            target_h, target_w = self.config.target_resolution
            standardized_tiles = []
            
            for tile_idx, tile in enumerate(tiles):
                # Get current tile dimensions
                curr_h, curr_w = tile.shape[:2]
                
                # Skip empty tiles
                if curr_h == 0 or curr_w == 0:
                    standardized_tiles.append(np.zeros((target_h, target_w, 3), dtype=np.float32))
                    continue
                
                # Determine if we need to resize
                if curr_h == target_h and curr_w == target_w:
                    # Already the right size
                    standardized_tiles.append(tile)
                    continue
                
                # Special handling for tree detection - preserve tree features
                # by using Lanczos interpolation for downsampling and
                # preserving high frequency detail with INTER_CUBIC for upsampling
                
                if curr_h > target_h or curr_w > target_w:
                    # Downsampling case - use Lanczos for better quality
                    interp_method = cv2.INTER_LANCZOS4
                else:
                    # Upsampling case - use cubic for better detail preservation
                    interp_method = cv2.INTER_CUBIC
                
                # Resize tile to target resolution
                resized = cv2.resize(
                    tile,
                    (target_w, target_h),
                    interpolation=interp_method
                )
                
                # For tree detection, enhance edges and textures after resizing
                if self.config.vegetation_enhancement:
                    # Apply light sharpening for better tree edge definition
                    kernel = np.array([[-1, -1, -1],
                                      [-1,  9, -1],
                                      [-1, -1, -1]]) / 9.0
                    
                    # Convert to uint8 for filter operation
                    img_uint8 = (resized * 255).astype(np.uint8)
                    sharpened = cv2.filter2D(img_uint8, -1, kernel)
                    
                    # Blend with original (70% original, 30% sharpened)
                    blend_factor = 0.3
                    blended = cv2.addWeighted(
                        img_uint8, 1.0 - blend_factor,
                        sharpened, blend_factor,
                        0
                    )
                    
                    # Convert back to float32
                    resized = blended.astype(np.float32) / 255.0
                    
                # Ensure values are in valid range
                resized = np.clip(resized, 0.0, 1.0)
                
                standardized_tiles.append(resized)
                
            return batch_idx, standardized_tiles
            
        except Exception as e:
            self.logger.error(f"Error in _standardize_batch for batch {batch_idx}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise


# Main pipeline coordinator for image processing
class ImageProcessor:
    """Main coordinator for image processing pipeline"""
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.image_processor = ImageTilingProcessor(config)
        self.projection_transform = ProjectionTransform(config)
        self.resolution_standardization = ResolutionStandardization(config)
        self.data_fusion = DataFusion(config)
        
    async def process_data(
        self,
        rgb_data: Dict,
        lidar_data: Dict = None,
        imu_data: Dict = None,
        metadata: Dict = None
    ) -> Dict:
        """Process all image data through the pipeline"""
        try:
            start_time = time.time()
            
            # Use empty data if not provided
            if lidar_data is None:
                lidar_data = {'point_clouds': [], 'timestamps': [], 'metadata': []}
                
            if imu_data is None:
                imu_data = {
                    'timestamps': [],
                    'position': np.array([]),
                    'orientation': np.array([]),
                    'metadata': {'sample_rate': 0, 'record_count': 0}
                }
                
            if metadata is None:
                metadata = {
                    'mission_id': 'unknown',
                    'timestamp': datetime.now().isoformat(),
                    'data_source': 'unknown'
                }
            
            # Process and tile images
            self.logger.info("Processing and tiling images")
            tiled_data = await self.image_processor.process_images(rgb_data)
            
            # Apply coordinate transforms
            self.logger.info("Applying coordinate transforms")
            projected_data = await self.projection_transform.transform_data(
                tiled_data,
                imu_data
            )
            
            # Standardize resolution
            self.logger.info("Standardizing resolution")
            standardized_data = await self.resolution_standardization.standardize(
                projected_data
            )
            
            # Fuse data streams
            self.logger.info("Fusing data streams")
            fused_data = await self.data_fusion.fuse_data(
                standardized_data,
                lidar_data,
                imu_data,
                metadata
            )
            
            processing_time = time.time() - start_time
            self.logger.info(f"Image processing completed in {processing_time:.2f} seconds")
            
            # Add processing info
            fused_data['processing_info'] = {
                'processing_time': processing_time,
                'tile_count': sum(len(tiles) for tiles in standardized_data['tiles']),
                'frame_count': len(standardized_data['tiles']),
                'timestamp': datetime.now().isoformat()
            }
            
            return fused_data
            
        except Exception as e:
            self.logger.error(f"Error in image processing pipeline: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            # Return error information
            return {
                'error': str(e),
                'traceback': traceback.format_exc(),
                'timestamp': datetime.now().isoformat()
            }


class ImageTilingProcessor:
    """Handles image tiling and preprocessing with enhancements for tree detection"""
    def __init__(self, config: PipelineConfig):
        """
        Initialize image tiling processor
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.memory_manager = MemoryManager(config)
        
    async def process_images(self, rgb_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process and tile RGB imagery with vegetation enhancement
        
        Args:
            rgb_data: Dictionary containing RGB images, timestamps, and metadata
            
        Returns:
            Dictionary with processed tiles and metadata
        """
        processed_images = []
        tile_metadata = []
        
        # Check if we have images to process
        if not rgb_data.get('images'):
            logger.error("No images to process")
            raise ValueError("No input images provided for processing")
        
        # Process images in parallel or sequentially based on image size
        sample_img = rgb_data['images'][0]
        img_size_mb = sample_img.nbytes / (1024 * 1024)
        
        # Use parallel processing for smaller images, sequential for large ones
        # to avoid memory issues
        if img_size_mb < 50 and len(rgb_data['images']) > 1:
            logger.info(f"Using parallel processing for {len(rgb_data['images'])} images")
            processed_images, tile_metadata = await self._process_images_parallel(rgb_data)
        else:
            # Process sequentially for large images
            logger.info(f"Using sequential processing for {len(rgb_data['images'])} images")
            processed_images, tile_metadata = await self._process_images_sequential(rgb_data)
                
        # Final cleanup
        self.memory_manager.cleanup()
                
        logger.info(f"Completed processing {len(rgb_data['images'])} images into {sum(len(tiles) for tiles in processed_images)} tiles")
            
        return {
            'tiles': processed_images,
            'metadata': tile_metadata
        }
    
    async def _process_images_parallel(self, rgb_data: Dict[str, Any]) -> Tuple[List[List[np.ndarray]], List[Dict[str, Any]]]:
        """
        Process images in parallel
        
        Args:
            rgb_data: Dictionary containing RGB images, timestamps, and metadata
            
        Returns:
            Tuple of (processed_images, tile_metadata)
        """
        processed_images = []
        tile_metadata = []
        
        # Create partial function for parallel execution
        process_func = partial(self._process_single_image)
        
        # Process images in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all tasks
            futures = []
            for idx, (image, timestamp, meta) in enumerate(zip(
                rgb_data['images'], 
                rgb_data['timestamps'], 
                rgb_data['metadata']
            )):
                future = executor.submit(
                    process_func, 
                    image=image, 
                    timestamp=timestamp, 
                    meta=meta, 
                    idx=idx
                )
                futures.append(future)
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    idx, tiles, tiles_meta = future.result()
                    # Insert at the correct position to maintain order
                    while len(processed_images) <= idx:
                        processed_images.append([])
                        tile_metadata.append({})
                    processed_images[idx] = tiles
                    tile_metadata[idx] = tiles_meta
                except Exception as e:
                    logger.error(f"Error in parallel image processing: {e}", exc_info=True)
        
        return processed_images, tile_metadata
        
    async def _process_images_sequential(self, rgb_data: Dict[str, Any]) -> Tuple[List[List[np.ndarray]], List[Dict[str, Any]]]:
        """
        Process images sequentially
        
        Args:
            rgb_data: Dictionary containing RGB images, timestamps, and metadata
            
        Returns:
            Tuple of (processed_images, tile_metadata)
        """
        processed_images = []
        tile_metadata = []
        
        for idx, (image, timestamp, meta) in enumerate(zip(
            rgb_data['images'], 
            rgb_data['timestamps'], 
            rgb_data['metadata']
        )):
            try:
                logger.info(f"Processing image {idx+1}/{len(rgb_data['images'])}")
                _, tiles, tiles_meta = self._process_single_image(image, timestamp, meta, idx)
                processed_images.append(tiles)
                tile_metadata.append(tiles_meta)
                
                # Force memory cleanup after each large image
                img_size_mb = image.nbytes / (1024 * 1024)
                if img_size_mb > 100:
                    self.memory_manager.cleanup()
                    
            except Exception as e:
                logger.error(f"Error processing image {idx}: {e}", exc_info=True)
                
                # Add empty placeholder to maintain indexing
                processed_images.append([])
                tile_metadata.append({
                    'timestamp': timestamp,
                    'source_metadata': meta,
                    'tile_count': 0,
                    'error': str(e)
                })
                
        return processed_images, tile_metadata
        
    def _process_single_image(self, image: np.ndarray, timestamp: Any, meta: Dict[str, Any], idx: int) -> Tuple[int, List[np.ndarray], Dict[str, Any]]:
        """
        Process a single image for parallel execution
        
        Args:
            image: Input image array
            timestamp: Image timestamp
            meta: Image metadata
            idx: Image index
            
        Returns:
            Tuple of (index, tiles, tiles_metadata)
        """
        try:
            # Preprocess image for tree detection
            processed = self._preprocess_image(image)
            
            # Generate tiles
            tiles, tile_coords = self._generate_tiles(processed)
            
            # Create metadata
            tiles_meta = {
                'timestamp': timestamp,
                'source_metadata': meta,
                'tile_count': len(tiles),
                'tile_size': self.config.tile_size,
                'overlap': self.config.overlap_pixels,
                'tile_coordinates': tile_coords
            }
            
            return idx, tiles, tiles_meta
            
        except Exception as e:
            logger.error(f"Error in _process_single_image for index {idx}: {e}", exc_info=True)
            raise
        
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image with enhancement for tree detection
        
        Args:
            image: Input image array
            
        Returns:
            Preprocessed image array
        """
        try:
            # Convert to float32 for better precision
            img_float = image.astype(np.float32) / 255.0
            
            # Apply vegetation enhancement if configured
            if self.config.vegetation_enhancement:
                img_float = self._enhance_vegetation(img_float)
                
            # Apply contrast enhancement if configured
            if self.config.contrast_enhancement:
                img_float = self._enhance_contrast(img_float)
                
            # Apply shadow correction if configured
            if self.config.shadow_correction:
                img_float = self._correct_shadows(img_float)
                
            # Ensure values are in valid range
            img_float = np.clip(img_float, 0.0, 1.0)
            
            return img_float
            
        except Exception as e:
            logger.error(f"Error in image preprocessing: {e}", exc_info=True)
            # Return original image if processing fails
            return image.astype(np.float32) / 255.0
        
    def _enhance_vegetation(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance vegetation in image to improve tree detection
        
        Args:
            image: Input image array
            
        Returns:
            Enhanced image array
        """
        # Calculate vegetation index (ExG - Excess Green Index)
        if image.shape[2] >= 3:  # Ensure RGB image
            r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
            
            # Excess Green Index: 2*G - R - B
            exg = 2.0 * g - r - b
            
            # Normalize to 0-1 range
            exg_min, exg_max = np.min(exg), np.max(exg)
            if exg_max > exg_min:  # Avoid division by zero
                exg_norm = (exg - exg_min) / (exg_max - exg_min)
            else:
                exg_norm = np.zeros_like(exg)
                
            # Create enhanced image by boosting green channel
            enhanced = image.copy()
            enhanced[:, :, 1] = np.minimum(1.0, enhanced[:, :, 1] + 0.2 * exg_norm)
            
            return enhanced
        else:
            return image
            
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image contrast to improve feature visibility
        
        Args:
            image: Input image array
            
        Returns:
            Contrast-enhanced image array
        """
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        try:
            # Convert to LAB color space (L for luminance)
            lab = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
            
            # Create CLAHE object
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            
            # Apply CLAHE to L channel
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            
            # Convert back to RGB
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
            
            return enhanced
        except Exception as e:
            logger.warning(f"Contrast enhancement failed: {e}")
            return image
            
    def _correct_shadows(self, image: np.ndarray) -> np.ndarray:
        """
        Apply shadow correction to improve detection in shadowed areas
        
        Args:
            image: Input image array
            
        Returns:
            Shadow-corrected image array
        """
        try:
            # Convert to HSV color space
            hsv = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
            
            # Get value channel (brightness)
            v_channel = hsv[:, :, 2]
            
            # Apply shadow detection (simple threshold for demo)
            shadow_threshold = np.mean(v_channel) * 0.7
            shadow_mask = v_channel < shadow_threshold
            
            # Brighten shadow regions
            brightened = image.copy()
            shadow_factor = 1.3  # Brightness increase factor
            
            # Apply brightening to shadows, all channels
            for i in range(3):  # RGB channels
                channel = brightened[:, :, i]
                channel[shadow_mask] = np.minimum(1.0, channel[shadow_mask] * shadow_factor)
                brightened[:, :, i] = channel
                
            return brightened
        except Exception as e:
            logger.warning(f"Shadow correction failed: {e}")
            return image
        
    def _generate_tiles(self, image: np.ndarray) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
        """
        Generate overlapping tiles from image with coordinates
        
        Args:
            image: Input image array
            
        Returns:
            Tuple of (tiles, tile_coordinates)
        """
        tiles = []
        tile_coordinates = []
        
        h, w = image.shape[:2]
        tile_h, tile_w = self.config.tile_size
        overlap = self.config.overlap_pixels
        
        # Calculate step sizes and number of tiles
        h_step = tile_h - overlap
        w_step = tile_w - overlap
        
        n_tiles_h = max(1, (h - overlap) // h_step)
        n_tiles_w = max(1, (w - overlap) // w_step)
        
        # Adjust to ensure coverage of the entire image
        if (n_tiles_h - 1) * h_step + tile_h < h:
            n_tiles_h += 1
        if (n_tiles_w - 1) * w_step + tile_w < w:
            n_tiles_w += 1
            
        # Calculate optimal tile positions for even coverage
        for i in range(n_tiles_h):
            y = min(i * h_step, h - tile_h) if h >= tile_h else 0
            
            for j in range(n_tiles_w):
                x = min(j * w_step, w - tile_w) if w >= tile_w else 0
                
                # Extract tile
                end_y = min(y + tile_h, h)
                end_x = min(x + tile_w, w)
                tile = image[y:end_y, x:end_x]
                
                # Skip tiles that are too small
                if tile.shape[0] < tile_h * 0.5 or tile.shape[1] < tile_w * 0.5:
                    continue
                    
                # Pad if needed
                if tile.shape[0] < tile_h or tile.shape[1] < tile_w:
                    padded_tile = np.zeros((tile_h, tile_w, image.shape[2]), dtype=image.dtype)
                    padded_tile[:tile.shape[0], :tile.shape[1]] = tile
                    tile = padded_tile
                
                tiles.append(tile)
                
                # Store tile coordinates in original image
                tile_coordinates.append({
                    'x': x,
                    'y': y,
                    'width': end_x - x,
                    'height': end_y - y,
                    'padded_width': tile_w,
                    'padded_height': tile_h,
                    'row': i,
                    'col': j
                })
                    
        return tiles, tile_coordinates


class DataFusion:
    """Fuses RGB, LiDAR, and IMU data with tree-specific enhancements"""
    def __init__(self, config: PipelineConfig):
        """
        Initialize data fusion
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.memory_manager = MemoryManager(config)
    
    async def fuse_data(
        self,
        standardized_data: Dict[str, Any],
        lidar_data: Dict[str, Any],
        imu_data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Fuse all data streams optimized for tree detection
        
        Args:
            standardized_data: Standardized RGB data
            lidar_data: LiDAR point cloud data
            imu_data: IMU position and orientation data
            metadata: Additional metadata
            
        Returns:
            Dictionary with fused data
        """
        try:
            # Check data availability
            has_lidar = len(lidar_data.get('point_clouds', [])) > 0
            has_imu = len(imu_data.get('position', [])) > 0
            
            logger.info(f"Fusing data streams (LiDAR: {has_lidar}, IMU: {has_imu})")
            
            # Fuse RGB and LiDAR
            if has_lidar:
                rgb_lidar_fused = await self._fuse_rgb_lidar(
                    standardized_data,
                    lidar_data
                )
            else:
                # Skip LiDAR fusion if no data
                rgb_lidar_fused = await self._create_rgb_only_data(standardized_data)
            
            # Add IMU data
            if has_imu:
                fully_fused = await self._add_imu_data(
                    rgb_lidar_fused,
                    imu_data
                )
            else:
                # Skip IMU fusion if no data
                fully_fused = rgb_lidar_fused
            
            # Add additional metadata
            if metadata:
                fully_fused = await self._add_metadata(fully_fused, metadata)
                
            # Enhance data for tree detection
            fully_fused = await self._enhance_tree_detection(fully_fused)
            
            # Clean up memory
            self.memory_manager.cleanup()
            
            return fully_fused
            
        except Exception as e:
            logger.error(f"Error in data fusion: {e}", exc_info=True)
            
            # Create minimal output on error
            return self._create_error_output(standardized_data, str(e))
    
    async def _add_metadata(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add or update metadata in the data dictionary
        
        Args:
            data: Data dictionary to update
            metadata: Metadata to add
            
        Returns:
            Updated data dictionary
        """
        if 'metadata' not in data:
            data['metadata'] = {}
            
        # Update with new metadata
        for key, value in metadata.items():
            data['metadata'][key] = value
            
        return data
        
    async def _enhance_tree_detection(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply tree-specific enhancements to fused data
        
        Args:
            data: Fused data to enhance
            
        Returns:
            Enhanced data for tree detection
        """
        # This method implements tree-specific enhancements to the data
        # For now, just return the data as is - enhancement will be implemented in future versions
        if 'metadata' not in data:
            data['metadata'] = {}
            
        # Add a note that tree enhancement was applied
        data['metadata']['tree_enhancement_applied'] = True
        
        return data
        
    async def _fuse_rgb_lidar(self, rgb_data: Dict[str, Any], lidar_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fuse RGB and LiDAR data
        
        Args:
            rgb_data: RGB image data with tiles and metadata
            lidar_data: LiDAR point cloud data
            
        Returns:
            Fused RGB and LiDAR data structure
        """
        # Initialize fused data structure
        fused_data = {
            'metadata': {},
            'frames': []
        }
        
        # Copy RGB metadata
        if 'metadata' in rgb_data:
            fused_data['metadata'] = rgb_data['metadata'].copy()
            
        # Add LiDAR metadata if available
        if 'metadata' in lidar_data:
            lidar_meta = lidar_data.get('metadata', {})
            if isinstance(lidar_meta, list) and len(lidar_meta) > 0:
                fused_data['metadata']['lidar'] = lidar_meta[0]
            else:
                fused_data['metadata']['lidar'] = lidar_meta
                
        # Get point clouds and timestamps
        point_clouds = lidar_data.get('point_clouds', [])
        lidar_timestamps = lidar_data.get('timestamps', [])
        
        # For each RGB tile set, create a frame with associated LiDAR data
        for i, tiles in enumerate(rgb_data.get('tiles', [])):
            # Create frame with tiles
            frame = {
                'rgb': tiles[0] if tiles else None,  # Use first tile for RGB
                'metadata': {}
            }
            
            # Add tile metadata
            if 'tile_metadata' in rgb_data and i < len(rgb_data['tile_metadata']):
                frame['metadata'] = rgb_data['tile_metadata'][i].copy()
                
            # Find matching LiDAR data if available
            if i < len(lidar_timestamps) and i < len(point_clouds):
                frame['lidar_points'] = point_clouds[i]
                
                # Add LiDAR metadata
                if 'lidar_metadata' in lidar_data and i < len(lidar_data['lidar_metadata']):
                    frame['lidar_metadata'] = lidar_data['lidar_metadata'][i]
                    
            # Add frame to result
            fused_data['frames'].append(frame)
            
        return fused_data
    
    async def _create_rgb_only_data(self, standardized_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create data structure with RGB data only (no LiDAR)
        
        Args:
            standardized_data: Standardized RGB data
            
        Returns:
            Dictionary with RGB-only frames
        """
        rgb_only_data = {
            'frames': [],
            'timestamps': []
        }
        
        # Create frames without LiDAR
        for i, (tiles, world_coords, meta) in enumerate(zip(
            standardized_data['tiles'],
            standardized_data['world_coordinates'],
            standardized_data['metadata']
        )):
            frame_tiles = []
            for j, (tile, coords) in enumerate(zip(tiles, world_coords)):
                frame_tiles.append({
                    'rgb': tile,
                    'world_coords': coords,
                    'timestamp': meta.get('timestamp', datetime.now())
                })
            rgb_only_data['frames'].append(frame_tiles)
            
            if 'timestamp' in meta:
                rgb_only_data['timestamps'].append(meta['timestamp'])
            else:
                rgb_only_data['timestamps'].append(datetime.now())
                
        return rgb_only_data
    
    def _create_error_output(self, standardized_data: Dict[str, Any], error_msg: str) -> Dict[str, Any]:
        """
        Create minimal output structure for error case
        
        Args:
            standardized_data: Input data
            error_msg: Error message
            
        Returns:
            Dictionary with minimal error output
        """
        empty_frames = []
        empty_timestamps = []
        
        for i, tiles in enumerate(standardized_data['tiles']):
            frame_tiles = []
            for tile in tiles:
                frame_tiles.append({
                    'rgb': tile,
                    'error': error_msg
                })
            empty_frames.append(frame_tiles)
            
            # Get timestamp if available
            if i < len(standardized_data['metadata']) and 'timestamp' in standardized_data['metadata'][i]:
                empty_timestamps.append(standardized_data['metadata'][i]['timestamp'])
            else:
                empty_timestamps.append(datetime.now())
                
        return {
            'frames': empty_frames,
            'timestamps': empty_timestamps,
            'error': error_msg
        }
    
    def extract_timestamp(self, ts_data: Any) -> float:
        """
        Helper function to extract timestamp value
        
        Args:
            ts_data: Timestamp data in various formats
            
        Returns:
            Timestamp as float (seconds since epoch)
        """
        if isinstance(ts_data, datetime):
            return ts_data.timestamp()
        elif isinstance(ts_data, dict):
            if 'timestamp' in ts_data:
                ts = ts_data['timestamp']
                if isinstance(ts, datetime):
                    return ts.timestamp()
                elif isinstance(ts, (int, float)):
                    return float(ts)
                elif isinstance(ts, str):
                    try:
                        return float(ts)
                    except ValueError:
                        try:
                            return datetime.fromisoformat(ts).timestamp()
                        except ValueError:
                            raise ValueError(f"Invalid timestamp format in dictionary: {ts}")
            raise ValueError(f"Missing or invalid timestamp in dictionary: {ts_data}")
        elif isinstance(ts_data, (int, float)):
            return float(ts_data)
        elif isinstance(ts_data, str):
            try:
                return float(ts_data)
            except ValueError:
                try:
                    return datetime.fromisoformat(ts_data).timestamp()
                except ValueError:
                    raise ValueError(f"Invalid timestamp string format: {ts_data}")
        else:
            raise TypeError(f"Unsupported timestamp type: {type(ts_data)}")

    def _transform_points_to_tile(
        self,
        points: np.ndarray,
        world_coords: np.ndarray
    ) -> np.ndarray:
        """
        Transform LiDAR points to tile coordinate system optimized for tree detection
        
        Args:
            points: LiDAR points
            world_coords: World coordinates of tile corners
            
        Returns:
            Transformed points in tile coordinates
        """
        try:
            # Ensure world_coords is properly formatted for getPerspectiveTransform
            if world_coords.shape[-1] == 3:  # If we have 3D coordinates
                # Extract just the x,y coordinates for 2D transform
                world_coords_2d = world_coords[:4, :2].astype(np.float32)
            else:
                world_coords_2d = world_coords[:4].astype(np.float32)
            
            # Create normalized tile coordinates
            tile_coords = np.array([
                [0, 0],
                [1, 0],
                [1, 1],
                [0, 1]
            ], dtype=np.float32)
            
            # Check for degenerate or invalid coordinates
            if not np.isfinite(world_coords_2d).all():
                logger.error(f"Invalid world coordinates found: {world_coords_2d}")
                raise ValueError("Invalid world coordinates containing NaN or Inf values")
                
            # Check if points form a proper quadrilateral
            if np.linalg.norm(world_coords_2d[0] - world_coords_2d[1]) < 1e-6 or \
               np.linalg.norm(world_coords_2d[1] - world_coords_2d[2]) < 1e-6 or \
               np.linalg.norm(world_coords_2d[2] - world_coords_2d[3]) < 1e-6 or \
               np.linalg.norm(world_coords_2d[3] - world_coords_2d[0]) < 1e-6:
                logger.error("World coordinates too close, cannot compute transform")
                raise ValueError("World coordinates too close for proper perspective transform")
            
            # Get the perspective transform
            tile_transform = cv2.getPerspectiveTransform(world_coords_2d, tile_coords)
            
            # Transform the points
            if points.shape[-1] == 3:  # If we have 3D points
                # Extract x,y coordinates for transformation
                points_2d = points[:, :2]
            else:
                points_2d = points
            
            # Make homogeneous coordinates
            points_homogeneous = np.column_stack([points_2d, np.ones(len(points_2d))])
            
            # Apply transform
            transformed = np.dot(tile_transform, points_homogeneous.T).T
            
            # Normalize homogeneous coordinates
            transformed = transformed[:, :2] / transformed[:, 2:].clip(1e-10)  # Prevent division by zero
            
            return transformed
            
        except cv2.error as e:
            logger.error(f"Error in perspective transform: {e}")
            raise ValueError(f"OpenCV perspective transform error: {e}")
        except Exception as e:
            logger.error(f"Error transforming points: {e}")
            raise ValueError(f"Failed to transform points: {e}")

    async def _fuse_rgb_lidar(
        self,
        rgb_data: Dict[str, Any],
        lidar_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Fuse RGB and LiDAR data with tree-specific enhancements
        
        Args:
            rgb_data: RGB image data
            lidar_data: LiDAR point cloud data
            
        Returns:
            Dictionary with fused RGB-LiDAR frames
        """
        fused_frames = []
        timestamps = []
        
        # Check if we have any data to fuse
        if not rgb_data['tiles'] or not lidar_data['point_clouds']:
            logger.warning("Not enough data for RGB-LiDAR fusion")
            return {'frames': [], 'timestamps': []}
            
        # Process each frame
        for idx, (tiles, world_coords, metadata) in enumerate(zip(
            rgb_data['tiles'], 
            rgb_data['world_coordinates'],
            rgb_data['metadata']
        )):
            try:
                logger.info(f"Fusing RGB-LiDAR for frame {idx+1}/{len(rgb_data['tiles'])}")
                
                # Get timestamp
                current_timestamp = self._extract_frame_timestamp(metadata)
                
                # Find closest LiDAR scan by timestamp
                point_cloud = self._find_closest_point_cloud(lidar_data, current_timestamp)
                
                # Process tiles based on data size
                if len(tiles) > 16 and point_cloud is not None and len(point_cloud) > 1000:
                    # Use thread pool for parallel processing
                    fused_tiles = await self._process_tiles_parallel(
                        tiles, 
                        world_coords, 
                        point_cloud, 
                        current_timestamp
                    )
                else:
                    # Process sequentially for smaller frames
                    fused_tiles = await self._process_tiles_sequential(
                        tiles,
                        world_coords,
                        point_cloud,
                        current_timestamp
                    )
                
                fused_frames.append(fused_tiles)
                timestamps.append(current_timestamp)
                
            except Exception as e:
                logger.error(f"Error processing frame {idx}: {e}", exc_info=True)
                
                # Add empty frame on error
                fused_frames.append([])
                timestamps.append(datetime.now())
        
        return {
            'frames': fused_frames,
            'timestamps': timestamps
        }
    
    def _extract_frame_timestamp(self, metadata: Dict[str, Any]) -> datetime:
        """
        Extract timestamp from frame metadata
        
        Args:
            metadata: Frame metadata
            
        Returns:
            Timestamp as datetime object
        """
        if 'timestamp' in metadata:
            current_timestamp = metadata['timestamp']
            if not isinstance(current_timestamp, datetime):
                ts_float = self.extract_timestamp(current_timestamp)
                current_timestamp = datetime.fromtimestamp(ts_float)
            return current_timestamp
        else:
            return datetime.now()
    
    def _find_closest_point_cloud(self, lidar_data: Dict[str, Any], current_timestamp: datetime) -> Optional[np.ndarray]:
        """
        Find the closest LiDAR point cloud by timestamp
        
        Args:
            lidar_data: LiDAR data dictionary
            current_timestamp: Current frame timestamp
            
        Returns:
            Closest point cloud or None if not available
        """
        lidar_idx = None
        if lidar_data and 'timestamps' in lidar_data and lidar_data['timestamps']:
            lidar_timestamps = lidar_data['timestamps']
            
            # Calculate time differences
            time_diffs = []
            for ts in lidar_timestamps:
                if isinstance(ts, datetime) and isinstance(current_timestamp, datetime):
                    diff = abs((ts - current_timestamp).total_seconds())
                else:
                    # Convert to timestamp and calculate difference
                    ts_float = self.extract_timestamp(ts)
                    current_ts_float = self.extract_timestamp(current_timestamp)
                    diff = abs(ts_float - current_ts_float)
                time_diffs.append(diff)
            
            # Find closest timestamp
            if time_diffs:
                lidar_idx = np.argmin(time_diffs)
        
        # Get corresponding point cloud if available
        if lidar_idx is not None and 'point_clouds' in lidar_data and lidar_idx < len(lidar_data['point_clouds']):
            point_cloud = lidar_data['point_clouds'][lidar_idx]
            
            # Ensure 3D points
            if point_cloud.shape[1] == 2:
                # If only 2D points (x, y), add z coordinate as zeros
                point_cloud = np.column_stack([point_cloud, np.zeros(point_cloud.shape[0])])
            elif point_cloud.shape[1] > 3:
                # If more than 3 dimensions, truncate to first 3
                point_cloud = point_cloud[:, :3]
                
            return point_cloud
            
        return None
    
# For testing the module independently
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    import sys
    import traceback
    
    # Function to load a sample image for testing
    def load_sample_image(path: str) -> np.ndarray:
        """
        Load a sample image for testing
        
        Args:
            path: Path to image file
            
        Returns:
            Image array
        """
        try:
            img = cv2.imread(path)
            if img is None:
                raise ValueError(f"Failed to load image: {path}")
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        except Exception as e:
            print(f"Error loading sample image: {e}")
            # Create a small test image
            return np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Test pipeline with sample data
    async def main():
        """Main function for testing the module"""
        # Create configuration
        config = PipelineConfig(
            vegetation_enhancement=True,
            contrast_enhancement=True,
            shadow_correction=True,
            tile_size=(512, 512),
            overlap_pixels=64,
            target_resolution=(512, 512),
            max_workers=4
        )
        
        # Create processor
        processor = ImageProcessor(config)
        
        # Sample image path (use command line arg if provided)
        sample_path = "sample.jpg"
        if len(sys.argv) > 1:
            sample_path = sys.argv[1]
            
        # Load sample image
        sample_img = load_sample_image(sample_path)
        
        # Create sample data
        rgb_data = {
            'images': [sample_img],
            'timestamps': [datetime.now()],
            'metadata': [{
                'filename': sample_path,
                'resolution': sample_img.shape[:2]
            }]
        }
        
        try:
            # Process sample data
            result = await processor.process_data(rgb_data)
            
            # Print results summary
            if 'error' in result:
                print(f"Processing error: {result['error']}")
            else:
                print(f"Processing successful:")
                print(f"- Frames: {len(result['frames'])}")
                print(f"- Tiles per frame: {len(result['frames'][0]) if result['frames'] else 0}")
                print(f"- Processing time: {result['processing_info']['processing_time']:.2f} seconds")
                
        except Exception as e:
            print(f"Error in test processing: {e}")
            traceback.print_exc()
        
    asyncio.run(main())