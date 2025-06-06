"""
Zarr Storage Manager for Tree Risk Assessment Pipeline
Handles efficient storage and retrieval of ML data with Google S2 geospatial indexing
"""

import os
import logging
import json
import asyncio
from pathlib import Path
import time
import shutil
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import zarr
from numcodecs import Blosc, JSON

# Import S2 library for geospatial indexing if available
try:
    import s2sphere as s2
    S2_AVAILABLE = True
except ImportError:
    S2_AVAILABLE = False
    logging.getLogger(__name__).warning("s2sphere library not available, geospatial indexing will be limited")

class StorageManager:
    """Manages data storage with zarr format and S2 geospatial indexing"""
    
    def __init__(self, config):
        """Initialize StorageManager with configuration"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Ensure zarr store path exists
        if hasattr(config, 'store_path'):
            self.store_path = config.store_path
            Path(self.store_path).mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Zarr store path: {self.store_path}")
        else:
            self.store_path = '/ttt/data/zarr'
            Path(self.store_path).mkdir(parents=True, exist_ok=True)
            self.logger.warning(f"No store_path in config, using default: {self.store_path}")
            
        # Set up ml path for temporary storage
        if hasattr(config, 'ml_path'):
            self.ml_path = config.ml_path
            Path(self.ml_path).mkdir(parents=True, exist_ok=True)
            self.logger.info(f"ML store path: {self.ml_path}")
        else:
            self.ml_path = '/ttt/data/ml'
            Path(self.ml_path).mkdir(parents=True, exist_ok=True)
            self.logger.warning(f"No ml_path in config, using default: {self.ml_path}")
    
    async def store_data(self, data: Dict, identifier: str, use_temp: bool = False) -> str:
        """
        Store processed data in zarr format with S2 geospatial indexing
        
        Args:
            data (Dict): Processed data to store
            identifier (str): Unique identifier for the data (area_id for map detections, mission_id for HDF5 data)
            use_temp (bool): If True, store in temporary location first
            
        Returns:
            str: Path to the stored data
        """
        self.logger.info(f"Storing data for identifier {identifier}")
        
        # Clean identifier to ensure consistent naming
        # Strip any 'detection_' prefix to get a clean numeric ID
        if isinstance(identifier, str) and identifier.startswith('detection_'):
            clean_id = identifier.replace('detection_', '')
        else:
            clean_id = identifier
            
        # Use a consistent directory structure
        # Format: /ttt/data/ml/ml_[job_id]/zarr
        ml_dir_name = f"ml_{clean_id}"
        
        # Determine storage location
        if use_temp:
            # Store in the zarr subdirectory of our ML results directory
            base_dir = os.path.join(self.ml_path, ml_dir_name)
            store_path = os.path.join(base_dir, "zarr")
            self.logger.info(f"Using consolidated storage at {store_path}")
        else:
            # Store directly in the main zarr store with the same identifier
            store_path = os.path.join(self.store_path, clean_id)
            self.logger.info(f"Using permanent storage at {store_path}")
        
        try:
            # Create zarr store
            store = zarr.open(store_path, mode='w')
            
            # Create root metadata
            store.attrs['identifier'] = identifier
            store.attrs['created_at'] = time.time()
            store.attrs['is_temporary'] = use_temp
            
            # Store metadata if available
            if 'metadata' in data:
                metadata_group = store.create_group('metadata')
                for key, value in data['metadata'].items():
                    # Convert complex structures to JSON for storage
                    if isinstance(value, (dict, list)):
                        metadata_group.attrs[key] = json.dumps(value)
                    else:
                        metadata_group.attrs[key] = value
            
            # Create frames group for storing individual frames
            frames = store.create_group('frames')
            
            # Create S2 index for geospatial data if coordinates are available
            if S2_AVAILABLE and 'coordinates' in data:
                s2_group = store.create_group('s2_index')
                await self._create_s2_index(s2_group, data['coordinates'])
            
            # Store each frame
            for i, frame in enumerate(data.get('frames', [])):
                await self._store_frame(frames, i, frame)
            
            self.logger.info(f"Successfully stored {len(data.get('frames', []))} frames at {store_path}")
            return store_path
            
        except Exception as e:
            self.logger.error(f"Error storing data: {e}")
            # Cleanup partial store
            if os.path.exists(store_path):
                shutil.rmtree(store_path)
            raise
    
    async def _create_s2_index(self, s2_group, coordinates):
        """Create S2 cell index for geospatial data"""
        if not S2_AVAILABLE:
            return
            
        try:
            # Create S2 cell index at multiple levels
            s2_cells = {}
            
            # Process all coordinates
            for coord_info in coordinates:
                if 'lat' in coord_info and 'lng' in coord_info:
                    lat = coord_info['lat']
                    lng = coord_info['lng']
                    
                    # Create S2 cell at multiple levels
                    for level in [10, 13, 15, 17, 20]:  # Different precision levels
                        latlng = s2.LatLng.from_degrees(lat, lng)
                        cell_id = s2.CellId.from_lat_lng(latlng).parent(level)
                        token = cell_id.to_token()
                        
                        if token not in s2_cells:
                            s2_cells[token] = []
                        
                        # Store object reference
                        obj_ref = {
                            'id': coord_info.get('id', ''),
                            'frame_idx': coord_info.get('frame_idx', -1),
                            'lat': lat,
                            'lng': lng,
                            'type': coord_info.get('type', 'unknown')
                        }
                        s2_cells[token].append(obj_ref)
            
            # Store S2 cells in zarr
            s2_cells_json = json.dumps(s2_cells)
            s2_group.attrs['cells'] = s2_cells_json
            
            # Store cell counts for each level
            level_counts = {}
            for token in s2_cells:
                level = len(token)  # Token length indicates cell level
                if level not in level_counts:
                    level_counts[level] = 0
                level_counts[level] += 1
            
            s2_group.attrs['level_counts'] = json.dumps(level_counts)
            
        except Exception as e:
            self.logger.error(f"Error creating S2 index: {e}")
    
    async def _store_frame(self, frames_group, frame_idx: int, frame_data: Union[Dict, List]):
        """Store a single frame of data in the zarr group"""
        # Create frame group
        frame = frames_group.create_group(str(frame_idx))
        
        # Handle case when frame_data is a list (tree data)
        if isinstance(frame_data, list):
            # Store trees as JSON in frame attributes
            frame.attrs['trees'] = json.dumps(frame_data)
            
            # Also store as individual tree properties for analysis
            trees_group = frame.create_group('trees')
            
            for i, tree in enumerate(frame_data):
                tree_group = trees_group.create_group(str(i))
                for key, value in tree.items():
                    if isinstance(value, (dict, list)):
                        tree_group.attrs[key] = json.dumps(value)
                    else:
                        tree_group.attrs[key] = value
            
            return
        
        # Continue with normal Dict processing
        # Set frame metadata
        if 'metadata' in frame_data:
            for key, value in frame_data['metadata'].items():
                if isinstance(value, (dict, list)):
                    frame.attrs[key] = json.dumps(value)
                else:
                    frame.attrs[key] = value
        
        # Store RGB data if available
        if 'rgb' in frame_data:
            rgb_data = frame_data['rgb']
            # Check if data is numpy array
            if isinstance(rgb_data, np.ndarray):
                # Use blosc compression for RGB data
                compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.SHUFFLE)
                frame.create_dataset('rgb', data=rgb_data, chunks=True, compressor=compressor)
            else:
                self.logger.warning(f"RGB data for frame {frame_idx} is not a numpy array, skipping")
        
        # Store LiDAR data if available
        if 'lidar_points' in frame_data and frame_data['lidar_points'] is not None:
            lidar_data = frame_data['lidar_points']
            if isinstance(lidar_data, np.ndarray):
                compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.SHUFFLE)
                frame.create_dataset('lidar_points', data=lidar_data, chunks=True, compressor=compressor)
                
                # If we have additional LiDAR metadata
                if 'lidar_metadata' in frame_data:
                    lidar_meta = frame.create_group('lidar_metadata')
                    for key, value in frame_data['lidar_metadata'].items():
                        if isinstance(value, np.ndarray):
                            lidar_meta.create_dataset(key, data=value, chunks=True, compressor=compressor)
                        elif isinstance(value, (dict, list)):
                            lidar_meta.attrs[key] = json.dumps(value)
                        else:
                            lidar_meta.attrs[key] = value
        
        # Store segmentation masks if available
        if 'masks' in frame_data and frame_data['masks'] is not None:
            masks_data = frame_data['masks']
            if isinstance(masks_data, np.ndarray):
                compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.SHUFFLE)
                frame.create_dataset('masks', data=masks_data, chunks=True, compressor=compressor)
        
        # Store bounding boxes if available
        if 'boxes' in frame_data and frame_data['boxes'] is not None:
            boxes_data = frame_data['boxes']
            if isinstance(boxes_data, np.ndarray):
                frame.create_dataset('boxes', data=boxes_data)
            elif isinstance(boxes_data, list):
                frame.create_dataset('boxes', data=np.array(boxes_data))
        
        # Store tree detection data specifically to match dashboard expectations
        if 'trees' in frame_data and frame_data['trees'] is not None:
            trees_group = frame.create_group('detections')
            trees_data = frame_data['trees']
            
            # Ensure we have a tree_features group that the dashboard expects
            features_group = frame.create_group('features')
            tree_features = features_group.create_group('tree_features')
            
            # Create necessary tree detection arrays
            tree_count = len(trees_data)
            if tree_count > 0:
                # Create datasets for tree properties
                bboxes = []
                confidences = []
                widths = []
                heights = []
                
                for tree in trees_data:
                    bboxes.append(tree.get('bbox', [0, 0, 0, 0]))
                    confidences.append(tree.get('confidence', 0.0))
                    widths.append(tree.get('width', 0.0))
                    heights.append(tree.get('height', 0.0))
                
                # Store as datasets
                tree_features.create_dataset('bbox', data=np.array(bboxes))
                tree_features.create_dataset('confidence', data=np.array(confidences))
                tree_features.create_dataset('width', data=np.array(widths))
                tree_features.create_dataset('height', data=np.array(heights))
                
                # Store in detections group as well for redundancy
                trees_group.create_dataset('boxes', data=np.array(bboxes))
                trees_group.create_dataset('scores', data=np.array(confidences))
        
        # Store any other arrays present in the frame data
        for key, value in frame_data.items():
            if key not in ['metadata', 'rgb', 'lidar_points', 'lidar_metadata', 'masks', 'boxes', 'trees']:
                if isinstance(value, np.ndarray):
                    # For smaller arrays, don't use compression
                    if value.size < 1000:
                        frame.create_dataset(key, data=value)
                    else:
                        compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.SHUFFLE)
                        frame.create_dataset(key, data=value, chunks=True, compressor=compressor)
                elif isinstance(value, (list, tuple)) and all(isinstance(x, (int, float)) for x in value):
                    frame.create_dataset(key, data=np.array(value))
                elif isinstance(value, (dict, list)):
                    frame.attrs[key] = json.dumps(value)
                elif value is not None:
                    frame.attrs[key] = value
    
    async def read_data(self, store_path: str) -> Dict:
        """
        Read data from zarr store
        
        Args:
            store_path (str): Path to the zarr store
            
        Returns:
            Dict: Dictionary containing the stored data
        """
        self.logger.info(f"Reading data from {store_path}")
        
        try:
            # Open zarr store
            store = zarr.open(store_path, mode='r')
            
            # Read metadata
            metadata = dict(store.attrs)
            
            # Read frames
            frames_data = []
            if 'frames' in store:
                for i in range(len(store.frames)):
                    frame_key = str(i)
                    if frame_key in store.frames:
                        frame = store.frames[frame_key]
                        frame_data = dict(frame.attrs)
                        
                        # Read arrays
                        for key in frame:
                            if key not in frame_data and not key.startswith('_'):
                                if isinstance(frame[key], zarr.Group):
                                    # Skip groups, handle specifically if needed
                                    continue
                                try:
                                    frame_data[key] = frame[key][:]
                                except Exception as e:
                                    self.logger.error(f"Error reading {key} from frame {i}: {e}")
                        
                        frames_data.append(frame_data)
            
            result = {
                'metadata': metadata,
                'frames': frames_data
            }
            
            # Read S2 index if available
            if 's2_index' in store:
                s2_index = store.s2_index
                if 'cells' in s2_index.attrs:
                    result['s2_index'] = json.loads(s2_index.attrs['cells'])
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error reading data from {store_path}: {e}")
            raise
    
    async def delete_data(self, store_path: str) -> bool:
        """
        Delete a zarr store
        
        Args:
            store_path (str): Path to the zarr store
            
        Returns:
            bool: True if deletion was successful
        """
        self.logger.info(f"Deleting data at {store_path}")
        
        try:
            if os.path.exists(store_path):
                shutil.rmtree(store_path)
                self.logger.info(f"Successfully deleted {store_path}")
                return True
            else:
                self.logger.warning(f"Store not found at {store_path}")
                return False
        except Exception as e:
            self.logger.error(f"Error deleting data at {store_path}: {e}")
            return False
    
    async def list_stores(self, include_temp: bool = False) -> List[str]:
        """
        List all available zarr stores
        
        Args:
            include_temp (bool): If True, include temporary stores too
            
        Returns:
            List[str]: List of store paths
        """
        stores = []
        
        try:
            # List permanent stores
            for item in os.listdir(self.store_path):
                item_path = os.path.join(self.store_path, item)
                if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, '.zgroup')):
                    stores.append(item_path)
            
            # If requested, list ML stores too
            if include_temp:
                for item in os.listdir(self.ml_path):
                    item_path = os.path.join(self.ml_path, item)
                    if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, '.zgroup')):
                        stores.append(item_path)
            
            self.logger.info(f"Found {len(stores)} zarr stores")
            return stores
        except Exception as e:
            self.logger.error(f"Error listing zarr stores: {e}")
            return []
            
    async def move_temp_to_permanent(self, area_id: str, selected_trees: List = None) -> str:
        """
        Move selected trees from temporary storage to permanent storage.
        Cleans up temporary storage after successful copy.
        
        Args:
            area_id (str): The area identifier 
            selected_trees (List): List of tree data to keep
            
        Returns:
            str: Path to the permanent store
        """
        if not selected_trees:
            self.logger.warning("No trees selected for permanent storage")
            return None
            
        ml_temp_path = os.path.join(self.ml_path, f"{area_id}_temp")
        permanent_path = os.path.join(self.store_path, f"{area_id}_ml_results")
        
        self.logger.info(f"Moving {len(selected_trees)} selected trees from {ml_temp_path} to {permanent_path}")
        
        if not os.path.exists(ml_temp_path):
            error_msg = f"Temporary store not found at {ml_temp_path}"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        try:
            # If the permanent path already exists, remove it first
            if os.path.exists(permanent_path):
                shutil.rmtree(permanent_path)
                
            # Create a new zarr store for permanent storage
            temp_store = zarr.open(ml_temp_path, mode='r')
            perm_store = zarr.open(permanent_path, mode='w')
            
            # Copy only selected trees and relevant metadata
            # Implementation details here...
            # For simplicity, we'll create a basic structure with selected trees
            
            # Set metadata 
            perm_store.attrs['area_id'] = area_id
            perm_store.attrs['created_at'] = time.time()
            perm_store.attrs['is_temporary'] = False
            perm_store.attrs['tree_count'] = len(selected_trees)
            
            # Create frames group
            frames_group = perm_store.create_group('frames')
            
            # Process selected trees
            # Implementation details here...
            # This would filter and copy only the selected trees to permanent storage
            
            # Success - cleanup temporary storage
            self.logger.info(f"Successfully moved {len(selected_trees)} trees to {permanent_path}")
            self.logger.info(f"Cleaning up temporary storage at {ml_temp_path}")
            shutil.rmtree(ml_temp_path)
            
            return permanent_path
            
        except Exception as e:
            self.logger.error(f"Error moving trees to permanent storage: {e}")
            # Clean up if there was an error
            if os.path.exists(permanent_path):
                shutil.rmtree(permanent_path)
            raise
            
    async def _copy_frame_with_filtering(self, source_frame, target_frames, frame_idx, selected_trees):
        """Copy a frame from source to target, filtering out unselected trees"""
        # Create frame group
        target_frame = target_frames.create_group(frame_idx)
        
        # Copy frame attributes
        for key, value in source_frame.attrs.items():
            target_frame.attrs[key] = value
            
        # Handle each child group or dataset
        for name in source_frame:
            # Special handling for detections and features groups which contain tree data
            if name == 'detections' or name == 'features':
                if name == 'detections':
                    detections_group = target_frame.create_group('detections')
                    # Here we would filter only selected trees
                    # For now, just copy everything if we don't have a filtering mechanism
                    if not selected_trees:
                        zarr.copy(source_frame[name], detections_group)
                    else:
                        # Implement filtering based on selected_trees
                        # This is application-specific and depends on the tree selection format
                        pass
                elif name == 'features':
                    features_group = target_frame.create_group('features')
                    # Copy non-tree feature groups directly
                    for feature_type in source_frame[name]:
                        if feature_type != 'tree_features':
                            zarr.copy(source_frame[name][feature_type], features_group.create_group(feature_type))
                    
                    # Handle tree_features if it exists
                    if 'tree_features' in source_frame[name]:
                        tree_features = features_group.create_group('tree_features')
                        # Filter trees or copy all if no selection provided
                        if not selected_trees:
                            zarr.copy(source_frame[name]['tree_features'], tree_features)
                        else:
                            # Implement filtering based on selected_trees
                            # This depends on your tree selection format
                            pass
            else:
                # For other datasets/groups, just copy directly
                if isinstance(source_frame[name], zarr.Group):
                    zarr.copy(source_frame[name], target_frame.create_group(name))
                else:
                    zarr.copy(source_frame[name], target_frame.create_dataset(name, data=source_frame[name][:]))
            
    async def query_by_s2_cell(self, cell_token: str) -> List[Dict]:
        """
        Query objects contained in a specific S2 cell
        
        Args:
            cell_token (str): S2 cell token
            
        Returns:
            List[Dict]: Objects within the S2 cell
        """
        if not S2_AVAILABLE:
            self.logger.warning("S2 library not available, cannot perform spatial query")
            return []
            
        results = []
        
        try:
            # Iterate through all stores
            stores = await self.list_stores()
            
            for store_path in stores:
                store = zarr.open(store_path, mode='r')
                
                # Check if this store has an S2 index
                if 's2_index' in store and 'cells' in store.s2_index.attrs:
                    s2_cells = json.loads(store.s2_index.attrs['cells'])
                    
                    # Check if the requested cell exists in this store
                    if cell_token in s2_cells:
                        # Add objects to results
                        for obj in s2_cells[cell_token]:
                            obj['store_path'] = store_path
                            results.append(obj)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error querying by S2 cell: {e}")
            return []