# backend/services/validation_service.py
import os
import json
import logging
import uuid
from datetime import datetime
import sys
import numpy as np

# Import config
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import APP_MODE, MOCK_DIR, ZARR_DIR, TEMP_DIR, LIDAR_DIR

# Import zarr if available
try:
    import zarr
    ZARR_AVAILABLE = True
except ImportError:
    ZARR_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Zarr package not available. Falling back to mock data.")

logger = logging.getLogger(__name__)

class ValidationService:
    """Service to handle validation of tree risk assessments"""
    
    def __init__(self):
        self.app_mode = APP_MODE
        self.zarr_dir = ZARR_DIR
        self.temp_dir = TEMP_DIR
        self.lidar_dir = LIDAR_DIR
        logger.info(f"ValidationService initialized in {self.app_mode} mode")
        
    def _load_mock_data(self, filename):
        """Load mock data from JSON file"""
        try:
            file_path = os.path.join(MOCK_DIR, filename)
            if os.path.exists(file_path):
                with open(file_path, 'r') as file:
                    return json.load(file)
            else:
                logger.warning(f"Mock data file not found: {filename}")
                return []
        except Exception as e:
            logger.error(f"Error loading mock data: {str(e)}")
            return []
            
    def prepare_trees_from_temp_zarr(self, area_id):
        """
        Prepare tree detection results from temporary Zarr store for validation.
        
        Args:
            area_id (str): ID of the area to prepare
            
        Returns:
            list: List of tree data ready for validation
        """
        if not ZARR_AVAILABLE:
            logger.warning("Zarr not available, cannot prepare trees")
            return []
            
        try:
            # Define path to temp zarr store
            temp_zarr_path = os.path.join(self.temp_dir, f"{area_id}_temp")
            
            if not os.path.exists(temp_zarr_path):
                logger.warning(f"Temporary Zarr store not found at {temp_zarr_path}")
                return []
                
            logger.info(f"Opening temporary zarr store at {temp_zarr_path}")
            
            # Open zarr store
            store = zarr.open(temp_zarr_path, mode='r')
            
            # List to hold prepared tree data
            tree_data = []
            
            # Process frames to extract tree data
            if hasattr(store, 'frames'):
                for frame_idx in range(len(store.frames)):
                    frame = store.frames[str(frame_idx)]
                    
                    # Check if this frame has tree features
                    if hasattr(frame, 'features') and hasattr(frame.features, 'tree_features'):
                        tree_features = frame.features.tree_features
                        
                        # Get number of trees in this frame
                        if hasattr(tree_features, 'bbox') and len(tree_features.bbox) > 0:
                            for tree_idx in range(len(tree_features.bbox)):
                                # Generate unique ID for the tree
                                tree_id = f"{area_id}_{frame_idx}_{tree_idx}"
                                
                                # Extract basic tree properties
                                tree = {
                                    'id': tree_id,
                                    'bbox': tree_features.bbox[tree_idx].tolist() if hasattr(tree_features, 'bbox') else [0, 0, 0, 0],
                                    'confidence': float(tree_features.confidence[tree_idx]) if hasattr(tree_features, 'confidence') else 0.0,
                                    'area_id': area_id,
                                    'frame_idx': frame_idx,
                                    'tree_idx': tree_idx
                                }
                                
                                # Get centroid from segmentation if available
                                if hasattr(frame.features, 'segmentation_features') and hasattr(frame.features.segmentation_features, 'centroid'):
                                    centroids = frame.features.segmentation_features.centroid
                                    if tree_idx < len(centroids):
                                        tree['centroid'] = centroids[tree_idx].tolist()
                                        
                                        # Calculate geo coordinates if geo_transform is available
                                        if hasattr(frame, 'metadata') and hasattr(frame.metadata, 'geo_transform'):
                                            geo_transform = frame.metadata.geo_transform[:]
                                            centroid = centroids[tree_idx]
                                            lng = geo_transform[0] + centroid[0] * geo_transform[1] + centroid[1] * geo_transform[2]
                                            lat = geo_transform[3] + centroid[0] * geo_transform[4] + centroid[1] * geo_transform[5]
                                            tree['location'] = [lng, lat]
                                
                                # Only add trees with location data
                                if 'location' in tree:
                                    tree_data.append(tree)
            
            logger.info(f"Prepared {len(tree_data)} trees from temporary store")
            return tree_data
            
        except Exception as e:
            logger.error(f"Error preparing trees from temp zarr: {str(e)}")
            return []
            
    def _load_lidar_data(self, tree_id):
        """Load LiDAR data for a specific tree"""
        try:
            logger.info(f"Looking for LiDAR data for tree {tree_id} in {self.lidar_dir}")
            file_path = os.path.join(self.lidar_dir, f"{tree_id}.json")
            logger.info(f"Full LiDAR file path: {file_path}")
            
            if os.path.exists(file_path):
                logger.info(f"LiDAR file found at {file_path}")
                with open(file_path, 'r') as file:
                    data = json.load(file)
                    logger.info(f"Successfully loaded LiDAR data with keys: {list(data.keys())}")
                    return data
            else:
                logger.warning(f"LiDAR data not found for tree: {tree_id}")
                return None
        except Exception as e:
            logger.error(f"Error loading LiDAR data for {tree_id}: {str(e)}")
            return None
    
    def _load_zarr_data(self, dataset_name, filters=None):
        """
        Load data from structured Zarr storage
        
        Args:
            dataset_name (str): Name of the dataset to load
            filters (dict, optional): Filtering criteria to apply
            
        Returns:
            list: Data loaded from storage
        """
        try:
            logger.info(f"Loading data for {dataset_name}")
            # Load from zarr storage
            if dataset_name == 'validation_queue':
                # Should get validation data from zarr store
                return []
            return []
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return []
    
    def _get_data(self, dataset_name, filters=None):
        """Get data from zarr store"""
        data = self._load_zarr_data(dataset_name, filters)
            
        if not data:
            logger.warning(f"No data found for {dataset_name}")
            
        return data
        
    def _apply_filters(self, items, filters):
        """Apply filters to validation queue items"""
        if not filters:
            return items
            
        filtered_items = items
        
        # Apply filters
        if 'status' in filters:
            filtered_items = [item for item in filtered_items if item['status'] == filters['status']]
            
        if 'property_id' in filters:
            filtered_items = [item for item in filtered_items if item['property_id'] == filters['property_id']]
            
        if 'risk_level' in filters:
            # Filter by highest risk factor level
            risk_level = filters['risk_level']
            filtered_items = [
                item for item in filtered_items 
                if any(factor['level'] == risk_level for factor in item.get('riskFactors', []))
            ]
            
        return filtered_items
    
    def get_validation_queue(self, filters=None):
        """Get validation queue items with optional filtering"""
        data = self._get_data('validation_queue')
        filtered_data = self._apply_filters(data, filters)
        
        # Enhance with LiDAR data
        for item in filtered_data:
            lidar_data = self._load_lidar_data(item['tree_id'])
            if lidar_data:
                # Add relevant LiDAR fields to validation queue item if they don't exist
                if not item.get('canopy_width') and lidar_data.get('canopy_width'):
                    item['canopy_width'] = lidar_data['canopy_width']
                if not item.get('canopy_area') and lidar_data.get('canopy_area'):
                    item['canopy_area'] = lidar_data['canopy_area']
                if not item.get('biomass_estimate') and lidar_data.get('biomass_estimate'):
                    item['biomass_estimate'] = lidar_data['biomass_estimate']
                if not item.get('carbon_sequestration') and lidar_data.get('carbon_sequestration'):
                    item['carbon_sequestration'] = lidar_data['carbon_sequestration']
                
                # Add LiDAR links
                item['lidar_data_available'] = True
                
                # Fix URLs to point to the correct locations
                if lidar_data.get('model_url') and lidar_data.get('model_url').startswith('/mock/'):
                    item['model_url'] = lidar_data.get('model_url').replace('/mock/', '/data/')
                else:
                    item['model_url'] = lidar_data.get('model_url')
                    
                if lidar_data.get('point_cloud_url') and lidar_data.get('point_cloud_url').startswith('/mock/'):
                    item['point_cloud_url'] = lidar_data.get('point_cloud_url').replace('/mock/', '/data/')
                else:
                    item['point_cloud_url'] = lidar_data.get('point_cloud_url')
                    
                if lidar_data.get('thumbnail_url') and lidar_data.get('thumbnail_url').startswith('/mock/'):
                    item['thumbnail_url'] = lidar_data.get('thumbnail_url').replace('/mock/', '/data/')
                else:
                    item['thumbnail_url'] = lidar_data.get('thumbnail_url')
                    
                # Log what we're returning
                logger.info(f"Added LiDAR links: {item['model_url']}, {item['point_cloud_url']}, {item['thumbnail_url']}")
        
        return filtered_data
        
    def get_tree_lidar_data(self, tree_id):
        """
        Retrieve LiDAR data for a specific tree
        
        Args:
            tree_id (str): Unique identifier for the tree
            
        Returns:
            dict: LiDAR data including point cloud and analysis results,
                  or None if data not found
        """
        lidar_data = self._load_lidar_data(tree_id)
        if lidar_data:
            # Fix URLs to point to the correct locations
            if lidar_data.get('model_url') and lidar_data.get('model_url').startswith('/mock/'):
                lidar_data['model_url'] = lidar_data.get('model_url').replace('/mock/', '/data/')
                
            if lidar_data.get('point_cloud_url') and lidar_data.get('point_cloud_url').startswith('/mock/'):
                lidar_data['point_cloud_url'] = lidar_data.get('point_cloud_url').replace('/mock/', '/data/')
                
            if lidar_data.get('thumbnail_url') and lidar_data.get('thumbnail_url').startswith('/mock/'):
                lidar_data['thumbnail_url'] = lidar_data.get('thumbnail_url').replace('/mock/', '/data/')
                
            logger.info(f"Returning LiDAR data for tree {tree_id} with updated URLs")
        return lidar_data
    
    def sync_trees_from_zarr(self, area_id):
        """
        Sync tree detection results from Zarr store into validation queue
        
        Args:
            area_id (str): ID of the area to sync
            
        Returns:
            bool: True if sync was successful, False otherwise
        """
        if not ZARR_AVAILABLE:
            logger.warning("Zarr not available, cannot sync trees")
            return False
            
        try:
            # Define path to zarr store
            zarr_path = os.path.join(self.zarr_dir, f"{area_id}_ml_results")
            
            if not os.path.exists(zarr_path):
                logger.warning(f"Zarr store not found at {zarr_path}")
                return False
                
            logger.info(f"Opening zarr store at {zarr_path}")
            
            # Open zarr store
            store = zarr.open(zarr_path, mode='r')
            
            # Get validation queue
            validation_queue = self._get_data('validation_queue')
            
            # Track new items to add to queue
            new_items = []
            
            # Process tree detections from zarr
            if hasattr(store, 'frames'):
                for frame_idx in range(len(store.frames)):
                    frame_group = store.frames[str(frame_idx)]
                    
                    if hasattr(frame_group, 'detections') and hasattr(frame_group, 'features') and hasattr(frame_group, 'risk'):
                        # Extract detections
                        detections = frame_group.detections
                        features = frame_group.features
                        risk = frame_group.risk
                        
                        # Check for tree-specific features
                        if hasattr(features, 'tree_features'):
                            tree_features = features.tree_features
                            
                            # Get number of trees in this frame
                            if hasattr(tree_features, 'bbox') and len(tree_features.bbox) > 0:
                                for tree_idx in range(len(tree_features.bbox)):
                                    # Use the S2 cell ID as the tree ID if available, 
                                    # otherwise generate a fallback ID
                                    if hasattr(tree_features, 'tree_id') and tree_idx < len(tree_features.tree_id):
                                        tree_id = str(tree_features.tree_id[tree_idx])
                                    else:
                                        tree_id = f"{area_id}_{frame_idx}_{tree_idx}"
                                    
                                    # Check if tree is already in validation queue
                                    if not any(item['id'] == tree_id for item in validation_queue):
                                        # Extract tree properties
                                        bbox = tree_features.bbox[tree_idx].tolist() if hasattr(tree_features, 'bbox') else [0, 0, 0, 0]
                                        confidence = float(tree_features.confidence[tree_idx]) if hasattr(tree_features, 'confidence') else 0.0
                                        width = float(tree_features.width[tree_idx]) if hasattr(tree_features, 'width') else 0.0
                                        height = float(tree_features.height[tree_idx]) if hasattr(tree_features, 'height') else 0.0
                                        
                                        # Extract risk factors if available
                                        risk_factors = []
                                        risk_level = "low"
                                        
                                        if hasattr(risk, 'risk_factors'):
                                            risk_group = risk.risk_factors
                                            if hasattr(risk_group, 'tree_id') and len(risk_group.tree_id) > 0:
                                                # For S2 cell IDs, we need to compare strings
                                                tree_risk_idx = []
                                                for i, risk_tree_id in enumerate(risk_group.tree_id):
                                                    if str(risk_tree_id) == tree_id:
                                                        tree_risk_idx.append(i)
                                                
                                                if len(tree_risk_idx) > 0:
                                                    idx = tree_risk_idx[0]
                                                    risk_level = str(risk_group.risk_level[idx]) if hasattr(risk_group, 'risk_level') else "low"
                                                    risk_score = float(risk_group.risk_score[idx]) if hasattr(risk_group, 'risk_score') else 0.0
                                                    
                                                    # Get risk factors
                                                    if hasattr(risk_group, 'risk_factors') and isinstance(risk_group.risk_factors[idx], list):
                                                        for factor in risk_group.risk_factors[idx]:
                                                            risk_factors.append({
                                                                "description": factor,
                                                                "level": risk_level,
                                                                "score": risk_score
                                                            })
                                        
                                        # Set default risk factors if none found
                                        if not risk_factors:
                                            risk_factors.append({
                                                "description": "Unknown risk factor",
                                                "level": risk_level,
                                                "score": 0.0
                                            })
                                        
                                        # Convert bbox from list to a proper format with normalized coordinates
                                        normalized_bbox = bbox
                                        
                                        # Initialize location coordinates
                                        location_coords = None
                                        
                                        # Get tree centroid from segmentation features if available
                                        if hasattr(features, 'segmentation_features'):
                                            seg_features = features.segmentation_features
                                            if hasattr(seg_features, 'centroid') and tree_idx < len(seg_features.centroid):
                                                pixel_centroid = seg_features.centroid[tree_idx]
                                                
                                                # Get transform data from frame metadata if available
                                                if hasattr(frame_group, 'metadata') and hasattr(frame_group.metadata, 'geo_transform'):
                                                    # Apply geo transform to convert pixel coordinates to lat/lng
                                                    transform = frame_group.metadata.geo_transform[:]
                                                    try:
                                                        # Apply affine transform: [lng, lat] = transform * [x, y, 1]
                                                        lng = transform[0] + pixel_centroid[0] * transform[1] + pixel_centroid[1] * transform[2]
                                                        lat = transform[3] + pixel_centroid[0] * transform[4] + pixel_centroid[1] * transform[5]
                                                        location_coords = [lng, lat]
                                                        logger.info(f"Used centroid {pixel_centroid} with transform to get coordinates {location_coords}")
                                                    except Exception as e:
                                                        logger.error(f"Error applying geo transform: {e}")
                                        
                                        # If no location coordinates could be calculated from segmentation centroid,
                                        # skip this tree as we don't want to use placeholder coordinates
                                        if not location_coords:
                                            logger.warning(f"Skipping tree {tree_id} due to missing geo transform or centroid")
                                            continue
                                        
                                        # Create new validation item with proper bounding box and location
                                        new_item = {
                                            "id": tree_id,
                                            "tree_id": tree_id,
                                            "tree_species": "Unknown Species",  # Default, could be detected or inferred
                                            "tree_height": int(height * 5) if height > 0 else 30,  # Convert relative height to feet
                                            "tree_diameter": int(width * 3) if width > 0 else 12,  # Estimate diameter
                                            "property_id": f"property_{area_id}",
                                            "property_type": "Residential",
                                            "status": "pending",
                                            "created_at": datetime.now().isoformat(),
                                            "updated_at": datetime.now().isoformat(),
                                            "confidence": confidence,
                                            "location": location_coords,  # Store coordinates directly for easier mapping
                                            "bbox": normalized_bbox,  # Store the bounding box for visualization
                                            "riskFactors": risk_factors,
                                            "proximity": "Near structure",
                                            "canopy_density": "Medium",
                                            "area_id": area_id,  # Important to track which detection run this came from
                                            "frame_idx": frame_idx,
                                            "detection_metadata": {
                                                "description": f"ML detected in area {area_id}, frame {frame_idx}",
                                                "ml_confidence": confidence,
                                                "detection_time": datetime.now().isoformat()
                                            }
                                        }
                                        
                                        new_items.append(new_item)
                                        logger.info(f"Added new tree {tree_id} to validation queue")
            
            # If new items were found, append them to the validation queue
            if new_items:
                updated_queue = validation_queue + new_items
                
                # Save updated queue to validation data
                validation_path = os.path.join(self.zarr_dir, 'validation_queue.json')
                with open(validation_path, 'w') as f:
                    json.dump(updated_queue, f, indent=2)
                    
                logger.info(f"Added {len(new_items)} new trees to validation queue")
                return True
            else:
                logger.info("No new trees found to add to validation queue")
                return False
                
        except Exception as e:
            logger.error(f"Error syncing trees from zarr: {str(e)}")
            return False
            
    def get_tree_count_from_zarr(self, zarr_path=None):
        """
        Get count of trees in a zarr store
        
        Args:
            zarr_path (str, optional): Path to the zarr store. If None, uses self.zarr_dir
            
        Returns:
            int: Number of trees detected in the zarr store
        """
        if not ZARR_AVAILABLE:
            return 0
            
        try:
            # Use provided zarr path or construct default
            if not zarr_path:
                zarr_path = self.zarr_dir
            
            if not os.path.exists(zarr_path):
                return 0
                
            store = zarr.open(zarr_path, mode='r')
            
            # Count trees across all frames
            tree_count = 0
            
            if hasattr(store, 'frames'):
                for frame_idx in range(len(store.frames)):
                    frame_group = store.frames[str(frame_idx)]
                    
                    if hasattr(frame_group, 'features') and hasattr(frame_group.features, 'tree_features'):
                        tree_features = frame_group.features.tree_features
                        
                        if hasattr(tree_features, 'bbox'):
                            tree_count += len(tree_features.bbox)
            
            return tree_count
            
        except Exception as e:
            logger.error(f"Error counting trees in zarr store: {str(e)}")
            return 0
    
    def update_validation_status(self, item_id, status, notes=None):
        """
        Update validation status for a tree assessment item
        
        Args:
            item_id (str): ID of the validation item to update
            status (str): New status value ('pending', 'approved', 'rejected')
            notes (dict, optional): Additional notes about the validation decision
            
        Returns:
            dict: Updated validation item or None if item not found
        """
        # Load queue items based on application mode
        if self.app_mode == 'test':
            queue_items = self._load_mock_data('validation_queue.json')
        else:
            queue_items = self._load_zarr_data('validation_queue')
        
        # Find and update the specific item
        for item in queue_items:
            if item['id'] == item_id:
                logger.info(f"Updating validation item {item_id} with status {status}")
                updated_item = {**item}
                updated_item['status'] = status
                updated_item['updated_at'] = datetime.now().isoformat()
                
                if notes:
                    updated_item['notes'] = notes
                    
                # Update the item in the mock data file
                if self.app_mode == 'test':
                    # Find the item index
                    item_index = next((i for i, x in enumerate(queue_items) if x['id'] == item_id), None)
                    if item_index is not None:
                        queue_items[item_index] = updated_item
                        
                        # Write back to the mock file
                        try:
                            with open(os.path.join(self.mock_dir, 'validation_queue.json'), 'w') as f:
                                json.dump(queue_items, f, indent=2)
                        except Exception as e:
                            logger.error(f"Error updating mock data: {str(e)}")
                            
                return updated_item
        
        logger.warning(f"Validation item {item_id} not found")
        return None
        
    def sync_validated_trees(self, trees):
        """
        Sync user-validated trees to the validation queue
        
        Args:
            trees (list): List of validated trees
            
        Returns:
            bool: True if sync was successful
        """
        try:
            # Get current validation queue
            validation_queue = self._get_data('validation_queue')
            
            # For each validated tree
            for tree in trees:
                # Check if already in queue
                existing_index = next((i for i, item in enumerate(validation_queue) 
                                     if item.get('id') == tree.get('id')), None)
                
                # If found, update it
                if existing_index is not None:
                    validation_queue[existing_index].update({
                        'user_validated': True,
                        'validation_date': tree.get('validation_date', datetime.now().isoformat()),
                        'species': tree.get('species', 'Unknown Species'),
                        'height': tree.get('height', 30),
                        'diameter': tree.get('diameter', 12),
                        'status': 'approved',  # User validated trees are automatically approved
                        'validated_bbox': tree.get('bbox'),  # Store the validated bounding box
                        'updated_at': datetime.now().isoformat()
                    })
                else:
                    # If not found, create new validation item
                    new_item = {
                        'id': tree['id'],
                        'tree_id': tree['id'],
                        'tree_species': tree.get('species', 'Unknown Species'),
                        'tree_height': tree.get('height', 30),
                        'tree_diameter': tree.get('diameter', 12),
                        'property_id': tree.get('property_id', "property_unknown"),
                        'property_type': tree.get('property_type', 'Residential'),
                        'status': 'approved',
                        'created_at': datetime.now().isoformat(),
                        'updated_at': datetime.now().isoformat(),
                        'confidence': tree.get('confidence', 1.0),
                        'location': tree.get('location', [0, 0]),
                        'bbox': tree.get('bbox'),
                        'riskFactors': tree.get('riskFactors', []),
                        'proximity': tree.get('proximity', 'Near structure'),
                        'canopy_density': tree.get('canopy_density', 'Medium'),
                        'user_validated': True,
                        'validation_date': tree.get('validation_date', datetime.now().isoformat())
                    }
                    
                    validation_queue.append(new_item)
            
            # Save back to validation data
            validation_path = os.path.join(self.zarr_dir, 'validation_queue.json')
            with open(validation_path, 'w') as f:
                json.dump(validation_queue, f, indent=2)
                
            logger.info(f"Synced {len(trees)} validated trees to validation queue")
            return True
                
        except Exception as e:
            logger.error(f"Error syncing validated trees: {str(e)}")
            return False
            
    def get_tree_count_from_temp_zarr(self, area_id):
        """
        Get count of trees in a temporary zarr store
        
        Args:
            area_id (str): ID of the area
            
        Returns:
            int: Number of trees detected in the area
        """
        if not ZARR_AVAILABLE:
            return 0
            
        try:
            # Define path to temporary zarr store
            zarr_path = os.path.join(self.temp_dir, f"{area_id}_temp")
            
            if not os.path.exists(zarr_path):
                return 0
                
            store = zarr.open(zarr_path, mode='r')
            
            # Count trees across all frames
            tree_count = 0
            
            if hasattr(store, 'frames'):
                for frame_idx in range(len(store.frames)):
                    frame_group = store.frames[str(frame_idx)]
                    
                    if hasattr(frame_group, 'features') and hasattr(frame_group.features, 'tree_features'):
                        tree_features = frame_group.features.tree_features
                        
                        if hasattr(tree_features, 'bbox'):
                            tree_count += len(tree_features.bbox)
            
            return tree_count
            
        except Exception as e:
            logger.error(f"Error counting trees in temporary zarr store: {str(e)}")
            return 0