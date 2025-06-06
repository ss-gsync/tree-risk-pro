# backend/services/tree_service.py
import os
import json
import logging
from datetime import datetime
import sys

# Import config
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import APP_MODE, MOCK_DIR, ZARR_DIR

logger = logging.getLogger(__name__)

class TreeService:
    """Service to handle tree data and risk assessments"""
    
    def __init__(self):
        self.app_mode = APP_MODE
        self.mock_dir = MOCK_DIR
        self.zarr_dir = ZARR_DIR
        logger.info(f"TreeService initialized in {self.app_mode} mode")
        
    def _load_mock_data(self, filename):
        """Load mock data from JSON file"""
        try:
            file_path = os.path.join(self.mock_dir, filename)
            if os.path.exists(file_path):
                with open(file_path, 'r') as file:
                    return json.load(file)
            else:
                logger.warning(f"Mock data file not found: {filename}")
                return []
        except Exception as e:
            logger.error(f"Error loading mock data: {str(e)}")
            return []
    
    def _load_zarr_data(self, dataset_name, filters=None):
        """
        Load data from Zarr store
        
        Args:
            dataset_name (str): Name of the dataset to load
            filters (dict, optional): Filtering criteria to apply
            
        Returns:
            list: Data loaded from Zarr store
        """
        try:
            logger.info(f"Loading data from Zarr store for {dataset_name}")
            
            # Only trees are implemented with Zarr storage for now
            if dataset_name == 'trees':
                return self._load_trees_from_zarr()
            
            # Return empty list for other data types
            logger.info(f"No data available for {dataset_name}")
            return []
        except Exception as e:
            logger.error(f"Error loading Zarr data: {str(e)}")
            return []
            
    def _load_trees_from_zarr(self):
        """
        Load trees from the Zarr store using S2 geospatial indexing
        
        Returns:
            list: Tree data from Zarr
        """
        try:
            import zarr
            import os
            import json
            
            trees = []
            zarr_dir = self.zarr_dir
            
            # Check if we have a Zarr store
            if not os.path.exists(zarr_dir):
                logger.warning(f"Zarr directory not found: {zarr_dir}")
                return []
            
            # Look for ML results in the Zarr directory
            # Find unique area_ids (without duplicates from multiple ML result directories)
            area_ids = set()
            detection_dirs = []
            
            for item in os.listdir(zarr_dir):
                # Match all possible ML result directories
                if (item.endswith('_ml_results') or item.endswith('_detection') or 
                    item.endswith('_results') or item.endswith('_trees')) and os.path.isdir(os.path.join(zarr_dir, item)):
                    zarr_path = os.path.join(zarr_dir, item)
                    
                    # Extract area_id from the directory name by removing common suffixes
                    area_id = item
                    for suffix in ['_ml_results', '_detection', '_results', '_trees']:
                        if area_id.endswith(suffix):
                            area_id = area_id.replace(suffix, '')
                            break
                    
                    # Add to set of unique area_ids
                    area_ids.add(area_id)
                    detection_dirs.append((area_id, zarr_path))
            
            # Process each unique area_id, using the most recently modified directory
            for area_id in area_ids:
                # Get all directories for this area_id
                area_dirs = [(aid, path) for aid, path in detection_dirs if aid == area_id]
                
                # Sort by modification time (newest first)
                area_dirs.sort(key=lambda x: os.path.getmtime(x[1]), reverse=True)
                
                # Process the most recent directory for this area_id
                if area_dirs:
                    _, zarr_path = area_dirs[0]
                    logger.info(f"Processing most recent Zarr store for area {area_id}: {zarr_path}")
                    
                    try:
                        # Open the Zarr store
                        store = zarr.open(zarr_path, mode='r')
                        
                        # Check if we have S2 index first
                        if 's2_index' in store:
                            # S2-indexed data
                            logger.info("Found S2 index in Zarr store")
                            
                            # Parse S2 index data
                            if 'cells' in store.s2_index.attrs:
                                try:
                                    s2_cells = json.loads(store.s2_index.attrs['cells'])
                                    
                                    # Process each S2 cell
                                    for s2_token, objects in s2_cells.items():
                                        for obj in objects:
                                            # Create tree object with S2 data
                                            tree = {
                                                'id': obj.get('id', f"s2_{s2_token}_{len(trees)}"),
                                                's2_cell': s2_token,
                                                'area_id': area_id,
                                                'lat': obj.get('lat'),
                                                'lng': obj.get('lng'),
                                                'frame_idx': obj.get('frame_idx', -1)
                                            }
                                            
                                            # Add to results only if we have geo coordinates
                                            if tree['lat'] is not None and tree['lng'] is not None:
                                                trees.append(tree)
                                except Exception as e:
                                    logger.error(f"Error parsing S2 index: {e}")
                        
                        # If no trees from S2 index or no S2 index at all, try frame-based approach
                        if (len([t for t in trees if t.get('area_id') == area_id]) == 0) and 'frames' in store:
                            # Process each frame
                            for frame_idx in store.frames:
                                frame = store.frames[frame_idx]
                                
                                # Look for tree features
                                if 'features' in frame and 'tree_features' in frame.features:
                                    tree_features = frame.features.tree_features
                                    
                                    # Check if we have bounding boxes
                                    if 'bbox' in tree_features:
                                        bboxes = tree_features.bbox[:]
                                        
                                        # Get confidence values if available
                                        confidences = tree_features.confidence[:] if 'confidence' in tree_features else None
                                        
                                        # Get centroids if available
                                        centroids = None
                                        if 'centroid' in tree_features:
                                            centroids = tree_features.centroid[:]
                                        
                                        # Process each tree
                                        for i in range(len(bboxes)):
                                            # Create tree object with information available
                                            tree = {
                                                'id': f"{area_id}_{frame_idx}_{i}",
                                                'area_id': area_id,
                                                'frame_idx': int(frame_idx),
                                                'bbox': bboxes[i].tolist(),
                                                'confidence': float(confidences[i]) if confidences is not None else 0.0
                                            }
                                            
                                            # Add centroid if available
                                            if centroids is not None and i < len(centroids):
                                                tree['centroid'] = centroids[i].tolist()
                                            
                                            # Check if tree has geographic coordinates
                                            if 'trees_json' in frame.attrs:
                                                try:
                                                    tree_json = json.loads(frame.attrs['trees_json'])
                                                    if i < len(tree_json) and 'location' in tree_json[i]:
                                                        # Add geo coordinates
                                                        location = tree_json[i]['location']
                                                        tree['lng'] = location[0]
                                                        tree['lat'] = location[1]
                                                except Exception as e:
                                                    logger.error(f"Error parsing trees_json: {e}")
                                            
                                            # Add to results
                                            trees.append(tree)
                    except Exception as e:
                        logger.error(f"Error processing Zarr store {zarr_path}: {e}")
            
            logger.info(f"Loaded {len(trees)} trees from Zarr stores")
            return trees
            
        except ImportError:
            logger.warning("Zarr not available")
            return []
        except Exception as e:
            logger.error(f"Error loading trees from Zarr: {e}")
            return []
    
    def _get_data(self, dataset_name, filters=None):
        """Get data from Zarr store"""
        return self._load_zarr_data(dataset_name, filters)
    
    def get_properties(self, filters=None):
        """Get all properties"""
        # Return empty list until implemented with real data
        logger.info("Returning empty properties list")
        return []
    
    def get_property(self, property_id):
        """Get property by ID"""
        # Return None until implemented with real data
        logger.info(f"Property {property_id} not found")
        return None
    
    def get_trees_by_property(self, property_id):
        """Get trees for a specific property"""
        # Use filter to get trees by property
        filters = {'property_id': property_id}
        return self._load_zarr_data('trees', filters)
    
    def get_trees(self, filters=None):
        """Get all trees with optional filtering"""
        trees = self._get_data('trees', filters)
        
        # Apply additional filters if provided
        if filters:
            if 'species' in filters and filters['species']:
                trees = [tree for tree in trees if tree.get('species') == filters['species']]
        
        return trees
        
    def get_tree_species(self):
        """Get a list of all unique tree species"""
        trees = self._get_data('trees')
        species_set = set()
        
        for tree in trees:
            if 'species' in tree and tree['species']:
                species_set.add(tree['species'])
                
        # Convert to list and sort alphabetically
        return sorted(list(species_set))
    
    def get_tree(self, tree_id):
        """Get tree by ID"""
        trees = self._get_data('trees')
        for tree in trees:
            if tree['id'] == tree_id:
                return tree
        return None
        
    def update_tree_assessment(self, tree_id, assessment_data):
        """Update tree assessment"""
        tree = self.get_tree(tree_id)
        if not tree:
            return None
            
        if self.app_mode == 'test':
            # In test mode, just return modified data without saving
            updated_tree = {**tree, **assessment_data}
            updated_tree['last_assessment'] = datetime.now().strftime('%Y-%m-%d')
            return updated_tree
        else:
            # Update data in production storage format
            logger.info(f"Updating tree {tree_id} with assessment data")
            updated_tree = {**tree, **assessment_data}
            updated_tree['last_assessment'] = datetime.now().strftime('%Y-%m-%d')
            return updated_tree
        
    def generate_report(self, property_id, options=None):
        """Generate a property report"""
        property_data = self.get_property(property_id)
        if not property_data:
            raise ValueError("Property not found")
            
        trees = self.get_trees_by_property(property_id)
        
        # In a real app, we would generate a real PDF
        # Here we're just returning a mock response
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        report_id = f"report-{property_id}-{timestamp}"
        
        return {
            "report_id": report_id,
            "property_id": property_id,
            "timestamp": datetime.now().isoformat(),
            "tree_count": len(trees),
            "status": "generated",
            "download_url": f"/api/reports/{report_id}"
        }
        
    def get_trees_by_area(self, area_id):
        """
        Get trees for a specific detection area
        
        Args:
            area_id (str): ID of the detection area
            
        Returns:
            list: Trees belonging to the specified area
        """
        try:
            # In test mode, read from mock data
            if self.app_mode == 'test':
                trees = self._load_mock_data('trees.json')
                
                # Filter trees by area_id
                trees_in_area = [tree for tree in trees if tree.get('area_id') == area_id]
                
                # If we don't have any trees for this area, check the validation queue
                if not trees_in_area:
                    validation_queue = self._load_mock_data('validation_queue.json')
                    trees_in_area = [tree for tree in validation_queue if tree.get('area_id') == area_id]
                
                return trees_in_area
            else:
                # Production implementation - query database for trees in area
                filters = {'area_id': area_id}
                trees = self._load_zarr_data('trees', filters)
                
                # Return empty list if no trees found - DO NOT USE MOCK DATA IN PRODUCTION MODE
                if not trees:
                    logger.info(f"No trees found for area {area_id} in Zarr storage - returning empty list")
                    return []
                    
                return trees
                
        except Exception as e:
            logger.error(f"Error getting trees for area {area_id}: {str(e)}")
            # Return empty list, not mock data
            return []
            
    def save_validated_trees(self, area_id, trees):
        """
        Save validated trees to Zarr store with S2 geospatial indexing
        
        Args:
            area_id (str): ID of the detection area
            trees (list): List of validated tree objects
            
        Returns:
            dict: Result of the save operation
        """
        try:
            # Production implementation to update Zarr store
            logger.info(f"Saving {len(trees)} validated trees for area {area_id}")
            
            # Import required modules
            import zarr
            import numpy as np
            import json
            
            # Try to import S2 library for geospatial indexing
            try:
                import s2sphere as s2
                s2_available = True
                logger.info("S2 library available for geospatial indexing")
            except ImportError:
                s2_available = False
                logger.warning("S2 library not available, geospatial indexing will be limited")
            
            # Create or open the Zarr store
            store_path = os.path.join(self.zarr_dir, f"{area_id}_ml_results")
            
            # Check if the store already exists
            if os.path.exists(store_path):
                logger.info(f"Opening existing Zarr store at {store_path}")
                store = zarr.open(store_path, mode='a')
            else:
                logger.info(f"Creating new Zarr store at {store_path}")
                store = zarr.open(store_path, mode='w')
                # Set up initial structure
                store.attrs['area_id'] = area_id
                store.attrs['created_at'] = datetime.now().isoformat()
                store.attrs['tree_count'] = 0
                
                # Create frames group if it doesn't exist
                if 'frames' not in store:
                    store.create_group('frames')
            
            # Get or create S2 index group
            if 's2_index' not in store and s2_available:
                s2_group = store.create_group('s2_index')
                s2_cells = {}
            elif 's2_index' in store and s2_available:
                s2_group = store.s2_index
                if 'cells' in s2_group.attrs:
                    try:
                        s2_cells = json.loads(s2_group.attrs['cells'])
                    except:
                        s2_cells = {}
                else:
                    s2_cells = {}
            
            # Process all validated trees
            saved_trees = []
            
            for tree in trees:
                # Verify the tree has a centroid or location
                if not ('centroid' in tree or ('lat' in tree and 'lng' in tree)):
                    logger.warning(f"Tree missing centroid or geo coordinates, skipping: {tree.get('id')}")
                    continue
                
                # Extract geographic coordinates
                lat, lng = None, None
                
                if 'lat' in tree and 'lng' in tree:
                    lat, lng = tree['lat'], tree['lng']
                elif 'location' in tree and isinstance(tree['location'], list) and len(tree['location']) >= 2:
                    lng, lat = tree['location'][0], tree['location'][1]
                
                # If we have geographic coordinates, create S2 cells
                if s2_available and lat is not None and lng is not None:
                    # Create S2 cell index at multiple levels
                    latlng = s2.LatLng.from_degrees(lat, lng)
                    
                    # Generate tokens for different precision levels
                    for level in [10, 13, 15, 17, 20]:
                        cell_id = s2.CellId.from_lat_lng(latlng).parent(level)
                        token = cell_id.to_token()
                        
                        if token not in s2_cells:
                            s2_cells[token] = []
                        
                        # Store object reference with unique ID
                        s2_id = f"s2_{token}_{len(s2_cells[token])}"
                        
                        # Original frame info
                        frame_idx = tree.get('frame_idx', -1)
                        
                        # Add to S2 cell
                        obj_ref = {
                            'id': tree.get('id', s2_id),
                            'frame_idx': frame_idx,
                            'lat': lat,
                            'lng': lng,
                            'type': 'tree'  # Object type
                        }
                        
                        # Add confidence if available
                        if 'confidence' in tree:
                            obj_ref['confidence'] = tree['confidence']
                            
                        s2_cells[token].append(obj_ref)
                
                # Save the tree data to the appropriate frame
                frame_idx = tree.get('frame_idx', 0)
                frame_key = str(frame_idx)
                
                # Make sure frame exists
                if frame_key not in store.frames:
                    store.frames.create_group(frame_key)
                
                frame = store.frames[frame_key]
                
                # Create or update tree features
                if 'features' not in frame:
                    features = frame.create_group('features')
                else:
                    features = frame.features
                    
                if 'tree_features' not in features:
                    tree_features = features.create_group('tree_features')
                    
                    # Initialize empty datasets
                    tree_features.create_dataset('bbox', data=np.array([]))
                    tree_features.create_dataset('confidence', data=np.array([]))
                    
                    if 'centroid' in tree:
                        tree_features.create_dataset('centroid', data=np.array([]))
                else:
                    tree_features = features.tree_features
                
                # Store validated trees JSON
                if 'trees_json' in frame.attrs:
                    try:
                        trees_json = json.loads(frame.attrs['trees_json'])
                    except:
                        trees_json = []
                else:
                    trees_json = []
                
                # Add or update the tree in JSON
                tree_found = False
                for i, existing_tree in enumerate(trees_json):
                    if existing_tree.get('id') == tree.get('id'):
                        trees_json[i] = tree
                        tree_found = True
                        break
                        
                if not tree_found:
                    trees_json.append(tree)
                
                # Save back to attributes
                frame.attrs['trees_json'] = json.dumps(trees_json)
                frame.attrs['validated'] = True
                frame.attrs['validation_date'] = datetime.now().isoformat()
                
                # Update tree count in frame
                frame.attrs['tree_count'] = len(trees_json)
                
                saved_trees.append(tree)
            
            # Update S2 index
            if s2_available and len(s2_cells) > 0:
                # Store S2 cells as JSON
                s2_group.attrs['cells'] = json.dumps(s2_cells)
                
                # Store cell counts by level
                level_counts = {}
                for token in s2_cells:
                    level = len(token)
                    if level not in level_counts:
                        level_counts[level] = 0
                    level_counts[level] += 1
                
                s2_group.attrs['level_counts'] = json.dumps(level_counts)
            
            # Update total tree count
            store.attrs['tree_count'] = len(saved_trees) + store.attrs.get('tree_count', 0)
            store.attrs['updated_at'] = datetime.now().isoformat()
            
            return {
                "trees_saved": len(saved_trees),
                "total_trees": store.attrs.get('tree_count', 0),
                "zarr_path": store_path
            }
                
        except Exception as e:
            logger.error(f"Error saving validated trees for area {area_id}: {str(e)}")
            raise