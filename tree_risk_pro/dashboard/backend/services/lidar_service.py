# backend/services/lidar_service.py
import os
import json
import logging
import random
import math
import sys

# Import config
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import APP_MODE, MOCK_DIR, ZARR_DIR, LIDAR_DIR

logger = logging.getLogger(__name__)

class LidarService:
    """Service to handle LiDAR data processing"""
    
    def __init__(self):
        self.app_mode = APP_MODE
        self.lidar_dir = LIDAR_DIR
        self.mock_dir = MOCK_DIR
        self.zarr_dir = ZARR_DIR
        logger.info(f"LidarService initialized in {self.app_mode} mode")
        
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
        """Load data from Zarr store (stub for future implementation)"""
        try:
            logger.info(f"Attempting to load Zarr data for {dataset_name}")
            # This is a placeholder for future Zarr implementation
            # In production, we'd use zarr.open_group() and access the arrays
            
            # For now, we'll fall back to mock data
            logger.warning("Zarr implementation not complete. Falling back to mock data.")
            return self._load_mock_data(f"{dataset_name}.json")
        except Exception as e:
            logger.error(f"Error loading Zarr data: {str(e)}")
            return []
    
    def _get_data(self, dataset_name, filters=None):
        """Get data based on app mode"""
        if self.app_mode == 'test':
            return self._load_mock_data(f"{dataset_name}.json")
        else:
            return self._load_zarr_data(dataset_name, filters)
    
    def get_lidar_data(self, property_id=None, tree_id=None):
        """
        Get LiDAR point cloud data for a property or specific tree
        
        Args:
            property_id (str, optional): ID of the property for which to retrieve LiDAR data
            tree_id (str, optional): ID of a specific tree for which to retrieve LiDAR data
            
        Returns:
            dict: JSON structure containing LiDAR point cloud data
        """
        # If tree_id is provided, try to get specific tree LiDAR data
        if tree_id:
            return self.get_tree_lidar_data(tree_id)
        
        # Check if we have a cached file for this property
        json_file_path = os.path.join(self.lidar_dir, f"{property_id}.json")
        
        if os.path.exists(json_file_path):
            try:
                with open(json_file_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading cached LiDAR data: {str(e)}")
                # If loading fails, generate new data
        
        # Try loading LiDAR data from Zarr store in production mode
        if self.app_mode == 'production':
            try:
                logger.info(f"Attempting to load LiDAR data from Zarr for property {property_id}")
                # This is a placeholder for future Zarr implementation
                # In production, we would load LiDAR point cloud data from Zarr
                logger.warning("Zarr LiDAR implementation not complete. Falling back to generated data.")
                # Continue to generated data as fallback
            except Exception as e:
                logger.error(f"Error loading LiDAR from Zarr: {str(e)}")
        
        # Get property center from properties data
        properties = self._get_data('properties')
        center_coords = None
        for prop in properties:
            if prop['id'] == property_id:
                center_coords = prop['center']
                break
        
        if not center_coords:
            # Use default coords if property not found
            center_coords = [-96.7970, 32.7767]  # Dallas
            logger.warning(f"Property {property_id} not found. Using default coordinates.")
        
        # Generate mock point cloud data
        num_points = 1000
        x_spread = 0.005
        y_spread = 0.005
        z_max = 30  # max height in meters
        
        # Get trees on the property for more realistic LiDAR data
        trees = self._get_data('trees')
        property_trees = [tree for tree in trees if tree['property_id'] == property_id]
        
        points = []
        
        # Generate ground-level points
        for i in range(num_points // 2):
            x = center_coords[0] + (random.random() - 0.5) * x_spread
            y = center_coords[1] + (random.random() - 0.5) * y_spread
            z = random.random() * 0.5  # Ground-level noise
            intensity = 0.2 + (random.random() * 0.2)  # Lower intensity for ground
            classification = 2  # Ground points in LAS standard
            
            points.append({
                "x": x,
                "y": y,
                "z": z,
                "intensity": intensity,
                "classification": classification
            })
        
        # Generate tree points if we have tree data
        if property_trees:
            for tree in property_trees:
                tree_x, tree_y = tree['location']
                tree_height = tree['height']
                canopy_spread = tree.get('canopy_spread', 20)  # Default if not specified
                
                # Number of points proportional to tree size
                tree_points = int((tree_height / 30) * 200)
                
                for i in range(tree_points):
                    # Generate points in a cone shape for the tree
                    # Distance from trunk center
                    distance = random.random() * (canopy_spread / 2)
                    # Angle around trunk
                    angle = random.random() * 2 * math.pi
                    # Height along trunk
                    height_pct = random.random()
                    
                    # Calculate x, y based on distance and angle
                    x = tree_x + distance * math.cos(angle) * (height_pct * 0.5 + 0.5)
                    y = tree_y + distance * math.sin(angle) * (height_pct * 0.5 + 0.5)
                    
                    # Height: taller near trunk, shorter at edges
                    z = tree_height * (height_pct * 0.9 + 0.1)
                    
                    # Intensity higher for leaves, lower for trunk
                    if distance < 1 and height_pct < 0.7:
                        intensity = 0.3 + (random.random() * 0.2)  # Trunk
                        classification = 3  # Low vegetation
                    else:
                        intensity = 0.6 + (random.random() * 0.4)  # Leaves/branches
                        classification = 5  # High vegetation
                    
                    points.append({
                        "x": x,
                        "y": y,
                        "z": z,
                        "intensity": intensity,
                        "classification": classification
                    })
        else:
            # If no tree data, generate random vegetation points
            logger.warning(f"No trees found for property {property_id}. Generating random vegetation.")
            for i in range(num_points // 2):
                x = center_coords[0] + (random.random() - 0.5) * x_spread
                y = center_coords[1] + (random.random() - 0.5) * y_spread
                z = 2 + (random.random() * z_max)  # Above ground
                intensity = 0.5 + (random.random() * 0.5)
                classification = 5  # High vegetation
                
                points.append({
                    "x": x,
                    "y": y,
                    "z": z,
                    "intensity": intensity,
                    "classification": classification
                })
        
        # Create result
        result = {
            "property_id": property_id,
            "points": points,
            "pointCount": len(points),
            "bounds": {
                "min": [
                    center_coords[0] - x_spread/2,
                    center_coords[1] - y_spread/2,
                    0
                ],
                "max": [
                    center_coords[0] + x_spread/2,
                    center_coords[1] + y_spread/2,
                    z_max
                ]
            }
        }
        
        # Save to cache file
        try:
            os.makedirs(self.lidar_dir, exist_ok=True)
            with open(json_file_path, 'w') as f:
                json.dump(result, f)
        except Exception as e:
            logger.error(f"Error saving LiDAR data: {str(e)}")
        
        return result
    
    def get_tree_lidar_data(self, tree_id):
        """
        Get detailed LiDAR data specific to an individual tree
        
        Args:
            tree_id (str): Unique identifier for the tree
            
        Returns:
            dict: LiDAR analysis data for the specified tree including measurements
                  and detected risk indicators
        """
        # First check in lidar directory for tree-specific LiDAR data
        if self.app_mode == 'test':
            mock_lidar_path = os.path.join(self.lidar_dir, f"{tree_id}.json")
            if os.path.exists(mock_lidar_path):
                try:
                    with open(mock_lidar_path, 'r') as f:
                        logger.info(f"Loaded mock LiDAR data for tree {tree_id}")
                        return json.load(f)
                except Exception as e:
                    logger.error(f"Error loading mock tree LiDAR data: {str(e)}")
                    # Continue to fallback generation
        
        # In production mode, try to load from Zarr store
        if self.app_mode == 'production':
            try:
                logger.info(f"Attempting to load Zarr LiDAR data for tree {tree_id}")
                # Placeholder for future implementation
                logger.warning("Zarr tree LiDAR implementation not complete. Falling back to mock data.")
                # Try mock data again as fallback
                mock_lidar_path = os.path.join(self.lidar_dir, f"{tree_id}.json")
                if os.path.exists(mock_lidar_path):
                    with open(mock_lidar_path, 'r') as f:
                        return json.load(f)
            except Exception as e:
                logger.error(f"Error loading Zarr tree LiDAR: {str(e)}")
        
        # If we reach here, we need to generate placeholder data
        # Get tree data first
        trees = self._get_data('trees')
        tree = None
        for t in trees:
            if t['id'] == tree_id:
                tree = t
                break
        
        if not tree:
            logger.error(f"Tree {tree_id} not found for LiDAR data")
            return {"error": "Tree not found"}
        
        # Generate a basic placeholder response
        species = tree.get('species', 'Unknown')
        height = tree.get('height', 30)
        dbh = tree.get('dbh', 20)
        canopy_spread = tree.get('canopy_spread', 20)
        risk_factors = tree.get('risk_factors', [])
        
        # Map risk factors to risk indicators
        risk_indicators = {}
        for risk in risk_factors:
            risk_type = risk.get('type', 'unknown')
            risk_level = risk.get('level', 'low')
            risk_desc = risk.get('description', '')
            
            risk_indicators[risk_type] = {
                "detected": True,
                "severity": risk_level,
                "description": risk_desc,
                "confidence": 0.8 + (random.random() * 0.15)
            }
        
        result = {
            "tree_id": tree_id,
            "species": species,
            "scan_date": tree.get('last_assessment', '2025-03-01'),
            "scan_type": "Aerial + Ground",
            "point_count": int(5000 + (random.random() * 10000)),
            "height": height + (random.random() * 0.5),
            "dbh": dbh + (random.random() * 0.5),
            "canopy_width": canopy_spread + (random.random() * 1.0),
            "risk_indicators": risk_indicators,
            "thumbnail_url": f"/assets/lidar/{tree_id}-thumb.png",
            "model_url": f"/assets/models/{tree_id}.glb"
        }
        
        return result