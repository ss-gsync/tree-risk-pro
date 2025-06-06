#!/usr/bin/env python3
"""
Test script for S2 geospatial indexing in the detection service
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# Add backend directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from services.detection_service import DetectionService, S2IndexManager

async def test_s2_index():
    """Test S2 geospatial indexing functionality"""
    
    print("Testing S2 geospatial indexing...")
    
    # Initialize the S2 index manager
    s2_manager = S2IndexManager()
    
    # Test with sample coordinates 
    lat, lng = 32.8600, -96.7800  # Dallas, TX
    
    # Test cell ID generation at different levels
    print("\nTesting cell IDs at different levels:")
    for level_name, level in s2_manager.cell_levels.items():
        cell_id = s2_manager.get_cell_id(lat, lng, level_name)
        print(f"  {level_name.capitalize()} level ({level}): {cell_id}")
    
    # Test cell IDs for a tree
    print("\nGetting all cell IDs for a single location:")
    cell_ids = s2_manager.get_cell_ids_for_tree(lat, lng)
    print(json.dumps(cell_ids, indent=2))
    
    # Test getting cells for bounds
    bounds = [
        [lng - 0.01, lat - 0.01],  # SW corner
        [lng + 0.01, lat + 0.01]   # NE corner
    ]
    
    print("\nGetting cells for bounds at different levels:")
    for level_name in s2_manager.cell_levels:
        cells = s2_manager.get_cells_for_bounds(bounds, level_name)
        print(f"  {level_name.capitalize()} level: {len(cells)} cells")
        if len(cells) <= 10:
            print(f"    Cell IDs: {cells}")
        else:
            print(f"    First 5 cell IDs: {cells[:5]}")
    
    # Test neighbor cells
    print("\nGetting neighbor cells:")
    for level_name in s2_manager.cell_levels:
        neighbors = s2_manager.get_neighbors(lat, lng, level_name, 8)
        print(f"  {level_name.capitalize()} level: {len(neighbors)} neighbors")
        if len(neighbors) <= 8:
            print(f"    Neighbor cell IDs: {neighbors}")
    
    # Initialize the detection service
    print("\nInitializing DetectionService...")
    detection_service = DetectionService()
    
    # Test finding trees (will return empty list if no data exists)
    print("\nTesting tree queries:")
    
    # Get cell ID at property level
    property_cell_id = s2_manager.get_cell_id(lat, lng, 'property')
    
    # Use query_trees_by_s2_cell (this may not return data if there are no stored trees)
    trees = await detection_service.query_trees_by_s2_cell(property_cell_id, 'property')
    print(f"Found {len(trees)} trees in cell ID {property_cell_id}")
    
    # Get neighboring trees
    neighbor_trees = await detection_service.get_neighboring_trees(lat, lng, 'block', 8)
    print(f"Found {len(neighbor_trees)} trees in neighboring cells")
    
    # Test grouping trees (using dummy data if no real trees found)
    print("\nTesting tree grouping:")
    
    # Create dummy trees if no real ones found
    if not trees:
        print("Creating dummy tree data for testing...")
        dummy_trees = []
        for i in range(10):
            # Vary positions slightly within the area
            dummy_lat = lat + (i - 5) * 0.001
            dummy_lng = lng + (i % 3 - 1) * 0.001
            
            # Create S2 cell IDs for each tree
            dummy_cell_ids = s2_manager.get_cell_ids_for_tree(dummy_lat, dummy_lng)
            
            # Create tree with S2 cells
            dummy_tree = {
                'id': f'dummy_tree_{i}',
                'location': [dummy_lng, dummy_lat],
                'risk_level': ['low', 'medium', 'high'][i % 3],
                's2_cells': dummy_cell_ids
            }
            dummy_trees.append(dummy_tree)
        
        # Group the dummy trees
        grouped_trees = await detection_service.group_trees_by_s2_cell(dummy_trees, 'block')
        print(f"Grouped {len(dummy_trees)} dummy trees into {len(grouped_trees)} S2 cells")
        
        # Calculate statistics for the groups
        cell_stats = detection_service.calculate_s2_statistics(grouped_trees)
        print("\nS2 cell statistics:")
        print(json.dumps(cell_stats, indent=2))
    else:
        # Group the real trees
        grouped_trees = await detection_service.group_trees_by_s2_cell(trees, 'block')
        print(f"Grouped {len(trees)} real trees into {len(grouped_trees)} S2 cells")
        
        # Calculate statistics for the groups
        cell_stats = detection_service.calculate_s2_statistics(grouped_trees)
        print("\nS2 cell statistics:")
        print(json.dumps(cell_stats, indent=2))
    
    print("\nS2 geospatial indexing tests completed.")

if __name__ == "__main__":
    # Run the async test function
    asyncio.run(test_s2_index())