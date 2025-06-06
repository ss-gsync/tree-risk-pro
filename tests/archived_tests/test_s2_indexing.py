#!/usr/bin/env python3
"""
S2 Geospatial Indexing Test

This script tests the S2 geospatial indexing integration with tree detection data.
It creates synthetic tree detection data and applies S2 indexing to demonstrate
the grouping and spatial querying capabilities.
"""

import os
import sys
import json
import logging
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('s2_indexing_test')

# Add necessary paths
sys.path.append('/ttt')

# Import the S2 indexing manager
from tree_ml.dashboard.backend.services.detection_service import S2IndexManager

# Test constants - Central Park area
TEST_BOUNDS = [
    [-73.9776, 40.7614],  # SW corner
    [-73.9499, 40.7968]   # NE corner
]

def create_synthetic_trees(num_trees=50, bounds=TEST_BOUNDS):
    """Create synthetic tree detection data within the given bounds"""
    logger.info(f"Creating {num_trees} synthetic trees within bounds")
    
    # Extract bounds coordinates
    sw_lng, sw_lat = bounds[0]
    ne_lng, ne_lat = bounds[1]
    
    # Calculate spans
    lng_span = ne_lng - sw_lng
    lat_span = ne_lat - sw_lat
    
    # Create random tree data
    trees = []
    for i in range(num_trees):
        # Random position within bounds
        lng = sw_lng + np.random.random() * lng_span
        lat = sw_lat + np.random.random() * lat_span
        
        # Random tree properties
        height = np.random.uniform(5.0, 30.0)
        confidence = np.random.uniform(0.6, 0.95)
        
        # Random risk level with bias towards medium
        risk_level_probs = np.array([0.2, 0.6, 0.2])  # [low, medium, high]
        risk_level_idx = np.random.choice(3, p=risk_level_probs)
        risk_level = ['low', 'medium', 'high'][risk_level_idx]
        
        # Create tree object
        tree = {
            'id': f"tree_{i+1}",
            'location': [lng, lat],
            'height': height,
            'confidence': confidence,
            'risk_level': risk_level,
            'species': 'Unknown',
            'bbox': [0.0, 0.0, 0.0, 0.0]  # Placeholder bounding box
        }
        
        trees.append(tree)
    
    logger.info(f"Created {len(trees)} synthetic trees")
    return trees

def add_s2_indexing(trees):
    """Add S2 cell IDs to tree detections"""
    logger.info("Adding S2 indexing to trees")
    
    # Create S2 index manager
    s2_manager = S2IndexManager()
    
    # Add S2 cell IDs to each tree
    for tree in trees:
        if 'location' in tree:
            lng, lat = tree['location']
            # Get S2 cell IDs for all levels
            s2_cells = s2_manager.get_cell_ids_for_tree(lat, lng)
            tree['s2_cells'] = s2_cells
    
    logger.info(f"Added S2 indexing to {len(trees)} trees")
    return trees

def group_trees_by_s2_cell(trees, level='block'):
    """Group trees by S2 cell at the specified level"""
    logger.info(f"Grouping trees by S2 cell at {level} level")
    
    # Create groups
    grouped_trees = {}
    
    # Group trees by cell ID
    for tree in trees:
        if 's2_cells' in tree and level in tree['s2_cells']:
            cell_id = tree['s2_cells'][level]
            
            if cell_id not in grouped_trees:
                grouped_trees[cell_id] = []
            
            grouped_trees[cell_id].append(tree)
    
    logger.info(f"Grouped {len(trees)} trees into {len(grouped_trees)} S2 cells")
    return grouped_trees

def calculate_s2_statistics(grouped_trees):
    """Calculate statistics for each S2 cell group"""
    logger.info(f"Calculating statistics for {len(grouped_trees)} S2 cells")
    
    # Create statistics for each cell
    cell_stats = {}
    
    # Risk level mapping
    risk_levels = {'low': 1, 'medium': 2, 'high': 3}
    
    # Process each cell
    for cell_id, trees in grouped_trees.items():
        # Skip empty cells
        if not trees:
            continue
        
        # Calculate tree count
        tree_count = len(trees)
        
        # Calculate average location
        lats = [tree['location'][1] for tree in trees]
        lngs = [tree['location'][0] for tree in trees]
        avg_lat = sum(lats) / tree_count
        avg_lng = sum(lngs) / tree_count
        
        # Calculate risk statistics
        risk_values = []
        for tree in trees:
            risk_level = tree.get('risk_level', 'medium').lower()
            risk_value = risk_levels.get(risk_level, 2)
            risk_values.append(risk_value)
        
        avg_risk = sum(risk_values) / tree_count
        
        # Determine dominant risk level
        if avg_risk < 1.5:
            dominant_risk = 'low'
        elif avg_risk < 2.5:
            dominant_risk = 'medium'
        else:
            dominant_risk = 'high'
        
        # Store cell statistics
        cell_stats[cell_id] = {
            'tree_count': tree_count,
            'center': [avg_lng, avg_lat],
            'avg_risk_value': avg_risk,
            'dominant_risk': dominant_risk,
            'trees': [tree['id'] for tree in trees]
        }
    
    logger.info(f"Calculated statistics for {len(cell_stats)} S2 cells")
    return cell_stats

def find_neighboring_trees(trees, lat, lng, level='block'):
    """Find trees in neighboring S2 cells"""
    logger.info(f"Finding trees in neighboring S2 cells at {level} level")
    
    # Create S2 index manager
    s2_manager = S2IndexManager()
    
    # Get cell ID for the reference point
    reference_cell = s2_manager.get_cell_id(lat, lng, level)
    logger.info(f"Reference cell ID: {reference_cell}")
    
    # Get neighboring cells
    neighbor_cells = s2_manager.get_neighbors(lat, lng, level, k=8)
    logger.info(f"Found {len(neighbor_cells)} neighboring cells")
    
    # Include the reference cell itself
    all_cells = [reference_cell] + neighbor_cells
    
    # Find trees in these cells
    trees_in_neighbors = []
    for tree in trees:
        if 's2_cells' in tree and level in tree['s2_cells']:
            cell_id = tree['s2_cells'][level]
            if cell_id in all_cells:
                trees_in_neighbors.append(tree)
    
    logger.info(f"Found {len(trees_in_neighbors)} trees in neighboring cells")
    return trees_in_neighbors

def test_s2_hierarchical_lookup(trees):
    """Test hierarchical S2 lookup at different zoom levels"""
    logger.info("Testing hierarchical S2 lookup at different zoom levels")
    
    # Create S2 index manager
    s2_manager = S2IndexManager()
    
    # Define a specific point in Central Park
    test_lat, test_lng = 40.7789, -73.9692
    logger.info(f"Using test point: [{test_lat}, {test_lng}]")
    
    # Test each level
    levels = ['city', 'neighborhood', 'block', 'property']
    trees_at_levels = {}
    
    for level in levels:
        # Get cell ID for the test point
        cell_id = s2_manager.get_cell_id(test_lat, test_lng, level)
        logger.info(f"Level {level}: Cell ID {cell_id}")
        
        # Find trees in this cell
        trees_in_cell = []
        for tree in trees:
            if 's2_cells' in tree and level in tree['s2_cells']:
                if tree['s2_cells'][level] == cell_id:
                    trees_in_cell.append(tree)
        
        trees_at_levels[level] = trees_in_cell
        logger.info(f"Found {len(trees_in_cell)} trees at {level} level")
    
    # Analyze the hierarchical relationship
    logger.info("\nHierarchical relationship analysis:")
    for i, level1 in enumerate(levels):
        for level2 in levels[i+1:]:
            overlap = set(t['id'] for t in trees_at_levels[level1]) & set(t['id'] for t in trees_at_levels[level2])
            logger.info(f"Trees in both {level1} and {level2}: {len(overlap)}")
    
    return trees_at_levels

def save_results(trees, grouped_trees, cell_stats, output_dir='/ttt/data/temp/s2_indexing_test'):
    """Save test results to files"""
    logger.info(f"Saving results to {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save trees
    with open(os.path.join(output_dir, 'trees.json'), 'w') as f:
        json.dump(trees, f, indent=2)
    
    # Save grouped trees (convert keys to strings for JSON)
    with open(os.path.join(output_dir, 'grouped_trees.json'), 'w') as f:
        # Convert to serializable format (cell IDs as keys)
        serializable_grouped = {str(k): v for k, v in grouped_trees.items()}
        json.dump(serializable_grouped, f, indent=2)
    
    # Save cell statistics (convert keys to strings for JSON)
    with open(os.path.join(output_dir, 's2_cell_stats.json'), 'w') as f:
        # Convert to serializable format (cell IDs as keys)
        serializable_stats = {str(k): v for k, v in cell_stats.items()}
        json.dump(serializable_stats, f, indent=2)
    
    logger.info(f"Results saved to {output_dir}")
    return output_dir

def main():
    """Run the S2 indexing test"""
    logger.info("Starting S2 Geospatial Indexing Test")
    
    # Create synthetic trees
    trees = create_synthetic_trees(num_trees=100)
    
    # Add S2 indexing
    trees = add_s2_indexing(trees)
    
    # Group trees by S2 cell at block level
    grouped_trees = group_trees_by_s2_cell(trees, level='block')
    
    # Calculate statistics for each cell
    cell_stats = calculate_s2_statistics(grouped_trees)
    
    # Test neighbor finding
    central_park_lat, central_park_lng = 40.7789, -73.9692
    neighboring_trees = find_neighboring_trees(trees, central_park_lat, central_park_lng, level='block')
    
    # Test hierarchical lookup
    hierarchical_trees = test_s2_hierarchical_lookup(trees)
    
    # Save results
    output_dir = save_results(trees, grouped_trees, cell_stats)
    
    logger.info(f"\nS2 Indexing Test completed successfully!")
    logger.info(f"Results saved to {output_dir}")
    
    # Print summary statistics
    logger.info("\nSummary Statistics:")
    logger.info(f"Total trees: {len(trees)}")
    logger.info(f"S2 cells at block level: {len(grouped_trees)}")
    logger.info(f"Trees near Central Park: {len(neighboring_trees)}")
    logger.info(f"Trees at city level: {len(hierarchical_trees['city'])}")
    logger.info(f"Trees at neighborhood level: {len(hierarchical_trees['neighborhood'])}")
    logger.info(f"Trees at block level: {len(hierarchical_trees['block'])}")
    logger.info(f"Trees at property level: {len(hierarchical_trees['property'])}")

if __name__ == "__main__":
    main()