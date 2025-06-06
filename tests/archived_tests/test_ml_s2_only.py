#!/usr/bin/env python3
"""
Test script for S2 indexing with the NYC satellite image
This bypasses the ML detection pipeline completely and focuses on S2 indexing
"""

import asyncio
import json
import os
import sys
from pathlib import Path
import time
from datetime import datetime
import random

# Add necessary paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))
sys.path.append('/ttt')

# Import only the S2IndexManager to test S2 indexing directly
from services.detection_service import S2IndexManager

async def test_s2_with_nyc_image():
    """Test S2 indexing with the NYC satellite image"""
    
    print("\n=== Testing S2 Indexing with NYC Satellite Image ===\n")
    
    # Initialize the S2 index manager
    s2_manager = S2IndexManager()
    
    # Image path and information
    image_path = "/ttt/data/temp/ml_results_test/satellite_40.7791_-73.96375_16_1746715815.jpg"
    
    # NYC area bounds (from the filename coordinates)
    lat = 40.7791
    lng = -73.96375
    
    # Create a slightly larger bounding box around the point
    bounds = [
        [lng - 0.005, lat - 0.005],  # SW corner
        [lng + 0.005, lat + 0.005]   # NE corner
    ]
    
    print(f"Input image: {image_path}")
    print(f"Center coordinates: [{lat}, {lng}]")
    print(f"Bounds: SW {bounds[0]}, NE {bounds[1]}")
    
    # Create job ID with timestamp
    timestamp = int(time.time())
    job_id = f"s2_test_{timestamp}"
    
    # Create output directory
    output_dir = f"/ttt/data/temp/s2_test_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get S2 cell IDs for the center point
    print("\nS2 cell IDs for center point:")
    for level_name, level in s2_manager.cell_levels.items():
        cell_id = s2_manager.get_cell_id(lat, lng, level_name)
        print(f"  {level_name.capitalize()} level ({level}): {cell_id}")
    
    # Get all cells covering the bounds
    print("\nS2 cells covering the bounds:")
    bounds_cells = {}
    for level_name in s2_manager.cell_levels:
        cells = s2_manager.get_cells_for_bounds(bounds, level_name)
        bounds_cells[level_name] = cells
        print(f"  {level_name.capitalize()}: {len(cells)} cells")
        if len(cells) <= 10:
            print(f"    {cells}")
        else:
            print(f"    First 5: {cells[:5]}")
    
    # Create simulated trees within the bounds
    print("\nCreating simulated trees within bounds...")
    
    # Use specific NYC tree species for more realistic simulation
    nyc_tree_species = [
        "London Planetree", "Norway Maple", "Pin Oak", "Honey Locust", 
        "Callery Pear", "Red Maple", "American Linden", "Ginkgo", 
        "Japanese Zelkova", "Silver Maple"
    ]
    
    # Create simulated trees
    num_trees = 25
    simulated_trees = []
    
    for i in range(num_trees):
        # Random location within bounds
        tree_lat = bounds[0][1] + (bounds[1][1] - bounds[0][1]) * random.random()
        tree_lng = bounds[0][0] + (bounds[1][0] - bounds[0][0]) * random.random()
        
        # Get S2 cells for this tree
        s2_cells = s2_manager.get_cell_ids_for_tree(tree_lat, tree_lng)
        
        # Generate realistic height and risk level
        height = 20 + 15 * random.random()  # 20-35 feet
        risk_levels = ['low', 'medium', 'high']
        risk_weights = [0.5, 0.3, 0.2]  # More trees are low risk
        risk_level = random.choices(risk_levels, weights=risk_weights, k=1)[0]
        
        # Generate risk factors based on risk level
        risk_factors = []
        if risk_level in ['medium', 'high']:
            potential_factors = [
                "Dead branches", "Trunk decay", "Root damage", 
                "Leaning", "Cracks", "Previous failure",
                "Weak branch union", "Canopy dieback"
            ]
            # High risk trees have more factors
            num_factors = 3 if risk_level == 'high' else 1
            risk_factors = random.sample(potential_factors, min(num_factors, len(potential_factors)))
        
        # Create tree object
        tree = {
            'id': f"{job_id}_tree_{i}",
            'location': [tree_lng, tree_lat],
            'bbox': [0.2, 0.2, 0.3, 0.3],  # Dummy bounding box
            'confidence': 0.7 + 0.3 * random.random(),
            'height': height,
            'species': random.choice(nyc_tree_species),
            'risk_level': risk_level,
            'risk_factors': risk_factors,
            's2_cells': s2_cells
        }
        simulated_trees.append(tree)
    
    print(f"Created {len(simulated_trees)} simulated trees")
    
    # Group trees by S2 cell at block level
    print("\nGrouping trees by S2 cell (block level)...")
    grouped_trees = {}
    
    for tree in simulated_trees:
        # Get the cell ID for this tree at the block level
        cell_id = tree['s2_cells']['block']
        
        # Add to the appropriate group
        if cell_id not in grouped_trees:
            grouped_trees[cell_id] = []
            
        grouped_trees[cell_id].append(tree)
    
    print(f"Grouped {len(simulated_trees)} trees into {len(grouped_trees)} S2 cells")
    
    # Calculate statistics for each cell group
    print("\nCalculating statistics for cell groups...")
    cell_stats = {}
    
    # Risk level mapping
    risk_levels = {
        'low': 1,
        'medium': 2,
        'high': 3
    }
    
    for cell_id, trees in grouped_trees.items():
        # Calculate basic statistics
        tree_count = len(trees)
        
        # Calculate average position
        avg_lat = sum(tree['location'][1] for tree in trees) / tree_count
        avg_lng = sum(tree['location'][0] for tree in trees) / tree_count
        
        # Calculate average risk level
        risk_values = [risk_levels[tree['risk_level']] for tree in trees]
        avg_risk = sum(risk_values) / tree_count if risk_values else 0
        
        # Determine dominant risk level
        if avg_risk < 1.5:
            dominant_risk = 'low'
        elif avg_risk < 2.5:
            dominant_risk = 'medium'
        else:
            dominant_risk = 'high'
        
        # Count trees by risk level
        risk_counts = {level: 0 for level in risk_levels.keys()}
        for tree in trees:
            risk_counts[tree['risk_level']] += 1
        
        # Calculate dominant species
        species_counts = {}
        for tree in trees:
            species = tree['species']
            if species not in species_counts:
                species_counts[species] = 0
            species_counts[species] += 1
        
        # Get the most common species
        dominant_species = max(species_counts.items(), key=lambda x: x[1])[0] if species_counts else "Unknown"
        
        # Store the cell statistics
        cell_stats[cell_id] = {
            'tree_count': tree_count,
            'center': [avg_lng, avg_lat],
            'avg_risk_value': avg_risk,
            'dominant_risk': dominant_risk,
            'risk_counts': risk_counts,
            'dominant_species': dominant_species,
            'trees': [tree['id'] for tree in trees]
        }
    
    # Print statistics for the first three cells
    print("\nStatistics for the first 3 cell groups:")
    
    for i, (cell_id, stats) in enumerate(list(cell_stats.items())[:3]):
        print(f"\nS2 Cell {i+1}: {cell_id}")
        print(f"  Tree Count: {stats['tree_count']}")
        print(f"  Center: {stats['center']}")
        print(f"  Dominant Risk: {stats['dominant_risk']} (avg value: {stats['avg_risk_value']:.2f})")
        print(f"  Risk Distribution: {stats['risk_counts']}")
        print(f"  Dominant Species: {stats['dominant_species']}")
        print(f"  Trees: {stats['trees']}")
    
    # Get neighboring cells of a specific location
    print("\nFinding neighboring cells:")
    
    # Choose a point near the center
    neighbor_lat = lat + 0.001
    neighbor_lng = lng + 0.001
    
    # Get S2 cell ID for this point
    cell_id = s2_manager.get_cell_id(neighbor_lat, neighbor_lng, 'block')
    print(f"S2 cell ID for point [{neighbor_lat}, {neighbor_lng}]: {cell_id}")
    
    # Get neighboring cells
    neighbors = s2_manager.get_neighbors(neighbor_lat, neighbor_lng, 'block')
    print(f"Found {len(neighbors)} neighboring cells:")
    for i, neighbor_id in enumerate(neighbors):
        print(f"  Neighbor {i+1}: {neighbor_id}")
    
    # Save the results to a JSON file
    print("\nSaving results...")
    
    results = {
        'metadata': {
            'image': image_path,
            'center': [lng, lat],
            'bounds': bounds,
            'job_id': job_id,
            'timestamp': datetime.now().isoformat()
        },
        'center_point_cells': {level: s2_manager.get_cell_id(lat, lng, level) 
                              for level in s2_manager.cell_levels},
        'bounds_cells': bounds_cells,
        'simulated_trees': simulated_trees,
        'grouped_trees': {str(k): [t['id'] for t in v] for k, v in grouped_trees.items()},
        'cell_stats': cell_stats,
        'neighbor_point': [neighbor_lng, neighbor_lat],
        'neighbor_cell': cell_id,
        'neighboring_cells': neighbors
    }
    
    output_file = os.path.join(output_dir, "s2_indexing_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")
    print("\n=== S2 Indexing Test Completed ===")

if __name__ == "__main__":
    # Run the async test function
    asyncio.run(test_s2_with_nyc_image())