#!/usr/bin/env python
"""
Tree Detection Diagnostic Tool
Checks a trees.json file to ensure it's properly formatted for the updated frontend display
"""

import os
import sys
import json
import argparse
from pathlib import Path

def check_trees_json(job_id):
    """Check the trees.json file for a detection job"""
    # Constants
    BASE_DIR = "/ttt"
    ML_DIR = os.path.join(BASE_DIR, "data", "ml")
    
    # Tree risk categories
    TREE_CATEGORIES = [
        'healthy_tree',
        'hazardous_tree',
        'dead_tree',
        'low_canopy_tree',
        'pest_disease_tree',
        'flood_prone_tree',
        'utility_conflict_tree',
        'structural_hazard_tree',
        'fire_risk_tree'
    ]
    
    # Construct path to trees.json
    job_dir = os.path.join(ML_DIR, job_id)
    ml_response_dir = os.path.join(job_dir, "ml_response")
    trees_json_path = os.path.join(ml_response_dir, "trees.json")
    
    # Check if directories and file exist
    if not os.path.exists(job_dir):
        print(f"Error: Job directory {job_dir} does not exist")
        return
    
    if not os.path.exists(ml_response_dir):
        print(f"Error: ML response directory {ml_response_dir} does not exist")
        return
    
    if not os.path.exists(trees_json_path):
        print(f"Error: trees.json file {trees_json_path} does not exist")
        return
    
    print(f"Checking trees.json for job {job_id}...")
    print("-" * 60)
    
    try:
        # Load trees.json
        with open(trees_json_path, "r") as f:
            data = json.load(f)
        
        # Check file size
        file_size_mb = os.path.getsize(trees_json_path) / (1024 * 1024)
        print(f"File size: {file_size_mb:.2f} MB")
        
        if file_size_mb > 10:
            print(f"WARNING: File size is large (> 10MB). This may cause performance issues.")
            
            # Check if trees contain segmentation data
            if "trees" in data and isinstance(data["trees"], list) and len(data["trees"]) > 0:
                sample_tree = data["trees"][0]
                if "segmentation" in sample_tree:
                    print("ISSUE DETECTED: Trees contain 'segmentation' data, which should be stored separately.")
                    print("This is likely causing the large file size.")
        
        # Check if data has tree categories
        has_categories = False
        categorized_tree_count = 0
        
        for category in TREE_CATEGORIES:
            if category in data and isinstance(data[category], list):
                has_categories = True
                categorized_tree_count += len(data[category])
                print(f"Category '{category}': {len(data[category])} trees")
        
        if not has_categories:
            print("ISSUE DETECTED: Data does not contain specific tree categories.")
            print("The frontend UI may not display trees in the correct categories.")
        
        # Check if trees have category field
        if "trees" in data and isinstance(data["trees"], list) and len(data["trees"]) > 0:
            total_trees = len(data["trees"])
            trees_with_category = sum(1 for tree in data["trees"] if "category" in tree)
            
            print(f"\nTrees with 'category' field: {trees_with_category}/{total_trees}")
            
            if trees_with_category < total_trees:
                print("ISSUE DETECTED: Some trees don't have the 'category' field.")
                print("These trees may not display correctly in the UI.")
            
            # Sample tree data
            print("\nSample tree data:")
            print(json.dumps(data["trees"][0], indent=2))
        
        # Check for masks directory
        if "metadata" in data and "masks_directory" in data["metadata"]:
            masks_dir = data["metadata"]["masks_directory"]
            full_masks_dir = os.path.join(ML_DIR, masks_dir)
            
            print(f"\nMasks directory: {masks_dir}")
            
            if os.path.exists(full_masks_dir):
                mask_files = os.listdir(full_masks_dir)
                print(f"Found {len(mask_files)} mask files")
                
                # Check if trees reference mask files
                if "trees" in data and isinstance(data["trees"], list) and len(data["trees"]) > 0:
                    trees_with_mask_path = sum(1 for tree in data["trees"] if "mask_path" in tree)
                    print(f"Trees with 'mask_path' field: {trees_with_mask_path}/{total_trees}")
                    
                    if trees_with_mask_path < total_trees:
                        print("ISSUE DETECTED: Some trees don't have the 'mask_path' field.")
                        print("These trees may not be able to load their masks on demand.")
            else:
                print(f"ISSUE DETECTED: Masks directory {full_masks_dir} does not exist")
        
        print("-" * 60)
        print("Recommendations:")
        
        if file_size_mb > 10:
            print("1. Update object_recognition.py to store mask data in separate files")
            print("   - In _add_segmentation method, save masks as NPZ files")
            print("   - In generate_json_results, ensure segmentation data is not included in the trees")
        
        if not has_categories:
            print("2. Update object_recognition.py to categorize trees by type")
            print("   - Add _assign_tree_category method to assign category based on features")
            print("   - In generate_json_results, group trees by category in the output")
        
        if has_categories and categorized_tree_count > 0:
            print("✓ Tree categories are properly defined in trees.json")
        
        print("\nTo fix these issues, run the updated ML pipeline on this image.")
        
    except Exception as e:
        print(f"Error analyzing trees.json: {e}")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Check trees.json for proper formatting")
    parser.add_argument("job_id", help="Detection job ID (e.g., detection_1234567890)")
    args = parser.parse_args()
    
    # Add 'detection_' prefix if not provided
    job_id = args.job_id
    if not job_id.startswith("detection_"):
        job_id = f"detection_{job_id}"
    
    check_trees_json(job_id)