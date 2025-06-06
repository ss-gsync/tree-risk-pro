#!/usr/bin/env python3
"""
S2 Coordinate Integration Tester for Tree Detection ML Pipeline

This script verifies that S2 cell indexing is working correctly for precise
coordinate mapping between the detection pipeline and the map visualization.
It focuses on ensuring proper geo-coordinate projection and avoiding whole-image
detections.

The script tests:
1. S2 cell indexing accuracy
2. Coordinate transformation with S2 cells
3. Detection quality with adjusted thresholds
"""

import os
import sys
import json
import asyncio
import logging
from datetime import datetime
import numpy as np
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('s2_coordinate_tester')

# Default paths
DATA_DIR = '/ttt/data/tests'
TEST_DIR = DATA_DIR  # Use the main tests directory
os.makedirs(TEST_DIR, exist_ok=True)

def test_s2_indexing():
    """Test basic S2 cell indexing functionality"""
    logger.info("Testing S2 cell indexing")
    
    try:
        # Import s2sphere directly
        import s2sphere
        
        # Test with Dallas, TX coordinates
        lat, lng = 32.7767, -96.7970
        
        # Create S2 cell at different levels
        s2_cells = {}
        for level_name, level in [
            ('city', 10),
            ('neighborhood', 13),
            ('block', 15),
            ('property', 18)
        ]:
            latlng = s2sphere.LatLng.from_degrees(lat, lng)
            cell = s2sphere.CellId.from_lat_lng(latlng).parent(level)
            s2_cells[level_name] = str(cell.id())
        
        logger.info(f"S2 cells for Dallas, TX ({lat}, {lng}):")
        for level, cell_id in s2_cells.items():
            logger.info(f"  {level}: {cell_id}")
        
        # Get neighboring cells at block level
        block_cell = s2sphere.CellId.from_lat_lng(
            s2sphere.LatLng.from_degrees(lat, lng)).parent(15)
        neighbors = []
        
        # Get face neighbors
        for i in range(4):
            neighbors.append(str(block_cell.get_edge_neighbors()[i].id()))
        
        logger.info(f"Found {len(neighbors)} neighboring cells at block level")
        
        # Create region for the Dallas area
        sw_lat, sw_lng = lat - 0.01, lng - 0.01
        ne_lat, ne_lng = lat + 0.01, lng + 0.01
        
        region = s2sphere.LatLngRect(
            s2sphere.LatLng.from_degrees(sw_lat, sw_lng),
            s2sphere.LatLng.from_degrees(ne_lat, ne_lng)
        )
        
        # Get cells covering the region
        coverer = s2sphere.RegionCoverer()
        coverer.min_level = 15  # block level
        coverer.max_level = 15
        coverer.max_cells = 100
        
        covering_cells = coverer.get_covering(region)
        
        logger.info(f"Found {len(covering_cells)} cells covering the Dallas test area")
        
        # Save test results
        results = {
            "test": "s2_indexing",
            "coordinates": {
                "lat": lat,
                "lng": lng
            },
            "cells": s2_cells,
            "neighbors_count": len(neighbors),
            "covering_count": len(covering_cells),
            "success": True
        }
        
        results_path = os.path.join(TEST_DIR, "s2_indexing_test.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"S2 indexing test results saved to {results_path}")
        
        return True, results
        
    except ImportError as e:
        logger.error(f"s2sphere not installed: {e}")
        return False, {"error": "s2sphere not installed"}
    except Exception as e:
        logger.error(f"Error testing S2 indexing: {e}")
        return False, {"error": str(e)}

def test_s2_coordinate_mapping():
    """Test S2 cell coordinate mapping with the detection service"""
    logger.info("Testing S2 coordinate mapping with detection service")
    
    try:
        # Import necessary components
        sys.path.append('/ttt')
        from tree_ml.dashboard.backend.services.detection_service import S2IndexManager, DetectionService
        
        # Initialize S2 manager and detection service
        s2_manager = S2IndexManager()
        detection_service = DetectionService()
        
        # Test with Dallas, TX coordinates
        lat, lng = 32.7767, -96.7970
        
        # Get S2 cell IDs at different levels
        cell_ids = s2_manager.get_cell_ids_for_tree(lat, lng)
        logger.info(f"S2 cell IDs for ({lat}, {lng}):")
        for level, cell_id in cell_ids.items():
            logger.info(f"  {level}: {cell_id}")
        
        # Test pixel-to-latlon mapping for a synthetic image
        # Create a test mapping
        test_width, test_height = 1200, 800
        test_bounds = [
            [lng - 0.01, lat - 0.01],  # SW corner
            [lng + 0.01, lat + 0.01]   # NE corner
        ]
        
        # Get test mapping
        mapping = {
            'width': test_width,
            'height': test_height,
            'bounds': test_bounds,
            'sw_lng': test_bounds[0][0],
            'sw_lat': test_bounds[0][1],
            'ne_lng': test_bounds[1][0],
            'ne_lat': test_bounds[1][1],
            'lng_per_pixel': (test_bounds[1][0] - test_bounds[0][0]) / test_width,
            'lat_per_pixel': (test_bounds[1][1] - test_bounds[0][1]) / test_height
        }
        
        # Generate S2 cells for the mapping
        s2_cells_mapping = detection_service._get_s2_cells_for_bounds(test_bounds, mapping)
        
        # Add S2 cells to mapping
        if s2_cells_mapping:
            mapping['s2_cells'] = s2_cells_mapping
            logger.info("Successfully added S2 cell mapping")
        else:
            logger.warning("No S2 cell mapping generated")
        
        # Test coordinate conversions with and without S2
        test_points = [
            (0.25, 0.25),  # Top left quarter
            (0.5, 0.5),    # Center
            (0.75, 0.75)   # Bottom right quarter
        ]
        
        conversion_results = []
        for x, y in test_points:
            # Convert normalized coords to geo with standard method
            lng1, lat1 = detection_service._normalize_image_coords_to_geo(x, y, mapping)
            
            # Get S2 cell ID for this location
            s2_cell_id = s2_manager.get_cell_id(lat1, lng1, 'block')
            
            conversion_results.append({
                "normalized": {"x": x, "y": y},
                "geo": {"lat": lat1, "lng": lng1},
                "s2_cell": s2_cell_id
            })
            
            logger.info(f"Normalized ({x}, {y}) -> Geo ({lng1}, {lat1}) -> S2 cell {s2_cell_id}")
        
        # Save test results
        results = {
            "test": "s2_coordinate_mapping",
            "mapping": {
                "width": mapping['width'],
                "height": mapping['height'],
                "bounds": mapping['bounds']
            },
            "s2_cells": s2_cells_mapping is not None,
            "conversions": conversion_results,
            "success": True
        }
        
        results_path = os.path.join(TEST_DIR, "s2_coordinate_mapping_test.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"S2 coordinate mapping test results saved to {results_path}")
        
        return True, results
        
    except ImportError as e:
        logger.error(f"Required modules not available: {e}")
        return False, {"error": f"Import error: {e}"}
    except Exception as e:
        logger.error(f"Error testing S2 coordinate mapping: {e}")
        return False, {"error": str(e)}

async def test_detection_with_s2():
    """Test tree detection with S2 coordinate mapping"""
    logger.info("Testing tree detection with S2 coordinate mapping")
    
    try:
        # Import necessary components
        sys.path.append('/ttt')
        from tree_ml.dashboard.backend.services.detection_service import DetectionService
        
        # Initialize detection service
        detection_service = DetectionService()
        
        # Test with Dallas, TX coordinates
        lat, lng = 32.7767, -96.7970
        
        # Create test bounds
        bounds = [
            [lng - 0.01, lat - 0.01],  # SW corner
            [lng + 0.01, lat + 0.01]   # NE corner
        ]
        
        # Create a simulated view_data
        view_data = {
            'center': [lng, lat],
            'zoom': 17,
            'bounds': bounds,
            'mapWidth': 1200,
            'mapHeight': 800
        }
        
        # Create job ID
        job_id = f"test_{int(datetime.now().timestamp())}"
        
        # Run detect_trees_from_map_view
        logger.info(f"Running detection with job ID: {job_id}")
        result = await detection_service.detect_trees_from_map_view(
            {'viewData': view_data}, job_id
        )
        
        # Check for success
        if result['success']:
            logger.info(f"Detection successful with {len(result.get('trees', []))} trees")
            
            # Check if coordinate system is S2
            coordinate_system = result.get('coordinate_system', 'standard')
            logger.info(f"Using coordinate system: {coordinate_system}")
            
            # Check for whole-image detections
            whole_image_count = 0
            tree_areas = []
            for tree in result.get('trees', []):
                bbox = tree.get('detection', {}).get('bbox', [0, 0, 0, 0])
                if len(bbox) == 4:
                    width = bbox[2] - bbox[0]
                    height = bbox[3] - bbox[1]
                    area = width * height
                    tree_areas.append(area)
                    if area > 0.8:
                        whole_image_count += 1
                        
            logger.info(f"Detected {whole_image_count} whole-image detections")
            if tree_areas:
                logger.info(f"Average tree detection area: {sum(tree_areas)/len(tree_areas):.4f}")
                logger.info(f"Max tree detection area: {max(tree_areas):.4f}")
            
            # Save results
            results = {
                "test": "detection_with_s2",
                "success": True,
                "tree_count": len(result.get('trees', [])),
                "coordinate_system": coordinate_system,
                "whole_image_count": whole_image_count,
                "job_id": job_id,
                "tree_areas": tree_areas
            }
            
            results_path = os.path.join(TEST_DIR, "detection_with_s2_test.json")
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Detection test results saved to {results_path}")
            
            # Also save the full result for further analysis
            full_results_path = os.path.join(TEST_DIR, "detection_with_s2_full.json")
            with open(full_results_path, 'w') as f:
                # We need to manually clean the trees to make them JSON serializable
                clean_result = result.copy()
                if 'trees' in clean_result:
                    clean_trees = []
                    for tree in clean_result['trees']:
                        clean_tree = {}
                        for key, val in tree.items():
                            # Skip segmentation which is too large
                            if key != 'segmentation':
                                clean_tree[key] = val
                        clean_trees.append(clean_tree)
                    clean_result['trees'] = clean_trees
                
                json.dump(clean_result, f, indent=2)
            logger.info(f"Full detection results saved to {full_results_path}")
            
            return True, results
        else:
            logger.error(f"Detection failed: {result.get('message', 'Unknown error')}")
            return False, {"error": result.get('message', 'Unknown error')}
        
    except ImportError as e:
        logger.error(f"Required modules not available: {e}")
        return False, {"error": f"Import error: {e}"}
    except Exception as e:
        logger.error(f"Error testing detection with S2: {e}")
        import traceback
        traceback.print_exc()
        return False, {"error": str(e)}

async def main():
    """Main function to run S2 tests"""
    logger.info("=" * 60)
    logger.info("S2 Coordinate Integration Tester")
    logger.info("=" * 60)
    
    # Track results
    results = {}
    success_count = 0
    tests_run = 0
    
    # Run basic S2 indexing test
    logger.info("=" * 60)
    logger.info("Test 1: Basic S2 Indexing")
    logger.info("=" * 60)
    s2_success, s2_results = test_s2_indexing()
    results["s2_indexing"] = s2_results
    tests_run += 1
    if s2_success:
        success_count += 1
        logger.info("✅ S2 indexing test passed")
    else:
        logger.error("❌ S2 indexing test failed")
    
    # Run S2 coordinate mapping test
    logger.info("=" * 60)
    logger.info("Test 2: S2 Coordinate Mapping")
    logger.info("=" * 60)
    mapping_success, mapping_results = test_s2_coordinate_mapping()
    results["s2_coordinate_mapping"] = mapping_results
    tests_run += 1
    if mapping_success:
        success_count += 1
        logger.info("✅ S2 coordinate mapping test passed")
    else:
        logger.error("❌ S2 coordinate mapping test failed")
    
    # Run detection with S2
    logger.info("=" * 60)
    logger.info("Test 3: Detection with S2")
    logger.info("=" * 60)
    detection_success, detection_results = await test_detection_with_s2()
    results["detection_with_s2"] = detection_results
    tests_run += 1
    if detection_success:
        success_count += 1
        logger.info("✅ Detection with S2 test passed")
    else:
        logger.error("❌ Detection with S2 test failed")
    
    # Save overall test report
    report = {
        "timestamp": datetime.now().isoformat(),
        "tests_run": tests_run,
        "tests_passed": success_count,
        "results": results
    }
    
    # Save directly to the main test directory
    report_path = os.path.join(TEST_DIR, "s2_testing_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
        
    logger.info(f"Test report saved to {report_path}")
    
    # Print summary
    logger.info("=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)
    logger.info(f"Tests run: {tests_run}")
    logger.info(f"Tests passed: {success_count}")
    
    if success_count == tests_run:
        logger.info("✅ All tests passed!")
    else:
        logger.warning(f"⚠️ {tests_run - success_count} tests failed")
    
    return 0 if success_count == tests_run else 1

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unhandled error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)