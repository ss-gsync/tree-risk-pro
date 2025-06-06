#!/usr/bin/env python3
"""
Unified Test Runner for Tree ML Pipeline
========================================

This script provides a centralized interface for running all tests for the
Tree ML project, including ML model tests and S2 coordinate tests. It ensures
consistent test paths, data formats, and reporting.

Features:
- Runs both ML and S2 coordinate tests
- Generates consolidated test reports
- Uses consistent test images and Dallas coordinates
- Provides comprehensive logging and visualization

Usage examples:
  # Run all tests
  python unified_test_runner.py

  # Run specific test suite
  python unified_test_runner.py --test ml
  python unified_test_runner.py --test s2

  # Run specific ML test
  python unified_test_runner.py --test ml --ml-test pipeline
  python unified_test_runner.py --test ml --ml-test deepforest

  # Force CPU mode
  python unified_test_runner.py --cpu
"""

import os
import sys
import json
import time
import logging
import argparse
import subprocess
import asyncio
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('unified_test_runner')

# Default directories
DATA_DIR = "/ttt/data/tests"
TEST_IMAGES_DIR = os.path.join(DATA_DIR, "test_images")
TEST_RESULTS_DIR = DATA_DIR  # All test results now go to the main tests directory

# Ensure results directory exists
os.makedirs(TEST_RESULTS_DIR, exist_ok=True)
os.makedirs(TEST_IMAGES_DIR, exist_ok=True)

# Dallas, TX coordinates for consistent testing
DALLAS_LAT = 32.7767
DALLAS_LNG = -96.7970

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Unified Test Runner for Tree ML Pipeline')
    parser.add_argument('--test', choices=['all', 'ml', 's2'], default='all',
                        help='Test suite to run (all, ml, s2)')
    parser.add_argument('--ml-test', choices=['all', 'pipeline', 'deepforest', 'sam', 'batch'],
                        default='all', help='ML test to run (all, pipeline, deepforest, sam, batch)')
    parser.add_argument('--cpu', action='store_true', help='Force CPU mode')
    parser.add_argument('--image', type=str, help='Custom test image path')
    parser.add_argument('--output', type=str, help='Custom output directory')
    return parser.parse_args()

def run_ml_tests(args):
    """Run ML tests using ml_tester.py"""
    logger.info("=" * 60)
    logger.info("Running ML Tests")
    logger.info("=" * 60)
    
    # Build command using poetry
    cmd = ["poetry", "run", "python", "/ttt/tests/ml_tester.py"]
    
    # Add test type
    if args.ml_test != 'all':
        cmd.extend(["--test", args.ml_test])
        
    # Add CPU flag if needed
    if args.cpu:
        cmd.append("--cpu")
        
    # Add custom image if provided
    if args.image:
        cmd.extend(["--image", args.image])
        
    # Add custom output directory if provided
    if args.output:
        cmd.extend(["--output", args.output])
    
    # Run command
    logger.info(f"Running command: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        logger.info("ML tests completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"ML tests failed with exit code {e.returncode}")
        return False

async def run_s2_tests(args):
    """Run S2 coordinate tests using s2_coordinate_tester.py"""
    logger.info("=" * 60)
    logger.info("Running S2 Coordinate Tests")
    logger.info("=" * 60)
    
    try:
        # Import s2_coordinate_tester directly
        sys.path.append('/ttt/tests')
        from s2_coordinate_tester import main as s2_main
        
        # Run the main function
        result = await s2_main()
        
        if result == 0:
            logger.info("S2 coordinate tests completed successfully")
            return True
        else:
            logger.error(f"S2 coordinate tests failed with exit code {result}")
            return False
    except Exception as e:
        logger.error(f"Error running S2 coordinate tests: {e}")
        return False

def generate_unified_report(ml_success, s2_success):
    """Generate a unified test report"""
    report = {
        "timestamp": datetime.now().isoformat(),
        "ml_tests": {
            "success": ml_success,
            "results_dir": TEST_RESULTS_DIR
        },
        "s2_tests": {
            "success": s2_success,
            "results_dir": TEST_RESULTS_DIR
        },
        "overall_success": ml_success and s2_success
    }
    
    # Load detailed reports if available
    try:
        ml_report_path = os.path.join(TEST_RESULTS_DIR, "batch_test_report.json")
        if os.path.exists(ml_report_path):
            with open(ml_report_path, 'r') as f:
                report["ml_tests"]["details"] = json.load(f)
    except Exception as e:
        logger.warning(f"Error loading ML test report: {e}")
    
    try:
        s2_report_path = os.path.join(TEST_RESULTS_DIR, "s2_testing_report.json")
        if os.path.exists(s2_report_path):
            with open(s2_report_path, 'r') as f:
                report["s2_tests"]["details"] = json.load(f)
    except Exception as e:
        logger.warning(f"Error loading S2 test report: {e}")
    
    # Save unified report
    report_path = os.path.join(TEST_RESULTS_DIR, "test_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    logger.info(f"Test report saved to {report_path}")
    
    return report

async def main():
    """Main function to run unified tests"""
    args = parse_args()
    
    # Track test results
    ml_success = True
    s2_success = True
    
    start_time = time.time()
    
    # Run ML tests if requested
    if args.test in ['all', 'ml']:
        ml_success = run_ml_tests(args)
    
    # Run S2 tests if requested
    if args.test in ['all', 's2']:
        s2_success = await run_s2_tests(args)
    
    # Generate unified report
    report = generate_unified_report(ml_success, s2_success)
    
    # Print summary
    total_time = time.time() - start_time
    logger.info("=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)
    logger.info(f"Total execution time: {total_time:.2f} seconds")
    logger.info(f"ML tests: {'PASSED' if ml_success else 'FAILED'}")
    logger.info(f"S2 tests: {'PASSED' if s2_success else 'FAILED'}")
    logger.info(f"Overall: {'PASSED' if report['overall_success'] else 'FAILED'}")
    
    # Return exit code
    return 0 if report['overall_success'] else 1

if __name__ == "__main__":
    asyncio.run(main())