#!/usr/bin/env python3
"""
Unified Test Runner for Tree ML Platform v0.2.3
===============================================

This script provides a centralized interface for running all tests for the
Tree ML project, including:
- Model server tests
- T4 GPU integration tests
- ML pipeline tests
- S2 coordinate tests
- Dashboard tests

Features:
- Runs all test suites or specific ones as requested
- Provides comprehensive logging
- Consistent test paths and data
- Supports testing against running T4 model server

Usage examples:
  # Run all tests
  python unified_test_runner.py

  # Run specific test suites
  python unified_test_runner.py --model-tests
  python unified_test_runner.py --t4-tests --server-url http://localhost:8000
  python unified_test_runner.py --ml-tests
  python unified_test_runner.py --s2-tests
  python unified_test_runner.py --dashboard-tests

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

# Get the project root directory
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Default directories
DATA_DIR = os.path.join(ROOT_DIR, "data/tests")
TEST_IMAGES_DIR = os.path.join(DATA_DIR, "test_images")
TEST_RESULTS_DIR = DATA_DIR

# Ensure results directory exists
os.makedirs(TEST_RESULTS_DIR, exist_ok=True)
os.makedirs(TEST_IMAGES_DIR, exist_ok=True)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Unified Test Runner for Tree ML Platform')
    
    # Test selection arguments
    parser.add_argument('--all', action='store_true', help='Run all tests')
    parser.add_argument('--model-tests', action='store_true', help='Run model server tests')
    parser.add_argument('--t4-tests', action='store_true', help='Run T4 integration tests')
    parser.add_argument('--ml-tests', action='store_true', help='Run ML pipeline tests')
    parser.add_argument('--s2-tests', action='store_true', help='Run S2 coordinate tests')
    parser.add_argument('--dashboard-tests', action='store_true', help='Run dashboard tests')
    
    # Configuration arguments
    parser.add_argument('--cpu', action='store_true', help='Force CPU mode for ML tests')
    parser.add_argument('--server-url', default="http://localhost:8000", 
                        help='URL of the model server for T4 tests')
    parser.add_argument('--image', type=str, help='Custom test image path')
    parser.add_argument('--output', type=str, help='Custom output directory')
    
    return parser.parse_args()

def run_command(command, description=None):
    """Run a shell command and display its output."""
    if description:
        logger.info("=" * 60)
        logger.info(description)
        logger.info("=" * 60)
    
    logger.info(f"Running command: {command}")
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        for line in result.stdout.splitlines():
            logger.info(line)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}")
        for line in e.stdout.splitlines():
            logger.error(line)
        return False

def run_model_tests():
    """Run model server tests."""
    return run_command(
        f"cd {ROOT_DIR} && ./tests/run_model_tests.sh",
        "Running Model Server Tests"
    )

def run_t4_tests(server_url):
    """Run T4 integration tests."""
    return run_command(
        f"cd {ROOT_DIR} && ./tests/run_t4_tests.sh {server_url}",
        f"Running T4 Integration Tests against {server_url}"
    )

def run_ml_pipeline_tests(args):
    """Run ML pipeline tests."""
    # Build command
    cmd = f"cd {ROOT_DIR}/tree_ml/pipeline/tests && poetry run python ml_test.py"
    
    # Add CPU flag if needed
    if args.cpu:
        cmd += " --cpu"
    
    # Add image if provided
    if args.image:
        cmd += f" --image {args.image}"
    
    return run_command(cmd, "Running ML Pipeline Tests")

async def run_s2_tests(args):
    """Run S2 coordinate tests."""
    logger.info("=" * 60)
    logger.info("Running S2 Coordinate Tests")
    logger.info("=" * 60)
    
    try:
        # Import s2_coordinate_tester directly
        sys.path.append(str(ROOT_DIR / "tests"))
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

def run_dashboard_tests():
    """Run dashboard integration tests."""
    return run_command(
        f"cd {ROOT_DIR}/tree_ml/dashboard && poetry run pytest",
        "Running Dashboard Tests"
    )

def generate_unified_report(results):
    """Generate a unified test report."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "results": results,
        "overall_success": all(results.values())
    }
    
    # Save unified report
    report_path = os.path.join(TEST_RESULTS_DIR, "unified_test_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    logger.info(f"Unified test report saved to {report_path}")
    
    return report

async def main():
    """Main function to run unified tests."""
    args = parse_args()
    
    # If no specific tests are selected, run all tests
    if not (args.all or args.model_tests or args.t4_tests or 
            args.ml_tests or args.s2_tests or args.dashboard_tests):
        args.all = True
    
    # Track test results
    results = {}
    start_time = time.time()
    
    # Run tests based on arguments
    if args.all or args.model_tests:
        results["model_tests"] = run_model_tests()
    
    if args.all or args.t4_tests:
        results["t4_tests"] = run_t4_tests(args.server_url)
    
    if args.all or args.ml_tests:
        results["ml_tests"] = run_ml_pipeline_tests(args)
    
    if args.all or args.s2_tests:
        results["s2_tests"] = await run_s2_tests(args)
    
    if args.all or args.dashboard_tests:
        results["dashboard_tests"] = run_dashboard_tests()
    
    # Generate unified report
    report = generate_unified_report(results)
    
    # Print summary
    total_time = time.time() - start_time
    logger.info("=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)
    logger.info(f"Total execution time: {total_time:.2f} seconds")
    
    for test_name, success in results.items():
        logger.info(f"{test_name}: {'PASSED' if success else 'FAILED'}")
    
    logger.info(f"Overall: {'PASSED' if report['overall_success'] else 'FAILED'}")
    
    # Return exit code
    return 0 if report['overall_success'] else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)