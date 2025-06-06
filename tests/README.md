# Tree ML Test Suite

This directory contains unified test scripts for the Tree ML project, focusing on ML model evaluation and S2 coordinate integration testing.

## Test Structure

The test suite is organized as follows:

```
/ttt/tests/           # Main test scripts
  ├── unified_test_runner.py   # Unified test runner for all tests
  ├── ml_tester.py             # ML model testing (DeepForest/SAM)
  ├── s2_coordinate_tester.py  # S2 coordinate integration testing
  └── archived_tests/          # Previous test scripts (reference only)

/ttt/data/tests/      # Test data and results
  ├── ml_test_images/         # ML test images
  ├── ml_test_results/        # ML test results
  ├── s2_test_results/        # S2 coordinate test results
  └── unified_test_results/   # Combined test results
```

## Running Tests

### Unified Test Runner

The unified test runner can run all tests with consistent settings:

```bash
# Run all tests
poetry run python unified_test_runner.py

# Run only ML tests
poetry run python unified_test_runner.py --test ml

# Run only S2 tests
poetry run python unified_test_runner.py --test s2

# Run ML pipeline test
poetry run python unified_test_runner.py --test ml --ml-test pipeline

# Force CPU mode
poetry run python unified_test_runner.py --cpu
```

### ML Tests

The ML tests can be run directly:

```bash
# Run pipeline test
poetry run python ml_tester.py --test pipeline

# Run DeepForest test
poetry run python ml_tester.py --test deepforest

# Run SAM test
poetry run python ml_tester.py --test sam

# Run batch test on all images
poetry run python ml_tester.py --test batch

# Use Dallas coordinates (for S2 compatibility)
poetry run python ml_tester.py --test pipeline --dallas
```

### S2 Coordinate Tests

The S2 coordinate tests verify proper geo-coordinate projection:

```bash
# Run S2 coordinate tests
poetry run python s2_coordinate_tester.py
```

## Test Images

Test images are stored in `/ttt/data/tests/ml_test_images/`. The default test images are satellite images with trees and varying resolutions.

For S2 coordinate compatibility, tests use Dallas, TX coordinates (32.7767, -96.7970) as a reference point.

## Compatibility with Map Overlay

The tests are designed to validate the map overlay functionality, ensuring proper geo-coordinate projection using S2 cells for precise mapping.

Key features:
- Tests tree detection quality
- Verifies S2 cell-based coordinate mapping
- Ensures no whole-image detections are returned
- Provides detailed test reports in JSON format

## Adding New Tests

To add a new test:
1. Add your test function to the appropriate test file
2. Update the argument parser to include your test option
3. Add your test to the main function switch

## Test Reports

All tests generate detailed reports in JSON format:
- ML tests: `/ttt/data/tests/ml_test_results/ml_test_report.json`
- S2 tests: `/ttt/data/tests/s2_test_results/s2_testing_report.json`
- Unified: `/ttt/data/tests/unified_test_results/unified_test_report.json`