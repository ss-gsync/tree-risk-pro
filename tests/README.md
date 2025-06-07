# Tree ML Testing Suite

This directory contains test suites for the Tree ML platform v0.2.3. These tests ensure that all components of the platform work correctly and reliably.

## Test Structure

The test suite is organized into several components:

```
/ttt/tests/                   # Main test directory
  ├── unified_test_runner.py  # Unified test runner for all tests
  ├── run_model_tests.sh      # Script to run model server tests
  ├── run_t4_tests.sh         # Script to test T4 GPU integration
  ├── model_server/           # Model server test files
  │   ├── test_basic.py       # Basic dependency tests
  │   ├── test_model_server.py # Core server functionality tests
  │   ├── test_detect.py      # Detection API tests
  │   ├── test_integration.py # Integration tests
  │   └── test_mock.py        # Mocked tests (no model required)
  ├── ml_tester.py            # ML model testing (DeepForest/SAM)
  ├── s2_coordinate_tester.py # S2 coordinate integration testing
  └── archived_tests/         # Previous test scripts (reference only)

/ttt/data/tests/              # Test data and results
  ├── test_images/            # Test images
  ├── ml_test_results/        # ML test results
  └── s2_test_results/        # S2 coordinate test results
```

## Test Suites

1. **Model Server Tests** (`/tests/model_server/`)
   - Tests for the T4 GPU model server
   - FastAPI endpoint tests
   - Client integration tests
   - Mock tests that don't require actual models

2. **ML Pipeline Tests** (`/tree_ml/pipeline/tests/`)
   - Tests for the machine learning pipeline
   - Object detection and segmentation tests
   - CUDA compatibility tests

3. **S2 Coordinate Tests** (`/tests/s2_coordinate_tester.py`)
   - Tests for S2 coordinate system conversions
   - Validation of coordinate mapping

4. **Dashboard Tests** (`/tree_ml/dashboard/tests/`)
   - Tests for the web dashboard
   - API integration tests
   - User interface components

## Running Tests

### Running All Tests

To run all tests, use the unified test runner:

```bash
cd /ttt
./tests/unified_test_runner.py
```

### Running Specific Test Suites

```bash
# Model server tests
./tests/run_model_tests.sh

# Run unit tests only
./tests/run_model_tests.sh unit

# Run integration tests only
./tests/run_model_tests.sh integration

# T4 integration tests (requires a running model server)
./tests/run_t4_tests.sh http://localhost:8000

# ML pipeline tests
./tests/unified_test_runner.py --ml-tests

# S2 coordinate tests
./tests/unified_test_runner.py --s2-tests

# Dashboard tests
./tests/unified_test_runner.py --dashboard-tests
```

### Running Tests with Specific Options

```bash
# Run tests in CPU mode (for environments without CUDA)
./tests/unified_test_runner.py --cpu

# Run tests with a custom test image
./tests/unified_test_runner.py --image /path/to/test/image.jpg

# Run tests with custom output directory
./tests/unified_test_runner.py --output /path/to/output/dir
```

## Test Data

The tests use sample data located in:

- `/ttt/data/tests/test_images/`: Sample satellite images
- `/ttt/data/ml/`: Sample ML detection results
- `/ttt/data/tests/`: Test reports and output

For S2 coordinate compatibility, tests use Dallas, TX coordinates (32.7767, -96.7970) as a reference point.

## T4 GPU Integration

The T4 GPU integration tests verify that:

1. The model server can be reached at the specified URL
2. The model server can load ML models correctly
3. The model server can process detection requests
4. The client can retrieve detection results

These tests require a running model server. To start the model server:

```bash
cd /ttt/tree_ml/pipeline
python model_server.py
```

## Test Reports

Test results are saved to the following locations:

- `/ttt/data/tests/unified_test_report.json`: Consolidated test report
- `/ttt/data/tests/ml_test_report.json`: ML pipeline test report
- `/ttt/data/tests/s2_testing_report.json`: S2 coordinate test report

## Deployment Testing

Before deploying to production, run the full test suite with:

```bash
cd /ttt
./tests/unified_test_runner.py --all

# If you have a T4 model server running
./tests/unified_test_runner.py --t4-tests --server-url http://your-server:8000
```

## Adding New Tests

To add a new test:
1. Add your test function to the appropriate test file
2. Update the argument parser to include your test option if needed
3. Add your test to the main function switch if needed
4. Update the unified test runner if necessary