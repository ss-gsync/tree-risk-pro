# Machine Learning Testing Framework

## Overview

This document describes the testing framework used to validate the ML components of the Tree ML system. The testing framework ensures reliable operation of tree detection and segmentation across various environments and conditions.

## Test Suite Structure

The test suite is organized in the `/tests/ml` directory with three main components:

```
/tests/ml/
├── cuda_test.py           # Tests CUDA and GPU compatibility
├── run_tests.py           # Main test runner script
└── test_deepforest_sam.py # Tests DeepForest detection and SAM segmentation
└── test_pipeline.py       # Tests the complete ML pipeline
```

### Test Components

1. **CUDA Test**: Verifies GPU compatibility and proper configuration
2. **DeepForest/SAM Test**: Tests tree detection and segmentation in isolation
3. **Pipeline Test**: Tests the complete end-to-end ML pipeline

## Running Tests

```bash
# Run all tests
poetry run python tests/ml/run_tests.py

# Run specific test
poetry run python tests/ml/run_tests.py cuda          # Test CUDA functionality
poetry run python tests/ml/run_tests.py deepforest_sam # Test DeepForest and SAM models
poetry run python tests/ml/run_tests.py pipeline      # Test the full ML pipeline

# Force CPU mode for testing
poetry run python tests/ml/run_tests.py all --cpu
```

## Test Results

Test results are stored in `/data/tests/ml_test_results/` with the following structure:

```
/data/tests/ml_test_results/
├── detection/                # Tree detection results
│   ├── deepforest_boxes.json # Detection bounding boxes
│   └── deepforest_prediction.png # Visualization of detections
├── pipeline/                 # Pipeline test results
│   ├── detection_boxes.json  # Detection results
│   ├── detection.png         # Detection visualization
│   ├── mask_*.png            # Individual tree masks
│   ├── pipeline_report.json  # Performance metrics
│   ├── segmentation_masks.json # Segmentation metadata
│   └── segmentation_visualization.png # Combined visualization
├── reports/                  # Test reports
│   └── ml_test_report_*.json # Timestamped test reports
└── segmentation/             # Segmentation results
    ├── combined_visualization.png # Visualization with all masks
    ├── sam_mask_*.png        # Individual tree masks
    └── sam_masks.json        # Segmentation metadata
```

## Test Metrics

Each test collects and reports metrics including:

- **Detection Count**: Number of trees detected
- **Detection Time**: Time to perform detection
- **Segmentation Time**: Time to segment all detected trees
- **Segmentation Scores**: Confidence scores for each segmentation mask
- **Total Pipeline Time**: End-to-end processing time
- **CUDA Status**: Whether CUDA was available and used

## Test Images

Tests use a standard satellite image of Central Park (40.7791, -73.96375) to ensure consistent results. The test can also generate synthetic test images when needed.

## GPU Acceleration

The test suite verifies CUDA compatibility and GPU acceleration. It:

1. Checks if CUDA is available
2. Tests basic tensor operations on GPU
3. Verifies models can be moved to GPU
4. Measures performance with and without GPU
5. Ensures CPU fallback works correctly

## Integration with CI/CD

The test suite is designed to be integrated with CI/CD pipelines:

1. **Automated Testing**: Can be run automatically on code changes
2. **Performance Monitoring**: Tracks metrics over time
3. **Environment Verification**: Tests in different deployment environments

## Testing In-Memory ML Service

The in-memory ML service can be tested through the following methods:

1. **API Endpoint Test**:
   ```bash
   # Start the backend server
   cd /ttt/tree_ml/dashboard/backend
   python app.py
   
   # In another terminal, check the ML model status
   curl http://localhost:5000/api/ml/status
   ```

2. **Dashboard Integration Test**:
   - Open the dashboard in a web browser
   - Go to the Tree Detection panel
   - Enable the "Use tree segmentation" checkbox
   - Run detection
   - Verify the combined visualization displays in the results

3. **Performance Comparison**:
   - Run detection with and without the in-memory service
   - Compare inference times in the logs
   - Verify that subsequent detection requests are faster with in-memory service

The in-memory service should show significant performance improvements, especially for repeated detection requests, as models are only loaded once during initialization.

## Future Test Enhancements

Planned enhancements to the test suite include:

1. **Multi-environment Testing**: Expand tests to different hardware configurations
2. **Benchmark Mode**: Detailed performance benchmarking mode
3. **Dataset Expansion**: Tests with a wider variety of satellite imagery
4. **Accuracy Metrics**: Formal precision/recall evaluation with ground truth data
5. **In-Memory Service Benchmarks**: Compare performance metrics between standard and in-memory processing