# Model Server Test Suite

This directory contains tests for the Tree ML model server that integrates with the T4 GPU instance.

## Test Overview

The test suite consists of several components:

- **Basic Tests**: Verify dependencies and imports
- **Model Server Tests**: Test the core model server functionality
- **External Model Service Tests**: Test the client integration with the model server
- **Integration Tests**: End-to-end tests of the complete detection pipeline
- **Mock Tests**: Tests using mocks to avoid requiring the actual model

## Running the Tests

### Option 1: Run All Tests

To run all tests with pytest:

```bash
# From the project root
./tests/run_model_tests.sh
```

### Option 2: Run Specific Tests

To run a specific set of tests:

```bash
# Unit tests only
./tests/run_model_tests.sh unit

# Integration tests only
./tests/run_model_tests.sh integration
```

### Option 3: Test with a Running Model Server

If you have a model server running (locally or on a T4 instance), you can run the full integration test suite:

```bash
# Test against a local model server
./tests/run_t4_tests.sh http://localhost:8000

# Test against a remote T4 model server
./tests/run_t4_tests.sh http://t4-instance-ip:8000
```

## Required Data

The tests expect the following:

1. Sample satellite image at `/ttt/data/tests/test_images/sample.jpg`
2. Sample detection results at `/ttt/data/ml/detection_1748997756/`

## Test Files

- `conftest.py`: Pytest configuration and fixtures
- `test_basic.py`: Basic dependency and import tests
- `test_model_server.py`: Core model server functionality tests
- `test_detect.py`: Detection API endpoint tests
- `test_external_model_service.py`: Client integration tests
- `test_integration.py`: End-to-end integration tests
- `test_mock.py`: Tests using mocks instead of actual model
- `test_imports.py`: Import tests
- `test_client.py`: Helper for testing FastAPI endpoints

## Preparing for Deployment

Before deploying the model server to production, make sure all tests pass using:

```bash
# Run all tests
./tests/run_model_tests.sh

# If you have a model server running
./tests/run_t4_tests.sh http://your-model-server:8000
```

## Troubleshooting

If tests fail:

1. **Import errors**: Check that all dependencies are installed (`poetry install`)
2. **Connection errors**: Ensure the model server is running and accessible
3. **Missing sample data**: Verify the sample data paths exist
4. **CUDA errors**: Ensure CUDA is properly installed on the system

## GPU Testing Notes

Some tests require CUDA and will be skipped if it's not available. To fully test GPU functionality, run on a system with CUDA installed.

## Adding New Tests

When adding new tests, follow these guidelines:

1. Use `pytest.mark.skipif` for tests that require specific conditions
2. Use fixtures from `conftest.py` for common test resources
3. Mock external dependencies when possible to keep tests fast and reliable
4. Add new test files to `tests/run_model_tests.sh`