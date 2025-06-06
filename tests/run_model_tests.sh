#!/bin/bash
# Run model server tests using Poetry

set -e

# Change to the project root directory
cd "$(dirname "$0")/.."

# Run the tests with Poetry
echo "Running model server tests..."
poetry run pytest tests/model_server -v

# Run specific tests if needed
if [ "$1" == "unit" ]; then
    echo "Running unit tests only..."
    poetry run pytest tests/model_server/test_model_server.py -v
elif [ "$1" == "integration" ]; then
    echo "Running integration tests only..."
    poetry run pytest tests/model_server/test_integration.py -v -m integration
fi

echo "Tests completed!"