#!/bin/bash

# Run ML overlay optimization tests

echo "Running ML Overlay Optimization Tests"
echo "===================================="

# Run performance test
echo "Running performance test..."
node test_overlay_performance.js

# Open HTML test page if browser is available
if command -v xdg-open >/dev/null 2>&1; then
    echo -e "\nOpening interactive test page..."
    xdg-open test_optimized_overlay.html
elif command -v open >/dev/null 2>&1; then
    echo -e "\nOpening interactive test page..."
    open test_optimized_overlay.html
else
    echo -e "\nTo run the interactive test, open test_optimized_overlay.html in your browser"
fi

echo -e "\nTests completed."