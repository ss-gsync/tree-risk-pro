# ML Overlay Optimization Tests

This directory contains tests for the optimized ML overlay and detection scripts.

## Test Files

- `test_optimized_overlay.html`: Interactive test page for comparing original and optimized ML overlay implementations
- `test_overlay_performance.js`: Node.js script for measuring performance metrics of optimized ML overlay

## Running the Tests

### Interactive Test Page

1. Open `test_optimized_overlay.html` in a web browser
2. Click "Load Test Detection" to load the test detection data
3. Use the controls to adjust visualization options
4. Click "Compare Performance" to compare original and optimized implementations

### Performance Test Script

Run the performance test script using Node.js:

```bash
node test_overlay_performance.js
```

## Test Data

Test data is stored in `/ttt/data/tests/test_1748726908/` and includes:

- Satellite imagery
- ML detection results with tree data
- Segmentation masks
- S2 cell mapping for coordinate transformation

## Key Optimizations

The optimized implementation includes the following improvements:

1. **Memory-efficient segmentation data handling**
   - Only essential segmentation data is stored
   - Lazy loading of segmentation masks using IntersectionObserver

2. **Improved rendering performance**
   - Batch DOM operations using document fragments
   - Virtual rendering for objects outside the viewport
   - Canvas-based rendering for segmentation masks

3. **Precise coordinate mapping**
   - S2 cell-based coordinate calculation
   - Multi-level fallback mechanisms for coordinate transformation
   - Proper handling of pixel-to-geo transformation

4. **Resource optimization**
   - Reduced memory footprint
   - More efficient data structures
   - Better error handling and performance metrics

## Comparison Metrics

The performance comparison includes the following metrics:

- Load time
- Render time
- Number of objects rendered
- Number of segmentations rendered
- Memory usage

## Expected Improvements

- **Memory usage**: 50-70% reduction in segmentation data size
- **Rendering performance**: 30-50% faster rendering
- **Coordinate precision**: More accurate placement of markers and overlays