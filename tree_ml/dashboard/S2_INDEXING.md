# S2 Geospatial Indexing Integration

This document describes the integration of S2 geospatial indexing into the Tree Risk Pro system.

## Overview

S2 is Google's hierarchical spatial indexing system that divides the earth into cells at different levels of granularity. This integration enhances the Tree Risk Pro platform with efficient spatial queries, hierarchical grouping, and optimized visualization of tree data.

## Features

- **Multi-level indexing**: Trees are indexed at multiple S2 cell levels (city, neighborhood, block, property)
- **Spatial queries**: Find trees in a specific geographic area 
- **Neighbor lookups**: Find trees near a specified location
- **Statistical aggregation**: Calculate statistics for trees grouped by S2 cell
- **Visualization grouping**: Group trees for efficient map visualization at different zoom levels

## How It Works

The system integrates S2 indexing at multiple points in the Tree Risk Pro pipeline:

1. **During Detection**: When trees are detected, they're assigned S2 cell IDs at multiple levels
2. **In Zarr Storage**: Tree data is stored with S2 indexing to enable efficient spatial queries
3. **For Visualization**: Trees are grouped by S2 cell for optimized map rendering
4. **For Analysis**: Statistical analysis can be performed on S2 cell groups

## Cell Levels

The system uses four standard S2 cell levels:

- **Level 10 (City)**: ~0.3km² - For city-wide view (low zoom)
- **Level 13 (Neighborhood)**: ~0.02km² - For neighborhood view (medium zoom)
- **Level 15 (Block)**: ~1400m² - For block view (medium-high zoom)
- **Level 18 (Property)**: ~40m² - For property level (high zoom)

## Integration with ML Pipeline

The S2 indexing system integrates with the real ML pipeline without fallbacks:

1. **YOLO Detection**: Trees are detected using real YOLO models:
   ```python
   # In object_recognition.py, YOLO models are loaded from these paths
   yolo_model_paths = [
       os.path.join(self.config.model_path, "yolo11s.pt"),
       os.path.join(self.config.model_path, "yolo11l.pt"),
       os.path.join("/ttt/tree_risk_pro/dashboard/backend", "yolov8m.pt")
   ]
   ```

2. **SAM Segmentation**: Actual SAM models generate segmentation masks:
   ```python
   # SAM models are loaded from these paths
   sam_checkpoint_paths = [
       os.path.join(model_dir, "sam2.1_hiera_small.pt"),
       os.path.join(model_dir, "sam2.1_hiera_base_plus.pt"),
       os.path.join(model_dir, "sam_vit_h_4b8939.pth")
   ]
   
   # Generate masks with SAM
   masks, scores, _ = predictor.predict(
       point_coords=input_point,
       point_labels=input_label,
       multimask_output=True
   )
   ```

3. **S2 Indexing**: Detected trees are assigned S2 cell IDs at multiple levels:
   ```python
   # Get S2 cell IDs for a tree
   s2_cells = s2_manager.get_cell_ids_for_tree(lat, lng)
   tree['s2_cells'] = s2_cells  # Store in tree data
   ```

4. **Zarr Storage**: Tree data with S2 indexing is stored in the Zarr format:
   ```python
   # Store results with S2 indexing
   store_path = await self.storage_manager.store_data(zarr_data, job_id)
   ```

5. **Gemini Analytics**: Analytics layer can analyze trees grouped by S2 cell

## API

The system provides several API methods for working with S2 cells:

- `get_cell_id(lat, lng, level)`: Get a cell ID for a specific location and level
- `get_cell_ids_for_tree(lat, lng)`: Get cell IDs at all levels for a tree location
- `get_cells_for_bounds(bounds, level)`: Get all cells covering a geographic area
- `get_neighbors(lat, lng, level, k)`: Get neighboring cells around a location
- `query_trees_by_s2_cell(cell_id, level)`: Find trees in a specific S2 cell
- `get_neighboring_trees(lat, lng, level, k)`: Find trees in neighboring cells
- `group_trees_by_s2_cell(trees, level)`: Group trees by S2 cell
- `calculate_s2_statistics(grouped_trees)`: Calculate statistics for cell groups

## Example Usage

### Basic S2 Operations

```python
# Initialize the S2 index manager
from services.detection_service import S2IndexManager
s2_manager = S2IndexManager()

# Get cell ID for a location at block level (New York City coordinates)
cell_id = s2_manager.get_cell_id(40.7128, -74.0060, 'block')
print(f"Block-level cell ID: {cell_id}")  # Example: 9749613089417207808

# Get all cell IDs for a tree at different levels
cell_ids = s2_manager.get_cell_ids_for_tree(40.7128, -74.0060)
print(cell_ids)
# Example output:
# {
#   'city': '9749613089149648896',
#   'neighborhood': '9749613089405952000',
#   'block': '9749613089417207808',
#   'property': '9749613089417730048'
# }

# Find neighboring cells at block level
neighbors = s2_manager.get_neighbors(40.7128, -74.0060, 'block', 8)
print(f"Found {len(neighbors)} neighboring cells")

# Get cells covering a geographic area (Central Park bounding box)
bounds = [
    [-73.981, 40.764],  # SW corner
    [-73.949, 40.800]   # NE corner
]
cells = s2_manager.get_cells_for_bounds(bounds, 'neighborhood')
print(f"Found {len(cells)} cells covering the area")
```

### Integration with ML Pipeline

```python
from services.detection_service import DetectionService

# Initialize the detection service
detection_service = DetectionService()

# Run tree detection with S2 indexing on a satellite image
result = await detection_service.detect_trees_from_satellite(
    image_path="/ttt/data/temp/ml_results_test/satellite_40.7791_-73.96375_16_1746715815.jpg",
    bounds=[[-73.970, 40.775], [-73.960, 40.785]],
    job_id="test_detection_1"
)

# Each tree now has S2 cell IDs
tree = result['trees'][0]
print(f"Tree at {tree['location']} has S2 cells: {tree['s2_cells']}")

# Query trees in a specific cell
trees = await detection_service.query_trees_by_s2_cell(cell_id, 'block')
print(f"Found {len(trees)} trees in cell {cell_id}")

# Find trees in neighboring cells
neighbor_trees = await detection_service.get_neighboring_trees(40.7128, -74.0060, 'block', 8)
print(f"Found {len(neighbor_trees)} trees in neighboring cells")

# Group trees by S2 cell for visualization and analysis
grouped_trees = await detection_service.group_trees_by_s2_cell(result['trees'], 'block')
print(f"Grouped {len(result['trees'])} trees into {len(grouped_trees)} cell groups")

# Calculate statistics for each cell group
cell_stats = detection_service.calculate_s2_statistics(grouped_trees)
for cell_id, stats in cell_stats.items():
    print(f"Cell {cell_id}: {stats['tree_count']} trees, risk level: {stats['dominant_risk']}")
```

### Full ML Pipeline with S2 Indexing

For a complete example of the ML pipeline with S2 indexing, see the `test_ml_pipeline.py` script:

```python
# Initialize the ML pipeline
pipeline = MLPipelineRunner()

# Process an image through the entire pipeline
results = await pipeline.process_image(
    image_path="/ttt/data/temp/ml_results_test/satellite_40.7791_-73.96375_16_1746715815.jpg",
    bounds=[[-73.970, 40.775], [-73.960, 40.785]],
    output_dir="/ttt/data/temp/ml_pipeline_test"
)

# Results include:
# - Tree detections with S2 cell IDs
# - Segmentation masks generated by SAM
# - Trees grouped by S2 cell
# - Statistics for each S2 cell group
```

## Testing and Demonstration

### Test Scripts

Several test scripts are available to verify the S2 indexing implementation:

1. **Full ML Pipeline Test**:
   ```bash
   cd /ttt
   python tree_risk_pro/dashboard/test_ml_pipeline.py
   ```
   This script runs the complete ML pipeline with YOLO, SAM, and S2 indexing on a sample satellite image. It:
   - Detects trees using YOLO
   - Generates segmentation masks using SAM
   - Adds S2 cell IDs to trees at multiple levels
   - Groups trees by S2 cell
   - Calculates statistics for each cell group
   - Saves results to a JSON file and visualization images

2. **S2 Indexing Test**:
   ```bash
   cd /ttt
   python tree_risk_pro/dashboard/test_s2_index.py
   ```
   This script focuses specifically on testing the S2 indexing functionality, verifying:
   - S2 cell ID generation at different levels
   - Finding neighboring cells
   - Getting cells covering geographic bounds
   - Converting between S2 cells and geographic coordinates

### Interactive Notebook

A Jupyter notebook demonstrating the S2 indexing functionality interactively is available at:
`/ttt/tree_risk_pro/dashboard/s2_indexing_demo.ipynb`

To run the notebook:
```bash
cd /ttt
jupyter notebook tree_risk_pro/dashboard/s2_indexing_demo.ipynb
```

This notebook includes:
- Visualizations of S2 cells at different levels
- Examples of S2 cell operations (neighbors, bounds)
- Interactive map visualizations
- Code examples for S2 integration with tree data

## Benefits 

This integration provides several key benefits:

1. **Improved Performance**: Efficient spatial queries and reduced data transfer
2. **Better Visualization**: Optimized grouping at different zoom levels
3. **Enhanced Analytics**: Geographic clustering for more meaningful analysis
4. **Scalability**: Hierarchical structure adapts to dataset size and zoom level
5. **Integration with ML**: Works seamlessly with YOLO/DeepForest and SAM pipelines

## Future Enhancements

Potential future enhancements to the S2 indexing system:

1. Addition of temporal indexing for time-series analysis
2. Implementing real-time collaboration using S2 cells as synchronization units
3. Adding polygon-based queries for irregular area selection
4. Optimizing visualization with LOD (Level of Detail) based on S2 cell level