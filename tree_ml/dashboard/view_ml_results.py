#!/usr/bin/env python3
"""
Utility to view ML pipeline results with S2 indexing visualization.

This script displays results from the ML pipeline tests, including:
1. YOLO detection results
2. SAM segmentation masks (ML overlay)
3. S2 cell grouping visualization
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ml_results_viewer')

# Default results directory
DEFAULT_RESULTS_DIR = '/ttt/data/temp/ml_results_test'

def find_latest_results(base_dir=DEFAULT_RESULTS_DIR):
    """Find the most recent test results"""
    if not os.path.exists(base_dir):
        logger.error(f"Results directory {base_dir} does not exist")
        return None
    
    # Look for detection and segmentation directories
    detection_dir = os.path.join(base_dir, 'detection')
    segmentation_dir = os.path.join(base_dir, 'segmentation')
    
    if not os.path.exists(detection_dir) or not os.path.exists(segmentation_dir):
        logger.warning(f"Could not find detection or segmentation directories in {base_dir}")
    
    # Look for ML response directories
    detection_ml_dir = os.path.join(detection_dir, 'ml_response')
    segmentation_ml_dir = os.path.join(segmentation_dir, 'ml_response')
    
    results = {
        'base_dir': base_dir,
        'detection_dir': detection_dir if os.path.exists(detection_dir) else None,
        'segmentation_dir': segmentation_dir if os.path.exists(segmentation_dir) else None,
        'detection_ml_dir': detection_ml_dir if os.path.exists(detection_ml_dir) else None,
        'segmentation_ml_dir': segmentation_ml_dir if os.path.exists(segmentation_ml_dir) else None,
        'detection_files': [],
        'segmentation_files': [],
        'test_image': None
    }
    
    # Find satellite test image
    for file in os.listdir(base_dir):
        if file.startswith('satellite_') and file.endswith('.jpg'):
            results['test_image'] = os.path.join(base_dir, file)
            break
    
    # Get detection files
    if results['detection_ml_dir']:
        results['detection_files'] = os.listdir(results['detection_ml_dir'])
        
        # Look for key files
        for file in results['detection_files']:
            if file == 'trees.json':
                results['detection_trees'] = os.path.join(results['detection_ml_dir'], file)
            elif file == 's2_grouping.json':
                results['detection_s2_grouping'] = os.path.join(results['detection_ml_dir'], file)
            elif file == 's2_statistics.json':
                results['detection_s2_stats'] = os.path.join(results['detection_ml_dir'], file)
    
    # Get segmentation files
    if results['segmentation_ml_dir']:
        results['segmentation_files'] = os.listdir(results['segmentation_ml_dir'])
        
        # Look for key files
        for file in results['segmentation_files']:
            if file == 'combined_segmentation.png':
                results['segmentation_mask'] = os.path.join(results['segmentation_ml_dir'], file)
            elif file == 'segmentation_overlay.png':
                results['segmentation_overlay'] = os.path.join(results['segmentation_ml_dir'], file)
            elif file == 's2_segmentation_grouping.json':
                results['segmentation_s2_grouping'] = os.path.join(results['segmentation_ml_dir'], file)
            elif file == 'segmentation_metadata.json':
                results['segmentation_metadata'] = os.path.join(results['segmentation_ml_dir'], file)
    
    return results

def load_detection_results(results):
    """Load detection results from JSON files"""
    detection_data = {}
    
    # Load tree detection data
    if 'detection_trees' in results and os.path.exists(results['detection_trees']):
        try:
            with open(results['detection_trees'], 'r') as f:
                detection_data['trees'] = json.load(f)
                logger.info(f"Loaded tree detection data with {len(detection_data['trees'].get('frames', [{}])[0].get('trees', []))} trees")
        except Exception as e:
            logger.error(f"Error loading tree detection data: {e}")
    
    # Load S2 grouping data
    if 'detection_s2_grouping' in results and os.path.exists(results['detection_s2_grouping']):
        try:
            with open(results['detection_s2_grouping'], 'r') as f:
                detection_data['s2_grouping'] = json.load(f)
                logger.info(f"Loaded S2 grouping data with {len(detection_data['s2_grouping'])} cells")
        except Exception as e:
            logger.error(f"Error loading S2 grouping data: {e}")
    
    # Load S2 statistics data
    if 'detection_s2_stats' in results and os.path.exists(results['detection_s2_stats']):
        try:
            with open(results['detection_s2_stats'], 'r') as f:
                detection_data['s2_stats'] = json.load(f)
                logger.info(f"Loaded S2 statistics data with {len(detection_data['s2_stats'])} cells")
        except Exception as e:
            logger.error(f"Error loading S2 statistics data: {e}")
    
    return detection_data

def load_segmentation_results(results):
    """Load segmentation results"""
    segmentation_data = {}
    
    # Load segmentation mask
    if 'segmentation_mask' in results and os.path.exists(results['segmentation_mask']):
        try:
            segmentation_data['mask'] = Image.open(results['segmentation_mask'])
            logger.info(f"Loaded segmentation mask with size {segmentation_data['mask'].size}")
        except Exception as e:
            logger.error(f"Error loading segmentation mask: {e}")
    
    # Load segmentation overlay
    if 'segmentation_overlay' in results and os.path.exists(results['segmentation_overlay']):
        try:
            segmentation_data['overlay'] = Image.open(results['segmentation_overlay'])
            logger.info(f"Loaded segmentation overlay with size {segmentation_data['overlay'].size}")
        except Exception as e:
            logger.error(f"Error loading segmentation overlay: {e}")
    
    # Load segmentation S2 grouping data
    if 'segmentation_s2_grouping' in results and os.path.exists(results['segmentation_s2_grouping']):
        try:
            with open(results['segmentation_s2_grouping'], 'r') as f:
                segmentation_data['s2_grouping'] = json.load(f)
                logger.info(f"Loaded segmentation S2 grouping data with {len(segmentation_data['s2_grouping'])} cells")
        except Exception as e:
            logger.error(f"Error loading segmentation S2 grouping data: {e}")
    
    # Load segmentation metadata
    if 'segmentation_metadata' in results and os.path.exists(results['segmentation_metadata']):
        try:
            with open(results['segmentation_metadata'], 'r') as f:
                segmentation_data['metadata'] = json.load(f)
                logger.info(f"Loaded segmentation metadata")
        except Exception as e:
            logger.error(f"Error loading segmentation metadata: {e}")
    
    return segmentation_data

def visualize_detection_results(results, detection_data, output_dir):
    """Visualize YOLO detection results with bounding boxes"""
    if 'test_image' not in results or not results['test_image']:
        logger.error("No test image found for visualization")
        return
    
    if 'trees' not in detection_data or not detection_data['trees']:
        logger.error("No tree detection data found for visualization")
        return
    
    try:
        # Load the test image
        image = Image.open(results['test_image'])
        width, height = image.size
        
        # Create a copy for drawing
        detection_image = image.copy().convert('RGBA')
        draw = ImageDraw.Draw(detection_image)
        
        # Get tree data
        trees = detection_data['trees'].get('frames', [{}])[0].get('trees', [])
        logger.info(f"Visualizing {len(trees)} detected trees")
        
        # Draw boxes for each tree
        for i, tree in enumerate(trees):
            if 'bbox' in tree:
                x1, y1, x2, y2 = tree['bbox']
                
                # Convert normalized coordinates to pixels if needed
                if max(x1, y1, x2, y2) <= 1.0:
                    x1, y1, x2, y2 = x1 * width, y1 * height, x2 * width, y2 * height
                
                # Draw bounding box
                draw.rectangle([(x1, y1), (x2, y2)], outline=(255, 0, 0, 255), width=2)
                
                # Draw tree ID
                tree_id = tree.get('id', f"Tree {i+1}")
                confidence = tree.get('confidence', 0.0)
                draw.text((x1, y1-10), f"{tree_id.split('_')[-1]} ({confidence:.2f})", fill=(255, 0, 0, 255))
        
        # Save the visualization
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "detection_visualization.png")
        detection_image.save(output_path)
        logger.info(f"Saved detection visualization to {output_path}")
        
        return output_path
    
    except Exception as e:
        logger.error(f"Error visualizing detection results: {e}")
        return None

def visualize_s2_grouping(results, detection_data, output_dir):
    """Visualize S2 cell grouping"""
    if 'test_image' not in results or not results['test_image']:
        logger.error("No test image found for visualization")
        return
    
    if 's2_grouping' not in detection_data or not detection_data['s2_grouping']:
        logger.error("No S2 grouping data found for visualization")
        return
    
    if 's2_stats' not in detection_data or not detection_data['s2_stats']:
        logger.error("No S2 statistics data found for visualization")
        return
    
    try:
        # Load the test image
        image = Image.open(results['test_image'])
        width, height = image.size
        
        # Create a copy for drawing
        s2_image = image.copy().convert('RGBA')
        draw = ImageDraw.Draw(s2_image)
        
        # Get S2 grouping and statistics
        s2_grouping = detection_data['s2_grouping']
        s2_stats = detection_data['s2_stats']
        
        # Risk level colors
        risk_colors = {
            'low': (0, 255, 0, 128),      # Green
            'medium': (255, 165, 0, 128), # Orange
            'high': (255, 0, 0, 128),     # Red
            'unknown': (100, 100, 255, 128) # Blue
        }
        
        # Get trees data for coordinate lookup
        trees = {}
        if 'trees' in detection_data and detection_data['trees']:
            for tree in detection_data['trees'].get('frames', [{}])[0].get('trees', []):
                if 'id' in tree:
                    trees[tree['id']] = tree
        
        # Draw cells
        for cell_id, stats in s2_stats.items():
            # Get trees in this cell
            tree_ids = stats.get('trees', [])
            
            # Skip cells with no trees
            if not tree_ids:
                continue
            
            # Get center of cell
            if 'center' in stats:
                # Center is in geographic coordinates - need to convert to pixels
                center_lng, center_lat = stats['center']
                
                # Get bounds from tree metadata (assuming all trees use the same bounds)
                bounds = None
                if 'trees' in detection_data and detection_data['trees']:
                    metadata = detection_data['trees'].get('metadata', {})
                    bounds = metadata.get('bounds')
                
                # If bounds are available, convert to pixel coordinates
                if bounds:
                    sw_lng, sw_lat = bounds[0]
                    ne_lng, ne_lat = bounds[1]
                    
                    # Convert center to normalized coordinates (0-1)
                    x_norm = (center_lng - sw_lng) / (ne_lng - sw_lng)
                    y_norm = (center_lat - sw_lat) / (ne_lat - sw_lat)
                    
                    # Convert to pixel coordinates
                    center_x = int(x_norm * width)
                    center_y = int(y_norm * height)
                    
                    # Get risk level and color
                    risk_level = stats.get('dominant_risk', 'medium').lower()
                    color = risk_colors.get(risk_level, risk_colors['unknown'])
                    
                    # Determine cell radius based on tree count (larger cells for more trees)
                    tree_count = stats.get('tree_count', 1)
                    cell_radius = min(100, max(30, 20 * tree_count))
                    
                    # Draw cell
                    draw.ellipse(
                        [(center_x - cell_radius, center_y - cell_radius), 
                         (center_x + cell_radius, center_y + cell_radius)], 
                        fill=color, outline=(255, 255, 255, 200), width=2
                    )
                    
                    # Add cell ID and tree count
                    draw.text(
                        (center_x - 20, center_y - 10), 
                        f"Cell: {cell_id[-4:]}\nTrees: {tree_count}",
                        fill=(255, 255, 255, 255)
                    )
        
        # Save the visualization
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "s2_grouping_visualization.png")
        s2_image.save(output_path)
        logger.info(f"Saved S2 grouping visualization to {output_path}")
        
        return output_path
    
    except Exception as e:
        logger.error(f"Error visualizing S2 grouping: {e}")
        return None

def create_report(results, detection_data, segmentation_data, output_dir):
    """Create an HTML report with all visualization results"""
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate visualizations
        detection_vis_path = visualize_detection_results(results, detection_data, output_dir)
        s2_grouping_vis_path = visualize_s2_grouping(results, detection_data, output_dir)
        
        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>ML Pipeline with S2 Indexing Results</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                .image-container {{ margin: 20px 0; }}
                .image-container img {{ max-width: 100%; border: 1px solid #ddd; }}
                .stats {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                table, th, td {{ border: 1px solid #ddd; }}
                th, td {{ padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>ML Pipeline with S2 Indexing Results</h1>
            
            <h2>Test Information</h2>
            <div class="stats">
                <p><strong>Base Directory:</strong> {results['base_dir']}</p>
                <p><strong>Test Image:</strong> {os.path.basename(results['test_image']) if 'test_image' in results and results['test_image'] else 'None'}</p>
            </div>
            
            <h2>YOLO Tree Detection</h2>
        """
        
        # Add detection results
        if 'trees' in detection_data and detection_data['trees']:
            trees = detection_data['trees'].get('frames', [{}])[0].get('trees', [])
            html_content += f"""
            <div class="stats">
                <p><strong>Total Trees Detected:</strong> {len(trees)}</p>
            </div>
            
            <h3>Detection Visualization</h3>
            <div class="image-container">
                <img src="{os.path.basename(detection_vis_path) if detection_vis_path else 'No visualization available'}" alt="Tree Detection">
            </div>
            
            <h3>Sample Tree Detections</h3>
            <table>
                <tr>
                    <th>ID</th>
                    <th>Confidence</th>
                    <th>Location</th>
                    <th>S2 Cell (Property)</th>
                </tr>
            """
            
            # Add sample tree rows
            for i, tree in enumerate(trees[:10]):  # Show first 10 trees
                tree_id = tree.get('id', f"Tree {i+1}")
                confidence = tree.get('confidence', 0.0)
                location = tree.get('location', ['N/A', 'N/A'])
                s2_cell = tree.get('s2_cells', {}).get('property', 'N/A')
                
                html_content += f"""
                <tr>
                    <td>{tree_id}</td>
                    <td>{confidence:.4f}</td>
                    <td>{location[1]}, {location[0]}</td>
                    <td>{s2_cell}</td>
                </tr>
                """
            
            html_content += "</table>"
        else:
            html_content += "<p>No tree detection data available</p>"
        
        # Add S2 grouping section
        html_content += """
            <h2>S2 Geospatial Indexing</h2>
        """
        
        if 's2_grouping' in detection_data and detection_data['s2_grouping']:
            html_content += f"""
            <div class="stats">
                <p><strong>Total S2 Cells:</strong> {len(detection_data['s2_grouping'])}</p>
            </div>
            
            <h3>S2 Cell Grouping Visualization</h3>
            <div class="image-container">
                <img src="{os.path.basename(s2_grouping_vis_path) if s2_grouping_vis_path else 'No visualization available'}" alt="S2 Cell Grouping">
            </div>
            
            <h3>S2 Cell Statistics</h3>
            <table>
                <tr>
                    <th>Cell ID</th>
                    <th>Tree Count</th>
                    <th>Risk Level</th>
                    <th>Center</th>
                </tr>
            """
            
            # Add S2 cell statistics
            if 's2_stats' in detection_data and detection_data['s2_stats']:
                cells = list(detection_data['s2_stats'].items())
                for i, (cell_id, stats) in enumerate(cells[:10]):  # Show first 10 cells
                    tree_count = stats.get('tree_count', 0)
                    risk_level = stats.get('dominant_risk', 'N/A')
                    center = stats.get('center', ['N/A', 'N/A'])
                    
                    html_content += f"""
                    <tr>
                        <td>{cell_id[-8:]}</td>
                        <td>{tree_count}</td>
                        <td>{risk_level}</td>
                        <td>{center[1]}, {center[0]}</td>
                    </tr>
                    """
            
            html_content += "</table>"
        else:
            html_content += "<p>No S2 grouping data available</p>"
        
        # Add SAM segmentation section
        html_content += """
            <h2>SAM Segmentation (ML Overlay)</h2>
        """
        
        if 'overlay' in segmentation_data and segmentation_data['overlay']:
            # Copy the segmentation overlay to output directory
            overlay_path = os.path.join(output_dir, "segmentation_overlay.png")
            segmentation_data['overlay'].save(overlay_path)
            
            html_content += f"""
            <div class="stats">
                <p><strong>Segmentation Completed:</strong> Yes</p>
            </div>
            
            <h3>Segmentation Overlay</h3>
            <div class="image-container">
                <img src="{os.path.basename(overlay_path)}" alt="Segmentation Overlay">
            </div>
            """
        else:
            html_content += "<p>No segmentation data available</p>"
        
        # Close HTML
        html_content += """
        </body>
        </html>
        """
        
        # Write HTML to file
        report_path = os.path.join(output_dir, "ml_pipeline_report.html")
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Created ML pipeline report at {report_path}")
        return report_path
    
    except Exception as e:
        logger.error(f"Error creating report: {e}")
        return None

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="View ML pipeline results with S2 indexing")
    parser.add_argument("--dir", default=DEFAULT_RESULTS_DIR, help="Base directory for ML pipeline results")
    parser.add_argument("--output", default="/ttt/data/temp/ml_report", help="Output directory for report")
    args = parser.parse_args()
    
    # Find latest results
    results = find_latest_results(args.dir)
    if not results:
        logger.error(f"Could not find ML pipeline results in {args.dir}")
        return
    
    # Load detection results
    detection_data = load_detection_results(results)
    
    # Load segmentation results
    segmentation_data = load_segmentation_results(results)
    
    # Create report
    report_path = create_report(results, detection_data, segmentation_data, args.output)
    if report_path:
        print(f"\nML Pipeline Report created at: {report_path}")
        print(f"Open this file in a web browser to view the results.")
    else:
        print("\nFailed to create ML Pipeline Report.")

if __name__ == "__main__":
    main()