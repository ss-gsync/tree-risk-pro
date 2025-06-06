#!/usr/bin/env python3
"""
Image Processor - Batch satellite image processing for Tree ML project

This script processes multiple satellite/aerial images through the Tree ML
detection pipeline, saving results in a standardized format. It's designed
for batch processing of test images to validate ML model performance.

Features:
- Process single or multiple satellite images
- Support for different ML models (DeepForest, YOLO)
- Detailed result reporting
- Visualization generation
- Standardized output directory structure

Usage examples:
  # Process all images in the default directory
  python image_processor.py

  # Process specific images
  python image_processor.py --images /path/to/images

  # Clean results directory before processing
  python image_processor.py --clean

  # Use a specific model
  python image_processor.py --model deepforest

  # Filter images by name
  python image_processor.py --filter satellite
"""

import os
import sys
import json
import shutil
import logging
import argparse
import subprocess
from glob import glob
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('image_processor')

# Default paths
DATA_DIR = "/ttt/data/tests"
TEST_IMAGES_DIR = os.path.join(DATA_DIR, "ml_test_images")
TEST_RESULTS_DIR = os.path.join(DATA_DIR, "ml_test_results")


class ImageProcessor:
    """Main class for processing satellite images"""
    
    def __init__(self, args):
        """Initialize the image processor with arguments"""
        self.args = args
        self.images_dir = args.images or TEST_IMAGES_DIR
        self.results_dir = args.results or TEST_RESULTS_DIR
        self.model_type = args.model
        self.filter_pattern = args.filter
        
        # Set up directories
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Clean results if requested
        if args.clean:
            self.clean_results_directory()
        
        # Keep track of processing statistics
        self.stats = {
            "total_images": 0,
            "processed_images": 0,
            "failed_images": 0,
            "total_trees_detected": 0,
            "start_time": datetime.now().isoformat()
        }
    
    def clean_results_directory(self):
        """Clean the results directory"""
        logger.info(f"Cleaning results directory: {self.results_dir}")
        
        for item in os.listdir(self.results_dir):
            item_path = os.path.join(self.results_dir, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            elif os.path.isfile(item_path):
                os.remove(item_path)
        
        logger.info("Results directory cleaned")
    
    def get_images(self) -> List[Path]:
        """Get all image files to process"""
        if not os.path.exists(self.images_dir):
            logger.error(f"Images directory not found: {self.images_dir}")
            return []
        
        # Get all image files with common image extensions
        images = []
        for ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
            images.extend(Path(self.images_dir).glob(f"*{ext}"))
        
        # Sort by name
        images = sorted(images)
        
        if not images:
            logger.error(f"No images found in {self.images_dir}")
            return []
        
        # Apply filter if specified
        if self.filter_pattern:
            filtered_images = [img for img in images if self.filter_pattern in img.name]
            if not filtered_images:
                logger.error(f"No images match the filter pattern: {self.filter_pattern}")
                return []
            logger.info(f"Filtered to {len(filtered_images)} images matching '{self.filter_pattern}'")
            images = filtered_images
        
        # Update stats
        self.stats["total_images"] = len(images)
        
        # Log image list
        logger.info(f"Found {len(images)} images to process:")
        for img in images[:5]:  # Show first 5 only
            logger.info(f"  - {img.name}")
        
        if len(images) > 5:
            logger.info(f"  ... and {len(images) - 5} more")
        
        return images
    
    def process_image(self, image_path: Path) -> bool:
        """Process a single image and return success status"""
        image_name = image_path.stem
        logger.info(f"Processing image: {image_path.name}")
        
        # Create results directory for this image
        image_results_dir = os.path.join(self.results_dir, image_name)
        os.makedirs(image_results_dir, exist_ok=True)
        
        # Define output paths
        detection_json = os.path.join(image_results_dir, "detection_results.json")
        visualization_path = os.path.join(image_results_dir, "visualization.png")
        
        try:
            # Run detection with the Tree ML pipeline
            script = self._create_detection_script(
                image_path=str(image_path), 
                output_json=detection_json,
                image_name=image_name,
                visualization_path=visualization_path
            )
            
            # Execute the detection command
            cmd = f"cd /ttt && poetry run python -c \"{script}\""
            process = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if process.returncode != 0:
                logger.error(f"Error processing {image_path.name}:")
                logger.error(process.stderr)
                self.stats["failed_images"] += 1
                return False
            
            # Check if results were created
            if not os.path.exists(detection_json):
                logger.error(f"No detection results created for {image_path.name}")
                self.stats["failed_images"] += 1
                return False
            
            # Parse results and create summary
            with open(detection_json, 'r') as f:
                results = json.load(f)
            
            tree_count = results.get("tree_count", 0)
            logger.info(f"Detected {tree_count} trees in {image_path.name}")
            
            # Update stats
            self.stats["processed_images"] += 1
            self.stats["total_trees_detected"] += tree_count
            
            # Create summary file
            self._create_summary_file(
                image_path=image_path,
                results_dir=image_results_dir,
                results=results
            )
            
            return True
        
        except Exception as e:
            logger.error(f"Error processing {image_path.name}: {e}")
            self.stats["failed_images"] += 1
            return False
    
    def _create_detection_script(self, image_path: str, output_json: str, 
                                image_name: str, visualization_path: str) -> str:
        """Create the Python script for detection based on model type"""
        script = f"""
import os
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

# Import the appropriate detection function
from tree_ml.pipeline.object_recognition import detect_trees

# Ensure results directory exists
os.makedirs('{os.path.dirname(output_json)}', exist_ok=True)

# Run detection
results = detect_trees(
    '{image_path}', 
    '{output_json}', 
    job_id='{image_name}'
)

# Create visualization if trees were detected
tree_count = results.get('tree_count', 0)
if tree_count > 0:
    try:
        # Load original image
        original_img = Image.open('{image_path}')
        draw = ImageDraw.Draw(original_img)
        
        # Draw boxes for each tree
        for tree in results.get('trees', []):
            if 'bbox' in tree:
                bbox = tree['bbox']
                # Check bbox format
                if isinstance(bbox, list) and len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    confidence = tree.get('confidence', 0)
                    # Color based on confidence (green to red)
                    color = (int(255 * (1-confidence)), int(255 * confidence), 0)
                    # Draw rectangle
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                    # Add confidence text
                    draw.text((x1, y1-10), f"{{confidence:.2f}}", fill=color)
        
        # Save visualization
        original_img.save('{visualization_path}')
    except Exception as e:
        print(f"Error creating visualization: {{e}}")

# Print summary for logging
print(f"Processed {{tree_count}} trees in image {image_name}")
"""
        return script
    
    def _create_summary_file(self, image_path: Path, results_dir: str, results: Dict[str, Any]):
        """Create a human-readable summary file"""
        summary_path = os.path.join(results_dir, "summary.txt")
        
        try:
            tree_count = results.get("tree_count", 0)
            
            with open(summary_path, 'w') as f:
                f.write(f"Image: {image_path}\n")
                f.write(f"Processed: {datetime.now().isoformat()}\n")
                f.write(f"Model: {self.model_type}\n")
                f.write(f"Trees detected: {tree_count}\n\n")
                
                if tree_count > 0:
                    f.write("Tree Details:\n")
                    for i, tree in enumerate(results.get("trees", [])):
                        f.write(f"Tree {i+1}:\n")
                        
                        # Write confidence if available
                        confidence = tree.get("confidence", None)
                        if confidence is not None:
                            f.write(f"  Confidence: {confidence:.3f}\n")
                        
                        # Write bounding box if available
                        bbox = tree.get("bbox", None)
                        if bbox is not None:
                            f.write(f"  Bounding Box: {bbox}\n")
                        
                        # Write detection type if available
                        detection_type = tree.get("detection_type", None)
                        if detection_type is not None:
                            f.write(f"  Detection Type: {detection_type}\n")
                        
                        f.write("\n")
            
            logger.info(f"Summary created: {summary_path}")
        
        except Exception as e:
            logger.error(f"Error creating summary file: {e}")
    
    def create_batch_report(self):
        """Create a batch processing report"""
        self.stats["end_time"] = datetime.now().isoformat()
        
        # Calculate processing time
        start_time = datetime.fromisoformat(self.stats["start_time"])
        end_time = datetime.fromisoformat(self.stats["end_time"])
        processing_time = (end_time - start_time).total_seconds()
        self.stats["processing_time_seconds"] = processing_time
        
        # Add average stats
        if self.stats["processed_images"] > 0:
            self.stats["avg_trees_per_image"] = self.stats["total_trees_detected"] / self.stats["processed_images"]
            self.stats["avg_time_per_image"] = processing_time / self.stats["processed_images"]
        
        # Save report
        report_path = os.path.join(self.results_dir, "batch_processing_report.json")
        try:
            with open(report_path, 'w') as f:
                json.dump(self.stats, f, indent=2)
            logger.info(f"Batch processing report saved to {report_path}")
        except Exception as e:
            logger.error(f"Error saving batch report: {e}")
    
    def run(self):
        """Run batch processing on all images"""
        logger.info("=" * 60)
        logger.info("Tree ML Image Processor")
        logger.info("=" * 60)
        logger.info(f"Images directory: {self.images_dir}")
        logger.info(f"Results directory: {self.results_dir}")
        logger.info(f"Model: {self.model_type}")
        
        # Get images to process
        images = self.get_images()
        if not images:
            logger.error("No images to process, exiting")
            return False
        
        # Process each image
        for i, image_path in enumerate(images):
            logger.info(f"Processing image {i+1}/{len(images)}: {image_path.name}")
            self.process_image(image_path)
        
        # Create batch report
        self.create_batch_report()
        
        # Print summary
        logger.info("=" * 60)
        logger.info("Batch Processing Complete")
        logger.info("=" * 60)
        logger.info(f"Total images: {self.stats['total_images']}")
        logger.info(f"Successfully processed: {self.stats['processed_images']}")
        logger.info(f"Failed: {self.stats['failed_images']}")
        logger.info(f"Total trees detected: {self.stats['total_trees_detected']}")
        logger.info(f"Total processing time: {self.stats['processing_time_seconds']:.2f} seconds")
        
        # Return success if at least one image was processed
        return self.stats["processed_images"] > 0


def main():
    """Main function to parse arguments and run batch processing"""
    parser = argparse.ArgumentParser(description="Tree ML Image Processor")
    parser.add_argument(
        "--images", 
        help="Directory containing images to process (default: /ttt/data/tests/ml_test_images)"
    )
    parser.add_argument(
        "--results", 
        help="Directory to store results (default: /ttt/data/tests/ml_test_results)"
    )
    parser.add_argument(
        "--model", 
        choices=["deepforest", "yolo", "gemini"], 
        default="deepforest",
        help="ML model to use for detection (default: deepforest)"
    )
    parser.add_argument(
        "--clean", 
        action="store_true",
        help="Clean results directory before processing"
    )
    parser.add_argument(
        "--filter", 
        help="Only process images with names containing this pattern"
    )
    
    args = parser.parse_args()
    
    # Run processor
    processor = ImageProcessor(args)
    success = processor.run()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())