#!/usr/bin/env python3
"""
Tree Risk Assessment Pipeline
Main entry point that connects data collection, image processing, and ML analysis components
"""

import os
import sys
import logging
import asyncio
import time
import json
import argparse
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Import required packages
import yaml
import redis
import zarr
import numpy as np
import cv2

# Import component modules using proper package structure
from tree_ml.pipeline.data_collection import PipelineConfig, DataCollector
from tree_ml.pipeline.image_processing import ImageProcessor
from tree_ml.pipeline.zarr_store import StorageManager
from tree_ml.pipeline.object_recognition import MLConfig, ModelManager, InferenceEngine, TreeFeatureExtractor, TreeRiskAssessment

class Pipeline:
    """Main pipeline for tree risk assessment that connects all components"""
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.data_collector = DataCollector(config)
        self.image_processor = ImageProcessor(config)
        self.storage_manager = StorageManager(config)
        
        # Initialize ML pipeline if configured
        self.ml_pipeline = None
        if hasattr(config, 'use_ml') and config.use_ml:
            # Convert PipelineConfig to MLConfig
            ml_config = self._create_ml_config(config)
            self.ml_pipeline = MLPipeline(ml_config)
            # Set direct ML processing by default
            if not hasattr(config, 'run_ml_directly'):
                config.run_ml_directly = True
            self.logger.info("ML pipeline enabled")
        else:
            self.logger.info("ML pipeline disabled")
        
        self.redis_client = None
    
    def _create_ml_config(self, pipeline_config: PipelineConfig) -> MLConfig:
        """Convert pipeline config to ML config"""
        ml_config = MLConfig()
        
        # Config mappings for more maintainable code
        param_mappings = {
            # Path settings
            'model_path': 'model_path',
            'export_path': 'export_path',
            'store_path': 'store_path',
            'ml_path': 'ml_path',
            
            # Hardware settings
            'gpu_device_id': 'gpu_device_id',
            'gpu_vram_gb': 'gpu_vram_gb',
            'num_cpu_cores': 'num_cpu_cores',
            
            # Redis settings
            'redis_host': 'redis_host',
            'redis_ports': 'redis_ports',
            
            # Model settings
            'yolo_confidence': 'yolo_confidence',
            'sam_iou_threshold': 'sam_iou_threshold',
            'deepforest_score_threshold': 'deepforest_score_threshold'
        }
        
        # Copy attributes if they exist
        for pipeline_attr, ml_attr in param_mappings.items():
            if hasattr(pipeline_config, pipeline_attr):
                setattr(ml_config, ml_attr, getattr(pipeline_config, pipeline_attr))
                
        return ml_config
    
    async def process_mission(self, mission_id: str) -> Dict:
        """Process a complete mission through all pipeline stages"""
        start_time = time.time()
        
        try:
            # Update mission status
            self._update_mission_status(mission_id, "initializing", "Starting mission processing")
            
            # Step 1: Collect data
            self.logger.info(f"[1/4] Collecting data for mission {mission_id}")
            self._update_mission_status(mission_id, "collecting", "Collecting sensor data")
            raw_data = await self.data_collector.collect_data(mission_id)
            
            # Check if we have RGB data
            if not raw_data['rgb']['images']:
                return self._create_error_response(mission_id, "No RGB data found for mission", start_time)
                
            # Step 2: Process images
            self.logger.info("[2/4] Processing images")
            self._update_mission_status(mission_id, "processing", "Processing sensor data")
            processed_data = await self.image_processor.process_data(
                raw_data['rgb'],
                raw_data['lidar'],
                raw_data['imu'],
                raw_data['metadata']
            )
            
            # Check for processing errors
            if 'error' in processed_data:
                return self._create_error_response(
                    mission_id, 
                    f"Error during image processing: {processed_data['error']}", 
                    start_time
                )
                
            # Step 3: Store data
            self.logger.info("[3/4] Storing processed data")
            self._update_mission_status(mission_id, "storing", "Storing processed data")
            store_path = await self.storage_manager.store_data(processed_data, mission_id)
            
            # Step 4: Run ML analysis directly
            ml_job_id = None
            ml_results = None
            
            if self.ml_pipeline:
                # Run ML analysis directly
                self.logger.info("[4/4] Running ML analysis")
                self._update_mission_status(mission_id, "analyzing", "Running ML analysis")
                
                try:
                    # Initialize ML pipeline if needed
                    await self._initialize_ml_pipeline()
                    
                    # Process the stored data
                    ml_results = await self.ml_pipeline.process_zarr_store(store_path, mission_id)
                    ml_job_id = f"direct_{mission_id}_{int(time.time())}"
                except Exception as e:
                    self.logger.error(f"Error in ML pipeline processing: {e}")
                    raise RuntimeError(f"ML pipeline processing failed: {e}")
            
            # Calculate statistics
            processing_time = time.time() - start_time
            
            # Create result summary
            frame_count = len(processed_data['frames']) if 'frames' in processed_data else 0
            tile_count = sum(len(frame) for frame in processed_data['frames']) if 'frames' in processed_data else 0
            
            # Check if ML results contain trees
            tree_count = 0
            if ml_results and 'trees' in ml_results and ml_results['trees']:
                tree_count = len(ml_results['trees'])
                self.logger.info(f"ML pipeline detected {tree_count} trees")
            else:
                self.logger.warning("ML pipeline did not detect any trees")
            
            result = {
                'mission_id': mission_id,
                'store_path': store_path,
                'ml_job_id': ml_job_id,
                'ml_results': ml_results,
                'frame_count': frame_count,
                'tile_count': tile_count,
                'tree_count': tree_count,
                'processing_time': processing_time,
                'start_time': start_time,
                'end_time': time.time(),
                'status': 'complete'
            }
            
            # Log completion
            self.logger.info(f"Mission {mission_id} completed in {processing_time:.2f} seconds")
            self.logger.info(f"Processed {frame_count} frames with {tile_count} tiles")
            self.logger.info(f"Data stored at {store_path}")
            
            # Update mission status
            completion_message = f"Processing completed in {processing_time:.2f}s. Data stored at {store_path}"
            if ml_results and 'risk_assessment' in ml_results:
                risk_level = ml_results['risk_assessment'].get('risk_level', 'Unknown')
                completion_message += f" Risk assessment: {risk_level}"
                
            self._update_mission_status(
                mission_id, 
                "complete", 
                completion_message
            )
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Error processing mission {mission_id}: {str(e)}"
            
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            
            # Update mission status
            self._update_mission_status(mission_id, "error", error_msg)
            
            return {
                'mission_id': mission_id,
                'status': 'error',
                'error': str(e),
                'traceback': traceback.format_exc(),
                'processing_time': processing_time
            }
    
    def _update_mission_status(self, mission_id: str, status: str, message: str):
        """Update mission status"""
        self.logger.info(f"Mission {mission_id} status: {status} - {message}")
        
    async def _initialize_ml_pipeline(self):
        """Initialize ML pipeline if not already initialized"""
        if not self.ml_pipeline._initialized:
            try:
                self.logger.info("Initializing ML pipeline")
                init_success = await self.ml_pipeline.initialize()
                if init_success:
                    self.ml_pipeline._initialized = True
                    self.logger.info("ML pipeline initialized successfully")
                else:
                    self.logger.error("ML pipeline initialization failed")
                    raise RuntimeError("ML pipeline initialization failed")
            except Exception as e:
                self.logger.error(f"ML pipeline initialization failed with error: {e}")
                raise RuntimeError(f"Failed to initialize ML pipeline: {e}")
        
    def _create_error_response(self, mission_id: str, error_msg: str, start_time: float) -> Dict:
        """Create standardized error response"""
        self.logger.error(error_msg)
        self._update_mission_status(mission_id, "error", error_msg)
        return {
            'mission_id': mission_id,
            'status': 'error',
            'error': error_msg,
            'processing_time': time.time() - start_time
        }
        
    def _create_batch_error_response(self, batch_id: str, error_msg: str, start_time: float) -> Dict:
        """Create standardized error response for batch operations"""
        self.logger.error(error_msg)
        return {
            'batch_id': batch_id,
            'status': 'error',
            'error': error_msg,
            'processing_time': time.time() - start_time
        }
    
    # Removed ML queuing functionality to simplify code
    
    async def process_batch(self, images: List, mission_id: str = None, map_coordinates: Dict = None) -> Dict:
        """Process a batch of images or map coordinates without a full mission context"""
        start_time = time.time()
        
        try:
            # Generate a mission ID if not provided
            if mission_id is None:
                mission_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
            # Determine if we're processing map coordinates or images
            if map_coordinates:
                # Load coordinates from a file if a file path is provided
                coords_data = map_coordinates
                self.logger.info(f"Processing with map coordinates: {coords_data}")
                
                # Use data collector to fetch and process map imagery 
                raw_data = await self.data_collector.collect_data(mission_id, coords_data)
                
                # Process the collected data
                self.logger.info(f"Processing map imagery for coordinates")
                processed_data = await self.image_processor.process_data(raw_data['rgb'])
                
            elif len(images) > 0:
                # Process provided images
                timestamps = [datetime.now() for _ in images]
                metadata = [{
                    'timestamp': ts,
                    'batch_processing': True,
                    'is_map_detection': False
                } for ts in timestamps]
                
                rgb_data = {
                    'images': images,
                    'timestamps': timestamps,
                    'metadata': metadata,
                    'is_map_detection': False
                }
                
                # Process images
                self.logger.info(f"Processing batch of {len(images)} images")
                processed_data = await self.image_processor.process_data(rgb_data)
                
            else:
                # If no images and no coordinates, return empty results
                self.logger.warning("No images or coordinates provided for processing")
                return self._create_batch_error_response(mission_id, "No images or map coordinates provided", start_time)
            
            # Store data
            store_path = await self.storage_manager.store_data(processed_data, mission_id)
            
            # Run ML analysis if configured
            ml_results = None
            if self.ml_pipeline and hasattr(self.config, 'run_ml_directly') and self.config.run_ml_directly:
                self.logger.info("Running ML analysis on batch")
                
                try:
                    # Initialize ML pipeline if needed
                    await self._initialize_ml_pipeline()
                    
                    # Process the stored data
                    ml_results = await self.ml_pipeline.process_zarr_store(store_path, mission_id)
                except Exception as e:
                    self.logger.error(f"Error in ML pipeline processing: {e}")
                    raise RuntimeError(f"ML pipeline processing failed: {e}")
            
            # Calculate statistics
            processing_time = time.time() - start_time
            frame_count = len(processed_data['frames']) if 'frames' in processed_data else 0
            tile_count = sum(len(frame) for frame in processed_data['frames']) if 'frames' in processed_data else 0
            
            result = {
                'batch_id': mission_id,
                'store_path': store_path,
                'frame_count': frame_count,
                'tile_count': tile_count,
                'processing_time': processing_time,
                'ml_results': ml_results,
                'status': 'complete'
            }
            
            self.logger.info(f"Batch processing completed in {processing_time:.2f} seconds")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            self.logger.error(f"Error in batch processing: {str(e)}")
            self.logger.error(traceback.format_exc())
            
            if mission_id:
                return self._create_batch_error_response(mission_id, str(e), start_time)
            else:
                return {
                    'status': 'error',
                    'error': str(e),
                    'traceback': traceback.format_exc(),
                    'processing_time': processing_time
                }
    
    async def list_missions(self) -> List[Dict]:
        """List all available missions"""
        try:
            missions = []
            
            # Look for mission directories in the input path
            input_path = Path(self.config.input_path)
            if not input_path.exists():
                return missions
                
            # Check each subdirectory
            for item in input_path.iterdir():
                if item.is_dir():
                    mission_id = item.name
                    
                    # Check if this looks like a mission directory
                    if (item / 'rgb').exists() or (item / 'lidar').exists() or (item / 'imu').exists():
                        # Get basic info
                        mission_info = {
                            'mission_id': mission_id,
                            'path': str(item),
                            'has_rgb': (item / 'rgb').exists(),
                            'has_lidar': (item / 'lidar').exists(),
                            'has_imu': (item / 'imu').exists()
                        }
                        
                        # Try to get more info from metadata
                        metadata_file = item / 'metadata.json'
                        if metadata_file.exists():
                            try:
                                with open(metadata_file) as f:
                                    metadata = json.load(f)
                                    mission_info.update(metadata)
                            except Exception as e:
                                self.logger.warning(f"Error reading metadata for mission {mission_id}: {e}")
                                
                        # Check for ML results
                        ml_results_path = Path(self.config.store_path) / f"{mission_id}_ml_results"
                        if ml_results_path.exists():
                            mission_info['has_ml_results'] = True
                            
                            # Try to get risk assessment info
                            try:
                                results_store = zarr.open(str(ml_results_path), mode='r')
                                if 'mission' in results_store and 'risk' in results_store.mission:
                                    risk_level = results_store.mission.risk.attrs.get('risk_level', 'Unknown')
                                    mission_info['risk_level'] = risk_level
                            except Exception as e:
                                self.logger.warning(f"Error reading ML results for mission {mission_id}: {e}")
                        else:
                            mission_info['has_ml_results'] = False
                                
                        # Add to list
                        missions.append(mission_info)
                        
            return missions
            
        except Exception as e:
            self.logger.error(f"Error listing missions: {e}")
            return []

class MLPipeline:
    """ML Pipeline that integrates with the main data processing pipeline"""
    def __init__(self, config: MLConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._initialized = False
        
        # Initialize components
        self.model_manager = ModelManager(config)
        self.inference_engine = InferenceEngine(config)
        self.feature_extractor = TreeFeatureExtractor(config)
        self.risk_assessment = TreeRiskAssessment(config)
        
    async def initialize(self):
        """Initialize models and engines"""
        try:
            # Initialize model
            self.logger.info("Initializing ML model...")
            await self.model_manager.initialize()
            
            # Export model to ONNX
            self.logger.info("Exporting model to ONNX...")
            onnx_path = await self.model_manager.export_onnx()
            
            # Initialize inference engine
            self.logger.info("Initializing inference engine...")
            await self.inference_engine.initialize(onnx_path)
            
            self._initialized = True
            self.logger.info("ML Pipeline initialization complete")
            return True
        except Exception as e:
            self.logger.error(f"Error initializing ML pipeline: {e}")
            self._initialized = False
            return False
            
    async def process_zarr_store(self, store_path: str, mission_id: str) -> dict:
        """Process data from Zarr store"""
        self.logger.info(f"Processing Zarr store: {store_path}")
        start_time = time.time()
        
        try:
            # Create a new results store inside the detection_ml_results directory
            # Use a simple "results" directory to store frame data
            results_path = os.path.join(os.path.dirname(store_path), "results")
            self.logger.info(f"Using results path: {results_path}")
            
            # Remove any existing results directory to start fresh
            if os.path.exists(results_path):
                self.logger.info(f"Removing existing results directory: {results_path}")
                import shutil
                shutil.rmtree(results_path)
            
            # Create the directory and new zarr store
            os.makedirs(results_path, exist_ok=True)
            results_store = zarr.open(results_path, mode='w')
            
            # Create frames group in the results store
            frames_group = results_store.create_group("frames")
            self.logger.info("Created frames group in results store")
            
            # Try to load satellite image directly since there's no input store with frames
            try:
                # Look for any jpg files in the ML directory
                ml_dir = os.path.dirname(os.path.dirname(results_path))
                image_files = [f for f in os.listdir(ml_dir) if f.endswith('.jpg')]
                
                if image_files:
                    self.logger.info(f"Found satellite images in ML dir: {image_files}")
                    # Process one frame per satellite image
                    frame_count = 1
                else:
                    self.logger.warning("No satellite images found in ML directory")
                    frame_count = 0
            except Exception as e:
                self.logger.error(f"Error checking for satellite images: {e}")
                frame_count = 0
            
            # If no frames found in input, create at least one frame for processing
            if frame_count == 0:
                self.logger.info("No frames found in input, creating one frame for processing")
                frame_count = 1
                
            all_risk_assessments = []
            
            # Process frames
            for frame_idx in range(frame_count):
                try:
                    self.logger.info(f"Processing frame {frame_idx+1}/{frame_count}")
                    
                    # Initialize variables
                    rgb_data = None
                    lidar_data = None
                    geo_transform = None
                    input_store = None  # Define the store variable previously referenced but not defined
                    
                    # Try to load satellite image from ML directory
                    try:
                        # Look for any jpg files in the ML directory
                        ml_dir = os.path.dirname(os.path.dirname(results_path))
                        image_files = [f for f in os.listdir(ml_dir) if f.endswith('.jpg')]
                        
                        if image_files:
                            # Load the first satellite image found
                            image_path = os.path.join(ml_dir, image_files[0])
                            self.logger.info(f"Loading satellite image from: {image_path}")
                            
                            rgb_data = cv2.imread(image_path)
                            # Convert from BGR to RGB
                            rgb_data = cv2.cvtColor(rgb_data, cv2.COLOR_BGR2RGB)
                            self.logger.info(f"Loaded satellite image with shape: {rgb_data.shape}")
                        else:
                            self.logger.warning("No satellite images found in ML directory")
                    except Exception as e:
                        self.logger.error(f"Error loading satellite image: {e}")
                    
                    # If no RGB data, abort processing
                    if rgb_data is None:
                        self.logger.error("No RGB data available for processing")
                        raise ValueError("No RGB data available for processing")
                    
                    # Run inference on the RGB data
                    detections = await self.inference_engine.infer(rgb_data)
                    self.logger.info(f"Inference successful with {len(detections.get('yolo_detections', []))} YOLO detections")
                    
                    # Extract features from detections
                    features = await self.feature_extractor.extract(
                        detections, 
                        rgb_data, 
                        lidar_data
                    )
                    self.logger.info(f"Feature extraction successful with {len(features.get('tree_features', []))} tree features")
                    
                    # Assess risk
                    risk_assessment = await self.risk_assessment.assess_risk(features)
                    self.logger.info(f"Risk assessment completed successfully")
                    
                    # Store results in frames group
                    # Create frame group
                    frame_group = frames_group.create_group(str(frame_idx))
                    self.logger.info(f"Created frame group {frame_idx}")
                    
                    # Store detections
                    detections_group = frame_group.create_group('detections')
                    for key, value in detections.items():
                        if key != 'inference_time' and isinstance(value, np.ndarray):
                            detections_group.create_dataset(key, data=value)
                    
                    # Store inference time
                    frame_group.create_dataset('inference_time', data=np.array([detections.get('inference_time', 0)]))
                    
                    # Store metadata with geo_transform if available
                    if geo_transform is not None:
                        metadata_group = frame_group.create_group('metadata')
                        metadata_group.create_dataset('geo_transform', data=geo_transform)
                        self.logger.info(f"Stored geo_transform for frame {frame_idx}")
                    
                    # Store risk assessment
                    risk_group = frame_group.create_group('risk')
                    risk_group.attrs['risk_level'] = risk_assessment['overall_risk']['risk_level']
                    risk_group.attrs['risk_score'] = float(risk_assessment['overall_risk']['risk_score'])
                    
                    self.logger.info(f"Successfully stored frame {frame_idx} results")
                    
                    all_risk_assessments.append(risk_assessment)
                    
                except Exception as e:
                    self.logger.error(f"Error processing frame {frame_idx}: {e}", exc_info=True)
            
            # Calculate mission-level risk assessment
            mission_risk = self._calculate_mission_risk(all_risk_assessments)
            
            # Store mission-level results
            self._store_mission_results(results_store, mission_risk)
            
            processing_time = time.time() - start_time
            self.logger.info(f"ML processing completed in {processing_time:.2f} seconds")
            
            # Generate JSON output if requested
            trees_json = None
            if hasattr(self.config, 'output_json') and self.config.output_json:
                # Get geo bounds from the original data if available
                geo_bounds = None
                input_store = zarr.open(store_path, mode='r')
                if hasattr(input_store, 'metadata') and 'coordinates' in input_store.metadata.attrs:
                    try:
                        coordinates = json.loads(input_store.metadata.attrs['coordinates'])
                        if 'bounds' in coordinates:
                            geo_bounds = coordinates['bounds']
                    except:
                        self.logger.warning("Failed to parse geo bounds from store metadata")
                
                # Generate JSON results
                trees_json = await self.risk_assessment.generate_json_results(
                    features,
                    mission_id,
                    geo_bounds
                )
                
                # Write JSON output if path is specified
                if self.config.output_path:
                    try:
                        os.makedirs(os.path.dirname(self.config.output_path), exist_ok=True)
                        with open(self.config.output_path, 'w') as f:
                            json.dump(trees_json, f, indent=2)
                        self.logger.info(f"Detection results written to {self.config.output_path}")
                    except Exception as e:
                        self.logger.error(f"Error writing JSON output: {e}")
            
            # Redundant JSON writing was removed (already done above)
            
            return {
                'mission_id': mission_id,
                'store_path': results_path,
                'frame_count': frame_count,
                'processing_time': processing_time,
                'risk_assessment': mission_risk,
                'trees': trees_json['trees'] if trees_json else [],
                'tree_count': len(trees_json['trees']) if trees_json and 'trees' in trees_json else 0,
                'status': 'complete'
            }
            
        except Exception as e:
            self.logger.error(f"Error processing Zarr store: {e}", exc_info=True)
            
            return {
                'mission_id': mission_id,
                'store_path': store_path,
                'status': 'error',
                'error': str(e)
            }
    
    def _convert_features_to_arrays(self, features):
        """Convert list of feature dictionaries to arrays for storage"""
        if not features:
            return {}
            
        result = {}
        
        # Find all keys
        all_keys = set()
        for feature in features:
            all_keys.update(feature.keys())
            
        # Convert each key to an array
        for key in all_keys:
            values = []
            for feature in features:
                if key in feature:
                    value = feature[key]
                    if isinstance(value, (list, tuple, np.ndarray)) and len(value) > 0:
                        values.append(np.array(value))
                    else:
                        values.append(value)
                else:
                    # Use appropriate empty value
                    if any(isinstance(f.get(key, None), (list, tuple, np.ndarray)) for f in features if key in f):
                        # Determine shape from other features
                        for f in features:
                            if key in f and isinstance(f[key], (list, tuple, np.ndarray)):
                                empty_shape = np.array(f[key]).shape
                                values.append(np.zeros(empty_shape))
                                break
                    else:
                        values.append(0)
                        
            # Convert to numpy array
            try:
                result[key] = np.array(values)
            except ValueError:
                # If values have different shapes, store as object array
                result[key] = np.array(values, dtype=object)
                
        return result
        
    def _store_mission_results(self, store, mission_risk):
        """Store mission-level results"""
        # Create mission group
        mission = store.create_group('mission')
        
        # Store mission risk assessment
        risk_group = mission.create_group('risk')
        
        # Store aggregated risk metrics
        if 'risk_level' in mission_risk:
            risk_group.attrs['risk_level'] = mission_risk['risk_level']
        if 'risk_score' in mission_risk:
            risk_group.attrs['risk_score'] = mission_risk['risk_score']
        if 'high_risk_areas' in mission_risk:
            risk_group.create_dataset('high_risk_areas', data=np.array(mission_risk['high_risk_areas']))
        
        # Store additional mission statistics
        if 'tree_count' in mission_risk:
            mission.attrs['tree_count'] = mission_risk['tree_count']
        if 'high_risk_count' in mission_risk:
            mission.attrs['high_risk_count'] = mission_risk['high_risk_count']
        if 'medium_risk_count' in mission_risk:
            mission.attrs['medium_risk_count'] = mission_risk['medium_risk_count']
        if 'low_risk_count' in mission_risk:
            mission.attrs['low_risk_count'] = mission_risk['low_risk_count']
            
    def _calculate_mission_risk(self, frame_risk_assessments):
        """Calculate mission-level risk assessment from frame assessments"""
        if not frame_risk_assessments:
            return {
                'risk_level': 'Low',
                'risk_score': 0,
                'tree_count': 0,
                'high_risk_count': 0,
                'medium_risk_count': 0,
                'low_risk_count': 0,
                'high_risk_areas': []
            }
            
        # Aggregate all trees and risk levels
        all_trees = []
        high_risk_areas = []
        
        high_risk_count = 0
        medium_risk_count = 0
        low_risk_count = 0
        
        # Aggregate risk counts from all frames
        for frame_idx, risk in enumerate(frame_risk_assessments):
            if 'overall_risk' in risk:
                high_risk_count += risk['overall_risk'].get('high_risk_count', 0)
                medium_risk_count += risk['overall_risk'].get('medium_risk_count', 0)
                low_risk_count += risk['overall_risk'].get('low_risk_count', 0)
                
            if 'risk_factors' in risk:
                # Add frame index to each tree for tracking
                for tree_risk in risk['risk_factors']:
                    tree_risk['frame_idx'] = frame_idx
                    all_trees.append(tree_risk)
                    
                    # Track high risk areas
                    if tree_risk['risk_level'] == 'High':
                        high_risk_areas.append({
                            'frame_idx': frame_idx,
                            'tree_id': tree_risk['tree_id'],
                            'bbox': tree_risk['bbox'],
                            'risk_score': tree_risk['risk_score'],
                            'risk_factors': tree_risk['risk_factors']
                        })
                    
        # Calculate overall mission risk
        tree_count = len(all_trees)
        
        if tree_count == 0:
            avg_risk_score = 0
        else:
            avg_risk_score = sum(tree['risk_score'] for tree in all_trees) / tree_count
        
        # Determine overall mission risk level
        if high_risk_count > 0 or avg_risk_score > 5.0:
            mission_risk_level = 'High'
        elif medium_risk_count > 0 or avg_risk_score > 3.0:
            mission_risk_level = 'Medium'
        else:
            mission_risk_level = 'Low'
            
        return {
            'risk_level': mission_risk_level,
            'risk_score': float(avg_risk_score),
            'tree_count': tree_count,
            'high_risk_count': high_risk_count,
            'medium_risk_count': medium_risk_count,
            'low_risk_count': low_risk_count,
            'high_risk_areas': high_risk_areas
        }
        
    # ML worker functionality removed to simplify code

def load_config(config_file: str = None) -> PipelineConfig:
    """Load configuration from file or environment variables"""
    config = PipelineConfig()
    
    # Load from config file if provided
    if config_file and os.path.exists(config_file):
        try:
            # Determine format based on extension
            if config_file.endswith('.json'):
                with open(config_file) as f:
                    config_data = json.load(f)
            elif config_file.endswith(('.yaml', '.yml')):
                with open(config_file) as f:
                    config_data = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_file}")
                
            # Update config with file values
            for key, value in config_data.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                    
            logging.info(f"Loaded configuration from {config_file}")
            
        except Exception as e:
            logging.error(f"Error loading config from {config_file}: {e}")
    
    # Override with environment variables
    for key in dir(config):
        # Skip special attributes and methods
        if key.startswith('_') or callable(getattr(config, key)):
            continue
            
        # Check for corresponding environment variable
        env_key = f"TREE_PIPELINE_{key.upper()}"
        if env_key in os.environ:
            env_value = os.environ[env_key]
            
            # Try to convert to the correct type
            current_value = getattr(config, key)
            if isinstance(current_value, bool):
                setattr(config, key, env_value.lower() in ('true', 'yes', '1'))
            elif isinstance(current_value, int):
                setattr(config, key, int(env_value))
            elif isinstance(current_value, float):
                setattr(config, key, float(env_value))
            elif isinstance(current_value, list) or isinstance(current_value, tuple):
                setattr(config, key, env_value.split(','))
            elif isinstance(current_value, dict):
                try:
                    setattr(config, key, json.loads(env_value))
                except:
                    logging.warning(f"Could not parse dictionary from {env_key}")
            else:
                setattr(config, key, env_value)
                
            logging.info(f"Set {key} from environment variable {env_key}")
    
    return config

async def main():
    """Main entry point for the pipeline"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Tree Risk Assessment Pipeline")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--mission", help="Mission ID to process")
    parser.add_argument("--list", action="store_true", help="List available missions")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--image", help="Process a single image (batch mode)")
    parser.add_argument("--coordinates", help="JSON file with map coordinates for processing")
    parser.add_argument("--zarr", help="Path to zarr directory for persistent storage")
    parser.add_argument("--mlstore", help="Path to directory for ML detection storage")
    parser.add_argument("--temp", action="store_true", help="Use temporary storage for processing")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument("--output", help="Path to output file for JSON results")
    parser.add_argument("--include-bounding-boxes", action="store_true", help="Include bounding boxes in results")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load configuration
    config = load_config(args.config)
    
    # Override zarr directory if specified
    if args.zarr:
        config.store_path = args.zarr
    
    # Override ML store directory if specified
    if args.mlstore:
        config.store_path = args.mlstore
        # Set flag to use the mlstore directory directly without adding suffixes
        config.use_direct_mlstore = True
        print(f"Using ML store path directly: {config.store_path}")
        
    # Set up ML path if using temporary storage
    if args.temp:
        config.use_temp_storage = True
        # Default ML directory is under the main store path
        config.ml_path = os.path.join(config.store_path, 'ml')
        print(f"Using ML storage at {config.ml_path}")
    else:
        config.use_temp_storage = False
        
    # Set JSON output options
    if args.json:
        config.output_json = True
        config.include_bounding_boxes = args.include_bounding_boxes
        if args.output:
            config.output_path = args.output
    else:
        config.output_json = False
        
    # Always enable ML pipeline for detection
    config.use_ml = True
    
    # Create pipeline
    pipeline = Pipeline(config)
    
    # Process command
    if args.list:
        # List available missions
        missions = await pipeline.list_missions()
        print(f"Found {len(missions)} missions:")
        for mission in missions:
            print(f"  {mission['mission_id']}")
            print(f"    Path: {mission['path']}")
            print(f"    Data: RGB={'Yes' if mission['has_rgb'] else 'No'}, "
                  f"LiDAR={'Yes' if mission['has_lidar'] else 'No'}, "
                  f"IMU={'Yes' if mission['has_imu'] else 'No'}")
            
            # Print ML results if available
            if mission.get('has_ml_results', False):
                print(f"    ML Results: Yes")
                if 'risk_level' in mission:
                    print(f"    Risk Level: {mission['risk_level']}")
            else:
                print(f"    ML Results: No")
                
            # Print additional info if available
            for key in mission:
                if key not in ['mission_id', 'path', 'has_rgb', 'has_lidar', 'has_imu', 'has_ml_results', 'risk_level']:
                    print(f"    {key}: {mission[key]}")
            print()
            
    elif args.coordinates:
        # Process a region based on map coordinates
        try:
            # Load coordinates from file
            with open(args.coordinates, 'r') as f:
                coordinates_data = json.load(f)
                
            print(f"Processing coordinates: {json.dumps(coordinates_data)}")
            
            # Add mission_id or use a default
            mission_id = args.mission or f"coord_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Pass the coordinates directly
            result = await pipeline.process_batch([], mission_id, coordinates_data)
            
            # Check if JSON output is requested
            if args.json and args.output:
                # Extract tree detections in a simplified format
                json_output = {
                    "mission_id": mission_id,
                    "status": "complete",
                    "timestamp": datetime.now().isoformat(),
                    "tree_count": 0,
                    "trees": []
                }
                
                # Add metadata
                json_output["metadata"] = {
                    "coordinates": coordinates_data,
                    "include_bounding_boxes": args.include_bounding_boxes
                }
                
                # Extract actual tree data from ML results
                if result.get('ml_results') and result.get('ml_results').get('trees'):
                    json_output["trees"] = result['ml_results']['trees']
                else:
                    # No trees were detected or ML processing failed
                    json_output["status"] = "complete_no_detections"
                
                json_output["tree_count"] = len(json_output["trees"])
                
                # Write output to file
                with open(args.output, 'w') as f:
                    json.dump(json_output, f, indent=2)
                print(f"Detection results written to {args.output}")
            else:
                # Print standard output
                print("Coordinate processing result:")
                for key, value in result.items():
                    if key not in ['ml_results', 'traceback']:  # Skip printing complex fields
                        print(f"  {key}: {value}")
                
        except Exception as e:
            print(f"Error processing coordinates: {e}")
            traceback.print_exc()
                
    elif args.image:
        # Process a single image
        try:
            # Load image
            img = cv2.imread(args.image)
            if img is None:
                print(f"Error: Could not load image {args.image}")
                return
                
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Add mission_id or use a default
            mission_id = args.mission or f"img_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            print(f"Processing image file: {args.image} with shape {img.shape}")
            
            # Process image
            result = await pipeline.process_batch([img], mission_id)
            
            # Print results
            print("Batch processing result:")
            for key, value in result.items():
                if key not in ['ml_results', 'traceback']:  # Skip printing complex fields
                    print(f"  {key}: {value}")
                
        except Exception as e:
            print(f"Error processing image: {e}")
            traceback.print_exc()
            
    elif args.mission:
        # Process a mission
        result = await pipeline.process_mission(args.mission)
        
        # Print results
        print("Mission processing result:")
        for key, value in result.items():
            if key not in ['traceback', 'ml_results']:  # Skip printing complex fields
                print(f"  {key}: {value}")
                
    else:
        parser.print_help()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Process interrupted by user")
        sys.exit(1)