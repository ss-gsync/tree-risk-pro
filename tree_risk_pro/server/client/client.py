#!/usr/bin/env python3
"""
Client Test Script for Edge Cache Server
Generates sample data and tests server connectivity

# Install dependencies
pip install aiohttp numpy opencv-python tqdm

# Run test with small dataset (default)
python client_test.py --host 192.168.4.35 --port 3000 --size small
"""

import ssl
import asyncio
import aiohttp
import numpy as np
import cv2
import logging
import json
import os
from pathlib import Path
from datetime import datetime, timedelta
import argparse
from tqdm import tqdm
import sys

class TestDataGenerator:
    """Generates sample mission data for testing"""
    def __init__(self, output_path: Path, size: str = 'small'):
        self.output_path = output_path
        
        # Configure data size
        self.sizes = {
            'small': {
                'frame_count': 10,
                'resolution': (1920, 1080),
                'lidar_scans': 5,
                'points_per_scan': 1000,
                'imu_records': 100
            },
            'medium': {
                'frame_count': 50,
                'resolution': (1920, 1080),
                'lidar_scans': 25,
                'points_per_scan': 5000,
                'imu_records': 500
            },
            'large': {
                'frame_count': 100,
                'resolution': (1920, 1080),
                'lidar_scans': 50,
                'points_per_scan': 10000,
                'imu_records': 1000
            }
        }
        self.config = self.sizes[size]

    async def generate_mission(self, mission_id: str) -> Path:
        """Generate a complete mission dataset"""
        mission_path = self.output_path / mission_id
        
        # Ensure we start with a clean slate
        if mission_path.exists():
            import shutil
            shutil.rmtree(mission_path)
        
        # Create directory structure
        (mission_path / 'imagery').mkdir(parents=True, exist_ok=True)
        (mission_path / 'lidar').mkdir(parents=True, exist_ok=True)
        (mission_path / 'imu').mkdir(parents=True, exist_ok=True)

        print("\nGenerating test data...")
        
        # Generate data with progress bars
        with tqdm(total=3, desc="Generating", unit="dataset") as pbar:
            start_time = datetime.now()
            
            # Generate RGB frames
            await self._generate_rgb_data(mission_path / 'imagery', start_time)
            pbar.update(1)
            
            # Generate LiDAR scans
            await self._generate_lidar_data(mission_path / 'lidar', start_time)
            pbar.update(1)
            
            # Generate IMU data
            await self._generate_imu_data(mission_path / 'imu', start_time)
            pbar.update(1)

        # Create metadata
        metadata = {
            'mission_id': mission_id,
            'timestamp': start_time.isoformat(),
            'data_size': self.config,
            'coordinate_system': 'EPSG:4326'
        }
        
        with open(mission_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
            
        return mission_path

    async def _generate_rgb_data(self, path: Path, start_time: datetime):
        """Generate RGB frames with test patterns"""
        frame_interval = timedelta(milliseconds=33)  # ~30fps
        
        for i in range(self.config['frame_count']):
            # Create test image with timestamp and patterns
            image = np.zeros((*self.config['resolution'], 3), dtype=np.uint8)
            
            # Add some dynamic test patterns
            cv2.putText(
                image,
                f"Frame {i} - {start_time + frame_interval * i}",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )
            
            # Add moving objects
            center_x = int(self.config['resolution'][0] * (0.5 + 0.3 * np.sin(i * 0.1)))
            center_y = int(self.config['resolution'][1] * (0.5 + np.cos(i * 0.1)))
            
            cv2.circle(image, (center_x, center_y), 50, (0, 0, 255), -1)
            cv2.rectangle(
                image,
                (100, 100),
                (300, 300),
                (0, 255, 0),
                3
            )
            
            cv2.imwrite(str(path / f'frame_{i:05d}.tiff'), image)

    async def _generate_lidar_data(self, path: Path, start_time: datetime):
        """Generate LiDAR point cloud data"""
        for i in range(self.config['lidar_scans']):
            # Generate point cloud with randomized 3D points
            points = np.random.uniform(-10, 10, (self.config['points_per_scan'], 3)).astype(np.float32)
            
            # Save as .laz (LiDAR compressed format)
            with open(path / f'scan_{i:05d}.laz', 'wb') as f:
                # Simplified point cloud binary representation
                f.write(points.tobytes())

    async def _generate_imu_data(self, path: Path, start_time: datetime):
        """Generate IMU trajectory data for JSON ingestion"""
        with open(path / 'trajectory.json', 'w') as f:
            records = []
            for i in range(self.config['imu_records']):
                record = {
                    "timestamp": (start_time + timedelta(milliseconds=i*10)).isoformat(),
                    "latitude": 37.7749 + 0.001 * np.sin(i * 0.02),
                    "longitude": -122.4194 + 0.001 * np.cos(i * 0.02),
                    "altitude": 100.0 + 5 * np.sin(i * 0.01),
                    "roll": 10 * np.sin(i * 0.05),
                    "pitch": 10 * np.cos(i * 0.05),
                    "yaw": i / self.config['imu_records'] * 360
                }
                records.append(record)
            
            # Write as newline-delimited JSON
            for record in records:
                json.dump(record, f)
                f.write('\n')

async def test_server_connection(url: str) -> bool:
    """Test server connectivity with comprehensive error handling"""
    print(f"\nTesting connection to {url}")
    
    # Create SSL context with explicit settings
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    try:
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(
                    f'{url}/api/system/status',
                    ssl=ssl_context
                ) as response:
                    print(f"Response status: {response.status}")
                    if response.status == 200:
                        response_text = await response.text()
                        print(f"Response text: {response_text}")
                        
                        try:
                            data = json.loads(response_text)
                            print(f"Server status: {data}")
                            return True
                        except json.JSONDecodeError:
                            print("Could not parse JSON response")
                            return False
                    else:
                        print(f"Unexpected response status: {response.status}")
                        return False
            
            except Exception as conn_error:
                print(f"Connection Error: {conn_error}")
                return False
    
    except Exception as overall_error:
        print(f"Comprehensive Connection Test Failed: {overall_error}")
        return False
    
async def validate_mission_structure(mission_path: Path) -> bool:
    """Validate that the mission directory has the required structure and files"""
    try:
        # Check if directory exists
        if not mission_path.exists():
            print(f"Error: Mission directory {mission_path} does not exist")
            return False
            
        # Check required subdirectories
        required_dirs = ['imagery', 'lidar', 'imu']
        for dir_name in required_dirs:
            if not (mission_path / dir_name).is_dir():
                print(f"Error: Required directory '{dir_name}' not found in {mission_path}")
                return False
        
        # Check for image files
        image_files = list((mission_path / 'imagery').glob('*.tiff'))
        if not image_files:
            print(f"Warning: No .tiff files found in {mission_path}/imagery")
            return False
            
        # Check for LiDAR files
        lidar_files = list((mission_path / 'lidar').glob('*.laz'))
        if not lidar_files:
            print(f"Warning: No .laz files found in {mission_path}/lidar")
            return False
            
        # Check for IMU trajectory file
        if not (mission_path / 'imu' / 'trajectory.json').exists():
            print(f"Error: trajectory.json not found in {mission_path}/imu")
            return False
            
        return True
        
    except Exception as e:
        print(f"Error validating mission structure: {str(e)}")
        return False

async def upload_mission(url: str, mission_path: Path) -> bool:
    """Upload mission data to Edge Cache Server"""
    mission_id = mission_path.name
    print(f"\nPreparing to upload mission data:")
    print(f"Mission ID: {mission_id}")
    print(f"Full URL: {url}")
    
    # Create SSL context with explicit settings
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    try:
        async with aiohttp.ClientSession() as session:
            # Start mission
            print("Starting mission...")
            try:
                async with session.post(
                    f'{url}/api/mission/{mission_id}/start',
                    ssl=ssl_context
                ) as response:
                    print(f"Mission start response status: {response.status}")
                    if response.status not in [200, 202]:
                        print(f"Failed to start mission: {response.status}")
                        try:
                            error_text = await response.text()
                            print(f"Error response: {error_text}")
                        except:
                            pass
                        return False
                    print("Mission started successfully")
            except Exception as start_error:
                print(f"Mission start failed: {str(start_error)}")
                return False
            
            # Upload RGB frames
            print("\nUploading RGB frames...")
            try:
                rgb_files = list((mission_path / 'imagery').glob('*.tiff'))
                for file_path in tqdm(rgb_files, desc="RGB Frames"):
                    data = aiohttp.FormData()
                    data.add_field('file', 
                                   open(file_path, 'rb'), 
                                   filename=file_path.name)
                    
                    async with session.post(
                        f'{url}/api/mission/{mission_id}/rgb', 
                        data=data,
                        ssl=ssl_context
                    ) as response:
                        if response.status != 200:
                            print(f"RGB upload failed for {file_path.name}")
                            return False
            except Exception as rgb_error:
                print(f"RGB upload error: {str(rgb_error)}")
                return False
            
            # Upload LiDAR scans
            print("\nUploading LiDAR scans...")
            try:
                lidar_files = list((mission_path / 'lidar').glob('*.laz'))
                for file_path in tqdm(lidar_files, desc="LiDAR Scans"):
                    data = aiohttp.FormData()
                    data.add_field('file', 
                                   open(file_path, 'rb'), 
                                   filename=file_path.name)
                    
                    async with session.post(
                        f'{url}/api/mission/{mission_id}/lidar', 
                        data=data,
                        ssl=ssl_context
                    ) as response:
                        if response.status != 200:
                            print(f"LiDAR upload failed for {file_path.name}")
                            return False
            except Exception as lidar_error:
                print(f"LiDAR upload error: {str(lidar_error)}")
                return False
            
            # Upload IMU data
            print("\nUploading IMU data...")
            try:
                imu_file = mission_path / 'imu' / 'trajectory.json'
                with open(imu_file, 'r') as f:
                    for line in tqdm(f, desc="IMU Records"):
                        imu_record = json.loads(line.strip())
                        async with session.post(
                            f'{url}/api/mission/{mission_id}/imu', 
                            json=imu_record,
                            ssl=ssl_context
                        ) as response:
                            if response.status != 200:
                                print("IMU upload failed")
                                return False
            except Exception as imu_error:
                print(f"IMU upload error: {str(imu_error)}")
                return False
            
            print("\nAll mission data uploaded successfully")
            return True
    
    except Exception as e:
        print(f"Mission upload failed: {str(e)}")
        return False

async def main():
    parser = argparse.ArgumentParser(description='Upload mission data to Edge Cache Server')
    parser.add_argument('--host', default='192.168.4.35', help='Server hostname/IP')
    parser.add_argument('--port', type=int, default=3000, help='Server port')
    parser.add_argument('--input', type=Path, help='Path to existing mission data')
    parser.add_argument('--generate-test', action='store_true', 
                      help='Generate and use test data instead of real data')
    parser.add_argument('--size', choices=['small', 'medium', 'large'], default='small',
                      help='Size of test dataset (only used with --generate-test)')
    parser.add_argument('--mission-id', type=str,
                      help='Optional mission ID (default: derived from timestamp)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.generate_test and not args.input:
        parser.error("Either --input or --generate-test must be specified")
    
    if args.generate_test and args.input:
        parser.error("Cannot specify both --input and --generate-test")
    
    # Use HTTPS URL
    server_url = f'https://{args.host}:{args.port}'
    
    # Test server connection
    if not await test_server_connection(server_url):
        print("Server connection failed. Exiting.")
        sys.exit(1)
    
    try:
        # Generate mission ID if not provided
        mission_id = args.mission_id or f'mission_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        
        if args.generate_test:
            # Generate test data
            print("\nGenerating test data...")
            generator = TestDataGenerator(Path('/tmp'), args.size)
            mission_path = await generator.generate_mission(mission_id)
            print(f"Generated test data at: {mission_path}")
        else:
            # Use existing mission data
            mission_path = args.input
            print(f"\nUsing existing mission data from: {mission_path}")
            
            # Validate mission directory structure
            if not await validate_mission_structure(mission_path):
                print("Mission directory validation failed. Exiting.")
                sys.exit(1)
        
        # Upload mission to server
        success = await upload_mission(server_url, mission_path)
        
        if not success:
            print("Mission upload failed.")
            sys.exit(1)
        
        # Check mission status
        try:
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f'{server_url}/api/mission/{mission_id}/status',
                    ssl=ssl_context
                ) as response:
                    if response.status == 200:
                        status = await response.json()
                        print("\nMission Status:")
                        print(json.dumps(status, indent=2))
                    else:
                        print("Could not retrieve mission status")
        except Exception as status_error:
            print(f"Error checking mission status: {status_error}")
        
    except Exception as e:
        print(f"\nError during execution: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    asyncio.run(main())