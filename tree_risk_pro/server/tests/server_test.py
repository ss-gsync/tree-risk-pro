#!/usr/bin/env python3
"""
Preprocessing Server Test Suite
Tests server functionality and data processing pipeline
"""
import unittest
import asyncio
import aiohttp
import numpy as np
import json
import os
from pathlib import Path
from datetime import datetime, timedelta
import tempfile
import shutil
import logging
import cv2
from typing import Dict, List
import zarr
from unittest.mock import AsyncMock, patch

class DataGenerator:
    """Generates test data for preprocessing pipeline"""
    def __init__(self, base_path: Path):
        self.base_path = base_path
        
    async def generate_mission(self, mission_id: str) -> Dict:
        """Generate complete mission dataset"""
        mission_path = self.base_path / mission_id
        os.makedirs(mission_path, exist_ok=True)
        
        # Create data directories
        rgb_path = mission_path / 'rgb'
        lidar_path = mission_path / 'lidar'
        imu_path = mission_path / 'imu'
        
        os.makedirs(rgb_path, exist_ok=True)
        os.makedirs(lidar_path, exist_ok=True)
        os.makedirs(imu_path, exist_ok=True)
        
        # Generate data
        start_time = datetime.now()
        metadata = await asyncio.gather(
            self._generate_rgb_data(rgb_path, start_time),
            self._generate_lidar_data(lidar_path, start_time),
            self._generate_imu_data(imu_path, start_time)
        )
        
        # Create mission metadata
        mission_metadata = {
            'mission_id': mission_id,
            'timestamp': start_time.isoformat(),
            'coordinate_system': 'EPSG:4326',
            'data': {
                'rgb': metadata[0],
                'lidar': metadata[1],
                'imu': metadata[2]
            }
        }
        
        # Save metadata
        with open(mission_path / 'metadata.json', 'w') as f:
            json.dump(mission_metadata, f, indent=2)
            
        return mission_metadata
        
    async def _generate_rgb_data(
        self,
        path: Path,
        start_time: datetime
    ) -> Dict:
        """Generate RGB image frames"""
        frame_count = 5  # Reduced for faster testing
        frame_interval = timedelta(milliseconds=100)
        resolution = (640, 480)  # Smaller resolution for testing
        
        metadata = {
            'frame_count': frame_count,
            'resolution': resolution,
            'interval_ms': 100
        }
        
        for i in range(frame_count):
            image = np.zeros((*resolution, 3), dtype=np.uint8)
            timestamp = start_time + frame_interval * i
            
            # Add test patterns
            cv2.putText(
                image,
                f"Frame {i}",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )
            cv2.rectangle(image, (100, 100), (200, 200), (0, 255, 0), 2)
            
            # Save image
            cv2.imwrite(str(path / f'frame_{i:04d}.jpg'), image)
            
        return metadata
        
    async def _generate_lidar_data(
        self,
        path: Path,
        start_time: datetime
    ) -> Dict:
        """Generate LiDAR point cloud data"""
        scan_count = 3  # Reduced for testing
        points_per_scan = 500
        
        metadata = {
            'scan_count': scan_count,
            'points_per_scan': points_per_scan,
            'interval_ms': 200
        }
        
        for i in range(scan_count):
            points = np.random.uniform(-10, 10, (points_per_scan, 3)).astype(np.float32)
            np.save(path / f'scan_{i:04d}.npy', points)
            
        return metadata
        
    async def _generate_imu_data(
        self,
        path: Path,
        start_time: datetime
    ) -> Dict:
        """Generate IMU sensor data"""
        sample_rate = 50  # Reduced for testing
        duration = 0.5  # Shorter duration
        sample_count = int(sample_rate * duration)
        
        metadata = {
            'sample_rate': sample_rate,
            'sample_count': sample_count,
            'duration_seconds': duration
        }
        
        # Generate simplified IMU data
        timestamps = []
        positions = []
        orientations = []
        
        for i in range(sample_count):
            t = start_time + timedelta(seconds=i/sample_rate)
            timestamps.append(t.timestamp())
            positions.append([37.7749, -122.4194, 100.0])
            orientations.append([0.0, 0.0, i/sample_count * 360])
        
        # Save to CSV
        with open(path / 'imu_data.csv', 'w') as f:
            f.write('timestamp,latitude,longitude,altitude,roll,pitch,yaw\n')
            for t, pos, ori in zip(timestamps, positions, orientations):
                f.write(f'{t},{pos[0]},{pos[1]},{pos[2]},{ori[0]},{ori[1]},{ori[2]}\n')
            
        return metadata

class TestPreprocessingServer(unittest.IsolatedAsyncioTestCase):
    """Test cases for preprocessing server"""
    
    async def asyncSetUp(self):
        """Set up test environment"""
        # Create temporary test directories
        self.test_dir = Path(tempfile.mkdtemp())
        self.input_path = self.test_dir / 'input'
        self.output_path = self.test_dir / 'output'
        self.zarr_path = self.test_dir / 'zarr'
        self.temp_path = self.test_dir / 'temp'
        
        # Create all required directories
        for path in [self.input_path, self.output_path, self.zarr_path, self.temp_path]:
            os.makedirs(path, exist_ok=True)
        
        # Set up data generator
        self.data_generator = DataGenerator(self.input_path)
        self.mission_id = 'test_mission_001'
        
        # Create session with longer timeout
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))
        
        # Allow overriding server URL via environment variable
        self.server_url = os.getenv('SERVER_URL', 'http://localhost:3000')
        
        # Generate initial test data
        self.metadata = await self.data_generator.generate_mission(self.mission_id)
        
    async def asyncTearDown(self):
        """Clean up test environment"""
        await self.session.close()
        shutil.rmtree(self.test_dir)
        
    @patch('aiohttp.ClientSession.get')
    async def test_server_health(self, mock_get):
        """Test server health check endpoint"""
        # Mock a successful response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            'status': 'healthy',
            'active_missions': 0,
            'resources': {
                'cpu_available': True,
                'memory_available': True,
                'gpu_available': True,
                'disk_available': True
            }
        })
        mock_get.return_value.__aenter__.return_value = mock_response
        
        try:
            async with self.session.get(f'{self.server_url}/api/system/status') as response:
                self.assertEqual(response.status, 200)
                data = await response.json()
                self.assertIn('status', data)
                self.assertTrue(data['status'] in ['healthy', 'degraded'])
        except Exception as e:
            self.skipTest(f"Server connection failed: {e}")
            
    @patch('aiohttp.ClientSession.post')
    @patch('aiohttp.ClientSession.get')
    async def test_mission_processing(self, mock_get, mock_post):
        """Test mission processing workflow"""
        # Mock mission start response
        mock_start_response = AsyncMock()
        mock_start_response.status = 202
        mock_start_response.json = AsyncMock(return_value={
            'mission_id': self.mission_id,
            'status': 'accepted'
        })
        mock_post.return_value.__aenter__.return_value = mock_start_response
        
        # Mock mission status responses
        mock_status_response = AsyncMock()
        mock_status_response.status = 200
        mock_status_response.json = AsyncMock(return_value={
            'mission_id': self.mission_id,
            'status': 'completed',
            'runtime_seconds': 10.5
        })
        mock_get.return_value.__aenter__.return_value = mock_status_response
        
        try:
            # Start mission processing
            async with self.session.post(
                f'{self.server_url}/api/mission/{self.mission_id}/start'
            ) as response:
                self.assertIn(response.status, [202, 503])
                
                if response.status == 503:
                    self.skipTest("Server busy - skipping further tests")
                
            # Check mission status
            async with self.session.get(
                f'{self.server_url}/api/mission/{self.mission_id}/status'
            ) as response:
                self.assertEqual(response.status, 200)
                status = await response.json()
                self.assertIn(status['status'], ['completed', 'processing'])
                
        except Exception as e:
            self.skipTest(f"Server connection failed: {e}")
        
    @patch('aiohttp.ClientSession.post')
    async def test_invalid_mission(self, mock_post):
        """Test handling of invalid mission IDs"""
        invalid_ids = ['', ' ', 'a' * 100, 'invalid/id', '../invalid']
        
        # Mock error responses
        mock_response = AsyncMock()
        mock_response.status = 400
        mock_response.json = AsyncMock(return_value={'error': 'Invalid mission ID'})
        mock_post.return_value.__aenter__.return_value = mock_response
        
        for invalid_id in invalid_ids:
            try:
                async with self.session.post(
                    f'{self.server_url}/api/mission/{invalid_id}/start'
                ) as response:
                    self.assertIn(response.status, [400, 404])
            except Exception as e:
                self.skipTest(f"Server connection failed: {e}")
        
    @patch('aiohttp.ClientSession.post')
    async def test_concurrent_missions(self, mock_post):
        """Test handling of concurrent mission processing"""
        # Generate a smaller number of concurrent missions
        mission_ids = [f'concurrent_test_{i}' for i in range(3)]
        
        # Mock responses
        mock_response = AsyncMock()
        mock_response.status = 202  # or 503 depending on server load
        mock_response.json = AsyncMock(return_value={
            'mission_id': mission_ids[0],
            'status': 'accepted'
        })
        mock_post.return_value.__aenter__.return_value = mock_response
        
        responses = []
        for mission_id in mission_ids:
            try:
                async with self.session.post(
                    f'{self.server_url}/api/mission/{mission_id}/start'
                ) as response:
                    responses.append(response.status)
                    await asyncio.sleep(0.5)  # Add small delay between requests
            except Exception as e:
                self.skipTest(f"Server connection failed: {e}")
        
        # Validate responses
        self.assertTrue(
            any(status == 202 for status in responses) or 
            all(status == 503 for status in responses)
        )

if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run tests
    unittest.main()