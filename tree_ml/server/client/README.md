# Edge Cache Client

A Python client script for uploading mission data to the Edge Cache Server. The client supports both real mission data uploads and test data generation for validation purposes.

## Prerequisites

- Python 3.7 or higher
- Required Python packages:
  ```bash
  pip install aiohttp numpy opencv-python tqdm
  ```

## Usage

### Uploading Real Mission Data

To upload existing mission data to the server:

```bash
python client.py --host <server_host> --port <server_port> --input /path/to/mission/data
```

Your mission data directory should have the following structure:
```
mission_data/
├── imagery/
│   └── *.tiff files
├── lidar/
│   └── *.laz files
└── imu/
    └── trajectory.json
```

### Generating and Uploading Test Data

For testing or validation purposes, you can generate and upload synthetic test data:

```bash
python client.py --host <server_host> --port <server_port> --generate-test --size <small|medium|large>
```

### Optional Arguments

- `--mission-id`: Specify a custom mission ID (default: timestamp-based)
- `--size`: Choose test data size when using `--generate-test`:
  - `small`: 10 frames, 5 LiDAR scans, 100 IMU records
  - `medium`: 50 frames, 25 LiDAR scans, 500 IMU records
  - `large`: 100 frames, 50 LiDAR scans, 1000 IMU records

### Examples

1. Upload real mission data:
```bash
python client.py --host 192.168.4.35 --port 3000 --input /mnt/missions/mission_001
```

2. Generate and upload small test dataset:
```bash
python client.py --host 192.168.4.35 --port 3000 --generate-test --size small
```

3. Upload with custom mission ID:
```bash
python client.py --host 192.168.4.35 --port 3000 --input /mnt/missions/mission_001 --mission-id custom_mission_123
```