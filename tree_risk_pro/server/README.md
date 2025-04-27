# Edge Cache Server
High-performance edge caching system for drone data with h5serv integration

## Overview

The Edge Cache server provides a robust caching layer for drone imagery, IMU, and LiDAR data. It uses h5serv for efficient data storage and retrieval, making it ideal for edge processing of large datasets.

## System Requirements

- Ubuntu Server 24.04 LTS or equivalent
- Minimum 8GB RAM (16GB recommended)
- 1TB NVMe SSD
- Gigabit Ethernet
- Python 3.11+
- Poetry for dependency management

## Installation

1. Install system dependencies:
```bash
sudo apt-get update
sudo apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    build-essential \
    libhdf5-dev
```

2. Install h5serv:
```bash
git clone https://github.com/HDFGroup/h5serv.git
cd h5serv
python3 -m pip install -r requirements.txt
python3 setup.py install
```

3. Clone the Edge Cache repository:
```bash
git clone <repo-url>
cd edge-cache
```

4. Install Poetry:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

5. Install dependencies:
```bash
poetry install
```

6. Create the edge-cache user:
```bash
sudo useradd -r -s /bin/false edge-cache
```

7. Create required directories:
```bash
sudo mkdir -p /data/{missions,system/logs}
sudo mkdir -p /var/run/edge-cache
sudo mkdir -p /var/log/edge-cache
sudo mkdir -p /opt/edge-cache
sudo chown -R edge-cache:edge-cache /data /var/run/edge-cache /var/log/edge-cache /opt/edge-cache
```

## Configuration

1. Generate SSL certificates:
```bash
cd /opt/edge-cache
sudo openssl req -x509 -newkey rsa:4096 -keyout server.key -out server.cert -days 365 -nodes
sudo chown edge-cache:edge-cache server.key server.cert
```

2. Configure h5serv:
```bash
sudo cp h5serv/config/default.cfg /opt/edge-cache/server.cfg
```

Edit `/opt/edge-cache/server.cfg`:
```ini
[server]
port = 3000
debug = True  # Set to True for detailed logging during testing
datapath = /data/
endpoint = https://192.168.4.35:3000  # https since the server uses SSL

[cors]
enabled = true  # Enable for testing with the preprocessing server
allow_credentials = true
allowed_origins = ["https://192.168.4.35:3000"]

[hdf5]
datapath = /data/hdf5  # Directory for HDF5 storage
dirpath = /data/hdf5/.hdf5  # Hidden directory for HDF5 metadata
disable_h5py = false
compression_enabled = true
compression_threshold = 1000000
chunking_enabled = true
min_chunk_size = 1048576 # 1MB chunks
max_size_gb = 2048  # 2TB limit for testing

[logging]
log_file = /data/edge/server.log  # Match server's log location
log_level = DEBUG  # Match server's debug level

[cache]
max_chunk_size = 1048576  # 1MB chunks
max_chunk_count = 2000    # Increased for better performance
scan_interval = 5         # More frequent scans during testing
cache_dir = /data/temp  # Using temp directory for cache

[startup]
init_domains = true
auto_scan = true
wait_for_process = true

[storage]
input_path = /data/input    # Match server paths
output_path = /data/output
zarr_store_path = /data/zarr
temp_path = /data/temp

[security]
domain_required = false
allow_cross_origin = true  # Enable for testing
ssl_cert = server.cert     # Match server's SSL config
ssl_key = server.key
```

## Running the Server

### Using the startup script:

1. Start the server:
```bash
sudo ./edge-cache-server.sh start
```

2. Check server status:
```bash
sudo ./edge-cache-server.sh status
```

3. View logs:
```bash
sudo ./edge-cache-server.sh logs
```

4. Stop the server:
```bash
sudo ./edge-cache-server.sh stop
```

### Using systemd:

1. Install the service:
```bash
sudo cp edge-cache.service /etc/systemd/system/
sudo systemctl daemon-reload
```

2. Start the service:
```bash
sudo systemctl start edge-cache
```

3. Enable on boot:
```bash
sudo systemctl enable edge-cache
```

4. Monitor logs:
```bash
sudo journalctl -u edge-cache -f
```

## API Endpoints

### Mission Management
- `POST /api/mission/{mission_id}/start` - Start a new mission
- `POST /api/mission/{mission_id}/rgb` - Upload RGB frame data
- `POST /api/mission/{mission_id}/imu` - Upload IMU data
- `POST /api/mission/{mission_id}/lidar` - Upload LiDAR scan data
- `GET /api/mission/{mission_id}/status` - Get mission status
- `GET /api/system/status` - Get system status

### h5serv Integration
- `GET /` - h5serv root endpoint (port 3000)
- `GET /{domain}` - List all datasets in a domain
- `GET /{domain}/{dataset}` - Get dataset metadata
- `GET /{domain}/{dataset}/value` - Get dataset values

## Example Usage

### Starting a new mission:
```bash
curl -X POST -k https://https://192.168.4.35:3000/api/mission/mission001/start
```

### Uploading RGB data:
```bash
curl -X POST -k https://https://192.168.4.35:3000/api/mission/mission001/rgb \
  -F "file=@frame.tiff"
```

### Checking mission status:
```bash
curl -k https://https://192.168.4.35:3000/api/mission/mission001/status
```

### Accessing data via h5serv:
```bash
curl http://https://192.168.4.35:3000/missions/mission001/rgb/frames
```

## Directory Structure

```
/data/
├── missions/
│   └── YYYYMMDD_HHMMSS/
│       ├── imagery/
│       ├── lidar/
│       ├── imu/
│       └── metadata.json
└── system/
    └── logs/

/opt/edge-cache/
├── venv/
├── server.key
├── server.cert
└── h5serv.cfg
```

## Monitoring

The server provides extensive monitoring through:
- System status endpoint
- Detailed logging
- Systemd journal integration
- h5serv statistics

## Troubleshooting

1. Check server logs:
```bash
sudo ./edge-cache-server.sh logs
```

2. Verify h5serv is running:
```bash
curl http://localhost:5000/about
```

3. Check system resources:
```bash
htop  # CPU and memory usage
df -h # Disk space
```

4. Common issues:
   - Permission denied: Check edge-cache user ownership
   - Port in use: Check for conflicting services
   - SSL errors: Verify certificate permissions
   - Memory errors: Check available RAM
