# Tree Risk Pro - Deployment Guide (v0.2.1)

> This is the deployment guide for the entire Tree Risk Pro project, including both the dashboard and backend components.

This guide provides instructions for deploying the Tree Risk Pro system on our Google Cloud Platform (GCP) instance.

## Deployment Summary

Our production server is deployed at: **https://34.125.120.78/**

**Current access credentials:**
- Username: `TestAdmin`
- Password: `trp345!`

## v0.2.1 Release Deployment Notes

v0.2.1 includes these key improvements requiring deployment attention:
- S2 geospatial indexing with Zarr store integration
- New API endpoints for S2 cell-based report management
- ML overlay with persistent opacity settings
- Enhanced DOM handling requiring proper browser caching settings
- Fixed Components/Detection sidebar functionality
- Performance improvements for transitions between modes
- Improved backend security
- Streamlined GCP deployment process

## Prerequisites

- Ubuntu 22.04 LTS VM with at least 2 vCPU, 4 GB memory
- Python 3.12+ and pip
- Node.js 18+
- Git
- Nginx

## Required API Keys

**IMPORTANT: You must obtain these API keys before deployment:**

1. **Google Maps API Key**: 
   - Go to https://console.cloud.google.com/
   - Navigate to Google Maps Platform > Credentials
   - Create an API key with Maps JavaScript API access

2. **Google Maps Map ID**:
   - Go to Google Maps Platform > Map Management
   - Create a new Map ID (or use existing one: e2f2ee8160dbcea0)

3. **Gemini API Key**:
   - Go to https://aistudio.google.com/app/apikey
   - Create a new API key for Gemini model access
   - Ensure your account has access to gemini-2.0-flash model

## Deployment Process

### 1. Kill Existing Processes

Before deployment, ensure all existing processes are properly terminated:

```bash
# Stop the systemd service if it exists
sudo systemctl stop dashboard-backend

# Kill any Python processes related to the application
sudo pkill -f "python.*app.py"
sudo pkill -f "gunicorn.*app:app"

# Kill any Node.js processes that might be running the server
sudo pkill -f "node.*server.js" 

# Kill any processes running on the server ports
sudo fuser -k 5000/tcp  # For the Flask backend
sudo fuser -k 5173/tcp  # For the Vite dev server

# Verify all processes are stopped
ps aux | grep -E "(python.*app|gunicorn|node.*server)"
```

### 2. Clone the Repository

```bash
# Clone the repository
git clone https://github.com/your-org/tree-risk-pro.git ~/tree-risk-pro
cd ~/tree-risk-pro
```

### 3. Configure Environment

Create environment files with your API keys:

```bash
# Frontend environment (.env)
cat > .env << EOF
# Important: Empty string is intentional to avoid path duplication
VITE_API_URL=
VITE_GOOGLE_MAPS_API_KEY=your_google_maps_api_key
VITE_GOOGLE_MAPS_MAP_ID=your_map_id
EOF

# Backend environment (backend/.env)
cat > tree_risk_pro/dashboard/backend/.env << EOF
APP_MODE=production
SKIP_AUTH=false
DASHBOARD_USERNAME=TestAdmin
DASHBOARD_PASSWORD=trp345!
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-2.0-flash
EOF
```

**IMPORTANT**: Replace the placeholder values with your actual API keys before proceeding:
- `your_google_maps_api_key` with your Google Maps API key
- `your_map_id` with your Google Maps Map ID
- `your_gemini_api_key` with your Gemini API key

Alternatively, you can use our setup script with production flag:

```bash
./scripts/setup.sh -d -p your-server-ip
# Then edit both .env files to update API keys
```

### 4. Build Frontend

```bash
# Install dependencies and build
cd tree_risk_pro/dashboard
npm install
npm run build
```

### 5. Setup Deployment Directory

```bash
# Clear previous deployment to prevent any cached or old files
sudo rm -rf /opt/dashboard/dist/*
sudo rm -rf /opt/dashboard/backend/*
sudo rm -f /opt/dashboard/backend/.env

# Create directory structure
sudo mkdir -p /opt/dashboard/{backend,dist}
sudo mkdir -p /opt/dashboard/backend/{logs,data/temp,data/zarr,data/reports,data/exports}

# Copy files
sudo cp -r dist/* /opt/dashboard/dist/
sudo cp -r backend/* /opt/dashboard/backend/
sudo cp ../../pyproject.toml /opt/dashboard/backend/
sudo cp .env /opt/dashboard/
sudo cp backend/.env /opt/dashboard/backend/

# Set permissions
sudo chmod -R 755 /opt/dashboard/backend/logs
sudo chmod -R 755 /opt/dashboard/backend/data
```

**IMPORTANT**: The step to clear previous deployment files is critical to ensure no cached or outdated files remain from previous versions.

### 6. Install Backend Dependencies

```bash
cd /opt/dashboard/backend
sudo pip install poetry
sudo poetry install

# Extend config.py to include environment variables for Gemini
sudo tee -a /opt/dashboard/backend/config.py > /dev/null << EOF

# Gemini Configuration
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')
GEMINI_MODEL = os.environ.get('GEMINI_MODEL', 'gemini-2.0-flash')
EOF
```

### 7. Configure Nginx

```bash
# Generate self-signed certificate if needed
sudo mkdir -p /etc/ssl/private
sudo openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout /etc/ssl/private/dashboard-selfsigned.key \
    -out /etc/ssl/certs/dashboard-selfsigned.crt \
    -subj "/C=US/ST=State/L=City/O=Organization/CN=your-server-ip"

# Create Nginx configuration
sudo tee /etc/nginx/sites-available/dashboard.conf > /dev/null << EOF
server {
    listen 80;
    server_name _;
    
    # Redirect all HTTP traffic to HTTPS
    return 301 https://\$host\$request_uri;
}

server {
    listen 443 ssl;
    server_name _;
    
    # SSL configuration
    ssl_certificate /etc/ssl/certs/dashboard-selfsigned.crt;
    ssl_certificate_key /etc/ssl/private/dashboard-selfsigned.key;
    
    # Frontend static files
    location / {
        root /opt/dashboard/dist;
        try_files \$uri \$uri/ /index.html;

        # Add cache control for static assets
        location /assets {
            expires 7d;
            add_header Cache-Control "public, max-age=604800";
        }
    }

    # Backend API
    location /api {
        proxy_pass http://localhost:5000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_read_timeout 90;
    }
}
EOF

# Replace "your-server-ip" with your actual server IP in the Nginx config
sudo sed -i "s/your-server-ip/$(curl -s ifconfig.me)/g" /etc/nginx/sites-available/dashboard.conf

sudo ln -sf /etc/nginx/sites-available/dashboard.conf /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl restart nginx
```

### 8. Create Systemd Service

```bash
# First find the exact path to gunicorn in the Poetry environment
cd /opt/dashboard/backend
GUNICORN_PATH=$(sudo poetry env info --path)
echo "Poetry virtualenv path: $GUNICORN_PATH"

# Create the systemd service file with the FULL PATH to gunicorn
sudo tee /etc/systemd/system/dashboard-backend.service > /dev/null << EOF
[Unit]
Description=Tree Risk Pro Dashboard Backend
After=network.target

[Service]
User=root
WorkingDirectory=/opt/dashboard/backend
ExecStart=$GUNICORN_PATH/bin/gunicorn -w 4 -b 127.0.0.1:5000 app:app
Environment="DASHBOARD_USERNAME=TestAdmin"
Environment="DASHBOARD_PASSWORD=trp345!"
Environment="APP_MODE=production" 
Environment="GEMINI_API_KEY=$(grep GEMINI_API_KEY /opt/dashboard/backend/.env | cut -d= -f2)"
Environment="GEMINI_MODEL=$(grep GEMINI_MODEL /opt/dashboard/backend/.env | cut -d= -f2)"
Restart=always

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable dashboard-backend
sudo systemctl restart dashboard-backend
```

### 9. Verify Deployment

```bash
# Check backend status
sudo systemctl status dashboard-backend

# Verify API is running
curl -u TestAdmin:trp345! http://localhost:5000/api/config

# Verify the HTTPS site is working (ignore self-signed certificate warning)
curl https://$(curl -s ifconfig.me)/api/config -k
```

After completing these steps, your Tree Risk Pro Dashboard should be accessible via HTTPS at your server's IP address.

## Troubleshooting

### Complete Rebuild and Cache Clear

If you're experiencing version display issues or cached content problems:

1. **Stop all services and kill any related processes**:
   ```bash
   # Stop the systemd service if it exists
   sudo systemctl stop dashboard-backend
   
   # Kill any Python processes related to the application
   sudo pkill -f "python.*app.py"
   sudo pkill -f "gunicorn.*app:app"
   
   # Kill any Node.js processes that might be running the server
   sudo pkill -f "node.*server.js"
   
   # Kill any processes running on the server ports
   sudo fuser -k 5000/tcp  # For the Flask backend
   sudo fuser -k 5173/tcp  # For the Vite dev server
   
   # Verify all processes are stopped
   ps aux | grep -E "(python.*app|gunicorn|node.*server)"
   ```

2. **Create a fresh deployment directory**:
   ```bash
   sudo rm -rf /opt/dashboard
   sudo mkdir -p /opt/dashboard
   ```

3. **Get the latest code**:
   ```bash
   cd ~/tree-risk-pro
   git fetch
   git reset --hard origin/main
   ```

4. **Clean and rebuild the frontend**:
   ```bash
   cd tree_risk_pro/dashboard
   rm -rf node_modules dist
   npm cache clean --force
   npm install
   npm run build
   ```

5. **Follow the deployment steps** from section 5 onwards.

6. **Clear all browser caches completely** before testing.

### CORS or API Connection Issues

If you see CORS errors or API connection problems:

1. **Check your API_BASE_URL in the frontend**:
   
   The key issue with this deployment is ensuring the frontend doesn't try to connect directly to `http://localhost:5000`. 
   
   In the source code:
   ```javascript
   // src/services/api/apiService.js
   const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';
   ```

   This is why we set `VITE_API_URL=` (empty string) in the .env file. When built with an empty string, API calls become relative paths (e.g., `/api/config`) that work correctly with the Nginx proxy.

2. **Verify built JavaScript doesn't contain localhost references**:
   ```bash
   cd /opt/dashboard/dist
   grep -r "localhost:5000" --include="*.js" .
   ```

3. **Check browser console for the actual URL being used**:
   - If you see `API_BASE_URL: http://localhost:5000`, the frontend was built with the wrong environment settings
   - If you see `API_BASE_URL: /api/api`, there's a path duplication issue
   
4. **Clear browser cache thoroughly**:
   - Use Ctrl+Shift+R or Cmd+Shift+R for a hard refresh
   - In Chrome DevTools (F12), go to Application > Clear Storage > Clear site data
   - Try an incognito/private window to test without cache

5. **Check Nginx logs for proxy errors**:
   ```bash
   sudo tail -f /var/log/nginx/error.log
   ```

### Backend Service Issues

If the backend service fails to start:

1. **Poetry path issues**:
   - Find the correct path to the virtualenv: `sudo poetry env info --path`
   - Use the exact path to gunicorn in the systemd service file

2. **Missing Gemini API configuration**:
   - Confirm config.py has been updated with Gemini environment variables
   - Check backend logs: `sudo journalctl -u dashboard-backend -e`

3. **Run backend manually to debug**:
   ```bash
   cd /opt/dashboard/backend
   sudo poetry run python app.py
   ```

### Missing Google Maps Components

If you see the error: "The map is initialized without a valid Map ID, which will prevent use of Advanced Markers":

1. **Get a valid Map ID from Google Cloud Console**:
   - Go to the [Google Cloud Console](https://console.cloud.google.com/)
   - Navigate to Google Maps Platform > Map Management
   - Create a new Map ID or use an existing one

2. **Make sure .env contains the correct Map ID**:
   ```
   VITE_GOOGLE_MAPS_MAP_ID=your_map_id
   ```

3. **Rebuild the frontend** after updating the environment variables

## Service Management for v0.2.1

### Monitoring Logs

```bash
# Backend Flask logs
sudo journalctl -u dashboard-backend.service -f
tail -f /opt/dashboard/backend/logs/app.log

# Tree detection logs (job-specific)
ls -la /opt/dashboard/backend/logs/tree_detection_*
cat /opt/dashboard/backend/logs/tree_detection_<timestamp>.log

# Nginx logs
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log
```

### Managing Services

```bash
# Backend service management
sudo systemctl status dashboard-backend
sudo systemctl restart dashboard-backend
sudo systemctl stop dashboard-backend
sudo systemctl start dashboard-backend

# Nginx service management
sudo systemctl status nginx
sudo systemctl restart nginx

# Check current CPU and memory usage
top -u $(whoami)

# Monitor disk space
df -h /opt/dashboard
```

### Backup and Restore

```bash
# Backup the entire application
sudo tar -czf /tmp/dashboard-backup-$(date +%Y%m%d).tar.gz /opt/dashboard

# Backup just the data directories
sudo tar -czf /tmp/dashboard-data-$(date +%Y%m%d).tar.gz /opt/dashboard/backend/data

# Restore from backup
sudo tar -xzf /tmp/dashboard-backup-20250426.tar.gz -C /
sudo systemctl restart dashboard-backend
```

## Security Checklist

1. **Authentication**:
   - Update the default credentials after deployment
   - Use a strong password (12+ characters with mixed case, numbers, symbols)
   - Current auth implementation is suitable for beta testing

2. **SSL/TLS**:
   - Our deployment script sets up a self-signed certificate by default
   - Current setup on 34.125.120.78 uses this self-signed certificate
   - For domain-based deployment, use Let's Encrypt:
     ```bash
     sudo systemctl status certbot.timer  # Check auto-renewal status
     ```

3. **Firewall**:
   - Our server has these ports open:
     ```bash
     sudo ufw status  # Should show only 22, 80, 443
     ```

4. **Updates**:
   - Update system packages monthly:
     ```bash
     sudo apt update && sudo apt upgrade -y
     ```
   - Check for outdated dependencies:
     ```bash
     cd /opt/dashboard && npm outdated
     cd /opt/dashboard/backend && source venv/bin/activate && pip list --outdated
     ```

5. **API Security**:
   - Keep API keys secure
   - Our Gemini API key is restricted to the production VM
   - Do not commit credentials to git

6. **Gemini API Usage**:
   - Monitor usage at https://console.cloud.google.com
   - Current quota: 60 requests/minute
   - Current billing: Pay-as-you-go with monthly budget alert