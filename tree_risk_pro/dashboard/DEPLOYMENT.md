# Tree Risk Pro Dashboard - Deployment Guide (Beta v0.3)

This guide provides instructions for deploying the Tree Risk Pro Dashboard on our Google Cloud Platform (GCP) instance.

## Deployment Summary

Our production server is deployed at: **https://34.125.1.171/**

**Authentication:**
- Custom username and password required
- No default credentials - must be set during deployment
- Environment variables: DASHBOARD_USERNAME and DASHBOARD_PASSWORD

**Note**: Authentication credentials are never hardcoded and must be provided during deployment either through environment variables or interactive prompts.

## Beta v0.3 Release Highlights

Beta v0.3 includes these key improvements:
- Removed hardcoded default credentials for enhanced security
- Interactive credential prompts during deployment
- Fixed CORS issues in production environment
- Improved authentication error handling
- Fixed directory permission issues
- Frontend properly handles empty API URL in production

## Beta v0.2 Release Highlights

Beta v0.2 included these improvements:
- UI refinements in header and sidebar navigation
- Renamed "Save to Database" to "Save for Review" for clarity
- Fixed 3D map state preservation between views
- Added visual separator lines in sidebars
- Added thorough code documentation
- Improved backend security
- Streamlined GCP deployment process

## Quick Start for Development

1. **Clone the repository and navigate to the dashboard directory**
   ```bash
   cd dashboard
   ```

2. **Frontend setup:**
   ```bash
   npm install
   cp .env.example .env
   # Edit .env to add Google Maps API key and Map ID
   npm run dev
   ```

3. **Backend setup (choose one):**
   
   **Option 1: Python Flask backend (full features)**
   ```bash
   cd backend
   poetry install
   poetry run python app.py
   ```
   
   **Option 2: Node.js Express backend (mock data)**
   ```bash
   cd backend
   npm install
   npm run dev
   ```

4. **Access the dashboard** at http://localhost:5173/

## Deployment to GCP - Beta v0.2 Guide

1. **Create a GCP VM instance**
   - Ubuntu 22.04 LTS
   - e2-medium (2 vCPU, 4 GB memory) minimum, e2-standard-2 (2 vCPU, 8 GB memory) recommended
   - Allow HTTP/HTTPS traffic
   - Add 50GB standard persistent disk

2. **Install required software**
   ```bash
   sudo apt update && sudo apt upgrade -y
   sudo apt install -y git python3 python3-pip python3-venv nginx build-essential certbot python3-certbot-nginx
   curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
   sudo apt install -y nodejs
   ```

3. **Clone the repository**
   ```bash
   git clone https://github.com/your-org/tree-risk-pro.git
   cd tree-risk-pro
   ```

4. **Set up backend**
   ```bash
   cd backend
   python3 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install poetry
   poetry install --no-root
   
   # Create necessary directories
   mkdir -p logs data/temp data/zarr
   chmod 755 logs
   cd ..
   ```

5. **Set up environment variables**
   ```bash
   # Backend environment configuration
   cat > backend/.env << EOF
   APP_MODE=production
   SKIP_AUTH=false
   # Note: DASHBOARD_USERNAME and DASHBOARD_PASSWORD must be set in the 
   # environment or provided during deployment. They are no longer hardcoded.
   GEMINI_API_KEY=your_gemini_api_key
   GEMINI_MODEL=gemini-2.0-flash
   EOF

   # Frontend environment configuration
   # IMPORTANT: Use empty VITE_API_URL for production to use relative paths
   cat > .env << EOF
   VITE_API_URL=
   VITE_GOOGLE_MAPS_API_KEY=your_google_maps_api_key
   VITE_GOOGLE_MAPS_MAP_ID=your_map_id
   EOF
   ```

6. **Build frontend**
   ```bash
   npm install
   npm run build
   ```

7. **Deploy with our automated script**
   
   Run our deployment script to handle Nginx configuration, SSL setup, and service creation:
   
   ```bash
   # Optionally set environment variables (or the script will prompt for them)
   # export DASHBOARD_USERNAME=your_custom_username
   # export DASHBOARD_PASSWORD=your_secure_password
   export GOOGLE_MAPS_API_KEY=your_google_maps_api_key
   
   # Run the deployment script (it will prompt for credentials if not set)
   ./gcp-deploy.sh
   ```
   
   This script:
   - Configures Nginx with HTTPS
   - Sets up the backend service
   - Installs all files to their locations
   - Sets up firewall rules
   - Starts all required services

8. **Alternative: Manual deployment**
   
   If you prefer to deploy manually instead of using the script:
   
   ```bash
   # Set up Nginx
   sudo cp deployment/nginx.conf /etc/nginx/sites-available/dashboard.conf
   sudo ln -sf /etc/nginx/sites-available/dashboard.conf /etc/nginx/sites-enabled/
   sudo rm -f /etc/nginx/sites-enabled/default
   sudo nginx -t
   
   # Deploy application files
   sudo mkdir -p /opt/dashboard/{backend,dist}
   sudo cp -r backend/* /opt/dashboard/backend/
   sudo cp -r dist/* /opt/dashboard/dist/
   
   # Create and start systemd service
   sudo cp deployment/dashboard-backend.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable dashboard-backend
   sudo systemctl start dashboard-backend
   sudo systemctl restart nginx
   ```

9. **Set up SSL with Let's Encrypt** (if you have a domain)
   ```bash
   sudo certbot --nginx -d your-domain.com
   ```

## Troubleshooting Google Maps Issues

If you see the error: "The map is initialized without a valid Map ID, which will prevent use of Advanced Markers":

1. **Get a valid Map ID from Google Cloud Console**:
   - Go to the [Google Cloud Console](https://console.cloud.google.com/)
   - Navigate to Google Maps Platform > Map Management
   - Create a new Map ID or use an existing one

2. **Update your environment configuration**:
   ```
   # Add to .env file
   VITE_GOOGLE_MAPS_API_KEY=your_google_maps_api_key
   VITE_GOOGLE_MAPS_MAP_ID=your_map_id
   ```

3. **Restart your development server** to apply the changes

## Service Management for Beta v0.2

### Authentication Management

To update authentication credentials on a running system:

```bash
# Set the environment variables in the systemd service file
sudo systemctl edit dashboard-backend

# Add the following lines
[Service]
Environment="DASHBOARD_USERNAME=your_new_username"
Environment="DASHBOARD_PASSWORD=your_new_password"

# Save and exit the editor (Ctrl+X, Y in nano)

# Restart the backend service
sudo systemctl restart dashboard-backend

# Check logs to confirm credentials are being loaded
tail -f /opt/dashboard/backend/logs/app.log
```

You should see log messages indicating credentials are being loaded from environment variables.

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
   - No default credentials - authentication credentials must be provided
   - DASHBOARD_USERNAME and DASHBOARD_PASSWORD environment variables are required
   - Use a strong password (12+ characters with mixed case, numbers, symbols)
   - Password is stored as a hash, not in plaintext
   - Deployment script will interactively prompt for credentials if not set

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