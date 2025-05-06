#!/bin/bash
set -e

# Project path
PROJECT_ROOT="/ttt/tree_risk_pro/server"
SERVER_SCRIPT="$PROJECT_ROOT/server.py"
LOG_DIR="/ttt/system/logs"
LOG_FILE="$LOG_DIR/edge-cache.log"

# SSL certificate paths
SSL_CERT="$PROJECT_ROOT/server.cert"
SSL_KEY="$PROJECT_ROOT/server.key"

# Get original user when running with sudo
if [ "$SUDO_USER" ]; then
    REAL_USER="$SUDO_USER"
else
    REAL_USER="$(whoami)"
fi

echo "Running as user: $(whoami), real user: $REAL_USER"

# Get Poetry's virtual environment path - use the real user's environment
if [ "$SUDO_USER" ]; then
    POETRY_ENV=$(sudo -u $REAL_USER bash -c "cd $PROJECT_ROOT && poetry env info --path")
else
    POETRY_ENV=$(cd $PROJECT_ROOT && poetry env info --path)
fi

if [ -z "$POETRY_ENV" ]; then
    echo "Error: Could not find Poetry virtual environment"
    echo "Please run 'cd $PROJECT_ROOT && poetry install' first"
    exit 1
fi

echo "Using Poetry environment at: $POETRY_ENV"

# Create log directory if it doesn't exist
if [ ! -d "$LOG_DIR" ]; then
    echo "Creating log directory: $LOG_DIR"
    mkdir -p "$LOG_DIR" || { echo "Failed to create log directory. Try running with sudo."; exit 1; }
fi

# Setup directories for server
setup_dirs() {
    echo "Setting up required directories..."
    # Create directories
    mkdir -p /ttt/missions /ttt/input /ttt/.hdf5 /ttt/temp
    
    # Make sure the directories have proper permissions
    chmod 755 /ttt/missions /ttt/input /ttt/.hdf5 /ttt/temp
}

# Setup virtual environment
setup_venv() {
    echo "Setting up environment using Poetry..."
    
    # Install dependencies via Poetry - use real user if sudo
    cd "$PROJECT_ROOT"
    if [ "$SUDO_USER" ]; then
        sudo -u $REAL_USER bash -c "cd $PROJECT_ROOT && poetry install"
    else
        poetry install
    fi
    
    # If there's a local h5serv package, install it
    if [[ -d "$PROJECT_ROOT/h5serv" ]]; then
        echo "Installing local h5serv package..."
        if [ "$SUDO_USER" ]; then
            sudo -u $REAL_USER bash -c "cd $PROJECT_ROOT && poetry run pip install -e $PROJECT_ROOT/h5serv"
        else
            poetry run pip install -e "$PROJECT_ROOT/h5serv"
        fi
    fi
}

# Check SSL certificates
check_ssl() {
    if [[ ! -f "$SSL_CERT" ]] || [[ ! -f "$SSL_KEY" ]]; then
        echo "Error: SSL certificates not found."
        echo "Make sure $SSL_CERT and $SSL_KEY exist."
        exit 1
    fi
}

# Start server
start_server() {
    echo "Starting Edge Cache Server..."
    
    # Check config
    if [[ ! -f "$PROJECT_ROOT/server.cfg" ]]; then
        echo "Warning: server.cfg not found at $PROJECT_ROOT/server.cfg"
    fi
    
    # Start the server using Poetry
    cd "$PROJECT_ROOT"  # Change directory to ensure SSL certs are found
    
    # When running with sudo, we need to run the server as the real user
    if [ "$SUDO_USER" ]; then
        # Start server as the real user but with sudo privileges
        sudo -E -u $REAL_USER bash -c "cd $PROJECT_ROOT && PYTHONPATH=$PROJECT_ROOT $POETRY_ENV/bin/python $SERVER_SCRIPT" 2>&1 | tee -a "$LOG_FILE" &
        echo $! > /tmp/edge-cache-server.pid
        echo "Server started with PID $(cat /tmp/edge-cache-server.pid)"
    else
        # Start directly
        PYTHONPATH="$PROJECT_ROOT" $POETRY_ENV/bin/python "$SERVER_SCRIPT" 2>&1 | tee -a "$LOG_FILE" &
        echo $! > /tmp/edge-cache-server.pid
        echo "Server started with PID $(cat /tmp/edge-cache-server.pid)"
    fi
    
    # Give it a moment to start up
    sleep 2
    
    # Check if it's running
    if ps -p $(cat /tmp/edge-cache-server.pid) > /dev/null; then
        echo "Server successfully started"
    else
        echo "ERROR: Server failed to start. Check logs at $LOG_FILE"
        exit 1
    fi
}

# Stop server
stop_server() {
    PID=$(lsof -t -i:3000)
    if [ -n "$PID" ]; then
        echo "Stopping server (PID: $PID)..."
        kill $PID
        sleep 2
        # Check if process still exists
        if ps -p $PID > /dev/null; then
            echo "Forcing kill..."
            kill -9 $PID
        fi
        echo "Server stopped."
    else
        echo "Server is not running."
    fi
}

# Check server status
status_server() {
    if [ -f /tmp/edge-cache-server.pid ]; then
        PID=$(cat /tmp/edge-cache-server.pid)
        if ps -p $PID > /dev/null; then
            echo "Edge Cache Server is running with PID $PID"
            return 0
        else
            echo "Edge Cache Server is not running (PID file exists but process is gone)"
            return 1
        fi
    elif pgrep -f "$SERVER_SCRIPT" > /dev/null; then
        PID=$(pgrep -f "$SERVER_SCRIPT")
        echo "Edge Cache Server is running with PID $PID (no PID file)"
        return 0
    else
        echo "Edge Cache Server is not running"
        return 1
    fi
}

# Main execution
case "$1" in
    setup)
        setup_dirs
        setup_venv
        check_ssl
        echo "Setup completed"
        ;;
    start)
        setup_dirs
        setup_venv
        check_ssl
        start_server
        ;;
    stop)
        stop_server
        ;;
    restart)
        stop_server
        sleep 2
        setup_dirs
        setup_venv
        start_server
        ;;
    status)
        status_server
        ;;
    *)
        echo "Usage: $0 {setup|start|stop|restart|status}"
        exit 1
        ;;
esac
