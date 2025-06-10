#!/bin/bash
# Script to fix the GroundingDINO CUDA extension path issue

echo "Fixing GroundingDINO CUDA extension path issue..."

# Define paths
SITE_PACKAGES_DIR="/ttt/.venv/lib/python3.12/site-packages"
GROUNDINGDINO_DIR="${SITE_PACKAGES_DIR}/groundingdino"
BUILD_DIR="/ttt/build/lib.linux-x86_64-cpython-312/groundingdino"
EXTENSION_FILE="_C.cpython-312-x86_64-linux-gnu.so"

# Create the groundingdino directory in site-packages if it doesn't exist
if [ ! -d "${GROUNDINGDINO_DIR}" ]; then
    echo "Creating groundingdino directory at ${GROUNDINGDINO_DIR}"
    mkdir -p "${GROUNDINGDINO_DIR}"
    chmod 755 "${GROUNDINGDINO_DIR}"
    
    # Create an empty __init__.py file in the directory
    touch "${GROUNDINGDINO_DIR}/__init__.py"
    chmod 644 "${GROUNDINGDINO_DIR}/__init__.py"
    
    echo "Successfully created ${GROUNDINGDINO_DIR}"
else
    echo "Directory ${GROUNDINGDINO_DIR} already exists"
fi

# Copy the extension file if it exists in the build directory
if [ -f "${BUILD_DIR}/${EXTENSION_FILE}" ]; then
    echo "Copying extension file from ${BUILD_DIR}/${EXTENSION_FILE} to ${GROUNDINGDINO_DIR}/${EXTENSION_FILE}"
    cp "${BUILD_DIR}/${EXTENSION_FILE}" "${GROUNDINGDINO_DIR}/${EXTENSION_FILE}"
    chmod 755 "${GROUNDINGDINO_DIR}/${EXTENSION_FILE}"
    echo "Successfully copied extension file"
else
    echo "Extension file not found at ${BUILD_DIR}/${EXTENSION_FILE}"
    # Try to find the extension file
    find /ttt -name "${EXTENSION_FILE}" -type f | while read file; do
        echo "Found extension file at: $file"
        cp "$file" "${GROUNDINGDINO_DIR}/${EXTENSION_FILE}"
        chmod 755 "${GROUNDINGDINO_DIR}/${EXTENSION_FILE}"
        echo "Copied extension file from $file"
        break
    done
fi

echo "Checking if fix was successful..."
if [ -f "${GROUNDINGDINO_DIR}/${EXTENSION_FILE}" ]; then
    echo "SUCCESS: Extension file exists at ${GROUNDINGDINO_DIR}/${EXTENSION_FILE}"
else
    echo "ERROR: Extension file still missing at ${GROUNDINGDINO_DIR}/${EXTENSION_FILE}"
fi