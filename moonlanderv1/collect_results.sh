#!/bin/bash

# --- Configuration ---
# Identity file path
KEY_PEM="/Users/henriquerio/Documents/IIT/FALL2025-HW/CS595/hmr-test.pem"

# Remote Server Details
REMOTE_USER="cc"
REMOTE_IP="129.114.108.135"
# Assuming the previous upload placed 'moonlander' in the home directory (~/)
REMOTE_SOURCE_BASE="~/moonlander"

# Local Destination Path
# I've constructed this based on your upload path + the requested subfolder
LOCAL_DEST_DIR="/Users/henriquerio/Documents/IIT/CS595-Project-FedRL/moonlander/sv_results/v1"

# List of folders to collect
FOLDERS_TO_COLLECT=("comparison_plots" "plots" "models")

# --- Execution ---

# 1. Create the local directory if it doesn't exist
echo "Checking local directory..."
if [ ! -d "$LOCAL_DEST_DIR" ]; then
    echo "Creating directory: $LOCAL_DEST_DIR"
    mkdir -p "$LOCAL_DEST_DIR"
else
    echo "Directory exists: $LOCAL_DEST_DIR"
fi

# 2. Loop through folders and SCP them down
for FOLDER in "${FOLDERS_TO_COLLECT[@]}"; do
    echo "------------------------------------------------"
    echo "Downloading remote folder: $FOLDER"
    
    # Construct the full remote path
    REMOTE_PATH="${REMOTE_USER}@${REMOTE_IP}:${REMOTE_SOURCE_BASE}/${FOLDER}"
    
    # Run SCP
    # -r: recursive copy
    # -i: use the specific identity file
    scp -r -i "$KEY_PEM" "$REMOTE_PATH" "$LOCAL_DEST_DIR"
    
    if [ $? -eq 0 ]; then
        echo "Successfully downloaded $FOLDER"
    else
        echo "Error downloading $FOLDER. Please check if it exists on the server."
    fi
done

echo "------------------------------------------------"
echo "Collection complete. Files stored in: $LOCAL_DEST_DIR"