#!/bin/bash

# Set environment path for cron
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

echo "Starting daily sync at $(date)"

# 1. Sync the Google Drive folder to your RunPod Workspace
# Replace 'gdrive:/TFF_Docs' with the actual path in your Google Drive
rclone sync gdrive:/TFF_Docs /workspace/TFF_Docs

# 2. Run your embedding script
python3 /workspace/EmbeddedModel.py

echo "Daily sync complete at $(date)"
