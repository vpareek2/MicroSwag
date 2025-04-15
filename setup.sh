#!/bin/bash

# Update system packages
apt-get update

# Install basic tools
apt-get install git vim nano unzip zip rsync ninja-build tmux -y

# Optional: switch remote to allow push access
git config --global user.email "veerpareek12@gmail.com"
git config --global user.name "veer"

# Configure git for proper line endings
git config --global core.autocrlf false

# Install uv
pip install uv

# Backup: Install uv from script if pip fails
curl -LsSf https://astral.sh/uv/install.sh | sh
sudo snap install astral-uv --classic

# Create and activate virtual environment
uv venv
source .venv/bin/activate

# Install Python dependencies
uv pip install -r requirements.txt

echo "Setup completed and repo ready!"
