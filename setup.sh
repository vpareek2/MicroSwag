#!/bin/bash

# Update system packages
sudo apt-get update

# Install basic tools
sudo apt-get install vim nano git-lfs unzip zip -y

# Initialize git-lfs
git lfs install

# Configure git for proper line endings
git config --global core.autocrlf false

# Install uv
pip install uv

# Pip currently failing
curl -LsSf https://astral.sh/uv/install.sh | sh
sudo snap install astral-uv --classic
# Create and activate virtual environment
uv venv
source .venv/bin/activate

# Install requirements using uv
uv pip install -r requirements.txt

echo "Setup completed!"
