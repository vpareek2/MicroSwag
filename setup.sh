#!/bin/bash

# Update system packages
sudo apt-get update

# Install basic tools
sudo apt-get install git vim nano unzip zip -y

# Clone the GitHub repo (public read access)
git clone https://github.com/vpareek2/MicroSwag.git
cd MicroSwag

# Optional: switch remote to allow push access

git remote set-url origin https://<vpareek2>:<>@github.com/vpareek2/MicroSwag.git

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
