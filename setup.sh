#!/bin/bash

set -e  # エラーで止める

echo "[Step] Updating apt and installing base packages..."
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-pip python3-venv \
                    build-essential cmake \
                    libgl1-mesa-dev libgl1-mesa-glx libglu1-mesa-dev \
                    libglew-dev freeglut3-dev x11-utils \
                    git curl unzip

echo "[Step] Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "[Step] Upgrading pip and installing Python packages..."
pip install --upgrade pip
pip install -r requirements.txt

echo "[Step] Appending DISPLAY settings (WSLg)..."
if ! grep -q "export DISPLAY=:0" ~/.bashrc; then
    echo 'export DISPLAY=:0' >> ~/.bashrc
fi

echo "[Done] Setup complete. Run 'source venv/bin/activate' to start the virtual environment."
