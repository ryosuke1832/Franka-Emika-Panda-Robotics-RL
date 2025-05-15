#!/bin/bash

set -e  # エラーで止める

echo "[Step] Updating apt and installing base packages..."
sudo apt update
sudo apt install -y python3 python3-pip python3-venv \
                    build-essential cmake \
                    libgl1-mesa-dev libgl1-mesa-glx libglu1-mesa-dev \
                    libglew-dev freeglut3-dev x11-utils \
                    git curl unzip

echo "[Step] Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "[Step] Upgrading pip and installing Python packages..."
pip install --upgrade pip setuptools wheel

# パッケージを個別にインストールして依存関係の問題を回避
echo "[Step] Installing core packages first..."
pip install numpy matplotlib

echo "[Step] Installing PyBullet..."
pip install pybullet

echo "[Step] Installing Gymnasium (replacement for gym)..."
pip install gymnasium==0.28.1

echo "[Step] Installing Panda-Gym..."
pip install panda-gym==3.0.0

echo "[Step] Installing TensorFlow..."
pip install tensorflow

echo "[Step] Appending DISPLAY settings (WSLg)..."
if ! grep -q "export DISPLAY=:0" ~/.bashrc; then
    echo 'export DISPLAY=:0' >> ~/.bashrc
fi

echo "[Done] Setup complete. Run 'source venv/bin/activate' to start the virtual environment."