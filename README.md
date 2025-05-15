# Franka Emika Pandaロボットシミュレーション / Franka Emika Panda Robot Simulation

WSL（Windows Subsystem for Linux）環境でFranka Emika Pandaロボットのシミュレーションを行うためのプロジェクトです。  
This project enables Franka Emika Panda robot simulation in WSL (Windows Subsystem for Linux) environment.

## 環境セットアップ / Environment Setup

以下のコマンドでセットアップを行います：  
Set up the environment with the following commands:

```bash
# セットアップスクリプトの実行権限を付与 / Grant execution permission to the setup script
chmod +x setup.sh

# セットアップスクリプトを実行 / Run the setup script
./setup.sh

# 仮想環境を有効化（新しいターミナルを開いた場合） / Activate the virtual environment (if opening a new terminal)
source venv/bin/activate
```

## 利用可能なスクリプト / Available Scripts

このプロジェクトには以下のスクリプトが含まれています：  
This project includes the following scripts:

### 1. main_simple.py

最もシンプルなバージョンで、Pandaロボットアームの「Reach」タスク（目標位置に到達する）をランダム方策で実行します。  
The simplest version that executes the "Reach" task (reaching a target position) with a random policy.

```bash
python main_simple.py
```

### 2. panda_tasks.py

様々なPandaロボットのタスク（Reach、Push、Pick and Place、Stack、Flip、Slide）を選択して実行できます。  
Allows you to select and execute various Panda robot tasks (Reach, Push, Pick and Place, Stack, Flip, Slide).

```bash
python panda_tasks.py
```

### 3. dqn_simple.py

Deep Q-Network（DQN）強化学習アルゴリズムを使用して、Pandaロボットの「Reach」タスクを学習させるデモンストレーションです。  
A demonstration that trains the Panda robot on the "Reach" task using the Deep Q-Network (DQN) reinforcement learning algorithm.

```bash
python dqn_simple.py
```

### 4. camera_control.py

様々なカメラアングルからPandaロボットを観察するデモスクリプトです。  
A demonstration script for observing the Panda robot from various camera angles.

```bash
python camera_control.py
```

### 5. main_with_camera.py

カメラ視点を変更できる機能を追加したメインスクリプトです。  
The main script with added functionality to change the camera perspective.

```bash
python main_with_camera.py
```

## トラブルシューティング / Troubleshooting

### X11表示の問題 / X11 Display Issues

WSL環境でGUIを表示するための一般的な設定：  
Common settings for displaying GUI in WSL environment:

```bash
# .bashrcに以下を追加 / Add the following to .bashrc
export DISPLAY=:0
```

または：  
Or:

```bash
# WSL2の場合は以下の設定も試してみてください / For WSL2, try the following settings
export DISPLAY=$(grep -m 1 nameserver /etc/resolv.conf | awk '{print $2}'):0.0
```

### フォントの警告 / Font Warnings

matplotlibで日本語を表示する際に発生するフォントの警告を解決するには：  
To resolve font warnings when displaying Japanese in matplotlib:

```bash
# 日本語フォントをインストール / Install Japanese fonts
sudo apt install fonts-ipafont-gothic fonts-ipafont-mincho

# matplotlibの設定ファイルを作成 / Create matplotlib configuration file
mkdir -p ~/.config/matplotlib
echo "font.family : IPAGothic" > ~/.config/matplotlib/matplotlibrc
```

## 参考情報 / Reference Information

- [panda-gym公式ドキュメント / panda-gym Official Documentation](https://panda-gym.readthedocs.io/)
- [ソースコードリポジトリ / Source Code Repository](https://github.com/qgallouedec/panda-gym)

## ライセンス / License

MITライセンス / MIT License