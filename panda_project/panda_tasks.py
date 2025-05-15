#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Franka Emika Pandaロボットの様々なタスクを実行するサンプルスクリプト
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from panda_gym.envs import (
    PandaReachEnv,
    PandaPushEnv,
    PandaPickAndPlaceEnv,
    PandaStackEnv,
    PandaFlipEnv,
    PandaSlideEnv
)

def run_random_policy(env, episodes=2, max_steps=1000):
    """
    ランダム方策でPandaロボットを制御し、エピソードごとの報酬を記録します。
    
    Args:
        env: gym環境
        episodes: 実行するエピソード数
        max_steps: 各エピソードの最大ステップ数
    
    Returns:
        all_rewards: 各エピソードの累積報酬のリスト
    """
    all_rewards = []
    
    for episode in range(episodes):
        print(f"エピソード {episode+1}/{episodes}")
        observation, info = env.reset()
        cumulative_reward = 0
        
        for step in range(max_steps):
            # ランダムなアクションを実行
            action = env.action_space.sample()
            
            # 環境を1ステップ進める
            observation, reward, terminated, truncated, info = env.step(action)
            cumulative_reward += reward
            
            # 環境をレンダリング
            env.render()
            
            # わずかに待機して可視化を見やすくする
            time.sleep(0.01)
            
            done = terminated or truncated
            if done:
                print(f"  ステップ {step+1}: 完了")
                break
        
        print(f"  累積報酬: {cumulative_reward:.4f}")
        all_rewards.append(cumulative_reward)
    
    return all_rewards

def main():
    """
    メイン実行関数
    """
    # 利用可能なタスクのリスト
    tasks = {
        "1": ("Reach", PandaReachEnv),
        "2": ("Push", PandaPushEnv),
        "3": ("Pick and Place", PandaPickAndPlaceEnv),
        "4": ("Stack", PandaStackEnv),
        "5": ("Flip", PandaFlipEnv),
        "6": ("Slide", PandaSlideEnv)
    }
    
    # タスク選択
    print("実行するタスクを選択してください:")
    for key, (name, _) in tasks.items():
        print(f"  {key}. {name}")
    
    choice = input("選択 (1-6): ").strip()
    
    if choice not in tasks:
        print("有効な選択ではありません。デフォルトでReachタスクを実行します。")
        choice = "1"
    
    task_name, task_class = tasks[choice]
    
    # 環境の作成
    print(f"Panda-Gym {task_name}環境を初期化中...")
    env = task_class(render=True)
    
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    try:
        # ランダム方策で実行
        print(f"ランダム方策でPandaロボットの{task_name}タスクを制御します...")
        run_random_policy(env, episodes=2)
        
    finally:
        # 環境をクローズ
        env.close()
        print("環境をクローズしました。")

if __name__ == "__main__":
    main()