#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Franka Emika Pandaロボットのサンプル実行スクリプト
シンプルに整理されたバージョン
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from panda_gym.envs import PandaReachEnv

def run_random_policy(env, episodes=3, max_steps=1000):
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

def plot_rewards(rewards):
    """
    エピソードごとの報酬をプロットします。
    
    Args:
        rewards: 各エピソードの累積報酬のリスト
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(rewards) + 1), rewards, marker='o')
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.grid(True)
    plt.savefig('rewards.png')
    # フォントの警告を抑えるためにタイトルを英語にしました

def main():
    """
    メイン実行関数
    """
    # 環境の作成
    print("Panda-Gym環境を初期化中...")
    env = PandaReachEnv(render=True)
    
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    try:
        # ランダム方策で実行
        print("ランダム方策でPandaロボットを制御します...")
        rewards = run_random_policy(env, episodes=3)
        
        # 結果をプロット
        plot_rewards(rewards)
        
    finally:
        # 環境をクローズ
        env.close()
        print("環境をクローズしました。")

if __name__ == "__main__":
    main()