#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Franka Emika Pandaロボットのサンプル実行スクリプト
panda-gym 3.0.0に対応し、環境を明示的に登録
"""

import numpy as np
import time
import matplotlib.pyplot as plt

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
            
            # 環境をレンダリング（render()メソッドを使用）
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
    plt.title('エピソードごとの累積報酬')
    plt.xlabel('エピソード')
    plt.ylabel('累積報酬')
    plt.grid(True)
    plt.savefig('rewards.png')
    plt.show()

def main():
    """
    メイン実行関数
    """
    # 環境の作成（panda_gymから直接環境クラスをインポート）
    print("Panda-Gym環境を初期化中...")
    
    try:
        # panda-gymの環境クラスを直接インポート
        from panda_gym.envs import PandaReachEnv
        
        # 環境を直接インスタンス化
        env = PandaReachEnv(render=True)
        
        print("環境が正常に作成されました。")
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
            
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        print("代替方法を試みます...")
        
        try:
            # 代替方法: 明示的に環境を登録してからmakeで作成
            import gymnasium as gym
            import panda_gym
            
            # 環境の登録
            print("環境を明示的に登録します...")
            
            # 全ての利用可能な環境を登録
            from gymnasium.envs import registry
            
            # 登録がない場合のみ登録
            env_id = "PandaReach-v3"
            if env_id not in registry.keys():
                from panda_gym.envs import PandaReachEnv
                gym.register(
                    id=env_id,
                    entry_point="panda_gym.envs:PandaReachEnv",
                    max_episode_steps=100,
                )
                print(f"{env_id}を登録しました")
            
            # 環境の作成
            env = gym.make(env_id, render=True)
            
            print("環境が正常に作成されました。")
            print(f"Observation space: {env.observation_space}")
            print(f"Action space: {env.action_space}")
            
            # ランダム方策で実行
            print("ランダム方策でPandaロボットを制御します...")
            rewards = run_random_policy(env, episodes=3)
            
            # 結果をプロット
            plot_rewards(rewards)
            
            # 環境をクローズ
            env.close()
            print("環境をクローズしました。")
        
        except Exception as e2:
            print(f"代替方法も失敗しました: {e2}")
            print("\nもう一つの方法を試みます: 直接PandaReachEnvを使用...")
            
            try:
                # PandaReachEnvを直接使用
                from panda_gym.envs import PandaReachEnv
                env = PandaReachEnv(render=True)
                
                observation, info = env.reset()
                
                print("\nランダムな行動をいくつか実行します...")
                for _ in range(100):
                    action = env.action_space.sample()
                    observation, reward, terminated, truncated, info = env.step(action)
                    env.render()
                    time.sleep(0.01)
                    
                    if terminated or truncated:
                        break
                
                env.close()
                print("環境をクローズしました。")
            
            except Exception as e3:
                print(f"すべての方法が失敗しました: {e3}")
                print("\n問題のトラブルシューティング:")
                print("1. panda-gymのバージョンを確認: pip show panda-gym")
                print("2. gymnasiumがインストールされているか確認: pip show gymnasium")
                print("3. 以下の簡単なコードを別のファイルで試してください:")
                print("""
# test_panda.py
from panda_gym.envs import PandaReachEnv

env = PandaReachEnv(render=True)
obs, info = env.reset()

for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    
    if terminated or truncated:
        break

env.close()
                """)

if __name__ == "__main__":
    main()