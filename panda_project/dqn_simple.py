#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Franka Emika Pandaロボットの強化学習サンプルスクリプト（簡易版）
TensorFlowを使用したDQN（Deep Q-Network）でPandaロボットを制御します。

Reinforcement Learning Sample Script for Franka Emika Panda Robot (Simplified Version)
Control the Panda robot using DQN (Deep Q-Network) with TensorFlow.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import time
from collections import deque
import random
from panda_gym.envs import PandaReachEnv

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # 割引率 / discount factor
        self.epsilon = 1.0   # 探索率 / exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
    
    def _build_model(self):
        # ニューラルネットワークの構築 / Build neural network
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def remember(self, state, action, reward, next_state, done):
        # 経験の記憶 / Remember experience
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        # ε-グリーディー方策に基づく行動選択 / Select action based on ε-greedy policy
        if np.random.rand() <= self.epsilon:
            # 離散行動空間から選択した行動を連続空間にマッピング
            # Map selected action from discrete action space to continuous space
            raw_action = np.random.randint(0, self.action_size)
            return self._map_to_continuous(raw_action)
        
        act_values = self.model.predict(state, verbose=0)
        raw_action = np.argmax(act_values[0])
        return self._map_to_continuous(raw_action)
    
    def _map_to_continuous(self, action_idx):
        # 離散アクションから連続アクションへのマッピング / Map from discrete to continuous actions
        action_map = {
            0: [0.1, 0, 0],   # x+
            1: [-0.1, 0, 0],  # x-
            2: [0, 0.1, 0],   # y+
            3: [0, -0.1, 0],  # y-
            4: [0, 0, 0.1],   # z+
            5: [0, 0, -0.1],  # z-
        }
        return action_map.get(action_idx, [0, 0, 0])
    
    def replay(self, batch_size):
        # 経験再生 / Experience replay
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                # Q学習の更新式 / Q-learning update formula
                action_idx = self._get_action_idx(action)
                target = (reward + self.gamma * 
                         np.amax(self.model.predict(next_state, verbose=0)[0]))
            
            target_f = self.model.predict(state, verbose=0)
            action_idx = self._get_action_idx(action)
            target_f[0][action_idx] = target
            
            # モデルの更新 / Update model
            self.model.fit(state, target_f, epochs=1, verbose=0)
        
        # εの減衰 / Decay ε
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def _get_action_idx(self, action):
        # 連続アクションから離散アクションインデックスへの逆マッピング
        # Reverse mapping from continuous actions to discrete action indices
        if action[0] > 0:
            return 0
        elif action[0] < 0:
            return 1
        elif action[1] > 0:
            return 2
        elif action[1] < 0:
            return 3
        elif action[2] > 0:
            return 4
        elif action[2] < 0:
            return 5
        return 0  # デフォルト / default

def preprocess_state(state):
    # 観測の前処理 / Preprocess observation
    flattened = np.concatenate([
        state['observation'], 
        state['achieved_goal'], 
        state['desired_goal']
    ])
    return np.reshape(flattened, [1, len(flattened)])

def train_dqn(env, episodes=15, max_steps=1000):
    """
    DQNエージェントを使用してPandaロボットを訓練します。
    
    Train the Panda robot using a DQN agent.
    
    Args:
        env: gym環境 / gym environment
        episodes: 訓練エピソード数 / number of training episodes
        max_steps: 各エピソードの最大ステップ数 / maximum number of steps per episode
    
    Returns:
        scores: 各エピソードのスコア / scores for each episode
    """
    # 状態空間と行動空間のサイズ / Size of state and action spaces
    observation, info = env.reset()
    processed_state = preprocess_state(observation)
    state_size = processed_state.shape[1]
    action_size = 6  # 単純化された離散行動空間（各軸の正負） / Simplified discrete action space (positive and negative for each axis)
    
    # エージェントの初期化 / Initialize agent
    agent = DQNAgent(state_size, action_size)
    batch_size = 32
    
    scores = []
    
    for episode in range(episodes):
        # 環境のリセット / Reset environment
        state, info = env.reset()
        processed_state = preprocess_state(state)
        total_reward = 0
        
        for step in range(max_steps):
            # レンダリング / Rendering
            env.render()
            time.sleep(0.01)
            
            # 行動の選択 / Select action
            action = agent.act(processed_state)
            
            # 環境を1ステップ進める / Advance environment by one step
            next_state, reward, terminated, truncated, info = env.step(action)
            processed_next_state = preprocess_state(next_state)
            done = terminated or truncated
            
            # 経験の記憶 / Remember experience
            agent.remember(processed_state, action, reward, processed_next_state, done)
            
            # 状態の更新 / Update state
            processed_state = processed_next_state
            total_reward += reward
            
            if done:
                break
        
        # 報酬の記録 / Record reward
        scores.append(total_reward)
        
        # 経験再生 / Experience replay
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
        
        print(f"エピソード / Episode: {episode+1}/{episodes}, スコア / Score: {total_reward}, ε: {agent.epsilon:.2f}")
    
    return scores

def plot_training_results(scores):
    """
    訓練結果をプロットします。
    
    Plot the training results.
    """
    plt.figure(figsize=(12, 6))
    
    # 累積報酬のプロット / Plot cumulative rewards
    plt.plot(range(1, len(scores) + 1), scores)
    plt.title('DQN Training Progress / DQN訓練の進捗')
    plt.xlabel('Episode / エピソード')
    plt.ylabel('Total Reward / 合計報酬')
    plt.grid(True)
    
    # 移動平均のプロット / Plot moving average
    window_size = min(5, len(scores))
    if window_size > 0:
        moving_avg = np.convolve(scores, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size, len(scores) + 1), moving_avg, 'r--', 
                 label=f'{window_size}-Episode Moving Average / {window_size}エピソード移動平均')
    
    plt.legend()
    plt.savefig('dqn_training_results.png')

def main():
    """
    メイン実行関数
    
    Main execution function
    """
    # 環境の作成 / Create environment
    print("Panda-Gym環境を初期化中... / Initializing Panda-Gym environment...")
    env = PandaReachEnv(render=True)
    
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    try:
        # エピソード数の指定 / Specify number of episodes
        episodes = input("訓練エピソード数を入力してください (デフォルト: 15) / Enter the number of training episodes (default: 15): ").strip()
        if episodes and episodes.isdigit():
            episodes = int(episodes)
        else:
            episodes = 15
        
        # DQNで訓練 / Train with DQN
        print(f"DQNでPandaロボットを{episodes}エピソード訓練します... / Training the Panda robot for {episodes} episodes with DQN...")
        scores = train_dqn(env, episodes=episodes)
        
        # 結果をプロット / Plot results
        plot_training_results(scores)
        
    finally:
        # 環境をクローズ / Close environment
        env.close()
        print("環境をクローズしました。/ Environment closed.")

if __name__ == "__main__":
    main()