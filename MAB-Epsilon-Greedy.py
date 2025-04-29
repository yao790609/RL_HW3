# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 17:19:11 2025

@author: yao79
"""

import numpy as np
import matplotlib.pyplot as plt

class EpsilonGreedyBandit:
    def __init__(self, k=10, epsilon=0.1, steps=1000, runs=2000):
        self.k = k
        self.epsilon = epsilon
        self.steps = steps
        self.runs = runs

    def run(self):
        average_rewards = np.zeros(self.steps)
        for run in range(self.runs):
            q_true = np.random.normal(0, 1, self.k)   # 真實期望值
            q_est = np.zeros(self.k)                  # 預估值
            N = np.zeros(self.k)                      # 每臂選擇次數
            rewards = []

            for t in range(self.steps):
                if np.random.rand() < self.epsilon:
                    action = np.random.choice(self.k)
                else:
                    action = np.argmax(q_est)

                reward = np.random.normal(q_true[action], 1)
                N[action] += 1
                q_est[action] += (reward - q_est[action]) / N[action]
                rewards.append(reward)

            average_rewards += (np.array(rewards) - average_rewards) / (run + 1)
        return average_rewards

# 多個 epsilon 比較
epsilons = [0, 0.01, 0.1]
steps = 1000
results = []

for eps in epsilons:
    bandit = EpsilonGreedyBandit(epsilon=eps, steps=steps)
    rewards = bandit.run()
    results.append(rewards)

# 畫圖
for i, eps in enumerate(epsilons):
    plt.plot(results[i], label=f"ε = {eps}")
plt.xlabel("Steps")
plt.ylabel("Average Reward")
plt.title("Epsilon-Greedy Performance Comparison")
plt.legend()
plt.grid()
plt.show()
