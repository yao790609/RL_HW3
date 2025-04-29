# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 17:22:15 2025

@author: yao79
"""

import numpy as np
import matplotlib.pyplot as plt

class UCBBandit:
    def __init__(self, k=10, c=2, steps=1000, runs=2000):
        self.k = k
        self.c = c
        self.steps = steps
        self.runs = runs

    def run(self):
        average_rewards = np.zeros(self.steps)
        for run in range(self.runs):
            q_true = np.random.normal(0, 1, self.k)  # 真實平均報酬
            q_est = np.zeros(self.k)                # 預估平均報酬
            N = np.zeros(self.k)                    # 每臂選擇次數
            rewards = []

            for t in range(self.steps):
                if t < self.k:
                    action = t  # 每個臂先嘗試一次
                else:
                    ucb_values = q_est + self.c * np.sqrt(np.log(t + 1) / (N + 1e-5))
                    action = np.argmax(ucb_values)

                reward = np.random.normal(q_true[action], 1)
                N[action] += 1
                q_est[action] += (reward - q_est[action]) / N[action]
                rewards.append(reward)

            average_rewards += (np.array(rewards) - average_rewards) / (run + 1)
        return average_rewards

# 設定與執行
steps = 1000
ucb_bandit = UCBBandit(c=2, steps=steps)
ucb_rewards = ucb_bandit.run()

# 畫圖
plt.plot(ucb_rewards, label="UCB (c=2)")
plt.xlabel("Steps")
plt.ylabel("Average Reward")
plt.title("UCB Performance")
plt.grid()
plt.legend()
plt.show()
