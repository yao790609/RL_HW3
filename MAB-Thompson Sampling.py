# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 17:37:49 2025

@author: yao79
"""

import numpy as np
import matplotlib.pyplot as plt

class ThompsonSamplingBandit:
    def __init__(self, k=10, steps=1000, runs=2000):
        self.k = k
        self.steps = steps
        self.runs = runs

    def run(self):
        average_rewards = np.zeros(self.steps)

        for run in range(self.runs):
            # 每次 run 隨機生成真實機率值
            true_probs = np.random.rand(self.k)
            alpha = np.ones(self.k)
            beta = np.ones(self.k)
            rewards = []

            for t in range(self.steps):
                # 從每個 arm 的 Beta 分布抽樣
                theta = np.random.beta(alpha, beta)
                action = np.argmax(theta)
                reward = np.random.binomial(1, true_probs[action])
                rewards.append(reward)

                # 更新參數
                alpha[action] += reward
                beta[action] += (1 - reward)

            average_rewards += (np.array(rewards) - average_rewards) / (run + 1)
        return average_rewards

# 執行
steps = 1000
ts_bandit = ThompsonSamplingBandit(steps=steps)
ts_rewards = ts_bandit.run()

# 繪圖
plt.plot(ts_rewards, label="Thompson Sampling")
plt.xlabel("Steps")
plt.ylabel("Average Reward")
plt.title("Thompson Sampling Performance")
plt.grid()
plt.legend()
plt.show()
