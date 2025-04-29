# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 17:23:20 2025

@author: yao79
"""

import numpy as np
import matplotlib.pyplot as plt

class SoftmaxBandit:
    def __init__(self, k=10, tau=0.1, steps=1000, runs=2000):
        self.k = k
        self.tau = tau
        self.steps = steps
        self.runs = runs

    def softmax(self, q_values):
        exp_q = np.exp(q_values / self.tau)
        return exp_q / np.sum(exp_q)

    def run(self):
        average_rewards = np.zeros(self.steps)
        for run in range(self.runs):
            q_true = np.random.normal(0, 1, self.k)
            q_est = np.zeros(self.k)
            N = np.zeros(self.k)
            rewards = []

            for t in range(self.steps):
                probs = self.softmax(q_est)
                action = np.random.choice(self.k, p=probs)

                reward = np.random.normal(q_true[action], 1)
                N[action] += 1
                q_est[action] += (reward - q_est[action]) / N[action]
                rewards.append(reward)

            average_rewards += (np.array(rewards) - average_rewards) / (run + 1)
        return average_rewards

# 設定並執行
steps = 1000
softmax_bandit = SoftmaxBandit(tau=0.1, steps=steps)
softmax_rewards = softmax_bandit.run()

# 繪圖
plt.plot(softmax_rewards, label="Softmax (τ=0.1)")
plt.xlabel("Steps")
plt.ylabel("Average Reward")
plt.title("Softmax Bandit Performance")
plt.grid()
plt.legend()
plt.show()
