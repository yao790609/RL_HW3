# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 17:48:35 2025

@author: yao79
"""

import numpy as np
import matplotlib.pyplot as plt

class BanditSimulator:
    def __init__(self, k=10, steps=1000, runs=2000):
        self.k = k
        self.steps = steps
        self.runs = runs

    def simulate(self):
        ts_rewards = np.zeros(self.steps)
        eg_rewards = np.zeros(self.steps)
        ucb_rewards = np.zeros(self.steps)
        softmax_rewards = np.zeros(self.steps)

        ts_best_counts = 0
        eg_best_counts = 0
        ucb_best_counts = 0
        softmax_best_counts = 0

        ts_converge_steps = []
        eg_converge_steps = []
        ucb_converge_steps = []
        softmax_converge_steps = []

        for run in range(self.runs):
            true_probs = np.random.rand(self.k)
            best_arm = np.argmax(true_probs)

            # 各演算法
            r_ts, b_ts, c_ts = self.run_thompson(true_probs, best_arm)
            r_eg, b_eg, c_eg = self.run_epsilon_greedy(true_probs, best_arm)
            r_ucb, b_ucb, c_ucb = self.run_ucb(true_probs, best_arm)
            r_sm, b_sm, c_sm = self.run_softmax(true_probs, best_arm)

            ts_rewards += r_ts
            eg_rewards += r_eg
            ucb_rewards += r_ucb
            softmax_rewards += r_sm

            ts_best_counts += b_ts
            eg_best_counts += b_eg
            ucb_best_counts += b_ucb
            softmax_best_counts += b_sm

            ts_converge_steps.append(c_ts)
            eg_converge_steps.append(c_eg)
            ucb_converge_steps.append(c_ucb)
            softmax_converge_steps.append(c_sm)

        # 平均處理
        ts_rewards /= self.runs
        eg_rewards /= self.runs
        ucb_rewards /= self.runs
        softmax_rewards /= self.runs

        return {
            "rewards": {
                "Thompson Sampling": ts_rewards,
                "Epsilon-Greedy": eg_rewards,
                "UCB": ucb_rewards,
                "Softmax": softmax_rewards
            },
            "best_counts": {
                "Thompson Sampling": ts_best_counts,
                "Epsilon-Greedy": eg_best_counts,
                "UCB": ucb_best_counts,
                "Softmax": softmax_best_counts
            },
            "converge_steps": {
                "Thompson Sampling": np.mean(ts_converge_steps),
                "Epsilon-Greedy": np.mean(eg_converge_steps),
                "UCB": np.mean(ucb_converge_steps),
                "Softmax": np.mean(softmax_converge_steps)
            }
        }

    def run_thompson(self, true_probs, best_arm):
        alpha = np.ones(self.k)
        beta = np.ones(self.k)
        rewards, best_count = [], 0

        for t in range(self.steps):
            theta = np.random.beta(alpha, beta)
            action = np.argmax(theta)
            reward = np.random.binomial(1, true_probs[action])
            alpha[action] += reward
            beta[action] += (1 - reward)
            best_count += (action == best_arm)
            rewards.append(reward)

        return np.array(rewards), best_count, self._converge_step(rewards)

    def run_epsilon_greedy(self, true_probs, best_arm, epsilon=0.1):
        Q = np.zeros(self.k)
        N = np.zeros(self.k)
        rewards, best_count = [], 0

        for t in range(self.steps):
            if np.random.rand() < epsilon:
                action = np.random.randint(self.k)
            else:
                action = np.argmax(Q)

            reward = np.random.binomial(1, true_probs[action])
            N[action] += 1
            Q[action] += (reward - Q[action]) / N[action]
            best_count += (action == best_arm)
            rewards.append(reward)

        return np.array(rewards), best_count, self._converge_step(rewards)

    def run_ucb(self, true_probs, best_arm, c=2):
        Q = np.zeros(self.k)
        N = np.zeros(self.k)
        rewards, best_count = [], 0

        for t in range(1, self.steps + 1):
            ucb_values = Q + c * np.sqrt(np.log(t + 1) / (N + 1e-5))
            action = np.argmax(ucb_values)
            reward = np.random.binomial(1, true_probs[action])
            N[action] += 1
            Q[action] += (reward - Q[action]) / N[action]
            best_count += (action == best_arm)
            rewards.append(reward)

        return np.array(rewards), best_count, self._converge_step(rewards)

    def run_softmax(self, true_probs, best_arm, tau=0.2):
        Q = np.zeros(self.k)
        N = np.zeros(self.k)
        rewards, best_count = [], 0

        for t in range(self.steps):
            exp_q = np.exp(Q / tau)
            probs = exp_q / np.sum(exp_q)
            action = np.random.choice(np.arange(self.k), p=probs)
            reward = np.random.binomial(1, true_probs[action])
            N[action] += 1
            Q[action] += (reward - Q[action]) / N[action]
            best_count += (action == best_arm)
            rewards.append(reward)

        return np.array(rewards), best_count, self._converge_step(rewards)

    def _converge_step(self, rewards, window=50, threshold=0.95):
        # 平滑窗口內的平均超過 threshold 時視為收斂
        for i in range(len(rewards) - window):
            if np.mean(rewards[i:i+window]) > threshold:
                return i
        return len(rewards)  # 未收斂

# 執行模擬
sim = BanditSimulator(k=10, steps=1000, runs=2000)
result = sim.simulate()

# 畫圖
for name, rewards in result["rewards"].items():
    plt.plot(rewards, label=name)
plt.xlabel("Steps")
plt.ylabel("Average Reward")
plt.title("Multi-Armed Bandit - Performance Comparison")
plt.legend()
plt.grid()
plt.show()

# 顯示統計
print("✅ Best Arm Selection Count (out of", sim.runs * sim.steps, "):")
for name, count in result["best_counts"].items():
    print(f"{name:20}: {count}")

print("\n📉 Convergence Step (avg steps until stable reward > 0.95):")
for name, step in result["converge_steps"].items():
    print(f"{name:20}: {step:.2f}")
