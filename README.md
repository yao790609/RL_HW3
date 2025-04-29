# 🎰 Multi-Armed Bandit Performance Analysis

This project analyzes and compares four popular strategies for the **Multi-Armed Bandit (MAB)** problem using simulation:

- 🎯 **Epsilon-Greedy**
- 📈 **Upper Confidence Bound (UCB)**
- 🔥 **Softmax Action Selection**
- 🧠 **Thompson Sampling**

---

## 📘 Algorithm Overview

### Epsilon-Greedy

Selects a random action with probability $\varepsilon$ and the best-known action with probability $1 - \varepsilon$:

$$
A_t = 
\begin{cases}
\text{random action} & \text{with probability } \varepsilon \\
\arg\max_a Q_t(a) & \text{with probability } 1 - \varepsilon
\end{cases}
$$

---

### Upper Confidence Bound (UCB)

Selects the action with the highest upper confidence estimate:

$$
A_t = \arg\max_a \left[ Q_t(a) + c \cdot \sqrt{\frac{\ln t}{N_t(a)}} \right]
$$

- $Q_t(a)$: Estimated value of action $a$
- $N_t(a)$: Times action $a$ has been selected
- $c$: Confidence level parameter

---

### Softmax Action Selection

Actions are selected probabilistically using the softmax distribution:

$$
P(a) = \frac{e^{Q_t(a) / \tau}}{\sum_b e^{Q_t(b) / \tau}}
$$

- $\tau$: Temperature parameter (controls exploration)

---

### Thompson Sampling

Samples from the Beta distribution for each action:

$$
\theta_a \sim \text{Beta}(\alpha_a, \beta_a), \quad A_t = \arg\max_a \theta_a
$$

After receiving reward $r$:

- $\alpha_a \leftarrow \alpha_a + r$
- $\beta_a \leftarrow \beta_a + (1 - r)$

---

## 🧪 Simulation Details

- Arms: 10
- Steps: 1000
- Runs: 2000
- Rewards: Binary (Bernoulli)
- Metrics tracked:
  - Average reward per step
  - Number of times the best arm was selected
  - Convergence step (when average reward exceeds threshold for a window)

---

## 📊 Results

### 1. Average Reward

A reward vs. step plot is generated to show how each algorithm improves over time.

### 2. Best Arm Selection

Counts how many times each method selected the true best arm.

### 3. Convergence Step

Measures how fast each algorithm converges to a near-optimal average reward.

---

## 💻 How to Run

```bash
pip install matplotlib numpy
python bandit_simulation.py
# RL_HW3
