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

- **Arms**: 10
- **Steps**: 1000
- **Runs**: 2000
- **Rewards**: Binary (Bernoulli)
- **Metrics tracked**:
  - Average reward per step
  - Number of times the best arm was selected
  - Convergence step (when average reward exceeds threshold for a window)

---

## 🔄 Discussion Prompt

I am currently working on a deep reinforcement learning assignment related to Multiple Arm Bandit performance analysis, using the **Epsilon-Greedy**, **UCB**, **Softmax**, and **Thompson Sampling** method. The additional requirements are as follows:

1. Algorithm formula: Present each algorithm’s mathematical formula in LaTeX format.

2. Code and charts: Provide Python code for the algorithm and use charts to illustrate the results (e.g., cumulative rewards, convergence speed, etc.).

3. Result interpretation: Analyze the results in terms of space and time complexity.

After code for each method generated, I am going to do comparison. The goal was to compare these strategies based on their performance, especially in terms of **average reward**, **best arm selection count**, and **convergence speed**. I implemented the algorithms in Python and ran simulations to evaluate how each algorithm performs under the same conditions.

---

## 🧑‍💻 Code Organization

The project is organized as follows:

1. Each algorithm is implemented separately:
    - **Epsilon-Greedy Algorithm** code is in `MAB-Epsilon-Greedy.py`.
    - **Upper Confidence Bound (UCB)** code is in `MAB-UCB.py`.
    - **Softmax Algorithm** code is in `MAB-Softmax.py`.
    - **Thompson Sampling Algorithm** code is in `MAB-Thompson Sampling.py`.

2. Once each of the algorithms has been tested individually, the final code to compare all four methods is placed in a unified file `MAB-Comparison.py`. This script runs the simulations for all four algorithms, tracks the metrics (average reward, best arm selection count, and convergence step), and generates the comparison plots.

---

## 📊 Results

### 1. Average Reward

A reward vs. step plot is generated to show how each algorithm improves over time.

![image](https://github.com/yao790609/RL_HW3/blob/main/MAB-Epsilon-Greedy.png)
Epsilon-Greedy

![image](https://github.com/yao790609/RL_HW3/blob/main/MAB-Softmax.png)
Softmax

![image](https://github.com/yao790609/RL_HW3/blob/main/MAB-UCB.png)
UCB

![image](https://github.com/yao790609/RL_HW3/blob/main/MAB-Thompson%20Sampling.png)
Thompson Sampling

---

## 📊 Significance of Comparison Plot

The comparison plot shows the **performance over time** of each algorithm. Here's how you can interpret the chart:

- **Average Reward**: The plot shows the cumulative reward each algorithm gathers over time. A higher average reward indicates a better-performing strategy.
  
- **Convergence Speed**: The rate at which the algorithms approach their optimal reward is visible in the graph. Algorithms that reach a high reward faster are considered more efficient.
  
- **Best Arm Selection**: By comparing the number of times each algorithm selects the optimal arm (as seen in the plot), we can determine which method is more likely to identify and exploit the best arm over time.

![image](https://github.com/yao790609/RL_HW3/blob/main/methods_comparison.png)

---

## 💻 How to Run

pip install matplotlib numpy
python MAB-Thompson Sampling.py
python MAB-UCB.py
python MAB-Epsilon-Greedy.py
python MAB-Thompson Sampling.py
python MAB-Comparison.py
