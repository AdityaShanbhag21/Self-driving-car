# 🚗 Self-Driving Car in a GridWorld (Reinforcement Learning)

This project simulates a basic self-driving car using Q-learning in a 10x10 GridWorld. The environment includes:
- ⛔ Obstacles
- 🟩🟥 Traffic signals (red: penalty, green: bonus)
- ⭐ Multiple goal destinations

The agent learns to navigate toward the goals efficiently using Q-learning and an epsilon-greedy exploration strategy.

## 📌 Features
- Grid-based environment with symbolic rendering
- Q-learning from scratch (no gym)
- Traffic signal interaction logic
- Real-time training visualization
- Final policy replay

## 📈 Learning Curve
Training progress is visualized using Matplotlib — you’ll see rewards improving over time as the agent learns.

## 🧠 Algorithm

Uses tabular **Q-learning**:
Q(s, a) = Q(s, a) + α [r + γ * max Q(s', a') − Q(s, a)]

- Exploration rate (ε) decays over time.
- Learning rate (α), discount factor (γ) are tunable.

## ▶️ Run It
