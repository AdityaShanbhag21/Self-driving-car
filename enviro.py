import numpy as np
import random
global EPSILON
import matplotlib.pyplot as plt
GRID_SIZE = 10
MAX_STEPS = 200

ACTIONS = [0,1,2,3]      #0=Up,1=Right,2=Down,3=Left

SYMBOLS = {
    "car": "🚗",
    "goal": "⭐",
    "obstacle": "🟥",
    "empty": "⬜"
}

class GridWorld:            #Environment class
    def __init__(self):
        self.grid_size= GRID_SIZE
        self.start_pos=(0,0)
        self.goal_positions=[(9,9),(9,5)]
        self.obstacles=[(1,1),(3,4),(4,8),(7,1),(6,6),(8,3)]
        self.traffic_signals={(3,2):"red", (5,5):"green", (7,7):"green", (2,8):"green"}
        self.visited_signals = set()

        
        self.reset()

    def reset(self):
        self.agent_pos=self.start_pos
        self.visited_signals = set()
        return self.agent_pos

    def is_valid(self, pos):
        return (
            0 <= pos[0]<self.grid_size and
            0 <= pos[1]<self.grid_size and
            pos not in self.obstacles
        )

    def step(self, action):
        x, y = self.agent_pos

        # Determine new position based on action
        if action == 0:  # Up
            new_pos = (x - 1, y)
        elif action == 1:  # Right
            new_pos = (x, y + 1)
        elif action == 2:  # Down
            new_pos = (x + 1, y)
        elif action == 3:  # Left
            new_pos = (x, y - 1)
        else:
            new_pos = self.agent_pos  # Invalid action

        # Default reward for taking a step
        reward = -0.5
        done = False

        # Check if new position is valid (not off-grid or into obstacle)
        if self.is_valid(new_pos):
            self.agent_pos = new_pos
        else:
            # Invalid move = crash
            return self.agent_pos, -10, True

        # Traffic signal handling (after moving)
        if self.agent_pos in self.traffic_signals:
            signal = self.traffic_signals[self.agent_pos]
            if signal == "red":
                reward -= 5
            elif signal == "green":
                # Only reward once
                if self.agent_pos not in self.visited_signals:
                    reward += 5
                    self.visited_signals.add(self.agent_pos)

        # Check for goal AFTER applying traffic signal reward
        if self.agent_pos in self.goal_positions:
            reward += 100
            done = True

        return self.agent_pos, reward, done


    def render(self):
        grid = [[SYMBOLS["empty"] for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        for (i, j), signal in self.traffic_signals.items():
            grid[i][j] = "🟩" if signal == "green" else "🟥"

        for (i, j) in self.obstacles:
            grid[i][j] = SYMBOLS["obstacle"]
        x, y = self.agent_pos
        #gx, gy = self.goal_pos
        grid[x][y] = SYMBOLS["car"]
        for (gx, gy) in self.goal_positions:
            grid[gx][gy] = SYMBOLS["goal"]


        for row in grid:
            print(" ".join(row))
        print("\n")

env = GridWorld()
state = env.reset()

done = False
while not done:
    env.render()
    action = random.choice(ACTIONS)
    state, reward, done = env.step(action)
    print(f"Action: {action}, New State: {state}, Reward: {reward}")

# Q-learning parameters
EPISODES = 1000
LEARNING_RATE = 0.1     # α
DISCOUNT_FACTOR = 0.9   # γ
EPSILON = 1.0           # Initial exploration rate
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01

q_table = {}

def get_q(state, action):
    return q_table.get((state, action), 0.0)

def choose_action(state):
    if np.random.rand() < EPSILON:
        return random.choice(ACTIONS)  #explore
    else:
        q_vals = [get_q(state, a) for a in ACTIONS]
        return np.argmax(q_vals)  #exploit

def update_q(state, action, reward, next_state):
    max_future_q = max([get_q(next_state, a) for a in ACTIONS])
    old_q = get_q(state, action)
    new_q = old_q + LEARNING_RATE * (reward + DISCOUNT_FACTOR * max_future_q - old_q)
    q_table[(state, action)] = new_q

env = GridWorld()

rewards_per_episode = []


for episode in range(EPISODES):
    state = env.reset()
    total_reward = 0
    done = False
    steps = 0

    while not done and steps < MAX_STEPS: 
        action = choose_action(state)
        next_state, reward, done = env.step(action)

        update_q(state, action, reward, next_state)
        state = next_state
        total_reward += reward
        steps += 1  


    rewards_per_episode.append(total_reward)
    
    EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)

    if (episode + 1) % 100 == 0:
        print(f"Episode {episode+1}: Total Reward = {total_reward:.2f}, Epsilon = {EPSILON:.3f}")



plt.plot(rewards_per_episode)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Self-Driving Car Learning Progress")
plt.show()

import time
import os

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def visualize_policy():
    state = env.reset()
    done = False
    steps = 0

    while not done and steps < 50:
        clear_screen()
        env.render()
        time.sleep(0.4)

        # Choose best action from Q-table (greedy)
        q_vals = [get_q(state, a) for a in ACTIONS]
        action = np.argmax(q_vals)
        next_state, reward, done = env.step(action)
        state = next_state
        steps += 1

    env.render()
    if state in env.goal_positions:
        print("🎉 Car reached the goal!")
    else:
        print("🚧 Car got stuck or took too long.")

visualize_policy()
