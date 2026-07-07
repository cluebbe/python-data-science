"""
Workshop Tutorial: Reinforcement Learning with Q-Learning (from scratch)
=========================================================================

Reinforcement learning (RL) is not supervised or unsupervised learning —
there's no labeled dataset. Instead, an *agent* learns by acting inside an
*environment*, receiving a *reward* after each action, and gradually
figuring out which actions lead to the most reward over time.

This tutorial builds a tiny 5x5 GridWorld from scratch (no gym/RL library
needed) and trains a Q-learning agent to navigate it:

    Q(s, a) <- Q(s, a) + alpha * [ r + gamma * max_a' Q(s', a') - Q(s, a) ]

The agent starts at S, must reach the goal G (+10 reward), and must learn
to avoid a trap X (-10 reward) that sits on many of the shortest paths.
Every other move costs -1, so the agent is also pushed toward efficiency.
"""

# ------------------------------------------------------------
# Step 0 — Imports
# ------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Step 1 — Build the GridWorld environment
# ------------------------------------------------------------
print("=== PART 1: The GridWorld Environment ===\n")

GRID_SIZE = 5
START = (0, 0)
GOAL = (4, 4)
TRAP = (2, 2)

ACTIONS = ["up", "down", "left", "right"]
ACTION_DELTAS = {
    0: (-1, 0),   # up
    1: (1, 0),    # down
    2: (0, -1),   # left
    3: (0, 1),    # right
}
N_ACTIONS = len(ACTIONS)


def step_env(state, action):
    """Apply `action` to `state`. Returns (next_state, reward, done)."""
    row, col = state
    d_row, d_col = ACTION_DELTAS[action]
    # Clip to the grid so moving into a wall just bumps in place.
    next_row = min(max(row + d_row, 0), GRID_SIZE - 1)
    next_col = min(max(col + d_col, 0), GRID_SIZE - 1)
    next_state = (next_row, next_col)

    if next_state == GOAL:
        return next_state, 10.0, True
    if next_state == TRAP:
        return next_state, -10.0, True
    return next_state, -1.0, False


def render_grid():
    rows = []
    for r in range(GRID_SIZE):
        row_chars = []
        for c in range(GRID_SIZE):
            cell = (r, c)
            if cell == START:
                row_chars.append("S")
            elif cell == GOAL:
                row_chars.append("G")
            elif cell == TRAP:
                row_chars.append("X")
            else:
                row_chars.append(".")
        rows.append(" ".join(row_chars))
    return "\n".join(rows)


print(render_grid())
print("\nS = start, G = goal (+10), X = trap (-10), . = empty (-1 per step)\n")

# ------------------------------------------------------------
# Step 2 — Initialize the Q-table and hyperparameters
# ------------------------------------------------------------
print("=== PART 2: Q-table and Hyperparameters ===\n")

q_table = np.zeros((GRID_SIZE, GRID_SIZE, N_ACTIONS))

ALPHA = 0.1            # learning rate
GAMMA = 0.9             # discount factor
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995
N_EPISODES = 500
MAX_STEPS = 50

print(f"Q-table shape: {q_table.shape}  (rows x cols x actions)")
print(f"alpha={ALPHA}, gamma={GAMMA}, episodes={N_EPISODES}\n")

# ------------------------------------------------------------
# Step 3 — Epsilon-greedy action selection
# ------------------------------------------------------------
print("=== PART 3: Epsilon-Greedy Action Selection ===\n")


def choose_action(state, q_table, epsilon):
    if np.random.random() < epsilon:
        return np.random.randint(N_ACTIONS)
    row, col = state
    return int(np.argmax(q_table[row, col]))


np.random.seed(0)
demo_epsilon = 0.3
samples = [choose_action((0, 0), q_table, demo_epsilon) for _ in range(2000)]
non_zero_fraction = np.mean([a != 0 for a in samples])
print(f"epsilon={demo_epsilon} -> {non_zero_fraction:.3f} of picks were a non-zero action")
print(f"(expected ~{demo_epsilon * 3 / 4:.3f}, since 3 of 4 actions are non-zero and the "
      f"all-zero Q-table always exploits to action 0)\n")

# ------------------------------------------------------------
# Step 4 — Train the agent
# ------------------------------------------------------------
print("=== PART 4: Training the Agent ===\n")

np.random.seed(42)
q_table = np.zeros((GRID_SIZE, GRID_SIZE, N_ACTIONS))
epsilon = EPSILON_START
rewards_per_episode = []

for episode in range(N_EPISODES):
    state = START
    total_reward = 0.0

    for _ in range(MAX_STEPS):
        action = choose_action(state, q_table, epsilon)
        next_state, reward, done = step_env(state, action)

        row, col = state
        next_row, next_col = next_state
        best_next_q = np.max(q_table[next_row, next_col])
        td_target = reward if done else reward + GAMMA * best_next_q
        td_error = td_target - q_table[row, col, action]
        q_table[row, col, action] += ALPHA * td_error

        state = next_state
        total_reward += reward
        if done:
            break

    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
    rewards_per_episode.append(total_reward)

print(f"Average reward, first 10 episodes : {np.mean(rewards_per_episode[:10]):.2f}")
print(f"Average reward, last 10 episodes  : {np.mean(rewards_per_episode[-10:]):.2f}")
print(f"Final epsilon                     : {epsilon:.4f}\n")

# ------------------------------------------------------------
# Step 5 — Visualize the learning curve
# ------------------------------------------------------------
window = 20
smoothed = np.convolve(rewards_per_episode, np.ones(window) / window, mode="valid")

plt.figure(figsize=(8, 4))
plt.plot(rewards_per_episode, alpha=0.3, label="Reward per episode")
plt.plot(
    range(window - 1, len(rewards_per_episode)),
    smoothed,
    label=f"{window}-episode moving average",
)
plt.axhline(y=3, color="green", linestyle="--", label="Theoretical optimum (3)")
plt.xlabel("Episode")
plt.ylabel("Total reward")
plt.title("Q-Learning Reward Over Training")
plt.legend()
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# Step 6 — Extract and evaluate the learned policy
# ------------------------------------------------------------
print("=== PART 6: Extracting and Evaluating the Learned Policy ===\n")

ARROWS = {0: "^", 1: "v", 2: "<", 3: ">"}


def render_policy(q_table):
    rows = []
    for r in range(GRID_SIZE):
        row_chars = []
        for c in range(GRID_SIZE):
            cell = (r, c)
            if cell == GOAL:
                row_chars.append("G")
            elif cell == TRAP:
                row_chars.append("X")
            else:
                best_action = int(np.argmax(q_table[r, c]))
                row_chars.append(ARROWS[best_action])
        rows.append(" ".join(row_chars))
    return "\n".join(rows)


print("Learned greedy policy:")
print(render_policy(q_table))
print()


def run_episode(policy_fn, seed=None):
    if seed is not None:
        np.random.seed(seed)
    state = START
    path = [state]
    total_reward = 0.0
    for _ in range(MAX_STEPS):
        action = policy_fn(state)
        state, reward, done = step_env(state, action)
        path.append(state)
        total_reward += reward
        if done:
            break
    return path, total_reward


greedy_policy = lambda s: int(np.argmax(q_table[s[0], s[1]]))
random_policy = lambda s: np.random.randint(N_ACTIONS)

greedy_path, greedy_reward = run_episode(greedy_policy)
random_path, random_reward = run_episode(random_policy, seed=1)

print(f"Greedy policy : {len(greedy_path) - 1} steps, total reward {greedy_reward:.1f}")
print(f"Path: {greedy_path}\n")
print(f"Random policy : {len(random_path) - 1} steps, total reward {random_reward:.1f}")
print(f"Path: {random_path}\n")
