import numpy as np
import random
import gym
from gym import spaces
import matplotlib.pyplot as plt

COLOR_MAP = {
    0: "#cdc1b4", 2: "#eee4da", 4: "#ede0c8", 8: "#f2b179",
    16: "#f59563", 32: "#f67c5f", 64: "#f65e3b", 128: "#edcf72",
    256: "#edcc61", 512: "#edc850", 1024: "#edc53f", 2048: "#edc22e",
    4096: "#3c3a32", 8192: "#3c3a32", 16384: "#3c3a32", 32768: "#3c3a32"
}
TEXT_COLOR = {
    2: "#776e65", 4: "#776e65", 8: "#f9f6f2", 16: "#f9f6f2",
    32: "#f9f6f2", 64: "#f9f6f2", 128: "#f9f6f2", 256: "#f9f6f2",
    512: "#f9f6f2", 1024: "#f9f6f2", 2048: "#f9f6f2", 4096: "#f9f6f2"
}

class Game2048Env(gym.Env):
    def __init__(self):
        super(Game2048Env, self).__init__()

        self.size = 4
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0

        # Action space: 0: up, 1: down, 2: left, 3: right
        self.action_space = spaces.Discrete(4)
        self.actions = ["up", "down", "left", "right"]

        self.last_move_valid = True

        # if board is not None:
        #     self.board = board
        # else:
        self.reset()

    def reset(self):
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        self.add_random_tile()
        self.add_random_tile()
        return self.board

    def add_random_tile(self):
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            x, y = random.choice(empty_cells)
            self.board[x, y] = 2 if random.random() < 0.9 else 4

    def compress(self, row):
        new_row = row[row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        return new_row

    def merge(self, row):
        for i in range(len(row) - 1):
            if row[i] == row[i + 1] and row[i] != 0:
                row[i] *= 2
                row[i + 1] = 0
                self.score += row[i]
        return row

    def move_left(self):
        moved = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            new_row = self.compress(self.board[i])
            new_row = self.merge(new_row)
            new_row = self.compress(new_row)
            self.board[i] = new_row
            if not np.array_equal(original_row, self.board[i]):
                moved = True
        return moved

    def move_right(self):
        moved = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            reversed_row = self.board[i][::-1]
            reversed_row = self.compress(reversed_row)
            reversed_row = self.merge(reversed_row)
            reversed_row = self.compress(reversed_row)
            self.board[i] = reversed_row[::-1]
            if not np.array_equal(original_row, self.board[i]):
                moved = True
        return moved

    def move_up(self):
        moved = False
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            col = self.compress(self.board[:, j])
            col = self.merge(col)
            col = self.compress(col)
            self.board[:, j] = col
            if not np.array_equal(original_col, self.board[:, j]):
                moved = True
        return moved

    def move_down(self):
        moved = False
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            reversed_col = self.board[:, j][::-1]
            reversed_col = self.compress(reversed_col)
            reversed_col = self.merge(reversed_col)
            reversed_col = self.compress(reversed_col)
            self.board[:, j] = reversed_col[::-1]
            if not np.array_equal(original_col, self.board[:, j]):
                moved = True
        return moved

    def is_game_over(self):
        if np.any(self.board == 0):
            return False
        for i in range(self.size):
            for j in range(self.size - 1):
                if self.board[i, j] == self.board[i, j+1]:
                    return False
        for j in range(self.size):
            for i in range(self.size - 1):
                if self.board[i, j] == self.board[i+1, j]:
                    return False

        return True

    def step(self, action):
        assert self.action_space.contains(action), "Invalid action"

        if action == 0:
            moved = self.move_up()
        elif action == 1:
            moved = self.move_down()
        elif action == 2:
            moved = self.move_left()
        elif action == 3:
            moved = self.move_right()
        else:
            moved = False

        self.last_move_valid = moved

        # Store the state before adding random tile
        state_before_random = self.board.copy()

        if moved:
            self.add_random_tile()

        done = self.is_game_over()

        return self.board, self.score, done, {"state_before_random": state_before_random}

    def render(self, mode="human", action=None):
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, self.size - 0.5)
        ax.set_ylim(-0.5, self.size - 0.5)

        for i in range(self.size):
            for j in range(self.size):
                value = self.board[i, j]
                color = COLOR_MAP.get(value, "#3c3a32")
                text_color = TEXT_COLOR.get(value, "white")
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=color, edgecolor="black")
                ax.add_patch(rect)

                if value != 0:
                    ax.text(j, i, str(value), ha='center', va='center',
                            fontsize=16, fontweight='bold', color=text_color)
        title = f"score: {self.score}"
        if action is not None:
            title += f" | action: {self.actions[action]}"
        plt.title(title)
        plt.gca().invert_yaxis()
        plt.show()

    def simulate_row_move(self, row):
        new_row = row[row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        for i in range(len(new_row) - 1):
            if new_row[i] == new_row[i + 1] and new_row[i] != 0:
                new_row[i] *= 2
                new_row[i + 1] = 0
        new_row = new_row[new_row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        return new_row

    def is_move_legal(self, action):
        temp_board = self.board.copy()

        if action == 0:  # Move up
            for j in range(self.size):
                col = temp_board[:, j]
                new_col = self.simulate_row_move(col)
                temp_board[:, j] = new_col
        elif action == 1:  # Move down
            for j in range(self.size):
                col = temp_board[:, j][::-1]
                new_col = self.simulate_row_move(col)
                temp_board[:, j] = new_col[::-1]
        elif action == 2:  # Move left
            for i in range(self.size):
                row = temp_board[i]
                temp_board[i] = self.simulate_row_move(row)
        elif action == 3:  # Move right
            for i in range(self.size):
                row = temp_board[i][::-1]
                new_row = self.simulate_row_move(row)
                temp_board[i] = new_row[::-1]
        else:
            raise ValueError("Invalid action")
        return not np.array_equal(self.board, temp_board)
 
import copy
import random
import math
import numpy as np
from collections import defaultdict

class NTupleApproximator:
    def __init__(self, board_size, patterns):
        self.board_size = board_size
        self.weights = [defaultdict(float) for _ in patterns]
        self.patterns = patterns

    def tile_to_index(self, tile):
        """
        Converts tile values to an index for the lookup table.
        Ensures all tiles are converted to a consistent index representation.
        """
        if tile == 0:
            return 0
        return int(math.log2(tile))

    def get_feature(self, board, coords):
        """
        Extract feature tuple from board based on given coordinates.
        """
        return tuple(self.tile_to_index(board[x, y]) for x, y in coords)

    def value(self, board):
        """
        Estimate the board value by summing values from all patterns.
        """
        return sum(
            self.weights[i][self.get_feature(board, pattern)] 
            for i, pattern in enumerate(self.patterns)
        )

    def update(self, board, delta, alpha):
        """
        Update weights for all patterns based on the TD error.
        """
        for i, pattern in enumerate(self.patterns):
            feature = self.get_feature(board, pattern)
            self.weights[i][feature] += alpha * delta
            
    def average_value(self):
        """
        Compute the average weight value over all patterns.
        If no weights are present, return 1.0 to avoid division by zero.
        """
        total = 0.0
        count = 0
        for weight_dict in self.weights:
            for value in weight_dict.values():
                total += value
                count += 1
        return total / count if count > 0 else 1.0

def choose_best_action(env, approximator, legal_moves, gamma):
    """
    Choose the best action based on estimated future value.
    """
    action_values = []
    for action in legal_moves:
        # Create a deep copy to simulate the move
        test_env = copy.deepcopy(env)
        prev_score = test_env.score
        
        # Simulate the move
        next_state, new_score, _, info = test_env.step(action)
        
        # Calculate incremental reward
        incremental_reward = new_score - prev_score
        
        # Use the state before random tile for value estimation
        next_state_before_random = info["state_before_random"]
        
        # Combine immediate reward with estimated future value
        action_value = incremental_reward + gamma * approximator.value(next_state_before_random)
        action_values.append((action, action_value))
    
    # Return the action with the highest estimated value
    return max(action_values, key=lambda x: x[1])[0]

def td_learning(env, approximator, num_episodes=50000, alpha=0.001, gamma=0.99, epsilon=0.1):
    """
    Train the 2048 agent using TD-Learning with after-state learning.
    """
    final_scores = []
    success_rates = []

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        previous_score = 0
        max_tile = 0
        trajectory = []

        while not done:
            # Find legal moves
            legal_moves = [a for a in range(4) if env.is_move_legal(a)]
            
            if not legal_moves:
                break

            # Choose action using epsilon-greedy strategy
            action = choose_best_action(env, approximator, legal_moves, gamma)

            # Take the action
            next_state, new_score, done, info = env.step(action)
            
            # Calculate incremental reward
            incremental_reward = new_score - previous_score
            previous_score = new_score
            
            # Track maximum tile
            max_tile = max(max_tile, np.max(next_state))

            # Store transition: 
            # (current_state, action, reward, state_before_random, next_state)
            trajectory.append((
                state, 
                action, 
                incremental_reward, 
                info["state_before_random"], 
                next_state
            ))

            # Update state for next iteration
            state = next_state

        # Perform backward TD learning
        next_value = 0
        for s, a, r, s_after, s_next in reversed(trajectory):
            # Compute TD error
            delta = r + gamma * next_value - approximator.value(s_after)
            
            # Update weights
            approximator.update(s_after, delta, alpha)
            
            # Update next value for next iteration
            next_value = approximator.value(s_after)

        # Record episode statistics
        final_scores.append(previous_score)
        success_rates.append(1 if max_tile >= 2048 else 0)

        # Print progress periodically
        if (episode + 1) % 100 == 0:
            avg_score = np.mean(final_scores[-100:])
            success_rate = np.mean(success_rates[-100:])
            print(f"Episode {episode+1}/{num_episodes} | "
                  f"Avg Score: {avg_score:.2f} | "
                  f"Success Rate: {success_rate:.2f}")

    return final_scores

# Define patterns (same as before)
patterns = [
    # horizontal 4-tuples
    [(0, 0), (0, 1), (0, 2), (0, 3)],
    [(1, 0), (1, 1), (1, 2), (1, 3)],
    [(2, 0), (2, 1), (2, 2), (2, 3)],
    [(3, 0), (3, 1), (3, 2), (3, 3)],
    
    # vertical 4-tuples
    [(0, 0), (1, 0), (2, 0), (3, 0)],
    [(0, 1), (1, 1), (2, 1), (3, 1)],
    [(0, 2), (1, 2), (2, 2), (3, 2)],
    [(0, 3), (1, 3), (2, 3), (3, 3)],
    
    # all 4-tile squares
    [(0, 0), (0, 1), (1, 0), (1, 1)],
    [(0, 1), (0, 2), (1, 1), (1, 2)],
    [(0, 2), (0, 3), (1, 2), (1, 3)],
    [(1, 0), (1, 1), (2, 0), (2, 1)],
    [(1, 1), (1, 2), (2, 1), (2, 2)],
    [(1, 2), (1, 3), (2, 2), (2, 3)],
    [(2, 0), (2, 1), (3, 0), (3, 1)],
    [(2, 1), (2, 2), (3, 1), (3, 2)],
    [(2, 2), (2, 3), (3, 2), (3, 3)],
]

# Create approximator and environment
approximator = NTupleApproximator(board_size=4, patterns=patterns)
env = Game2048Env()

# Run TD-Learning training
final_scores = td_learning(
    env, 
    approximator, 
    num_episodes=3000,  # Can increase for better learning
    alpha=0.01,  # Smaller learning rate for stability
    gamma=0.99, 
    epsilon=0.1
)