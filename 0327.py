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


# -------------------------------
# TODO: Define transformation functions (rotation and reflection), i.e., rot90, rot180, ..., etc.
# -------------------------------

# def rot90(pattern, board_size):
#     """Rotate pattern 90 degrees clockwise"""
#     return [(y, board_size - 1 - x) for (x, y) in pattern]

# def rot180(pattern, board_size):
#     """Rotate pattern 180 degrees"""
#     return [(board_size - 1 - x, board_size - 1 - y) for (x, y) in pattern]

# def rot270(pattern, board_size):
#     """Rotate pattern 270 degrees clockwise"""
#     return [(board_size - 1 - y, x) for (x, y) in pattern]

# def reflect_horizontal(pattern, board_size):
#     """Reflect pattern horizontally"""
#     return [(x, board_size - 1 - y) for (x, y) in pattern]

# def reflect_vertical(pattern, board_size):
#     """Reflect pattern vertically"""
#     return [(board_size - 1 - x, y) for (x, y) in pattern]

# def reflect_diagonal(pattern, board_size):
#     """Reflect pattern along the main diagonal"""
#     return [(y, x) for (x, y) in pattern]

# def reflect_antidiagonal(pattern, board_size):
#     """Reflect pattern along the anti-diagonal"""
#     return [(board_size - 1 - y, board_size - 1 - x) for (x, y) in pattern]


class NTupleApproximator:
    def __init__(self, board_size, patterns):
        """
        Initializes the N-Tuple approximator.
        Hint: you can adjust these if you want
        """
        self.board_size = board_size
        self.patterns = patterns
        # Create a weight dictionary for each pattern (shared within a pattern group)
        self.weights = [defaultdict(float) for _ in patterns]
        # Generate symmetrical transformations for each pattern
        # self.symmetry_patterns = []
        # for pattern in self.patterns:
        #     syms = self.generate_symmetries(pattern)
        #     for syms_ in syms:
        #         self.symmetry_patterns.append(syms_)
        self.patterns = []
        for pattern in patterns:
            self.patterns.append(pattern)
            
            
    # def generate_symmetries(self, pattern):
    #     # TODO: Generate 8 symmetrical transformations of the given pattern.
    #     symmetries = []
    #     # Original pattern
    #     symmetries.append(pattern)
    #     # Rotations
    #     rot90_pattern = rot90(pattern, self.board_size - 1)
    #     symmetries.append(rot90_pattern)
    #     rot180_pattern = rot180(pattern, self.board_size - 1)
    #     symmetries.append(rot180_pattern)
    #     rot270_pattern = rot270(pattern, self.board_size - 1)
    #     symmetries.append(rot270_pattern)
    #     # Reflections
    #     refl_h = reflect_horizontal(pattern, self.board_size - 1)
    #     symmetries.append(refl_h)
    #     refl_v = reflect_vertical(pattern, self.board_size - 1)
    #     symmetries.append(refl_v)
    #     refl_d = reflect_diagonal(pattern, self.board_size - 1)
    #     symmetries.append(refl_d)
    #     refl_ad = reflect_antidiagonal(pattern, self.board_size - 1)
    #     symmetries.append(refl_ad)
        
    #     # Remove duplicates
    #     unique_symmetries = []
    #     for sym in symmetries:
    #         if sym not in unique_symmetries:
    #             unique_symmetries.append(sym)
        
    #     return unique_symmetries


    def tile_to_index(self, tile):
        """
        Converts tile values to an index for the lookup table.
        """
        if tile == 0:
            return 0
        else:
            return int(math.log(tile, 2))

    def get_feature(self, board, coords):
        # Extract tile values from the board based on the given coordinates and convert them into a feature tuple.
        feature = []
        for x, y in coords:
            tile_value = board[x, y]  # Direct 2D indexing
            feature.append(self.tile_to_index(tile_value))
        return tuple(feature)


    def value(self, board):
        # TODO: Estimate the board value: sum the evaluations from all patterns.
        total_value = 0
        
        for i, pattern in enumerate(self.patterns):
            # Get the pattern index in the original patterns list
            # pattern_idx = i % len(self.patterns)
            # Extract feature and lookup its weight
            feature = self.get_feature(board, pattern)
            total_value += self.weights[i][feature]
        
        return total_value

    def update(self, board, delta, alpha):
        # TODO: Update weights based on the TD error.
        for i, pattern in enumerate(self.patterns):
            # Get the pattern index in the original patterns list
            # pattern_idx = i % len(self.patterns)
            # Extract feature
            feature = self.get_feature(board, pattern)
            # Update weight
            self.weights[i][feature] += alpha * delta # delta is the TD error

def choose_best_action(env, approximator, legal_moves, gamma):
    action_values = []
    for a in legal_moves:
        # Simulate the move
        test_env = copy.deepcopy(env)
        prev_score = test_env.score
        sim_next_state, sim_new_score, _, sim_info = test_env.step(a)
        sim_incre_reward = sim_new_score - prev_score # sim_previous_score is zero in simulation
        sim_next_state_before_random = sim_info["state_before_random"]
        # Include reward in the value calculation, but use state before random tile
        action_values.append((a, sim_incre_reward + gamma * approximator.value(sim_next_state_before_random)))
    
    # Choose action with highest value
    best_action = max(action_values, key=lambda x: x[1])[0]
    return best_action

def td_learning(env, approximator, num_episodes=50000, alpha=0.01, gamma=0.99, epsilon=0.1):
    """
    Trains the 2048 agent using TD-Learning.

    Args:
        env: The 2048 game environment.
        approximator: NTupleApproximator instance.
        num_episodes: Number of training episodes.
        alpha: Learning rate.
        gamma: Discount factor.
        epsilon: Epsilon-greedy exploration rate.
    """
    final_scores = []
    success_flags = []

    for episode in range(num_episodes):
        state = env.reset()
        # prev_state_before_random = state.copy()
        trajectory = []  # Store (state, action, incremental_reward, state_before_random, state_after_random) tuples
        previous_score = 0
        done = False
        max_tile = np.max(state)
        # legal_moves = [a for a in range(4) if env.is_move_legal(a)]

        while not done:
            legal_moves = [a for a in range(4) if env.is_move_legal(a)]
            if not legal_moves:
                break
            if random.random() < epsilon:
                # Exploration: choose a random legal move
                action = random.choice(legal_moves)
            else:
                # Exploitation: choose the best action based on current values
                action = choose_best_action(env, approximator, legal_moves, gamma)


            next_state, new_score, _, info = env.step(action)
            next_state_before_random = info["state_before_random"]
            # next_score_before_random = info["score_before_random"]
            incremental_reward = new_score - previous_score
            previous_score = new_score
            max_tile = max(max_tile, np.max(next_state))

            # Store transition in trajectory with all state information
            trajectory.append((state, action, incremental_reward, next_state_before_random, next_state))

            # Update previous states for the next iteration
            # prev_state_before_random = state_before_random
            state = next_state

        # # Learn from trajectory in reverse order (except terminal transition)
        # if len(trajectory) > 1:  # Only if we have more than one transition
        #     # First, handle the terminal transition
        #     last_state, last_action, last_reward, last_state_before_random, terminal_state = trajectory[-1]
        #     next_value = 0  # Terminal state has value 0
        #     current_value = approximator.value(last_state)
        #     td_error = last_reward + gamma * next_value - current_value
        #     approximator.update(last_state_before_random, td_error, alpha)

        # Then handle all other transitions in reverse order
                # Then handle all transitions in reverse order (except the terminal one)
        for s, a, r, s_after, s_next in trajectory[:-1][::-1]:
            # Create a temporary environment for s_next
            test_env = Game2048Env()
            test_env.board = s_next.copy()  # initialize with the s_next board state
            
            # Determine legal moves in the simulated environment
            legal_moves_next = [act for act in range(4) if test_env.is_move_legal(act)]
            if legal_moves_next:
                # Choose the best action from s_next
                a_next = choose_best_action(test_env, approximator, legal_moves_next, gamma)
                try:
                    # Record current score before the simulated move
                    score_before = test_env.score
                    # Take the action; step returns the new score, among other things
                    sim_next_state, sim_new_score, done, sim_info = test_env.step(a_next)
                    # Immediate reward: the change in score during simulation
                    r_next = sim_new_score - score_before
                    s_after_next = sim_info["state_before_random"]
                    v_after_next = approximator.value(s_after_next)
                except Exception:
                    r_next = 0
                    v_after_next = 0
            else:
                r_next = 0
                v_after_next = 0

            # TD error: target value is the immediate reward from s_next and the approximator value 
            # of the state before the random tile; update based on s_after
            delta = r_next + v_after_next - approximator.value(s_after)
            approximator.update(s_after, delta, alpha)

        final_scores.append(env.score)
        success_flags.append(1 if max_tile >= 2048 else 0)

        if (episode + 1) % 100 == 0:
            avg_score = np.mean(final_scores[-100:])
            success_rate = np.sum(success_flags[-100:]) / 100
            print(f"Episode {episode+1}/{num_episodes} | Avg Score: {avg_score:.2f} | Success Rate: {success_rate:.2f}")

    return final_scores


# TODO: Define your own n-tuple patterns
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
    
    # # Special snake patterns
    # [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2), (3, 2)]
]

approximator = NTupleApproximator(board_size=4, patterns=patterns)

env = Game2048Env()

# Run TD-Learning training
# Note: To achieve significantly better performance, you will likely need to train for over 100,000 episodes.
# However, to quickly verify that your implementation is working correctly, you can start by running it for 1,000 episodes before scaling up.
final_scores = td_learning(env, approximator, num_episodes=20000, alpha=0.1, gamma=0.99, epsilon=0.1)