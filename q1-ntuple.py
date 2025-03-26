import numpy as np
import random
import gym
from gym import spaces
import matplotlib.pyplot as plt
from collections import namedtuple

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

# Define the same namedtuples as in main.py
Transition = namedtuple("Transition", "s, a, r, s_after, s_next")
Gameplay = namedtuple("Gameplay", "transition_history game_reward max_tile")

class Game2048Env(gym.Env):
    def __init__(self):
        super(Game2048Env, self).__init__()

        self.size = 4
        self.board = [0] * 16  # Change to 1D list like main.py
        self.score = 0
        self.last_score = 0

        # Action space: 0: up, 1: down, 2: left, 3: right
        self.action_space = spaces.Discrete(4)
        self.actions = ["up", "down", "left", "right"]

        self.last_move_valid = True

        self.reset()

    def reset(self):
        self.board = [0] * 16
        self.score = 0
        self.last_score = 0
        self.add_random_tile()
        self.add_random_tile()
        return self.board

    def add_random_tile(self):
        empty_cells = [i for i, v in enumerate(self.board) if v == 0]
        if empty_cells:
            idx = random.choice(empty_cells)
            self.board[idx] = 2 if random.random() < 0.9 else 4

    def compress(self, row):
        return [x for x in row if x != 0]

    def merge(self, row):
        row = self.compress(row)
        reward = 0
        r = []
        hold = -1
        while len(row) > 0:
            v = row.pop(0)
            if hold != -1:
                if hold == v:
                    reward = reward + (2 ** (hold + 1))
                    r.append(hold + 1)
                    hold = -1
                else:
                    r.append(hold)
                    hold = v
            else:
                hold = v
        if hold != -1:
            r.append(hold)
            hold = -1
        while len(r) < 4:
            r.append(0)
        return reward, r

    def move_left(self):
        moved = False
        for i in range(4):
            idx = i * 4
            row = self.board[idx:idx + 4]
            original_row = row.copy()
            row_reward, new_row = self.merge(row)
            self.score += row_reward
            self.board[idx:idx + 4] = new_row
            if not original_row == new_row:
                moved = True
        return moved

    def move_right(self):
        moved = False
        for i in range(4):
            idx = i * 4
            row = self.board[idx:idx + 4][::-1]
            original_row = row.copy()
            row_reward, new_row = self.merge(row)
            self.score += row_reward
            self.board[idx:idx + 4] = new_row[::-1]
            if not original_row == new_row[::-1]:
                moved = True
        return moved

    def move_up(self):
        moved = False
        for j in range(4):
            col = self.board[j::4]
            original_col = col.copy()
            col_reward, new_col = self.merge(col)
            self.score += col_reward
            for i in range(4):
                self.board[i * 4 + j] = new_col[i]
            if not original_col == new_col:
                moved = True
        return moved

    def move_down(self):
        moved = False
        for j in range(4):
            col = self.board[j::4][::-1]
            original_col = col.copy()
            col_reward, new_col = self.merge(col)
            self.score += col_reward
            for i in range(4):
                self.board[i * 4 + j] = new_col[::-1][i]
            if not original_col == new_col[::-1]:
                moved = True
        return moved

    def is_game_over(self):
        if 0 in self.board:
            return False
        for i in range(4):
            for j in range(3):
                if self.board[i * 4 + j] == self.board[i * 4 + j + 1]:
                    return False
                if self.board[j * 4 + i] == self.board[(j + 1) * 4 + i]:
                    return False
        return True

    def get_afterstate(self, action):
        temp_board = self.board.copy()
        temp_score = self.score
        temp_last_score = self.last_score
        
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

        afterstate = self.board.copy()
        afterstate_score = self.score
        merge_reward = afterstate_score - temp_score  # Calculate merge reward
        
        # Restore original state
        self.board = temp_board
        self.score = temp_score
        self.last_score = temp_last_score
        
        return afterstate, merge_reward, moved

    def step(self, action):
        assert self.action_space.contains(action), "Invalid action"

        # Get afterstate
        afterstate, afterstate_score, moved = self.get_afterstate(action)
        self.last_move_valid = moved

        if moved:
            # Update board to afterstate
            self.board = afterstate
            self.last_score = self.score
            self.score = afterstate_score
            # Add random tile
            self.add_random_tile()

        done = self.is_game_over()

        # Return the merge reward directly from the last action
        reward = afterstate_score - self.last_score

        return self.board, reward, done, {"afterstate": afterstate, "afterstate_score": afterstate_score}

    def render(self, mode="human", action=None):
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, self.size - 0.5)
        ax.set_ylim(-0.5, self.size - 0.5)

        for i in range(self.size):
            for j in range(self.size):
                value = self.board[i * 4 + j]
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
        # Convert row to list if it's not already
        row = list(row)
        # Remove zeros
        new_row = [x for x in row if x != 0]
        # Pad with zeros
        while len(new_row) < 4:
            new_row.append(0)
        # Merge tiles
        for i in range(len(new_row) - 1):
            if new_row[i] == new_row[i + 1] and new_row[i] != 0:
                new_row[i] *= 2
                new_row[i + 1] = 0
        # Remove zeros again
        new_row = [x for x in new_row if x != 0]
        # Pad with zeros
        while len(new_row) < 4:
            new_row.append(0)
        return new_row

    def is_move_legal(self, action):
        temp_board = self.board.copy()

        if action == 0:  # Move up
            for j in range(self.size):
                col = temp_board[j::self.size]
                new_col = self.simulate_row_move(col)
                for i in range(4):
                    temp_board[i * 4 + j] = new_col[i]
        elif action == 1:  # Move down
            for j in range(self.size):
                col = temp_board[j::self.size][::-1]
                new_col = self.simulate_row_move(col)
                for i in range(4):
                    temp_board[i * 4 + j] = new_col[::-1][i]
        elif action == 2:  # Move left
            for i in range(self.size):
                row = temp_board[i*self.size:i*self.size+self.size]
                new_row = self.simulate_row_move(row)
                temp_board[i*self.size:i*self.size+self.size] = new_row
        elif action == 3:  # Move right
            for i in range(self.size):
                row = temp_board[i*self.size:i*self.size+self.size][::-1]
                new_row = self.simulate_row_move(row)
                temp_board[i*self.size:i*self.size+self.size] = new_row[::-1]
        else:
            raise ValueError("Invalid action")
        return self.board != temp_board

import copy
import random
import math
import numpy as np
from collections import defaultdict


# -------------------------------
# TODO: Define transformation functions (rotation and reflection), i.e., rot90, rot180, ..., etc.
# -------------------------------

def rot90(pattern, board_size):
    """Rotate pattern 90 degrees clockwise"""
    return [(y, board_size - 1 - x) for (x, y) in pattern]

def rot180(pattern, board_size):
    """Rotate pattern 180 degrees"""
    return [(board_size - 1 - x, board_size - 1 - y) for (x, y) in pattern]

def rot270(pattern, board_size):
    """Rotate pattern 270 degrees clockwise"""
    return [(board_size - 1 - y, x) for (x, y) in pattern]

def reflect_horizontal(pattern, board_size):
    """Reflect pattern horizontally"""
    return [(x, board_size - 1 - y) for (x, y) in pattern]

def reflect_vertical(pattern, board_size):
    """Reflect pattern vertically"""
    return [(board_size - 1 - x, y) for (x, y) in pattern]

def reflect_diagonal(pattern, board_size):
    """Reflect pattern along the main diagonal"""
    return [(y, x) for (x, y) in pattern]

def reflect_antidiagonal(pattern, board_size):
    """Reflect pattern along the anti-diagonal"""
    return [(board_size - 1 - y, board_size - 1 - x) for (x, y) in pattern]


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
        self.symmetry_patterns = []
        for pattern in self.patterns:
            syms = self.generate_symmetries(pattern)
            for syms_ in syms:
                self.symmetry_patterns.append(syms_)

    def generate_symmetries(self, pattern):
        # TODO: Generate 8 symmetrical transformations of the given pattern.
        symmetries = []
        # Original pattern
        symmetries.append(pattern)
        # Rotations
        rot90_pattern = rot90(pattern, self.board_size - 1)
        symmetries.append(rot90_pattern)
        rot180_pattern = rot180(pattern, self.board_size - 1)
        symmetries.append(rot180_pattern)
        rot270_pattern = rot270(pattern, self.board_size - 1)
        symmetries.append(rot270_pattern)
        # Reflections
        refl_h = reflect_horizontal(pattern, self.board_size - 1)
        symmetries.append(refl_h)
        refl_v = reflect_vertical(pattern, self.board_size - 1)
        symmetries.append(refl_v)
        refl_d = reflect_diagonal(pattern, self.board_size - 1)
        symmetries.append(refl_d)
        refl_ad = reflect_antidiagonal(pattern, self.board_size - 1)
        symmetries.append(refl_ad)
        
        # Remove duplicates
        unique_symmetries = []
        for sym in symmetries:
            if sym not in unique_symmetries:
                unique_symmetries.append(sym)
        
        return unique_symmetries

    def tile_to_index(self, tile):
        """
        Converts tile values to an index for the lookup table.
        """
        if tile == 0:
            return 0
        else:
            return int(math.log(tile, 2))

    def get_feature(self, board, coords):
        # Extract tile values from the board based on the given coordinates and convert them into a feature tuple
        feature = []
        for x, y in coords:
            # Convert 2D coordinates to 1D index
            idx = x * self.board_size + y
            tile_value = board[idx]  # Direct list access instead of flatten()
            feature.append(self.tile_to_index(tile_value))
        return tuple(feature)

    def value(self, board, delta=None):
        """Return the expected total future rewards of the board.
        Updates the weights if a delta is given and return the updated value.
        """
        vals = []
        for i, pattern in enumerate(self.symmetry_patterns):
            pattern_idx = i % len(self.patterns)
            feature = self.get_feature(board, pattern)
            if delta is not None:
                self.weights[pattern_idx][feature] += delta
            vals.append(self.weights[pattern_idx][feature])
        return np.mean(vals)  # Use mean instead of sum

    def update(self, board, delta, alpha):
        """Update weights based on the TD error."""
        self.value(board, alpha * delta)  # Let value method handle the update

    def evaluate(self, board, action, env):
        """Evaluate an action on the given board state"""
        # Get afterstate and reward
        afterstate, afterstate_score, moved = env.get_afterstate(action)
        if not moved:
            return 0
        return afterstate_score + self.value(afterstate)

    def best_action(self, board, env):
        """Returns the action with the highest expected total rewards"""
        a_best = None
        r_best = -1
        for a in range(4):
            if env.is_move_legal(a):
                r = self.evaluate(board, a, env)
                if r > r_best:
                    r_best = r
                    a_best = a
        return a_best

    def learn(self, s, a, r, s_after, s_next, env, alpha=0.01):
        """Learn from a transition experience using afterstate learning"""
        # Get best action for next state
        a_next = self.best_action(s_next, env)
        if a_next is not None:
            # Get afterstate and reward for next action
            s_after_next, r_next, _ = env.get_afterstate(a_next)
            v_after_next = self.value(s_after_next)
        else:
            r_next = 0
            v_after_next = 0

        # Calculate TD error: r_next + v_after_next - v_after
        delta = r_next + v_after_next - self.value(s_after)
        
        # Update value function
        self.update(s_after, alpha * delta, alpha)

def play(env, approximator, board=None, spawn_random_tile=False):
    """Return a gameplay of playing the given (board) until terminal states."""
    if board is not None:
        env.board = board.copy()
    else:
        env.reset()
    
    r_game = 0
    a_cnt = 0
    transition_history = []
    
    while True:
        a_best = approximator.best_action(env.board, env)
        s = env.board.copy()
        reward = None  # Initialize reward as None
        s_after = None  # Initialize s_after as None
        s_next = None  # Initialize s_next as None
        
        try:
            # Get afterstate and reward
            afterstate, afterstate_score, moved = env.get_afterstate(a_best)
            if not moved:
                break
                
            # Take action and get next state
            next_state, reward, done, info = env.step(a_best)
            r_game += reward
            
            s_after = afterstate.copy()
            s_next = next_state.copy()
            
        except Exception as e:
            # game ends when agent makes illegal moves or board is full
            break
            
        finally:
            a_cnt += 1
            transition_history.append(
                Transition(s=s, a=a_best, r=reward, s_after=s_after, s_next=s_next)
            )
    
    gp = Gameplay(
        transition_history=transition_history,
        game_reward=r_game,
        max_tile=2 ** max(env.board),
    )
    return gp

def learn_from_gameplay(approximator, gp, env, alpha=0.1):
    """Learn transitions in reverse order except the terminal transition"""
    for tr in gp.transition_history[:-1][::-1]:
        approximator.learn(
            tr.s, tr.a, tr.r, tr.s_after, tr.s_next,
            env, alpha=alpha
        )

def td_learning(env, approximator, num_episodes=50000, alpha=0.01, gamma=0.99, epsilon=0.1):
    """
    Trains the 2048 agent using TD-Learning with afterstate learning.
    """
    final_scores = []
    success_flags = []

    for episode in range(num_episodes):
        # Use the play function to get a complete gameplay
        gp = play(env, approximator, spawn_random_tile=True)
        
        # Learn from the gameplay
        learn_from_gameplay(approximator, gp, env, alpha=alpha)
        
        final_scores.append(gp.game_reward)
        success_flags.append(1 if gp.max_tile >= 2048 else 0)

        if (episode + 1) % 100 == 0:
            avg_score = np.mean(final_scores[-100:])
            success_rate = np.sum(success_flags[-100:]) / 100
            print(f"Episode {episode+1}/{num_episodes} | Avg Score: {avg_score:.2f} | Success Rate: {success_rate:.2f} | Max Tile: {gp.max_tile}")

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
]

approximator = NTupleApproximator(board_size=4, patterns=patterns)

env = Game2048Env()

# Run TD-Learning training
# Note: To achieve significantly better performance, you will likely need to train for over 100,000 episodes.
# However, to quickly verify that your implementation is working correctly, you can start by running it for 1,000 episodes before scaling up.
final_scores = td_learning(env, approximator, num_episodes=20000, alpha=0.1, gamma=0.99, epsilon=0.1)