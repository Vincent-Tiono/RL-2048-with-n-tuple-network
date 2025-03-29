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

class IllegalAction(Exception):
    pass


class GameOver(Exception):
    pass

class Game2048Env(gym.Env):
    def __init__(self, board=None):
        super(Game2048Env, self).__init__()

        self.size = 4
        if board is not None:
            self.board = board
        else:
            self.board = [0] * 16  # Use 1D list instead of 2D array
        self.score = 0

        # Action space: 0: up, 1: down, 2: left, 3: right
        self.action_space = spaces.Discrete(4)
        self.actions = ["up", "down", "left", "right"]

        self.last_move_valid = True

        self.reset()

    def reset(self):
        self.board = [0] * 16
        self.score = 0
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
                    reward += hold * 2
                    r.append(hold * 2)
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
            if not original_row == new_row:
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
            if not original_col == new_col:
                moved = True
        return moved

    def rotate(self):
        """Rotate the board 90 degrees clockwise inplace."""
        size = 4
        b = []
        for i in range(size):
            b.extend(self.board[i::4][::-1])
        self.board = b
        return self

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

    def step(self, action):
        assert self.action_space.contains(action), "Invalid action"
        
        # Store current score to calculate immediate reward
        old_score = self.score
        
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

        if moved:
            self.add_random_tile()

        done = self.is_game_over()
        
        # Calculate immediate reward as difference in scores
        immediate_reward = self.score - old_score

        return self.board, self.score, immediate_reward, done, {}

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
                col = temp_board[j::4]
                new_col = self.simulate_row_move(col)
                temp_board[j::4] = new_col
        elif action == 1:  # Move down
            for j in range(self.size):
                col = temp_board[j::4][::-1]
                new_col = self.simulate_row_move(col)
                temp_board[j::4] = new_col[::-1]
        elif action == 2:  # Move left
            for i in range(self.size):
                row = temp_board[i * 4:i * 4 + 4]
                temp_board[i * 4:i * 4 + 4] = self.simulate_row_move(row)
        elif action == 3:  # Move right
            for i in range(self.size):
                row = temp_board[i * 4:i * 4 + 4][::-1]
                new_row = self.simulate_row_move(row)
                temp_board[i * 4:i * 4 + 4] = new_row[::-1]
        else:
            raise ValueError("Invalid action")
        return not np.array_equal(self.board, temp_board)
    
    def copyboard(self):
        """Create a deep copy of the 2D board"""
        return self.board.copy()  # numpy's copy() creates a deep copy of the array

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
        """Extract tile values from the board based on the given coordinates and convert them into a feature index.
        Ensures consistent spatial processing by maintaining the order of coordinates.
        """
        # Convert 2D coordinates to values in a consistent order
        values = []
        for x, y in coords:
            values.append(board[x * 4 + y])
        
        # Process values in a consistent order (left-to-right, top-to-bottom)
        k = 1
        n = 0
        for v in values:
            if v == 0:
                idx = 0
            else:
                idx = int(math.log(v, 2))
            if idx >= 15:
                raise ValueError(f"Value {v} is too large")
            n += idx * k
            k *= 15
        return n

    def value(self, board, delta=None):
        """
        Estimate the board value: expected value of the evaluations from all patterns.
        """
        vals = []
        for pattern_idx, pattern in enumerate(self.patterns):
            feature = self.get_feature(board, pattern)
            if delta is not None:
                self.weights[pattern_idx][feature] += delta
            vals.append(self.weights[pattern_idx][feature])
        return np.mean(vals)

    # def update(self, board, delta, alpha):
    #     """Update weights based on the TD error."""
    #     self.value(board, alpha * delta)  # Let value method handle the update

    def evaluate(self, s, a):
        """Return expected total rewards of performing 
        action (a) on the given board state (s)"""
        # Get afterstate and reward
        b = Game2048Env(s)
        try:
            # r = b.act(a)
            _, _, r, _, _ = b.step(a)
            s_after = b.copyboard() # s_after is afterstate, ie. the state after the action before adding a random tile
        except IllegalAction:
            return 0
        return r + self.value(s_after)

    def best_action(self, s, epsilon=0.1):
        """Returns the action with the highest expected total rewards on the state (s)"""
        # Exploration: choose random action with probability epsilon
        if random.random() < epsilon:
            return random.randint(0, 3)
            
        # Exploitation: choose best action
        a_best = None
        r_best = -1
        # 0: up, 1: down, 2: left, 3: right
        for a in range(4):
            r = self.evaluate(s, a)
            if r > r_best:
                r_best = r
                a_best = a
        return a_best

    def learn(self, s, a, r, s_after, s_next, alpha=0.1):
        """
        Learn from a transition experience by updating the belief
        on the after state (s_after) towards 
        the sum of the next transition rewards (r_next) and
        the belief on the next after state (s_after_next).
        ie. val - pred = (value of s_after_next + r_next) - value of s_after
        """
        # Get best action for next state
        a_next = self.best_action(s_next)
        b = Game2048Env(s_next)
        try:
            _, _, r_next, _, _ = b.step(a_next)
            s_after_next = b.copyboard()
            v_after_next = self.value(s_after_next)
        except IllegalAction:
            r_next = 0
            v_after_next = 0
        
        delta = r_next + v_after_next - self.value(s_after)
        self.value(s_after, alpha * delta)
    

from collections import namedtuple

"""
Vocabulary
--------------

Transition: A Transition shows how a board transformed from a state to the next state. It contains:
- s: The board state before the action
- a: The action performed
- r: The reward received
- s_after: The board's "after state" after applying the action
- s_next: The board's "next state" after adding a random tile
- score: The score after the action
- max_tile: The maximum tile value on the board
- empty_cells: Number of empty cells
- merge_count: Number of merges in this transition
- move_valid: Whether the move was valid

Gameplay: A series of transitions on the board (transition_history). Also reports:
- transition_history: List of all transitions
- game_reward: Total reward of playing the game
- max_tile: Maximum tile reached
- num_moves: Total number of moves made
- final_score: Final score achieved
- empty_cells_history: History of empty cells count
- merge_history: History of merge counts
"""
Transition = namedtuple("Transition", "s, a, r, s_after, s_next, score, max_tile, empty_cells, merge_count, move_valid")
Gameplay = namedtuple("Gameplay", "transition_history game_reward max_tile num_moves final_score empty_cells_history merge_history")

def count_empty_cells(board):
    """Count number of empty cells (zeros) in the board"""
    return sum(1 for x in board if x == 0)

def count_merges(board, after_board):
    """Count number of merges between two board states"""
    merges = 0
    for i in range(len(board)):
        if board[i] != 0 and board[i] == after_board[i]:
            merges += 1
    return merges

def play(agent, board, spawn_random_tile=False):
    "Return a gameplay of playing the given (board) until terminal states."
    b = Game2048Env(board)
    r_game = 0  # total reward
    a_cnt = 0  # number of actions
    transition_history = []  # history of transitions
    empty_cells_history = []  # history of empty cells
    merge_history = []  # history of merge counts
    
    while True:
        a_best = agent.best_action(b.board)
        s = b.copyboard()
        try:
            # Get current state info
            current_empty_cells = count_empty_cells(s)
            current_max_tile = np.max(s)
            
            # Take action
            _, _, r, _, _ = b.step(a_best)
            r_game += r
            s_after = b.copyboard()
            
            # Count merges
            merge_count = count_merges(s, s_after)
            
            # Add random tile
            b.add_random_tile()
            s_next = b.copyboard()
            
            # Store transition with additional info
            transition = Transition(
                s=s,
                a=a_best,
                r=r,
                s_after=s_after,
                s_next=s_next,
                score=b.score,
                max_tile=current_max_tile,
                empty_cells=current_empty_cells,
                merge_count=merge_count,
                move_valid=True
            )
            
            # Update histories
            empty_cells_history.append(current_empty_cells)
            merge_history.append(merge_count)
            
        except (IllegalAction, GameOver) as e:
            # Store final transition with None values for invalid states
            transition = Transition(
                s=s,
                a=a_best,
                r=None,
                s_after=None,
                s_next=None,
                score=b.score,
                max_tile=np.max(s),
                empty_cells=count_empty_cells(s),
                merge_count=0,
                move_valid=False
            )
            break
        finally:
            a_cnt += 1
            transition_history.append(transition)
    
    gp = Gameplay(
        transition_history=transition_history,
        game_reward=r_game,
        max_tile=np.max(b.board),
        num_moves=a_cnt,
        final_score=b.score,
        empty_cells_history=empty_cells_history,
        merge_history=merge_history
    )
    learn_from_gameplay(agent, gp)
    return gp


def learn_from_gameplay(agent, gp, alpha=0.1):
    "Learn transitions in reverse order except the terminal transition"
    for tr in gp.transition_history[:-1][::-1]:
        agent.learn(tr.s, tr.a, tr.r, tr.s_after, tr.s_next, alpha=alpha)


# def td_learning(env, approximator, num_episodes=50000, alpha=0.01, gamma=0.99, epsilon=0.1):
#     """
#     Trains the 2048 agent using TD-Learning with afterstate learning.

#     Args:
#         env: The 2048 game environment.
#         approximator: NTupleApproximator instance.
#         num_episodes: Number of training episodes.
#         alpha: Learning rate.
#         gamma: Discount factor.
#         epsilon: Epsilon-greedy exploration rate.
#     """
#     final_scores = []
#     success_flags = []

#     for episode in range(num_episodes):
#         state = env.reset()
#         trajectory = []  # Store transitions for reverse-order learning
#         done = False
#         max_tile = np.max(state)

#         while not done:
#             # Action selection
#             if random.random() < epsilon:
#                 # Exploration: choose a random legal move
#                 legal_moves = [a for a in range(4) if env.is_move_legal(a)]
#                 if not legal_moves:
#                     break
#                 action = random.choice(legal_moves)
#             else:
#                 # Exploitation: choose the best action based on current values
#                 action = approximator.best_action(state)
#                 if action is None:
#                     break

#             # Get afterstate and reward
#             afterstate, afterstate_score, moved = env.get_afterstate(action)
#             if not moved:
#                 break

#             # Take action and get next state
#             next_state, new_score, done, info = env.step(action)
            
#             # Store transition
#             trajectory.append({
#                 'state': state.copy(),
#                 'action': action,
#                 'reward': afterstate_score,
#                 'afterstate': afterstate.copy(),
#                 'next_state': next_state.copy()
#             })

#             state = next_state
#             max_tile = max(max_tile, np.max(state))

#         # Learn from trajectory in reverse order
#         for transition in reversed(trajectory[:-1]):  # Skip the last transition
#             approximator.learn(
#                 transition['state'],
#                 transition['action'],
#                 transition['reward'],
#                 transition['afterstate'],
#                 transition['next_state'],
#                 env,
#                 alpha=alpha
#             )

#         final_scores.append(env.score)
#         success_flags.append(1 if max_tile >= 2048 else 0)

#         if (episode + 1) % 100 == 0:
#             avg_score = np.mean(final_scores[-100:])
#             success_rate = np.sum(success_flags[-100:]) / 100
#             print(f"Episode {episode+1}/{num_episodes} | Avg Score: {avg_score:.2f} | Success Rate: {success_rate:.2f} | Max Tile: {max_tile}")

#     return final_scores

# TODO: Define your own n-tuple patterns
patterns = [
    # horizontal 4-tuples
    [(0,0), (0,1), (0,2), (0,3)],
    [(1,0), (1,1), (1,2), (1,3)],
    [(2,0), (2,1), (2,2), (2,3)],
    [(3,0), (3,1), (3,2), (3,3)],
    # vertical 4-tuples
    [(0,0), (1,0), (2,0), (3,0)],
    [(0,1), (1,1), (2,1), (3,1)],
    [(0,2), (1,2), (2,2), (3,2)],
    [(0,3), (1,3), (2,3), (3,3)],
    # all 4-tile squares
    [(0,0), (0,1), (1,0), (1,1)],
    [(1,0), (1,1), (2,0), (2,1)],
    [(2,0), (2,1), (3,0), (3,1)],
    [(0,1), (0,2), (1,1), (1,2)],
    [(1,1), (1,2), (2,1), (2,2)],
    [(2,1), (2,2), (3,1), (3,2)],
    [(0,2), (0,3), (1,2), (1,3)],
    [(1,2), (1,3), (2,2), (2,3)],
    [(2,2), (2,3), (3,2), (3,3)]
]


def td_learning(approximator, num_sessions, num_episodes):
    n_games = 0
    for i_session in range(num_sessions):
        gameplays = []
        for i_episode in range(num_episodes):
            gp = play(approximator, None, spawn_random_tile=True)
            gameplays.append(gp)
            n_games += 1
        n2048 = sum([1 for gp in gameplays if gp.max_tile == 2048])
        mean_maxtile = np.mean([gp.max_tile for gp in gameplays])
        maxtile = np.max([gp.max_tile for gp in gameplays])
        mean_gamerewards = np.mean([gp.game_reward for gp in gameplays])
        print(f"Game# {n_games}, tot. {n_games/1000}k games, " + "mean game rewards {:.0f}, mean max tile {:.0f}, 2048 rate {:.0%}, maxtile {:.0f}".format(
            mean_gamerewards, mean_maxtile, n2048 / len(gameplays), maxtile))
        
approximator = NTupleApproximator(board_size=4, patterns=patterns) #approximator is the agent
# env = Game2048Env()
td_learning(approximator, num_sessions=20, num_episodes=100)
        


# # Run TD-Learning training
# # Note: To achieve significantly better performance, you will likely need to train for over 100,000 episodes.
# # However, to quickly verify that your implementation is working correctly, you can start by running it for 1,000 episodes before scaling up.
# final_scores = td_learning(env, approximator, num_episodes=20000, alpha=0.1, gamma=0.99, epsilon=0.1)

# Training parameters
LEARNING_RATE = 0.01  # Reduced from 0.1 for more stable learning
GAMMA = 0.99  # Increased from 0.95 for better long-term planning
EPSILON_START = 0.2  # Increased from 0.1 for more exploration
EPSILON_END = 0.01
EPSILON_DECAY = 0.999  # Slower decay for better exploration
TRAINING_EPISODES = 50000  # Increased from 10000 for more training
EVALUATION_EPISODES = 100
EVALUATION_INTERVAL = 100  # Evaluate more frequently