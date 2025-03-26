import random
import numpy as np

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

def action_name(a):
    return "UP DOWN LEFT RIGHT".split()[a]


class IllegalAction(Exception):
    pass


class GameOver(Exception):
    pass


def compress(row):
    "remove 0s in the list"
    return [x for x in row if x != 0]


def merge(row):
    row = compress(row)
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


from copy import copy


class Game2048Env:
    def __init__(self, board=None):
        """board is a list of 16 integers"""
        if board is not None:
            self.board = board
        else:
            self.reset()

    def reset(self):
        self.clear()
        self.board[random.choice(self.empty_tiles())] = 4 if np.random.rand() <= 0.1 else 2
        self.board[random.choice(self.empty_tiles())] = 4 if np.random.rand() <= 0.1 else 2

    def spawn_tile(self, random_tile=False):
        empty_tiles = self.empty_tiles()
        if len(empty_tiles) == 0:
            raise GameOver("Game2048Env is full. Cannot spawn any tile.")
        if random_tile:
            k = 4 if np.random.rand() <= 0.1 else 2
            self.board[random.choice(empty_tiles)] = k
        else:
            self.board[empty_tiles[0]] = 2

    def clear(self):
        self.board = [0] * 16

    def empty_tiles(self):
        return [i for (i, v) in enumerate(self.board) if v == 0]

    def display(self):
        def format_row(lst):
            s = ""
            for l in lst:
                s += " {:3d}".format(l)
            return s

        for row in range(4):
            idx = row * 4
            print(format_row(self.board[idx : idx + 4]))

    def act(self, a):
        original = self.board
        if a == LEFT:
            r = self.merge_to_left()
        if a == RIGHT:
            r = self.rotate().rotate().merge_to_left()
            self.rotate().rotate()
        if a == UP:
            r = self.rotate().rotate().rotate().merge_to_left()
            self.rotate()
        if a == DOWN:
            r = self.rotate().merge_to_left()
            self.rotate().rotate().rotate()
        if original == self.board:
            raise IllegalAction("Action did not move any tile.")
        return r

    def rotate(self):
        "Rotate the board inplace 90 degress clockwise."
        size = 4
        b = []
        for i in range(size):
            b.extend(self.board[i::4][::-1])
        self.board = b
        return self

    def merge_to_left(self):
        "merge board to the left, returns the reward for mering tiles"
        "Raises IllegalAction exception if the action does not move any tile."
        r = []
        board_reward = 0
        for nrow in range(4):
            idx = nrow * 4
            row = self.board[idx : idx + 4]
            row_reward, row = merge(row)
            board_reward = board_reward + row_reward
            r.extend(row)
        self.board = r
        return board_reward

    def copyboard(self):
        return copy(self.board)
    

import numpy as np
# from game import Game2048Env, UP, RIGHT, DOWN, LEFT, action_name
# from game import IllegalAction, GameOver


class nTupleNewrok:
    def __init__(self, tuples):
        self.TUPLES = tuples
        self.TARGET_VALUE = 32768  # Maximum value we expect to see
        self.LUTS = self.initialize_LUTS(self.TUPLES)

    def initialize_LUTS(self, tuples):
        LUTS = []
        for tp in tuples:
            # Calculate the number of possible values for each tuple
            # We need to account for values up to 2048 (2^11)
            # We'll use 12 as the base to be safe (0-11 for values 0,2,4,8,...,2048)
            LUTS.append(np.zeros(16 ** len(tp)))
        return LUTS

    def tuple_id(self, values):
        values = values[::-1]
        k = 1
        n = 0
        for v in values:
            # Convert actual value to index (0 for 0, 1 for 2, 2 for 4, etc.)
            if v == 0:
                idx = 0
            else:
                idx = int(np.log2(v))
            if idx >= 15:  # Safety check
                raise ValueError(f"Value {v} is too large")
            n += idx * k
            k *= 15
        return n

    def V(self, board, delta=None, debug=False):
        """Return the expected total future rewards of the board.
        Updates the LUTs if a delta is given and return the updated value.
        """
        if debug:
            print(f"V({board})")
        vals = []
        for i, (tp, LUT) in enumerate(zip(self.TUPLES, self.LUTS)):
            tiles = [board[i] for i in tp]
            tpid = self.tuple_id(tiles)
            if delta is not None:
                LUT[tpid] += delta
            v = LUT[tpid]
            if debug:
                print(f"LUTS[{i}][{tiles}]={v}")
            vals.append(v)
        return np.mean(vals)

    def evaluate(self, s, a):
        "Return expected total rewards of performing action (a) on the given board state (s)"
        b = Game2048Env(s)
        try:
            r = b.act(a)
            s_after = b.copyboard()
        except IllegalAction:
            return 0
        return r + self.V(s_after)

    def best_action(self, s):
        "returns the action with the highest expected total rewards on the state (s)"
        a_best = None
        r_best = -1
        for a in [UP, DOWN, LEFT, RIGHT]:  # Updated action order to match environment
            r = self.evaluate(s, a)
            if r > r_best:
                r_best = r
                a_best = a
        return a_best

    def learn(self, s, a, r, s_after, s_next, alpha=0.01, debug=False):
        """Learn from a transition experience by updating the belief
        on the after state (s_after) towards the sum of the next transition rewards (r_next) and
        the belief on the next after state (s_after_next).

        """
        a_next = self.best_action(s_next)
        b = Game2048Env(s_next)
        try:
            r_next = b.act(a_next)
            s_after_next = b.copyboard()
            v_after_next = self.V(s_after_next)
        except IllegalAction:
            r_next = 0
            v_after_next = 0

        delta = r_next + v_after_next - self.V(s_after)

        if debug:
            print("s_next")
            Game2048Env(s_next).display()
            print("a_next", action_name(a_next), "r_next", r_next)
            print("s_after_next")
            Game2048Env(s_after_next).display()
            self.V(s_after_next, debug=True)
            print(
                f"delta ({delta:.2f}) = r_next ({r_next:.2f}) + v_after_next ({v_after_next:.2f}) - V(s_after) ({V(s_after):.2f})"
            )
            print(
                f"V(s_after) <- V(s_after) ({V(s_after):.2f}) + alpha * delta ({alpha} * {delta:.1f})"
            )
        self.V(s_after, alpha * delta)

import numpy as np
# from game import Game2048Env
# from game import IllegalAction, GameOver
# from agent import nTupleNewrok
import pickle

from collections import namedtuple

"""
Vocabulary
--------------

Transition: A Transition shows how a board transfromed from a state to the next state. It contains the board state (s), the action performed (a), 
the reward received by performing the action (r), the board's "after state" after applying the action (s_after), and the board's "next state" (s_next) after adding a random tile to the "after state".

Gameplay: A series of transitions on the board (transition_history). Also reports the total reward of playing the game (game_reward) and the maximum tile reached (max_tile).
"""
Transition = namedtuple("Transition", "s, a, r, s_after, s_next")
Gameplay = namedtuple("Gameplay", "transition_history game_reward max_tile")


def play(agent, board, spawn_random_tile=False):
    "Return a gameplay of playing the given (board) until terminal states."
    b = Game2048Env(board)
    r_game = 0
    a_cnt = 0
    transition_history = []
    while True:
        a_best = agent.best_action(b.board)
        s = b.copyboard()
        try:
            r = b.act(a_best)
            r_game += r
            s_after = b.copyboard()
            b.spawn_tile(random_tile=spawn_random_tile)
            s_next = b.copyboard()
        except (IllegalAction, GameOver) as e:
            # game ends when agent makes illegal moves or board is full
            r = None
            s_after = None
            s_next = None
            break
        finally:
            a_cnt += 1
            transition_history.append(
                Transition(s=s, a=a_best, r=r, s_after=s_after, s_next=s_next)
            )
    gp = Gameplay(
        transition_history=transition_history,
        game_reward=r_game,
        max_tile=max(b.board),
    )
    learn_from_gameplay(agent, gp)
    return gp


def learn_from_gameplay(agent, gp, alpha=0.1):
    "Learn transitions in reverse order except the terminal transition"
    for tr in gp.transition_history[:-1][::-1]:
        agent.learn(tr.s, tr.a, tr.r, tr.s_after, tr.s_next, alpha=alpha)


def load_agent(path):
    return pickle.load(path.open("rb"))


# map board state to LUT
TUPLES = [
    # horizontal 4-tuples
    [0, 1, 2, 3],
    [4, 5, 6, 7],
    [8, 9, 10, 11],
    [12, 13, 14, 15],
    # vertical 4-tuples
    [0, 4, 8, 12],
    [1, 5, 9, 13],
    [2, 6, 10, 14],
    [3, 7, 11, 15],
    # all 4-tile squares
    [0, 1, 4, 5],
    [4, 5, 8, 9],
    [8, 9, 12, 13],
    [1, 2, 5, 6],
    [5, 6, 9, 10],
    [9, 10, 13, 14],
    [2, 3, 6, 7],
    [6, 7, 10, 11],
    [10, 11, 14, 15],
]

if __name__ == "__main__":
    import numpy as np

    agent = None
    # prompt to load saved agents
    from pathlib import Path

    path = Path("tmp")
    saves = list(path.glob("*.pkl"))
    if len(saves) > 0:
        print("Found %d saved agents:" % len(saves))
        for i, f in enumerate(saves):
            print("{:2d} - {}".format(i, str(f)))
        k = input(
            "input the id to load an agent, input nothing to create a fresh agent:"
        )
        if k.strip() != "":
            k = int(k)
            n_games, agent = load_agent(saves[k])
            print("load agent {}, {} games played".format(saves[k].stem, n_games))
    if agent is None:
        print("initialize agent")
        n_games = 0
        agent = nTupleNewrok(TUPLES)

    n_session = 5000
    n_episode = 100
    print("training")
    try:
        for i_se in range(n_session):
            gameplays = []
            for i_ep in range(n_episode):
                gp = play(agent, None, spawn_random_tile=True)
                gameplays.append(gp)
                n_games += 1
            n2048 = sum([1 for gp in gameplays if gp.max_tile == 2048])
            mean_maxtile = np.mean([gp.max_tile for gp in gameplays])
            maxtile = max([gp.max_tile for gp in gameplays])
            mean_gamerewards = np.mean([gp.game_reward for gp in gameplays])
            print(
                "Game# %d, tot. %dk games, " % (n_games, n_games / 1000)
                + "mean game rewards {:.0f}, mean max tile {:.0f}, 2048 rate {:.0%}, maxtile {}".format(
                    mean_gamerewards, mean_maxtile, n2048 / len(gameplays), maxtile
                ),
            )
    except KeyboardInterrupt:
        print("training interrupted")
        print("{} games played by the agent".format(n_games))
        if input("save the agent? (y/n)") == "y":
            # fout = "tmp/{}_{}games.pkl".format(agent.__class__.__name__, n_games)
            fout = "new.pkl".format(agent.__class__.__name__, n_games)
            pickle.dump((n_games, agent), open(fout, "wb"))
            print("agent saved to", fout)
