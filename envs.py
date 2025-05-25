import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import itertools


class TradingEnv(gym.Env):
    """
    A multi-stock trading environment for reinforcement learning.

    State: [# of stock owned, current stock prices, cash in hand]
      - length: n_stock * 2 + 1

    Action: encoded as integers ∈ [0, 3^n_stock[
      - Each action corresponds to a combination of (sell, hold, buy)
    """

    metadata = {"render.modes": ["human"]}

    # train_data : matrice numpy de taille (n_stock, n_step) représentant les prix historiques.
    # init_invest : investissement initial de l'agent (en cash)
    # max_steps : nombre maximum d'étapes dans un épisode
    def __init__(self, train_data, init_invest=20000, max_steps=200):
        assert init_invest > 0, "Initial investment must be positive"
        self.stock_price_history = np.around(train_data).astype(int)
        self.n_stock, self.n_step = self.stock_price_history.shape
        print('self.n_stock, self.n_step:', self.n_stock, self.n_step)
        assert self.n_stock > 0 and self.n_step > 1, "Invalid training data shape"

        self.init_invest = init_invest

        # Action space: all combinations of 0 (sell), 1 (hold), 2 (buy) per stock
        self.action_space = spaces.Discrete(3 ** self.n_stock)

        # Observation space construction
        stock_max_price = self.stock_price_history.max(axis=1)
        stock_range = [[0, max(1, init_invest * 2 // p)] for p in stock_max_price]
        price_range = [[0, max(1, int(p))] for p in stock_max_price]
        cash_in_hand_range = [[0, max(1, init_invest * 2)]]
        full_range = stock_range + price_range + cash_in_hand_range
        nvec = [hi - lo + 1 for (lo, hi) in full_range]

        self.observation_space = spaces.MultiDiscrete(nvec)

        self.max_steps = max_steps
        self._seed()
        self.reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # Choisit un point de départ aléatoire dans la série temporelle
    # Initialise : portefeuille vide, cash, prix initial
    def reset(self):
        self.start_step = self.np_random.integers(0, self.n_step - self.max_steps)
        self.cur_step = self.start_step
        self.stock_owned = np.zeros(self.n_stock, dtype=int)
        self.stock_price = self.stock_price_history[:, self.cur_step]
        self.cash_in_hand = self.init_invest
        return self._get_obs()

    # Applique l'action (décode via _trade)
    # Calcule la récompense : variation de valeur de portefeuille
    # Retourne :
    #   l’observation (état),
    #   la récompense,
    #   si l’épisode est fini (done),
    #   des infos utiles (prix, action, portefeuille)
    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        prev_val = self._get_val()
        self.cur_step += 1
        done = self.cur_step >= self.start_step + self.max_steps

        self.stock_price = self.stock_price_history[:, self.cur_step]
        self._trade(action)
        cur_val = self._get_val()

        reward = cur_val - prev_val
        obs = self._get_obs()
        info = {
            "cur_val": cur_val,
            "price": self.stock_price.copy(),
            "action_vec": list(itertools.product([0, 1, 2], repeat=self.n_stock))[action],
            "stock_owned": self.stock_owned.copy(),
            "cash_in_hand": self.cash_in_hand
        }
        return obs, reward, done, info

    # Concatène les stocks détenus, prix actuels et cash → observation complète
    def _get_obs(self):
        obs = np.concatenate((self.stock_owned, self.stock_price, [self.cash_in_hand]))
        return obs.astype(int)

    # Calcule la valeur totale du portefeuille à un instant donné
    def _get_val(self):
        return int(np.sum(self.stock_owned * self.stock_price) + self.cash_in_hand)

    def _trade(self, action):
        # Décode l’action entière en un vecteur [0, 1, 2]^n_stock
        action_vec = list(itertools.product([0, 1, 2], repeat=self.n_stock))[action]
        sell_index = [i for i, a in enumerate(action_vec) if a == 0]
        buy_index = [i for i, a in enumerate(action_vec) if a == 2]

        # Vend d’abord les actions sélectionnées
        for i in sell_index:
            self.cash_in_hand += self.stock_price[i] * self.stock_owned[i]
            self.stock_owned[i] = 0

        # Essaie d’acheter autant que possible les actions demandées, en boucle (tant que le cash le permet)
        can_buy = True
        while can_buy and buy_index:
            can_buy = False
            for i in buy_index:
                if self.cash_in_hand >= self.stock_price[i]:
                    self.stock_owned[i] += 1
                    self.cash_in_hand -= self.stock_price[i]
                    can_buy = True

