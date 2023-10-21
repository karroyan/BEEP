import random
import gym
from gym import spaces
import numpy as np
import config
from SimEnv import Environment

class Liner_Environment(gym.Env):
    '''
    Environment for the container sales in liner shipping scenerio.
    Applied the constraints described in BEEP
    '''

    def __init__(self, price: float, c_bar: int, K: int, path: str, action_range: int):
        '''
        self.state will not be initlized here, need be reset

        Input:
        - price(float): the initial price of each episode 
        - c_bar(int): the number of corrupted episodes
        - K(int): the number of total episodes
        - path(str): the path of history sale data, end in .csv
        - action_range(int): the absolute action range
        '''
        self.state
        self.price = price
        self.H = 70 # 14days * 5/day
        self.remaining_time = self.H 
        self.SALE = 50
        self.PRICE = price
        self.c_bar = c_bar
        self.K = K

        self.action_range = action_range
        self.num_seg = config.NUM_SEG
        self.max_price = config.MAX_PRICE
        self.min_price = config.MIN_PRICE
        self.max_sale = config.MAX_SALE

        self.env = Environment(path, 'gamma')
        
        # reward calculate parameters
        self.Nk = {}
        self.S = 70*self.num_seg
        self.A = 2*self.action_range
        self.d = 1
        self.delta = 0.1
        self.eta = 0.1
        self.xi = 200
        self.sum_cxi = 0
        self.lambda_i = 0
        self.max_reward = self.max_price * 50
        self.f = lambda x: int((x-self.min_price) / (self.max_price-self.min_price) * self.num_seg)
        self.f_inv = lambda x: self.min_price + x * (self.max_price-self.min_price) / self.num_seg
        self.f_reward = lambda x:x/self.max_reward

        assert self.max_price <= self.min_price, 'The max price should be larger than the min price'
        # The action space is the discrete space within 2*action_range, with one yuan as an interval
        # Notice the action here is not real action, it should be map to [-action_range, action_range]
        self.action_space = spaces.Discrete(2*self.action_range)
        # The state space has two dimension, price and remaining time
        self.observation_space = spaces.Box(low=np.array([self.min_price, 0]), \
                                            high=np.array([self.max_price, self.remaining_time]), dtype=np.float32)

    def reset(self):
        '''
        reset the initial price and remainingtime

        Output:
        - state(np.array(float, float)): The current state of [price, remaining_time]
        '''
        self.price = self.PRICE
        self.remaining_time = 70
        self.state = np.array([self.price, self.remaining_time])

        return self.state

    def step(self, action: int):
        '''
        Move one step in the environment and return result.

        Input:
        - action(int): the fake action to change, which should be in [0, 2*action_range]
        
        Output:
        - state(np.array(float, float)): the state containing price and remaining time
        - reward(float): the constrained added reward
        - done(bool): whether the episode finished
        '''
        p = random.random()
        real_action = action - self.action_range
        # price can not be changed out of the price range: [self.min_price, self.max_price]
        self.price = max(self.min_price, min(self.max_price, self.price + real_action))
        self.remaining_time -= 1

        if (p < self.c_bar/self.K):
            # This episode is corrputed, the sale will be random sampled from [0, max_sale]
            sale = random.randrange(0, self.max_sale)
        else:
            # If the episode is not corrputed, the sale will be estimated from the history sale data
            sale = self.env.ave_sale(self.price)

        self.state = np.array([self.price, self.remaining_time])
        if self.remaining_time <= 0:
            done = True
        
        reward = self.reward_cal(self.state, action, done, int(sale))

        return self.state, reward, done, {}

    def reward_cal(self, state, action: int, done: bool, sale: int):
        '''
        Calculated the contrained reward r_lambdain BEEP:
        r = r_bar + bk
        for each c:
            c = c_bar -bk
        r_lambda = r + sum(c)

        '''
        # fake state is used to record the visit times Nk
        fake_state = np.array([self.f(state[0]), state[1]])

        if (tuple(fake_state),action) in self.Nk:
            self.Nk[(tuple(fake_state),action)] += 1
        else:
            self.Nk[(tuple(fake_state),action)] = 1
        bk = min(2*self.H, self.H*np.sqrt(2*np.log(64*self.S*self.A*self.H*(self.d+1)*\
                                                   (self.K**2)/self.delta)/self.Nk[(tuple(fake_state),action)]) \
                 + self.c_bar * (self.H**2) / self.Nk[(tuple(fake_state),action)])

        rk = sale*state[0]
        rk_norm = self.f_reward(rk)
        r = rk_norm + bk
        c = sale - bk
        r_lambda = r + self.lambda_i * (c-self.xi)

        self.sum_cxi += (c - self.xi)

        if done:
            self.lambda_i = min(0, self.lambda_i - self.eta*self.sum_cxi/self.H)

        return r_lambda

    def expectation(self, price):
        sale = self.env.expectation_sale(price)
        return sale
