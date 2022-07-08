import random
import pandas as pd
import numpy as np
import pickle
import config
from SimEnv import Environment

class Simulation:
    '''
    Simulate the sailing of liner shipping and generate samples
    '''
    def __init__(self, price, c_bar, K, path):
        self.price = price
        self.remaining_time = 70 # 14days * 5/day
        self.SALE = 50
        self.PRICE = price
        self.c_bar = c_bar
        self.K = K
        self.max_price = config.MAX_PRICE
        self.min_price = config.MIN_PRICE
        self.max_sale = config.MAX_SALE
        self.env = Environment(path, 'gamma')
        # self.cnt_distribution = self.read_wbl_distribution(wbl_path)

    # def get_sample(self):
    #     sample = np.array([self.price, self.util, self.remaining_time])
    #     return sample

    # def slow_step(self, action):
    #     '''
    #     the environment response instantly
    #     action means the price change [action]
    #     :return:
    #     '''
    #     # if the price is nonzero, clear the wbl_list and change price
    #     # then generate a new nonzero sale for the remaning time
    #     self.price = max(self.price + action, 0)
    #     if len(self.wbl_list) != 0 and action != 0:
    #         #print(self.wbl_list)
    #         sale = int(self.env.ave_sale_nonzero(self.price) * self.wbl_list[0][1])
    #         if sale != 0:
    #             self.wbl_list = self.time_distribution(self.split_wbl(sale), interval = self.wbl_list[0][1])
    #     elif len(self.wbl_list) == 0:
    #         sale = self.env.ave_sale(self.price)  # total sale in two hours
    #         if sale == 0:
    #             self.wbl_list = [[0, 1]]
    #         else:
    #             self.wbl_list = self.time_distribution(self.split_wbl(sale))
    #
    #     slow_sale = self.wbl_list[0][0]
    #     slow_time = self.wbl_list[0][1]
    #     self.wbl_list = self.wbl_list[1:]
    #     reward = slow_sale * self.price
    #     self.util += slow_sale
    #     self.remaining_time -= slow_time
    #
    #
    #     if self.remaining_time <= 0 or self.util >= self.max_util:
    #         sample = np.array([self.price, self.util, self.remaining_time])
    #         done = 1
    #         self.reset()
    #     else:
    #         sample = np.array([self.price, self.util, self.remaining_time])
    #         done = 0
    #
    #     return reward, sample, done

    def step(self, state, action):
        '''
        the environment response within two hours
        action means the price change [action]
        :return:
        '''
        p = random.random()
        if (p < self.c_bar/self.K):
            self.price = max(self.price + action, 0)
            self.remaining_time -= 1
            sale = random.randrange(0, self.max_sale)
            return sale, [self.price, self.remaining_time]
        self.price = state[0]
        self.remaining_time -= 1
        self.price = max(self.price + action, 0) # price change
        #sale = max(int(self.SALE - 1 * self.price / 100), 0)
        sale = self.env.ave_sale(self.price)
        # sale = min(self.max_util - self.util, sale)
        # self.util += sale # utilization change
        # self.remaining_time -= 1
        #
        # if self.remaining_time <= 0 or self.util >= self.max_util:
        #     self.reset()
        #     sample = np.array([self.price, self.util, self.remaining_time])
        #     done = 1
        # else:
        #     sample = np.array([self.price, self.util, self.remaining_time])
        #     done = 0

        return sale, [self.price, self.remaining_time]

    def expectation(self, price):
        sale = self.env.expectation_sale(price)
        return sale

    def reset(self):
        # self.done = 0
        # self.util = 0
        self.price = self.PRICE
        self.remaining_time = 70
        # self.wbl_list = []

    # def split_wbl(self, x):
    #     wbl_list = []
    #     if x < 1:
    #         wbl_list.append(0)
    #         return wbl_list
    #     while x >= 1:
    #         sum = 0
    #         list = []
    #         for key, value in self.cnt_distribution.items():
    #             if key <= x:
    #                 sum += value
    #                 tmplist = [key for _ in range(value)]
    #                 list.extend(tmplist)
    #         cntr_cnt = random.choice(list)
    #         wbl_list.append(cntr_cnt)
    #         x -= cntr_cnt
    #     return wbl_list
    #
    # def time_distribution(self, wbl_list, interval = 1):
    #     pair_list = []
    #     for wbl in wbl_list:
    #         time = random.random() * interval
    #         pair_list.append([wbl, time])
    #     pair_list.sort(key=lambda x: x[1])
    #     #print('pair_list: {}'.format(pair_list))
    #     return pair_list
    #
    # def read_wbl_distribution(self, path):
    #     f = open(path, 'rb')
    #     return pickle.load(f)
