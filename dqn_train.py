import numpy as np
from matplotlib import pyplot as plt

from DQN import DQN
from SimEnv import Environment
from sample_gen import Simulation
import config
import random

random.seed(10)

min_price = 1500
max_price = 3500
num_seg = 200
action_range = 40  # [-40, 40]

f = lambda x: int((x - min_price) / (max_price - min_price) * num_seg)
f_inv = lambda x: min_price + x * (max_price - min_price) * num_seg
f_inv_action = lambda x: x - action_range

K = 2000  # num of periods (not 14)
H = 70  #
delta = 0.01  # Waiting for training
c_bar = 10  # need to change
S = 70 * num_seg  # size of state set 20*70
S_dim = 2  # [price, remaining time]
A = 2 * action_range  # size of action set (40)-(-40)
d = 1
xi = 200
eta = 0.1
init_price = 2500
init_rt = 70
average_reward = 0

dqn = DQN(S_dim, A)
sim = Simulation(init_price, c_bar, K, config.SIM_SALE_PATH)  # initial price is 1500
# dqn.load_weight(config.WEIGHT_PATH)

Nk = {}

lambda_i = 0  # P23-4

recoder = []
total_reward_recorder = []

for k in range(K):

    # initialize the state

    true_price = init_price
    sim.reset()

    true_s = np.array([true_price, 70])
    fake_s = np.array([f(true_price), 70])

    sum_cxi = 0
    total_reward = 0
    recoder = []

    for h in range(H):
        a = dqn.choose_action(fake_s)
        recoder.append([true_s, f_inv_action(a)])

        if (tuple(fake_s), a) in Nk:
            Nk[(tuple(fake_s), a)] += 1
        else:
            Nk[(tuple(fake_s), a)] = 1
        #bk = min(2 * H, H * np.sqrt(2 * np.log(64 * S * A * H * (d + 1) * (K ** 2) / delta) / Nk[(tuple(fake_s), a)]))
        bk = 0

        # print("Round k=({}),h=({}):{}".format(k, h, bk))

        sale, true_s_ = sim.step(true_s, f_inv_action(a))

        rk = sale * true_s_[0]
        # r = rk + bk
        # ck = sale
        # c = ck - bk
        total_reward += rk

        # r_lambda = r + lambda_i * (c - xi)

        # sum_cxi += (c - xi)

        fake_s_ = np.array([f(fake_s[0]), fake_s[1]])
        dqn.store_transition(fake_s, a, rk, fake_s_)
        # update s
        true_s = true_s_
        fake_s = fake_s_

    average_reward += total_reward
    if k % 10 == 0 and k > 0:
        print('Total reward for episode {} is {}'.format(k, average_reward / 10))
        total_reward_recorder.append(average_reward / 10)
        average_reward = 0
        print('The last pricing process is {}'.format(recoder))
    dqn.learn()
    lambda_i = min(0, lambda_i - eta * sum_cxi / H)

dqn.save_weight(config.WEIGHT_PATH)
plt.plot(total_reward_recorder)
plt.savefig("fig.png")
