import numpy as np
from matplotlib import pyplot as plt

from RL_brain import SarsaTable, QLearningTable
from SimEnv import Environment
from sample_gen import Simulation
import config
import random
import csv

random.seed(10)

min_price = 1500
max_price = 3500
num_seg = 1000
max_reward = max_price * 50
action_range = 40  # [-40, 40]

f = lambda x: int((x - min_price) / (max_price - min_price) * num_seg)
f_inv = lambda x: min_price + x * (max_price - min_price) * num_seg
f_inv_action = lambda x: x - action_range
f_reward = lambda x: x / max_reward

K = 5000  # num of periods (not 14)
H = 70  #
delta = 0.1  # Waiting for training
c_bar = 10  # need to change
S = 70 * num_seg  # size of state set 20*70
S_dim = 3  # [price, rest capacity, remaining time]
A = 2 * action_range  # size of action set (40)-(-40)
d = 1
xi = 200
eta = 0.1
init_price = 1800
init_rt = 70
average_reward = 0
action_space = []
for i in range(action_range * 2):
    action_space.append(i)

#sarsa = SarsaTable(actions=action_space)
qlearning = QLearningTable(actions=action_space)
sim = Simulation(init_price, c_bar, K, config.SIM_SALE_PATH)  # initial price is 1500

recoder = []
total_reward_recorder = []

for k in range(K):

    # initialize the state

    true_price = init_price
    sim.reset()

    rest = xi

    true_s = np.array([true_price, rest, 70])
    fake_s = np.array([f(true_price), rest, 70])

    total_reward = 0
    recoder = []

    for h in range(H):
        a = qlearning.choose_action(str([fake_s[0], fake_s[1], fake_s[2]]))
        # recoder.append([true_s, f_inv_action(a)])

        bk = 0

        # print("Round k=({}),h=({}):{}".format(k, h, bk))

        sale, true_s_ = sim.step(true_s, f_inv_action(a))
        if rest < sale:
            sale = rest
        true_s_ = np.array([true_s_[0], rest - sale, true_s_[1]])
        recoder.append([true_s, f_inv_action(a), sale])

        rk = sale * true_s_[0]
        rk_norm = f_reward(rk)
        # r = rk + bk
        # ck = sale
        # c = ck - bk
        total_reward += rk  # renvenue
        # r_lambda = r

        fake_s_ = np.array([f(true_s_[0]), rest - sale, true_s_[1]])
        #a_ = sarsa.choose_action(str([fake_s_[0], fake_s_[1], fake_s_[2]]))
        if fake_s_[1] == 0:
            fake_s_ = "terminal"
        qlearning.learn(str([fake_s[0], fake_s[1], fake_s[2]]), a, rk_norm, str([fake_s_[0], fake_s_[1], fake_s_[2]]))
        # update s
        if fake_s_[1] == 0:
            break
        true_s = true_s_
        fake_s = fake_s_
        rest = rest - sale

    average_reward += total_reward
    if k % 10 == 0 and k > 0:
        print('Total reward for episode {} is {}'.format(k, average_reward / 10))
        total_reward_recorder.append(average_reward / 10)
        average_reward = 0
        print('The last pricing process is {}'.format(recoder))

plt.plot(total_reward_recorder)
plt.savefig("fig-q_learning-02-06-19-30.png")
with open("log-q_learning-02-06-19-30.csv", "w") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(total_reward_recorder)