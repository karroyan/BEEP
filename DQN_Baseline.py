import numpy as np
from matplotlib import pyplot as plt

from DQN import DQN
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
action_range = 40 # [-40, 40]

f = lambda x: int((x-min_price) / (max_price-min_price) * num_seg)
f_inv = lambda x: min_price + x * (max_price-min_price) * num_seg
f_inv_action = lambda x: x - action_range
f_reward = lambda x: x/max_reward

K = 3500 #num of periods (not 14)
H = 70 #
delta = 0.1 # Waiting for training
c_bar = 20 # need to change
S = 70*num_seg # size of state set 20*70
S_dim = 3 # [price, rest capacity, remaining time]
A = 2*action_range # size of action set (40)-(-40)
d = 1
xi = 200
eta = 0.1
init_price = 1800
init_rt = 70
average_reward = 0

dqn = DQN(S_dim,A)
sim = Simulation(init_price, c_bar, K, config.SIM_SALE_PATH) # initial price is 1500
#dqn.load_weight(config.WEIGHT_PATH)



recoder = []
total_reward_recorder = []
recorder_100 = []

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
        a = dqn.choose_action(fake_s)
        #recoder.append([true_s, f_inv_action(a)])

        bk = 0

        #print("Round k=({}),h=({}):{}".format(k, h, bk))


        sale, true_s_= sim.step([true_s[0], true_s[2]],f_inv_action(a))
        if rest < sale:
            sale = rest
        true_s_ = np.array([true_s_[0], rest-sale, true_s_[1]])
        recoder.append([true_s, f_inv_action(a), sale])

        rk = sale*true_s_[0]
        rk_norm = f_reward(rk)
        # r = rk + bk
        # ck = sale
        # c = ck - bk
        total_reward += rk # renvenue
        
        # r_lambda = r


        fake_s_ = np.array([f(true_s_[0]), rest-sale, true_s_[1]])
        dqn.store_transition(fake_s,a,rk_norm,fake_s_)
        # update s
        true_s = true_s_
        fake_s = fake_s_
        rest = rest-sale
        if rest <= 0:
            break

    average_reward += total_reward
    if k %10 == 0 and k > 0:
        if k%100 == 0:
            print('Total reward for episode {} is {}'.format(k, average_reward/10))
        total_reward_recorder.append(average_reward / 10)
        average_reward = 0
        #print('The last pricing process is {}'.format(recoder))
    if k >  (K-101):
        recorder_100.append(total_reward)
    dqn.learn()


dqn.save_weight(config.WEIGHT_PATH)
plt.plot(total_reward_recorder)
plt.savefig("fig-dqn-c-20-third.png")
with open("log-dqn-c-20-third-reward.txt", "w") as file:
    for r in total_reward_recorder:
        file.write(str(r))
        file.write('\n')
with open("log-dqn-c-20-third.txt", "w") as file:
    file.write(str(sum(recorder_100) / len(recorder_100)))
    file.write(str(max(recorder_100)))

with open("log-dqn-c-20-third.csv", "w") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(list(total_reward_recorder))