'''
Hyperparameter in DQN
'''
BATCH_SIZE = 32
LR = 0.0005                   # learning rate
EPSILON = 0.9               # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 100   # target update frequency
MEMORY_CAPACITY = 20000
MEMORY_COUNTER = 0          # for store experience
LEARNING_STEP_COUNTER = 0   # for target updating
MAX_PRICE = 3500
MIN_PRICE = 1500
NUM_SEG= 1000
MAX_SALE = 20
SIM_SALE_PATH = '../data/2hours_price_setting_env.csv'
WEIGHT_PATH = './weights/'
