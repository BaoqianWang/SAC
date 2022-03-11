import math
import random
import sys
from scipy import special
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import json
import pickle
from experiments.helper.sac_rl import MultiAccessNetworkRL
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser("SAC in WMC")
    # Environment
    parser.add_argument("--seed", type=int, default=6, help="random seed")
    parser.add_argument("--ddl", type=int, default=10, help="packets queue length")
    parser.add_argument("--grid-size", type=int, default=3, help="grid size")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--T", type=int, default=10, help="time steps, episode length")
    parser.add_argument("--save-rate", type=int, default=200)
    parser.add_argument("--evalM", type=int, default=15)
    parser.add_argument("--save-dir", type=str, default="../trained_policy/")
    parser.add_argument("--max-iteration", type=int, default=30000)

    return parser.parse_args()



def touch_path(path):
    dirname = os.path.dirname(path)
    if any(dirname):
        os.makedirs(dirname, exist_ok=True)

if __name__ == "__main__":
    arglist = parse_args()
    seed = arglist.seed
    ddl = arglist.ddl
    grid_size = arglist.grid_size
    T = arglist.T
    evalInterval = arglist.save_rate
    evalM = arglist.evalM
    gamma = arglist.gamma

    M = arglist.max_iteration
    nodeNum = grid_size ** 2
    save_dir = arglist.save_dir + '%d_agents_%d_ddl_%d_iteration_%d_seed/' %(nodeNum, ddl, evalM, seed)
    touch_path(save_dir)
    # Set the random seed
    np.random.seed(seed)
    random.seed(seed)

    # hard coded parameters
    maxK = 1
    arrivalProb = None
    transmitProb = 'random'
    restartIntervalQ = 10
    restartIntervalPolicy = 10
    k = 1

    networkRLModel = MultiAccessNetworkRL(ddl = ddl, graphType = 'grid', nodeNum = nodeNum, maxK = maxK ,arrivalProb = arrivalProb,transmitProb = transmitProb, gridW = grid_size, gridH = grid_size, gamma = gamma)
    policyRewardSmooth, global_time = networkRLModel.train(k = k, M = M, T= T, evalInterval = evalInterval,restartIntervalQ = restartIntervalQ, restartIntervalPolicy = restartIntervalPolicy, evalM = evalM, clearPolicy = True)
    reward_file =  save_dir +  'reward.pkl'
    time_file = save_dir + 'time.pkl'

    with open(rewagit rd_file, 'wb') as fp:
        pickle.dump(policyRewardSmooth, fp)

    with open(time_file, 'wb') as fp:
        pickle.dump(global_time, fp)


    plt.figure(figsize=(5.8,4.5))
    font_size = 18
    plt.plot(global_time, policyRewardSmooth, linestyle=':', linewidth=2, label='SAC')
    plt.ylabel('Reward', fontsize=font_size)
    plt.xlabel('Training iteration', fontsize=font_size)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.legend(fontsize=font_size-2)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.subplots_adjust(bottom=0.15, left=0.2, top=0.95, wspace=0, hspace=0)
    plt.grid()
    #plt.savefig('../figures/sac0308', transparent = False)
    plt.show()
