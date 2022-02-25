import math
import random
import sys
from scipy import special
from tqdm import trange
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from experiments.helper.sac_rl import MultiAccessNetworkRL



if __name__ == "__main__":

    ddl = 20
    np.random.seed(1)

    gridW = 3
    gridH = 3
    nodeNum = 9 # number of nodes in the network
    maxK = 1 # the size of neighborhood we use in localized learning
    arrivalProb = None
    transmitProb = 'random'
    gamma = 0.7
    networkRLModel = MultiAccessNetworkRL(ddl = 2, graphType = 'grid', nodeNum = nodeNum, maxK = maxK ,arrivalProb = arrivalProb,transmitProb = transmitProb, gridW = gridW, gridH = gridH, gamma = 0.7)
    M = 1000
    T = 25

    evalInterval = 200 #evaluate the policy every evalInterval rounds (outer loop)
    evalM = 150
    restartIntervalQ = 10
    restartIntervalPolicy = 10
    np.random.seed()
    k = 1
    policyRewardSmooth = networkRLModel.train(k = k, M = M, T= T, evalInterval = evalInterval,restartIntervalQ = restartIntervalQ, restartIntervalPolicy = restartIntervalPolicy, clearPolicy = True)
    plt.rc('font', size=14)
    plt.plot(np.linspace(1,M,len(policyRewardSmooth)), policyRewardSmooth, label = 'SAC $\kappa = '+str(1)+'$')
    plt.xlim(0,M)
    plt.legend(loc = 'lower right')
    plt.ylabel('Total Discounted Reward')
    plt.xlabel('$m$')
    plt.show()
