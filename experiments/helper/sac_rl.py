from experiments.helper.sac_network import GlobalNetwork
from experiments.helper.sac_network import AccessNetwork
from experiments.helper.sac_network import constructGridNetwork
from experiments.helper.sac_node import accessNodeDiscounted
from experiments.helper.sac_node import Node
import numpy as np
from tqdm import trange
import time

class MultiAccessNetworkRL:
    def __init__(self,ddl = 2, graphType = 'line', nodeNum = 10, maxK = 3 ,arrivalProb = None, transmitProb = 'random', gridW = 1, gridH = 1, gamma = 0.7):
        self.ddl = ddl
        self.nodeNum = nodeNum
        self.gamma = gamma
        self.maxK = maxK
        if(arrivalProb == None):
            self.arrivalProb = np.random.rand(nodeNum)
        else:
            self.arrivalProb = arrivalProb


        self.accessNetwork =  constructGridNetwork(nodeNum = nodeNum, width = gridW, height = gridH, k = maxK, transmitProb= transmitProb)

        self.nodeList = []
        for i in range(nodeNum):
            self.nodeList.append(accessNodeDiscounted(index = i, ddl = ddl, arrivalProb = self.arrivalProb[i], accessNetwork = self.accessNetwork, gamma = gamma, nodeNum = nodeNum) )


    def train(self, k = 1, M = 10000, T = 20, evalInterval = 500, restartIntervalQ = 50, restartIntervalPolicy = 50, evalM = 500, clearPolicy = True):

        # print(T)
        policyRewardList = []
        policyRewardSmooth = []
        global_time = []
        #print('iteration', M)
        start_time = time.time()

        for m in range(M):
            discountedReward = 0.0 # total reward for this

            print('iteration.............', m)
            # restart node
            for i in range(self.nodeNum):
                if(m == 0):
                    self.nodeList[i].restart(clearPolicy = clearPolicy, clearQ = True)
                else:
                    self.nodeList[i].restart(clearPolicy = False, clearQ = False)
                self.nodeList[i].initializeState()
                #print('initialize........')
                self.nodeList[i].updateAction()

            tmpReward = 0.0
            for i in range(self.nodeNum):
                neighborList = []
                for j in self.accessNetwork.findNeighbors(i, 1):
                    neighborList.append(self.nodeList[j])
                self.nodeList[i].updateReward(neighborList,self.accessNetwork)
                tmpReward += self.nodeList[i].reward[-1]
            discountedReward += tmpReward/self.nodeNum

            for t in range(T):
                #print(t)
                for i in range(self.nodeNum): #update state-action
                    self.nodeList[i].updateState()
                    self.nodeList[i].updateAction()
                tmpReward = 0.0
                for i in range(self.nodeNum):
                    neighborList = []
                    for j in self.accessNetwork.findNeighbors(i, 1):
                        neighborList.append(self.nodeList[j])
                    self.nodeList[i].updateReward(neighborList,self.accessNetwork)

                    tmpReward += self.nodeList[i].reward[-1] # add latest reward
                discountedReward += (self.gamma**(t+1))*tmpReward/self.nodeNum
                # Update Q-function
                for i in range(self.nodeNum):
                    neighborList = []
                    for j in self.accessNetwork.findNeighbors(i, k):
                        neighborList.append(self.nodeList[j])
                    self.nodeList[i].updateQ(neighborList, 1/pow((m%restartIntervalQ)+1,.4)  )

            policyRewardList.append(discountedReward)


            #perform the grad update
            for i in range(self.nodeNum):
                neighborList = []
                for j in self.accessNetwork.findNeighbors(i, k):
                    neighborList.append(self.nodeList[j])
                self.nodeList[i].updateParams(neighborList,  1.0* 1/pow((m%restartIntervalPolicy)+1,.6))

            #print(m%restartIntervalPolicy)
            if m > M*0.9: # for the last 10% of running, no restarting
                restartIntervalQ = max(int(M*0.5),restartIntervalQ)
                restartIntervalPolicy = max(int(M*0.5),restartIntervalPolicy)

            # perform a policy evaluation
            if m%evalInterval == 0:
                end_time = time.time()
                policyRewardSmooth.append(self.policyEval(evalM, T))
                global_time.append(end_time - start_time)

        #print('done')
        return policyRewardSmooth, global_time


    def policyEval(self, evalM, T):
        aveReward = 0.0
        for m in range(evalM):
            # restart policy
            discountedReward = []

            for i in range(self.nodeNum):
                self.nodeList[i].restart(clearPolicy = False, clearQ = False)
                self.nodeList[i].initializeState()
                self.nodeList[i].updateAction()

            tmpReward = 0.0
            for i in range(self.nodeNum):
                neighborList = []
                for j in self.accessNetwork.findNeighbors(i, 1):
                    neighborList.append(self.nodeList[j])
                self.nodeList[i].updateReward(neighborList,self.accessNetwork)
                tmpReward += self.nodeList[i].reward[-1]

            #discountedReward.append(tmpReward)
            #print(T)
            for t in range(T):
                for i in range(self.nodeNum): #update state-action
                    self.nodeList[i].updateState()
                    self.nodeList[i].updateAction()
                #tmpReward = 0.0
                for i in range(self.nodeNum):
                    neighborList = []
                    for j in self.accessNetwork.findNeighbors(i, 1):
                        neighborList.append(self.nodeList[j])
                    self.nodeList[i].updateReward(neighborList,self.accessNetwork)

                    tmpReward += self.nodeList[i].reward[-1] # add latest reward
                #print(tmpReward)
            discountedReward.append(tmpReward)

        return np.mean(discountedReward)
