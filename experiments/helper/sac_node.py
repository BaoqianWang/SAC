import numpy as np
from scipy import special

class Node:
    def __init__(self, index):
        self.index = index
        self.state = [] #The list of local state at different time steps
        self.action = [] #The list of local actions at different time steps
        self.reward = [] #The list of local actions at different time steps
        self.currentTimeStep = 0 #Record the current time step.
        self.paramsDict = {} #use a hash map to query the parameters given a state (or neighbors' states)
        self.QDict = {} #use a hash map to query to the Q value given a (state, action) pair
        self.kHop = [] #The list to record the (state, action) pairs of k-hop neighbors
    #get the local Q at timeStep
    def getQ(self, kHopStateAction):
        #if the Q value of kHopStateAction hasn't been queried before, return 0.0 (initial value)
        return self.QDict.get(kHopStateAction, 0.0)

    #initialize the local state
    def initializeState(self):
        pass
    #update the local state, it may depends on the states of other nodes at the last time step.
    #Remember to increase self.currentTimeStep by 1
    def updateState(self):
        pass
    #update the local action
    def updateAction(self):
        pass
    #update the local reward
    def updateReward(self):
        pass
    #update the local Q value
    def updateQ(self):
        pass
    #update the local parameter
    def updateParams(self):
        pass
    #clear the record. Called when a new inner loop starts.
    def restart(self, clearPolicy = False, clearQ = False):
        self.state.clear()
        self.action.clear()
        self.reward.clear()

        if(clearPolicy == True):
            self.paramsDict.clear()
        if(clearQ == True):
            self.QDict.clear()
        self.kHop = []
        self.currentTimeStep = 0


class accessNodeDiscounted(Node):
    def __init__(self, index, ddl, arrivalProb, accessNetwork, gamma, nodeNum):
        super(accessNodeDiscounted, self).__init__(index)
        self.ddl = ddl #the initial deadline of each packet
        self.arrivalProb = arrivalProb #the arrival probability at each timestep
        #we use packetQueue to represent the current local state, which is (e_1, e_2, ..., e_d)
        self.packetQueue = [np.random.choice(2) for i in range(self.ddl)]#np.zeros(self.ddl, dtype = int) #use 1 to represent a packet with this remaining time, otherwise 0
        #print(self.packetQueue)
        self.accessPoints = accessNetwork.findAccess(i=index) #find and cache the access points this node can access
        self.accessNum = len(self.accessPoints) #the number of access points
        self.actionNum = self.accessNum  + 1 #the number of possible actions
        #print(self.actionNum)
        self.stateNum = 2**self.ddl # number of possible states
        self.gamma = gamma # discounting factor
        self.nodeNum = nodeNum
        #construct a list of possible actions
        self.actionList = [-1] #(-1, -1) is an empty action that does nothing
        for a in self.accessPoints:
            self.actionList.append(a)
        self.index = index
        # set default policy
        self.defaultPolicy = np.zeros(self.actionNum)
        self.defaultPolicy[0] = 2

    #remove the first element in packetQueue, and add packetState to the end
    def rotateAdd(self, packetState):
        #print('self.packetQueue[1:] =',self.packetQueue[1:],'self.ddl = ',self.ddl, 'packetState = ',packetState )
        #print('before = ',self.packetQueue)
        self.packetQueue = np.insert(self.packetQueue[1:], self.ddl - 1, packetState)
        #print('after = ',self.packetQueue)

    #initialize the local state (called at the beginning of the training process)
    def initializeState(self):
        self.packetQueue = [np.random.choice(2) for i in range(self.ddl)]#np.zeros(self.ddl, dtype = int) #use 1 to represent a packet with this remaining time, otherwise 0

        newPacketState = np.random.choice(2)#np.random.binomial(1, 0.5) #Is there a packer arriving at time step 0?
        self.rotateAdd(newPacketState) #get the packet queue at time step 0
        self.state.append(tuple(self.packetQueue)) #append this state to state record
        # print('call initializeState')
        # print(newPacketState)
        #print(self.index, self.packetQueue)
        #print(self.index, self.state)

    #At each time step t, call updateState, updateAction, updateReward, updateQ in this order
    def updateState(self):
        self.currentTimeStep += 1
        lastAction = self.action[-1]

        # find the earliest slot
        nonEmptySlots = np.nonzero(self.packetQueue == 1)

        if len(nonEmptySlots) >0: # queue not empty
            #if the reward at the last time step is positive, we have successfully send out a packet
            if self.reward[-1] > 0:
                self.packetQueue[nonEmptySlots[0]] = 0 # earliest packet is sent

        #sample whether next packet comes
        newPacketState = np.random.choice(2)#np.random.binomial(1, self.arrivalProb) #Is there a packer arriving at time step 0?
        self.rotateAdd(newPacketState) #get the packet queue at time step 0
        self.state.append(tuple(self.packetQueue)) #append this state to state record
        # print('update state', self.state)


    def updateAction(self):
        # get the current state
        currentState = tuple(self.packetQueue)
        # fetch the params based on the current state. If haven't updated before, return all zeros
        params = self.paramsDict.get(currentState, np.zeros(self.actionNum))
        # compute the probability vector
        probVec = special.softmax(params)

        # randomly select an action based on probVec
        currentAction = self.actionList[np.random.choice(a = self.actionNum, p = probVec)]
        #print(currentAction)

        self.action.append(currentAction)

    #oneHopNeighbors is a list of accessNodes
    def updateReward(self, oneHopNeighbors, accessNetwork):
        #decide if a packet is successfully sending out
        currentAction = self.action[-1]
        if currentAction == -1: # the do nothing action
            self.reward.append(0.0)
            return
        currentState = np.array(self.state[-1])

        #check if the node try to send out an empty slot
        if np.all(currentState == 0): # if the current queue is empty
            # zero reward
            self.reward.append(0.0)
            #return

        for neighbor in oneHopNeighbors:
            if neighbor.index == self.index:
                continue
            neighborAction = neighbor.action[-1]

            if neighborAction != currentAction:
                continue
            else:
                neighborState = np.array(neighbor.state[-1])
                #print('neighborState', neighborState)
                if np.any(neighborState == 1): # neighbor queue non empty, conflict!
                    #print('conflict!')
                    self.reward.append(0.0)
                    return

        # no conflict, send
        transmitSuccess = np.random.binomial(1, accessNetwork.transmitProb[currentAction])
        if transmitSuccess == 1:
            self.reward.append(1.0)
        else:
            self.reward.append(0.0)



    #kHopNeighbors is a list of accessNodes, alpha is learning rate
    def updateQ(self, kHopNeighbors, alpha):
        lastStateAction = []
        currentStateAction = []
        #print('neighbors', self.index, len(kHopNeighbors))
        #construct a list of the state-action pairs of k-hop neighbors
        for neighbor in kHopNeighbors:
            neighborLastState = neighbor.state[-2]
            neighborCurrentState = neighbor.state[-1]
            neighborLastAction = neighbor.action[-2]
            neighborCurrentAction = neighbor.action[-1]
            lastStateAction.append((neighborLastState, neighborLastAction))
            currentStateAction.append((neighborCurrentState, neighborCurrentAction))
        lastStateAction = tuple(lastStateAction)
        currentStateAction = tuple(currentStateAction)
        #fetch the Q value based on neighbors' states and actions
        lastQTerm1 = self.QDict.get(lastStateAction, 0.0)
        lastQTerm2 = self.QDict.get(currentStateAction, 0.0)
        #compute the temporal difference
        temporalDiff = self.reward[-2] +  self.gamma*lastQTerm2 - lastQTerm1
        #print('lastStateAction',lastStateAction)
        #perform the Q value update
        self.QDict[lastStateAction] = lastQTerm1 + alpha * temporalDiff

        # if this time step 1, we should also put lastStateAction into history record
        if len(self.kHop) == 0:
            self.kHop.append(lastStateAction)

        #put currentStateAction into history record
        self.kHop.append(currentStateAction)

    #eta is the learning rate
    def updateParams(self, kHopNeighbors, eta):
        #for t = 0, 1, ..., T, compute the term in g_{i, t}(m) before \nabla
        mutiplier1 = np.zeros(self.currentTimeStep + 1)
        for neighbor in kHopNeighbors:
            for t in range(self.currentTimeStep + 1):
                neighborKHop = neighbor.kHop[t]
                neighborQ = neighbor.getQ(neighborKHop)
                mutiplier1[t] += neighborQ

        for t in range(self.currentTimeStep + 1):
            mutiplier1[t] *= pow(self.gamma, t)
            mutiplier1[t] /= self.nodeNum

        #finish constructing mutiplier1
        #compute the gradient with respect to the parameters associated with s_i(t)
        for t in range(self.currentTimeStep + 1):
            currentState = self.state[t]
            currentAction = self.action[t]
            params = self.paramsDict.get(currentState, self.defaultPolicy)
            probVec = special.softmax(params)
            grad = -probVec
            actionIndex = self.actionList.index(currentAction)
            grad[actionIndex] += 1.0
            self.paramsDict[currentState] = params + eta * mutiplier1[t] * grad


    def setBenchmarkPolicy(self,accessNetwork,noactionProb): # set a naive benchmarkPolicy
        proportionAction = []
        for actionCounter in range(self.actionNum):
            if self.actionList[actionCounter] == -1:
                proportionAction.append(np.log(100*noactionProb/4.0))
            else:
                numNodePerAccess = float(accessNetwork.numNodePerAccess[self.actionList[actionCounter]])
                transmitProb = float(accessNetwork.transmitProb[self.actionList[actionCounter]])
                print('numNodePerAccess = ',numNodePerAccess,' transmitProb = ',transmitProb)
                proportionAction.append( np.log(100*transmitProb/numNodePerAccess))


        for stateInt in range(self.stateNum): # enumerate state
            currentState = self.int2state(stateInt) # turn state integer into binary list
            actionParams = np.ones(self.actionNum,dtype = float) * (-10) # default to be all negative


            if np.all( currentState == 0): # no packet in queue
                actionParams[0] = 10.0 # do nothing
            else:
                actionParams = np.array(proportionAction) # proportional action
            # update paramsDict
            self.paramsDict[tuple(currentState)] = actionParams



    def int2state(self,stateInt):
        currentState = np.zeros(self.ddl,dtype = int)
        stateIntIterate = stateInt
        for i in range(self.ddl):
            currentState[i] = stateIntIterate% self.ddl
            stateIntIterate = stateIntIterate//self.ddl
        return currentState
