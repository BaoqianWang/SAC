import numpy as np

class GlobalNetwork:
    def __init__(self, nodeNum, k):
        self.nodeNum = nodeNum #the total number of nodes in this network
        self.adjacencyMatrix = np.eye(self.nodeNum, dtype = int) #initialize the adjacency matrix of the global network
        self.k = k #the number of hops used in learning
        self.adjacencyMatrixPower = [np.eye(self.nodeNum, dtype = int)] #cache the powers of the adjacency matrix
        self.neighborDict = {} #use a hashmap to store the ((node, dist), neighbors) pairs which we have computed
        self.addingEdgesFinished = False #have we finished adding edges?


    #finish adding edges, so we can construct the k-hop neighborhood after adding edges
    def finishAddingEdges(self):
        temp = np.eye(self.nodeNum, dtype = int)
        #the d-hop adjacency matrix is stored in self.adjacencyMatrixPower[d]
        for _ in range(self.k):
            temp = np.matmul(temp, self.adjacencyMatrix)
            self.adjacencyMatrixPower.append(temp)
        self.addingEdgesFinished = True
        #print(self.adjacencyMatrixPower)

    #query the d-hop neighborhood of node i, return a list of node indices.
    def findNeighbors(self, i, d):
        if not self.addingEdgesFinished:
            print("Please finish adding edges before call findNeighbors!")
            return -1
        if (i, d) in self.neighborDict: #if we have computed the answer before, return it
            return self.neighborDict[(i, d)]
        neighbors = []
        for j in range(self.nodeNum):
            if self.adjacencyMatrixPower[d][i, j] > 0: #this element > 0 implies that dist(i, j) <= d
                neighbors.append(j)
        self.neighborDict[(i, d)] = neighbors #cache the answer so we can reuse later
        return neighbors

class AccessNetwork(GlobalNetwork):
    def __init__(self, nodeNum, k, accessNum):
        super(AccessNetwork, self).__init__(nodeNum, k)
        self.accessNum = accessNum
        self.accessMatrix = np.zeros((nodeNum, accessNum), dtype = int)

    #add an access point a for node i
    def addAccess(self, i, a):
        self.accessMatrix[i, a] = 1

    #finish adding access points. we can construct the neighbor graph
    def finishAddingAccess(self):
        #use accessMatrix to construct the adjacency matrix of (user) nodes
        self.adjacencyMatrix = np.matmul(self.accessMatrix, np.transpose(self.accessMatrix))

        #calculate the number of users sharing each access point
        self.numNodePerAccess = np.sum(self.accessMatrix,axis = 0)

        super(AccessNetwork, self).finishAddingEdges()

    #find the access points for node i
    def findAccess(self, i):
        accessPoints = []
        for j in range(self.accessNum):
            if self.accessMatrix[i, j] > 0:
                accessPoints.append(j)
        return accessPoints

    def setTransmitProb(self,transmitProb):
        self.transmitProb = transmitProb

def constructGridNetwork(nodeNum, width, height, k,  transmitProb = 'allone'):
    if nodeNum != width * height:
        print("nodeNum does not satisfy the requirement of grid network!", nodeNum, width, height)
        return null

    accessNum = (width + 1) * (height + 1)
    accessNetwork = AccessNetwork(nodeNum = nodeNum, k = k, accessNum = accessNum)

    for i in range(nodeNum):
        upperLeft = i // width  * (width + 1) + i % width
        upperRight = upperLeft + 1
        lowerLeft = upperLeft + width + 1
        lowerRight = lowerLeft + 1
        for a in [upperLeft, upperRight, lowerLeft, lowerRight]:
            accessNetwork.addAccess(i, a)

    accessNetwork.finishAddingAccess()

    # setting transmitProb
    transmitProb = np.random.rand(accessNum)

    accessNetwork.setTransmitProb(transmitProb)

    return accessNetwork
