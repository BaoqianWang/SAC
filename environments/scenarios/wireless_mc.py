import numpy as np
from environments.scenarios.core import UserWorld, UserAgent

class Scenario():
    def __init__(self, grid_size, deadlines):
        self.grid_size = grid_size
        self.deadlines = deadlines
        self.num_agents = grid_size ** 2
        self.num_neighbor = 9 # including itself

    def _calc_mask(self, agent, shape_size):
        delta = [(-1, 1), (-1, -1), (1, -1), (1, 1)]
        for dt in delta:
            row = agent.state.p_pos[0]
            col = agent.state.p_pos[1]
            row_dt = row + dt[0]
            col_dt = col + dt[1]
            if row_dt in range(0, shape_size):
              agent.spin_mask[agent.state.id + shape_size * dt[0]] = 1

            if col_dt in range(0, shape_size):
              agent.spin_mask[agent.state.id + dt[1]] = 1

            if row_dt in range(0, shape_size) and col_dt in range(0, shape_size):
              agent.spin_mask[agent.state.id + shape_size * dt[0] + dt[1]] = 1


    def make_world(self):
        world = UserWorld()
        world.num_neighbor = self.num_neighbor
        world.num_agents = self.num_agents
        world.deadlines = self.deadlines
        world.agents = [UserAgent() for i in range(self.num_agents)]
        world.shape_size = self.grid_size
        world.transmit_rate = np.random.rand((self.grid_size + 1)**2)
        world.arrival_rate = np.random.rand(self.grid_size**2)
        self.reset_world(world)
        return world

    def reset_world(self, world):
        world_mat = np.array(range(np.power(world.shape_size, world.dim_pos)))
        world_mat = world_mat.reshape((world.shape_size,) * world.dim_pos)
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.color = np.array([0.35, 0.35, 0.85])
            agent.state.id = i
            agent.state.p_pos = np.where(world_mat == i)
            #agent.state.packets = [np.random.choice(2) for i in range(self.deadlines)]
            agent.spin_mask = np.zeros(world.num_agents)
            agent.access_rate = world.transmit_rate
            agent.arrival_rate = world.arrival_rate[i]
            #agent.state.packets = [0 for i in range(self.deadlines)]
            # if np.random.rand() < agent.arrival_rate:
            #     agent.state.packets[-1] = 1
            agent.state.packets = [np.random.choice(2) for i in range(self.deadlines)]
            #print(world.num_agents)
            self._calc_mask(agent, world.shape_size)
            #print(agent.spin_mask)
            agent.neighbors = [i for i in range(self.num_agents) if agent.spin_mask[i] == 1]
            #print(agent.neighbors)
            agent.transmit_succ = False

        for i, agent in enumerate(world.agents):
            upperLeft = i // self.grid_size  * (self.grid_size + 1) + i % self.grid_size
            upperRight = upperLeft + 1
            lowerLeft = upperLeft + self.grid_size + 1
            lowerRight = lowerLeft + 1
            agent.state.access = [upperLeft, upperRight, lowerLeft, lowerRight]

    # agent action = [0, 1, 2, 3, 4] 0 means null
    def reward(self, agent, world):

        return 1 if agent.transmit_succ else 0



    def observation(self, agent, world):
        packets = []
        packets += agent.state.packets
        for neighbor in agent.neighbors:
            packets += world.agents[neighbor].state.packets
        packets += [0 for j in range(self.deadlines * (self.num_neighbor - 1 - len(agent.neighbors)))]

        return np.asarray(packets)


    def info(self, agent, world):
        return 0

    def done(self, agent, world):
        return False
