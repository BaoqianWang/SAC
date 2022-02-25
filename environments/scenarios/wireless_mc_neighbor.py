import numpy as np
from environments.scenarios.core_neighbor import UserWorld, UserAgent

class Scenario():
    def __init__(self, grid_size, deadlines):
        self.grid_size = grid_size
        self.deadlines = deadlines
        self.num_agents = grid_size ** 2
        self.num_neighbor = 5 # including itself

    def _calc_mask(self, agent, shape_size):
        delta = [-1, 1]
        for dt in delta:
            row = agent.state.p_pos[0]
            col = agent.state.p_pos[1]
            row_dt = row + dt
            col_dt = col + dt
            if row_dt in range(0, shape_size):
              agent.spin_mask[agent.state.id + shape_size * dt] = 1

            if col_dt in range(0, shape_size):
              agent.spin_mask[agent.state.id + dt] = 1


    def make_world(self):
        world = UserWorld()
        world.num_neighbor = self.num_neighbor
        world.num_agents = self.num_agents
        world.deadlines = self.deadlines
        world.agents = [UserAgent() for i in range(self.num_agents)]
        world.shape_size = self.grid_size
        world.transmit_rate = np.random.rand((self.grid_size + 1)**2)
        world.arrival_rate = np.random.rand(self.grid_size**2)
        self.reset_world(world, 0)
        return world

    def reset_world(self, world, agent_id):
        world_mat = np.array(range(np.power(world.shape_size, world.dim_pos)))
        world_mat = world_mat.reshape((world.shape_size,) * world.dim_pos)

        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.color = np.array([0.35, 0.35, 0.85])
            agent.state.id = i
            agent.state.p_pos = np.where(world_mat == i)
            agent.state.packets = [np.random.choice(2) for i in range(self.deadlines)]
            agent.spin_mask = np.zeros(world.num_agents)
            agent.access_rate = world.transmit_rate
            agent.arrival_rate = world.arrival_rate[i]
            self._calc_mask(agent, world.shape_size)
            agent.neighbors = [i for i in range(self.num_agents) if agent.spin_mask[i] == 1]
            agent.transmit_succ = False

        for i, agent in enumerate(world.agents):
            upperLeft = i // self.grid_size  * (self.grid_size + 1) + i % self.grid_size
            upperRight = upperLeft + 1
            lowerLeft = upperLeft + self.grid_size + 1
            lowerRight = lowerLeft + 1
            agent.state.access = [upperLeft, upperRight, lowerLeft, lowerRight]


        agent = world.agents[agent_id]
        self_agent_neg = list(np.where(agent.spin_mask == 1)[0])
        neighbor_agent_neighbor = []
        for neighbor_index in self_agent_neg:
            neighbor_index_neg = list(np.where(world.agents[neighbor_index].spin_mask == 1)[0])
            neighbor_agent_neighbor += neighbor_index_neg

        action_agents = self_agent_neg + neighbor_agent_neighbor + [agent_id]
        action_agents = list(set(action_agents))
        return action_agents, self_agent_neg


            #print(agent.neighbors)
    # agent action = [0, 1, 2, 3, 4] 0 means null
    def reward(self, index, world):
        agent = world.agents[index]

        return 1 if agent.transmit_succ else 0



    def observation(self, index, world):
        agent = world.agents[index]
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
