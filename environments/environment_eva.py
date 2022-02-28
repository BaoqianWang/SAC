import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np
import os

# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class MultiAgentEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array']
    }

    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, shared_viewer=True, export_episode=False):

        # np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
        self.world = world
        self.agents = self.world.policy_agents
        # set required vectorized gym env property
        self.n = len(world.policy_agents)
        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback
        # environment parameters
        self.discrete_action_space = True
        self.action_set = [-1, 0, 1, 2, 3]
        # configure spaces
        self.action_space = []
        self.observation_space = []
        for i in range(9):
            self.action_space.append(spaces.Discrete(2))
            self.observation_space.append(spaces.MultiBinary(9 * self.world.deadlines))


    def step(self, action_n):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = []
        self.agents = self.world.policy_agents
        # set action for each agent
        for i, agent in enumerate(self.agents):
            self._set_action(action_n[i], agent)
        # advance world state
        self.world.step()
        # record observation for each agent
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
            reward_n.append(self._get_reward(agent))
            done_n.append(self._get_done(agent))
            info_n.append(self._get_info(agent))
            agent.transmit_succ = False


        return obs_n, reward_n, done_n, info_n

    def reset(self):
        self.reset_callback(self.world)
        obs_n = []
        self.agents = self.world.policy_agents
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
        return obs_n

    # get info used for benchmarking
    def _get_info(self, agent):
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)

    # get observation for a particular agent
    def _get_obs(self, agent):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(agent, self.world)

    # get dones for a particular agent
    # unused right now -- agents are allowed to go beyond the viewing screen
    def _get_done(self, agent):
        if self.done_callback is None:
            return False
        return self.done_callback(agent, self.world)

    # get reward for a particular agent
    def _get_reward(self, agent):
        if self.reward_callback is None:
            return 0.0
        ret = self.reward_callback(agent, self.world)
        # print("ass", agent.color)
        return ret

    # set env action for a particular agent
    def _set_action(self, action, agent):
        # print(action)
        # action = np.random.choice(a = self.action_set, p = action)
        # if action == -1:
        #     agent.action.a = -1
        # else:
        #     agent.action.a = agent.state.access[action]

        # agent.action.a = action
        if action[0] <= 0.2:
            agent.action.a = -1

        if action[0] > 0.2 and action[0] <= 0.4:
            agent.action.a = agent.state.access[0]

        if action[0] > 0.4 and action[0] <= 0.6:
            agent.action.a = agent.state.access[1]

        if action[0] > 0.6 and action[0] <= 0.8:
            agent.action.a = agent.state.access[2]

        if action[0] > 0.8:
            agent.action.a = agent.state.access[3]
