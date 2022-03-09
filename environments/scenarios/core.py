import numpy as np


class UserAgentState():
  def __init__(self):
    self.id = None
    self.p_pos = None
    self.packets = None


class UserAction(object):
  def __init__(self):
    # action
    self.a = None


class UserAgent():
  def __init__(self):
    # agents are movable by default
    self.movable = False
    self.name = ''
    # -1: observe the whole state, 0: itself, 1: neighbour of 1 unit
    self.spin_mask = None  # the mask for who is neighbours
    self.movable = False
    self.color = None
    # state
    self.state = UserAgentState()
    # action
    self.action = UserAction()
    # script behavior to execute
    self.action_callback = None


# multi-agent world
class UserWorld(object):
  def __init__(self):
    # list of agents and entities (can change at execution-time!)
    self.agents = []
    self.num_agents = 1
    # position dimensionality
    self.dim_pos = 2
    # state dimension
    # color dimensionality
    self.dim_color = 3
    # world size
    self.shape_size = 1
    # User specific
    self.global_state = None  # log all spins
    self.bernulli_p = 0.5

  # return all entities in the world
  @property
  def entities(self):
    return self.agents

  # return all agents controllable by external policies
  @property
  def policy_agents(self):
    return [agent for agent in self.agents if agent.action_callback is None]

  # return all agents controlled by world scripts, no use for now
  @property
  def scripted_agents(self):
    return [agent for agent in self.agents if agent.action_callback is not None]

  # update state of the world
  def step(self):

    # update agent state, and to the global_state
    for agent in self.agents:
      self.update_agent_state(agent)

  def update_agent_state(self, agent):
      sent = False

      if agent.action.a >= 0 and any([value > 0 for value in agent.state.packets]):
          sent = True
          for neighbor in agent.neighbors:
              other_agent = self.agents[neighbor]
              if other_agent.action.a == agent.action.a and any([value > 0 for value in other_agent.state.packets]):
                  sent = False
                  break

          if sent and np.random.uniform() < agent.access_rate[agent.action.a]:
              agent.transmit_succ = True
              for i, value in enumerate(agent.state.packets):
                  if value > 0:
                      agent.state.packets[i] = 0
                      break

      packets = [value for value in agent.state.packets[1:]]
      packets.append(np.random.choice(2))
      # if np.random.uniform() < agent.arrival_rate:
      #     packets.append(1)
      # else:
      #     packets.append(0)

      agent.state.packets = packets
