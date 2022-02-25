import tensorflow as tf
from learner import Learner
import time
import random
import copy
import numpy as np
import matplotlib.pyplot as plt
import time
from statistics import mean
import joblib

# Set the random number generation seed
random.seed(30)
gamma=0.95
epsilon=0.001
def make_env(scenario_name="wireless_mc"):
    import importlib
    from environments.environment import MultiAgentEnv
    from  environments.scenarios.wireless_mc import Scenario
    #import Scenario
    #module_name = "environments.scenarios.{}".format(scenario_name)
    #scenario_class = importlib.import_module(module_name).Scenario
    # load scenario from script
    # grid size: 3, d: 2
    scenario = Scenario(3, 2)
    # create world
    world = scenario.make_world()
    # create multiagent environment
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env

def get_learners(env, session, num_learners, obs_shape_n, num_actions):
    Learners=[]
    for i in range(num_learners):
        Learners.append(Learner(env, session,"agent_%d" %i, obs_shape_n, num_actions,i, gamma, epsilon))
    return Learners


def interact_with_environment(env, Learners, steps):
    step = 0
    obs_n = env.reset()
    rewards = []
    while step < steps:
        actions = [learner.action([obs_n[i]]) for i, learner in enumerate(Learners)]

        next_obs, rew_n, done_n, info_n = env.step(actions)
        rewards.append(sum(rew_n))
        for i, learner in enumerate(Learners):
            neighbor_actions = [actions[i]]

            for id in env.world.agents[i].neighbors:
                neighbor_actions.append(actions[id])

            neighbor_actions += [0 for j in range(env.world.num_neighbor-1-len(neighbor_actions))]
            learner.environment.append([obs_n[i], neighbor_actions, rew_n[i]])

        obs_n = copy.deepcopy(next_obs)
        step += 1

    return mean(rewards)



if __name__=="__main__":

    with tf.Session() as session:
        tf.set_random_seed(30)
        random.seed(30)

        num_learners=9 # This is determined in the environment
        num_actions=5
        iter=0
        max_iter=2000
        steps=25
        # Create environment
        env = make_env()

        # Environment initialization
        obs_shape_n = [env.observation_space[0].shape[0]]
        #print(obs_shape_n)
        # Learners initialization
        Learners = get_learners(env, session, num_learners, obs_shape_n, num_actions)

        #Initialize all global variables
        session.run(tf.global_variables_initializer())

        #Environment reset
        obs_n = env.reset()
        episode_reward=[]
        start_time=time.time()
        plot_reward=[]
        while(iter < max_iter):
            reward_average = interact_with_environment(env, Learners, steps)
            episode_reward.append(reward_average)
            iter += 1
            #Train learners
            start_train=time.time()
            Q_values = []
            for j in range(num_learners):
                Q_values.append(Learners[j].update_Q())

            neighbor_Q_values = []
            for k in range(num_learners):
                neighbor_Q_value = [0 for t in range(steps)]
                num_neighbor = len(env.world.agents[k].neighbors)
                for neighbor in env.world.agents[k].neighbors:
                    neighbor_Q_value = [(neighbor_Q_value[i] + Q_values[neighbor][i]) for i in range(steps)]
                neighbor_Q_value = [steps*gamma**i*value/num_neighbor for i, value in enumerate(neighbor_Q_value)]
                neighbor_Q_values.append(neighbor_Q_value)


            for i in range(num_learners):
                #print(len(neighbor_Q_values[i]))
                Learners[i].learn(neighbor_Q_values[i])

            end_train=time.time()

            # print(Learners[0].observation)
            #print('iteration', iter)
            if(iter%200==0):
                end_time=time.time()
                print("steps: {}, episode reward: {}, time: {}".format(iter, np.mean(episode_reward[-100:]), end_time-start_time))
                plot_reward.append(np.mean(episode_reward[-100:]))
                start_time=time.time()

            for i in range(num_learners):
                Learners[i].reset_episode()

        # plt.plot(plot_reward)
        # plt.show()
