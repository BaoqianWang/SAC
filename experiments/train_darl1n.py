import argparse
import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import pickle
from mpi4py import MPI
import random
import experiments.helper.tf_util as U
from experiments.helper.maddpg_neighbor import MADDPGAgentTrainer
from experiments.helper.policy_target_policy import PolicyTrainer, PolicyTargetPolicyTrainer
import tensorflow.contrib.layers as layers
import json
import imageio
import joblib

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--eva-max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--max-num-train", type=int, default=2000, help="number of train")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=128, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=32, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--save-dir", type=str, default="../trained_policy/", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=20, help="save model once every time this number of train are completed")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--num-good", type=int, default="0", help="num good")
    parser.add_argument("--num-agents", type=int, default="0", help="num agents")
    parser.add_argument("--num-learners", type=int, default="0", help="num learners")
    parser.add_argument("--max-num-neighbors", type=int, default="0", help="maximum number of  agents in neighbors area")
    parser.add_argument("--seed", type=int, default="1", help="seed for random number")
    return parser.parse_args()


def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out


def make_env(scenario_name, arglist, evaluate=False): ###################
    import importlib
    if evaluate:
        from environments.environment_eva import MultiAgentEnv
        module_name = "environments.scenarios.{}".format(scenario_name)
        scenario_class = importlib.import_module(module_name).Scenario
        # grid size 3, deadline 2.
        scenario = scenario_class(3, 10)
    else:
        from environments.environment_neighbor import MultiAgentEnv
        module_name = "environments.scenarios.{}_neighbor".format(scenario_name)
        scenario_class = importlib.import_module(module_name).Scenario
        scenario = scenario_class(3, 10)

    world = scenario.make_world()
    # create multiagent environment
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env


def get_trainers(env, num_agents, name, obs_shape_n, arglist, session):
    trainers = []
    model = mlp_model
    trainer = MADDPGAgentTrainer
    for i in range(num_agents):
        trainers.append(trainer(
            name+"agent_%d" % i, model, obs_shape_n, session, env.action_space, i, arglist,
            local_q_func=(arglist.good_policy=='ddpg')))
    return trainers


def evaluate_policy(evaluate_env, trainers, num_episode, display = False):
    good_episode_rewards = [0.0]
    adv_episode_rewards = [0.0]
    step = 0
    episode = 0
    frames = []
    obs_n = evaluate_env.reset()
    while True:
        action_n = [agent.action(obs) for agent, obs in zip(trainers,obs_n)]
        new_obs_n, rew_n, done_n, next_info_n = evaluate_env.step(action_n)
        #print(sum(rew_n))
        for i, rew in enumerate(rew_n):
            good_episode_rewards[-1] += rew

        step += 1
        done = all(done_n)
        terminal = (step > (arglist.eva_max_episode_len))

        obs_n = new_obs_n
        info_n = next_info_n

        if done or terminal:
            episode += 1
            if episode >= num_episode:
                break

            good_episode_rewards.append(0)
            adv_episode_rewards.append(0)
            obs_n = evaluate_env.reset()
            step = 0

    return np.mean(good_episode_rewards), np.mean(adv_episode_rewards)


def interact_with_environments(env, trainers, node_id, steps):
    act_d = env.action_space[0].n

    for k in range(steps):
        #obs_pot, neighbor = env.reset(node_id) # Neighbor does not include agent itself.
        obs_pot, neighbor = env.reset(node_id)
        action_n = [np.zeros((act_d))] * env.n # Actions for transition
        # print('before', node_id, obs_pot[node_id])
        action_neighbor = [np.zeros((act_d))] * arglist.max_num_neighbors #The neighbors include the agent itself
        target_action_neighbor = [np.zeros((act_d))] * arglist.max_num_neighbors

        self_action = trainers[node_id].action(obs_pot[node_id])

        action_n[node_id] = self_action
        action_neighbor[0] = self_action

        valid_neighbor = 1
        for i, obs in enumerate(obs_pot):
            if i == node_id: continue
            if len(obs) !=0 :
                other_action = trainers[i].action(obs)
                action_n[i] = other_action
                if neighbor and i in neighbor and valid_neighbor < arglist.max_num_neighbors:
                    action_neighbor[valid_neighbor] = other_action
                    valid_neighbor += 1

        new_obs_neighbor, rew, done_n, next_info_n = env.step(action_n) # Interaction within the neighbor area

        valid_neighbor = 1
        target_action_neighbor[0]=trainers[node_id].target_action(new_obs_neighbor[node_id])

        for k, next in enumerate(new_obs_neighbor):
            if k == node_id: continue
            if len(next) != 0 and valid_neighbor < arglist.max_num_neighbors:
                target_action_neighbor[valid_neighbor] = trainers[k].target_action(next)
                valid_neighbor += 1

        info_n = 0.1
        trainers[node_id].experience(obs_pot[node_id], action_neighbor, new_obs_neighbor[node_id], target_action_neighbor, rew)

        #print(target_action_neighbor)
    return


def touch_path(path):
    dirname = os.path.dirname(path)
    if any(dirname):
        os.makedirs(dirname, exist_ok=True)


def save_weights(trainers, index):
    weight_file_name = os.path.join(arglist.save_dir, 'agent%d.weights' %index)
    touch_path(weight_file_name)
    weight_dict = trainers[index].get_all_weights()
    joblib.dump(weight_dict, weight_file_name)


if __name__== "__main__":
    # MPI initialization.
    comm = MPI.COMM_WORLD
    num_node = comm.Get_size()
    node_id = comm.Get_rank()
    node_name = MPI.Get_processor_name()

    with tf.Session() as session:
        #Parse the parameters
        arglist = parse_args()
        seed = arglist.seed
        gamma = arglist.gamma
        num_agents = arglist.num_agents
        num_learners = arglist.num_learners # In two sided applications, we only train one side.
        assert num_node == num_learners + 1
        np.random.seed(seed)
        tf.set_random_seed(seed)
        random.seed(seed)
        CENTRAL_CONTROLLER = 0
        LEARNERS = [i+CENTRAL_CONTROLLER for i in range(1, 1+num_learners)]
        env = make_env(arglist.scenario, arglist, evaluate= False)
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]

        if (node_id == CENTRAL_CONTROLLER):
            trainers = []
            # The central controller only needs policy parameters to execute the policy for evaluation
            for i in range(num_agents):
                trainers.append(PolicyTrainer("actor" + "agent_%d" % i, mlp_model, obs_shape_n, session, env.action_space, i, arglist, local_q_func=False))
        else:
            trainers = []
            # Trainer needs the MADDPG trainer for its own agent, while only needs policy and target policy for good agents, and policy for adversary agents
            for i in range(num_agents):
                if node_id - 1 == i:
                    trainers.append(MADDPGAgentTrainer("actor" + "agent_%d" % i, mlp_model, obs_shape_n, session, env.action_space, i, arglist, local_q_func=False))
                else: # Adversary agents
                    trainers.append(PolicyTargetPolicyTrainer("actor" + "agent_%d" % i, mlp_model, obs_shape_n, session, env.action_space, i, arglist, local_q_func=False))


        U.initialize()
        final_good_rewards = []
        final_adv_rewards = []
        final_rewards = []
        train_time = []
        global_train_time = []
        ground_global_time = time.time()
        train_step = 0
        iter_step = 0
        num_train = 0

        if (node_id == CENTRAL_CONTROLLER):
            train_start_time = time.time()
            print('Computation scheme: ', 'DARL1N')
            print('Scenario: ', arglist.scenario)
            print('Number of agents: ', num_agents)
            touch_path(arglist.save_dir)
            # if arglist.load_dir == "":
            #     arglist.load_dir = arglist.save_dir
            evaluate_env = make_env(arglist.scenario, arglist, evaluate= True)

        comm.Barrier()
        print('Start training...')
        start_time = time.time()
        while True:
            comm.Barrier()
            if num_train > 0:
                start_master_weights=time.time()
                weights=comm.bcast(weights,root=0)
                end_master_weights=time.time()

            if (node_id in LEARNERS):
                # Receive parameters
                if num_train == 0:
                    env_time1 = time.time()
                    interact_with_environments(env, trainers, node_id-1, 5 * arglist.batch_size)
                    env_time2 = time.time()
                    print('Env interaction time', env_time2 - env_time1)
                else:
                    for i, agent in enumerate(trainers):
                        agent.set_weigths(weights[i+1])
                    interact_with_environments(env, trainers, node_id-1, 4 * arglist.eva_max_episode_len)

                loss = trainers[node_id-1].update(trainers)
                weights = trainers[node_id-1].get_weigths()

            if (node_id == CENTRAL_CONTROLLER):
                weights = None

            weights = comm.gather(weights, root = 0)

            if (node_id in LEARNERS):
                num_train += 1
                if num_train > arglist.max_num_train:
                    save_weights(trainers, node_id - 1)
                    break

            if(node_id == CENTRAL_CONTROLLER):
                if(num_train % arglist.save_rate == 0):
                    for i in range(num_agents):
                        trainers[i].set_weigths(weights[i+1])

                    end_train_time = time.time()
                    #U.save_state(arglist.save_dir, saver=saver)
                    good_reward, adv_reward = evaluate_policy(evaluate_env, trainers, 10, display = False)
                    final_good_rewards.append(good_reward)
                    final_adv_rewards.append(adv_reward)
                    train_time.append(end_train_time - start_time)
                    print('Num of training iteration:', num_train, 'Good Reward:', good_reward, 'Adv Reward:', adv_reward, 'Training time:', round(end_train_time - start_time, 3), 'Global training time:', round(end_train_time- ground_global_time, 3))
                    global_train_time.append(round(end_train_time - ground_global_time, 3))
                    start_time = time.time()

                num_train += 1
                if num_train > arglist.max_num_train:
                    #save_weights(trainers)
                    good_rew_file_name = arglist.save_dir + 'good_agent.pkl'
                    with open(good_rew_file_name, 'wb') as fp:
                        pickle.dump(final_good_rewards, fp)

                    adv_rew_file_name = arglist.save_dir  + 'adv_agent.pkl'
                    with open(adv_rew_file_name, 'wb') as fp:
                        pickle.dump(final_adv_rewards, fp)

                    time_file_name = arglist.save_dir + 'train_time.pkl'
                    with open(time_file_name, 'wb') as fp:
                        pickle.dump(train_time, fp)

                    global_time_file = arglist.save_dir + 'global_time.pkl'
                    with open(global_time_file, 'wb') as fp:
                        pickle.dump(global_train_time, fp)

                    train_end_time = time.time()
                    print('The total training time:', train_end_time - train_start_time)
                    print('Average train time', np.mean(train_time))
                    break
