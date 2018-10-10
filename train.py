import torch
import torch.optim as optim
from torch.autograd import Variable

import numpy as np

from wrappers import wrap_deepmind, make_atari, wrap_pytorch
from agent import Agent
from model import DQN
from memory import *

def get_env():
    env = make_atari("PongNoFrameskip-v4")
    env = wrap_deepmind(env)
    env = wrap_pytorch(env)
    return env

def get_epsilon(frame_idx, eps_start=1, eps_final=0.1, decay_win=30000):
    eps = eps_final + (eps_start - eps_final) * np.exp(-1 * frame_idx / decay_win)
    return eps

def train_agent(args, device, verbose=True):
    '''
    Main driver function which trains the agent
    : param args    : all args received from cmd
    : param device  : whther to train on cpu or gpu
    : param verbose : whether to print info on console while training
    '''

    env = get_env()

    # initialize agent according to algorithm
    agent = Agent(args.algorithm, env, args.lr, args.gamma, device)

    # initialize replay memory
    buffer = ReplayBuffer(args.capacity, args.batch_size)

    # initialize number of episodes, rewards and loss
    ep = 0
    total_loss = 0
    total_reward = 0

    S = env.reset()
    for idx in range(1, args.num_frames_to_train + 1):
        eps = get_epsilon(idx)
        var_S = Variable(torch.FloatTensor(S).unsqueeze(0)).to(device)
        A = agent.get_best_action(var_S, eps)
        S_prime, R, is_done, _ = env.step(A)

        buffer.push(S, A, R, S_prime, is_done)
        total_reward += R

        if is_done:
            S = env.reset()
            ep += 1
            if verbose:
                print("Episode: [%3d] complete - reward obtained: [%.2f]" %(ep, total_reward))
            agent.track_statistics(total_loss, total_reward)
            total_loss = 0
            total_reward = 0
        else:
            S = S_prime

        # perform weight updates by sampling from replay memory
        if len(buffer) > args.warm_up:
            batch_S, batch_A, batch_R, batch_Sp, batch_done = buffer.sample()

            # convert into variables
            batch_S = Variable(torch.FloatTensor(batch_S)).to(device)
            batch_Sp = Variable(torch.FloatTensor(batch_Sp)).to(device)
            batch_R = Variable(torch.FloatTensor(batch_R)).to(device)
            batch_done = Variable(torch.FloatTensor(batch_done)).to(device)
            batch_A = Variable(torch.LongTensor(batch_A)).to(device)

            loss = agent.update_params(batch_S, batch_Sp, batch_R, batch_done, batch_A)
            total_loss += loss.item()

        if idx % args.update_target == 0:
            agent.backup()

    # after training is complete, plot collected stats
    agent.save_statistics()
