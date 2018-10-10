import torch
import torch.optim as optim

import numpy as np
import os

from model import DQN, DuellingDQN

class Agent:
    def __init__(self, algo, env, lr, gamma, device):
        self.algorithm = algo
        self.env = env

        if self.algorithm == "duellingDQN":
            self.online_model = DuellingDQN(self.env.observation_space.shape, self.env.action_space.n).to(device)
            self.target_model = DuellingDQN(self.env.observation_space.shape, self.env.action_space.n).to(device)
        else:
            self.online_model = DQN(self.env.observation_space.shape, self.env.action_space.n).to(device)
            self.target_model = DQN(self.env.observation_space.shape, self.env.action_space.n).to(device)

        # target_dqn is initialized to same weights as online_dqn
        self.target_model.load_state_dict(self.online_model.state_dict())

        # adam optimizer for online dqn
        self.optimizer = optim.Adam(self.online_model.parameters(), lr=lr)

        self.gamma = gamma
        self.best_reward = 0
        self.reward_tracker = []
        self.loss_tracker = []

        model_save_path = "saved_models/"
        if not os.path.exists(model_save_path):
            os.mkdir(model_save_path)

        self.root = model_save_path

    def update_params(self, S, S_prime, R, done, A):
        '''
        calculates loss w.r.t algorithm specified
        All parameters are passed as variables on required device
        : param S       : (batch_size x 4 x 84 x 84)
        : param S_prime : (batch_size x 4 x 84 x 84)
        : param R       : (batch_size)
        : param done    : (batch_size)
        : param A       : (batch_size)
        '''
        if self.algorithm == "DQN" or self.algorithm == "duellingDQN":
            with torch.no_grad():
                # use bellman equation
                Q_Sp_A = self.target_model(S_prime).max(1)[0].squeeze()
                target = R + self.gamma * Q_Sp_A * (1 - done)

        elif self.algorithm == "doubleDQN" or self.algorithm == "duelling_doubleDQN":
            with torch.no_grad():

                # V(S') is calculated by using target net
                V_Sp = self.target_model(S_prime)

                # A' is calculated by using online net
                A_Sp = self.online_model(S_prime).max(1)[1].squeeze()

                # estimate of Q(S', A') which is used to evaluate target
                Q_Sp_A = V_Sp.squeeze().gather(1, A_Sp.unsqueeze(1)).squeeze()
                target = R + self.gamma * Q_Sp_A * (1 - done)

        # Q(S, A) is same for all algorithms
        y = self.online_model(S).squeeze().gather(1, A.unsqueeze(1)).squeeze()

        # MSE loss a/c to Bellman equation
        loss = (y - target).pow(2).mean()

        # update params
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def get_best_action(self, state, epsilon):
        '''
        Use epsilon greedy policy to select action
        '''
        if np.random.uniform() > epsilon:
            # exploit
            with torch.no_grad():
                q_value = self.online_model(state)
                action = q_value.max(1)[1].data[0]
        else:
            # explore
            action = np.random.randint(0, self.env.action_space.n)
        return action

    def track_statistics(self, loss, R):
        '''
        Keeps track of episodic rewards and losses, and saves best model on basis of score
        '''
        self.reward_tracker.append(R)
        self.loss_tracker.append(loss)
        if R >= self.best_reward:
            # update best reward and save model giving this result
            self.best_reward = R
            save_name = self.root + self.algorithm + "_best.pth"
            torch.save(self.online_model.state_dict(), save_name)


    def backup(self):
        self.target_model.load_state_dict(self.online_model.state_dict())

    def save_statistics(self):
        '''
        writes loss and rewards for each episode into separate log files
        '''
        f_name = self.root + self.algorithm + "_stats.log"
        with open(f_name, "w") as fp:
            fp.write(self.algorithm + ", rewards, " + str(self.reward_tracker[0]))
            for r in self.reward_tracker[1:]:
                fp.write(", " + str(r))

            fp.write("\n")
            fp.write(self.algorithm + ", loss, " + str(self.loss_tracker[0]))
            for l in self.loss_tracker[1:]:
                fp.write(", " + str(l))
