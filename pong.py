import torch
from torch.autograd import Variable

import argparse
import os
import time

from train import get_env
from model import *

parser = argparse.ArgumentParser()
parser.add_argument("--algorithm", type=str, default="DQN",
        choices=["DQN", "duellingDQN", "doubleDQN", "duelling_doubleDQN"])
args = parser.parse_args()

env = get_env()

if args.algorithm == "DQN" or args.algorithm == "doubleDQN":
    dqn = DQN(env.observation_space.shape, env.action_space.n)
else:
    dqn = DuellingDQN(env.observation_space.shape, env.action_space.n)

print("Using algorithm: %s" %(args.algorithm))

model_path = "saved_models/" + args.algorithm + "_best.pth"
if os.path.exists(model_path):
    print("Pretrained model found, using it for simulation")
    dqn.load_state_dict(torch.load(model_path))

S = env.reset()
done = False
while not done:
    S = Variable(torch.FloatTensor(S).unsqueeze(0))
    q_vals = dqn(S).max(1)
    A = q_vals[1].item()
    S_prime, R, done, _ = env.step(A)
    S = S_prime
    env.render()
    time.sleep(0.02)
