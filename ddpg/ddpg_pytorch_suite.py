import sys
sys.path.append("../utils")
sys.path.append("../models")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dm_control import suite
import numpy as np
import utils
import ReplayBuffer
import Noise

critic_lr = 1e-3
actor_lr = 1e-3
tau = 5e-3
batch_size = 100
buffer_size = 1e6
sigma = 0.2
gamma = 0.99
device = torch.device("cuda")


class DDPGActor(nn.Module):
    def __init__(self, state_dim, action_dim, actor_lr, device):
        super(DDPGActor, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actor_lr = actor_lr
        self.device = device

        self.fc1 = nn.Linear(state_dim, 400).to(device)
        self.fc2 = nn.Linear(400, 300).to(device)
        self.fc3 = nn.Linear(300, action_dim).to(device)
        nn.init.uniform_(tensor=self.fc3.weight, a=-3e-3, b=3e-3)
        nn.init.uniform_(tensor=self.fc3.bias, a=-3e-3, b=3e-3)

        self.optimizer = optim.Adam(self.parameters(), lr=actor_lr)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))

        return x


class DDPGCritic(nn.Module):
    def __init__(self, state_dim, action_dim, critic_lr, device):
        super(DDPGCritic, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.critic_lr = critic_lr
        self.device = device

        self.fc1 = nn.Linear(state_dim + action_dim, 400).to(device)
        self.fc2 = nn.Linear(400, 300).to(device)
        self.fc3 = nn.Linear(300, 1).to(device)
        nn.init.uniform_(tensor=self.fc3.weight, a=-3e-4, b=3e-4)
        nn.init.uniform_(tensor=self.fc3.bias, a=-3e-4, b=3e-4)

        self.optimizer = optim.Adam(self.parameters(), lr=self.critic_lr)

    def forward(self, s, a):
        x = torch.cat([s, a], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


def soft_target_update(main, target, tau):
    params_main = list(main.parameters())
    params_target = list(target.parameters())

    assert len(params_main) == len(params_target)

    for pi in range(len(params_main)):
        params_target[pi].data.copy_((1 - tau) * params_target[pi].data + tau * params_main[pi].data)


def target_initialize(main, target):
    params_main = list(main.parameters())
    params_target = list(target.parameters())

    assert len(params_main) == len(params_target)

    for pi in range(len(params_main)):
        params_target[pi].data.copy_(params_main[pi].data)

if __name__ == "__main__":

    env = suite.load(domain_name="cheetah", task_name="run")

    state_dim = utils.state_1d_dim_calc(env)[-1]

    action_dim = env.action_spec().shape[-1]

    replay_buffer = ReplayBuffer.ReplayBuffer(buffer_size=buffer_size)

    ou_noise = Noise.OrnsteinUhlenbeckActionNoise(mu=np.zeros([action_dim]), sigma=sigma * np.ones([action_dim]))

    MSEcriterion = nn.MSELoss()

    actor_main = DDPGActor(state_dim, action_dim, actor_lr, device)
    actor_target = DDPGActor(state_dim, action_dim, actor_lr, device)
    critic_main = DDPGCritic(state_dim, action_dim, critic_lr, device)
    critic_target = DDPGCritic(state_dim, action_dim, critic_lr, device)

    target_initialize(actor_main, actor_target)

    for epi_i in range(1, 1000 + 1):

        ou_noise.reset()
        timestep = env.reset()
        ep_reward = 0.0

        # timestep, reward, discount, observation
        _, _, _, s = timestep
        s = torch.FloatTensor(utils.state_1d_flat(s)).to(device)
        while timestep.last() != True:
            with torch.no_grad():
                a = actor_main.forward(s.view(-1,state_dim)).cpu().numpy()
                a = a + ou_noise()
                a = a[0]

            timestep = env.step(a)
            t, r, _, s2 = timestep

            s2 = torch.FloatTensor(utils.state_1d_flat(s2)).to(device)

            replay_buffer.add(s.cpu().numpy(), a, r, t, s2.cpu().numpy())

            s_batch, a_batch, r_batch, t_batch, s2_batch = replay_buffer.sample_batch(batch_size)
            s_batch = torch.FloatTensor(s_batch).to(device)
            a_batch = torch.FloatTensor(a_batch).to(device)
            r_batch = torch.FloatTensor(r_batch).to(device)
            s2_batch = torch.FloatTensor(s2_batch).to(device)

            with torch.no_grad():
                next_target_q = critic_target.forward(s2_batch,
                                                      actor_target.forward(s2_batch)
                                                      )
                y_i = r_batch.view([-1, 1]) + gamma * next_target_q

            q = critic_main.forward(s_batch, a_batch.view([-1, action_dim]))

            loss = MSEcriterion(q, y_i)

            critic_main.optimizer.zero_grad()
            loss.backward()
            critic_main.optimizer.step()

            # max_q print

            actor_main.optimizer.zero_grad()
            a_out = actor_main.forward(s_batch)
            loss = -critic_main.forward(s_batch, a_out).mean()
            loss.backward()
            actor_main.optimizer.step()

            soft_target_update(actor_main, actor_target, tau)
            soft_target_update(critic_main, critic_target, tau)

            s = s2
            ep_reward += r

        max_q_from_laststep = np.max(q.detach().cpu().numpy())
        print(epi_i,"***",ep_reward,"***",max_q_from_laststep)
