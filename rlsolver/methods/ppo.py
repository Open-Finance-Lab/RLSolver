import sys
import os
cur_path = os.path.dirname(os.path.abspath(__file__))
rlsolver_path = os.path.join(cur_path, '../../rlsolver')
sys.path.append(os.path.dirname(rlsolver_path))

import random
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from rlsolver.envs.env_ppo import EnvMaxcut
from rlsolver.methods.util_read_data import load_graph_list

@dataclass
class Config:
    start_str: str = 'AUGJpKpXeU9nDPfQf'
    start_val: int = 220
    num_nodes: int = 100

    num_envs: int = 100
    num_steps: int = 100
    num_iterations: int = 1000
    num_minibatches: int = 4
    update_epochs: int = 4
    seed: int = 0

    torch_deterministic: bool = True
    cuda: bool = True
    # Algorithm specific arguments
    learning_rate: float = 2.5e-4
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    # to be filled in runtime
    batch_size: int = 0
    minibatch_size: int = 0


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.num_nodes).prod(), 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.num_nodes).prod(), 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, envs.num_nodes), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


if __name__ == "__main__":
    Config.batch_size = int(Config.num_envs * Config.num_steps)
    Config.minibatch_size = int(Config.batch_size // Config.num_minibatches)

    # TRY NOT TO MODIFY: seeding
    random.seed(Config.seed)
    np.random.seed(Config.seed)
    torch.manual_seed(Config.seed)
    torch.backends.cudnn.deterministic = Config.torch_deterministic

    # simulator
    device = torch.device("cuda" if torch.cuda.is_available() and Config.cuda else "cpu")
    graph_type, num_nodes, graph_id = 'PowerLaw', Config.num_nodes, 0
    graph_name = f'{graph_type}_{num_nodes}_ID{graph_id}'
    graph_list = load_graph_list(graph_name=graph_name)
    envs = EnvMaxcut(args=Config, graph_list=graph_list, device=device, if_bidirectional=True)

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=Config.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros(Config.num_steps, Config.num_envs, envs.num_nodes).to(device)
    actions = torch.zeros((Config.num_steps, Config.num_envs)).to(device)
    logprobs = torch.zeros((Config.num_steps, Config.num_envs)).to(device)
    rewards = torch.zeros((Config.num_steps, Config.num_envs)).to(device)
    cut_values = torch.zeros((Config.num_steps, Config.num_envs)).to(device)
    dones = torch.zeros((Config.num_steps, Config.num_envs)).to(device)
    values = torch.zeros((Config.num_steps, Config.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    next_obs = envs.reset()
    next_done = torch.zeros(Config.num_envs).to(device)

    for iteration in range(1, Config.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if Config.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / Config.num_iterations
            lrnow = frac * Config.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, Config.num_steps):
            global_step += Config.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, next_done, cut_value = envs.step(action)
            rewards[step] = reward.view(-1)
            cut_values[step] = cut_value.view(-1)

        max_value = torch.max(cut_values).item()
        print(iteration, max_value)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(Config.num_steps)):
                if t == Config.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + Config.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + Config.gamma * Config.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        n_nodes = (envs.num_nodes,)
        b_obs = obs.reshape((-1,) + n_nodes)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + ())
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(Config.batch_size)
        clipfracs = []
        for _ in range(Config.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, Config.batch_size, Config.minibatch_size):
                end = start + Config.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds],
                                                                              b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_advantages = b_advantages[mb_inds]
                if Config.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - Config.clip_coef, 1 + Config.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if Config.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -Config.clip_coef,
                        Config.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - Config.ent_coef * entropy_loss + v_loss * Config.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), Config.max_grad_norm)
                optimizer.step()
