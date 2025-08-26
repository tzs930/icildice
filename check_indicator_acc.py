import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from core.preprocess import preprocess_dataset
import safety_gymnasium as safety_gym
from core.replay_buffer import MDPReplayBuffer
from argparse import ArgumentParser
from itertools import product

class CheckIndicatorAcc(nn.Module):
    def __init__(self, env, configs,
                 train_safe_replay_buffer=None, train_mixed_replay_buffer=None,
                 test_safe_replay_buffer=None, test_unsafe_replay_buffer=None,
                 seed=0):
        
        seed = configs['train']['seed']
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        super(CheckIndicatorAcc, self).__init__()

        self.env = env
        self.obs_dim = env.observation_space.low.size
        self.action_dim = env.action_space.low.size
        self.method = configs['method']

        self.train_safe_replay_buffer = train_safe_replay_buffer
        self.train_mixed_replay_buffer = train_mixed_replay_buffer
        self.test_safe_replay_buffer = test_safe_replay_buffer
        self.test_unsafe_replay_buffer = test_unsafe_replay_buffer

        self.device = configs['device']
        self.n_prior_nets = configs['train']['n_prior_nets']
        self.prior_nets = []
        self.prior_fit_nets = []
        self.prior_hidden_size = configs['train']['prior_net_size']
        self.fittor_hidden_size = configs['train']['fittor_net_size']
        self.prior_output_size = configs['train']['prior_output_size']
        self.gp_weight = float(configs['train']['gp_weight'])

        self.uncertainty_threshold = 100.
        self.uncertainty_safe_quantile = float(configs['train']['uncertainty_safe_quantile']) # e.g. 0.998 Quantile    

        if self.method == 'discriminator' or self.method == 'pu-discriminator':
            self.disc_hidden_size = configs['train']['fittor_net_size']
            self.disc_network = nn.Sequential(
                nn.Linear(self.obs_dim + self.action_dim, self.disc_hidden_size[0], device=self.device),
                # nn.LeakyReLU(0.2, inplace=True),
                nn.ReLU(inplace=True),
                nn.Linear(self.disc_hidden_size[1], 1, device=self.device),
                nn.Sigmoid()
            )
            self.disc_network.requires_grad_ = True
            self.disc_optimizer = optim.Adam(self.disc_network.parameters(), lr=float(configs['train']['lr_prior']))
            self.class_prior = float(configs['train']['class_prior'])

        else:
            for i in range(self.n_prior_nets):
                prior_net = nn.Sequential(
                    nn.Linear(self.obs_dim + self.action_dim, self.prior_hidden_size[0], device=self.device),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.prior_hidden_size[1], self.prior_output_size, device=self.device)
                )
                prior_net.requires_grad_ = False
                self.prior_nets.append(prior_net)
            
                prior_fit_net = nn.Sequential(
                    nn.Linear(self.obs_dim + self.action_dim, self.fittor_hidden_size[0], device=self.device),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.fittor_hidden_size[0], self.fittor_hidden_size[1], device=self.device),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.fittor_hidden_size[1], self.fittor_hidden_size[2], device=self.device),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.fittor_hidden_size[2], self.prior_output_size, device=self.device),
                )
                prior_fit_net.requires_grad_ = True
                self.prior_fit_nets.append(prior_fit_net)
                # self.prior_fit_nets[i].parameters().requires_grad_ = True

            self.prior_fit_params = []
            for i in range(self.n_prior_nets):
                self.prior_fit_params.extend(self.prior_fit_nets[i].parameters())
            self.prior_fit_optimizer = optim.Adam(self.prior_fit_params, lr=float(configs['train']['lr_prior']))
            
            self.uncertainty_estimates = \
                lambda x: [torch.mean((self.prior_nets[i](x) - self.prior_fit_nets[i](x)) ** 2, dim=-1) for i in range(self.n_prior_nets)]
            
            self.var_coef = 1.
            self.aleatoric_var = 0.
            self.current_uc_gradient_penalty = 0.

        self.use_wandb = bool(configs['use_wandb'])
        if self.use_wandb:
            import wandb
            self.wandb = wandb
            self.wandb.init(project=configs['wandb']['project'], entity=configs['wandb']['entity'], config=configs)
        else:
            self.wandb = None

        self.obs_standardize = configs['replay_buffer']['standardize_obs']
        self.act_standardize = configs['replay_buffer']['standardize_act']

    def calculate_gradient_penalty(self, network, expert_s_a, mixed_s_a, type='disc'):
    # Add gradient penalty term
    
        rand_coef = torch.rand(expert_s_a.size(0), 1, device=self.device)
        rand_coef = rand_coef.expand(expert_s_a.size())
        
        # Interpolate between expert and safe samples
        interpolated_s_a = rand_coef * expert_s_a + (1 - rand_coef) * mixed_s_a
        interpolated_s_a.requires_grad_(True)
        
        # Get discriminator output for interpolated samples
        if type == 'disc':
            interpolated_output = network(interpolated_s_a)
        elif type == 'prior':
            interpolated_output = torch.stack([network[i](interpolated_s_a) for i in range(self.n_prior_nets)]) # (n_prior_nets, batch_size, prior_output_size)
                        
        # Calculate gradients with respect to interpolated inputs
        gradients = torch.autograd.grad(
            outputs=interpolated_output,
            inputs=interpolated_s_a,
            grad_outputs=torch.ones_like(interpolated_output),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Calculate gradient penalty
        gradients = gradients.view(-1, gradients.size(-1)) # (n_prior_nets * batch_size, prior_output_size)
        gradient_penalty = ((gradients.norm(2, dim=-1) - 1) ** 2).mean()

        return gradient_penalty

    def train(self, total_iteration=1e6, eval_freq=1000, batch_size=1024):
        max_score = -100000.

        safe_n = self.train_safe_replay_buffer._size
        safe_obs_total = torch.tensor(self.train_safe_replay_buffer._observations[:safe_n], dtype=torch.float32, device=self.device)
        safe_actions_total = torch.tensor(self.train_safe_replay_buffer._actions[:safe_n], dtype=torch.float32, device=self.device)
        safe_s_a_total = torch.cat([safe_obs_total, safe_actions_total], dim=-1)

        for num in range(0, int(total_iteration + 1)):

            train_safe_batch = self.train_safe_replay_buffer.random_batch(batch_size, standardize=self.obs_standardize)
            train_safe_obs = torch.tensor(train_safe_batch['observations'], dtype=torch.float32, device=self.device)
            train_safe_actions = torch.tensor(train_safe_batch['actions'], dtype=torch.float32, device=self.device)
            train_safe_s_a = torch.cat([train_safe_obs, train_safe_actions], dim=-1)

            train_mixed_batch = self.train_mixed_replay_buffer.random_batch(batch_size, standardize=self.obs_standardize)
            train_mixed_obs = torch.tensor(train_mixed_batch['observations'], dtype=torch.float32, device=self.device)
            train_mixed_actions = torch.tensor(train_mixed_batch['actions'], dtype=torch.float32, device=self.device)
            train_mixed_s_a = torch.cat([train_mixed_obs, train_mixed_actions], dim=-1)

            if self.method == 'discriminator':
                self.disc_optimizer.zero_grad()

                expert_disc = self.disc_network(train_safe_s_a)
                expert_disc_loss = -torch.log(expert_disc + 1e-10).mean()

                mixed_disc = self.disc_network(train_mixed_s_a)
                mixed_disc_loss = -torch.log(1 - mixed_disc + 1e-10).mean()

                disc_loss = expert_disc_loss + mixed_disc_loss

                if self.gp_weight > 0:
                    self.current_disc_gradient_penalty = self.calculate_gradient_penalty(self.disc_network, train_safe_s_a, train_mixed_s_a, type='disc')
                    disc_loss += self.gp_weight * self.current_disc_gradient_penalty
                else:
                    self.current_disc_gradient_penalty = 0.

                disc_loss.backward()
                self.disc_optimizer.step()

            elif self.method == 'pu-discriminator':
                self.disc_optimizer.zero_grad()

                expert_disc = self.disc_network(train_safe_s_a)
                expert_disc_loss = -torch.log(expert_disc + 1e-10).mean() * self.class_prior

                mixed_disc = self.disc_network(train_mixed_s_a)
                mixed_disc_loss = -torch.log(1 - mixed_disc + 1e-10).mean()
                mixed_disc_loss += (- self.class_prior) * (-torch.log(1 - expert_disc + 1e-10).mean())

                disc_loss = expert_disc_loss + mixed_disc_loss

                if self.gp_weight > 0:
                    self.current_disc_gradient_penalty = self.calculate_gradient_penalty(self.disc_network, train_safe_s_a, train_mixed_s_a, type='disc')
                    disc_loss += self.gp_weight * self.current_disc_gradient_penalty
                else:
                    self.current_disc_gradient_penalty = 0.

                disc_loss.backward()
                self.disc_optimizer.step()

            else:
                self.prior_fit_optimizer.zero_grad()

                prior_outputs = [self.prior_nets[i](train_safe_s_a).detach() for i in range(self.n_prior_nets)]
                prior_fit_outputs = [self.prior_fit_nets[i](train_safe_s_a) for i in range(self.n_prior_nets)]

                prior_fit_loss = [torch.mean((prior_outputs[i] - prior_fit_outputs[i]) ** 2, dim=-1) for i in range(self.n_prior_nets)]
                prior_fit_loss = torch.stack(prior_fit_loss).mean()

                if self.gp_weight > 0:
                    self.current_uc_gradient_penalty = self.calculate_gradient_penalty(self.prior_fit_nets, train_safe_s_a, train_mixed_s_a, type='prior')
                    prior_fit_loss += self.gp_weight * self.current_uc_gradient_penalty
                else:
                    self.current_uc_gradient_penalty = 0.

                prior_fit_loss.backward()
                self.prior_fit_optimizer.step()

            if num % eval_freq == 0:
                if self.method == 'discriminator' or self.method == 'pu-discriminator':
                    train_safe_disc_total = self.disc_network(safe_s_a_total)
                    self.uncertainty_threshold = torch.quantile(train_safe_disc_total, 1 - self.uncertainty_safe_quantile)

                    test_safe_batch = self.test_safe_replay_buffer.random_batch(batch_size, standardize=self.obs_standardize)
                    test_safe_obs = torch.tensor(test_safe_batch['observations'], dtype=torch.float32, device=self.device)
                    test_safe_actions = torch.tensor(test_safe_batch['actions'], dtype=torch.float32, device=self.device)
                    test_safe_s_a = torch.cat([test_safe_obs, test_safe_actions], dim=-1)

                    test_unsafe_batch = self.test_unsafe_replay_buffer.random_batch(batch_size, standardize=self.obs_standardize)
                    test_unsafe_obs = torch.tensor(test_unsafe_batch['observations'], dtype=torch.float32, device=self.device)
                    test_unsafe_actions = torch.tensor(test_unsafe_batch['actions'], dtype=torch.float32, device=self.device)
                    test_unsafe_s_a = torch.cat([test_unsafe_obs, test_unsafe_actions], dim=-1)
                    test_unsafe_costs = torch.tensor(test_unsafe_batch['costs'], dtype=torch.float32, device=self.device).reshape(-1)

                    test_safe_disc = self.disc_network(test_safe_s_a)
                    test_unsafe_disc = self.disc_network(test_unsafe_s_a)

                    test_safe_n_correct = (test_safe_disc > self.uncertainty_threshold).sum()
                    test_n_correct_safe_acc = test_safe_n_correct / batch_size

                    test_unsafe_n_correct = (test_unsafe_disc < self.uncertainty_threshold).sum()
                    test_n_correct_unsafe_acc = test_unsafe_n_correct / batch_size

                    if test_unsafe_costs.sum() > 0:
                        cost_making_unsafe_indices = (test_unsafe_costs > 0).cpu().detach().numpy()
                        cost_making_unsafe_n_correct = (test_unsafe_disc[cost_making_unsafe_indices] < self.uncertainty_threshold).sum()
                        n_correct_cost_making_unsafe_acc = cost_making_unsafe_n_correct / cost_making_unsafe_indices.sum()
                    else:
                        n_correct_cost_making_unsafe_acc = 0.
                    
                    f1_score = 2 * test_n_correct_safe_acc * test_n_correct_unsafe_acc / (test_n_correct_safe_acc + test_n_correct_unsafe_acc)

                    if self.use_wandb:
                        self.wandb.log({
                            'test/safe_acc': test_n_correct_safe_acc,
                            'test/unsafe_acc': test_n_correct_unsafe_acc,
                            'test/cost_making_unsafe_acc': n_correct_cost_making_unsafe_acc,
                            'test/f1_score': f1_score,
                            'uncertainty_threshold': self.uncertainty_threshold,
                            'nonzero_cost_num': cost_making_unsafe_indices.sum(),
                        }, step=num)
                    else:
                        print(f'Iteration={num}: Test safe accuracy: {test_n_correct_safe_acc}, Test unsafe accuracy: {test_n_correct_unsafe_acc}, Test f1 score: {f1_score}, Cost making unsafe accuracy: {n_correct_cost_making_unsafe_acc}, Threshold: {self.uncertainty_threshold}')

                    
                else:
                    uncertainty_estimates = torch.stack(self.uncertainty_estimates(safe_s_a_total)) # (n_prior_nets, batch_size, prior_output_size)
                    uncertainty_estimates_mean = uncertainty_estimates.mean(dim=0) # (batch_size, prior_output_size)
                    self.uncertainty_threshold = np.quantile(uncertainty_estimates_mean.cpu().detach().numpy(), self.uncertainty_safe_quantile) # Select 99.9% Quantile as threshold

                    # Test safe
                    test_safe_batch = self.test_safe_replay_buffer.random_batch(batch_size, standardize=self.obs_standardize)
                    test_safe_obs = torch.tensor(test_safe_batch['observations'], dtype=torch.float32, device=self.device)
                    test_safe_actions = torch.tensor(test_safe_batch['actions'], dtype=torch.float32, device=self.device)
                    test_safe_s_a = torch.cat([test_safe_obs, test_safe_actions], dim=-1)

                    safe_error = torch.stack(self.uncertainty_estimates(test_safe_s_a))
                    safe_error_mean = safe_error.mean(dim=0)
                    safe_uncertainties = safe_error_mean #+ self.var_coef * safe_error_var - self.aleatoric_var
                    safe_uncertainties = safe_uncertainties.cpu().detach().numpy()
                    test_n_correct_safe = (safe_uncertainties < self.uncertainty_threshold).sum()
                    test_n_correct_safe_acc = test_n_correct_safe / batch_size

                    # Test unsafe
                    test_unsafe_batch = self.test_unsafe_replay_buffer.random_batch(batch_size, standardize=self.obs_standardize)
                    test_unsafe_obs = torch.tensor(test_unsafe_batch['observations'], dtype=torch.float32, device=self.device)
                    test_unsafe_actions = torch.tensor(test_unsafe_batch['actions'], dtype=torch.float32, device=self.device)
                    test_unsafe_s_a = torch.cat([test_unsafe_obs, test_unsafe_actions], dim=-1)

                    test_unsafe_costs = torch.tensor(test_unsafe_batch['costs'], dtype=torch.float32, device=self.device).reshape(-1)

                    unsafe_error = torch.stack(self.uncertainty_estimates(test_unsafe_s_a)) # (n_prior_nets, batch_size, )
                    unsafe_error_mean = unsafe_error.mean(dim=0) # (batch_size, )
                    unsafe_uncertainties = unsafe_error_mean #+ self.var_coef * unsafe_error_var - self.aleatoric_var
                    unsafe_uncertainties = unsafe_uncertainties.cpu().detach().numpy() # (batch_size, )
                    # print(f'unsafe_uncertainties: {unsafe_uncertainties}')
                    test_n_correct_unsafe = (unsafe_uncertainties >= self.uncertainty_threshold).sum()
                    test_n_correct_unsafe_acc = test_n_correct_unsafe / batch_size
                    nonzero_cost_indices = (test_unsafe_costs > 0).cpu().detach().numpy()
                
                    if nonzero_cost_indices.sum() > 0:
                        cost_making_unsafe_uncertainties = unsafe_uncertainties[nonzero_cost_indices]
                        n_correct_cost_making_unsafe = (cost_making_unsafe_uncertainties >= self.uncertainty_threshold).sum()
                        n_correct_cost_making_unsafe_acc = n_correct_cost_making_unsafe / nonzero_cost_indices.sum()
                    else:
                        n_correct_cost_making_unsafe_acc = 0.

                    if self.use_wandb:
                        self.wandb.log({
                        'test/safe_acc': test_n_correct_safe_acc,
                        'test/unsafe_acc': test_n_correct_unsafe_acc,
                        'test/cost_making_unsafe_acc': n_correct_cost_making_unsafe_acc,
                        'stats/mean_unsafe_uncertainties': unsafe_uncertainties.mean(),
                        'stats/std_unsafe_uncertainties': unsafe_uncertainties.std(),
                        'stats/mean_safe_uncertainties': safe_uncertainties.mean(),
                        'stats/std_safe_uncertainties': safe_uncertainties.std(),
                        'uncertainty_threshold': self.uncertainty_threshold,
                        'nonzero_cost_num': nonzero_cost_indices.sum(),
                        }, step=num)
                    else:
                        print(f'Iteration={num}: Test safe accuracy: {test_n_correct_safe_acc}, Test unsafe accuracy: {test_n_correct_unsafe_acc}, Mean safe uncertainties: {safe_uncertainties.mean()}, Mean unsafe uncertainties: {unsafe_uncertainties.mean()}, Threshold: {self.uncertainty_threshold}')

                
if __name__ == '__main__':
    configs = {
        'env_id': 'SafetyPointCircle1-v0',
        'method': 'prior',
        'device': 'cuda',
        'use_wandb': True,
        'wandb': {
            'project': 'check-indicator-acc-PU',
            'entity': 'tzs930',
        },
        'replay_buffer': {
            'max_replay_buffer_size': 10000000,
            'standardize_obs': False,
            'standardize_act': False,
        },
        'dataset': {
            'dataset_path': 'dataset/SafetyPointCircle1-v0',
            'train_safe': {
                'types': ['safe-expert-v0', 'safe-medium-v0'],
                'start_indices': [0, 0],
                'num_trajs': [100, 100],
            },
            'train_mixed': {
                'types': ['safe-expert-v0', 'safe-medium-v0', 'unsafe-expert-v0', 'unsafe-medium-v0'],
                'start_indices': [100, 100, 100, 100],
                'num_trajs': [200, 200, 200, 200],
            },
            'test_safe': {
                'types': ['safe-expert-v0', 'safe-medium-v0'],
                'start_indices': [500, 500],
                'num_trajs': [500, 500],
            },
            'test_unsafe': {
                'types': ['unsafe-expert-v0'],
                'start_indices': [0],
                'num_trajs': [1000],
            },
        },
        'train': {
            'seed': 0,
            'total_iteration': 500000,
            'eval_freq': 10000,
            'batch_size': 512,
            'n_prior_nets': 1,
            'prior_net_size': [128, 128],
            'fittor_net_size': [128, 128, 128, 128],
            'prior_output_size': 100,
            'gp_weight': 0.,
            'lr_prior': 3e-4,
            'uncertainty_safe_quantile': 0.999,
        }
    }

    parser = ArgumentParser()
    parser.add_argument("--pid", help="process_id", default=0, type=int)
    args = parser.parse_args()
    pid = args.pid

    configs['pid'] = pid

    hp_grids = {
        "method": ['pu-discriminator', 'discriminator'],
        "env_id": ['SafetyHopperVelocity-v1', 'SafetyPointCircle2-v0', 'SafetyPointCircle1-v0',  'SafetyHalfCheetahVelocity-v1', 'SafetyWalker2dVelocity-v1', 'SafetyAntVelocity-v1'],
        "train/uncertainty_safe_quantile":  [0.9, 0.95, 0.99],
        "train/gp_weight":                  [10.0, 0.0],
        # "train/n_prior_nets":               [2,  4,  8,  16, 32],
        # "train/prior_output_size":          [10, 20, 30, 40, 50],
        "train/seed":                       [0, 1, 2],
        "train/class_prior":                [0.5],
    }
    # 3*5*5*3 = 225

    hp_values = list(product(*hp_grids.values()))[pid]

    for key, value in zip(hp_grids.keys(), hp_values):
        if '/' in key:
            key1 = key.split('/')[0]
            key2 = key.split('/')[1]
            configs[key1][key2] = value
        else:
            configs[key] = value

    configs['dataset']['dataset_path'] = f'dataset/{configs["env_id"]}'

    env = safety_gym.make(configs['env_id'])
    obs_dim = env.observation_space.low.size
    action_dim = env.action_space.low.size

    train_safe_data_dict, n_train_safe = preprocess_dataset(configs['dataset']['dataset_path'],
                                                            configs['dataset']['train_safe']['types'], 
                                                            configs['dataset']['train_safe']['start_indices'], 
                                                            num_rollouts=configs['dataset']['train_safe']['num_trajs'])

    train_mixed_data_dict, n_train_mixed = preprocess_dataset(configs['dataset']['dataset_path'],
                                                            configs['dataset']['train_mixed']['types'], 
                                                            configs['dataset']['train_mixed']['start_indices'], 
                                                            num_rollouts=configs['dataset']['train_mixed']['num_trajs'])

    test_safe_data_dict, n_test_safe = preprocess_dataset(configs['dataset']['dataset_path'],
                                                            configs['dataset']['test_safe']['types'], 
                                                            configs['dataset']['test_safe']['start_indices'], 
                                                            num_rollouts=configs['dataset']['test_safe']['num_trajs'])

    test_unsafe_data_dict, n_test_unsafe = preprocess_dataset(configs['dataset']['dataset_path'],
                                                            configs['dataset']['test_unsafe']['types'], 
                                                            configs['dataset']['test_unsafe']['start_indices'], 
                                                            num_rollouts=configs['dataset']['test_unsafe']['num_trajs'])

    train_safe_replay_buffer = MDPReplayBuffer(
        configs['replay_buffer']['max_replay_buffer_size'],
        env,
        obs_dim=obs_dim
    )
    train_safe_replay_buffer.add_path(train_safe_data_dict)

    train_mixed_replay_buffer = MDPReplayBuffer(
        configs['replay_buffer']['max_replay_buffer_size'],
        env,
        obs_dim=obs_dim
    )
    train_mixed_replay_buffer.add_path(train_mixed_data_dict)

    test_safe_replay_buffer = MDPReplayBuffer(
        configs['replay_buffer']['max_replay_buffer_size'],
        env,
        obs_dim=obs_dim
    )
    test_safe_replay_buffer.add_path(test_safe_data_dict)

    test_unsafe_replay_buffer = MDPReplayBuffer(
        configs['replay_buffer']['max_replay_buffer_size'],
        env,
        obs_dim=obs_dim
    )
    test_unsafe_replay_buffer.add_path(test_unsafe_data_dict)

    check_indicator_acc = CheckIndicatorAcc(env, configs,
                                            train_safe_replay_buffer=train_safe_replay_buffer,
                                            train_mixed_replay_buffer=train_mixed_replay_buffer,
                                            test_safe_replay_buffer=test_safe_replay_buffer,
                                            test_unsafe_replay_buffer=test_unsafe_replay_buffer)
    check_indicator_acc.train(total_iteration=configs['train']['total_iteration'],
                              eval_freq=configs['train']['eval_freq'],
                              batch_size=configs['train']['batch_size'])
