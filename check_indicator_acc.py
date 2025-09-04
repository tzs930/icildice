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
import copy
from sklearn.metrics import roc_auc_score
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor

class CheckIndicatorAcc(nn.Module):
    def __init__(self, env, configs,
                 train_safe_replay_buffer=None, train_mixed_replay_buffer=None,
                 train_unsafe_replay_buffer=None,
                 valid_safe_replay_buffer=None, valid_mixed_replay_buffer=None,
                 valid_unsafe_replay_buffer=None,
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
        self.train_unsafe_replay_buffer = train_unsafe_replay_buffer
        self.valid_safe_replay_buffer = valid_safe_replay_buffer
        self.valid_mixed_replay_buffer = valid_mixed_replay_buffer
        self.valid_unsafe_replay_buffer = valid_unsafe_replay_buffer
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

        if self.method == 'discriminator' or self.method == 'pu-discriminator' or self.method == 'pn-discriminator':
            self.disc_hidden_size = configs['train']['fittor_net_size']
            self.disc_network = nn.Sequential(
                nn.Linear(self.obs_dim + self.action_dim, self.disc_hidden_size[0], device=self.device),
                nn.ReLU(inplace=True),
                nn.Linear(self.disc_hidden_size[0], self.disc_hidden_size[1], device=self.device),
                nn.ReLU(inplace=True),
                nn.Linear(self.disc_hidden_size[1], 1, device=self.device),
                nn.Sigmoid()
            )
            self.disc_network.requires_grad_ = True
            self.disc_optimizer = optim.Adam(self.disc_network.parameters(), lr=float(configs['train']['lr_prior']))
            self.class_prior = float(configs['train']['class_prior'])

        elif 'en-discriminator' in self.method:
            self.pretrain_steps = 500000
            self.disc_hidden_size = configs['train']['fittor_net_size']
            
            self.pu_disc_network = nn.Sequential(
                nn.Linear(self.obs_dim + self.action_dim, self.disc_hidden_size[0], device=self.device),
                nn.ReLU(inplace=True),
                nn.Linear(self.disc_hidden_size[0], self.disc_hidden_size[1], device=self.device),
                nn.ReLU(inplace=True),
                nn.Linear(self.disc_hidden_size[1], 1, device=self.device),
                nn.Sigmoid()
            )
            self.pu_disc_network.requires_grad_ = True
            self.pu_disc_optimizer = optim.Adam(self.pu_disc_network.parameters(), lr=float(configs['train']['lr_prior']))
            self.best_pu_disc_network = copy.deepcopy(self.pu_disc_network)

            self.pu_gp_weight = float(configs['train']['pu_gp_weight'])

            self.disc_network = nn.Sequential(
                nn.Linear(self.obs_dim + self.action_dim, self.disc_hidden_size[0], device=self.device),
                nn.ReLU(inplace=True),
                nn.Linear(self.disc_hidden_size[0], self.disc_hidden_size[1], device=self.device),
                nn.ReLU(inplace=True),
                nn.Linear(self.disc_hidden_size[1], 1, device=self.device),
                nn.Sigmoid()
            )

            self.best_disc_network = copy.deepcopy(self.disc_network)
            self.disc_network.requires_grad_ = True
            self.disc_optimizer = optim.Adam(self.disc_network.parameters(), lr=float(configs['train']['lr_prior']))

        elif self.method == 'ocsvm':
            self.ocsvm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.05)
            
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

        self.early_stopping_patience = 10000

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

        # safe_n = self.train_safe_replay_buffer._size
        # safe_obs_total = torch.tensor(self.train_safe_replay_buffer._observations[:safe_n], dtype=torch.float32, device=self.device)
        # safe_actions_total = torch.tensor(self.train_safe_replay_buffer._actions[:safe_n], dtype=torch.float32, device=self.device)
        # safe_s_a_total = torch.cat([safe_obs_total, safe_actions_total], dim=-1)

        train_safe_n = self.train_safe_replay_buffer._size
        train_safe_obs_total = torch.tensor(self.train_safe_replay_buffer._observations[:train_safe_n], dtype=torch.float32, device=self.device)
        train_safe_actions_total = torch.tensor(self.train_safe_replay_buffer._actions[:train_safe_n], dtype=torch.float32, device=self.device)
        train_safe_s_a_total = torch.cat([train_safe_obs_total, train_safe_actions_total], dim=-1)

        train_mixed_n = self.train_mixed_replay_buffer._size
        valid_mixed_n = self.valid_mixed_replay_buffer._size

        valid_safe_n = self.valid_safe_replay_buffer._size
        valid_safe_obs_total = torch.tensor(self.valid_safe_replay_buffer._observations[:valid_safe_n], dtype=torch.float32, device=self.device)
        valid_safe_actions_total = torch.tensor(self.valid_safe_replay_buffer._actions[:valid_safe_n], dtype=torch.float32, device=self.device)
        valid_safe_s_a_total = torch.cat([valid_safe_obs_total, valid_safe_actions_total], dim=-1)

        # if self.method == 'en-discriminator-prior':
        safe_s_a_total = torch.vstack([train_safe_s_a_total, valid_safe_s_a_total])
        total_safe_n = train_safe_n + valid_safe_n
        total_mixed_n = train_mixed_n + valid_mixed_n

        valid_mixed_n = self.valid_mixed_replay_buffer._size
        valid_mixed_obs_total = torch.tensor(self.valid_mixed_replay_buffer._observations[:valid_mixed_n], dtype=torch.float32, device=self.device)
        valid_mixed_actions_total = torch.tensor(self.valid_mixed_replay_buffer._actions[:valid_mixed_n], dtype=torch.float32, device=self.device)
        valid_mixed_s_a_total = torch.cat([valid_mixed_obs_total, valid_mixed_actions_total], dim=-1)
        valid_s_a_total = torch.vstack([valid_safe_s_a_total, valid_mixed_s_a_total])

        min_valid_loss = 10000.
        last_update_num = 0 # for early stopping

        if self.method == 'ocsvm':
            self.ocsvm.fit(train_safe_s_a_total.cpu().detach().numpy())

            test_safe_batch = self.test_safe_replay_buffer.random_batch(batch_size, standardize=self.obs_standardize)
            test_safe_obs = torch.tensor(test_safe_batch['observations'], dtype=torch.float32, device=self.device)
            test_safe_actions = torch.tensor(test_safe_batch['actions'], dtype=torch.float32, device=self.device)
            test_safe_s_a = torch.cat([test_safe_obs, test_safe_actions], dim=-1)

            test_unsafe_batch = self.test_unsafe_replay_buffer.random_batch(batch_size, standardize=self.obs_standardize)
            test_unsafe_obs = torch.tensor(test_unsafe_batch['observations'], dtype=torch.float32, device=self.device)
            test_unsafe_actions = torch.tensor(test_unsafe_batch['actions'], dtype=torch.float32, device=self.device)
            test_unsafe_s_a = torch.cat([test_unsafe_obs, test_unsafe_actions], dim=-1)
            test_unsafe_costs = torch.tensor(test_unsafe_batch['costs'], dtype=torch.float32, device=self.device).reshape(-1)

            # if 'ocsvm' in self.method:
            test_safe_disc = self.ocsvm.decision_function(test_safe_s_a.cpu().detach().numpy())
            test_unsafe_disc = self.ocsvm.decision_function(test_unsafe_s_a.cpu().detach().numpy())
            # test_predictions = torch.cat([test_safe_disc, test_unsafe_disc], dim=0).cpu().detach().numpy().flatten()
            test_predictions = np.concatenate([test_safe_disc, test_unsafe_disc])
            test_labels = np.concatenate([np.ones(test_safe_disc.shape[0]), np.zeros(test_unsafe_disc.shape[0])])
            test_auc_roc = roc_auc_score(test_labels, test_predictions)
            print(f'OCSVM: Test AUC-ROC: {test_auc_roc:.4f}')

            nonzero_cost_indices = (test_unsafe_costs > 0).cpu().detach().numpy()
            if nonzero_cost_indices.sum() > 0:
                # cost_making_unsafe_uncertainties = 
                cost_making_unsafe_s_a = test_unsafe_s_a.cpu().detach().numpy()[nonzero_cost_indices]
                n_correct_cost_making_unsafe = (self.ocsvm.predict(cost_making_unsafe_s_a) < 0).sum()
                n_correct_cost_making_unsafe_acc = n_correct_cost_making_unsafe / nonzero_cost_indices.sum()
            else:
                n_correct_cost_making_unsafe_acc = 0.
            
            test_positive_acc = (self.ocsvm.predict(test_safe_s_a.cpu().detach().numpy()) > 0).mean()
            test_negative_acc = (self.ocsvm.predict(test_unsafe_s_a.cpu().detach().numpy()) < 0).mean()
            print(f'OCSVM: Test positive accuracy: {test_positive_acc:.4f}, Test negative accuracy: {test_negative_acc:.4f}')
            print(f'OCSVM: Test cost-makers accuracy: {n_correct_cost_making_unsafe_acc:.4f}')

        if self.method == 'lof':
            lof = LocalOutlierFactor(n_neighbors=5, novelty=True)
            lof.fit(train_safe_s_a_total.cpu().detach().numpy())

            test_safe_batch = self.test_safe_replay_buffer.random_batch(batch_size, standardize=self.obs_standardize)
            test_safe_obs = torch.tensor(test_safe_batch['observations'], dtype=torch.float32, device=self.device)
            test_safe_actions = torch.tensor(test_safe_batch['actions'], dtype=torch.float32, device=self.device)
            test_safe_s_a = torch.cat([test_safe_obs, test_safe_actions], dim=-1)

            test_unsafe_batch = self.test_unsafe_replay_buffer.random_batch(batch_size, standardize=self.obs_standardize)
            test_unsafe_obs = torch.tensor(test_unsafe_batch['observations'], dtype=torch.float32, device=self.device)
            test_unsafe_actions = torch.tensor(test_unsafe_batch['actions'], dtype=torch.float32, device=self.device)
            test_unsafe_s_a = torch.cat([test_unsafe_obs, test_unsafe_actions], dim=-1)
            test_unsafe_costs = torch.tensor(test_unsafe_batch['costs'], dtype=torch.float32, device=self.device).reshape(-1)

            test_safe_disc = lof.decision_function(test_safe_s_a.cpu().detach().numpy())
            test_unsafe_disc = lof.decision_function(test_unsafe_s_a.cpu().detach().numpy())
            test_predictions = np.concatenate([test_safe_disc, test_unsafe_disc])
            test_labels = np.concatenate([np.ones(test_safe_disc.shape[0]), np.zeros(test_unsafe_disc.shape[0])])
            test_auc_roc = roc_auc_score(test_labels, test_predictions)
            print(f'LOF: Test AUC-ROC: {test_auc_roc:.4f}')

            nonzero_cost_indices = (test_unsafe_costs > 0).cpu().detach().numpy()
            if nonzero_cost_indices.sum() > 0:
                # cost_making_unsafe_uncertainties = 
                cost_making_unsafe_s_a = test_unsafe_s_a.cpu().detach().numpy()[nonzero_cost_indices]
                n_correct_cost_making_unsafe = (lof.predict(cost_making_unsafe_s_a) < 0).sum()
                n_correct_cost_making_unsafe_acc = n_correct_cost_making_unsafe / nonzero_cost_indices.sum()
            else:
                n_correct_cost_making_unsafe_acc = 0.
            
            test_positive_acc = (lof.predict(test_safe_s_a.cpu().detach().numpy()) > 0).mean()
            test_negative_acc = (lof.predict(test_unsafe_s_a.cpu().detach().numpy()) < 0).mean()
            print(f'LOF: Test positive accuracy: {test_positive_acc:.4f}, Test negative accuracy: {test_negative_acc:.4f}')
            print(f'LOF: Test cost-makers accuracy: {n_correct_cost_making_unsafe_acc:.4f}')

        else:
            if 'en-discriminator' in self.method:
                for num in range(0, int(self.pretrain_steps + 1)):
                    self.pu_disc_optimizer.zero_grad()
                    train_safe_batch = self.train_safe_replay_buffer.random_batch(batch_size, standardize=self.obs_standardize)
                    train_mixed_batch = self.train_mixed_replay_buffer.random_batch(batch_size, standardize=self.obs_standardize)

                    train_safe_obs = torch.tensor(train_safe_batch['observations'], dtype=torch.float32, device=self.device)
                    train_safe_actions = torch.tensor(train_safe_batch['actions'], dtype=torch.float32, device=self.device)
                    train_mixed_obs = torch.tensor(train_mixed_batch['observations'], dtype=torch.float32, device=self.device)
                    train_mixed_actions = torch.tensor(train_mixed_batch['actions'], dtype=torch.float32, device=self.device)

                    train_safe_s_a = torch.cat([train_safe_obs, train_safe_actions], dim=-1)
                    train_mixed_s_a = torch.cat([train_mixed_obs, train_mixed_actions], dim=-1)

                    train_safe_disc = self.pu_disc_network(train_safe_s_a)
                    train_mixed_disc = self.pu_disc_network(train_mixed_s_a)

                    disc_loss = -torch.log(train_safe_disc + 1e-10).mean() - torch.log(1 - train_mixed_disc + 1e-10).mean()

                    if self.pu_gp_weight > 0:
                        self.current_pu_disc_gradient_penalty = self.calculate_gradient_penalty(self.pu_disc_network, train_safe_s_a, train_mixed_s_a, type='disc')
                        disc_loss += self.pu_gp_weight * self.current_pu_disc_gradient_penalty
                    else:
                        self.current_pu_disc_gradient_penalty = 0.

                    disc_loss.backward()
                    self.pu_disc_optimizer.step()

                    if num % 1000 == 0:
                        self.pu_disc_optimizer.zero_grad()
                        valid_safe_disc = self.pu_disc_network(valid_safe_s_a_total)
                        valid_mixed_disc = self.pu_disc_network(valid_mixed_s_a_total)
                        valid_disc_loss = -torch.log(valid_safe_disc + 1e-10).mean() - torch.log(1 - valid_mixed_disc + 1e-10).mean()

                        if valid_disc_loss < min_valid_loss:
                            min_valid_loss = valid_disc_loss
                            last_update_num = num
                            self.best_pu_disc_network = copy.deepcopy(self.pu_disc_network)
                            print(f'Update best pu discriminator network at {num} iterations, valid loss: {valid_disc_loss.item():.2f}')

                        if num - last_update_num > self.early_stopping_patience:
                            print(f'Early stopping at {num} iterations...')
                            break
                            
                best_C = torch.mean(self.best_pu_disc_network(safe_s_a_total)).cpu().detach().numpy()
                self.class_prior = (total_safe_n / total_mixed_n) / (best_C + 1e-10) # n/m * (1/C)
                print(f'Best C: {best_C}, Class prior: {self.class_prior}') 
                    
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

                elif self.method == 'pn-discriminator':
                    train_unsafe_batch = self.train_unsafe_replay_buffer.random_batch(batch_size, standardize=self.obs_standardize)
                    train_unsafe_obs = torch.tensor(train_unsafe_batch['observations'], dtype=torch.float32, device=self.device)
                    train_unsafe_actions = torch.tensor(train_unsafe_batch['actions'], dtype=torch.float32, device=self.device)
                    train_unsafe_s_a = torch.cat([train_unsafe_obs, train_unsafe_actions], dim=-1)

                    safe_disc = self.disc_network(train_safe_s_a)
                    safe_disc_loss = -torch.log(safe_disc + 1e-10).mean()

                    unsafe_disc = self.disc_network(train_unsafe_s_a)
                    unsafe_disc_loss = -torch.log(1 - unsafe_disc + 1e-10).mean()

                    disc_loss = safe_disc_loss + unsafe_disc_loss

                    if self.gp_weight > 0:
                        self.current_disc_gradient_penalty = self.calculate_gradient_penalty(self.disc_network, train_safe_s_a, train_unsafe_s_a, type='disc')
                        disc_loss += self.gp_weight * self.current_disc_gradient_penalty
                    else:
                        self.current_disc_gradient_penalty = 0.

                    disc_loss.backward()
                    self.disc_optimizer.step()

                elif self.method == 'pu-discriminator':
                    self.disc_optimizer.zero_grad()

                    safe_disc = self.disc_network(train_safe_s_a)
                    safe_disc_loss = -torch.log(safe_disc + 1e-10).mean() * self.class_prior

                    mixed_disc = self.disc_network(train_mixed_s_a)
                    mixed_disc_loss = -torch.log(1 - mixed_disc + 1e-10).mean()
                    mixed_disc_loss += (- self.class_prior) * (-torch.log(1 - safe_disc + 1e-10).mean())

                    disc_loss = safe_disc_loss + mixed_disc_loss

                    if self.gp_weight > 0:
                        self.current_disc_gradient_penalty = self.calculate_gradient_penalty(self.disc_network, train_safe_s_a, train_mixed_s_a, type='disc')
                        disc_loss += self.gp_weight * self.current_disc_gradient_penalty
                    else:
                        self.current_disc_gradient_penalty = 0.

                    disc_loss.backward()
                    self.disc_optimizer.step()

                elif 'en-discriminator' in self.method: 
                    self.disc_optimizer.zero_grad()

                    safe_disc = self.disc_network(train_safe_s_a)
                    safe_disc_loss = -torch.log(safe_disc + 1e-10).mean() * self.class_prior

                    mixed_disc = self.disc_network(train_mixed_s_a)
                    mixed_disc_loss = -torch.log(1 - mixed_disc + 1e-10).mean()
                    mixed_disc_loss += (- self.class_prior) * (-torch.log(1 - safe_disc + 1e-10).mean())

                    disc_loss = safe_disc_loss + mixed_disc_loss

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
                    test_safe_batch = self.test_safe_replay_buffer.random_batch(batch_size, standardize=self.obs_standardize)
                    test_safe_obs = torch.tensor(test_safe_batch['observations'], dtype=torch.float32, device=self.device)
                    test_safe_actions = torch.tensor(test_safe_batch['actions'], dtype=torch.float32, device=self.device)
                    test_safe_s_a = torch.cat([test_safe_obs, test_safe_actions], dim=-1)

                    test_unsafe_batch = self.test_unsafe_replay_buffer.random_batch(batch_size, standardize=self.obs_standardize)
                    test_unsafe_obs = torch.tensor(test_unsafe_batch['observations'], dtype=torch.float32, device=self.device)
                    test_unsafe_actions = torch.tensor(test_unsafe_batch['actions'], dtype=torch.float32, device=self.device)
                    test_unsafe_s_a = torch.cat([test_unsafe_obs, test_unsafe_actions], dim=-1)
                    test_unsafe_costs = torch.tensor(test_unsafe_batch['costs'], dtype=torch.float32, device=self.device).reshape(-1)

                    # if 'ocsvm' in self.method:
                    #     test_safe_disc = self.ocsvm.decision_function(test_safe_s_a.cpu().detach().numpy())
                    #     test_unsafe_disc = self.ocsvm.decision_function(test_unsafe_s_a.cpu().detach().numpy())
                    #     # test_predictions = torch.cat([test_safe_disc, test_unsafe_disc], dim=0).cpu().detach().numpy().flatten()
                    #     test_predictions = np.concatenate([test_safe_disc, test_unsafe_disc])
                    #     test_labels = np.concatenate([np.ones(test_safe_disc.shape[0]), np.zeros(test_unsafe_disc.shape[0])])
                    #     test_auc_roc = roc_auc_score(test_labels, test_predictions)
                    #     print(f'Iteration={num}: Test AUC-ROC: {test_auc_roc:.4f}')

                    if 'discriminator' in self.method:
                        # if self.method == 'en-discriminator-prior':
                        #     train_safe_disc_total = self.disc_network(safe_s_a_total)
                        # else:
                        safe_disc_total = self.disc_network(safe_s_a_total)
                        
                        test_safe_disc = self.disc_network(test_safe_s_a)
                        test_unsafe_disc = self.disc_network(test_unsafe_s_a)
                        
                        # Calculate AUC-ROC for safe/unsafe discrimination
                        # Combine predictions and create labels (1 for safe, 0 for unsafe)
                        test_predictions = torch.cat([test_safe_disc, test_unsafe_disc], dim=0).cpu().detach().numpy().flatten()
                        test_labels = np.concatenate([np.ones(test_safe_disc.shape[0]), np.zeros(test_unsafe_disc.shape[0])])
                        test_auc_roc = roc_auc_score(test_labels, test_predictions)

                        valid_safe_batch = self.valid_safe_replay_buffer.random_batch(batch_size, standardize=self.obs_standardize)
                        valid_safe_obs = torch.tensor(valid_safe_batch['observations'], dtype=torch.float32, device=self.device)
                        valid_safe_actions = torch.tensor(valid_safe_batch['actions'], dtype=torch.float32, device=self.device)
                        valid_safe_s_a = torch.cat([valid_safe_obs, valid_safe_actions], dim=-1)

                        valid_mixed_batch = self.valid_mixed_replay_buffer.random_batch(batch_size, standardize=self.obs_standardize)
                        valid_mixed_obs = torch.tensor(valid_mixed_batch['observations'], dtype=torch.float32, device=self.device)
                        valid_mixed_actions = torch.tensor(valid_mixed_batch['actions'], dtype=torch.float32, device=self.device)
                        valid_mixed_s_a = torch.cat([valid_mixed_obs, valid_mixed_actions], dim=-1)

                        valid_safe_disc = self.disc_network(valid_safe_s_a)
                        valid_mixed_disc = self.disc_network(valid_mixed_s_a)
                        valid_pu_loss = -torch.log(valid_safe_disc + 1e-10).mean() - torch.log(1 - valid_mixed_disc + 1e-10).mean()
                        
                        valid_safe_disc_loss = -torch.log(valid_safe_disc + 1e-10).mean() * self.class_prior
                        valid_mixed_disc = self.disc_network(valid_mixed_s_a)
                        valid_mixed_disc_loss = -torch.log(1 - valid_mixed_disc + 1e-10).mean()
                        valid_mixed_disc_loss += (- self.class_prior) * (-torch.log(1 - valid_safe_disc + 1e-10).mean())
                        valid_pn_loss = valid_safe_disc_loss + valid_mixed_disc_loss

                        test_pn_loss = -torch.log(test_safe_disc + 1e-10).mean() - torch.log(1 - test_unsafe_disc + 1e-10).mean()

                        # Dictionary to store results for multiple thresholds
                        threshold_results = {}
                        quantile_list = [1.0, 0.9995, 0.999, 0.995, 0.99, 0.95, 0.90, 0.7]
                        
                        # Arrays for plotting
                        safe_acc_array = {}
                        unsafe_acc_array = {}
                        cost_making_unsafe_acc_array = {}
                        acc_array = {}
                        threshold_array = {}

                        best_acc = 0.
                        best_uncertainty_threshold = 0.
                        best_safe_acc = 0.
                        best_unsafe_acc = 0.
                        best_cost_making_unsafe_acc = 0.
                        best_quantile = 0.

                        for uncertainty_safe_quantile in quantile_list: 
                            self.uncertainty_threshold = torch.quantile(safe_disc_total, 1 - uncertainty_safe_quantile)

                            test_safe_n_correct = (test_safe_disc >= self.uncertainty_threshold).sum()
                            test_n_correct_safe_acc = test_safe_n_correct / batch_size

                            test_unsafe_n_correct = (test_unsafe_disc < self.uncertainty_threshold).sum()
                            test_n_correct_unsafe_acc = test_unsafe_n_correct / batch_size

                            if test_unsafe_costs.sum() > 0:
                                cost_making_unsafe_indices = (test_unsafe_costs > 0).cpu().detach().numpy()
                                cost_making_unsafe_n_correct = (test_unsafe_disc[cost_making_unsafe_indices] < self.uncertainty_threshold).sum()
                                n_correct_cost_making_unsafe_acc = cost_making_unsafe_n_correct / cost_making_unsafe_indices.sum()
                            else:
                                n_correct_cost_making_unsafe_acc = 0.
                        
                            acc = (test_n_correct_safe_acc + test_n_correct_unsafe_acc) / 2

                            if acc > best_acc:
                                best_acc = acc
                                best_uncertainty_threshold = self.uncertainty_threshold
                                best_safe_acc = test_n_correct_safe_acc
                                best_unsafe_acc = test_n_correct_unsafe_acc
                                best_cost_making_unsafe_acc = n_correct_cost_making_unsafe_acc
                                best_quantile = uncertainty_safe_quantile

                            # Store in dictionary
                            safe_acc_array[uncertainty_safe_quantile] = test_n_correct_safe_acc.item()
                            unsafe_acc_array[uncertainty_safe_quantile] = test_n_correct_unsafe_acc.item()
                            cost_making_unsafe_acc_array[uncertainty_safe_quantile] = n_correct_cost_making_unsafe_acc
                            acc_array[uncertainty_safe_quantile] = acc.item()
                            threshold_array[uncertainty_safe_quantile] = self.uncertainty_threshold.item()

                            if self.use_wandb:
                                self.wandb.log({
                                    f'quantile={uncertainty_safe_quantile}/safe_acc': safe_acc_array[uncertainty_safe_quantile],
                                    f'quantile={uncertainty_safe_quantile}/unsafe_acc': unsafe_acc_array[uncertainty_safe_quantile],
                                    f'quantile={uncertainty_safe_quantile}/cost_making_unsafe_acc': cost_making_unsafe_acc_array[uncertainty_safe_quantile],
                                    f'quantile={uncertainty_safe_quantile}/total_acc': acc_array[uncertainty_safe_quantile],
                                    f'quantile={uncertainty_safe_quantile}/threshold': threshold_array[uncertainty_safe_quantile],
                                }, step=num)
                            else:
                                print(f'Iteration={num}: Test safe accuracy: {test_n_correct_safe_acc}, Test unsafe accuracy: {test_n_correct_unsafe_acc}, Test total accuracy: {acc}, Cost making unsafe accuracy: {n_correct_cost_making_unsafe_acc}, Threshold: {self.uncertainty_threshold}')

                        if self.use_wandb:
                            self.wandb.log({
                                'best/acc': best_acc,
                                'best/uncertainty_threshold': best_uncertainty_threshold,
                                'best/safe_acc': best_safe_acc,
                                'best/unsafe_acc': best_unsafe_acc,
                                'best/cost_making_unsafe_acc': best_cost_making_unsafe_acc,
                                'best/quantile': best_quantile,
                                'best/class_prior': self.class_prior,
                                'eval/valid_PU_loss': valid_pu_loss.item(),
                                'eval/valid_PN_estimated_loss': valid_pn_loss.item(),
                                'eval/test_PN_true_loss': test_pn_loss.item(),
                                'eval/test_auc_roc': test_auc_roc,
                            }, step=num)
                            if 'en-discriminator' in self.method:
                                self.wandb.log({
                                    'best/C': best_C,
                                }, step=num)
                        else:
                            print(f'Iteration={num}: Test best accuracy: {best_acc}, Threshold: {best_uncertainty_threshold}, AUC-ROC: {test_auc_roc:.4f}')
                        
                    else:
                        uncertainty_estimates = torch.stack(self.uncertainty_estimates(safe_s_a_total)) # (n_prior_nets, batch_size, prior_output_size)
                        uncertainty_estimates_mean = uncertainty_estimates.mean(dim=0) # (batch_size, prior_output_size)

                        # Test safe
                        test_safe_batch = self.test_safe_replay_buffer.random_batch(batch_size, standardize=self.obs_standardize)
                        test_safe_obs = torch.tensor(test_safe_batch['observations'], dtype=torch.float32, device=self.device)
                        test_safe_actions = torch.tensor(test_safe_batch['actions'], dtype=torch.float32, device=self.device)
                        test_safe_s_a = torch.cat([test_safe_obs, test_safe_actions], dim=-1)
                        safe_error = torch.stack(self.uncertainty_estimates(test_safe_s_a))
                        safe_error_mean = safe_error.mean(dim=0)
                        safe_uncertainties = safe_error_mean #+ self.var_coef * safe_error_var - self.aleatoric_var
                        safe_uncertainties = safe_uncertainties.cpu().detach().numpy()

                        for uncertainty_safe_quantile in [0.9999, 0.9995, 0.999, 0.995, 0.99, 0.95, 0.90, 0.75, 0.5]:
                            self.uncertainty_threshold = np.quantile(uncertainty_estimates_mean.cpu().detach().numpy(), uncertainty_safe_quantile) # Select 99.9% Quantile as threshold

                        
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
            'project': 'check-indicator-acc-0903',
            'entity': 'tzs930',
        },
        'replay_buffer': {
            'max_replay_buffer_size': 10000000,
            'standardize_obs': True,
            'standardize_act': False,
        },
        'dataset': {
            'dataset_path': 'dataset/SafetyPointCircle1-v0',
            'train_safe': {
                'types': ['safe-expert-v0', 'safe-medium-v0'],
                'start_indices': [0, 0],
                'num_trajs': [10, 10],
            },
            'train_mixed': {
                'types': ['safe-expert-v0', 'safe-medium-v0', 'unsafe-expert-v0', 'unsafe-medium-v0'],
                'start_indices': [100, 100, 100, 100],
                'num_trajs': [20, 20, 10, 10],
            },
            'train_unsafe': {
                'types': ['unsafe-expert-v0', 'unsafe-medium-v0'],
                'start_indices': [200, 200],
                'num_trajs': [10, 10],
            },
            'valid_safe': {
                'types': ['safe-expert-v0', 'safe-medium-v0'],
                'start_indices': [200, 200],
                'num_trajs': [10, 10],
            },
            'valid_mixed': {
                'types': ['safe-expert-v0', 'safe-medium-v0', 'unsafe-expert-v0', 'unsafe-medium-v0'],
                'start_indices': [200, 200, 300, 300],
                'num_trajs': [20, 20, 10, 10],
            },
            'valid_unsafe': {
                'types': ['unsafe-expert-v0', 'unsafe-medium-v0'],
                'start_indices': [250, 250],
                'num_trajs': [10, 10],
            },
            'test_safe': {
                'types': ['safe-expert-v0', 'safe-medium-v0'],
                'start_indices': [500, 500],
                'num_trajs': [500, 500],
            },
            'test_unsafe': {
                'types': ['unsafe-expert-v0', 'unsafe-medium-v0'],
                'start_indices': [500, 500],
                'num_trajs': [500, 500],
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
            'pu_gp_weight': 10.,
            'lr_prior': 3e-5,
            'uncertainty_safe_quantile': 0.999,
        }
    }

    parser = ArgumentParser()
    parser.add_argument("--pid", help="process_id", default=0, type=int)
    args = parser.parse_args()
    pid = args.pid

    # if pid < 9:
    #     exit()

    configs['pid'] = pid

    hp_grids = {
        "method": ['pn-discriminator'], #'pu-discriminator/GT', 'en-discriminator-prior', 'discriminator', 'pu-discriminator/0.1', 'pu-discriminator/0.3', 'pu-discriminator/0.5', 'pu-discriminator/0.7', 'pu-discriminator/0.9'], #, 'pu-discriminator/GT', 'en-discriminator-prior', 'discriminator', 'pu-discriminator/0.1', 'pu-discriminator/0.3', 'pu-discriminator/0.5', 'pu-discriminator/0.7', 'pu-discriminator/0.9'
        "env_id": ['SafetyPointCircle1-v0', 'SafetyWalker2dVelocity-v1'], #'SafetyWalker2dVelocity-v0'],
        # "dataset/train_safe/num_trajs":     [[1,1], [5,5], [10,10], [50,50], [100,100]], 
        # "train/uncertainty_safe_quantile":  ['0.9', '0.99', '0.999'],
        "train/gp_weight":                  [0.0], #1.0, 10.0],
        "train/pu_gp_weight":               [0.0], #1.0, 10.0],
        # "train/n_prior_nets":               [2,  4,  8,  16, 32],
        # "train/prior_output_size":          [10, 20, 30, 40, 50],
        "train/seed":                       [0, 1, 2],
        "train/class_prior":                [-1],
    }
    # 3*5*5*3 = 225

    hp_values = list(product(*hp_grids.values()))[pid]

    for key, value in zip(hp_grids.keys(), hp_values):
        if '/' in key:
            if 'dataset' in key:
                key1 = key.split('/')[0]
                key2 = key.split('/')[1]
                key3 = key.split('/')[2]
                configs[key1][key2][key3] = value
                print(f'** {key1}/{key2}/{key3}: {value}')
            else:
                key1 = key.split('/')[0]
                key2 = key.split('/')[1]
                configs[key1][key2] = value
                print(f'** {key1}/{key2}: {value}')
        else:
            configs[key] = value

    if 'pu-discriminator' in configs['method']:
        method = configs['method'].split('/')[0]
        class_prior = configs['method'].split('/')[1]
        configs['method'] = method
        if 'GT' in class_prior:
            dataset_nums = configs['dataset']['train_mixed']['num_trajs']
            n_safe = dataset_nums[0] + dataset_nums[1]
            n_unsafe = dataset_nums[2] + dataset_nums[3]
            configs['train']['class_prior'] = n_safe / (n_safe + n_unsafe)
            print(f'** Class prior: {configs["train"]["class_prior"]}')
        else:
            configs['train']['class_prior'] = class_prior

    # if 'PUGP' in configs['method']:
    #     configs['method'] = configs['method'].split('/')[0]
        # configs['train']['pu_gp_weight'] = 10.


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
    train_unsafe_data_dict, n_train_unsafe = preprocess_dataset(configs['dataset']['dataset_path'],
                                                            configs['dataset']['train_unsafe']['types'], 
                                                            configs['dataset']['train_unsafe']['start_indices'], 
                                                            num_rollouts=configs['dataset']['train_unsafe']['num_trajs'])

    valid_safe_data_dict, n_valid_safe = preprocess_dataset(configs['dataset']['dataset_path'],
                                                            configs['dataset']['valid_safe']['types'], 
                                                            configs['dataset']['valid_safe']['start_indices'], 
                                                            num_rollouts=configs['dataset']['valid_safe']['num_trajs'])

    valid_mixed_data_dict, n_valid_mixed = preprocess_dataset(configs['dataset']['dataset_path'],
                                                            configs['dataset']['valid_mixed']['types'], 
                                                            configs['dataset']['valid_mixed']['start_indices'], 
                                                            num_rollouts=configs['dataset']['valid_mixed']['num_trajs'])

    valid_unsafe_data_dict, n_valid_unsafe = preprocess_dataset(configs['dataset']['dataset_path'],
                                                            configs['dataset']['valid_unsafe']['types'], 
                                                            configs['dataset']['valid_unsafe']['start_indices'], 
                                                            num_rollouts=configs['dataset']['valid_unsafe']['num_trajs'])

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

    train_unsafe_replay_buffer = MDPReplayBuffer(
        configs['replay_buffer']['max_replay_buffer_size'],
        env,
        obs_dim=obs_dim
    )
    train_unsafe_replay_buffer.add_path(train_unsafe_data_dict)

    valid_safe_replay_buffer = MDPReplayBuffer(
        configs['replay_buffer']['max_replay_buffer_size'],
        env,
        obs_dim=obs_dim
    )
    valid_safe_replay_buffer.add_path(valid_safe_data_dict)

    valid_mixed_replay_buffer = MDPReplayBuffer(
        configs['replay_buffer']['max_replay_buffer_size'],
        env,
        obs_dim=obs_dim
    )
    valid_mixed_replay_buffer.add_path(valid_mixed_data_dict)

    valid_unsafe_replay_buffer = MDPReplayBuffer(
        configs['replay_buffer']['max_replay_buffer_size'],
        env,
        obs_dim=obs_dim
    )
    valid_unsafe_replay_buffer.add_path(valid_unsafe_data_dict)

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

    if configs['replay_buffer']['standardize_obs'] or configs['replay_buffer']['standardize_act']:
        obs_mean, obs_std, act_mean, act_std = train_mixed_replay_buffer.calculate_statistics(
            standardize_obs=configs['replay_buffer']['standardize_obs'],
            standardize_act=configs['replay_buffer']['standardize_act']
        )
        if train_safe_data_dict is not None:
            train_safe_replay_buffer.set_statistics(obs_mean, obs_std, act_mean, act_std)
        if train_unsafe_data_dict is not None:
            train_unsafe_replay_buffer.set_statistics(obs_mean, obs_std, act_mean, act_std)
        if valid_safe_data_dict is not None:
            valid_safe_replay_buffer.set_statistics(obs_mean, obs_std, act_mean, act_std)
        if valid_mixed_data_dict is not None:
            valid_mixed_replay_buffer.set_statistics(obs_mean, obs_std, act_mean, act_std)
        if valid_unsafe_data_dict is not None:
            valid_unsafe_replay_buffer.set_statistics(obs_mean, obs_std, act_mean, act_std)
        if test_safe_data_dict is not None:
            test_safe_replay_buffer.set_statistics(obs_mean, obs_std, act_mean, act_std)
        if test_unsafe_data_dict is not None:
            test_unsafe_replay_buffer.set_statistics(obs_mean, obs_std, act_mean, act_std)
    else:
        obs_mean, obs_std, act_mean, act_std = None, None, None, None

    check_indicator_acc = CheckIndicatorAcc(env, configs,
                                            train_safe_replay_buffer=train_safe_replay_buffer,
                                            train_mixed_replay_buffer=train_mixed_replay_buffer,
                                            train_unsafe_replay_buffer=train_unsafe_replay_buffer,
                                            valid_safe_replay_buffer=valid_safe_replay_buffer,
                                            valid_unsafe_replay_buffer=valid_unsafe_replay_buffer,
                                            valid_mixed_replay_buffer=valid_mixed_replay_buffer,
                                            test_safe_replay_buffer=test_safe_replay_buffer,
                                            test_unsafe_replay_buffer=test_unsafe_replay_buffer)
    check_indicator_acc.train(total_iteration=configs['train']['total_iteration'],
                              eval_freq=configs['train']['eval_freq'],
                              batch_size=configs['train']['batch_size'])
