import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

import yaml

# performance_dict = {
#     'SafetyPointCircle1-v0': {
#         'safe_expert_score': 44.30,
#         'unsafe_expert_score': 54.55,
#         'random_score': 0., 
#         'cost_threshold': 25.,
#     }
# }

def copy_nn_module(source, target):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

class IcilDICEv3(nn.Module):
    def __init__(self, policy, env, configs, best_policy=None,
                 init_obs_buffer=None, expert_replay_buffer=None, safe_replay_buffer=None, mixed_replay_buffer=None,
                 seed=0, n_train=1, add_absorbing_state=False):
        
        seed = configs['train']['seed']
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        super(IcilDICEv3, self).__init__()

        self.env = env
        self.policy = policy
        self.best_policy = best_policy

        self.init_obs_buffer = init_obs_buffer
        self.expert_replay_buffer = expert_replay_buffer
        self.safe_replay_buffer = safe_replay_buffer
        self.mixed_replay_buffer = mixed_replay_buffer
        
        self.add_absorbing_state = add_absorbing_state
        
        self.device = configs['device']
        
        self.n_train = n_train
    
        self.obs_dim = env.observation_space.low.size
        self.action_dim = env.action_space.low.size
        
        self.gamma = configs['train']['gamma']
        
        self.nu_hidden_size = configs['train']['nu_hidden_size']
        self.nu_network = nn.Sequential(
            nn.Linear(self.obs_dim, self.nu_hidden_size[0], device=self.device),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.ReLU(inplace=True),
            nn.Linear(self.nu_hidden_size[0], self.nu_hidden_size[1], device=self.device),
            nn.ReLU(inplace=True),
            nn.Linear(self.nu_hidden_size[1], 1, device=self.device)
        )

        self.disc_hidden_size = configs['train']['disc_hidden_size']
        self.disc_network = nn.Sequential(
            nn.Linear(self.obs_dim + self.action_dim, self.disc_hidden_size[0], device=self.device),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.ReLU(inplace=True),
            nn.Linear(self.disc_hidden_size[0], self.disc_hidden_size[1], device=self.device),
            nn.ReLU(inplace=True),
            nn.Linear(self.disc_hidden_size[1], 1, device=self.device),
            nn.Sigmoid()
        )

        self.use_soft_indicator = bool(configs['train']['use_soft_indicator'])
        self.beta = float(configs['train']['beta'])
        self.lambda_ = nn.Parameter(torch.tensor(configs['train']['lambda_starts'], device=self.device))
        # self.lambda_ = torch.exp(self.lambda_param)

        self.uc_disc_hidden_size = configs['train']['uc_disc_hidden_size']
        self.uc_disc_network = nn.Sequential(
            nn.Linear(self.obs_dim + self.action_dim, self.uc_disc_hidden_size[0], device=self.device),
            nn.ReLU(inplace=True),
            nn.Linear(self.uc_disc_hidden_size[0], self.uc_disc_hidden_size[1], device=self.device),
            nn.ReLU(inplace=True),
            nn.Linear(self.uc_disc_hidden_size[1], 1, device=self.device),
            nn.Sigmoid()
        )
        
        self.uncertainty_safe_quantile = float(configs['train']['uncertainty_safe_quantile']) # e.g. 0.998 Quantile
        self.uncertainty_threshold_update_freq = int(configs['train']['uncertainty_threshold_update_freq'])
        self.pretrain_steps = int(configs['train']['pretrain_steps'])
        self.current_uc_gradient_penalty = 0.
        self.current_disc_gradient_penalty = 0.

        self.r_fn = \
            lambda x: - torch.log( 1 / (self.disc_network(x) + 1e-10) - 1 + 1e-10)
        
        self.nu_optimizer = optim.Adam(self.nu_network.parameters(), lr=float(configs['train']['lr_nu']))
        self.disc_optimizer = optim.Adam(self.disc_network.parameters(), lr=float(configs['train']['lr_disc']))
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=float(configs['train']['lr']))
        self.uc_disc_optimizer = optim.Adam(self.uc_disc_network.parameters(), lr=float(configs['train']['lr_disc']))
        self.lambda_optimizer = optim.Adam([self.lambda_], lr=float(configs['train']['lr_lambda']))
        
        self.alpha = float(configs['train']['alpha'])
        self.class_prior = float(configs['train']['class_prior'])
        self.disc_steps = int(configs['train']['disc_steps'])

        self.nu_gp_weight = float(configs['train']['nu_gp_weight'])
        self.disc_gp_weight = float(configs['train']['disc_gp_weight'])  # Gradient penalty coefficient
        self.uc_disc_gp_weight = float(configs['train']['uc_disc_gp_weight'])
        
        self.indicator_weight = float(configs['train']['indicator_weight'])
        self.prior_learning_during_train = bool(configs['train']['prior_learning_during_train'])
        
        self.num_eval_iteration = int(configs['train']['num_evals'])
        self.envname = configs['env_id']
        
        self.use_wandb = bool(configs['use_wandb'])
        if self.use_wandb:
            import wandb
            self.wandb = wandb
            self.wandb.init(project=configs['wandb']['project'], entity=configs['wandb']['entity'], config=configs)
        else:
            self.wandb = None

        self.save_policy_path = configs['train']['save_policy_path']        
        
        # For standardization
        self.obs_standardize = configs['replay_buffer']['standardize_obs']
        self.act_standardize = configs['replay_buffer']['standardize_act']

        if self.obs_standardize:
            self.obs_mean = self.safe_replay_buffer.obs_mean
            self.obs_std = self.safe_replay_buffer.obs_std 
            self.obs_mean_tt = torch.tensor(self.obs_mean, device=self.device)
            self.obs_std_tt = torch.tensor(self.obs_std, device=self.device)
        else:
            self.obs_mean = None
            self.obs_std = None
            self.obs_mean_tt = None
            self.obs_std_tt = None
            
        if self.act_standardize:
            self.act_mean = self.safe_replay_buffer.act_mean
            self.act_std = self.safe_replay_buffer.act_std
            self.act_mean_tt = torch.tensor(self.safe_replay_buffer.act_mean, device=self.device)
            self.act_std_tt = torch.tensor(self.safe_replay_buffer.act_std, device=self.device)
        else:
            self.act_mean = None
            self.act_std = None
            self.act_mean_tt = None
            self.act_std_tt = None

        try:
            safe_expert_file_stats = f'dataset/{self.envname}/stats/safe-expert-v0-stats.yaml'
            with open(safe_expert_file_stats, 'r') as f:
                safe_expert_stats = yaml.safe_load(f)
            self.max_score = safe_expert_stats['average_return']
            # self.cost_threshold = safe_expert_stats['cost_threshold']
        except:
            self.max_score = None

        self.cost_threshold = 25.

        try:
            random_file_stats = f'dataset/{self.envname}/stats/random-v0-stats.yaml'
            with open(random_file_stats, 'r') as f:
                random_stats = yaml.safe_load(f)
            self.min_score = random_stats['average_return']
        except:
            self.min_score = None

        print(f'** max_score: {self.max_score}, cost_threshold: {self.cost_threshold}, min_score: {self.min_score}')

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
        
        # batch_valid = self.replay_buffer_valid.random_batch(self.n_valid, standardize=self.standardize)
        
        # obs_valid = batch_valid['observations']
        # actions_valid = batch_valid['actions'][:, -self.action_dim:]
        # next_obs_valid = batch_valid['next_observations']
        # terminals_valid = batch_valid['terminals']
                
        # obs_valid = torch.tensor(obs_valid, dtype=torch.float32, device=self.device)
        # actions_valid = torch.tensor(actions_valid, dtype=torch.float32, device=self.device)
        # next_obs_valid = torch.tensor(next_obs_valid, dtype=torch.float32, device=self.device)
        # terminals_valid = torch.tensor(terminals_valid, dtype=torch.float32, device=self.device)

        safe_n = self.safe_replay_buffer._size
        safe_obs_total = torch.tensor(self.safe_replay_buffer._observations[:safe_n], dtype=torch.float32, device=self.device)
        safe_actions_total = torch.tensor(self.safe_replay_buffer._actions[:safe_n], dtype=torch.float32, device=self.device)
        safe_s_a_total = torch.cat([safe_obs_total, safe_actions_total], dim=-1)

        # mixed_n = self.mixed_replay_buffer._size
        # mixed_obs_total = torch.tensor(self.mixed_replay_buffer._observations[:mixed_n], dtype=torch.float32, device=self.device)
        # mixed_actions_total = torch.tensor(self.mixed_replay_buffer._actions[:mixed_n], dtype=torch.float32, device=self.device)
        # mixed_s_a_total = torch.cat([mixed_obs_total, mixed_actions_total], dim=-1)

        # Pretrain phase: Prior fitting or Expert discriminator fitting
        for num in range(0, int(self.pretrain_steps + 1)):
            self.disc_optimizer.zero_grad()
            
            expert_batch = self.expert_replay_buffer.random_batch(batch_size, standardize=self.obs_standardize)
            expert_obs = torch.tensor(expert_batch['observations'], dtype=torch.float32, device=self.device)
            expert_actions = torch.tensor(expert_batch['actions'], dtype=torch.float32, device=self.device)
            expert_s_a = torch.cat([expert_obs, expert_actions], dim=-1)
            
            mixed_batch = self.mixed_replay_buffer.random_batch(batch_size, standardize=self.obs_standardize)
            mixed_obs = torch.tensor(mixed_batch['observations'], dtype=torch.float32, device=self.device)
            mixed_actions = torch.tensor(mixed_batch['actions'], dtype=torch.float32, device=self.device)
            mixed_s_a = torch.cat([mixed_obs, mixed_actions], dim=-1)

            expert_disc = self.disc_network(expert_s_a)
            expert_disc_loss = -torch.log(expert_disc + 1e-10).mean()

            mixed_disc = self.disc_network(mixed_s_a)
            mixed_disc_loss = -torch.log(1 - mixed_disc + 1e-10).mean()

            disc_loss = expert_disc_loss + mixed_disc_loss

            if self.disc_gp_weight > 0:
                self.current_disc_gradient_penalty = self.calculate_gradient_penalty(self.disc_network, expert_s_a, mixed_s_a, type='disc')
                disc_loss += self.disc_gp_weight * self.current_disc_gradient_penalty
            else:
                self.current_disc_gradient_penalty = 0.

            disc_loss.backward()
            self.disc_optimizer.step()

            self.uc_disc_optimizer.zero_grad()
            safe_batch = self.safe_replay_buffer.random_batch(batch_size, standardize=self.obs_standardize)
            safe_obs = torch.tensor(safe_batch['observations'], dtype=torch.float32, device=self.device)
            safe_actions = torch.tensor(safe_batch['actions'], dtype=torch.float32, device=self.device)
            safe_s_a = torch.cat([safe_obs, safe_actions], dim=-1)

            safe_disc = self.uc_disc_network(safe_s_a)
            mixed_disc = self.uc_disc_network(mixed_s_a)

            # safe_disc_loss = -torch.log(safe_disc + 1e-10).mean() * self.class_prior
            # mixed_disc_loss = -torch.log(1 - mixed_disc + 1e-10).mean()
            # mixed_disc_loss += (- self.class_prior) * (-torch.log(1 - safe_disc + 1e-10).mean())
            # uc_disc_loss = safe_disc_loss + mixed_disc_loss
            uc_disc_loss = -torch.log(safe_disc + 1e-10).mean() - torch.log(1 - mixed_disc + 1e-10).mean()

            if self.uc_disc_gp_weight > 0:
                self.current_uc_gradient_penalty = self.calculate_gradient_penalty(self.uc_disc_network, safe_s_a, mixed_s_a, type='disc')
                uc_disc_loss += self.uc_disc_gp_weight * self.current_uc_gradient_penalty
            else:
                self.current_uc_gradient_penalty = 0.

            uc_disc_loss.backward()
            self.uc_disc_optimizer.step()

            if num % 10000 == 0:
                print(f'** pretrain iteration: {num}, disc loss: {disc_loss.item():.4f}, uc disc loss: {uc_disc_loss.item():.4f}')

        safe_disc_total = self.uc_disc_network(safe_s_a_total).reshape(-1).cpu().detach().numpy()
        safe_disc_threshold = np.quantile(safe_disc_total, 1 - self.uncertainty_safe_quantile)
        self.uncertainty_threshold = safe_disc_threshold
             
        for num in range(0, int(total_iteration)+1):
            if self.prior_learning_during_train:
                for _ in range(self.disc_steps):
                    self.disc_optimizer.zero_grad()
                    
                    expert_batch = self.expert_replay_buffer.random_batch(batch_size, standardize=self.obs_standardize)
                    mixed_batch = self.mixed_replay_buffer.random_batch(batch_size, standardize=self.obs_standardize)
                    
                    expert_obs = torch.tensor(expert_batch['observations'], dtype=torch.float32, device=self.device)
                    expert_actions = torch.tensor(expert_batch['actions'], dtype=torch.float32, device=self.device)

                    mixed_obs = torch.tensor(mixed_batch['observations'], dtype=torch.float32, device=self.device)
                    mixed_actions = torch.tensor(mixed_batch['actions'], dtype=torch.float32, device=self.device)

                    expert_s_a = torch.cat([expert_obs, expert_actions], dim=-1)
                    mixed_s_a = torch.cat([mixed_obs, mixed_actions], dim=-1)

                    expert_disc = self.disc_network(expert_s_a)
                    mixed_disc = self.disc_network(mixed_s_a)

                    disc_loss = -torch.log(expert_disc + 1e-10).mean() - torch.log(1 - mixed_disc + 1e-10).mean()
                    
                    # Add gradient penalty term
                    if self.disc_gp_weight > 0:
                        randperm = torch.randperm(batch_size)
                        self.current_disc_gradient_penalty = self.calculate_gradient_penalty(self.disc_network, mixed_s_a, mixed_s_a[randperm], type='disc')
                        disc_loss += self.disc_gp_weight * self.current_disc_gradient_penalty
                    else:
                        self.current_disc_gradient_penalty = 0.

                    disc_loss.backward()
                    self.disc_optimizer.step()

                    self.uc_disc_optimizer.zero_grad()
                    safe_batch = self.safe_replay_buffer.random_batch(batch_size, standardize=self.obs_standardize)
                    safe_obs = torch.tensor(safe_batch['observations'], dtype=torch.float32, device=self.device)
                    safe_actions = torch.tensor(safe_batch['actions'], dtype=torch.float32, device=self.device)
                    safe_s_a = torch.cat([safe_obs, safe_actions], dim=-1)
                        
                    safe_disc = self.uc_disc_network(safe_s_a)
                    mixed_disc = self.uc_disc_network(mixed_s_a)

                    # uc_disc_loss = self.class_prior * (-torch.log(safe_disc + 1e-10).mean())
                    # uc_disc_loss += (-torch.log(1 - mixed_disc + 1e-10).mean())
                    # uc_disc_loss += (- self.class_prior) * (-torch.log(1 - safe_disc + 1e-10).mean())
                    uc_disc_loss = -torch.log(safe_disc + 1e-10).mean() - torch.log(1 - mixed_disc + 1e-10).mean()

                    if self.uc_disc_gp_weight > 0:
                        randperm = torch.randperm(batch_size)
                        self.current_uc_gradient_penalty = self.calculate_gradient_penalty(self.uc_disc_network, mixed_s_a, mixed_s_a[randperm], type='disc')
                        uc_disc_loss += self.uc_disc_gp_weight * self.current_uc_gradient_penalty
                    else:
                        self.current_uc_gradient_penalty = 0.

                    uc_disc_loss.backward()
                    self.uc_disc_optimizer.step()

            self.nu_optimizer.zero_grad()
            self.lambda_optimizer.zero_grad()
            self.policy_optimizer.zero_grad()

            init_obs = self.init_obs_buffer.random_batch(batch_size, standardize=self.obs_standardize)
            expert_batch = self.expert_replay_buffer.random_batch(batch_size, standardize=self.obs_standardize)
            mixed_batch = self.mixed_replay_buffer.random_batch(batch_size, standardize=self.obs_standardize)

            # nu (lagrangian) training
            init_obs = torch.tensor(init_obs, dtype=torch.float32, device=self.device)

            mixed_obs = torch.tensor(mixed_batch['observations'], dtype=torch.float32, device=self.device)
            mixed_actions = torch.tensor(mixed_batch['actions'], dtype=torch.float32, device=self.device)
            mixed_next_obs = torch.tensor(mixed_batch['next_observations'], dtype=torch.float32, device=self.device)
            mixed_s_a = torch.cat([mixed_obs, mixed_actions], dim=-1)
            mixed_r = self.r_fn(mixed_s_a).reshape(-1)

            if self.use_soft_indicator:
                # mixed_disc = self.uc_disc_network(mixed_s_a)
                mixed_safe_indicator = self.uc_disc_network(mixed_s_a).reshape(-1).detach()
            else:
                mixed_disc = self.uc_disc_network(mixed_s_a)
                mixed_safe_indicator = (mixed_disc > self.uncertainty_threshold).reshape(-1).detach().float()

            init_nu = self.nu_network(init_obs).reshape(-1)
            mixed_nu = self.nu_network(mixed_obs).reshape(-1)
            mixed_nu_prime = self.nu_network(mixed_next_obs).reshape(-1)

            mixed_adv = mixed_r + self.lambda_ * self.indicator_weight * mixed_safe_indicator - self.beta * (1. - mixed_safe_indicator) + self.gamma * mixed_nu_prime - mixed_nu

            nu_loss0 = (1 - self.gamma) * init_nu.mean()
            nu_loss1 = (1 + self.alpha) * torch.logsumexp(mixed_adv / (1 + self.alpha), -1) #.mean()
            # nu_loss1 = (1+ self.alpha) * torch.exp(mixed_adv / (1 + self.alpha) - 1).mean()

            nu_loss = nu_loss0 + nu_loss1 - self.lambda_

            if self.nu_gp_weight > 0:
                randperm = torch.randperm(batch_size)
                self.current_nu_gradient_penalty = self.calculate_gradient_penalty(self.nu_network, mixed_obs, mixed_obs[randperm], type='disc')
                nu_loss += self.nu_gp_weight * self.current_nu_gradient_penalty
            else:
                self.current_nu_gradient_penalty = 0.

            nu_loss.backward()
            self.nu_optimizer.step()
            self.lambda_optimizer.step()

            # Policy training (weighted BC)
            # weights corresponds to KL (with reweighting)
            # Use logsumexp trick for numerical stability
            log_weights = (mixed_adv) / (self.alpha + 1) - 1
            log_weights_normalized = log_weights - torch.logsumexp(log_weights, dim=0)
            policy_weight = torch.exp(log_weights_normalized).detach()
            policy_loss = - (policy_weight * self.policy.log_prob(mixed_obs, mixed_actions)).mean()

            policy_loss.backward()
            self.policy_optimizer.step()

            if (num) % self.uncertainty_threshold_update_freq == 0  and self.prior_learning_during_train:
                safe_disc_total = self.uc_disc_network(safe_s_a_total).reshape(-1).cpu().detach().numpy()
                safe_disc_threshold = np.quantile(safe_disc_total, 1 - self.uncertainty_safe_quantile)
                self.uncertainty_threshold = safe_disc_threshold
            
            if (num) % eval_freq == 0:
                self.policy.eval()
                eval_ret_mean, eval_ret_std, eval_cost_mean, eval_cost_std, eval_violation_rate, eval_length_mean, feasible_ret_mean, feasible_ret_std = \
                    self.evaluate(self.env, self.policy, num_evaluation=self.num_eval_iteration)
                
                # mixed_error = torch.stack(self.uncertainty_estimates(mixed_s_a_total))
                # mixed_error_mean = mixed_error.mean(dim=0)
                # mixed_error_var = mixed_error.var(dim=0)
                # mixed_uncertainties = mixed_error_mean + self.var_coef * mixed_error_var - self.aleatoric_var
                # mixed_uncertainties = mixed_uncertainties.cpu().detach().numpy()
                # mixed_uncertainties = np.maximum(0, mixed_uncertainties)
                mixed_num_oods = batch_size - torch.sum(mixed_safe_indicator).item()

                mean_weight = torch.mean(policy_weight).detach().cpu().numpy()
                sum_weights = torch.sum(policy_weight).detach().cpu().numpy()
                sum_weights_squared = torch.sum(policy_weight ** 2).detach().cpu().numpy()
                effective_sample_size = (sum_weights ** 2) / (sum_weights_squared + 1e-8)
                
                print(f'** iter{num}: policy_loss={policy_loss.item():.2f}, ret={eval_ret_mean:.2f}+-{eval_ret_std:.2f}, cost={eval_cost_mean:.2f}+-{eval_cost_std:.2f}, violation_rate={eval_violation_rate:.2f}, length={eval_length_mean:.2f}, lambda={self.lambda_.item():.2f}, num_oods={mixed_num_oods}')
                
                if self.use_wandb:
                    self.wandb.log({'train/policy_loss':       policy_loss.item(), 
                                    'train/nu_loss':           nu_loss.item(),
                                    'train/disc_loss':         disc_loss.item(),
                                    'train/disc_gradient_penalty':  self.current_disc_gradient_penalty,
                                    'train/lambda':            self.lambda_.item(),
                                    'train/mixed_nu_mean':     mixed_nu.mean().item(),
                                    'train/mixed_adv_mean':    mixed_adv.mean().item(),
                                    'train/sum_weights':       sum_weights,
                                    'train/effective_sample_size': effective_sample_size,
                                    'train/mean_policy_weight':       mean_weight,
                                    'train/mixed_r_max':       mixed_r.max().item(),
                                    'train/mixed_r_min':       mixed_r.min().item(),
                                    'eval/episode_return':     eval_ret_mean,
                                    'eval/episode_cost':       eval_cost_mean,
                                    'eval/violation_rate':     eval_violation_rate,
                                    'eval/episode_length':     eval_length_mean,
                                    'eval/feasible_return':     feasible_ret_mean,
                                    'eval/feasible_return_std': feasible_ret_std,
                                    'prior/uncertainty_threshold': self.uncertainty_threshold,
                                    'prior/num_oods_in_mixed':     mixed_num_oods,
                                    'prior/fit_loss':              uc_disc_loss.item(),
                                    'prior/uc_gradient_penalty':  self.current_uc_gradient_penalty,
                               }, step=num+1)

                if eval_ret_mean > max_score:
                    print(f'** max score record! ')
                    max_score = eval_ret_mean
                    copy_nn_module(self.policy, self.best_policy)
                    
        if self.save_policy_path:
            print(f'** save model to ', f'{self.save_policy_path}/bc_actor_best.pt')
            os.makedirs(self.save_policy_path, exist_ok=True)
            torch.save(self.best_policy.state_dict(), 
                    f'{self.save_policy_path}/bc_actor_best.pt')
            
            print(f'** save model to ', f'{self.save_policy_path}/bc_actor_last.pt')
            os.makedirs(self.save_policy_path, exist_ok=True)
            torch.save(self.policy.state_dict(), 
                    f'{self.save_policy_path}/bc_actor_last.pt')
                    
        if self.save_policy_path:
            print(f'** save model to ', f'{self.save_policy_path}/bc_actor_best.pt')
            os.makedirs(self.save_policy_path, exist_ok=True)
            torch.save(self.best_policy.state_dict(), 
                    f'{self.save_policy_path}/bc_actor_best.pt')
            
            print(f'** save model to ', f'{self.save_policy_path}/bc_actor_last.pt')
            os.makedirs(self.save_policy_path, exist_ok=True)
            torch.save(self.policy.state_dict(), 
                    f'{self.save_policy_path}/bc_actor_last.pt')
    
                   
    def evaluate(self, env, policy, num_evaluation=5, deterministic=True):
        rets = []
        costs = []
        lengths = []
        rets_until_violation = []
        # cost_threshold = performance_dict[env.spec.id]['cost_threshold']
        # max_score = performance_dict[env.spec.id]['safe_expert_score']
        # min_score = performance_dict[env.spec.id]['random_score']
        cost_threshold = self.cost_threshold
        max_score = self.max_score
        min_score = self.min_score

        # maxtimestep = 1000
        for num in range(0, num_evaluation):
            obs_, info_ = env.reset()
            
            done = False
            t = 0
            ret = 0.
            cum_cost = 0.
            violation = False
            ret_until_violation = 0
            
            while not done:  #or t < maxtimestep
                if self.add_absorbing_state:
                    obs = np.concatenate([obs_, [0.]]) # absorbing
                else:
                    obs = obs_
                    
                if self.obs_standardize:
                    obs = (obs - self.obs_mean[0]) / (self.obs_std[0] + 1e-8)

                obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
                if deterministic:
                    action = policy(obs).mean.cpu().detach().numpy()
                else:
                    action = policy(obs).sample().cpu().detach().numpy()
                
                next_obs, rew, cost, terminate, truncated, info = env.step(action)

                ret += rew
                cum_cost += cost
                
                if cum_cost > cost_threshold:
                    violation = True
                if not violation:
                    ret_until_violation += rew
                
                obs_ = next_obs 
                done = terminate or truncated
                    
                t += 1

            if max_score is not None and min_score is not None:
                normalized_ret = (ret - min_score) / (max_score - min_score) * 100.
                normalized_ret_until_violation = (ret_until_violation - min_score) / (max_score - min_score) * 100.
            else:
                normalized_ret = ret
                normalized_ret_until_violation = ret_until_violation

            rets.append(normalized_ret)
            costs.append(cum_cost)
            lengths.append(t)
            rets_until_violation.append(normalized_ret_until_violation)

        violation_rate = np.mean(np.array(costs) > cost_threshold)

        return np.mean(rets), np.std(rets)/np.sqrt(num_evaluation), np.mean(costs), np.std(costs)/np.sqrt(num_evaluation), violation_rate, np.mean(lengths), np.mean(rets_until_violation), np.std(rets_until_violation)/np.sqrt(num_evaluation)
    
