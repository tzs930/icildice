import numpy as np
import random
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

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

class BCJointFiltered(nn.Module):
    def __init__(self, policy, env, configs, best_policy=None,
                expert_replay_buffer=None, safe_replay_buffer=None, mixed_replay_buffer=None, n_train=1):
        
        seed = configs['train']['seed']
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        super(BCJointFiltered, self).__init__()

        self.env = env
        self.policy = policy
        self.best_policy = best_policy
        self.expert_replay_buffer = expert_replay_buffer
        self.safe_replay_buffer = safe_replay_buffer
        self.mixed_replay_buffer = mixed_replay_buffer

        self.device = configs['device']
        self.add_absorbing_state = configs['replay_buffer']['use_absorbing_state']
        
        self.n_train = n_train
    
        if self.add_absorbing_state:
            self.obs_dim = env.observation_space.low.size + 1
        else:
            self.obs_dim = env.observation_space.low.size
        self.action_dim = env.action_space.low.size
        
        self.policy_optimizer = optim.Adam(policy.parameters(), lr=float(configs['train']['lr']))
        
        self.num_eval_iteration = configs['train']['num_evals']
        self.envname = configs['env_id']
        
        self.use_wandb = configs['use_wandb']
        if self.use_wandb:
            import wandb
            self.wandb = wandb
            self.wandb.init(project=configs['wandb']['project'], 
                            entity=configs['wandb']['entity'], 
                            config=configs)
        else:
            self.wandb = None

        self.save_policy_path = configs['train']['save_policy_path']        

        self.disc_hidden_size = configs['train']['disc_hidden_size']
        self.disc_gp_weight = configs['train']['disc_gp_weight']
        self.expert_disc_network = nn.Sequential(
            nn.Linear(self.obs_dim + self.action_dim, self.disc_hidden_size[0], device=self.device),
            nn.ReLU(inplace=True),
            nn.Linear(self.disc_hidden_size[0], self.disc_hidden_size[1], device=self.device),
            nn.ReLU(inplace=True),
            nn.Linear(self.disc_hidden_size[1], 1, device=self.device),
            nn.Sigmoid()
        )
        self.expert_disc_optimizer = optim.Adam(self.expert_disc_network.parameters(), lr=float(configs['train']['lr_disc']))

        self.safe_disc_network = nn.Sequential(
            nn.Linear(self.obs_dim + self.action_dim, self.disc_hidden_size[0], device=self.device),
            nn.ReLU(inplace=True),
            nn.Linear(self.disc_hidden_size[0], self.disc_hidden_size[1], device=self.device),
            nn.ReLU(inplace=True),
            nn.Linear(self.disc_hidden_size[1], 1, device=self.device),
            nn.Sigmoid()
        )
        self.safe_disc_optimizer = optim.Adam(self.safe_disc_network.parameters(), lr=float(configs['train']['lr_disc']))

        self.safe_disc_threshold = 0.
        self.safe_disc_quantile = float(configs['train']['disc_quantile'])
        self.expert_disc_threshold = 0.
        self.expert_disc_quantile = float(configs['train']['disc_quantile'])
        self.threshold_update_freq = int(configs['train']['threshold_update_freq'])
        self.disc_learning_during_train = configs['train']['disc_learning_during_train']

        self.current_expert_disc_gradient_penalty = 0.
        self.current_safe_disc_gradient_penalty = 0.
        
        # For standardization
        self.obs_standardize = configs['replay_buffer']['standardize_obs']
        self.act_standardize = configs['replay_buffer']['standardize_act']

        if self.obs_standardize:
            self.obs_mean = self.mixed_replay_buffer.obs_mean
            self.obs_std = self.mixed_replay_buffer.obs_std 
            self.obs_mean_tt = torch.tensor(self.obs_mean, device=self.device)
            self.obs_std_tt = torch.tensor(self.obs_std, device=self.device)
        else:
            self.obs_mean = None
            self.obs_std = None
            self.obs_mean_tt = None
            self.obs_std_tt = None
            
        if self.act_standardize:
            self.act_mean = self.mixed_replay_buffer.act_mean
            self.act_std = self.mixed_replay_buffer.act_std
            self.act_mean_tt = torch.tensor(self.mixed_replay_buffer.act_mean, device=self.device)
            self.act_std_tt = torch.tensor(self.mixed_replay_buffer.act_std, device=self.device)
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

    def calculate_gradient_penalty(self, network, expert_s_a, mixed_s_a):
        # Add gradient penalty term
        
        rand_coef = torch.rand(expert_s_a.size(0), 1, device=self.device)
        rand_coef = rand_coef.expand(expert_s_a.size())
        
        # Interpolate between expert and safe samples
        interpolated_s_a = rand_coef * expert_s_a + (1 - rand_coef) * mixed_s_a
        interpolated_s_a.requires_grad_(True)
        
        # Get discriminator output for interpolated samples
        interpolated_output = network(interpolated_s_a)
                        
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
        gradients = gradients.view(-1, gradients.size(-1)) # (batch_size, output_size)
        gradient_penalty = ((gradients.norm(2, dim=-1) - 1) ** 2).mean()

        return gradient_penalty
    
    def train(self, total_iteration=1e6, eval_freq=1000, batch_size=1024):
        
        max_score = -100000.

        safe_n = self.safe_replay_buffer._size
        safe_obs_total = torch.tensor(self.safe_replay_buffer._observations[:safe_n], dtype=torch.float32, device=self.device)
        safe_actions_total = torch.tensor(self.safe_replay_buffer._actions[:safe_n], dtype=torch.float32, device=self.device)
        safe_s_a_total = torch.cat([safe_obs_total, safe_actions_total], dim=-1)

        safe_disc_total = self.safe_disc_network(safe_s_a_total).reshape(-1).cpu().detach().numpy()
        safe_disc_threshold = np.quantile(safe_disc_total, 1 - self.safe_disc_quantile)
        self.safe_disc_threshold = safe_disc_threshold

        expert_n = self.expert_replay_buffer._size
        expert_obs_total = torch.tensor(self.expert_replay_buffer._observations[:expert_n], dtype=torch.float32, device=self.device)
        expert_actions_total = torch.tensor(self.expert_replay_buffer._actions[:expert_n], dtype=torch.float32, device=self.device)
        expert_s_a_total = torch.cat([expert_obs_total, expert_actions_total], dim=-1)

        expert_disc_total = self.expert_disc_network(expert_s_a_total).reshape(-1).cpu().detach().numpy()
        expert_disc_threshold = np.quantile(expert_disc_total, 1 - self.expert_disc_quantile)
        self.expert_disc_threshold = expert_disc_threshold

        for num in range(0, int(total_iteration)+1):
            # Safe Discriminator Training
            self.safe_disc_optimizer.zero_grad()
                    
            safe_batch = self.safe_replay_buffer.random_batch(batch_size, standardize=self.obs_standardize)
            mixed_batch = self.mixed_replay_buffer.random_batch(batch_size, standardize=self.obs_standardize)

            safe_obs = torch.tensor(safe_batch['observations'], dtype=torch.float32, device=self.device)
            safe_actions = torch.tensor(safe_batch['actions'], dtype=torch.float32, device=self.device)
            
            mixed_obs = torch.tensor(mixed_batch['observations'], dtype=torch.float32, device=self.device)
            mixed_actions = torch.tensor(mixed_batch['actions'], dtype=torch.float32, device=self.device)

            safe_s_a = torch.cat([safe_obs, safe_actions], dim=-1)
            mixed_s_a = torch.cat([mixed_obs, mixed_actions], dim=-1)

            safe_disc = self.safe_disc_network(safe_s_a)
            mixed_disc = self.safe_disc_network(mixed_s_a)

            disc_loss = -torch.log(safe_disc + 1e-10).mean() - torch.log(1 - mixed_disc + 1e-10).mean()
            
            # Add gradient penalty term
            if self.disc_gp_weight > 0:
                randperm = torch.randperm(batch_size)
                self.current_safe_disc_gradient_penalty = self.calculate_gradient_penalty(self.safe_disc_network, mixed_s_a, mixed_s_a[randperm], type='disc')
                disc_loss += self.disc_gp_weight * self.current_safe_disc_gradient_penalty
            else:
                self.current_safe_disc_gradient_penalty = 0.

            disc_loss.backward()
            self.safe_disc_optimizer.step()

            # Expert Discriminator Training
            self.expert_disc_optimizer.zero_grad()
                    
            expert_batch = self.expert_replay_buffer.random_batch(batch_size, standardize=self.obs_standardize)
            mixed_batch = self.mixed_replay_buffer.random_batch(batch_size, standardize=self.obs_standardize)

            expert_obs = torch.tensor(expert_batch['observations'], dtype=torch.float32, device=self.device)
            expert_actions = torch.tensor(expert_batch['actions'], dtype=torch.float32, device=self.device)
            
            mixed_obs = torch.tensor(mixed_batch['observations'], dtype=torch.float32, device=self.device)
            mixed_actions = torch.tensor(mixed_batch['actions'], dtype=torch.float32, device=self.device)

            expert_s_a = torch.cat([expert_obs, expert_actions], dim=-1)
            mixed_s_a = torch.cat([mixed_obs, mixed_actions], dim=-1)

            expert_disc = self.expert_disc_network(expert_s_a)
            mixed_disc = self.expert_disc_network(mixed_s_a)

            disc_loss = -torch.log(expert_disc + 1e-10).mean() - torch.log(1 - mixed_disc + 1e-10).mean()
            
            # Add gradient penalty term
            if self.disc_gp_weight > 0:
                randperm = torch.randperm(batch_size)
                self.current_expert_disc_gradient_penalty = self.calculate_gradient_penalty(self.expert_disc_network, mixed_s_a, mixed_s_a[randperm], type='disc')
                disc_loss += self.disc_gp_weight * self.current_expert_disc_gradient_penalty
            else:
                self.current_expert_disc_gradient_penalty = 0.

            disc_loss.backward()
            self.expert_disc_optimizer.step()

            mixed_batch = self.mixed_replay_buffer.random_batch(batch_size, standardize=self.obs_standardize)
            
            mixed_obs = torch.tensor(mixed_batch['observations'], dtype=torch.float32, device=self.device)
            mixed_actions = torch.tensor(mixed_batch['actions'], dtype=torch.float32, device=self.device)
            mixed_s_a = torch.cat([mixed_obs, mixed_actions], dim=-1)
            mixed_disc = self.expert_disc_network(mixed_s_a)
            
            expert_indicator = (mixed_disc > self.expert_disc_threshold).reshape(-1).detach().float()
            safe_indicator = (mixed_disc > self.safe_disc_threshold).reshape(-1).detach().float()

            train_loss = torch.mean(expert_indicator * safe_indicator * -self.policy.log_prob(mixed_obs, mixed_actions))
            
            self.policy_optimizer.zero_grad()
            train_loss.backward()
            self.policy_optimizer.step()

            if (num) % self.threshold_update_freq == 0  and self.disc_learning_during_train:
                safe_disc_total = self.safe_disc_network(safe_s_a_total).reshape(-1).cpu().detach().numpy()
                safe_disc_threshold = np.quantile(safe_disc_total, 1 - self.safe_disc_quantile)
                self.safe_disc_threshold = safe_disc_threshold
                
                expert_disc_total = self.expert_disc_network(expert_s_a_total).reshape(-1).cpu().detach().numpy()
                expert_disc_threshold = np.quantile(expert_disc_total, 1 - self.expert_disc_quantile)
                self.expert_disc_threshold = expert_disc_threshold

            if (num) % eval_freq == 0:
                self.policy.eval()
                eval_ret_mean, eval_ret_std, eval_cost_mean, eval_cost_std, eval_violation_rate, eval_length_mean, eval_feasible_ret_mean, eval_feasible_ret_std = \
                    self.evaluate(self.env, self.policy, num_evaluation=self.num_eval_iteration)

                print(f'** iter{num}: policy_loss={train_loss.item():.2f}, ret={eval_ret_mean:.2f}+-{eval_ret_std:.2f}, cost={eval_cost_mean:.2f}+-{eval_cost_std:.2f}, violation_rate={eval_violation_rate:.2f}, length={eval_length_mean:.2f}')
                
                num_nonzero_samples = (expert_indicator * safe_indicator).sum().item()
                num_nonzero_samples_expert = expert_indicator.sum().item()
                num_nonzero_samples_safe = safe_indicator.sum().item()

                if self.use_wandb:
                    self.wandb.log({'train/policy_loss':       train_loss.item(), 
                               'eval/episode_return':      eval_ret_mean,
                               'eval/episode_cost':        eval_cost_mean,
                               'eval/violation_rate':      eval_violation_rate,
                               'eval/episode_length':      eval_length_mean,
                               'eval/feasible_return':     eval_feasible_ret_mean,
                               'eval/feasible_return_std': eval_feasible_ret_std,
                               'eval/safe_disc_threshold': self.safe_disc_threshold,
                               'eval/expert_disc_threshold': self.expert_disc_threshold,
                               'eval/num_nonzero_samples': num_nonzero_samples,
                               'eval/num_nonzero_samples_expert': num_nonzero_samples_expert,
                               'eval/num_nonzero_samples_safe': num_nonzero_samples_safe,
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
    
