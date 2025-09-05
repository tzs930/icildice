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
#         'cost_threshold': 25.,
#         'safe_expert_score': 44.30,
#         'unsafe_expert_score': 54.55,
#         'random_score': 0., 
#     }
# }

def copy_nn_module(source, target):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class DemoDICESafe(nn.Module):
    def __init__(self, policy, env, configs, best_policy=None,
                 init_obs_buffer=None, expert_replay_buffer=None, safe_replay_buffer=None, seed=0, 
                 n_train=1):
        
        seed = configs['train']['seed']
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        super(DemoDICESafe, self).__init__()

        self.env = env
        self.policy = policy
        self.best_policy = best_policy

        self.init_obs_buffer = init_obs_buffer
        self.expert_replay_buffer = expert_replay_buffer
        self.safe_replay_buffer = safe_replay_buffer

        self.add_absorbing_state = configs['replay_buffer']['use_absorbing_state']
        
        self.device = configs['device']
        
        self.n_train = n_train
    
        if self.add_absorbing_state:
            self.obs_dim = env.observation_space.low.size + 1
        else:
            self.obs_dim = env.observation_space.low.size
        self.action_dim = env.action_space.low.size
        
        self.gamma = configs['train']['gamma']
        
        self.nu_hidden_size = configs['train']['nu_hidden_size']
        self.nu_network = nn.Sequential(
            nn.Linear(self.obs_dim, self.nu_hidden_size[0], device=self.device),
            nn.ReLU(inplace=True),
            nn.Linear(self.nu_hidden_size[0], self.nu_hidden_size[1], device=self.device),
            nn.ReLU(inplace=True),
            nn.Linear(self.nu_hidden_size[1], 1, device=self.device)
        )

        self.disc_hidden_size = configs['train']['disc_hidden_size']
        self.disc_network = nn.Sequential(
            nn.Linear(self.obs_dim + self.action_dim, self.disc_hidden_size[0], device=self.device),
            nn.ReLU(inplace=True),
            nn.Linear(self.disc_hidden_size[0], self.disc_hidden_size[1], device=self.device),
            nn.ReLU(inplace=True),
            nn.Linear(self.disc_hidden_size[1], 1, device=self.device),
            nn.Sigmoid()
        )
        
        self.r_fn = \
            lambda x: - torch.log( 1 / (self.disc_network(x) + 1e-10) - 1 + 1e-10)
        
        self.nu_optimizer = optim.Adam(self.nu_network.parameters(), lr=float(configs['train']['lr_nu']))
        self.disc_optimizer = optim.Adam(self.disc_network.parameters(), lr=float(configs['train']['lr_disc']))
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=float(configs['train']['lr']))
 
        self.alpha = float(configs['train']['alpha'])
        self.disc_steps = int(configs['train']['disc_steps'])
        self.disc_gp_weight = float(configs['train']['disc_gp_weight'])  # Gradient penalty coefficient
        self.pretrain_steps = int(configs['train']['pretrain_steps'])
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
            self.obs_mean = self.replay_buffer.obs_mean
            self.obs_std = self.replay_buffer.obs_std 
            self.obs_mean_tt = torch.tensor(self.obs_mean, device=self.device)
            self.obs_std_tt = torch.tensor(self.obs_std, device=self.device)
        else:
            self.obs_mean = None
            self.obs_std = None
            self.obs_mean_tt = None
            self.obs_std_tt = None
            
        if self.act_standardize:
            self.act_mean = self.replay_buffer.act_mean
            self.act_std = self.replay_buffer.act_std
            self.act_mean_tt = torch.tensor(self.replay_buffer.act_mean, device=self.device)
            self.act_std_tt = torch.tensor(self.replay_buffer.act_std, device=self.device)
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

        for num in range(0, int(self.pretrain_steps)):
            self.disc_optimizer.zero_grad()
            expert_batch = self.expert_replay_buffer.random_batch(batch_size, standardize=self.obs_standardize)
            safe_batch = self.safe_replay_buffer.random_batch(batch_size, standardize=self.obs_standardize)

            expert_obs = torch.tensor(expert_batch['observations'], dtype=torch.float32, device=self.device)
            expert_actions = torch.tensor(expert_batch['actions'], dtype=torch.float32, device=self.device)

            safe_obs = torch.tensor(safe_batch['observations'], dtype=torch.float32, device=self.device)
            safe_actions = torch.tensor(safe_batch['actions'], dtype=torch.float32, device=self.device)

            expert_s_a = torch.cat([expert_obs, expert_actions], dim=-1)
            safe_s_a = torch.cat([safe_obs, safe_actions], dim=-1)

            expert_disc = self.disc_network(expert_s_a)
            safe_disc = self.disc_network(safe_s_a)

            disc_loss = -torch.log(expert_disc + 1e-10).mean() - torch.log(1 - safe_disc + 1e-10).mean()
            
            # Add gradient penalty term
            if self.disc_gp_weight > 0:
                gradient_penalty = self.calculate_gradient_penalty(self.disc_network, expert_s_a, safe_s_a)
                disc_loss += self.disc_gp_weight * gradient_penalty
            else:
                gradient_penalty = 0.
            self.current_gradient_penalty = gradient_penalty                

            disc_loss.backward()
            self.disc_optimizer.step()
        
        for num in range(0, int(total_iteration)+1):
            for _ in range(self.disc_steps):
                self.disc_optimizer.zero_grad()
                expert_batch = self.expert_replay_buffer.random_batch(batch_size, standardize=self.obs_standardize)
            
                safe_batch = self.safe_replay_buffer.random_batch(batch_size, standardize=self.obs_standardize)

                expert_obs = torch.tensor(expert_batch['observations'], dtype=torch.float32, device=self.device)
                expert_actions = torch.tensor(expert_batch['actions'], dtype=torch.float32, device=self.device)

                safe_obs = torch.tensor(safe_batch['observations'], dtype=torch.float32, device=self.device)
                safe_actions = torch.tensor(safe_batch['actions'], dtype=torch.float32, device=self.device)

                expert_s_a = torch.cat([expert_obs, expert_actions], dim=-1)
                safe_s_a = torch.cat([safe_obs, safe_actions], dim=-1)

                expert_disc = self.disc_network(expert_s_a)
                safe_disc = self.disc_network(safe_s_a)

                disc_loss = -torch.log(expert_disc + 1e-10).mean() - torch.log(1 - safe_disc + 1e-10).mean()
                
                # Add gradient penalty term
                if self.disc_gp_weight > 0:
                    gradient_penalty = self.calculate_gradient_penalty(self.disc_network, expert_s_a, safe_s_a)
                    disc_loss += self.disc_gp_weight * gradient_penalty
                else:
                    gradient_penalty = 0.                
                self.current_gradient_penalty = gradient_penalty

                disc_loss.backward()
                self.disc_optimizer.step()

           
            self.nu_optimizer.zero_grad()

            init_obs = self.init_obs_buffer.random_batch(batch_size, standardize=self.obs_standardize)
            expert_batch = self.expert_replay_buffer.random_batch(batch_size, standardize=self.obs_standardize)
            safe_batch = self.safe_replay_buffer.random_batch(batch_size, standardize=self.obs_standardize)

            # nu (lagrangian) training
            init_obs = torch.tensor(init_obs, dtype=torch.float32, device=self.device)
            safe_obs = torch.tensor(safe_batch['observations'], dtype=torch.float32, device=self.device)
            safe_actions = torch.tensor(safe_batch['actions'], dtype=torch.float32, device=self.device)
            safe_next_obs = torch.tensor(safe_batch['next_observations'], dtype=torch.float32, device=self.device)

            safe_s_a = torch.cat([safe_obs, safe_actions], dim=-1)
            safe_r = self.r_fn(safe_s_a).reshape(-1)

            init_nu = self.nu_network(init_obs).reshape(-1)
            safe_nu = self.nu_network(safe_obs).reshape(-1)
            safe_nu_prime = self.nu_network(safe_next_obs).reshape(-1)

            safe_adv = safe_r.detach() + self.gamma * safe_nu_prime - safe_nu
            nu_loss0 = (1 - self.gamma) * init_nu.mean()
            nu_loss1 = (1 + self.alpha) * torch.logsumexp(safe_adv / (1+self.alpha), -1) #.mean()

            nu_loss = nu_loss0 + nu_loss1

            self.nu_optimizer.zero_grad()
            nu_loss.backward()
            self.nu_optimizer.step()

            # Policy training (weighted BC)
            self.policy_optimizer.zero_grad()

            # weights corresponds to KL (with reweighting)
            policy_weight = torch.exp( (safe_nu - safe_nu.max()) / (self.alpha + 1) )
            policy_weight = (policy_weight / policy_weight.sum()).detach()
            policy_loss = - (policy_weight * self.policy.log_prob(safe_obs, safe_actions)).mean()

            policy_loss.backward()
            self.policy_optimizer.step()
            
            if (num) % eval_freq == 0:
                self.policy.eval()
                eval_ret_mean, eval_ret_std, eval_cost_mean, eval_cost_std, eval_violation_rate, eval_length_mean, eval_feasible_ret_mean, eval_feasible_ret_std = \
                    self.evaluate(self.env, self.policy, num_evaluation=self.num_eval_iteration)
                
                print(f'** iter{num}: policy_loss={policy_loss.item():.2f}, ret={eval_ret_mean:.2f}+-{eval_ret_std:.2f}, cost={eval_cost_mean:.2f}+-{eval_cost_std:.2f}, violation_rate={eval_violation_rate:.2f}, length={eval_length_mean:.2f}')
                
                if self.use_wandb:
                    self.wandb.log({'train/policy_loss':       policy_loss.item(), 
                                    'train/nu_loss':           nu_loss.item(),
                                    'train/disc_loss':         disc_loss.item(),
                                    'train/disc_gradient_penalty':  self.current_gradient_penalty,
                                    'eval/episode_return':     eval_ret_mean,
                                    'eval/episode_cost':       eval_cost_mean,
                                    'eval/violation_rate':     eval_violation_rate,
                                    'eval/episode_length':     eval_length_mean,
                                    'eval/feasible_return':     eval_feasible_ret_mean,
                                    'eval/feasible_return_std': eval_feasible_ret_std,
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
    
