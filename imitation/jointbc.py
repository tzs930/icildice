import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

def copy_nn_module(source, target):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

class JointBC(nn.Module):
    def __init__(self, policy, env, configs, best_policy=None,
                 expert_replay_buffer=None, safe_replay_buffer=None, n_train=1,add_absorbing_state=False):
        
        seed = configs['seed']
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        super(JointBC, self).__init__()

        self.env = env
        self.policy = policy
        self.best_policy = best_policy
        self.expert_replay_buffer = expert_replay_buffer
        self.safe_replay_buffer = safe_replay_buffer

        self.device = configs['device']
        self.add_absorbing_state = add_absorbing_state
        
        self.n_train = n_train
    
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
        
        # For standardization
        self.obs_standardize = configs['replay_buffer']['standardize_obs']
        self.act_standardize = configs['replay_buffer']['standardize_act']

        if self.obs_standardize:
            self.obs_mean = self.expert_replay_buffer.obs_mean
            self.obs_std = self.expert_replay_buffer.obs_std 
            self.obs_mean_tt = torch.tensor(self.obs_mean, device=self.device)
            self.obs_std_tt = torch.tensor(self.obs_std, device=self.device)
        else:
            self.obs_mean = None
            self.obs_std = None
            self.obs_mean_tt = None
            self.obs_std_tt = None
            
        if self.act_standardize:
            self.act_mean = self.expert_replay_buffer.act_mean
            self.act_std = self.expert_replay_buffer.act_std
            self.act_mean_tt = torch.tensor(self.expert_replay_buffer.act_mean, device=self.device)
            self.act_std_tt = torch.tensor(self.expert_replay_buffer.act_std, device=self.device)
        else:
            self.act_mean = None
            self.act_std = None
            self.act_mean_tt = None
            self.act_std_tt = None


    def train(self, total_iteration=1e6, eval_freq=1000, batch_size=1024):
        
        max_score = -100000.
        
        for num in range(0, int(total_iteration)+1):
            self.policy.train()
            expert_batch = self.expert_replay_buffer.random_batch(batch_size, standardize=self.obs_standardize)
            safe_batch = self.safe_replay_buffer.random_batch(batch_size, standardize=self.obs_standardize)
            
            expert_obs = expert_batch['observations']
            expert_actions = expert_batch['actions']
            
            safe_obs = safe_batch['observations']
            safe_actions = safe_batch['actions']
            
            expert_obs = torch.tensor(expert_obs, dtype=torch.float32, device=self.device)
            expert_actions = torch.tensor(expert_actions, dtype=torch.float32, device=self.device)
            
            safe_obs = torch.tensor(safe_obs, dtype=torch.float32, device=self.device)
            safe_actions = torch.tensor(safe_actions, dtype=torch.float32, device=self.device)
            
            train_loss = - self.policy.log_prob(expert_obs, expert_actions).mean() \
                         - self.policy.log_prob(safe_obs, safe_actions).mean()
            
            self.policy_optimizer.zero_grad()
            train_loss.backward()
            self.policy_optimizer.step()

            if (num) % eval_freq == 0:
                self.policy.eval()
                eval_ret_mean, eval_ret_std, eval_cost_mean, eval_cost_std, eval_violation_rate, eval_length_mean = \
                    self.evaluate(self.env, self.policy, num_evaluation=self.num_eval_iteration)
                
                print(f'** iter{num}: policy_loss={train_loss.item():.2f}, ret={eval_ret_mean:.2f}+-{eval_ret_std:.2f}, cost={eval_cost_mean:.2f}+-{eval_cost_std:.2f}, violation_rate={eval_violation_rate:.2f}, length={eval_length_mean:.2f}')
                
                if self.use_wandb:
                    self.wandb.log({'train/policy_loss':       train_loss.item(), 
                               'eval/episode_return':     eval_ret_mean,
                               'eval/episode_cost':       eval_cost_mean,
                               'eval/violation_rate':     eval_violation_rate,
                               'eval/episode_length':     eval_length_mean,
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

        # maxtimestep = 1000
        for num in range(0, num_evaluation):
            obs_, info_ = env.reset()
            
            done = False
            t = 0
            ret = 0.
            cum_cost = 0.
            
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
                
                next_obs, rew, done, info = env.step(action)
                cost = info['cost']

                ret += rew
                cum_cost += cost
                
                obs_ = next_obs 
                    
                t += 1
            
            rets.append(ret)
            costs.append(cum_cost)
            lengths.append(t)

        violation_rate = np.mean(np.array(costs) > 0)

        return np.mean(rets), np.std(rets)/np.sqrt(num_evaluation), np.mean(costs), np.std(costs)/np.sqrt(num_evaluation), violation_rate, np.mean(lengths)
        # eval_ret_mean, eval_ret_std, eval_cost_mean, eval_cost_std, eval_violation_rate, eval_length_mean
