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

class BC(nn.Module):
    def __init__(self, policy, env, best_policy=None,
                 replay_buffer=None, replay_buffer_valid=None, seed=0, 
                 device='cpu', lr=3e-4, envname=None, wandb=None, save_policy_path=None, 
                 obs_dim=1, action_dim=1, stacksize=1, standardize=True, expert_policy=None,
                 n_train=1, n_valid=1, add_absorbing_state=False):
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        super(BC, self).__init__()

        self.env = env
        self.policy = policy
        self.best_policy = best_policy
        self.replay_buffer = replay_buffer
        self.replay_buffer_valid = replay_buffer_valid
        self.device = device
        self.add_absorbing_state = add_absorbing_state
        
        self.n_train = n_train
        self.n_valid = n_valid
    
        self.obs_dim = obs_dim
        self.action_dim = action_dim        
        self.stacksize = stacksize
        
        self.policy_optimizer = optim.Adam(policy.parameters(), lr=lr)
        
        self.num_eval_iteration = 100
        self.envname = envname
        
        self.wandb = None
        if wandb:
            self.wandb = wandb
            self.wandb.init()

        self.save_policy_path = save_policy_path        
        
        # For standardization
        self.standardize = standardize

        self.obs_mean_tt = torch.tensor(self.replay_buffer.obs_mean, device=device)
        self.obs_std_tt = torch.tensor(self.replay_buffer.obs_std, device=device)
        self.act_mean_tt = torch.tensor(self.replay_buffer.act_mean, device=device)
        self.act_std_tt = torch.tensor(self.replay_buffer.act_std, device=device)

        self.obs_mean = self.replay_buffer.obs_mean
        self.obs_std = self.replay_buffer.obs_std # For numerical stability
        self.act_mean = self.replay_buffer.act_mean
        self.act_std = self.replay_buffer.act_std

        self.expert_policy = expert_policy
        

    def train(self, total_iteration=1e6, eval_freq=1000, batch_size=1024):
        
        max_score = -100000.
        
        batch_valid = self.replay_buffer_valid.random_batch(self.n_valid, standardize=self.standardize)
        
        obs_valid = batch_valid['observations']
        actions_valid = batch_valid['actions'][:, -self.action_dim:]        
        prev_expert_action_valid = batch_valid['actions'][:, :-self.action_dim] # For debugging
                
        obs_valid = torch.tensor(obs_valid, dtype=torch.float32, device=self.device)
        actions_valid = torch.tensor(actions_valid, dtype=torch.float32, device=self.device)
        prev_expert_action_valid = torch.tensor(prev_expert_action_valid, dtype=torch.float32, device=self.device)
        
        for num in range(0, int(total_iteration)+1):
            batch = self.replay_buffer.random_batch(batch_size, standardize=self.standardize)
            
            obs = batch['observations']
            actions = batch['actions'][:, -self.action_dim:]
            
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
            actions = torch.tensor(actions, dtype=torch.float32, device=self.device)            

            # neg_likelihood = -self.policy.log_prob(obs, actions).mean()
            train_epsilon = self.policy(obs).mean - actions
            train_mse = (train_epsilon ** 2).mean()
            train_loss = train_mse
            # train_hsic = estimate_hsic(X=obs, Y= train_epsilon, Kx_sigma2=1., Ky_sigma2=1.)
            
            self.policy_optimizer.zero_grad()
            train_loss.backward()
            self.policy_optimizer.step()

            if (num) % eval_freq == 0:
                valid_epsilon = self.policy(obs_valid).mean - actions_valid
                valid_mse = (valid_epsilon ** 2).mean()
                valid_loss = valid_mse
                
                # valid_hsic = estimate_hsic(X=obs_valid, Y= valid_epsilon, Kx_sigma2=1., Ky_sigma2=1.)

                eval_ret_mean, eval_ret_std, true_mse, CVAR_1percent, CVAR_5percent, CVAR_25percent = self.evaluate(num_iteration=self.num_eval_iteration)
                
                print(f'** iter{num}: train_policy_loss={train_loss.item():.2f}, val_policy_loss={valid_loss.item():.2f}, eval_ret={eval_ret_mean:.2f}+-{eval_ret_std:.2f} ({obs_valid.shape[0]})',)
                print(f'-- MSE :  (train) {train_mse:.6f} (valid) {valid_mse:.6f} (true) {true_mse:.6f}')
                print(f'-- CVAR_1percent={CVAR_1percent:.6f}, CVAR_5percent={CVAR_5percent:.6f}, CVAR_25percent={CVAR_25percent:.6f}')
                # print(f'** HSIC : (train) {train_hsic:.6f} (valid) {valid_hsic:.6f}')
                
                if self.wandb:
                    self.wandb.log({'train_policy_loss':       train_loss.item(), 
                                    'valid_policy_loss':       valid_loss.item(),
                                    'train_mse':               train_mse.item(),
                                    'valid_mse':               valid_mse.item(),
                                    'target_mse':              true_mse,
                                    # 'train_HSIC':             train_hsic.item(),
                                    # 'valid_HSIC':             valid_hsic.item(),
                                    'eval/episode_return':     eval_ret_mean,
                                    'eval/CVAR1':              CVAR_1percent,
                                    'eval/CVAR5':              CVAR_5percent,
                                    'eval/CVAR25':             CVAR_25percent,
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
                    
                    
    def evaluate(self, num_iteration=5):
        rets = []
        true_se_list = []

        maxtimestep = 1000
        for num in range(0, num_iteration):
            obs_ = self.env.reset()
            
            done = False
            t = 0
            ret = 0.
            
            while not done and t < maxtimestep:
                # Before standardization
                expert_input = np.expand_dims(obs_, axis=0).astype(np.float32)                
                noise_input = np.zeros([1, self.env.action_space.shape[0]]).astype(np.float32)
                expert_action, _, _ = self.expert_policy.run(None, {'observations': expert_input, 'noise': noise_input})

                if self.add_absorbing_state:
                    obs = np.concatenate([obs_, [0.]]) # absorbing
                else:
                    obs = obs_
                    
                if self.standardize:
                    obs = (obs - self.obs_mean[0]) / (self.obs_std[0])
                obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
                action = self.policy(obs).mean.cpu().detach().numpy()

                true_se = np.mean((expert_action - action) ** 2)
                true_se_list.append(true_se)
                
                next_obs, rew, done, _ = self.env.step(action)
                ret += rew
                
                obs_ = next_obs 
                    
                t += 1
            
            rets.append(ret)

        num_CVAR_1percent = int(num_iteration * 0.01)
        CVAR_1percent = np.sort(rets)[:num_CVAR_1percent].mean()

        num_CVAR_5percent = int(num_iteration * 0.05)
        CVAR_5percent = np.sort(rets)[:num_CVAR_5percent].mean()

        num_CVAR_25percent = int(num_iteration * 0.25)
        CVAR_25percent = np.sort(rets)[:num_CVAR_25percent].mean()

        return np.mean(rets), np.std(rets), np.mean(true_se_list), CVAR_1percent, CVAR_5percent, CVAR_25percent
    
