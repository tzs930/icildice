import os
wandb_dir = './wandb_offline'
os.environ['WANDB_DIR'] = wandb_dir
os.environ['D4RL_DATASET_DIR'] = './dataset'
import wandb
# import envs
# import gym
import safety_gymnasium as safety_gym
import gymnasium as gym

import pickle5 as pickle
import yaml

import numpy as np
import torch
import time

from imitation.bc import BC
from imitation.bc_filtered import BCFiltered
from imitation.jointbc import JointBC
from imitation.bc_jointfiltered import BCJointFiltered
from imitation.demodice import DemoDICESafe
from imitation.demodice_reg import DemoDICEReg
from imitation.icildice import IcilDICE
from imitation.icildice_v2 import IcilDICEv2
from imitation.icildice_v3 import IcilDICEv3

from argparse import ArgumentParser
from itertools import product

from core.policy import TanhGaussianPolicy, GaussianPolicy
from core.replay_buffer import InitObsBuffer, MDPReplayBuffer
from core.preprocess import preprocess_dataset
from rlkit.envs.wrappers import NormalizedBoxEnv

# import onnx
# import onnxruntime as ort

STD_EPSILON = 1e-8

def train(configs):
    # env = NormalizedBoxEnv(gym.make(configs['env_id']))
    env = safety_gym.make(configs['env_id'])
    
    obs_dim    = env.observation_space.low.size 
    action_dim = env.action_space.low.size
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # expert_dataset_path = configs['dataset']['expert_path']
    
    envname = configs['env_id']
    print(f'-- Preprocessing dataset... ({envname})')
    expert_data_dict, n_expert = preprocess_dataset(configs['dataset']['dataset_path'],
                                                   configs['dataset']['expert']['types'], 
                                                   configs['dataset']['expert']['start_indices'], 
                                                   num_rollouts=configs['dataset']['expert']['num_trajs'])
    
    init_obs_list = [expert_data_dict['initial_observations']]
    if configs['dataset']['safe']['types'] is not None:
        if len(configs['dataset']['safe']['types']) == 0:
            safe_data_dict = None
            n_safe = 0
        else:
            safe_data_dict, n_safe = preprocess_dataset(configs['dataset']['dataset_path'],
                                                    configs['dataset']['safe']['types'], 
                                                    configs['dataset']['safe']['start_indices'], 
                                                    num_rollouts=configs['dataset']['safe']['num_trajs'])
            init_obs_list.append(safe_data_dict['initial_observations'])
    else:
        safe_data_dict = None
        n_safe = 0

    if configs['dataset']['mixed']['types'] is not None:
        if len(configs['dataset']['mixed']['types']) == 0:
            mixed_data_dict = None
            n_mixed = 0
        else:
            mixed_data_dict, n_mixed = preprocess_dataset(configs['dataset']['dataset_path'],
                                                    configs['dataset']['mixed']['types'], 
                                                    configs['dataset']['mixed']['start_indices'], 
                                                    num_rollouts=configs['dataset']['mixed']['num_trajs'])
            init_obs_list.append(mixed_data_dict['initial_observations'])
    else:
        mixed_data_dict = None
        n_mixed = 0

    init_obs_datapoints = np.concatenate(init_obs_list)

    print(f'** num. of expert data points : {n_expert}')
    print(f'** num. of safe data points : {n_safe}')
    print(f'** num. of mixed data points : {n_mixed}')
    n_train = n_expert + n_safe + n_mixed
    print(f'** num. of total train data points : {n_train}')

    expert_replay_buffer = MDPReplayBuffer(
        configs['replay_buffer']['max_replay_buffer_size'],
        env,
        obs_dim=obs_dim
    )
    expert_replay_buffer.add_path(expert_data_dict)

    if safe_data_dict is not None:
        safe_replay_buffer = MDPReplayBuffer(
            configs['replay_buffer']['max_replay_buffer_size'],
            env,
            obs_dim=obs_dim
        )
        
        safe_replay_buffer.add_path(safe_data_dict)

    if mixed_data_dict is not None:
        mixed_replay_buffer = MDPReplayBuffer(
            configs['replay_buffer']['max_replay_buffer_size'],
            env,
            obs_dim=obs_dim
        )
        mixed_replay_buffer.add_path(mixed_data_dict)
    
    init_obs_buffer = InitObsBuffer(
        env, init_obs_datapoints
    )
    
    if configs['replay_buffer']['standardize_obs'] or configs['replay_buffer']['standardize_act']:
        obs_mean, obs_std, act_mean, act_std = expert_replay_buffer.calculate_statistics(
            standardize_obs=configs['replay_buffer']['standardize_obs'],
            standardize_act=configs['replay_buffer']['standardize_act']
        )
        init_obs_buffer.set_statistics(obs_mean, obs_std)

        if safe_data_dict is not None:
            safe_replay_buffer.set_statistics(obs_mean, obs_std, act_mean, act_std)
        if mixed_data_dict is not None:
            mixed_replay_buffer.set_statistics(obs_mean, obs_std, act_mean, act_std)
    else:
        obs_mean, obs_std, act_mean, act_std = None, None, None, None
    
    # Setup Policy
    if 'GaussianPolicy' == configs['policy']['class']:
        policy = GaussianPolicy(
                hidden_sizes=configs['policy']['layer_sizes'],
                obs_dim=obs_dim,
                action_dim=action_dim,
                device=device
            )
            
        best_policy = GaussianPolicy(
            hidden_sizes=configs['policy']['layer_sizes'],
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=device
        )

    elif 'TanhGaussianPolicy' == configs['policy']['class']:
        policy = TanhGaussianPolicy(
            hidden_sizes=configs['policy']['layer_sizes'],
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=device
        )

        best_policy = TanhGaussianPolicy(
            hidden_sizes=configs['policy']['layer_sizes'],
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=device
        )
    else:
        raise NotImplementedError
    
    # Setup Trainer and Start Training
    if 'BC' == configs['method']:
        trainer = BC(
            policy = policy,
            env = env,
            configs=configs,
            best_policy = best_policy,
            replay_buffer=expert_replay_buffer,
            n_train=n_train,
        )

        trainer.train(total_iteration = configs['train']['total_iteration'],
                      eval_freq = configs['train']['eval_freq'],
                      batch_size = configs['train']['batch_size'])

    elif 'BCFiltered' == configs['method']:
        trainer = BCFiltered(
            policy = policy,
            env = env,
            configs=configs,
            best_policy = best_policy,
            expert_replay_buffer=expert_replay_buffer,
            safe_replay_buffer=safe_replay_buffer,
            mixed_replay_buffer=mixed_replay_buffer,
            n_train=n_train,
        )
        
        trainer.train(total_iteration = configs['train']['total_iteration'],
                      eval_freq = configs['train']['eval_freq'],
                      batch_size = configs['train']['batch_size'])

    elif 'BCJointFiltered' == configs['method']:
        trainer = BCJointFiltered(
            policy = policy,
            env = env,
            configs=configs,
            best_policy = best_policy,
            expert_replay_buffer=expert_replay_buffer,
            safe_replay_buffer=safe_replay_buffer,
            mixed_replay_buffer=mixed_replay_buffer,
            n_train=n_train,
        )
        trainer.train(total_iteration = configs['train']['total_iteration'],
                      eval_freq = configs['train']['eval_freq'],
                      batch_size = configs['train']['batch_size'])

    elif 'JointBC' == configs['method']:
        policy = GaussianPolicy(
                hidden_sizes=configs['policy']['layer_sizes'],
            obs_dim=obs_dim ,
            action_dim=action_dim,            
            device=device            
        )
        
        best_policy = GaussianPolicy(
            hidden_sizes=configs['policy']['layer_sizes'],
            obs_dim=obs_dim,
            action_dim=action_dim,            
            device=device            
        )
        
        trainer = JointBC(
            policy = policy,
            env = env,
            configs=configs,
            best_policy = best_policy,
            expert_replay_buffer=expert_replay_buffer,
            safe_replay_buffer=safe_replay_buffer,
            n_train=n_train,
        )

        trainer.train(total_iteration = configs['train']['total_iteration'],
                      eval_freq = configs['train']['eval_freq'],
                      batch_size = configs['train']['batch_size'])
    
    elif 'DemoDICESafe' == configs['method']:
        
        trainer = DemoDICESafe(
            policy = policy,
            best_policy = best_policy,
            env = env,
            configs = configs,
            init_obs_buffer = init_obs_buffer,
            expert_replay_buffer = expert_replay_buffer,
            safe_replay_buffer = safe_replay_buffer,
        )

        trainer.train(total_iteration = configs['train']['total_iteration'],
                      eval_freq = configs['train']['eval_freq'],
                      batch_size = configs['train']['batch_size'])

    elif 'DemoDICEReg' == configs['method']:

        trainer = DemoDICEReg(
            policy = policy,
            best_policy = best_policy,
            env = env,
            configs = configs,
            init_obs_buffer = init_obs_buffer,
            expert_replay_buffer = expert_replay_buffer,
            safe_replay_buffer = safe_replay_buffer,
            mixed_replay_buffer = mixed_replay_buffer,
        )

        trainer.train(total_iteration = configs['train']['total_iteration'],
                      eval_freq = configs['train']['eval_freq'],
                      batch_size = configs['train']['batch_size'])

    elif 'IcilDICEv3' in configs['method']:
        trainer = IcilDICEv3(
            policy = policy,
            best_policy = best_policy,
            env = env,
            configs = configs,
            init_obs_buffer = init_obs_buffer,
            expert_replay_buffer = expert_replay_buffer,
            safe_replay_buffer = safe_replay_buffer,
            mixed_replay_buffer = mixed_replay_buffer,
        )
        trainer.train(total_iteration = configs['train']['total_iteration'],
                      eval_freq = configs['train']['eval_freq'],
                      batch_size = configs['train']['batch_size'])

    elif 'IcilDICEv2' == configs['method']:
        trainer = IcilDICEv2(
            policy = policy,
            best_policy = best_policy,
            env = env,
            configs = configs,
            init_obs_buffer = init_obs_buffer,
            expert_replay_buffer = expert_replay_buffer,
            safe_replay_buffer = safe_replay_buffer,
            mixed_replay_buffer = mixed_replay_buffer,
        )
        trainer.train(total_iteration = configs['train']['total_iteration'],
                      eval_freq = configs['train']['eval_freq'],
                      batch_size = configs['train']['batch_size'])

    elif 'IcilDICE' == configs['method']:
        
        trainer = IcilDICE(
            policy = policy,
            best_policy = best_policy,
            env = env,
            configs = configs,
            init_obs_buffer = init_obs_buffer,
            expert_replay_buffer = expert_replay_buffer,
            safe_replay_buffer = safe_replay_buffer,
            mixed_replay_buffer = mixed_replay_buffer,
        )

        trainer.train(total_iteration = configs['train']['total_iteration'],
                      eval_freq = configs['train']['eval_freq'],
                      batch_size = configs['train']['batch_size'])
    
    else: 
        raise NotImplementedError       

 
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--pid", help="process_id", default=0, type=int)
    parser.add_argument("--config", help="config file", default="configs/SafetyPointCircle1-v0/bc_jointfiltered.yaml")
    args = parser.parse_args()
    pid = args.pid

    configs = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    configs['seed'] = pid

    # print(configs)
    train(configs)
