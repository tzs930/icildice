import numpy as np
import pickle5
import os

def preprocess_dataset(dataset_path, dataset_types, start_indices, num_rollouts, use_absorbing_state=False, max_episode_len=None):
    
    # e.g. dataset_types = ['safe-expert-v0', 'safe-medium-v0']
    #      dataset

    data_dict = {
            'initial_observations': [],
            'observations': [],
            'actions': [],
            'rewards': [],
            'costs': [],
            'next_observations': [],
            'terminals': [],
            'episode_indices': []
        }
    
    for dataset_type, start_idx, rollout_per_dataset in zip(dataset_types, start_indices, num_rollouts):
        filepath = f'{dataset_path}/{dataset_type}.pkl'
        
        with open(filepath, 'rb') as f:
            dataset = pickle5.load(f)

        # total_episodes = len(dataset)
        for episode_idx in range(start_idx, start_idx + rollout_per_dataset):
            episode = dataset[episode_idx]

            initial_observation = episode['observations'][0]
            observations = episode['observations']
            actions = episode['actions']
            rewards = episode['rewards']
            costs = episode['costs']
            terminals = episode['terminals']
            next_observations = episode['next_observations']
            episode_len = len(observations)

            if use_absorbing_state:
                initial_observation = np.concatenate([initial_observation, np.zeros((1), dtype=initial_observation.dtype)])
                observations = np.concatenate([observations, np.zeros((episode_len, 1), dtype=observations.dtype)], axis=1)
                next_observations = np.concatenate([next_observations, np.zeros((episode_len, 1), dtype=next_observations.dtype)], axis=1)

                if terminals[-1] and episode_len < max_episode_len:
                    absorbing_state = np.zeros(observations.shape[1], dtype=observations.dtype)
                    absorbing_state[-1] = 1.
                    next_observations[-1] = absorbing_state

                    observations = np.vstack([observations, absorbing_state[None]])
                    actions = np.vstack([actions, np.zeros((1, actions.shape[1]), dtype=actions.dtype)])
                    rewards = np.concatenate([rewards, np.array([0.], dtype=rewards.dtype)])
                    costs = np.concatenate([costs, np.array([0.], dtype=costs.dtype)])
                    terminals = np.concatenate([terminals, np.array([True], dtype=terminals.dtype)])
                    next_observations = np.vstack([next_observations, absorbing_state[None]])
            
            data_dict['initial_observations'].append(initial_observation)
            data_dict['observations'].extend(observations)
            data_dict['actions'].extend(actions)
            data_dict['rewards'].extend(rewards)
            data_dict['costs'].extend(costs)
            data_dict['terminals'].extend(terminals)
            data_dict['next_observations'].extend(next_observations)

    data_dict['initial_observations'] = np.array(data_dict['initial_observations'])
    data_dict['observations'] = np.array(data_dict['observations'])
    data_dict['actions'] = np.array(data_dict['actions'])
    data_dict['rewards'] = np.array(data_dict['rewards'])
    data_dict['costs'] = np.array(data_dict['costs'])
    data_dict['next_observations'] = np.array(data_dict['next_observations'])
    data_dict['terminals'] = np.array(data_dict['terminals'])

    n_train = len(data_dict['observations'])

    return data_dict, n_train


def preprocess_dataset_per_episode(filepath, start_idx=0, num_rollouts=100):
    
    # e.g. filepath = 'dataset/BlockedPendulum-v0/random-v0'
    #      dataset
    # with open(filepath, 'rb') as f:
    #     dataset = pickle5.load(f)

    data_dict = {
        'initial_observations': [],
        'observations': [],
        'actions': [],
        'rewards': [],
        'costs': [],
        'next_observations': [],
        'terminals': [],
        'episode_indices': []
    }

    for file in os.listdir(filepath):
        episode_idx = int(file.split('_')[0].split('-')[1])
        if episode_idx >= start_idx and episode_idx < start_idx + num_rollouts:
            with open(os.path.join(filepath, file), 'rb') as f:
                data = pickle5.load(f)
                # print(data)
                data_dict['initial_observations'].append(data['observations'][0])
                data_dict['observations'].extend(list(data['observations']))
                data_dict['actions'].extend(list(data['actions']))
                data_dict['rewards'].extend(list(data['rewards']))
                data_dict['costs'].extend(list(data['costs']))
                data_dict['next_observations'].extend(list(data['next_observations']))
                data_dict['terminals'].extend(list(data['terminals']))
                data_dict['episode_indices'].extend([episode_idx] * len(data['observations']))

    data_dict['initial_observations'] = np.array(data_dict['initial_observations'])
    data_dict['observations'] = np.array(data_dict['observations'])
    data_dict['actions'] = np.array(data_dict['actions'])
    data_dict['rewards'] = np.array(data_dict['rewards'])
    data_dict['costs'] = np.array(data_dict['costs'])
    data_dict['next_observations'] = np.array(data_dict['next_observations'])
    data_dict['episode_indices'] = np.array(data_dict['episode_indices'])
    data_dict['terminals'] = np.array(data_dict['terminals'])

    return data_dict, len(data_dict['initial_observations'])
