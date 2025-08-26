import pickle5 as pickle
import numpy as np

dataset_path = 'dataset/SafetyPointCircle1-v0'

with open(f'{dataset_path}-unsafe-medium-v0.pkl', 'rb') as f:
    expert_data_dict = pickle.load(f)

rets = []
ccosts = []
for eps in expert_data_dict:
    ret = np.sum(eps['rewards'])
    ccost = np.sum(eps['costs'])
    rets.append(ret)
    ccosts.append(ccost)

print('--Return Statistics')
print('Mean: ', np.mean(rets))
print('Std: ', np.std(rets))
print('Max: ', np.max(rets))
print('Min: ', np.min(rets))

print('--Cost Statistics')
print('Mean: ', np.mean(ccosts))
print('Std: ', np.std(ccosts))
print('Max: ', np.max(ccosts))
print('Min: ', np.min(ccosts))
