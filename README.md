# Implementation Code for IcilDICE (under review)

- This code contains official implementation codes of IcilDICE and baselines for offline inverse constrained imitation learning.

### 1. Prerequisites

- To run this code, first install the anaconda virtual environment and install D4RL:

```
conda env create -f environment.yml
conda activate icildice
```

- This implementation requires `safety-gymnasium==1.2.0`, which currently does not support PyPI installation due to package size constraints. 
To install, we should download the package file from GitHub repo and manually install from this:
```
wget https://github.com/PKU-Alignment/safety-gymnasium/archive/refs/heads/main.zip
unzip main.zip
cd safety-gymnasium-main
pip install -e .
```

- Download dataset from huggingface-hub. You need Huggingface access token that is accesible to [Dataset Repository](https://huggingface.co/datasets/kaist-sisk/offline-icil-dataset).
(TODO: should modify before submission)
```
git config --global credential.helper store
hf auth login
hf download kaist-sisk/offline-icil-dataset --repo-type dataset --local-dir ./dataset
```

### 2. Generate Subsampling Indices for Covariate Shift Scenarios
```
python generate_subsample_indices.py
```

### 3. Train & Evaluate DrilDICE
- Train imitation policies using `main.py`.

### Toy Domain:
- To run toy domain env., you should get MOSEK license. (To obtain academic licenses, see https://www.mosek.com/products/academic-licenses/ )
```
pip install cvxpy[mosek] mosek
```
- Run `cd fourrooms; python run_toy.py`
