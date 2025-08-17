# Implementation Code for IcilDICE (under review)

- This code contains official implementation codes of IcilDICE and baselines for offline inverse constrained imitation learning.

### 1. Prerequisites

- To run this code, first install the anaconda virtual environment and install D4RL:

```
conda env create -f environment.yml
conda activate drildice
pip install d4rl
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
