# ECNCECE

## Setup
1. Clone the repository
```bash
git clone https://github.com/ECNCECE-anonymous/ECNCECE.git
cd ECNCECE
```

2. Create and activate a virtual environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate
```
  (On windows: venv\Scripts\activate)

3. Install dependencies
```bash
pip install -r requirements.txt
```

## Evolution Parameters
NEAT evolution parameters (including RL hyperparameter mutations) are defined in the NEAT configuration file: neat_config_exp_v2

## Running an Evolution
> **Note:** Evolution runs can take a long time.  
> The default setup uses a population of 150 individuals, each trained with reinforcement learning for 100,000 timesteps over 400 generations.

To run a basic evolutionary experiment:
```bash
python evolve.py
```
To view all available command-line options:
```bash
python evolve.py -h
```
Example with Custom Arguments:
```bash
python evolve.py --n_gens 10 --n_seasons 3 --seed 42 --EC --n_procs 4
```

## Running with Weights & Biases (wandb) Tracking
To run an evolution and log metrics to wandb:
```bash
python wandb_exp.py
```
To view available command-line options:
```bash
python wandb_exp.py -h
```
Example with wandb Project and Run ID:
```bash
python wandb_exp.py --project ECNCECE --run_id exp3_s42_EC --n_gens 20 --n_seasons 4 --EC
```
