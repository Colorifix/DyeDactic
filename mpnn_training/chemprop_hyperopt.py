# A hyperparameter optimisation of MPNN model using approach
# from Greenman et al. https://doi.org/10.1039/D1SC05677H
# original training data was cleaned and enriched with natural colourants data
# training goes on GPU the chemprop must be configured accordingly

import numpy as np
from hyperopt import hp, fmin, tpe, Trials, space_eval, STATUS_OK
import json, os, subprocess, uuid

SAVE_DIR = "hyperopt"

def objective_fun(params):
    param_id = str(uuid.uuid4())
    params["id"] = param_id

    # make sure max learning rate is bigger than init_lr and final_lr
    params['init_lr'] = params['init_lr'] * params['max_lr']
    params['final_lr'] = params['final_lr'] * params['max_lr']

    trial_directory = os.path.join(SAVE_DIR, param_id)
    hyperopt_config_dir = os.path.join(trial_directory, "hyperopt.json")
    if not os.path.exists(trial_directory):
        os.makedirs(trial_directory)

    # keep the parameters saved
    with open(hyperopt_config_dir, "w") as outfile:
        json.dump(params, outfile)

    run_command = f"""python chemprop_train  \
                     --data_path data/train_all.csv \
                     --smiles_columns smiles solvent \
                     --dataset_type regression \
                     --target_columns peakwavs_max \
                     --loss_function mse \
                     --separate_val_path data/test_artificial.csv \
                     --separate_test_path data/test_natural.csv \
                     --seed 123 \
                     --pytorch_seed 42 \
                     --metric mae \
                     --extra_metrics rmse \
                     --cache_cutoff inf \
                     --save_dir {trial_directory} \
                     --batch_size {params["batch_size"]} \
                     --hidden_size {params["hidden_size"]} \
                     --activation {params["activation"]} \
                     --aggregation {params["aggregation"]} \
                     --depth {params["depth"]} \
                     --dropout {params["dropout"]} \
                     --ffn_num_layers {params["ffn_num_layers"]} \
                     --ffn_hidden_size {params["ffn_hidden_size"]} \
                     --warmup_epochs {params["warmup_epochs"]} \
                     --init_lr {params["init_lr"]} \
                     --max_lr {params["max_lr"]} \
                     --final_lr {params["final_lr"]} \
                     --adding_h \
                     --number_of_molecules 2 \
                     --epochs 30 \
                     --gpu 0 \
                     {params["bias"]} \
                     {params["atom_messages"]} \
                     --ensemble_size 1"""

    process = subprocess.Popen(run_command.split(),
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               )
    
    stdout, stderr = process.communicate()

    results_file = os.path.join(trial_directory, "verbose.log")
    with open(results_file, 'r') as f:
        results = f.readlines()
    final_val_mae = np.mean([float(y.split()[6]) for y in [x for x in results if 'best validation mae' in x]])

    return {'loss': final_val_mae, 'status': STATUS_OK}


param_space = {
    'bias': hp.choice('bias', ['--bias', '']),
    'hidden_size': hp.choice('hidden_size', range(100, 420, 20)),
    'activation': hp.choice('activation', ['ReLU', 'LeakyReLU', 'PReLU', 'tanh', 'SELU', 'ELU']),
    'aggregation': hp.choice('aggregation', ['mean', 'sum', 'norm']),
    'depth': hp.choice('depth', [3, 4, 5, 6]),
    'atom_messages': hp.choice('atom_messages', ['--atom_messages', '']),
    'dropout': hp.uniform('dropout', 0.0, 0.4),
    'ffn_num_layers': hp.choice('ffn_num_layers', [1, 2, 3]),
    'ffn_hidden_size': hp.choice('ffn_hidden_size', range(100, 420, 20)),
    'warmup_epochs': hp.choice('warmup_epochs', [2, 3, 4, 5, 6]),
    'batch_size': hp.choice('batch_size', range(10, 110, 20)),
    'init_lr': hp.loguniform('init_lr', -3, -1),
    'max_lr': hp.loguniform('max_lr', -5, -2),
    'final_lr': hp.loguniform('final_lr', -3, -1)}

trials = Trials()

best_params = fmin(
    fn=objective_fun,
    space=param_space,
    algo=tpe.suggest,
    max_evals=200,
    trials=trials)

with open(best_params, "w") as outfile:
    json.dump(params, outfile)
