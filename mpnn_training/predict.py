import numpy as np
import json, os, subprocess

ensemble_size = 4
n_folds = 5
best_params = "./hyperopt/best_params"

# model training - uncomment to retrain on gpu
'''

if not os.path.exists(best_params):
    os.makedirs(best_params)

with open("hyperopt/best_param.json", "r") as f:
    params = json.load(f)

params['init-lr'] = params['init-lr'] * params['max-lr']
params['final-lr'] = params['final-lr'] * params['max-lr']

for fold in range(n_folds):
    if not os.path.exists(best_params + f"/{fold+1}"):
        os.makedirs(best_params + f"/{fold+1}")):
    run_command = f"""chemprop train \
                    --data-path data/data_all{fold}.csv \
                    --smiles-columns smiles solvent \
                    --task-type regression \
                    --target-columns peakwavs_max \
                    --loss-function mse \
                    --splits-column split \
                    --data-seed 123 \
                    --pytorch-seed 42 \
                    --metric mae rmse \
                    --save-dir {best_params}/{fold+1} \
                    --message-hidden-dim {params["message-hidden-dim"]} \
                    --activation {params["activation"]} \
                    --aggregation {params["aggregation"]} \
                    --depth {params["depth"]} \
                    --dropout {params["dropout"]} \
                    --ffn-num-layers {params["ffn-num-layers"]} \
                    --ffn-hidden-dim {params["ffn-hidden-dim"]} \
                    --warmup-epochs {params["warmup-epochs"]} \
                    --init-lr {params["init-lr"]} \
                    --max-lr {params["max-lr"]} \
                    --final-lr {params["final-lr"]} \
                    --add-h \
                    --accelerator gpu \
                    --devices auto \
                    --num-workers 7 \
                    --epochs 500 \
                    {params["message-bias"]} \
                    --ensemble-size {ensemble_size}"""


    process = subprocess.Popen(run_command.split(),
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               )

    stdout, stderr = process.communicate()
    print(stdout.decode('utf-8'))
    print(stderr.decode('utf-8'))
'''

# prediction
for fold in range(n_folds):
    for i in range(ensemble_size):
        predict_command = f"""chemprop predict \
                            --test-path ./data/test_natural{fold+1}.csv \
                            --smiles-columns smiles solvent \
                            --add-h \
                            --accelerator cpu \
                            --devices auto \
                            --model-path {best_params}/{fold+1}/model_{i}/best.pt \
                            --preds-path ./data/preds_natural_fold{fold+1}_model{i}.csv"""

        process = subprocess.Popen(predict_command.split(),
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   )


        stdout, stderr = process.communicate()



