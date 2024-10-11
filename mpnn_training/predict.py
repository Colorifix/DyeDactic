import json, subprocess

with open("hyperopt/best_param.json", "r") as f:
    params = json.load(f)

for fold in [0, 1, 2, 4]: # range(5): fold_3 is not used as it is undertrained
    predict_command = f"""python chemprop_predict \
                          --test_path data/test_natural.csv \
                          --number_of_molecules 2 \
                          --smiles_columns smiles solvent \
                          --checkpoint_path hyperopt/best_params/fold_{fold}/model_0/model.pt \
                          --preds_path data/preds_natural_fold{fold}_{params["id"]}.csv"""


    process = subprocess.Popen(predict_command.split(),
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               )

    stdout, stderr = process.communicate()


