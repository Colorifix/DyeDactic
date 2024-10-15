## DyeDactic 
--- -
[//]: (Badges)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<!---
Link to Zenodo for  storage ![DOI](https://zenodo.org/badge/???????.svg)](https://zenodo.org/badge/latestdoi/????????) 
-->

A colour prediction workflow for biosynthetically produced dyes and pigments.  
A code repository to reproduce the results published by [Karlov et al.]()

### Command line set up
- [Poetry](https://python-poetry.org/docs/#installing-with-the-official-installer) is required to run the package I used version (1.8.3)
- Clone the repo `git clone git@github.com:dkarlov/dydactic.git`
- Download a release of XTB executable (6.6.1 was tested) and make sure executable is in your $PATH
- Make sure you are in the root directory `cd dyedactic`
- Please run `poetry install` to install dependencies
- `bash get_nn_weights.sh` will download the Chemprop model files from zenodo 
- Then any script can be launched using `poetry run /path/to/script.py`

### Description of package
```
.
├── README.md
├── __init__.py
├── calculate_hlgap.py
├── data
│   ├── orbital_energies.csv
│   ├── pigments.csv
│   └── tddft_results.csv
├── draw_pie_diagram.py
├── figs
│   ├── b2plyp.png
│   ├── b2plyp_ext.png
│   ├── bmk.png
│   ├── bmk_ext.png
│   ├── camb3lyp.png
│   ├── camb3lyp_ext.png
│   ├── m062x.png
│   ├── m062x_ext.png
│   ├── pbe0.png
│   ├── pbe0_ext.png
│   ├── pbeqihd.png
│   ├── pbeqihd_ext.png
│   ├── wb97xd.png
│   ├── wb97xd4_tda.png
│   ├── wb97xd_ext.png
│   ├── wpbepp86.png
│   ├── wpbepp86_ext.png
│   └── wpbepp86_main_text.png
├── generate_inputs.py
├── inputs
│   ├── 1,3-dihydroxy-2-methoxyanthraquinone
│   │   ├── 1,3-dihydroxy-2-methoxyanthraquinone_mm.xyz
│   │   ├── 1,3-dihydroxy-2-methoxyanthraquinone_xtb.xyz
│   │   ├── b2plyp.inp
│   │   ├── b2plyp_nosolv.inp
│   │   ├── bmk.inp
│   │   ├── bmk_nosolv.inp
│   │   ├── camb3lyp.inp
│   │   ├── camb3lyp_nosolv.inp
│   │   ├── m062x.inp
│   │   ├── m062x_nosolv.inp
│   │   ├── opt_pbe0.inp
│   │   ├── pbe0_nosolv.inp
│   │   ├── pbeqidh.inp
│   │   ├── pbeqidh_nosolv.inp
│   │   ├── wb97xd4.inp
│   │   ├── wb97xd4_nosolv.inp
│   │   ├── wb97xd4_polar.inp
│   │   ├── wb97xd4_polar_nosolv.inp
│   │   ├── wb97xd4_tda.inp
│   │   ├── wb97xd4_tda_nosolv.inp
│   │   ├── wpbepp86.inp
│   │   └── wpbepp86_nosolv.inp
... ...
│   └── xanthopurpurin
│       ├── b2plyp.inp
│       ├── b2plyp_nosolv.inp
│       ├── bmk.inp
│       ├── bmk_nosolv.inp
│       ├── camb3lyp.inp
│       ├── camb3lyp_nosolv.inp
│       ├── m062x.inp
│       ├── m062x_nosolv.inp
│       ├── opt_pbe0.inp
│       ├── pbe0_nosolv.inp
│       ├── pbeqidh.inp
│       ├── pbeqidh_nosolv.inp
│       ├── wb97xd4.inp
│       ├── wb97xd4_nosolv.inp
│       ├── wb97xd4_polar.inp
│       ├── wb97xd4_polar_nosolv.inp
│       ├── wb97xd4_tda.inp
│       ├── wb97xd4_tda_nosolv.inp
│       ├── wpbepp86.inp
│       ├── wpbepp86_nosolv.inp
│       ├── xanthopurpurin_mm.xyz
│       └── xanthopurpurin_xtb.xyz
├── logs
├── mpnn_training
│   ├── chemprop_hyperopt.py
│   ├── data
│   │   ├── 20210205_all_expt_data_no_duplicates_solvent_calcs.csv
│   │   ├── pigments_wb97xd4_tda_solv.csv
│   │   ├── preds_artificial_f8f937d8-e7f3-4073-a91c-e50fa78313d7.csv
│   │   ├── preds_natural_f8f937d8-e7f3-4073-a91c-e50fa78313d7.csv
│   │   ├── preds_natural_fold0_f8f937d8-e7f3-4073-a91c-e50fa78313d7.csv
│   │   ├── preds_natural_fold1_f8f937d8-e7f3-4073-a91c-e50fa78313d7.csv
│   │   ├── preds_natural_fold2_f8f937d8-e7f3-4073-a91c-e50fa78313d7.csv
│   │   ├── preds_natural_fold3_f8f937d8-e7f3-4073-a91c-e50fa78313d7.csv
│   │   ├── preds_natural_fold4_f8f937d8-e7f3-4073-a91c-e50fa78313d7.csv
│   │   ├── test_artificial.csv
│   │   ├── test_natural.csv
│   │   └── train_all.csv
│   ├── hyperopt
│   │   ├── best_param.json
│   │   └── best_params
│   │       ├── fold_0
│   │       │   ├── model_0
│   │       │   │   └── model.pt
│   │       │   └── test_scores.json
│   │       ├── fold_1
│   │       │   ├── model_0
│   │       │   │   └── model.pt
│   │       │   └── test_scores.json
│   │       ├── fold_2
│   │       │   ├── model_0
│   │       │   │   ├── events.out.tfevents.1719472128.ampere001.int.ada.nottingham.ac.uk
│   │       │   │   └── model.pt
│   │       │   └── test_scores.json
│   │       ├── fold_3
│   │       │   ├── model_0
│   │       │   │   ├── events.out.tfevents.1719504215.ampere001.int.ada.nottingham.ac.uk
│   │       │   │   └── model.pt
│   │       │   └── test_scores.json
│   │       ├── fold_4
│   │       │   ├── model_0
│   │       │   │   ├── events.out.tfevents.1719536571.ampere001.int.ada.nottingham.ac.uk
│   │       │   │   └── model.pt
│   │       │   └── test_scores.json
│   │       └── verbose.log
│   ├── predict.py
│   ├── prepare_dataset.ipynb
│   └── visualize_training.py
├── poetry.lock
├── pyproject.toml
├── src
│   ├── __init__.py
│   ├── convert_spectrum_to_colour.py
│   ├── optimize_xtb.py
│   ├── orca_inputs.py
│   ├── spectrum2colour
│   │   ├── AQ1_exp.csv
│   │   ├── AQ2_exp.csv
│   │   ├── AQ3_exp.csv
│   │   ├── AQ4_exp.csv
│   │   ├── AQ5_exp.csv
│   │   ├── AQ6_exp.csv
│   │   ├── AQ7_exp.csv
│   │   ├── AQ8_exp.csv
│   │   ├── IN10_exp.csv
│   │   ├── IN1_exp.csv
│   │   ├── __pycache__
│   │   └── biliverdin_exp.csv
│   ├── tautomers.py
│   └── utils.py
├── tddft_energy_correlation_SI_figures.py
├── tddft_energy_correlation_in_groups.py
├── tddft_energy_correlation_manuscript_figures.py
├── tddft_fosc_SI_figures.py
├── tddft_fosc_in_groups.py
├── tsne_natural_visualise.py
└── visualize_hlgap.py
```
### Description of important files
- `tests/spectrum2colour` several experimental absorption spectrum of several anthraquinone, indigo, and biliverdin derivatives
- ``
- `src/` directory contains all functions and classes from 3D structure generation to colour estimation. 
- `src/convert_spectrum_to_colour.py` contains functions to convert absorption spectra to RGB colours together with a test run
- `src/optimize_xtb.py` has wrapping functions to use XTB external program to optimise geometry, calculate energies, and HOMO-LUMO gaps  
- `src/orca_inputs.py` a class for generating ORCA (5.0.3) input files to optimise geometry and do TD-DFT calculations 
- `src/tauromers.py` tautomer enumeration functions which use euristics to prune tautomer generation tree and estimate energies using XTB energies
- `src/utils.py` miscellaneous helper functions
- `data/` contains the collected data set of natural colourants, calculated QM descriptors, TD-DFT results, and experimental absorption spectra 
- `data/pigments.csv` a csv file containing the database of collected natural compounds with experimental data and references
- `data/tddft_results.csv` a csv file containing TD-DFT calculation results including electron transition energy and oscillator strength parsed of ORCA OUTPUTS
- `data/orbital_energies.csv` a csv file containing quantum chemical descriptors (HOMO, LUMO energies, dipole moments, etc) to train a composite QC/ML model
- `data/pigment_pH_SI.csv` contains experimental absorption spectra for 4 colourants explored in the paper (*emodin*, *quinalizarin*, *biliverdin*, and *orcein*) at different pH levels
- `figs/` a folder for Figures generated by scripts in the package
- `inputs/` generated input files for ORCA calculations
- `mpnn_training/` a specified folder for chemprop based neural network model to predict absorption lowest light absorption energies
- `mpnn_training/data/` a directory for raw and clean training data 
- `mpnn_training/data/` a directory for raw and clean training data 
- `mpnn_training/data/20210205_all_expt_data_no_duplicates_solvent_calcs.csv` a training set provided by Greenman et al.
- `mpnn_training/data/pigments_wb97xd4_tda_solv.csv` a csv file with calculated transition energies (wB97XD4) for outlier deletion and solvents in SMILES format 
- `mpnn_training/data/train_all.csv` a csv file for MPNN training data (90% of natural and 90% of artificial colourants together after split) set with transitions and solvent
- `mpnn_training/data/test_artificial.csv` a validation set consisted of the artificial colourants (10% of the initial set) to select the best model parameters
- `mpnn_training/data/test_natural.csv` a natural colourant test set (10% of the collected set) solely to estimate prediction error
- `mpnn_training/data/preds_artificial_....csv` - prediction results for the validation set using a single MPNN trained on the best found parameter set
- `mpnn_training/data/preds_natural_....csv`- prediction results for the natural colourant set using a single MPNN trained on the best found parameter set
- `mpnn_training/data/preds_natural_fold?_....csv`- prediction results for the natural colourant set using an enemble of models trained during 5-fold CV
- `mpnn_training/hyperopt` - parameters and NN weights
- `mpnn_training/chemprop_hyperopt.py` a script to run hyperparameter optimisation (training is done using GPU)
- `mpnn_training/predict.py` - a prediction script which runs with test_natural.csv by default
- `mpnn_training/visualize_training.py` - prepare a Figure visualising training process prediction performance for paper
- `mpnn_training/prepare_dataset.ipynb` - prepare a train/test spilt for MPNN training and clean the initial data from outliers
- `tests` csv files for colour prediction test
- 