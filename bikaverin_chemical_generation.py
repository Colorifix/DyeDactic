import doranet.modules.synthetic as synthetic
import doranet.modules.enzymatic as enzymatic
import doranet.modules.post_processing as post_processing
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Mol, CanonSmiles, MolFromSmiles
from typing import List, Tuple
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image
import json, os, io
import pandas as pd
from src.utils import generate_count_fps, generate_colours
from dataclasses import dataclass
from itertools import compress
from lightning import pytorch as pl
from rdkit import Chem
from chemprop import data, featurizers
from chemprop.featurizers import MultiHotAtomFeaturizer, SimpleMoleculeMolGraphFeaturizer
from chemprop.models import multi
from chemprop.data.datasets import MulticomponentDataset, MoleculeDataset
import scipy.stats as st
import torch
from pathermo.properties import Hf

dir_path = os.path.dirname(os.path.realpath(__file__))
cofactor_file = os.path.join(dir_path, 'data/all_cofactors.tsv')
cofactor_df = pd.read_csv(cofactor_file, sep="\t")
cofactor_smiles = cofactor_df["SMILES"].to_list()
HELPERS = set(map(lambda smi: CanonSmiles(smi, useChiral=0), cofactor_smiles))

threshold = 2.40


def predict_abs_energy(input_colourants: List[str],
                       input_solvents: List[str] = ["CO"],
                       model_path: str = os.getcwd() + "/mpnn_training/hyperopt/best_params/",
                       model_indices: Tuple[int] = (1, 2, 3, 4, 5),
                       ensemble_size: int = 4) -> Tuple[List[float], List[float]]:


    """
    A function to predict colourants' lowest transition energies using MPNN
    NN weights were adjusted using natural colourant data and Deep4Chem dataset
    1. only cpu usage currently
    2. return mean value but also can return variance from predictions
    3. SMILES should not contain explicit hydrogen atoms
    params inputs_colourants: a list of SMILES string
    params inputs_solvents: a list of SMILES string
    params model_path: a path with to root catalogue containing models
    params model_indices: indices for models to use (in the default case model with index 3 did not train properly)
    return: a list of transition energies in eV
    """

    # check if the default argument is porvided and if yes assign "CO" to every colourant
    if len(input_solvents) == 1 and input_solvents[0] == "CO":
        input_solvents = ["CO"] * len(input_colourants)

    # check if there are any unexpected indices
    for idx in model_indices:
        for ens in range(ensemble_size):
            if not os.path.exists(f"{model_path}/{idx}/model_{ens}/best.pt"):
                raise ValueError(f"Supplied model indices ({idx}/model_{ens}/best.pt) does not correspond to any of the trained models")

    # Check the validity of SMILES inputs
    for smi in input_colourants:
        if smi == '':
            raise RuntimeError(f"Colourant SMILES {smi} cannot be empty string")

        mol_from_smiles = Chem.MolFromSmiles(smi)
        if mol_from_smiles is None:
            raise RuntimeError(f"Colourant SMILES {smi} cannot be converted to Mol")

    for smi in input_solvents:
        if smi == '':
            raise RuntimeError(f"Solvent SMILES {smi} cannot be empty string")
        mol_from_smiles = Chem.MolFromSmiles(smi)
        if mol_from_smiles is None:
            raise RuntimeError(f"Solvent SMILES {smi} cannot be converted to Mol")


    # SMILES should be supplied without hydrogens
    colourant_dataset = MoleculeDataset(
        [data.MoleculeDatapoint.from_smi(smi, add_h=True) for smi in input_colourants])

    solvent_dataset = MoleculeDataset(
        [data.MoleculeDatapoint.from_smi(smi, add_h=True) for smi in input_solvents])

    multi_dataset = MulticomponentDataset(datasets=[colourant_dataset,
                                                    solvent_dataset]
                                          )

    dataloader = data.build_dataloader(multi_dataset, shuffle=False)
    all_predictions = []

    # load and predict using all models from cross-validation
    for idx in model_indices:
        for ens in range(ensemble_size):
            model = multi.MulticomponentMPNN.load_from_file(f"{model_path}/{idx}/model_{ens}/best.pt", map_location='cpu')

            with torch.inference_mode():
                trainer = pl.Trainer(
                    logger=False,
                    enable_progress_bar=False,
                    accelerator="cpu",
                    devices=1
                )

                fold_predictions = trainer.predict(model, dataloader)
                fold_predictions = torch.cat(fold_predictions, 0)

                all_predictions.append(fold_predictions)

    predictions = np.concatenate(all_predictions, axis=1)

    means = np.mean(predictions, axis = 1)
    stds = np.std(predictions, axis = 1)

    if len(model_indices) != 1:
        # use t-distribution to estimate confidence intervals
        interval_025, interval_975 = st.t.interval(0.95,
                                                   df=len(model_indices),
                                                   loc=means,
                                                   scale=stds)

        mean, confidence = 0.5 * (interval_025 + interval_975), 0.5 * (interval_975 - interval_025)
        return np.round(mean, 3), np.round(confidence, 3)

    else:
        return np.round(means, 3), np.array([0.] * len(input_colourants))



def DrawMol(mol, title, molSize=(250, 250), kekulize=True):
    mc = Chem.MolFromSmiles(mol)
    if kekulize:
        try:
            Chem.Kekulize(mc)
        except:
            mc = Chem.Mol(mol.ToBinary())
    if not mc.GetNumConformers():
        Chem.rdDepictor.Compute2DCoords(mc)

    drawer = rdMolDraw2D.MolDraw2DCairo(*molSize)
    drawer.DrawMolecule(mc, legend = title)
    drawer.FinishDrawing()
    image = drawer.GetDrawingText()
    im = Image.open(io.BytesIO(image))
    return np.array(im)


# define how to handle hovering event
def hover(event):
    """ A helper function to visualise colourant structure when hovering over a point in the scatter plot """
    global reactions, abs_energies, fps_embedded
    # if the mouse is over the scatter points
    if line.contains(event)[0]:
        # find out the index within the array from the event
        # if two points are too close to each other then take the first point
        if len(line.contains(event)[1]["ind"]) > 1:
            ind = line.contains(event)[1]["ind"][0]
        else:
            ind, = line.contains(event)[1]["ind"]
        # get the figure size
        w, h = fig.get_size_inches()*fig.dpi
        ws = (event.x > w/2.)*-1 + (event.x <= w/2.)
        hs = (event.y > h/2.)*-1 + (event.y <= h/2.)
        # if event occurs in the top or right quadrant of the figure,
        # change the annotation box position relative to mouse.
        ab.xybox = (xybox[0]*ws, xybox[1]*hs)
        # make annotation box visible
        ab.set_visible(True)

        # place it at the position of the hovered scatter point
        ab.xy = (fps_embedded[ind, 0], fps_embedded[ind, 1])

        # set the image corresponding to that point
        im.set_data(DrawMol(reactions[ind].product_of_interest, f"E_abs = {abs_energies[ind]:.2f} +/- {abs_energies_confidence[ind]:.2f}"))

    else:
        #if the mouse is not over a scatter point
        ab.set_visible(False)

    fig.canvas.draw_idle()


def rxn_dH(rxn_dict):
    """ Reaction enthalpy estimation based on pathermo """
    reactants = rxn_dict["reactants"]
    products  = rxn_dict["products"]

    reactants_H = [Hf(mol) for mol in reactants]
    products_H = [Hf(mol) for mol in products]

    if (None in reactants_H) or (None in products_H):
        return 0.
    else:
        return sum(products_H) - sum(reactants_H)

@dataclass
class Reaction:
    """ A class to parse DORAnet generation results"""
    rule: str
    smiles: str
    product_of_interest: Mol | None
    enthalpy: float


def parse_generated_reaction(line: str) -> Reaction:
    """ Parse DORAnet generated reaction string"""
    beginning, rule, enthalpy, ending = line.split('>')
    reaction_smiles = beginning + '>>' + ending
    enthalpy = float(enthalpy.split('$')[0])

    products = ending.split(".")
    longest_smiles = sorted([smi for smi in products if CanonSmiles(smi, useChiral=0) not in HELPERS],
                            key=len,
                            reverse=True)[0]

    return Reaction(rule=rule, smiles=reaction_smiles, product_of_interest=longest_smiles, enthalpy=enthalpy)



job_name = "bikaverin_modification"
bikaverin = {
    "CC1=CC(=CC2=C1C(=O)C3=C(O2)C(=O)C4=C(C3=O)C(=CC(=C4O)OC)O)OC", # slightly unstable tautomer
    "COc4cc(C)c3c(=O)c2c(O)c1c(=O)cc(OC)c(=O)c1c(O)c2oc3c4" # stable tautomer
             }


helpers = {
    "NCCO", # ethanolamine
          }


synthetic_network = synthetic.generate_network(
    job_name = job_name + "_synthetic",
    starters = bikaverin,
    helpers = helpers,
    direction = "forward",
    molecule_thermo_calculator=Hf
)


enzymatic_network = enzymatic.generate_network(
    job_name = job_name + "_enzymatic",
    starters = bikaverin,
    direction = "forward",
    allow_multiple_reactants = True,
    rxn_thermo_calculator = rxn_dH
)


post_processing.pretreat_networks(
    networks = {
        synthetic_network,
        enzymatic_network
        },
    starters = bikaverin,
    helpers = helpers,
    total_generations = 1,
    job_name = job_name + "_synthetic_enzymatic"
    )


visible_ev = np.linspace(1.63, 3.26, 163)

with open(job_name + "_synthetic_enzymatic_network_pretreated.json") as f:
    reactions = json.load(f)

reactions = [parse_generated_reaction(line) for line in reactions]

fps = np.array([np.array(generate_count_fps(rxn.product_of_interest)) for rxn in reactions])
fps_embedded = TSNE(n_components = 2, perplexity = 30).fit_transform(fps)


abs_energies, abs_energies_confidence = predict_abs_energy([rxn.product_of_interest for rxn in reactions],
                                                           ["CO"] * len(reactions))

colours = generate_colours(abs_energies)

# create figure and plot scatter
fig = plt.figure()
ax = fig.add_subplot(111)

below_threshold = abs_energies < 2.4
above_threshold = abs_energies >= 2.4

products = [rxn.product_of_interest for rxn, e in zip(reactions, abs_energies) if e < 2.4]

bikaverin_generation_output_smi = ""
for i, product in enumerate(products):
    bikaverin_generation_output_smi += f"{product} mol{i}\n"

with open("bikaverin_generation_output.smi", "w") as f:
    f.write(bikaverin_generation_output_smi)

line = ax.scatter(fps_embedded[:, 0], fps_embedded[:, 1],
                  c=colours, edgecolors='black',
                  linewidth=3,
                  ls="", s=40,
                  marker="o",
                  alpha=1.0)

line2 = ax.scatter(fps_embedded[below_threshold, 0], fps_embedded[below_threshold, 1],
                  c=list(compress(colours, below_threshold)), edgecolors='black',
                  linewidth=3, ls="",
                  marker="s", s=60,
                  alpha=1.0)

ax.set_xlabel("TSNE component 1", fontsize = 24)
ax.set_ylabel("TSNE component 2", fontsize = 24)


# create the annotations box
im = OffsetImage(np.zeros((100, 100)), zoom=1)
xybox=(100., 100.)
ab = AnnotationBbox(im, (0,0), xybox=xybox, xycoords='data',
        boxcoords="offset points", pad=0.3, arrowprops=dict(arrowstyle="->"))

# add it to the axes and make it invisible
ax.add_artist(ab)
ab.set_visible(False)

# garbage cleaning
os.remove("*pgnet")
os.remove("*json")

# add callback for mouse moves
fig.canvas.mpl_connect('motion_notify_event', hover)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()

