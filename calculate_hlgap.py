import os
from rdkit import Chem
from src.optimize_xtb import run_xtb_energy
import pandas as pd


# This script takes optimised coordinates on the xtb level and
# does single point calculation yielding HOMO-LUMO gaps which will be further visualised

data = pd.read_csv("data/pigments.csv")
data_dirname = "inputs"

hlgap_list = []

# then create a set of directories with names from molecular titles
for row in data.iterrows():

    name, smiles, solvent = row[1]["name"], row[1]["smiles"], row[1]["solvent"]

    # a path to xyz file with optimized geometry
    path2xyz = os.path.join(data_dirname, name, name + "_xtb.xyz")

    # get total charge from SMILES
    mol = Chem.MolFromSmiles(smiles)
    total_charge = sum([atom.GetFormalCharge() for atom in mol.GetAtoms()])
    # check if solvent is in a list available solvents otherwise use water
    if solvent not in ["Acetone", "Acetonitrile", "Aniline", "Benzaldehyde", "Benzene", "CH2Cl2", "CHCl3",
                       "CS2", "Dioxane", "DMF", "DMSO", "Ether", "Ethylacetate", "Furane", "Hexadecane",
                       "Hexane", "Methanol", "Nitromethane", "Octanol", "Phenol", "Toluene", "THF", "Water"]:
        solvent = "h2o"

    _, hlgap = run_xtb_energy(path2xyz, solvent = solvent, charge = total_charge)
    hlgap_list.append(hlgap)

data["hlgap"] = hlgap_list
data.to_csv("data/pigments_with_hlgap.csv")
