import sys, os
from timeit import default_timer as timer
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.rdchem import Mol
from src.utils import DrawTautsFromList
from typing import Type, List
from src.optimize_xtb import get_lowest_energies_xtb, pyscf_energy
import numpy as np
from loguru import logger

logger.add("optimization.log", format="{time} {level} {message}",
           filter="my_module", level="INFO", rotation="10 MB")

SCRIPT_DIR = os.path.dirname(os.path.abspath("/Users/dkarlov/aq_indigo_reproduction/src"))
sys.path.append(os.path.dirname(SCRIPT_DIR))


def sp2_filtering_rule_match(taut: Type[Mol], input_mol: Type[Mol]) -> bool:
    # Check if we have the same ring sp2 carbon atom set after tautomeric transformations
    sp2_carbon = Chem.MolFromSmarts('[$([cX3](:*):*),$([cX2+](:*):*),$([CX3]=*),$([CX2+]=*)]')
    return set(taut.GetSubstructMatches(sp2_carbon)) == set(input_mol.GetSubstructMatches(sp2_carbon))

def sp2_filtering_rule_subset(taut: Type[Mol], input_mol: Type[Mol]) -> bool:
    # Check if we have the same ring sp2 carbon atom set after tautomeric transformations with additional sp2 carbons
    sp2_carbon_in_a_ring = Chem.MolFromSmarts('[$([cX3](:*):*),$([cX2+](:*):*),$([CX3]=*),$([CX2+]=*)!R0]')
    return set(input_mol.GetSubstructMatches(sp2_carbon_in_a_ring)).issubset(set(taut.GetSubstructMatches(sp2_carbon_in_a_ring)))

def get_max_sp2_path_length(mol: Type[Mol]) -> int:
    """
    Really boring check of the length of sp2 carbon substructure length
    start path length equals to number of sp2 carbon atoms
    then on every iteration we decrease the length of possible unsaturated substructure
    :param mol: rdkit Mol object instance
    :return length of a chain composed of unsaturated carbons
    """

    smarts_sp2_carbon = '[$([cX3](:*):*),$([cX2+](:*):*),$([CX3]=*),$([CX2+]=*)]'
    num_c = len(mol.GetSubstructMatches(Chem.MolFromSmarts(smarts_sp2_carbon)))
    # if molecule has no o just a single double bond return 0, 1, 2
    if num_c in [0, 1, 2]:
        return num_c

    not_found = True
    while not_found:
        # check if we have zero carbon atoms is a dangerous symptom that something goes utterly wrong
        if num_c == 0:
            logger.warning(f"{Chem.MolToSmiles(mol)}: {str(result.stderr)}")
            return num_c
        # combine a SMARTS string with number of carbon atoms equal to current length
        current_smarts = "~".join(num_c * [smarts_sp2_carbon])
        current_unsaturated_path = Chem.MolFromSmarts(current_smarts)
        # check if the molecule contain unsturated string of the current length
        if mol.HasSubstructMatch(current_unsaturated_path):
            not_found = False
        # decrease length by 1 and repeat
        else:
            num_c -= 1
    return num_c

def enumerate_tautomers(mol: Type[Mol],
                        solvent = "h2o",
                        filter_sp2: bool = True,
                        use_pyscf: bool = False) -> List[Type[Mol]]:
    """
    Enumerate tautomers using rdkit rules and rank them based on xtb energy
    The following euristics are used to avoid energy calculation of all possible tautomers (a huge number!):
    1. do not consider tautomers with sp2 -> sp3 change in rings but sp2 atoms can appear in linear segments
    2. to run energy calculation on a tautomer satisfied the above condition the tautomer has to have longer pi-system

    :param mol: rdkit Mol object instance
    :param solvent: solvent name taken from available options for ALPB xtb model
    :param filter_sp2: whether filter a generated set of molecules using predefined filtering rules

    :return length of a chain composed of unsaturated carbons
    """
    tautomer_params = rdMolStandardize.CleanupParameters()
    tautomer_params.tautomerRemoveSp3Stereo = False
    tautomer_params.tautomerRemoveBondStereo = False
    tautomer_params.tautomerRemoveIsotopicHs = False
    tautomer_params.maxTautomers = 10000
    tautomer_params.maxTransforms = 200000

    enumerator = rdMolStandardize.TautomerEnumerator(tautomer_params)
    tautomers = []
    min_energy = 0.
    min_dft_energy = 0.
    largest_mol_sp2_substructure = get_max_sp2_path_length(mol)

    for taut in enumerator.Enumerate(mol):
        if filter_sp2:
            # first check if we have all sp2 carbons in place
            if sp2_filtering_rule_match(taut, mol):
                energy, conf = get_lowest_energies_xtb(taut, solvent = solvent)
                if energy < min_energy:
                    min_energy = energy
                taut.SetProp("energy", str(energy))

                # in case a comparison with ab initio calculations are required
                if use_pyscf:
                    dft_energy = pyscf_energy(conf)
                    if dft_energy < min_dft_energy:
                        min_dft_energy = dft_energy
                    taut.SetProp("dft_energy", str(dft_energy))

                tautomers.append(taut)

            # if they are not in place check if the transformation extends the pi-system in the tautomer and the sp2 carbons in rings are intact
            elif sp2_filtering_rule_subset(taut, mol) and (get_max_sp2_path_length(taut) >= 1.3 * largest_mol_sp2_substructure):
                energy, conf = get_lowest_energies_xtb(taut, solvent = solvent)
                if energy < min_energy:
                    min_energy = energy
                taut.SetProp("energy", str(energy))

                # in case a comparison with ab initio calculations are required
                if use_pyscf:
                    dft_energy = pyscf_energy(conf)
                    if dft_energy < min_dft_energy:
                        min_dft_energy = dft_energy
                    taut.SetProp("dft_energy", str(dft_energy))

                tautomers.append(taut)
        #estimate energies for all possible tautomers which we do not want by default
        else:
            energy, conf = get_lowest_energies_xtb(taut, solvent=solvent)
            if energy < min_energy:
                min_energy = energy
            taut.SetProp("energy", str(energy))

            # in case a comparison with ab initio calculations are required
            if use_pyscf:
                dft_energy = pyscf_energy(conf)
                if dft_energy < min_dft_energy:
                    min_dft_energy = dft_energy
                taut.SetProp("dft_energy", str(dft_energy))

            tautomers.append(taut)

    # convert energies to kcal/mol
    for taut in tautomers:
        taut.SetProp("energy", str(np.round((float(taut.GetProp("energy")) - min_energy) * 627.5, decimals=1)))
        if use_pyscf:
            taut.SetProp("dft_energy", str(np.round((float(taut.GetProp("dft_energy")) - min_dft_energy) * 627.5, decimals=1)))

    # sort tautomers energy by XTB values
    tautomers = sorted(tautomers, key = lambda mol: float(mol.GetProp("energy")))

    return tautomers


if __name__ == "__main__":

    # Two observable tautomers of curcumin: enol curcumin is more stable and this is reproduced
    # but keto-curcumin cannot be generated using the defined euristics
    keto_curcumin = Chem.MolFromSmiles('O=C(\\C=C\\c1ccc(O)c(OC)c1)CC(=O)C=Cc2cc(OC)c(O)cc2')
    enol_curcumin = Chem.MolFromSmiles('O=C(/C=C(/C=C/c1ccc(O)c(OC)c1)O)/C=C/c2cc(OC)c(O)cc2')
    # several larger colourants
    resistomycin = Chem.MolFromSmiles('CC1=CC(=O)C2=C(C3=C4C5=C2C1=C(C=C5C(C(=C4C(=O)C=C3O)O)(C)C)O)O')
    prodigiosin = Chem.MolFromSmiles('CCCCCC1=C(NC(=C1)/C=C\\2/C(=CC(=N2)C3=CC=CN3)OC)C')
    bikaverin = Chem.MolFromSmiles('COc4cc(C)c3c(=O)c2c(O)c1c(=O)cc(OC)c(=O)c1c(O)c2oc3c4')

    mols = {'prodigiosin': prodigiosin,
            'keto_curcumin': keto_curcumin,
            'enol_curcumin': enol_curcumin,
            'resistomycin': resistomycin,
            'bikaverin': bikaverin}

    for name, mol in mols.items():
        start = timer()
        DrawTautsFromList(enumerate_tautomers(mol, use_pyscf=True), name)
        end = timer()
        logger.info(f'Elapsed time for {name}: {end - start}')
