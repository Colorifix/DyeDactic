import matplotlib.pyplot as plt
from src.ionisation import *

if __name__ == "__main__":
    """
    Equation for systematic error removal
    E_actual (in eV) = 0.397 + 0.731 * E_vert (in eV) for wB97X-D4 

    Bikaverin   |  After systematic error removal
    S11         |  2.55 (0.37); 3.11 (0.05)
    S10         |  2.20 (0.39); 2.92 (0.08)
    S01         |  2.00 (0.33); 3.08 (0.05)
    S00         |  1.98 (0.37); 2.84 (0.06)

    """
    mol = Molecule("bikaverin",
                   transition_energies = [(3, [2.55, 3.11]), (2, [2.20, 2.92]),
                                          (1, [2.00, 3.08]), (0, [1.98, 2.84])],
                                                                                                        
                   protonated_species = [(3, "COc4cc(C)c3c(=O)c2c(O)c1c(=O)cc(OC)c(=O)c1c(O)c2oc3c4"),        # neutral                                     (S11)
                                         (2, "COc4cc(C)c3c(=O)c2c(O)c1c(=O)cc(OC)c(=O)c1c([O-])c2oc3c4"),     # most acidic hydroxyl deprotonated           (S01)
                                         (1, "COc4cc(C)c3c(=O)c2c([O-])c1c(=O)cc(OC)c(=O)c1c(O)c2oc3c4"),     # hydroxyl between two carbonyls deprotonated (S10)
                                         (0, "COc4cc(C)c3c(=O)c2c([O-])c1c(=O)cc(OC)c(=O)c1c([O-])c2oc3c4")], # both hydroxyle deprotonated                  (S00)

                   oscillator_strengths = [(3, [0.37, 0.05]), (2, [0.39, 0.08]),
                                           (1, [0.33, 0.05]), (0, [0.37, 0.06])],
                   pKa = [(0, 8.25), (1, 10.16)],
                   vg_osc_str = {}
                   )

  
    mol.epsilon_from_osc_strength()
    mol.bikaverin_protonated_species()
    mol.generate_colour_vs_pH()

    mol.visualize_species_distribution()
    
    """
    ethanolaminobikaverin   |  After systematic error removal
    S11                     |  2.66 (0.13); 3.05 (0.27)
    S10                     |  2.18 (0.36); 3.02 (0.03)
    S01                     |  1.85 (0.31); 2.98 (0.03)

    """
    mol = Molecule("ethanolaminobikaverin",
                   transition_energies = [(2, [2.66, 3.05]),
                                          (1, [2.18, 3.02]),
                                          (0, [1.85, 2.98])],
                                                                                                        
                   protonated_species = [(2, "COc4cc(C)c3c(=O)c2c(O)c1c(=O)cc(OC)c(=O)c1c([NH2+]CCO)c2oc3c4"),  # positive aminogroup   (S11)
                                         (1, "COC1=CC(=O)c2c(c(NCCO)c3oc4cc(OC)cc(C)c4c(=O)c3c2O)C1=O"),        # neutral               (S01)
                                         (0, "COc4cc(C)c3c(=O)c2c([O-])c1c(=O)cc(OC)c(=O)c1c(NCCO)c2oc3c4")],   # monodeprotonated      (S00)


                   oscillator_strengths = [(2, [0.13, 0.27]),
                                           (1, [0.36, 0.03]),
                                           (0, [0.31, 0.03])],

                   pKa = [(0, 2.69), (1, 9.19)],
                   vg_osc_str = {}
                   )

    mol.epsilon_from_osc_strength()
    mol.ethanolaminobikaverin_protonated_species()
    mol.generate_colour_vs_pH()

    mol.visualize_species_distribution()
    




