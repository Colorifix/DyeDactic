import matplotlib.pyplot as plt
from src.ionisation import *

if __name__ == "__main__":
    """
    Equation for systematic error removal
    E_actual (in eV) = 0.397 + 0.731  * E_vert (in eV) for wB97X-D4  

    Quinalizarin  |    Calculated energies      | Energies after systematic error removal
    S1111         | 3.122 (0.42); 3.723 (0.06)  | 2.697 (0.42); 3.118 (0.06)
    S0111         | 2.600 (0.44); 3.429 (0.12)  | 2.298 (0.44); 2.903 (0.12)
    S0110         | 2.623 (0.41); 3.326 (0.05)  | 2.314 (0.41); 2.828 (0.05)
    S0101         | 2.653 (0.28); 3.419 (0.04)  | 2.328 (0.61); 2.734 (0.10)
    S0100         |               2.384 (0.69)  | 2.140 (0.69)
    S0001         | 2.532 (0.51); 3.562 (0.10)  | 2.248 (0.51); 3.000 (0.10)
    S0000         | 2.423 (0.62); 3.224 (0.07)  | 2.168 (0.62); 2.754 (0.07)
    """

    mol = Molecule("quinalizarin",
                   transition_energies=[(0, [2.697, 3.118]),  (1, [2.298, 2.903]),  (2, [2.314, 2.828]),
                                        (3, [2.328, 2.734]), (4, [2.140]), (5, [2.248, 3.000]), (6, [2.168, 2.754])],
                                                                                                         # See Scheme S3 for more info
                   protonated_species=[(6, "C1=CC(=C(C2=C1C(=O)C3=C(C=CC(=C3C2=O)O)O)O)O"),              # neutral                       (S1111)
                                       (5, "C1=CC(=C(C3=C1C(=O)C2=C(C=CC(=C2C3=O)O)O)O)[O-]"),           # 2-hydroxy deprotonated        (S0111)
                                       (4, "C1=CC(=C(C3=C1C(=O)C2=C(C=CC(=C2C3=O)O)[O-])O)[O-]"),        # 2,5-dyhydroxy deprotonated    (S0110)
                                       (3, "C1=CC(=C(C3=C1C(=O)C2=C(C=CC(=C2C3=O)[O-])O)O)[O-]"),        # 2,8-dyhydroxy deprotonated    (S0101)
                                       (2, "C1=CC(=C(C3=C1C(=O)C2=C(C=CC(=C2C3=O)[O-])[O-])O)[O-]"),     # 1,2,8-trihydroxy deprotonated (S0001)
                                       (1, "C1=CC(=C(C3=C1C(=O)C2=C(C=CC(=C2C3=O)[O-])O)[O-])[O-]"),     # 2,5,8-trihydroxy deprotonated (S0100)
                                       (0, "C1=CC(=C(C2=C1C(=O)C3=C(C=CC(=C3C2=O)[O-])[O-])[O-])[O-]")], # fully deprotonated            (S0000)

                   # trihydroxy deprotonated
                   oscillator_strengths=[(0, [0.42, 0.06]),  (1, [0.44, 0.12]),  (2, [0.41, 0.05]),
                                         (3, [0.61, 0.10]), (4, [0.69]), (5, [0.51, 0.1]), (6, [0.62, 0.07])],
                   pKa=[(0, 7.49), (1, 9.56), (2, 11.29), (3, 12.32)],
                   vg_osc_str = {"S0000": [0.8521, 0.00003, 0.0760, 0.00009, 0.0835, 0.00034, 0.0004, 0.0950, 0.02436, 0.01914],
                                 "S0001": [0.6955, 0.0004, 0.1769, 0.00005, 0.0323, 0.00014, 0.1492, 0.2249, 0.0003, 0.0008],
                                 "S0100": [0.8726, 0.0200, 0.0092, 0.0461, 0.0519, 0.0061, 0.0017, 0.0062, 0.0962, 0.0220],
                                 "S0101": [0.8595, 0.1272, 0.0003, 0.0254, 0.00012, 0.0853, 0.00005, 0.1493, 0.0002, 0.1297],
                                 "S0110": [0.5767, 0.0999, 0.0063, 0.0035, 0.00005, 0.5763, 0.0010, 0.0475, 0.0022, 0.0907],
                                 "S0111": [0.6288, 0.1948, 0.0432, 0.00008, 0.4442, 0.00123, 0.000008, 0.1387, 0.4270, 0.2815],
                                 "S1111": [0.6088, 0.0892, 0.00008, 0.0163, 0.00023, 0.1716, 0.2428, 0.6911, 0.0037, 0.5414]}

                   )

    mol.read_vibronic_spectra_from_files("quinalizarin")
    mol.epsilon_from_osc_strength()
    mol.quinalizarin_protonated_species()
    mol.generate_colour_vs_pH()
    mol.generate_colour_vs_pH_from_vibronic("quinalizarin")
    mol.visualize_species_distribution(vibronic=True)
