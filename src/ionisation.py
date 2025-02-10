from typing import List, Tuple
from rdkit import Chem
from src.convert_spectrum_to_colour import *
from src.utils import DrawMol, gauss_spec_with_extinction
import matplotlib.pyplot as plt
from typing import Dict
from tqdm import tqdm

class Molecule:

    """
    This class is used for calculation and visualisation of spectral dependency on pH
    We do not consider species with low population (< 0.01 molar fraction at any pH)
    A set of species is selected based on ChemAxon pKa and population predictions
    """

    def __init__(self,
                 name: str,
                 oscillator_strengths: List[Tuple[int, List[float]]],
                 transition_energies: List[Tuple[int, List[float]]],
                 protonated_species: List[Tuple[int, str]],
                 pKa: List[Tuple[int, float]],
                 concentration: float = 1e-4,
                 empirical_broadening: float = 0.4,
                 pH_low: float = 1.,
                 pH_high: float = 14.,
                 num_points: int = 200
    ) -> None:

        self.name = name                                    # mol name
        self.concentration = concentration                  # a standard concentration of 50 muM but can be changed
        self.pH_low = pH_low                               # - |
        self.pH_high = pH_high                             #   | - pH range for calculation and visualisation
        self.num_points = num_points                       # - |

        self.empirical_broadening = empirical_broadening    # the default values are provided

        self.oscillator_strengths = oscillator_strengths    # oscillator strengths for ionised forms as a list
                                                            # of tuples (index of species, list of osc strengths)
        self.transition_energies = transition_energies      # transition energies for ionised forms as a list
                                                            # of tuples (index of species, list of transition energies )
        self.protonated_species = protonated_species        # protonated species SMILES as a list of tuples
                                                            # (index of species, SMILES string)
        self.pKa = pKa                                      # [(index, pKa)] - SMILES species, pKas, osc strengths,
                                                            # wavelengths the same index species share the same attributes: pKas, osc strengths, etc.
        self.species = np.array([])                         # molar fractions at different pH levels as numpy array with shape
                                                            # (number of species, number of pH vales)
        self.colours_vs_pH = []                             # list of RGB colours in hexadecimal strings with length equal
                                                            # to the number of pH values to explore
        self.colours_vs_pH_sampled = []                     # the same thing as before but from sampled spectra
        self.spectra = []                                   # the same as previous but with the number a modeled spectra
                                                            # for each pH value
        self.spectra_from_sampling = {}                     # dict of modeled absorption spectra
        self.extinction_coefficients = []                   # will be calculated from oscillator strengths

        # some checks to perform
        if not len(self.oscillator_strengths) == len(self.transition_energies) == len(self.protonated_species):
            raise RuntimeError(f"Lengths of osc strengths, transition energies, and protonated species vectors are different: " +
                            f"{len(self.oscillator_strengths)}, {len(self.transition_energies)}, {len(self.protonated_species)}")

        # make some checks about pH range
        pH_low_ok = (self.pH_low <= 14.) and (self.pH_low >= 0.)
        pH_high_ok = (self.pH_high <= 14.) and (self.pH_high >= 0.)
        pH_high_more_than_pH_low = pH_low < pH_high
        num_points_ok = self.num_points > 1
        if not (pH_low_ok and pH_high_ok and pH_high_more_than_pH_low and num_points_ok):
            raise RuntimeError(f"pH range is out of normal scope: pH_low = {self.pH_low}, pH_high = {self.pH_high}, num_points = {self.num_points}")


    def emodin_protonated_species(self) -> None:
        """
        Based on predicted constants for emodin acid/base equilibrium is the following:

                    [AH3]
                      | K1
                   [AH2(-)]
                          \ K2
             [A'H(2-)]  [A''H(2-)]
                  \ K2   / K3
                   [A(3-)]

        all_species = ([H+]**3  / K3 / K2 / K1 + [H+]**2 / K3 / K2 + [H+] / K3 + [H+] / K2 + 1)
        [AH3]      =  [H+]**3 / K3 / K2 / K1 / all_species
        [AH2(-)]   =  [H+]**2 / K3 / K2 / all_species
        [A'H(2-)]  =  [H+] / K3 / all_species
        [A''H(2-)] =  [H+] / K2 / all_species
        [A(3-)]    =  1 / all_species

        This function updates self.species numpy array - np.array(num_points, number of protonated species)

        """

        pH = np.linspace(self.pH_low, self.pH_high, self.num_points)  # pH values used in calculation
        self.species = [np.zeros(self.num_points),
                        self.pKa[2][1] - pH,
                        self.pKa[1][1] - pH,
                        self.pKa[1][1] + self.pKa[2][1] - 2 * pH,
                        self.pKa[0][1] + self.pKa[1][1] + self.pKa[2][1] - 3 * pH]
        self.species = np.concatenate([arr.reshape(1, self.num_points) for arr in self.species])
        self.species = 10 ** self.species
        self.species = self.species / np.sum(self.species, 0)



    def quinalizarin_protonated_species(self) -> None:
        """
        Based on predicted constants for quinalizarin equilibrium is the following:

                       [AH4]
                        | K1
                     [AH3(-)]
                        | K2
                     [A'H2(2-)] [A''H2(2-)]
                        | K3  / K2
                     [A'H(3-)]  [A''H(3-)]
                        | K4  / K3
                     [A(4-)]

        all_species = ([H+]**4 /K4 / K3 / K2 / K1 + [H+]**3 / K4 / K3 / K2 + [H+]**2 / K4 / K3 + [H+]**2 / K4 / K2 + [H+] / K3 + [H+] / K4 + 1)
        [AH4]       = [H+]**4 /K4 / K3 / K2 / K1 / all_species
        [AH3(-)]    = [H+]**3 / K4 / K3 / K2 / all_species
        [A'H2(2-)]  = [H+]**2 / K4 / K3 / all_species
        [A''H2(2-)] = [H+]**2 / K4 / K2 / all_species
        [A'H(3-)]   = [H+] / K3 / all_species
        [A''H(3-)]  = [H+] / K4 / all_species
        [A(3-)]     = 1 / all_species

        This function updates self.species numpy array - np.array(num_points, number of protonated species)
        """

        pH = np.linspace(self.pH_low, self.pH_high, self.num_points)  # pH values used in calculation
        self.species = [np.zeros(self.num_points),
                        self.pKa[3][1] - pH,
                        self.pKa[2][1] - pH,
                        self.pKa[3][1] + self.pKa[2][1] - 2 * pH,
                        self.pKa[3][1] + self.pKa[1][1] - 2 * pH,
                        self.pKa[3][1] + self.pKa[2][1] + self.pKa[1][1] - 3 * pH,
                        self.pKa[3][1] + self.pKa[2][1] + self.pKa[1][1] + self.pKa[0][1] - 4 * pH]
        self.species = np.concatenate([arr.reshape(1, self.num_points) for arr in self.species])
        self.species = 10 ** self.species
        self.species = self.species / np.sum(self.species, 0)


    def aminoorcein_protonated_species(self) -> None:
        """
        Based on predicted constants for aminoorcein equilibrium is the following:

                     [AH4(2+)]
                        | K1
                     [AH3(+)]
                        | K2
                      [AH2]
                        | K3
                     [A'H(-)]  [A''H(-)]
                        | K4  / K3
                      [A(2-)]

        all_species     = [H+]**4 /K4 / K3 / K2 / K1 + [H+]**3 / K4 / K3 / K2 + [H+]**2 / K4 / K3 + [H+] / K3 + [H+] / K4 + 1
        [AH4(2+)]       = [H+]**4 /K4 / K3 / K2 / K1 / all_species
        [AH3(+)]        = [H+]**3 / K4 / K3 / K2 / all_species
        [AH2]           = [H+]**2 / K4 / K3 / all_species
        [A'H(-)]        = [H+] / K3 / all_species
        [A''H(-)]       = [H+] / K4 / all_species
        [A(2-)]         = 1 / all_species

        This function updates self.species numpy array - np.array(num_points, number of protonated species)
        """

        pH = np.linspace(self.pH_low, self.pH_high, self.num_points)  # pH values used in calculation
        self.species = [np.zeros(self.num_points),
                        self.pKa[3][1] - pH,
                        self.pKa[2][1] - pH,
                        self.pKa[3][1] + self.pKa[2][1] - 2 * pH,
                        self.pKa[3][1] + self.pKa[2][1] + self.pKa[1][1] - 3 * pH,
                        self.pKa[3][1] + self.pKa[2][1] + self.pKa[1][1] + self.pKa[0][1] - 4 * pH]
        self.species = np.concatenate([arr.reshape(1, self.num_points) for arr in self.species])
        self.species = 10 ** self.species
        self.species = self.species / np.sum(self.species, 0)


    def hydroxyorcein_protonated_species(self) -> None:
        """
        Based on predicted constants for hydroxyorcein equilibrium is the following:

             [A'H4(+)]
                | K1
               [AH3]
                | K2
             [AH2(-)]
                | K3
             [AH(2-)]
                | K4
             [A(3-)]

        all_species = [H+]**4 / K4 / K3 / K2 / K1 + [H+]**3 / K4 / K3 / K2 + [H+]**2 / K4 / K3 + [H+] / K4  +  1
        [A(3-)]   =  1 / all_species
        [AH(2-)]  =  [H+] / K4 / all_species
        [AH2(-)]  =  [H+]**2 / K4 / K3 / all_species
        [AH3]     =  [H+]**3 / K4 / K3 / K2 / all_species
        [AH4(+)]  =  [H+]**4 / K4 / K3 / K2 / K1 / all_species

        This function updates self.species numpy array - np.array(num_points, number of protonated species)
        """

        pH = np.linspace(self.pH_low, self.pH_high, self.num_points)  # pH values used in calculation
        self.species = [np.zeros(self.num_points)]
        for i in reversed(range(len(self.pKa))):
            last_species = self.species[-1]
            self.species.append(self.pKa[i][1] - pH + last_species)
        self.species = np.concatenate([arr.reshape(1, self.num_points) for arr in self.species])
        self.species = 10 ** self.species
        self.species = self.species / np.sum(self.species, 0)

    def biliverdin_protonated_species(self) -> None:
        """
        Based on predicted constants for biliverdin equilibrium is the following:

              [AH5(+)]
                | K1
               [AH4]   [A'H4]
                | K2  / K1
              [AH3(-)]
                | K3
             [AH2(2-)]
                | K4
             [AH(3-)]  [AH'(3-)]
                | K5  / K4
             [A(4-)]

        all_species = [H+]**4 / K5 / K4 / K3 / K2 / K1 + [H+]**4 / K5 / K4 / K3 / K1 + [H+]**4 / K5 / K4 / K3 / K2 +
                      [H+]**3 / K5 / K4 / K3 + [H+]**2 / K5 / K4 + [H+] / K4  +  [H+] / K5  + 1
        [A(4-)]   =  1 / all_species
        [AH(3-)]  =  [H+] / K5 / all_species
        [AH'(3-)] =  [H+] / K4 / all_species
        [AH2(2-)] =  [H+]**2 / K5 / K4 / all_species
        [AH3(-)]  =  [H+]**3 / K5 / K4 / K3 / all_species
        [AH4]     =  [H+]**4 / K5 / K4 / K3 / K2 / all_species
        [A'H4]    =  [H+]**4 / K5 / K4 / K3 / K1 / all_species
        [AH5(+)]  =  [H+]**5 / K5 / K4 / K3 / K2 / K1 / all_species

        This function updates self.species numpy array - np.array(num_points, number of protonated species)
        """

        pH = np.linspace(self.pH_low, self.pH_high, self.num_points)  # pH values used in calculation
        self.species = [np.zeros(self.num_points),
                        self.pKa[4][1] - pH,
                        self.pKa[3][1] - pH,
                        self.pKa[4][1] + self.pKa[3][1] - 2 * pH,
                        self.pKa[4][1] + self.pKa[3][1] + self.pKa[2][1] - 3  * pH,
                        self.pKa[4][1] + self.pKa[3][1] + self.pKa[2][1] + self.pKa[1][1] - 4 * pH,
                        self.pKa[4][1] + self.pKa[3][1] + self.pKa[2][1] + self.pKa[0][1] - 4 * pH,
                        self.pKa[4][1] + self.pKa[3][1] + self.pKa[2][1] + self.pKa[1][1] + self.pKa[0][1] - 5 * pH]
        self.species = np.concatenate([arr.reshape(1, self.num_points) for arr in self.species])
        self.species = 10 ** self.species
        self.species = self.species / np.sum(self.species, 0)


    def epsilon_from_osc_strength(self) -> None:
        """
        Molar absorption spectrum is calculated by the following formula from  Modern Molecular Photochemistry of Organic Molecules (N. Turro, 2010):

        eps(E) = (N_A * pi) / (3 * ln(10) * h_bar * eps_0 * c * (2pi)^0.5 * H) * sum_b( osc_strength_b * exp(-(E - Eg->b)^2/2H^2))
        where
        N_A - avogadro number
        pi - pi number
        e - electron charge
        eps_0 - vacuum dielectric permittivity
        H - empirical broadening parameter
        osc_strength - oscillator strength of b electronic transition
        Eg->b - transition energy from the ground state "g" excited state "b"

        it can be converted to a numerical coefficient:
        eps(E) = 11451.73 / H  * sum_b( osc_strength_b * exp(-(E - Eg->b)^2/2H^2)) in  M^(-1)*cm^(-1)

        NB:
        H and energies come in eV
        :return:
        """

        self.extinction_coefficients = []

        for osc in self.oscillator_strengths:
            current_osc = []
            for j in osc[1]:
                current_osc.append(11451.73 / self.empirical_broadening * j)
            self.extinction_coefficients.append((osc[0], current_osc))

    def absorption_spectrum_from_file(self,
                                      filename: str,
                                      calculated_transisitons: int = 10,
                                      num_spec_points: int = 500,
                                      s: float = 0.08) -> None:
        """
        opens a pandas csv file and gets absorption data for all conformations
        each line one conformation
        10 longest absorption wavelength and the corresponding oscillator strength are expected to be in the file
        :params filename: a path to a file with absorption data
        :params calculated_transitions: number of TDDFT roots calculated
        :params num_spec_points: number of points to approximate absorption spectra
        :params s: sigma parameter of a normal distribution
        :returns: a dictionary with simulated absorption spectra for each protonated species in optical density
        """

        visible_ev = np.linspace(1.63, 3.26, num_spec_points).reshape(-1, 1)

        try:
            data = pd.read_csv(filename)
        except:
            raise Exception(f"{filename} cannot be read by pandas")

        # expected columns
        wls = [f"w{i + 1}" for i in range(calculated_transisitons)]
        foscs = [f"f{i + 1}" for i in range(calculated_transisitons)]
        expected_columns = ["spec"] + wls + foscs

        # check the column format
        if any([column not in data.columns for column in expected_columns]):
            raise Exception(f"{filename} does not have all necessary columns columns")

        species = list(set(data["spec"]))
        final_spectra = {key: np.zeros(num_spec_points) for key in species}

        for spec in species:
            current_data = data[data["spec"] == spec]
            for j in range(1, calculated_transisitons + 1):

                # reshape the data
                wl_reshaped = np.repeat(current_data[f"w{j}"].to_numpy().reshape(1, -1), repeats=num_spec_points, axis=0)
                fosc_reshaped = np.repeat(current_data[f"f{j}"].to_numpy().reshape(1, -1), repeats=num_spec_points, axis=0)
                # for j transition place gaussian and add together j transition for all species
                w_exp = np.sum(fosc_reshaped * np.exp(-4 * np.log(2) * (visible_ev - wl_reshaped)**2 / s**2), axis=1)
                final_spectra[spec] += 1 / s * np.sqrt(4 * np.log(2) / np.pi) * w_exp

            # normalise spectrum and convert to optical density
            final_spectra[spec] = 11451.73 * self.concentration / len(current_data) * final_spectra[spec]

        self.spectra_from_sampling = final_spectra



    def generate_colour_vs_pH_from_sampled_spectra_biliverdin(self,
                                                              num_spec_points: int = 500) -> None:
        """
        A special function to visualise biliverdin colour from sampled spectra
        """

        visible_ev = np.linspace(1.63, 3.26, num_spec_points)

        # make sure the order of species is correct
        initial_spectra = [self.spectra_from_sampling['S11111'],
                           self.spectra_from_sampling['S11011'],
                           self.spectra_from_sampling['S10111'],
                           self.spectra_from_sampling['S10011'],
                           self.spectra_from_sampling['S10001_coil'],
                           self.spectra_from_sampling['S00001'],
                           self.spectra_from_sampling['S10000'],
                           self.spectra_from_sampling['S00000']]

        spectra = []

        for i in range(self.species.shape[1]):
            sp = np.zeros(num_spec_points)
            fractions = list(reversed(self.species[:, i]))
            for j in range(len(fractions)):
                sp += fractions[j] * initial_spectra[j]
            spectra.append(sp)

        self.interpolated_spectra = [np.interp(np.linspace(380, 780, 81), *zip(*sorted(zip(ev2nm(visible_ev), sp))))
                                for sp in spectra]
        self.colours_vs_pH_sampled = [Spectrum(np.linspace(380, 780, 81), sp).rgb_to_hex() for sp in self.interpolated_spectra]

    def generate_colour_vs_pH(self) -> None:
        """
        Given transition energies, ext. coefficients nad species distributions generate RGB colour vs pH
        """
        visible_ev = np.linspace(1.63, 3.26, 163)

        # currently use gaussian approximation of spectrum
        # takes each species and add gaussian function placed on the transition energy
        spectra = [gauss_spec_with_extinction([i[1] for i in self.transition_energies],        # transition energies here for each species
                                              list(reversed(self.species[:, i])),              # molar fractions of species at a specified pH
                                              [i[1] for i in self.extinction_coefficients],    # extinction coefficients for each species
                                              self.empirical_broadening, self.concentration) for i in range(self.species.shape[1])]

        spectra = [np.interp(np.linspace(380, 780, 81), *zip(*sorted(zip(ev2nm(visible_ev), sp)))) for sp in spectra]
        self.colours_vs_pH = [Spectrum(np.linspace(380, 780, 81), sp).rgb_to_hex() for sp in spectra]


    def visualize_species_distribution(self, sampled_spectrum = False) -> None:
        """
        Draw:
        1. Species distributions vs pH
        2. Chemical structures of the corresponding protonated forms
        3. Colour vs pH from single conformations
        4. Colour vs pH from an ensemble of conformations
        """
        if len(self.species) == 0:
            raise RuntimeError("A self.species is empty: first, generate distribution based on the nature of the colourant")

        x = np.linspace(self.pH_low, self.pH_high, self.num_points)

        if sampled_spectrum:
            fig, axs = plt.subplots(4)
        else:
            fig, axs = plt.subplots(3)

        for i in range(self.species.shape[0]):
            axs[0].plot(x, self.species[i, :], label=str(i))  # molar fraction vs pH for species

        axs[1].bar(x, np.ones(self.num_points), color=self.colours_vs_pH)  # bars marked by computed colour single conformations

        if sampled_spectrum:
            axs[2].bar(x, np.ones(self.num_points), color=self.colours_vs_pH_sampled)  # bars marked by computed colour single conformations

        # draw mols from SMILES
        for i in range(len(self.protonated_species)):
            newax = fig.add_axes([i / len(self.protonated_species),
                                  0.,
                                  1 / len(self.protonated_species),
                                  1/4], anchor='NE', zorder=1)
            newax.imshow(DrawMol(self.protonated_species[i][1], str(self.protonated_species[i][0])))
            newax.axis('off')

        axs[0].set_xlabel("pH", fontsize=15)
        axs[0].set_ylabel("Molar ratio", fontsize=15)

        box = axs[0].get_position()
        axs[0].set_position([box.x0, box.y0,
                             box.width, box.height * 0.9])
        # Put a legend upper current axis
        axs[0].legend(bbox_to_anchor=(0, 1.08, 1, 0.2), loc="lower left",
                      mode="expand", borderaxespad=0,
                      fancybox=True, shadow=True, fontsize=15, ncol=len(self.species))

        axs[0].tick_params(bottom=False, labelbottom=False, labelsize=15)
        axs[1].tick_params(bottom=False, labelbottom=False, labelsize=15)
        axs[2].tick_params(labelsize=15)
        axs[3].axis('off')

        plt.show()

