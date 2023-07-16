import argparse
import json

import numpy as np
from rich import print
from uncertainties import ufloat, unumpy

# Arguments
parser = argparse.ArgumentParser(
    description="Statistical analysis of the depths of the rods."
)
parser.add_argument(
    "-i",
    "--input",
    type=str,
    default="result_depths.json",
    help="Path to the input file.",
)
parser.add_argument(
    "-v", "--verbosity", type=int, default=1, help="Print more information."
)

args = parser.parse_args()

input = json.load(open(args.input))
output = {}

# Verbosity levels:
# 0: Only the final result is printed
# 1: The final result and the mean, standard deviation and difference for each rod is printed
# 2: The final result and the mean, standard deviation and difference for each rod and face is printed
verbose = 1

for rod in input.keys():
    print(f"\n[bold white]{rod}:[/bold white]")
    rod_means = np.array([])
    rod_weighted_means = np.array([])
    rod_weighted_stds = np.array([])
    rod_full_means = np.array([])
    for face in input[rod].keys():
        # Placing all the measurements for a given face in an array
        m_face = np.array([])
        for shot in input[rod][face].keys():
            m_face = np.append(
                m_face,
                ufloat(
                    input[rod][face][shot]["depth"], input[rod][face][shot]["error"]
                ),
            )
            rod_full_means = np.append(rod_full_means, input[rod][face][shot]["depth"])

        # Calculating the mean, standard deviation and difference for a given face
        # These are not used in the final result, but are printed for reference
        face_mean = unumpy.nominal_values(m_face).mean()
        face_std = unumpy.nominal_values(m_face).std()
        face_diff = (
            unumpy.nominal_values(m_face).max() - unumpy.nominal_values(m_face).min()
        ) / 2
        rod_means = np.append(rod_means, face_mean)

        # Calculating the weighted mean and standard deviation for a given face
        # These are used in the final result
        weights = 1 / (unumpy.std_devs(m_face) ** 2)
        face_weighted_mean = np.average(unumpy.nominal_values(m_face), weights=weights)
        rod_weighted_means = np.append(rod_weighted_means, face_weighted_mean)
        face_weighted_std = 1 / np.sqrt(np.sum(weights))
        rod_weighted_stds = np.append(rod_weighted_stds, face_weighted_std)

        # Printing the results for a given face
        if verbose > 1:
            print(f"\n\t{face}:")
            print(f"\t\tmean: \t{face_mean:.4f}")
            print(
                f"\t\tw mean: \t{face_weighted_mean:.4f} ± {face_weighted_std:.4f} ({face_weighted_std/face_weighted_mean*100:.2f}%)"
            )
            print(f"\t\tstd: \t{face_std:.4f} ({face_std/face_mean*100:.2f}%)")
            print(f"\t\tdiff: \t{face_diff:.4f} ({face_diff/face_mean*100:.2f}%)")

    # Calculating the mean, standard deviation and difference for a given rod
    # These are not used in the final result, but are printed for reference
    rod_mean = rod_means.mean()
    rod_std = rod_means.std()
    rod_diff = (rod_means.max() - rod_means.min()) / 2
    rod_full_std = rod_full_means.std()

    # Calculating the weighted mean and standard deviation for a given rod
    weights = 1 / (rod_weighted_stds**2)
    rod_weighted_mean = np.average(rod_weighted_means, weights=weights)
    rod_weighted_std = 1 / np.sqrt(np.sum(weights))

    # Saving the results
    # mean: The weighted mean of the weighted means for each face
    # std: The standard deviation of the weighted means for each face. Accounts for variation between faces
    # weight_std: The weighted standard deviation of the weighted means for each face. Accounts for the uncertainty coming for individual measurements
    output[rod] = {
        "mean": rod_weighted_mean,
        "std": rod_std,
        "weight_std": rod_weighted_std,
        "total_unc": np.sqrt(rod_weighted_std**2 + rod_std**2),
    }

    total_unc = np.sqrt(rod_std**2 + rod_weighted_std**2)

    # Printing the results for a given rod
    print("\n[bold]Summary:[/bold]")
    print(
        f"\tw mean:\t{rod_weighted_mean:.4f} ± {total_unc:.4f} ({total_unc/rod_weighted_mean*100:.2f}%)"
    )
    if verbose > 0:
        print(f"\tmean:  \t{rod_mean:.4f}")
        print(f"\tstd:   \t{rod_std:.4f} ({rod_std/rod_mean*100:.2f}%)")
        print(
            f"\tw unc: \t{rod_weighted_std:.4f} ({rod_weighted_std/rod_mean*100:.2f}%)"
        )
        print(f"\tf std: \t{rod_full_std:.4f} ({rod_full_std/rod_mean*100:.2f}%)")
        print(f"\tdiff:  \t{rod_diff:.4f} ({rod_diff/rod_mean*100:.2f}%)")


# Saving the results
json.dump(output, open(args.input.replace(".json", "_merged.json"), "w"), indent=2)
