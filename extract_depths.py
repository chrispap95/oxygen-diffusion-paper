# This script is used to extract the depths of the rods from the images.
# It uses the peak positions from the peak_positions.json file to calculate the depths.
# The depths are saved in the result_depths.json file.

# The script can be run in two steps:
#   1. The first step is used to find the peak positions in the images.
#      The peak positions are saved in the peak_positions.json file.
#   2. The second step is used to calculate the depths of the rods.
#      The depths are saved in the result_depths.json file.

# The script can be run with the following command:
# python extract_depths.py -i peak_positions.json -s 2 -v -p -o result_depths.json

import argparse
import json
from pathlib import Path

from rich import print

import depth_calculator

parser = argparse.ArgumentParser(
    description="Extract the depths of the rods from the images."
)
parser.add_argument(
    "-i",
    "--input",
    type=str,
    default="peak_positions.json",
    help="Path to the input file.",
)
parser.add_argument(
    "-s",
    "--step",
    type=int,
    default=2,
    help="Step to run. 1: Find peak positions. 2: Calculate depths.",
)
parser.add_argument(
    "-v", "--verbose", action="store_true", help="Print more information."
)
parser.add_argument("-p", "--plot_images", action="store_true", help="Plot the images.")
parser.add_argument(
    "--show_one",
    action="store_true",
    help="Only show the first image and exit.",
)
parser.add_argument(
    "-o",
    "--output",
    type=str,
    default="result_depths.json",
    help="Path to the output file.",
)
parser.add_argument(
    "-d",
    "--plots_dir",
    type=str,
    default="plots",
    help="Path to the directory where the plots are saved.",
)


def process_shot(config, rod, face, shot, args, output):
    frame_path = (
        Path(config["input_path"]) / Path(rod) / Path(face) / Path(f"frame{shot}.png")
    )

    if args.verbose:
        print(f"\nProcessing frame: {frame_path}")

    if args.step == 1:
        depth_calculator.first_step(
            str(frame_path),
            output_path=args.plots_dir,
            only_show=args.show_one,
            window_length=config["window_length"],
            polyorder=config["polyorder"],
            cut=config["cut"],
            prominence=config["prominence"],
            rel_height=config["rel_height"],
            d=config["d"],
            sigmaSpace=config["sigmaSpace"],
            sigmaColor=config["sigmaColor"],
            clipLimit=config["clipLimit"],
            tileGridSize=config["tileGridSize"],
        )
    elif args.step == 2:
        depth = depth_calculator.depth_calculation(
            str(frame_path),
            input_config[rod][face][shot],
            d=config["d"],
            sigmaColor=config["sigmaColor"],
            sigmaSpace=config["sigmaSpace"],
            clipLimit=config["clipLimit"],
            tileGridSize=config["tileGridSize"],
            cut=config["cut"],
            window_length=config["window_length"],
            polyorder=config["polyorder"],
            prominence=config["prominence"],
            rel_height=config["rel_height"],
            plot_image=args.plot_images,
        )

        output[rod][face][shot] = {
            "depth": depth.nominal_value,
            "error": depth.std_dev,
        }

        if args.verbose:
            print(f"\n\td = {depth} mm")
            relative_uncertainty = 100 * depth.std_dev / depth.nominal_value
            print(f"\n\tRelative uncertainty: {relative_uncertainty:.2f}%")
            print("\n\tUncertainty components:")
            for var, error in depth.error_components().items():
                print(f"\t\t{var.tag:10}:\t{error:>6.4f}")
    else:
        raise ValueError("Step must be 1 or 2")


args = parser.parse_args()

input_config = json.load(open(args.input))
output = {}
brk = False
default_config = input_config["default_config"]

for rod in input_config.keys():
    if rod == "default_config":
        continue
    output[rod] = {}
    config_rod = default_config.copy()
    if "config" in input_config[rod].keys():
        config_rod.update(input_config[rod]["config"])
    for face in input_config[rod].keys():
        if face == "config":
            continue
        output[rod][face] = {}
        config = config_rod.copy()
        if "config" in input_config[rod][face].keys():
            config.update(input_config[rod][face]["config"])
        for shot in input_config[rod][face].keys():
            if shot == "config":
                continue

            process_shot(config, rod, face, shot, args, output)

            brk = True
            if args.show_one:
                break
    if brk and args.show_one:
        break

if args.step == 2:
    json.dump(output, open(args.output, "w"), indent=2)
