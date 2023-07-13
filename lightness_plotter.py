import argparse
import json

import cv2
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from scipy.signal import find_peaks, peak_widths, savgol_filter

plt.style.use(hep.style.ROOT)

# Input parameters
parser = argparse.ArgumentParser()
parser.add_argument(
    "-f", "--filename", type=str, help="The filename of the image to process"
)
parser.add_argument(
    "-o",
    "--output",
    type=str,
    default="plots/lightness.pdf",
    help="The output file to save the plt.",
)
parser.add_argument(
    "-c",
    "--config",
    type=str,
    default="data/peak_positions_stills.json",
    help="The path to the config file. If not specified, the default config file will be used",
)


def plot_lightness(
    filename,
    d=9,
    sigmaColor=75,
    sigmaSpace=75,
    clipLimit=2.0,
    tileGridSize=8,
    cut=200,
    window_length=9,
    polyorder=3,
    prominence=0.8,
    rel_height=0.5,
    peaks_to_use=None,
    output=None,
):
    img = cv2.imread(filename)
    blur = cv2.bilateralFilter(img, d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)
    im_gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(
        clipLimit=clipLimit, tileGridSize=(tileGridSize, tileGridSize)
    )
    lightness_channel = clahe.apply(im_gray)
    lightness_channel = lightness_channel[30:-50, 515:1530]

    pad_size = 50
    lightness_channel = np.pad(
        lightness_channel, ((0, 0), (pad_size, 0)), mode="constant", constant_values=255
    )

    shape = lightness_channel.shape
    ratio = shape[0] / shape[1]
    size = 7

    # Slicing parameters
    mid = int(shape[0] / 2)
    slc = [slice(mid - cut, mid + cut), slice(pad_size, shape[1])]

    # Process the image
    lightness_profile = np.mean(lightness_channel[*slc], axis=0)
    abs_gradient = np.abs(np.gradient(lightness_profile))
    smoothened = savgol_filter(
        abs_gradient, window_length=window_length, polyorder=polyorder
    )
    peaks, _ = find_peaks(smoothened, prominence=prominence)
    widths, _, _, _ = peak_widths(smoothened, peaks, rel_height=rel_height)

    # Create a 2D plot of the lightness channel
    fig, ax1 = plt.subplots(figsize=(size * ratio * 1.3, size))
    ax1.imshow(lightness_channel, cmap="gray", origin="upper")
    ax1.axis("off")

    # Plot the lines for the window used to calculate the lightness profile
    ax1.plot([pad_size, shape[1]], [mid - cut, mid - cut], "r--")
    ax1.plot([pad_size, shape[1]], [mid + cut, mid + cut], "r--")

    # Create a 1D plot of the data
    ax2 = ax1.twinx()
    ax2.plot(
        range(pad_size, len(smoothened) + pad_size), smoothened, label="Data", lw=2
    )
    ax2.yaxis.tick_left()
    ax2.yaxis.set_ticks_position("both")
    ax2.yaxis.set_label_position("left")
    ax2.set_ylabel("Lightness profile gradient")

    # Plot the detected peaks with markers
    peaks = peaks[peaks_to_use]
    ax2.plot(
        peaks + pad_size,
        smoothened[peaks],
        "o",
        ms=8,
        mec="r",
        mfc="none",
        mew=2,
        label="Peaks",
    )

    # Add labels for peak positions and widths
    for i, (peak, width) in enumerate(zip(peaks, widths)):
        break
        ax2.text(
            peak + pad_size,
            1.1 * smoothened[peak],
            f"Peak {i+1}",
            ha="center",
            va="bottom",
            color="r",
            size=12,
        )
        ax2.text(
            peak + pad_size,
            1.1 * smoothened[peak] / 2,
            f"Width: {width:.2f}",
            ha="center",
            va="top",
            color="r",
            size=12,
        )

    ax2.set_yscale("log")
    ax2.set_ylim(0.5, 500)

    fig.tight_layout()
    # fig.subplots_adjust(right=1.01)

    fig.savefig(output, bbox_inches="tight")
    plt.show()


def get_config(filename, database="peak_positions.json"):
    input = json.load(open(database))
    default_config = input["default_config"]
    for rod in input.keys():
        if rod == "default_config" or rod not in filename:
            continue
        config = default_config.copy()
        if "config" in input[rod].keys():
            config.update(input[rod]["config"])
        return config


if "__main__" == __name__:
    args = parser.parse_args()

    database = args.config
    config = get_config(args.filename, database=database)

    peaks_to_use = np.array([0, 1, 6, 10])

    plot_lightness(
        args.filename,
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
        peaks_to_use=peaks_to_use,
        output=args.output,
    )
