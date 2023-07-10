import argparse
import json
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageCms
from rich import print
from scipy.signal import find_peaks, peak_widths, savgol_filter
from uncertainties import ufloat

# Input parameters
parser = argparse.ArgumentParser()
parser.add_argument(
    "-f", "--filename", type=str, help="The filename of the image to process"
)
parser.add_argument(
    "--plot_image", action="store_true", help="Whether to plot the image or not"
)
parser.add_argument(
    "-v",
    "--verbose",
    action="store_true",
    help="Whether to print intermediate values or not",
)
parser.add_argument(
    "-s",
    "--step",
    type=int,
    default=1,
    help="The step to run the script in. 1: Find the peaks, 2: Calculate the depths",
)
parser.add_argument(
    "-o",
    "--output",
    type=str,
    help="The output path to save the plots to. If not specified, the plots will be saved to a default location",
)
parser.add_argument(
    "-c",
    "--config",
    type=str,
    help="The path to the config file. If not specified, the default config file will be used",
)


def load_image(
    image_path, d=9, sigmaColor=75, sigmaSpace=75, clipLimit=2.0, tileGridSize=8
):
    img = cv2.imread(image_path)
    blur = cv2.bilateralFilter(img, d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)
    im_gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(
        clipLimit=clipLimit, tileGridSize=(tileGridSize, tileGridSize)
    )
    equalized_image = clahe.apply(im_gray)
    return equalized_image


def load_image_legacy(image_path):
    # Load the image
    image = Image.open(image_path)

    # Convert the image to the CIELAB color space
    srgb_profile = ImageCms.createProfile("sRGB")
    lab_profile = ImageCms.createProfile("LAB")
    transform = ImageCms.buildTransformFromOpenProfiles(
        srgb_profile, lab_profile, "RGB", "LAB"
    )
    lab_image = ImageCms.applyTransform(image, transform)

    # Convert the image to a numpy array
    lab_array = np.array(lab_image)

    # Extract the lightness channel (L*)
    lightness_channel = lab_array[:, :, 0]

    return lightness_channel


def process_image(
    lightness_channel,
    window_length=11,
    polyorder=3,
    prominence=0.8,
    rel_height=0.5,
    cut=100,
):
    """
    Process an image to extract the depths:
        Extract the lightness channel
        Calculate the gradient of the lightness channel
        Calculate the absolute gradient
        Smoothen the absolute gradient
        Find the peaks in the absolute gradient
        Calculate the widths of the peaks

    Parameters
    ----------
    image : PIL.Image
        The image to process

    Returns
    -------
    peaks : numpy.ndarray
        The positions of the peaks in the image
    widths : numpy.ndarray
        The widths of the peaks in the image
    """
    mid = int(lightness_channel.shape[0] / 2)
    slc = [slice(mid - cut, mid + cut), slice(None)]
    lightness_profile = np.mean(lightness_channel[*slc], axis=0)
    abs_gradient = np.abs(np.gradient(lightness_profile))
    smoothened = savgol_filter(
        abs_gradient, window_length=window_length, polyorder=polyorder
    )
    peaks, _ = find_peaks(smoothened, prominence=prominence)
    widths, _, _, _ = peak_widths(smoothened, peaks, rel_height=rel_height)
    return peaks, widths


def plot_lightness_channel(lightness_channel):
    # Create a 2D plot of the lightness channel
    plt.imshow(lightness_channel, cmap="gray", origin="upper")
    plt.colorbar(label="Lightness (L*)")
    plt.title("Lightness Channel (L*)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig("test1.pdf")


def plot_peaks(lightness_channel, peaks, widths):
    lightness_profile = np.mean(lightness_channel, axis=0)
    abs_gradient = np.abs(np.gradient(lightness_profile))
    smoothened = savgol_filter(abs_gradient, window_length=11, polyorder=3)

    # Create a 1D plot of the data
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(smoothened, label="Data")

    # Plot the detected peaks with markers
    ax.plot(
        peaks, smoothened[peaks], "o", ms=8, mec="r", mfc="none", mew=2, label="Peaks"
    )

    # Add labels for peak positions and widths
    for i, (peak, width) in enumerate(zip(peaks, widths)):
        ax.text(peak, 1.1 * smoothened[peak], f"Peak {i+1}", ha="center", va="bottom")
        ax.text(
            peak,
            1.1 * smoothened[peak] / 2,
            f"Width: {width:.2f}",
            ha="center",
            va="top",
        )

    ax.set_title("Absolute L* Profile Gradient with peaks")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_yscale("log")
    ax.set_ylim(0.01, 100)
    fig.legend()
    fig.savefig("test2.pdf")


def first_step(
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
    output_path=None,
    only_show=False,
):
    # Load the image
    lightness_channel = load_image(
        filename,
        d=d,
        sigmaColor=sigmaColor,
        sigmaSpace=sigmaSpace,
        clipLimit=clipLimit,
        tileGridSize=tileGridSize,
    )

    shape = lightness_channel.shape
    ratio = shape[0] / shape[1]
    size = 7

    # Slicing parameters
    mid = int(shape[0] / 2)
    slc = [slice(mid - cut, mid + cut), ...]

    # Process the image
    lightness_profile = np.mean(lightness_channel[*slc], axis=0)
    abs_gradient = np.abs(np.gradient(lightness_profile))
    smoothened = savgol_filter(
        abs_gradient, window_length=window_length, polyorder=polyorder
    )
    peaks, _ = find_peaks(smoothened, prominence=prominence)
    widths, _, _, _ = peak_widths(smoothened, peaks, rel_height=rel_height)

    # Create a 2D plot of the lightness channel
    fig, ax1 = plt.subplots(figsize=(size * ratio * 1.2, size))
    im = ax1.imshow(lightness_channel, cmap="gray", origin="upper")
    fig.colorbar(im, ax=ax1)

    # Plot the lines for the window used to calculate the lightness profile
    ax1.plot([0, shape[1]], [mid - cut, mid - cut], "r--")
    ax1.plot([0, shape[1]], [mid + cut, mid + cut], "r--")

    ax1.minorticks_on()

    # Create a 1D plot of the data
    ax2 = ax1.twinx()
    ax2.plot(smoothened, label="Data")

    # Plot the detected peaks with markers
    ax2.plot(
        peaks, smoothened[peaks], "o", ms=8, mec="r", mfc="none", mew=2, label="Peaks"
    )

    # Add labels for peak positions and widths
    for i, (peak, width) in enumerate(zip(peaks, widths)):
        ax2.text(peak, 1.1 * smoothened[peak], f"Peak {i+1}", ha="center", va="bottom")
        ax2.text(
            peak,
            1.1 * smoothened[peak] / 2,
            f"Width: {width:.2f}",
            ha="center",
            va="top",
        )

    ax2.set_yscale("log")
    ax2.set_ylim(0.1, 100)

    fig.tight_layout()
    fig.subplots_adjust(right=1.05)

    if only_show:
        plt.show()
        return

    initial_path = filename.split("/")[:-3]
    rod = filename.split("/")[-3]
    face = filename.split("/")[-2]
    shot = filename.split("/")[-1]

    output_file = Path(face + shot.replace("frame", "_").replace(".png", ".pdf"))
    if output_path is None:
        output_path = Path(
            initial_path.replace("O2_videos_frames", "O2_videos_plots")
        ) / Path(rod)
    else:
        output_path = Path(output_path) / Path(rod)
    os.makedirs(output_path, exist_ok=True)
    fig.savefig(output_path / output_file)


def get_image_scale_old(peaks, widths, rod_width_exp, rod_width_exp_uncertainty):
    rod_width_pix = abs(peaks[2] - peaks[0])  # pixels
    rod_width_pix_uncertainty = np.sqrt(widths[0] ** 2 + widths[2] ** 2)
    image_scale = rod_width_exp / rod_width_pix  # mm/pixel
    image_scale_uncertainty = np.sqrt(
        (rod_width_exp_uncertainty / rod_width_pix) ** 2
        + (rod_width_exp * rod_width_pix_uncertainty / rod_width_pix**2) ** 2
    )
    return image_scale, image_scale_uncertainty


def get_depth_old(peaks, widths, image_scale, image_scale_uncertainty):
    depth_pix = abs(peaks[1] - peaks[0])  # pixels
    depth_pix_uncertainty = np.sqrt(widths[0] ** 2 + widths[1] ** 2)
    depth = depth_pix * image_scale  # mm
    depth_uncertainty = np.sqrt(
        (depth_pix_uncertainty * image_scale) ** 2
        + (depth_pix * image_scale_uncertainty) ** 2
    )
    return depth, depth_uncertainty


def get_image_scale(peaks, rod_width_exp):
    rod_width_pix = abs(peaks[2] - peaks[0])  # pixels
    image_scale = rod_width_exp / rod_width_pix  # mm/pixel
    return image_scale


def get_depth(peaks, image_scale):
    depth_pix = abs(peaks[1] - peaks[0])  # pixels
    depth = depth_pix * image_scale  # mm
    return depth


def get_rod_width(image):
    measurements = json.load(open("rod_sizes.json"))
    if "FaceA" in image or "FaceC" in image:
        face = "AC"
    elif "FaceB" in image or "FaceD" in image:
        face = "BD"
    else:
        raise ValueError("The face could not be found")
    for material in measurements.keys():
        if material in image:
            for rod in measurements[material].keys():
                if rod in image:
                    rod_width_exp = ufloat(
                        measurements[material][rod][face][0],
                        measurements[material][rod][face][1],
                        "rod width",
                    )
                    return rod_width_exp
    raise ValueError("The rod width could not be found")


def depth_calculation(
    filename,
    peaks_to_use,
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
    plot_image=False,
    debug=False,
):
    lightness_channel = load_image(
        filename,
        d=d,
        sigmaColor=sigmaColor,
        sigmaSpace=sigmaSpace,
        clipLimit=clipLimit,
        tileGridSize=tileGridSize,
    )
    peaks, widths = process_image(
        lightness_channel,
        window_length=window_length,
        polyorder=polyorder,
        prominence=prominence,
        rel_height=rel_height,
        cut=cut,
    )

    peaks_ufloat = [
        ufloat(peaks[i - 1], widths[i - 1] / 2, f"peak {i}") for i in peaks_to_use
    ]

    if debug:
        print(f"Peaks: {peaks}")
        print(f"Widths: {widths}")

    # Plot the lightness channel and the peaks
    if plot_image:
        plot_lightness_channel(lightness_channel)
        plot_peaks(lightness_channel, peaks, widths)

    # Get the actual rod width in mm
    rod_width_exp = get_rod_width(filename)
    if debug:
        print(f"Rod width: {rod_width_exp}")

    # Estimate the scale of the image and the depth of the boundary
    image_scale = get_image_scale(peaks_ufloat, rod_width_exp)
    depth = get_depth(peaks_ufloat, image_scale)

    return depth


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


def get_peaks_to_use(filename, database="peak_positions.json"):
    input = json.load(open(database))
    for rod in input.keys():
        if rod == "default_config" or rod not in filename:
            continue
        for face in input[rod].keys():
            if face == "config" or face not in filename:
                continue
            for shot in input[rod][face].keys():
                if f"frame{shot}" not in filename:
                    continue
                return input[rod][face][shot]
    raise ValueError("The peaks to use could not be found")


if "__main__" == __name__:
    args = parser.parse_args()

    database = args.config
    if args.config is None:
        database = "peak_positions.json"
    config = get_config(args.filename, database=database)

    if args.step == 1:
        first_step(
            args.filename,
            only_show=False,
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
            output_path=args.output,
        )
    elif args.step == 2:
        peaks_to_use = get_peaks_to_use(args.filename)
        depth = depth_calculation(
            args.filename,
            peaks_to_use,
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
            plot_image=args.plot_image,
        )
        print(f"\n {args.filename} d = {depth} mm")

        if args.verbose:
            relative_uncertainty = 100 * depth.std_dev / depth.nominal_value
            print(f"\nRelative uncertainty: {relative_uncertainty:.2f}%")
            print("\nUncertainty components:")
            for var, error in depth.error_components().items():
                print(f"\t{var.tag:10}:\t{error:>6.4f}")
    else:
        raise ValueError("The step must be 1 or 2")
