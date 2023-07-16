import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageCms
from scipy.signal import find_peaks, peak_widths


def load_image_as_lab(image_path):
    # Load the image
    image = Image.open(image_path)

    # Convert the image to the CIELAB color space
    srgb_profile = ImageCms.createProfile("sRGB")
    lab_profile = ImageCms.createProfile("LAB")
    transform = ImageCms.buildTransformFromOpenProfiles(
        srgb_profile, lab_profile, "RGB", "LAB"
    )
    lab_image = ImageCms.applyTransform(image, transform)

    return lab_image


def get_lightness_channel(lab_image):
    # Convert the image to a numpy array
    lab_array = np.array(lab_image)

    # Extract the lightness channel (L*)
    lightness_channel = lab_array[:, :, 0]

    return lightness_channel


def plot_lightness_channel(lightness_channel, fig=None, ax=None):
    if fig is None:
        fig = plt.figure()

    if ax is None:
        ax = fig.add_subplot(111)

    # Create a 2D plot of the lightness channel
    ax.imshow(lightness_channel, cmap="gray", origin="upper")

    # Add a colorbar to show the lightness values
    ax.colorbar(label="Lightness (L*)")

    # Set the title and axis labels
    ax.title("Lightness Channel (L*)")
    ax.xlabel("x")
    ax.ylabel("y")

    # Show the plot
    fig.savefig("lightness_plots/lightness_channel.png")


def plot_peaks(data, peaks, widths, fig=None, ax=None):
    if fig is None:
        fig = plt.figure()

    if ax is None:
        ax = fig.add_subplot(111)

    # Create a 1D plot of the data
    ax.plot(data, label="Data")

    # Plot the detected peaks with markers
    ax.plot(peaks, data[peaks], "o", ms=8, mec="r", mfc="none", mew=2, label="Peaks")

    # Add labels for peak positions and widths
    for i, (peak, width) in enumerate(zip(peaks, widths)):
        ax.text(peak, data[peak], f"Peak {i+1}", ha="center", va="bottom")
        ax.text(peak, data[peak] / 2, f"Width: {width:.2f}", ha="center", va="top")

    # Set the title and axis labels
    ax.title("Absolute L* Profile Gradient with peaks")
    ax.xlabel("x")
    ax.ylabel("y")

    # Add a legend
    ax.legend()

    # Show the plot
    fig.savefig("lightness_plots/lightness_channel_peaks.png")


# check if output directory exists
if not os.path.exists("lightness_plots"):
    os.makedirs("lightness_plots")

image_path = "DSC_4576_00016.png"
lab_image = load_image_as_lab(image_path)
lightness_channel = get_lightness_channel(lab_image)
plot_lightness_channel(lightness_channel)

lightness_channel_profile_x = np.mean(lightness_channel, axis=0)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(lightness_channel_profile_x)
ax.title("Lightness Channel (L*) Profile")
ax.xlabel("x")
ax.ylabel("Lightness (L*)")
fig.savefig("lightness_plots/lightness_channel_profile_x.png")

lightness_channel_profile_x_med = np.median(lightness_channel, axis=0)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(lightness_channel_profile_x_med)
ax.title("Lightness Channel (L*) Profile")
ax.xlabel("x")
ax.ylabel("Lightness (L*)")
fig.savefig("lightness_plots/lightness_channel_profile_x_med.png")

lightness_channel_profile_x_gradient = abs(np.gradient(lightness_channel_profile_x))
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(lightness_channel_profile_x_gradient)
ax.title("Absolute Lightness Channel (L*) Profile Gradient")
ax.xlabel("x")
ax.ylabel("Absolute Lightness (L*) Gradient")
fig.savefig("lightness_plots/lightness_channel_profile_x_gradient.png")

peaks, _ = find_peaks(lightness_channel_profile_x_gradient, prominence=2)
widths, _, _, _ = peak_widths(
    lightness_channel_profile_x_gradient, peaks, rel_height=0.9
)
print("Peak positions:", peaks)
print("Peak widths:", widths)

plot_peaks(lightness_channel_profile_x_gradient, peaks, widths)
