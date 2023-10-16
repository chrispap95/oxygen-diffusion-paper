"""
Plot the depth vs dose rate for PS and/or PVT

Author: Christos Papageorgakis
"""

import argparse
import json

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from iminuit import Minuit
from iminuit.cost import LeastSquares
from jacobi import propagate

plt.style.use(hep.style.ROOT)


parser = argparse.ArgumentParser(description="Plot the depth vs dose rate")
parser.add_argument("-m", "--material", help="Material to plot", type=str, default="PS")
parser.add_argument("-b", "--both", help="Plot both materials", action="store_true")
parser.add_argument(
    "--input_irradiations",
    help="Input irradiations file",
    type=str,
    default="data/irradiations.json",
)
parser.add_argument(
    "--input_depths",
    help="Input depths file",
    type=str,
    default="data/result_depths_all_merged.json",
)
parser.add_argument(
    "--input_list",
    help="Input file with list of existing rods",
    type=str,
    default="data/rod_list.json",
)
parser.add_argument(
    "-n",
    "--include_no_boundaries",
    help="Include rods without boundaries",
    action="store_true",
)
parser.add_argument(
    "-o",
    "--output",
    help="Output file",
    type=str,
    default="plots/depth_vs_dose_rate.pdf",
)


def inv_sq_root(x, p):
    return p / x**0.5


def plotter(
    material, input_irradiations, input_depths, fig=None, ax=None, minimal=False
):
    """
    Plot the depth vs dose rate for a given material

    Parameters
    ----------
    material : str
        Material to plot (PS or PVT)
    input_irradiations : dict
        Dictionary with irradiation information
    input_depths : dict
        Dictionary with depth information
    fig : matplotlib.figure.Figure
        Figure to plot on (optional)
    ax : matplotlib.axes.Axes
        Axes to plot on (optional)
    minimal : bool
        Whether to plot a minimal version of the plot (optional)

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure with plot
    ax : matplotlib.axes.Axes
        Axes with plot
    """
    x = []
    y = []
    xerr = []
    yerr = []
    for rod in input_depths.keys():
        if material not in rod:
            continue
        if "temperature" in input_irradiations[rod].keys():
            continue
        if "mean" not in input_depths[rod].keys():
            continue
        if "L11R" in rod:
            continue
        x.append(input_irradiations[rod]["dose rate"])
        xerr.append(input_irradiations[rod]["rel dose rate unc"])
        y.append(input_depths[rod]["mean"])
        yerr.append(input_depths[rod]["total_unc"])
    x = np.array(x)
    y = np.array(y)
    xerr = np.array(xerr)
    yerr = np.array(yerr)

    # The initial xerr is relative, so we need to multiply it by x
    xerr = xerr * x

    # try to scale the yerr to make the fit better
    color = "C0"
    if material == "PS":
        marker = "o"
        scale = 0.08
    elif material == "PVT":
        marker = "x"
        color = "C1"
        scale = 0.05
    yerr = np.sqrt(yerr**2 + (y * scale) ** 2)

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    ax.errorbar(
        x, y, xerr=xerr, yerr=yerr, fmt=marker, capsize=3, label="Data", color=color
    )

    least_squares = LeastSquares(x, y, yerr, inv_sq_root)
    m = Minuit(least_squares, p=0.1)
    m.migrad()
    m.hesse()
    x_full = np.logspace(0, 5, 1000)
    y, ycov = propagate(lambda p: inv_sq_root(x_full, p), m.values, m.covariance)
    yerr_prop = np.diag(ycov) ** 0.5
    ax.plot(x_full, y, label="fit")
    ax.fill_between(x_full, y - yerr_prop, y + yerr_prop, facecolor=color, alpha=0.5)
    ax.set_xlabel("Dose rate (Gy/h)")
    ax.set_ylabel("Depth (mm)")
    ax.set_xlim(10, 10000)
    ax.set_xscale("log")
    ax.set_ylim(0, 5)

    # Plotting the fit info and legend
    if not minimal:
        fit_info = [
            f"$\\chi^2$/$n_\\mathrm{{dof}}$ = {m.fval:.1f} / {m.ndof:.0f} = {m.fmin.reduced_chi2:.1f}",
            "Equation: $z = p/\\sqrt{R}$",
        ]
        for p, v, e in zip(m.parameters, m.values, m.errors):
            fit_info.append(
                f"{p} = ${v:.1f} \\pm {e:.1f}$ $\\frac{{mm\\cdot h^{{1/2}}}}{{Gy^{{1/2}}}}$"
            )

        ax.set_title(f"Depth vs dose rate for {material}", fontsize=22)
        ax.legend(loc="lower left")
        ax.text(
            150,
            4.8,
            "\n".join(fit_info),
            ha="left",
            va="top",
            fontsize=22,
        )

    # Print the fit results
    print(f"\nFit results for {material}")
    print(f"  chi2 / ndof = {m.fval:.1f} / {m.ndof:.0f} = {m.fmin.reduced_chi2:.1f}")
    for p, v, e in zip(m.parameters, m.values, m.errors):
        print(f"\t{p} = {v:.3f} ± {e:.3f} mm*h^(1/2)/Gy^(1/2)")
        print(f"\tR_0 = {v**2 / 25:.3f} ± {2*v*e / 25:.3f} Gy/h")
    print()

    plt.tight_layout()
    return fig, ax


def plot_no_boundary(material, input_list, input_irradiations, ax):
    """
    Plot the depth vs dose rate for a given material for rods without boundaries

    Parameters
    ----------
    material : str
        Material to plot (PS or PVT)
    input_list : dict
        Dictionary with irradiation information
    input_irradiations : dict
        Dictionary with depth information
    ax : matplotlib.axes.Axes
        Axes to plot on (optional)

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes with plot
    """
    x = []
    xerr = []
    for rod in input_list["no boundary"]:
        if material not in rod:
            continue
        x.append(input_irradiations[rod]["dose rate"])
        xerr.append(input_irradiations[rod]["rel dose rate unc"])
    x = np.array(x)
    xerr = np.array(xerr)
    y = np.ones_like(x) * 5.05
    yerr = np.ones_like(x) * 0.25

    # The initial xerr is relative, so we need to multiply it by x
    xerr = xerr * x

    # try to scale the yerr to make the fit better
    if material == "PS":
        color = "C0"
    elif material == "PVT":
        color = "C1"

    ax.errorbar(
        x,
        y,
        xerr=xerr,
        yerr=yerr,
        fmt="o",
        capsize=3,
        label="Data",
        color=color,
        uplims=True,
    )
    # The factor is to avoid the labels from overlapping
    factor = 0.65
    ax.set_xlim(1 * factor, 10000 / factor)
    return ax


if __name__ == "__main__":
    args = parser.parse_args()

    input_irradiations = json.load(open(args.input_irradiations))
    input_depths = json.load(open(args.input_depths))
    input_list = json.load(open(args.input_list))

    # change default color cycle
    N = 2
    colors = plt.cm.cividis(np.linspace(0, 1, N))
    colors[-1] = (1, 0.549, 0, 1)
    color_cycle = plt.cycler(color=colors)
    plt.rcParams["axes.prop_cycle"] = color_cycle

    if args.both:
        fig, ax = plotter(
            material="PS",
            input_irradiations=input_irradiations,
            input_depths=input_depths,
            minimal=True,
        )
        fig, ax = plotter(
            material="PVT",
            input_irradiations=input_irradiations,
            input_depths=input_depths,
            fig=fig,
            ax=ax,
            minimal=True,
        )
        if args.include_no_boundaries:
            ax = plot_no_boundary(
                material="PS",
                input_list=input_list,
                input_irradiations=input_irradiations,
                ax=ax,
            )
            ax = plot_no_boundary(
                material="PVT",
                input_list=input_list,
                input_irradiations=input_irradiations,
                ax=ax,
            )
        # dummy_PS = ax.plot([], [], label="PS", color="C0", lw=3)[0]
        # dummy_PVT = ax.plot([], [], label="PS", color="C1", lw=3)[0]

        dummy_PS = ax.errorbar(
            [-100],
            [-100],
            xerr=[1],
            yerr=[1],
            fmt="o",
            capsize=3,
            label="data",
            color="C0",
        )
        dummy_PVT = ax.errorbar(
            [-100],
            [-100],
            xerr=[1],
            yerr=[1],
            fmt="x",
            capsize=3,
            label="data",
            color="C1",
        )

        dummy_line = ax.plot([], [], label="fit", color="black", lw=3)
        dummy_errorbar = ax.errorbar(
            [-100],
            [-100],
            xerr=[1],
            yerr=[1],
            fmt="o",
            capsize=3,
            label="data",
            color="black",
        )

        # Add legends
        color_legend = ax.legend([dummy_PS, dummy_PVT], ["PS", "PVT"], loc="lower left")
        ax.add_artist(color_legend)
        ax.legend(
            handles=[dummy_errorbar, dummy_line[0]],
            labels=["data", "fit"],
            loc="upper right",
        )
    else:
        fig, ax = plotter(
            input_irradiations=input_irradiations,
            input_depths=input_depths,
            material=args.material,
        )
        if args.include_no_boundaries:
            ax = plot_no_boundary(
                material=args.material,
                input_list=input_list,
                input_irradiations=input_irradiations,
                ax=ax,
            )

    plt.tight_layout()
    plt.savefig(args.output, bbox_inches="tight")
    plt.show()
