import argparse
import json

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from matplotlib.ticker import FixedLocator, LogLocator, MultipleLocator

plt.style.use(hep.style.ROOT)


parser = argparse.ArgumentParser(description="Plot the refractive index")
parser.add_argument("-m", "--material", help="Material to plot", type=str, default="PS")
parser.add_argument("-b", "--both", help="Plot both materials", action="store_true")
parser.add_argument(
    "--input_irradiations",
    help="Input irradiations file",
    type=str,
    default="data/irradiations.json",
)
parser.add_argument(
    "--input_indices",
    help="Input indices file",
    type=str,
    default="data/indices.json",
)
parser.add_argument(
    "--input_list",
    help="Input file with list of existing rods",
    type=str,
    default="data/rod_list.json",
)
parser.add_argument(
    "-o",
    "--output",
    help="Output file",
    type=str,
    default=None,
)


def loader(material, input_irradiations, input_indices, input_pre_irr):
    x = []
    xerr = []
    y_out = []
    yerr_out = []
    y_in = []
    yerr_in = []
    for rod in input_indices.keys():
        if material not in rod:
            continue
        if "temperature" in input_irradiations[rod].keys():
            continue
        if "L11R" in rod:
            continue
        x.append(input_irradiations[rod]["dose rate"])
        xerr.append(input_irradiations[rod]["rel dose rate unc"])
        y_out.append(input_indices[rod]["n_out"])
        yerr_out.append(input_indices[rod]["n_out_unc"])
        y_in.append(input_indices[rod]["n_in"])
        yerr_in.append(input_indices[rod]["n_in_unc"])
    x = np.array(x, dtype=float)
    xerr = np.array(xerr, dtype=float)
    y_out = np.array(y_out, dtype=float)
    yerr_out = np.array(yerr_out, dtype=float)
    y_in = np.array(y_in, dtype=float)
    yerr_in = np.array(yerr_in, dtype=float)

    index_unirr_arr = np.array([])
    index_unirr_arr_unc = np.array([])
    for meas in input_pre_irr[material]["470"]:
        index_unirr_arr = np.append(index_unirr_arr, meas[0])
        index_unirr_arr_unc = np.append(index_unirr_arr_unc, meas[1])

    return (
        x,
        xerr,
        y_out,
        yerr_out,
        y_in,
        yerr_in,
        index_unirr_arr,
        index_unirr_arr_unc,
    )


def plotter(
    material,
    input_irradiations,
    input_indices,
    input_pre_irr,
    fig=None,
    ax=None,
    minimal=False,
    use_multicolor=False,
):
    (
        x,
        xerr,
        y_out,
        yerr_out,
        y_in,
        yerr_in,
        index_unirr_arr,
        index_unirr_arr_unc,
    ) = loader(material, input_irradiations, input_indices, input_pre_irr)
    weights = 1 / (index_unirr_arr_unc**2)
    index_unirr = np.average(index_unirr_arr, weights=weights)
    index_unirr_unc = 1 / np.sqrt(np.sum(weights))

    # will use cividis colors but change the first one to orange
    N = 2
    cividis_colors = plt.cm.cividis(np.linspace(0, 1, N))
    cividis_colors[-1] = (1, 0.549, 0, 1)
    color_cycle = plt.cycler(color=cividis_colors)
    plt.rcParams["axes.prop_cycle"] = color_cycle

    # try to scale the yerr to make the fit better
    if use_multicolor:
        color_in = "C0"
        color_out = "C1"
    elif material == "PS":
        color_out = color_in = "C0"
        if use_multicolor:
            color_in = "C2"
    elif material == "PVT":
        color_out = color_in = "C1"
        if use_multicolor:
            color_in = "C3"

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))

    ax.hlines(index_unirr, 1, 10000, color="black", linestyles="-")
    ax.fill_between(
        np.linspace(1, 10000, 100),
        index_unirr - index_unirr_unc,
        index_unirr + index_unirr_unc,
        alpha=0.2,
        color="black",
    )

    # Plot the outter indices
    ax.errorbar(
        x,
        y_out,
        xerr=xerr,
        yerr=yerr_out,
        fmt="o",
        capsize=3,
        label=material + " $n_{out}$",
        color=color_out,
    )
    y_out_m = np.mean(y_out[~np.isnan(y_out)])
    y_out_std = np.std(y_out[~np.isnan(y_out)])
    ax.hlines(y_out_m, 1, 10000, color=color_out, linestyles="dashed")
    ax.fill_between(
        np.linspace(1, 10000, 100),
        y_out_m - y_out_std,
        y_out_m + y_out_std,
        alpha=0.2,
        color=color_out,
    )

    # Plot the inner indices
    ax.errorbar(
        x,
        y_in,
        xerr=xerr,
        yerr=yerr_in,
        fmt="x",
        capsize=3,
        label=material + " $n_{in}$",
        color=color_in,
    )
    y_in_m = np.mean(y_in[~np.isnan(y_in)])
    y_in_std = np.std(y_in[~np.isnan(y_in)])
    ax.hlines(y_in_m, 1, 10000, color=color_in, linestyles="dashed")
    ax.fill_between(
        np.linspace(1, 10000, 100),
        y_in_m - y_in_std,
        y_in_m + y_in_std,
        alpha=0.2,
        color=color_in,
    )

    if not minimal:
        ax.set_title(f"Index vs dose rate for {material}", fontsize=22)
    ax.set_xlabel("Dose rate (Gy/h)")
    ax.set_ylabel("refractive index")
    ax.set_xlim(1, 10000)
    ax.set_xscale("log")
    if material == "PS":
        ax.text(2.6, 1.685, material, fontsize=28)
        ax.set_ylim(1.595, 1.705)
    elif minimal and material == "PVT" and use_multicolor:
        ax.yaxis.set_major_locator(MultipleLocator(0.01))
        ax.yaxis.set_minor_locator(MultipleLocator(0.002))
        ax.text(2.6, 1.65, material, fontsize=28)
        ax.set_ylim(1.588, 1.662)

    plt.tight_layout()


if __name__ == "__main__":
    args = parser.parse_args()

    input_irradiations = json.load(open(args.input_irradiations))
    input_indices = json.load(open(args.input_indices))
    input_list = json.load(open(args.input_list))
    input_pre_irr = json.load(open("data/pre_irr_index.json"))

    if args.both:
        plotter(
            material="PS",
            input_irradiations=input_irradiations,
            input_indices=input_indices,
            input_pre_irr=input_pre_irr,
            minimal=True,
        )
        fig = plt.gcf()
        ax = plt.gca()
        plotter(
            material="PVT",
            input_irradiations=input_irradiations,
            input_indices=input_indices,
            input_pre_irr=input_pre_irr,
            fig=fig,
            ax=ax,
            minimal=True,
        )
        dummy_PS = ax.plot([], [], label="PS", color="C0", lw=3)[0]
        dummy_PVT = ax.plot([], [], label="PVT", color="C1", lw=3)[0]

        dummy_in = ax.errorbar(
            [-100],
            [-100],
            xerr=[1],
            yerr=[1],
            fmt="x",
            capsize=3,
            label="in",
            color="black",
        )
        dummy_out = ax.errorbar(
            [-100],
            [-100],
            xerr=[1],
            yerr=[1],
            fmt="o",
            capsize=3,
            label="out",
            color="black",
        )

        # Add legends
        color_legend = ax.legend([dummy_PS, dummy_PVT], ["PS", "PVT"], loc="upper left")
        ax.add_artist(color_legend)
        ax.legend(
            handles=[dummy_in, dummy_out],
            labels=["inner", "outer"],
            loc="upper right",
        )
        ax.minorticks_on()
    else:
        plotter(
            input_irradiations=input_irradiations,
            input_indices=input_indices,
            input_pre_irr=input_pre_irr,
            material=args.material,
            use_multicolor=True,
            minimal=True,
        )
        fig = plt.gcf()
        ax = plt.gca()
        dummy_in = ax.errorbar(
            [-100],
            [-100],
            xerr=[1],
            yerr=[1],
            fmt="x",
            capsize=3,
            label="inner",
            color="C0",
        )
        dummy_out = ax.errorbar(
            [-100],
            [-100],
            xerr=[1],
            yerr=[1],
            fmt="o",
            capsize=3,
            label="outer",
            color="C1",
        )
        dummy_unirr = ax.plot([], [], label="fit", color="black", lw=2)
        # color_legend = ax.legend([dummy_PS, dummy_PVT], ["PS", "PVT"], loc="upper left")
        # ax.add_artist(color_legend)
        ax.legend(
            handles=[dummy_in, dummy_out, dummy_unirr[0]],
            labels=["inner", "outer", "unirr."],
            loc="upper right",
        )

        minor_ticks = np.concatenate(
            [np.arange(2, 10, 2) * 10**exp for exp in range(5)]
        )
        minor_locator = FixedLocator(minor_ticks)
        major_locator = LogLocator(base=10.0, numticks=10)
        ax.xaxis.set_minor_locator(minor_locator)
        ax.xaxis.set_major_locator(major_locator)

    if args.output is None:
        output_name = f"plots/index_vs_dose_rate_{args.material}.pdf"
    else:
        output_name = args.output
    plt.savefig(output_name, bbox_inches="tight")
