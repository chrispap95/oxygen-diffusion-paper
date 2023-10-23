import json

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from iminuit import Minuit
from iminuit.cost import LeastSquares
from jacobi import propagate

plt.style.use(hep.style.ROOT)


def sellmeier_eq(wl, B1, C1):
    return np.sqrt((B1 * ((wl / 1000) ** 2)) / ((wl / 1000) ** 2 - C1) + 1)


def make_plot(wl, n, n_err, B1, C1, material):
    wl_space = np.linspace(380, 720, 1000)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.errorbar(wl, n, yerr=n_err, fmt="o", color="black", label="data", capsize=3)
    ax.plot(
        wl_space,
        sellmeier_eq(wl_space, B1, C1),
        color="red",
        label="reference",
        lw=2,
        linestyle="--",
    )
    ax.set_xlabel(r"Wavelength $\lambda$ (nm)")
    ax.set_ylabel("Refractive index $n$")
    ax.set_xlim(400, 700)
    if wl[0] == 400:
        ax.set_xlim(380, 720)
    if "BGO" in material:
        # ax.set_ylim(1.99, 2.26)
        ax.set_ylim(2.04, 2.26)
        ax.text(600, 2.2, material, fontsize=28)
    elif material == "PS":
        ax.set_ylim(1.568, 1.622)
        ax.text(620, 1.61, material, fontsize=28)

    least_squares = LeastSquares(wl, n, n_err, sellmeier_eq)
    m = Minuit(least_squares, B1=0.1, C1=0.1)
    m.migrad()
    m.hesse()
    y, ycov = propagate(lambda p: sellmeier_eq(wl_space, *p), m.values, m.covariance)
    yerr_prop = np.diag(ycov) ** 0.5
    ax.plot(wl_space, y, label="fit", color="black", lw=2)
    ax.fill_between(
        wl_space, y - yerr_prop, y + yerr_prop, facecolor="black", alpha=0.2
    )
    ax.legend(loc="lower left")

    # Plotting the fit info and legend
    minimal = True
    if not minimal:
        fit_info = [
            f"$\\chi^2$/$n_\\mathrm{{dof}}$ = {m.fval:.1f} / {m.ndof:.0f} = {m.fmin.reduced_chi2:.1f}",
            "Sellmeier equation",
        ]
        for p, v, e in zip(m.parameters, m.values, m.errors):
            fit_info.append(f"{p} = ${v:.1f} \\pm {e:.1f}$")

        ax.set_title(f"Refraction index for {material}", fontsize=22)
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
    print(f"\nFit results for {material}:")
    print(f"  chi2 / ndof = {m.fval:.1f} / {m.ndof:.0f} = {m.fmin.reduced_chi2:.1f}")
    for p, v, e in zip(m.parameters, m.values, m.errors):
        print(f"\t{p} = {v:.3f} ± {e:.3f}")
    print()

    plt.tight_layout()
    fig.savefig(
        f"plots/fig11_sellmeier_{material.replace(' ', '_')}.pdf", bbox_inches="tight"
    )


if __name__ == "__main__":
    input = json.load(open("data/validation_indices.json"))
    for material in input.keys():
        B1 = input[material]["B1"]
        C1 = input[material]["C1"]
        wl = np.array([int(i) for i in list(input[material]["measurements"].keys())])
        index = np.array([])
        index_unc = np.array([])
        for wl_i in input[material]["measurements"].keys():
            index = np.append(index, input[material]["measurements"][wl_i][0])
            index_unc = np.append(index_unc, input[material]["measurements"][wl_i][1])
        make_plot(wl, index, index_unc, B1, C1, material)
