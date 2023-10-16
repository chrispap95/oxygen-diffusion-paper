import json

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from iminuit import Minuit
from iminuit.cost import LeastSquares
from jacobi import propagate

plt.style.use(hep.style.CMS)


def sellmeier_eq(wl, B1, C1):
    return np.sqrt((B1 * ((wl / 1000) ** 2)) / ((wl / 1000) ** 2 - C1) + 1)


input = json.load(open("data/pre_irr_index.json"))

wavelengths = np.array([470, 527, 635])
wl_space = np.linspace(400, 700, 1000)

# will use cividis colors but change the first one to orange
N = 2
cividis_colors = plt.cm.cividis(np.linspace(0, 1, N))
cividis_colors[-1] = (1, 0.549, 0, 1)
color_cycle = plt.cycler(color=cividis_colors)
plt.rcParams["axes.prop_cycle"] = color_cycle

fig, ax = plt.subplots(figsize=(8, 6))

for material in input.keys():
    index = np.array([])
    index_unc = np.array([])
    for wl in input[material].keys():
        index_wl = np.array([])
        index_wl_unc = np.array([])
        for meas in input[material][wl]:
            index_wl = np.append(index_wl, meas[0])
            index_wl_unc = np.append(index_wl_unc, meas[1])
        weights = 1 / (index_wl_unc**2)
        index = np.append(index, np.average(index_wl, weights=weights))
        index_unc = np.append(index_unc, 1 / np.sqrt(np.sum(weights)))

    if material == "PS":
        color = "C0"
        marker = "o"
    elif material == "PVT":
        color = "C1"
        marker = "x"

    ax.errorbar(
        wavelengths,
        index,
        yerr=index_unc,
        fmt=marker,
        label=material,
        capsize=3,
        color=color,
    )

    least_squares = LeastSquares(wavelengths, index, index_unc, sellmeier_eq)
    m = Minuit(least_squares, B1=0.1, C1=0.1)
    m.migrad()
    m.hesse()
    y, ycov = propagate(lambda p: sellmeier_eq(wl_space, *p), m.values, m.covariance)
    yerr_prop = np.diag(ycov) ** 0.5
    ax.plot(wl_space, y, color=color, lw=2)
    ax.fill_between(wl_space, y - yerr_prop, y + yerr_prop, facecolor=color, alpha=0.2)

    # Print the fit results
    print(f"\nFit results for {material}:")
    print(f"  chi2 / ndof = {m.fval:.1f} / {m.ndof:.0f} = {m.fmin.reduced_chi2:.1f}")
    for p, v, e in zip(m.parameters, m.values, m.errors):
        print(f"\t{p} = {v:.4f} ± {e:.4f}")
    print()

ax.set_xlabel(r"Wavelength $\lambda$ (nm)")
ax.set_ylabel(r"Index of refraction $n$")
ax.set_xlim(400, 700)
ax.set_ylim(1.578, 1.632)
ax.legend()
fig.tight_layout()
fig.savefig("plots/pre_irr_index.pdf", bbox_inches="tight")
