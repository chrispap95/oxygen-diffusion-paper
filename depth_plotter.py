import json

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from iminuit import Minuit
from iminuit.cost import LeastSquares
from jacobi import propagate

plt.style.use(hep.style.ROOT)


def inv_sq_root(x, p):
    return p / x**0.5


material = "PVT"

input_irradations = json.load(open("irradiations.json"))
input_depths = json.load(open("result_depths_all_merged.json"))

x = []
y = []
xerr = []
yerr = []
for rod in input_depths.keys():
    if material in rod:
        continue
    if "temperature" in input_irradations[rod].keys():
        continue
    if "mean" not in input_depths[rod].keys():
        continue
    if "L11R" in rod:
        continue
    x.append(input_irradations[rod]["dose rate"])
    xerr.append(input_irradations[rod]["rel dose rate unc"])
    y.append(input_depths[rod]["mean"])
    yerr.append(input_depths[rod]["total_unc"])

x = np.array(x)
y = np.array(y)
xerr = np.array(xerr)
yerr = np.array(yerr)

# The initial xerr is relative, so we need to multiply it by x
xerr = xerr * x

# try to scale the yerr to make the fit better
if material == "PVT":
    scale = 0.08
elif material == "PS":
    scale = 0.05
yerr = np.sqrt(yerr**2 + (y * scale) ** 2)

fig, ax = plt.subplots(figsize=(7, 6))
ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="o", capsize=3, label="Data")
ax.set_xlabel("Dose rate (Gy/h)")
ax.set_ylabel("Depth (mm)")
ax.set_xlim(10, 10000)
ax.set_xscale("log")
ax.set_ylim(0, 5)


least_squares = LeastSquares(x, y, yerr, inv_sq_root)
m = Minuit(least_squares, p=0.1)
m.migrad()
m.hesse()
x_full = np.linspace(10, 10000, 9990)
y, ycov = propagate(lambda p: inv_sq_root(x_full, p), m.values, m.covariance)
yerr_prop = np.diag(ycov) ** 0.5
ax.plot(x_full, y, label="fit")
ax.fill_between(x_full, y - yerr_prop, y + yerr_prop, facecolor="C1", alpha=0.5)


# Plotting the fit info and legend
fit_info = [
    f"$\\chi^2$/$n_\\mathrm{{dof}}$ = {m.fval:.1f} / {m.ndof:.0f} = {m.fmin.reduced_chi2:.1f}",
    "Equation: $z = p/\\sqrt{R}$",
]
for p, v, e in zip(m.parameters, m.values, m.errors):
    fit_info.append(
        f"{p} = ${v:.1f} \\pm {e:.1f}$ $\\frac{{mm\\cdot h^{{1/2}}}}{{Gy^{{1/2}}}}$"
    )

ax.legend(loc="lower left")
ax.text(
    150,
    4.8,
    "\n".join(fit_info),
    ha="left",
    va="top",
    fontsize=22,
)
plt.tight_layout()

plt.show()
