import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from iminuit import Minuit
from iminuit.cost import LeastSquares
from jacobi import propagate

# from scipy.special import erfc

plt.style.use(hep.style.ROOT)


def sqrt_law(x, a):
    return a * np.sqrt(x)


if __name__ == "__main__":
    days = np.array([1, 5, 11, 18, 25])
    edays = np.ones_like(days) * 0.5

    time_array = np.linspace(0, 32, 300)

    z_index = np.array([1.23, 1.23, 1.24, 1.16, 1.14])
    z_color = np.array([1.59, 2.09, 2.68, 3.32, 4.09])

    ez_color = np.array([0.02, 0.04, 0.07, 0.11, 0.21])

    z_index_mean = np.mean(z_index)
    z_index_std = np.sqrt((np.std(z_index)) ** 2 + (0.05 * z_index_mean) ** 2)

    fig, ax = plt.subplots(figsize=(7, 6))
    plt.errorbar(
        days,
        z_color,
        yerr=ez_color,
        xerr=edays,
        fmt="o",
        color="black",
        ecolor="black",
        label="color depth",
        capsize=3,
    )
    ax.hlines(
        z_index_mean,
        -2,
        32,
        color="black",
        linestyles="dashed",
        label="index boundary",
        linewidth=2,
    )
    ax.fill_between(
        np.linspace(-2, 32, 100),
        z_index_mean - z_index_std,
        z_index_mean + z_index_std,
        alpha=0.2,
        color="black",
    )
    plt.xlabel("Days after irradiation")
    plt.ylabel("Annealing depth $z$ (mm)")
    plt.xlim(-2, 32)
    plt.ylim(0, 5)

    least_squares = LeastSquares(days, z_color - z_index, ez_color, sqrt_law)
    m = Minuit(least_squares, a=0.1)
    m.migrad()
    m.hesse()
    y_1, ycov = propagate(lambda p: sqrt_law(time_array, *p), m.values, m.covariance)
    y_1 += z_index_mean
    yerr_prop = np.diag(ycov) ** 0.5
    ax.plot(time_array, y_1, label="fit", color="red", lw=2)
    ax.fill_between(
        time_array, y_1 - yerr_prop, y_1 + yerr_prop, facecolor="red", alpha=0.2
    )

    # Print the fit results
    print("\nFit results:")
    print(f"  chi2 / ndof = {m.fval:.1f} / {m.ndof:.0f} = {m.fmin.reduced_chi2:.1f}")
    for p, v, e in zip(m.parameters, m.values, m.errors):
        print(f"\t{p} = {v:.4f} ± {e:.4f}")
    print(f"\tD = {v**2:.4f} ± {2*v*e:.4f}")
    print()

    plt.tight_layout()
    plt.legend(loc="upper left")
    plt.savefig("plots/annealing_depth.pdf")
    # plt.show()
