import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

plt.style.use(hep.style.ROOT)


def sqrt_law(x, a):
    return a * np.sqrt(x)


def plot_data_vs_sim(D, R, k1, k2, mode=None):
    """
    Plot the simulation results and the experimental data.

    Parameters
    ----------
    D : float
        Diffusion coefficient
    R : float
        Radical formation rate (includes the dose rate so it's Y*R)
    k1 : float
        Rate constant for the radical crosslinking reaction
    k2 : float
        Rate constant for the radical oxidation reaction
    mode: str
        The running mode:
         - "scan" mode will print the chi square and the parameters
         - "paper" mode will add the fit to the plot

    Returns
    -------
    None
    """
    filename = f"D_{D}_R_{R}_k1_{k1}_k2_{k2}".replace(".", "p")
    if mode == "scan":
        label = f"R={R} k1={k1} k2={k2}"
    else:
        label = "color depth simulation"
    maximum = np.load(f"../data/sim_{filename}.npy")

    times = np.linspace(0, 35, 9999)

    # find the saturation point
    t_saturation = times[np.argmax(maximum)]

    data_t = np.array([1, 5, 11, 18, 25])
    data_et = np.ones_like(data_t) * 0.5
    data_z = np.array([1.59, 2.09, 2.68, 3.32, 4.09])
    data_ez = np.array([0.02, 0.04, 0.07, 0.11, 0.21])

    z_index = np.array([1.23, 1.23, 1.24, 1.16, 1.14])
    z_index_mean = np.mean(z_index)
    z_index_std = np.sqrt((np.std(z_index)) ** 2 + (0.05 * z_index_mean) ** 2)

    # Days to a.u.
    times = times * (0.85 * times[-1] / t_saturation)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(times, maximum, label=label)
    ax.errorbar(
        data_t,
        data_z,
        fmt="o",
        capsize=3,
        xerr=data_et,
        yerr=data_ez,
        label="color depth data",
    )
    ax.hlines(
        z_index_mean,
        -10,
        100,
        color="black",
        linestyles="dashed",
        label="index boundary",
        linewidth=2,
    )
    ax.fill_between(
        np.linspace(-10, 100, 100),
        z_index_mean - z_index_std,
        z_index_mean + z_index_std,
        alpha=0.2,
        color="black",
    )

    # Calculate chi^2
    if mode == "scan":
        chi2 = np.sum((data_z - maximum[(np.rint(data_t)).astype(int)]) ** 2 / data_ez)
        chi2 += (z_index_mean - maximum[0]) ** 2 / z_index_std
        ax.text(t_saturation * 0.7, 0.3, rf"$\chi^2 = ${chi2:.4f}")

    if mode == "paper":
        from iminuit import Minuit
        from iminuit.cost import LeastSquares
        from jacobi import propagate

        least_squares = LeastSquares(data_t, data_z - z_index, data_ez, sqrt_law)
        m = Minuit(least_squares, a=0.1)
        m.migrad()
        m.hesse()
        y_1, ycov = propagate(lambda p: sqrt_law(times, *p), m.values, m.covariance)
        y_1 += z_index_mean
        yerr_prop = np.diag(ycov) ** 0.5
        ax.plot(times, y_1, label="fit", color="red", lw=2)
        ax.fill_between(
            times, y_1 - yerr_prop, y_1 + yerr_prop, facecolor="red", alpha=0.2
        )

    ax.set_xlim(-1, 36)
    ax.set_ylim(-0.2, 5.2)
    ax.set_xlabel("t (days)")
    ax.set_ylabel("z (mm)")
    fig.tight_layout()
    plt.legend(loc="upper left")
    plt.savefig("../plots/annealing_depth.pdf")
    plt.show()


if "__main__" == __name__:
    # These parameters will make the simulation match the experimental data
    D = 0.1
    R = 0.0006
    k1 = 0.02
    k2 = 0.2

    # run_sim(D, R, k1, k2)
    plot_data_vs_sim(D, R, k1, k2)
