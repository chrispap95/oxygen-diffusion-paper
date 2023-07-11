# Effects of oxygen on the optical properties of phenyl-based scintillators during irradiation and recovery

This repository contains the code for all the figues shown in the paper. The code is organized in three jupyter notebooks that can be just viewed on github or run locally. The notebooks are:

- Figure 2: `notebooks/jaeriCase1.ipynb`
- Figure 3: `notebooks/jaeriOxygenModel.ipynb`
- Figures 4 and 5: `notebooks/sandiaOxygenModel.ipynb`

The python packages required to run the notebooks are listed in `requirements.txt`. The notebooks can be run locally by installing the packages in a virtual environment and running the notebooks with jupyter:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The plots for the depths and the refractive indices can be reproduced by running the `depth_plotter.py` and `index_plotter.py` scripts. For example:

```bash
python depth_plotter.py -b
```

All the data used in the paper are stored in json files in `data/`.

The notebooks can also be run on binder by clicking on the badge below.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/chrispap95/oxygen-diffusion-simulation/main)
