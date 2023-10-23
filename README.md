# Effects of oxygen on the optical properties of phenyl-based scintillators during irradiation and recovery

This repository contains the code for all the figues shown in the paper. The code is organized in jupyter notebooks or python scripts that can be just viewed on github or run locally. The files are:

- Fig. 2: `notebooks/jaeriCase1.ipynb`
- Fig. 3: `notebooks/jaeriOxygenModel.ipynb`
- Fig. 4: `notebooks/RunGPUSimulation_visualization.ipynb`
- Fig. 5: `notebooks/RunGPUSimulation_scan.ipynb`
- Figs. 6 & 7: `notebooks/sandiaOxygenModel.ipynb`
- Fig. 8: `src/lightness_plotter.py`
- Fig. 11: `src/sellmeier.py`
- Fig. 12: `src/depth_plotter.py`
- Fig. 13: `src/annealing_depth.py`
- Fig. 14: `src/pre_irr_index.py`
- Fig. 15: `src/index_plotter.py`

The python packages required to run the notebooks are listed in `requirements.txt`. The notebooks can be run locally by installing the packages in a virtual environment and running the notebooks with jupyter:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The plots that were created by the python scripts can be reproduced by running them. For example, to reproduce Fig. 12 run:

```bash
python src/depth_plotter.py -b -n
```

All the data used in the paper are stored in json files in `data/`.

The notebooks can also be run on binder by clicking on the badge below.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/chrispap95/oxygen-diffusion-simulation/main)
