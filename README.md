# Deploying a 5G Network Project

This folder contains the code for the "Deployin a 5G network" project by Heloise Dupont de Dinechin, Ahmad Ajalloeian and Delio Vicini.

## Structure
The code is structured as follows:
* `RunOptimization.ipynb` is the main notebook to run our methods. It generates results using all the methods we implemented and can run on both provided datasets (G1, G2)
* `baseline.py`, `convexhull.py`, `clustering.py`, `smooth.py` contain the different implementations. Our best/final method is in `smooth.py`
* `util.py` contains some utility functions, e.g. to evaluate the objective function efficiently
* The remaining jupyter notebooks were used to generate the figures in the report
* The `legacy` folder contains some outdated code we used in some of our experiments


## Requirements
The code has been tested Python 3.8.5. The code requires `numpy`, `jupyter`, `scipy`, `matplotlib` and `tqdm` to be installed, e.g. using
```
pip install numpy scipy jupyter matplotlib tqdm
```