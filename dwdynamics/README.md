# Solving quantum-inspired dynamics on quantum and classical computers

**Authors:** Philipp Hanussek, Jakub Pawłowski, Zakaria Mzaouali, Bartłomiej Gardas

This repository accompanies the paper *Solving quantum-inspired dynamics on quantum and classical computers*.
The methods implemented here are based on:
Jałowiecki, K., Więckowski, A., Gawron, P. et al. Parallel in time dynamics with quantum annealers. Sci Rep 10, 13534 (2020). [https://doi.org/10.1038/s41598-020-70017-x](https://doi.org/10.1038/s41598-020-70017-x)

## Data

The data for the instances corresponding to the paper is located in `data/results/hessian`.
Filenames are of the form:
`precision_p_timepoints_t_idx.json` where `idx` distinguishes between samples.

## Installation

It is recommended to use a fresh virtualenv. To install, simply run:

```bash
pip install .
```

For development, use poetry:

```bash
pip install poetry
poetry install
```

## Usage

All methods to generate the plots from the paper are in `notebooks/tests.ipynb`.
